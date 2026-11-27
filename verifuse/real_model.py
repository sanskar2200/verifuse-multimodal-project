"""Frozen CLIP representations and supervised heads for a documented real corpus.

This module deliberately keeps the real-data model separate from the synthetic
shape benchmark. Dataset labels are classification targets, not forensic proof.
Heavy optional dependencies are imported only while encoding or training.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
import platform
import threading
import time
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .learning import sigmoid

MODEL_NAMES = ("text_only", "image_only", "feature_fusion", "late_fusion")
ENCODER_ID = "openai/clip-vit-base-patch32"
FEATURE_ORDER = ["clip_image_embedding", "tfidf_text_features"]


def runtime_versions():
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("numpy", "Pillow", "torch", "transformers", "scikit-learn", "huggingface-hub"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def normalize_embeddings(values):
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("Embeddings must be a finite two-dimensional matrix.")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("An encoder produced a zero embedding.")
    return array / norms


def feature_matrices(image, text):
    """Build modality-specific inputs: frozen CLIP vision and TF-IDF text."""
    image = normalize_embeddings(image)
    text = np.asarray(text, dtype=np.float32)
    if text.ndim != 2 or len(text) != len(image) or not np.isfinite(text).all():
        raise ValueError("Image and text feature rows must align and be finite.")
    return {"image_only": image, "text_only": text,
            "feature_fusion": np.concatenate([image, text], axis=1)}


def fit_text_vectorizer(records, train_mask, max_features=2048):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2,
                                 max_features=max_features, sublinear_tf=True,
                                 token_pattern=r"(?u)\b\w+\b", dtype=np.float32)
    texts = [row.get("text", "") for row in records]
    vectorizer.fit([text for text, is_train in zip(texts, train_mask) if is_train])
    values = {"vocabulary": {str(term): int(index) for term, index in vectorizer.vocabulary_.items()},
              "idf": vectorizer.idf_.tolist(),
              "max_features": max_features, "ngram_range": [1, 2], "min_df": 2,
              "sublinear_tf": True, "fit_split": "train"}
    return vectorizer, values


def load_text_vectorizer(values):
    from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=tuple(values["ngram_range"]),
                                 min_df=values["min_df"], max_features=values["max_features"],
                                 sublinear_tf=values["sublinear_tf"],
                                 token_pattern=r"(?u)\b\w+\b", vocabulary=values["vocabulary"],
                                 dtype=np.float32)
    vectorizer.vocabulary_ = {str(term): int(index) for term, index in values["vocabulary"].items()}
    vectorizer._tfidf = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True,
                                         sublinear_tf=values["sublinear_tf"])
    vectorizer._tfidf.idf_ = np.asarray(values["idf"], dtype=np.float32)
    return vectorizer


class LinearHead:
    """Portable logistic head; scaler and weights were fitted on training only."""
    def __init__(self, values):
        self.mean = np.asarray(values["mean"], dtype=np.float64)
        self.scale = np.asarray(values["scale"], dtype=np.float64)
        self.weights = np.asarray(values["weights"], dtype=np.float64)
        self.bias = float(values["bias"])
        if not (self.mean.shape == self.scale.shape == self.weights.shape) or self.weights.ndim != 1:
            raise ValueError("Invalid learned head dimensions.")
        if not all(np.isfinite(value).all() for value in (self.mean, self.scale, self.weights)) or not np.isfinite(self.bias):
            raise ValueError("Non-finite learned model parameters.")
        if np.any(self.scale <= 0):
            raise ValueError("Learned scaling factors must be positive.")

    def predict_proba(self, matrix):
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.weights) or not np.isfinite(matrix).all():
            raise ValueError("Invalid features for the learned head.")
        return sigmoid(((matrix - self.mean) / self.scale) @ self.weights + self.bias)


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_npz(path, **values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part.npz")
    np.savez_compressed(temporary, **values)
    temporary.replace(path)


def manifest_digest(records):
    return hashlib.sha256(json.dumps(records, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def validate_manifest(manifest):
    """Reject any shared supplied source identifier/hash across partitions.

    Checking every edge also prevents a connected component spanning splits.
    Absence of source/event IDs is reported as a limitation, never inferred.
    """
    records = manifest.get("records", [])
    if not records:
        raise ValueError("The real-data manifest contains no records.")
    ids, seen, populated = set(), {}, set()
    fields = {
        "group_id": "group", "event_id": "event", "story_id": "story",
        "source_image_id": "image_id", "image_id": "image_id",
        "source_text_id": "text_id", "text_id": "text_id",
        "source_caption_id": "text_id", "caption_id": "text_id",
        "image_sha256": "image_bytes", "pixel_sha256": "image_pixels",
        "text_sha256": "text_hash", "image_pixel_sha256": "image_pixels",
    }
    for row in records:
        identifier = str(row.get("id", ""))
        if not identifier or identifier in ids:
            raise ValueError("Every real-data record needs a unique nonempty id.")
        ids.add(identifier)
        if row.get("split") not in {"train", "val", "test"}:
            raise ValueError("Real-data splits must be train, val, or test.")
        if row.get("label") not in {"REAL", "FAKE"}:
            raise ValueError("Real-data labels must be REAL or FAKE.")
        if not isinstance(row.get("text", ""), str):
            raise ValueError("Record text must be a string.")
        for field, namespace in fields.items():
            value = row.get(field)
            if value is None or value == "":
                continue
            populated.add(field)
            key = (namespace, str(value))
            prior = seen.setdefault(key, row["split"])
            if prior != row["split"]:
                raise ValueError(f"Source leakage across splits for {field}: {value!s} ({prior}, {row['split']}).")
    splits = {}
    for split in ("train", "val", "test"):
        subset = [r for r in records if r["split"] == split]
        if {r["label"] for r in subset} != {"REAL", "FAKE"}:
            raise ValueError(f"The {split} split must contain both REAL and FAKE examples.")
        groups = {r.get("group_id", r.get("source_image_id", str(r["id"]))) for r in subset}
        splits[split] = {"records": len(subset), "groups": len(groups),
                         "REAL": sum(r["label"] == "REAL" for r in subset),
                         "FAKE": sum(r["label"] == "FAKE" for r in subset)}
    return {"splits": splits, "checked_fields": sorted(populated),
            "supplied_source_identifiers_disjoint": bool(populated),
            "pixel_hash_disjoint": bool({"pixel_sha256", "image_pixel_sha256"} & populated),
            "limitations": ["Only supplied source identifiers and hashes are audited; unidentified related events and near-duplicate images can remain.",
                            "Overlap with the frozen encoder's undisclosed pretraining corpus cannot be ruled out."]}


class ClipEncoder:
    """Local-only CLIP inference. Training never updates the CLIP parameters."""
    def __init__(self, model_path, batch_size=16, threads=4):
        self.model_path = Path(model_path)
        self.batch_size = int(batch_size)
        self.threads = int(threads)
        if self.batch_size < 1 or self.threads < 1:
            raise ValueError("Batch size and CPU thread count must be positive.")
        self._model = None
        self._lock = threading.Lock()
        self.truncated_texts = 0

    def _load(self):
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            if not (self.model_path / "config.json").is_file():
                raise FileNotFoundError(f"Local CLIP files are missing from {self.model_path}. Download the documented pinned encoder first.")
            import torch
            from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizerFast
            torch.set_num_threads(self.threads)
            self.torch = torch
            self.image_processor = CLIPImageProcessor.from_pretrained(str(self.model_path), local_files_only=True)
            self.tokenizer = CLIPTokenizerFast.from_pretrained(str(self.model_path), local_files_only=True)
            self._model = CLIPModel.from_pretrained(str(self.model_path), local_files_only=True).to("cpu").eval()
            for parameter in self._model.parameters():
                parameter.requires_grad_(False)

    def encode_images(self, images):
        self._load()
        images = [ImageOps.exif_transpose(image).convert("RGB") for image in images]
        encoded = self.image_processor(images=images, return_tensors="pt")
        with self.torch.inference_mode():
            features = self._model.get_image_features(**encoded)
        return normalize_embeddings(features.detach().cpu().numpy())

    def encode_texts(self, texts):
        self._load()
        token_lengths = self.tokenizer(texts, truncation=False, padding=False, verbose=False)["input_ids"]
        self.truncated_texts += sum(len(tokens) > 77 for tokens in token_lengths)
        encoded = self.tokenizer(texts, padding=True, truncation=True, max_length=77, return_tensors="pt")
        with self.torch.inference_mode():
            features = self._model.get_text_features(**encoded)
        return normalize_embeddings(features.detach().cpu().numpy())

    def encode(self, images, texts):
        if len(images) != len(texts):
            raise ValueError("Image and text batches must have equal length.")
        return self.encode_images(images), self.encode_texts(texts)


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _encoder_fingerprint(encoder):
    # Encoder identity must change when downloaded weights or processing config changes.
    # Download provenance normally pins the commit; config + local weight identity also
    # prevents reuse after a local weight file replacement.
    pieces = [str(encoder.model_path.resolve())]
    for name in ("config.json", "preprocessor_config.json", "tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt"):
        path = encoder.model_path / name
        if path.exists():
            pieces.append(name + ":" + _file_sha(path))
    for path in sorted(encoder.model_path.glob("*")):
        if path.is_file() and path.suffix in {".bin", ".safetensors"}:
            pieces.append(path.name + ":" + _file_sha(path))
    return hashlib.sha256("|".join(pieces).encode()).hexdigest()


def encode_manifest(root, manifest, encoder, output_path=None):
    """Resume feature extraction by image bytes/text, not labels or partition names."""
    started = time.perf_counter()
    started_at = utc_now()
    root = Path(root)
    output_path = Path(output_path or root / "data/real/embeddings.npz")
    cache_path = output_path.with_name("clip_feature_cache.npz")
    records = manifest["records"]
    fingerprint = _encoder_fingerprint(encoder)
    image_keys, text_keys, paths = [], [], []
    image_digests = {}
    for row in records:
        if not row.get("image_path") or not row.get("text", "").strip():
            raise ValueError("Local feature extraction requires an image_path and nonempty text for every pair.")
        path = (root / row["image_path"]).resolve()
        if path not in image_digests:
            image_digests[path] = _file_sha(path)
        image_keys.append(hashlib.sha256((fingerprint + image_digests[path]).encode()).hexdigest())
        text_keys.append(hashlib.sha256((fingerprint + "\0" + row["text"]).encode("utf-8")).hexdigest())
        paths.append(path)
    image_cache, text_cache = {}, {}
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as stored:
            image_cache = dict(zip(stored["image_keys"].tolist(), stored["image"].astype(np.float32)))
            text_cache = dict(zip(stored["text_keys"].tolist(), stored["text"].astype(np.float32)))
        print(f"Found cached features: {len(image_cache)} images, {len(text_cache)} texts.", flush=True)
    reused_images = len(set(image_keys) & image_cache.keys())
    reused_texts = len(set(text_keys) & text_cache.keys())

    def save_cache():
        _atomic_npz(cache_path, image_keys=np.asarray(list(image_cache), dtype=str),
                    image=np.asarray(list(image_cache.values()), dtype=np.float32),
                    text_keys=np.asarray(list(text_cache), dtype=str),
                    text=np.asarray(list(text_cache.values()), dtype=np.float32))

    unique_images = {key: path for key, path in zip(image_keys, paths) if key not in image_cache}
    pending = list(unique_images)
    for offset in range(0, len(pending), encoder.batch_size):
        keys = pending[offset:offset + encoder.batch_size]
        images = []
        for key in keys:
            with Image.open(unique_images[key]) as image:
                images.append(ImageOps.exif_transpose(image).convert("RGB"))
        for key, feature in zip(keys, encoder.encode_images(images)):
            image_cache[key] = feature
        if (offset // encoder.batch_size) % 8 == 0 or offset + encoder.batch_size >= len(pending):
            save_cache()
        print(f"CLIP images: {min(offset + encoder.batch_size, len(pending))}/{len(pending)} newly encoded", flush=True)
    unique_text = {key: row["text"] for key, row in zip(text_keys, records) if key not in text_cache}
    pending = list(unique_text)
    for offset in range(0, len(pending), encoder.batch_size):
        keys = pending[offset:offset + encoder.batch_size]
        for key, feature in zip(keys, encoder.encode_texts([unique_text[key] for key in keys])):
            text_cache[key] = feature
        if (offset // encoder.batch_size) % 16 == 0 or offset + encoder.batch_size >= len(pending):
            save_cache()
        print(f"CLIP texts: {min(offset + encoder.batch_size, len(pending))}/{len(pending)} newly encoded", flush=True)
    image = normalize_embeddings([image_cache[key] for key in image_keys])
    text = normalize_embeddings([text_cache[key] for key in text_keys])
    _atomic_npz(output_path, ids=np.asarray([str(r["id"]) for r in records], dtype=str), image=image, text=text,
                image_keys=np.asarray(image_keys), text_keys=np.asarray(text_keys), encoder_fingerprint=np.asarray(fingerprint))
    metadata = {"encoder": ENCODER_ID, "frozen": True, "encoder_fingerprint": fingerprint,
                "feature_count": int(image.shape[1]), "records": len(records),
                "unique_images": len(set(image_keys)), "unique_texts": len(set(text_keys)),
                "newly_encoded_truncated_texts": encoder.truncated_texts,
                "started_at_utc": started_at, "completed_at_utc": utc_now(),
                "encoding_seconds": round(time.perf_counter() - started, 3),
                "timing_scope": "Wall time for this extraction invocation including cache lookup and model loading; earlier interrupted invocations are excluded.",
                "reused_unique_images": reused_images, "reused_unique_texts": reused_texts,
                "batch_size": encoder.batch_size, "cpu_threads": getattr(encoder, "threads", None),
                "runtime_versions": runtime_versions(),
                "cache_key_inputs": "encoder files + image file bytes / exact caption text; no labels or split names",
                "extraction": "local CLIP image/text encoders on supplied natural-media files"}
    receipt_path = encoder.model_path / "download_receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        metadata["revision"] = receipt.get("revision")
        metadata["download_receipt_sha256"] = _file_sha(receipt_path)
    _atomic_json(output_path.with_suffix(".metadata.json"), metadata)
    return image, text, metadata


def load_embeddings(path, records):
    """Align externally supplied embeddings by explicit record IDs."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as stored:
        if not {"ids", "image", "text"}.issubset(stored.files):
            raise ValueError("Embedding NPZ must contain ids, image and text arrays.")
        ids = [str(value) for value in stored["ids"].tolist()]
        if len(set(ids)) != len(ids):
            raise ValueError("Embedding IDs must be unique.")
        images, texts = np.asarray(stored["image"]), np.asarray(stored["text"])
    if len(images) != len(ids) or len(texts) != len(ids):
        raise ValueError("Embedding IDs and matrices have different lengths.")
    lookup = {identifier: index for index, identifier in enumerate(ids)}
    missing = [str(r["id"]) for r in records if str(r["id"]) not in lookup]
    if missing:
        raise ValueError(f"Embeddings missing for {len(missing)} manifest records.")
    order = [lookup[str(r["id"])] for r in records]
    image, text = normalize_embeddings(images[order]), normalize_embeddings(texts[order])
    if image.shape != text.shape:
        raise ValueError("Image/text embedding dimensions differ.")
    metadata_path = path.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    metadata.setdefault("extraction", "externally supplied embeddings aligned by record ID")
    metadata.setdefault("limitations", ["Input-to-embedding correspondence is supplied by the embedding provider and has not been independently re-encoded."])
    metadata["embedding_file_sha256"] = _file_sha(path)
    return image, text, metadata


def fit_head(train_x, train_y, val_x, val_y, candidates=(.001, .01, .1, 1.0, 10.0), max_iter=1500):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    from sklearn.exceptions import ConvergenceWarning
    train_x = np.asarray(train_x, dtype=np.float64)
    mean, scale = train_x.mean(axis=0), train_x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    train_scaled, val_scaled = (train_x - mean) / scale, (np.asarray(val_x) - mean) / scale
    best, trials = None, []
    for c_value in candidates:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model = LogisticRegression(C=float(c_value), solver="lbfgs", max_iter=max_iter, random_state=42)
            model.fit(train_scaled, train_y)
        loss = float(log_loss(val_y, model.predict_proba(val_scaled)[:, 1], labels=[0, 1]))
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        trials.append({"C": float(c_value), "validation_log_loss": loss, "iterations": int(model.n_iter_[0]), "converged": converged})
        if best is None or loss < best[0]:
            best = (loss, model, float(c_value), converged)
    _, model, selected_c, converged = best
    values = {"weights": model.coef_[0].tolist(), "bias": float(model.intercept_[0]), "mean": mean.tolist(),
              "scale": scale.tolist(), "C": selected_c, "fit_split": "train", "selection_split": "val",
              "selection_metric": "log_loss", "converged": converged, "trials": trials}
    # Serialized inference must reproduce the sklearn estimator before saving.
    np.testing.assert_allclose(LinearHead(values).predict_proba(val_x), model.predict_proba(val_scaled)[:, 1], atol=1e-10)
    return values


def binary_metrics(labels, probabilities):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, average_precision_score, confusion_matrix, f1_score,
                                 log_loss, precision_recall_curve, precision_recall_fscore_support,
                                 precision_score, recall_score, roc_auc_score, roc_curve, brier_score_loss)
    labels, probabilities = np.asarray(labels, dtype=int), np.asarray(probabilities, dtype=float)
    predicted = (probabilities >= .5).astype(int)
    precision, recall, _, support = precision_recall_fscore_support(labels, predicted, labels=[0, 1], zero_division=0)
    per_f1 = [2 * p * r / (p + r) if p + r else 0.0 for p, r in zip(precision, recall)]
    both_classes = len(np.unique(labels)) == 2
    if both_classes:
        fpr, tpr, _ = roc_curve(labels, probabilities)
        pr_precision, pr_recall, _ = precision_recall_curve(labels, probabilities)
    else:
        fpr = tpr = pr_precision = pr_recall = np.asarray([])
    return {"accuracy": float(accuracy_score(labels, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
            "precision": float(precision_score(labels, predicted, zero_division=0)),
            "recall": float(recall_score(labels, predicted, zero_division=0)),
            "f1": float(f1_score(labels, predicted, zero_division=0)),
            "macro_f1": float(f1_score(labels, predicted, labels=[0, 1], average="macro", zero_division=0)),
            "roc_auc": float(roc_auc_score(labels, probabilities)) if both_classes else None,
            "pr_auc": float(average_precision_score(labels, probabilities)) if np.any(labels == 1) else None,
            "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
            "brier_score": float(brier_score_loss(labels, probabilities)),
            "confusion_matrix": confusion_matrix(labels, predicted, labels=[0, 1]).tolist(), "support": len(labels),
            "per_class": {name: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(per_f1[i]), "support": int(support[i])}
                          for i, name in enumerate(("REAL", "FAKE"))},
            "curves": {"false_positive_rate": fpr.tolist(), "true_positive_rate": tpr.tolist(),
                       "precision": pr_precision.tolist(), "recall": pr_recall.tolist()}}


def train(root, manifest, image_embeddings, text_embeddings, encoder_metadata=None, candidates=(.001, .01, .1, 1.0, 10.0)):
    """Fit supervised heads and evaluate the untouched supplied test partition."""
    from sklearn.metrics import log_loss
    import sklearn
    started = time.perf_counter()
    started_at = utc_now()
    root = Path(root)
    records = manifest["records"]
    audit = validate_manifest(manifest)
    train_mask = np.asarray([row["split"] == "train" for row in records])
    text_vectorizer, text_vectorizer_values = fit_text_vectorizer(records, train_mask)
    text_features = text_vectorizer.transform([row.get("text", "") for row in records]).toarray()
    matrices = feature_matrices(image_embeddings, text_features)
    if len(matrices["image_only"]) != len(records):
        raise ValueError("Embedding row count differs from the manifest.")
    if matrices["image_only"].shape[1] != 512:
        raise ValueError("This real-data profile requires 512-dimensional CLIP ViT-B/32 embeddings.")
    masks = {split: np.asarray([row["split"] == split for row in records]) for split in ("train", "val", "test")}
    y = np.asarray([row["label"] == "FAKE" for row in records], dtype=int)
    models, scores = {}, {}
    for name in ("text_only", "image_only", "feature_fusion"):
        print(f"Training {name}: {int(masks['train'].sum())} examples, {matrices[name].shape[1]} features", flush=True)
        models[name] = fit_head(matrices[name][masks["train"]], y[masks["train"]], matrices[name][masks["val"]], y[masks["val"]], candidates)
        scores[name] = LinearHead(models[name]).predict_proba(matrices[name])
        print(f"Selected {name} C={models[name]['C']}; validation loss={min(t['validation_log_loss'] for t in models[name]['trials']):.5f}", flush=True)
    alphas = np.linspace(0, 1, 21)
    losses = [float(log_loss(y[masks["val"]], alpha * scores["text_only"][masks["val"]] + (1 - alpha) * scores["image_only"][masks["val"]], labels=[0, 1])) for alpha in alphas]
    alpha = float(alphas[int(np.argmin(losses))])
    scores["late_fusion"] = alpha * scores["text_only"] + (1 - alpha) * scores["image_only"]
    validation_metrics = {name: binary_metrics(y[masks["val"]], score[masks["val"]]) for name, score in scores.items()}
    selected_model = min(MODEL_NAMES, key=lambda name: validation_metrics[name]["log_loss"])
    train_prevalence = float(np.mean(y[masks["train"]]))
    majority_baseline = {"name": "Training-majority baseline", "class": "FAKE" if train_prevalence >= .5 else "REAL",
                         "training_fake_prevalence": train_prevalence,
                         "description": "Always predict the most frequent training label. The test distribution is not used to choose the class.",
                         "metrics": binary_metrics(y[masks["test"]], np.full(int(masks["test"].sum()), train_prevalence))}
    majority_baseline["test_accuracy"] = majority_baseline["metrics"]["accuracy"]
    label_definition = manifest.get("label_definition", "REAL and FAKE are inherited dataset labels. They do not establish authenticity or factual truth.")
    dataset_info = manifest.get("dataset", {})
    fallback_name = dataset_info.get("name", "Real image-text corpus") if isinstance(dataset_info, dict) else str(dataset_info)
    dataset_name = manifest.get("name") or fallback_name
    encoder_metadata = dict(encoder_metadata or {})
    encoder_metadata.update({"model_id": ENCODER_ID, "local_path": "models/real/clip", "frozen": True,
                             "normalized": True, "dimensions": matrices["image_only"].shape[1],
                             "role": "image-only visual feature extractor"})
    bundle = {"version": 1, "profile": "real", "real_data": True, "dataset_name": dataset_name,
              "dataset_sha256": manifest_digest(records), "label_definition": label_definition,
              "scope": "Supervised classification of the documented real-media corpus's inherited labels",
              "encoder": encoder_metadata, "text_vectorizer": text_vectorizer_values,
              "feature_order": FEATURE_ORDER, "models": models,
              "late_fusion_alpha": alpha, "decision_threshold": .5, "selected_model": selected_model,
              "probability_calibrated": False,
              "training_started_at_utc": started_at, "runtime_versions": runtime_versions(),
              "fit": {"head": "L2 logistic regression", "optimizer": "sklearn lbfgs", "sklearn_version": sklearn.__version__,
                      "class_weight": None,
                      "standardization_fit_split": "train", "regularization_selection_split": "val",
                      "regularization_candidates": list(candidates), "late_fusion_selection_split": "val",
                      "selection_metric": "log_loss", "test_used_for_selection": False,
                      "clip_fine_tuned": False, "text_model": "train-fitted TF-IDF unigram/bigram features with L2 logistic regression",
                      "text_max_features": len(text_vectorizer_values["vocabulary"]), "seed": manifest.get("seed", 42)}}
    output = root / "results/real"
    output.mkdir(parents=True, exist_ok=True)
    descriptions = {"text_only": "L2 logistic regression on train-fitted TF-IDF word and bigram features.",
                    "image_only": "L2 logistic regression on frozen normalized CLIP image embeddings.",
                    "feature_fusion": "L2 logistic regression on concatenated CLIP image and TF-IDF text features.",
                    "late_fusion": "Independent image/text head probabilities blended by validation-selected weight."}
    names = {"text_only": ("text_baseline", "Text-only baseline"), "image_only": ("image_baseline", "Image-only baseline"),
             "feature_fusion": ("feature_fusion", "Feature-level fusion"), "late_fusion": ("late_fusion", "Decision-level fusion")}
    experiments = [{"id": names[name][0], "name": names[name][1], "model": name, "description": descriptions[name],
                    "evaluation_split": "test", "metrics": binary_metrics(y[masks["test"]], scores[name][masks["test"]])} for name in MODEL_NAMES]
    image_dim = matrices["image_only"].shape[1]
    text_dim = matrices["text_only"].shape[1]
    head = LinearHead(models["feature_fusion"])
    ablations = []
    columns = {"without_text": list(range(image_dim, image_dim + text_dim)),
               "without_image": list(range(image_dim))}
    for name, indices in columns.items():
        altered = matrices["feature_fusion"][masks["test"]].copy()
        altered[:, indices] = head.mean[indices]
        ablations.append({"id": name, "name": name.replace("_", " ").title(),
                          "description": "Replace the removed modality and its joint features with training means; no retraining.",
                          "metrics": binary_metrics(y[masks["test"]], head.predict_proba(altered))})
    experiments += [{"id": "modality_ablation", "name": "Modality ablation", "model": "feature_fusion", "evaluation_split": "test",
                     "description": "Inference ablation of the trained fusion model using training means; this intervention is outside the original input distribution.",
                     "metrics": ablations[0]["metrics"], "variants": ablations},
                    {"id": "ood_generalization", "name": "External-domain evaluation", "model": "feature_fusion", "evaluation_split": None,
                     "description": "Not evaluated: this run has no independently held-out source domain or deepfake benchmark.",
                     "metrics": None, "variants": [], "status": "not_available"}]
    difficult = []
    test_indices = np.flatnonzero(masks["test"])
    for label in (0, 1):
        indices = [i for i in test_indices if y[i] == label]
        for i in sorted(indices, key=lambda index: abs(scores["feature_fusion"][index] - .5))[:8]:
            row, p = records[i], float(scores["feature_fusion"][i])
            difficult.append({"id": row["id"], "text": row.get("text", ""), "image_path": row.get("image_path", ""),
                              "label": row["label"], "prediction": "FAKE" if p >= .5 else "REAL", "fake_probability": p,
                              "correct": (p >= .5) == bool(y[i]), "manipulation": row.get("source_label_name", "dataset label"),
                              "explanation": "Low-margin fusion prediction on inherited corpus labels; no human-verified semantic explanation is available.", "evidence": []})
    supplied_limits = manifest.get("limitations", [])
    if isinstance(supplied_limits, str):
        supplied_limits = [supplied_limits]
    limitations = list(supplied_limits) + audit["limitations"] + [
        "This is a subset/mirror evaluation unless the dataset manifest explicitly establishes otherwise; full published benchmark performance is not claimed.",
        "Dataset labels may reflect weak supervision or source conventions; the classifier does not establish real-world truth.",
        "The image branch uses frozen CLIP vision features; the text branch is a separate train-fitted TF-IDF plus logistic regression model.",
        "Scores are uncalibrated dataset class estimates; arbitrary uploads have not been validated as in-domain.",
        "No real-world face-swap, synthetic speech, video, human explanation, or external-domain evaluation was performed."]
    result = {"project": "VeriFuse", "profile": "real", "real_data": True, "controlled_benchmark": False,
              "benchmark": dataset_name, "seed": manifest.get("seed", 42), "dataset_sha256": bundle["dataset_sha256"],
              "dataset": {"total_records": len(records), "scene_groups": sum(s["groups"] for s in audit["splits"].values()),
                          "splits": audit["splits"], "label_definition": label_definition,
                          "group_disjoint": audit["supplied_source_identifiers_disjoint"], "pixel_hash_disjoint": audit["pixel_hash_disjoint"],
                          "audit": audit, "limitations": limitations, "provenance": manifest.get("provenance", manifest.get("dataset", {}))},
              "experiments": experiments, "validation_metrics": validation_metrics, "selected_model": selected_model,
              "majority_baseline": majority_baseline,
              "validation_trials": {"regularization": {name: values["trials"] for name, values in models.items()},
                                    "late_fusion": [{"alpha_text": float(a), "validation_log_loss": loss} for a, loss in zip(alphas, losses)]},
              "models": {"features": {name: matrix.shape[1] for name, matrix in matrices.items()}, "training": bundle["fit"],
                         "late_fusion_alpha": alpha, "threshold": .5, "selected_model": selected_model,
                         "selection": "Lowest validation log loss among the four models; test labels unused for selection.",
                         "score_meaning": "Uncalibrated probability estimate for the dataset's FAKE label."},
              "encoder": encoder_metadata, "difficult_cases": sorted(difficult, key=lambda row: abs(row["fake_probability"] - .5)),
              "difficult_case_selection": "Eight lowest-margin test examples per class from feature fusion, after training was fixed.",
              "faithfulness": {}, "sanity_checks": [{"id": "supplied_split_identifiers", "passed": True, "detail": audit["checked_fields"]}],
              "notes": ["FAKE is the positive class. Confusion matrices use true rows/predicted columns in order REAL, FAKE.",
                        "PR-AUC is average precision. All binary test metrics use a fixed 0.5 threshold.",
                        "No region or token explanation faithfulness result is claimed for this real-data model."],
              "training_seconds": round(time.perf_counter() - started, 3),
              "training_started_at_utc": started_at, "completed_at_utc": utc_now(), "runtime_versions": runtime_versions()}
    for field in ("source_label", "source_label_name", "six_way_label", "6_way_label"):
        if all(field in records[i] for i in test_indices):
            result["test_by_source_label"] = []
            for value in sorted({str(records[i][field]) for i in test_indices}):
                selected = np.asarray([i for i in test_indices if str(records[i][field]) == value])
                p = scores["feature_fusion"][selected]
                result["test_by_source_label"].append({"source_label": value, "count": len(selected), "accuracy": float(np.mean((p >= .5) == y[selected]))})
            break
    predictions = [{"id": row["id"], "split": row["split"], "label": row["label"],
                    "probabilities": {name: float(values[i]) for name, values in scores.items()}} for i, row in enumerate(records)]
    _atomic_json(root / "models/real/bundle.json", bundle)
    _atomic_json(output / "experiments.json", result)
    _atomic_json(output / "predictions.json", predictions)
    write_report(result, output / "report.md")
    print(f"Real-data training complete. Validation-selected model: {selected_model}", flush=True)
    return result


def write_report(result, path):
    lines = ["# VeriFuse: real-data training report", "", result["benchmark"], "", result["dataset"]["label_definition"], "",
             "The text model is train-fitted TF-IDF with an L2 logistic classifier. The image model uses frozen CLIP vision features with a separately fitted L2 logistic classifier.", "",
             "## Dataset partitions", "", "| Partition | Pairs | REAL | FAKE |", "|---|---:|---:|---:|"]
    for name, split in result["dataset"]["splits"].items():
        lines.append(f"| {name} | {split['records']} | {split['REAL']} | {split['FAKE']} |")
    lines += ["", "Scalers and heads fit on training only. Validation log loss selects regularization, the late-fusion weight, and the recommended model. Test labels are not used for selection.", "",
              "## Held-out test measurements", "", "| Model | Accuracy | F1 (FAKE) | Macro F1 | ROC-AUC | Average precision | Log loss |", "|---|---:|---:|---:|---:|---:|---:|"]
    baseline = result["majority_baseline"]
    lines.append("| Training-majority baseline | " + " | ".join(f"{baseline['metrics'][k]:.4f}" if baseline["metrics"][k] is not None else "N/A" for k in ("accuracy", "f1", "macro_f1", "roc_auc", "pr_auc", "log_loss")) + " |")
    for experiment in result["experiments"][:4]:
        metric = experiment["metrics"]
        lines.append("| " + experiment["name"] + " | " + " | ".join(f"{metric[k]:.4f}" if metric[k] is not None else "N/A" for k in ("accuracy", "f1", "macro_f1", "roc_auc", "pr_auc", "log_loss")) + " |")
    lines += ["", f"Validation-selected model: **{result['selected_model']}**. Fixed decision threshold: 0.5. Scores are uncalibrated class estimates.", "",
              f"The majority baseline always predicts **{baseline['class']}**, chosen from the training distribution. Its test accuracy is **{baseline['test_accuracy']:.1%}**; inspect macro F1 and each class's recall when labels are imbalanced.", "",
              "Confusion matrices use true rows/predicted columns, in REAL then FAKE order.", ""]
    for experiment in result["experiments"][:4]:
        lines.append(f"- {experiment['name']}: `{experiment['metrics']['confusion_matrix']}`")
    selected = next(e for e in result["experiments"][:4] if e["model"] == result["selected_model"])
    lines += ["", "## Selected model by class", "", "| Dataset class | Precision | Recall | F1 | Test examples |", "|---|---:|---:|---:|---:|"]
    for label, values in selected["metrics"]["per_class"].items():
        lines.append(f"| {label} | {values['precision']:.4f} | {values['recall']:.4f} | {values['f1']:.4f} | {values['support']} |")
    lines += ["", f"Balanced accuracy: **{selected['metrics']['balanced_accuracy']:.4f}**. The majority class has greater recall; overall accuracy alone hides the lower recall on REAL-labeled posts."]
    lines += ["", "## Audits and limits", "", "Checked source/hash fields: " + ", ".join(result["dataset"]["audit"]["checked_fields"]) + ".", ""]
    lines.extend("- " + limitation for limitation in result["dataset"]["limitations"])
    provenance = result["dataset"].get("provenance", {})
    lines += ["", "## Sources", "", "- [Fakeddit paper](https://aclanthology.org/2020.lrec-1.755/)",
              "- [Community data mirror](https://huggingface.co/datasets/ams-99/fakeddit_9k)",
              "- [Author-provided label mapping](https://github.com/entitize/Fakeddit/issues/14#issuecomment-815355653)",
              "- [Pretrained CLIP image encoder](https://huggingface.co/openai/clip-vit-base-patch32)", "",
              f"Dataset mirror revision: `{provenance.get('revision', 'not supplied')}`. Download checksums and exact exclusions are retained locally; see `docs/REAL_DATASET.md`."]
    lines += ["", "## Reproduction", "", "Run `python scripts/train_real.py` after preparing `data/real/manifest.json` and the pinned local encoder in `models/real/clip`. Extraction checkpoints resume automatically by image/text content. Use `--embeddings` only for a documented supplied feature file aligned by record ID.", "",
              f"Dataset manifest SHA-256: `{result['dataset_sha256']}`.", "",
              f"Training started (UTC): {result['training_started_at_utc']}. Head fitting/evaluation wall time: {result['training_seconds']:.3f} seconds.", "",
              f"CLIP revision: `{result['encoder'].get('revision', 'not supplied')}`. Extraction wall time for the recorded invocation: {result['encoder'].get('encoding_seconds', 'not recorded')} seconds. {result['encoder'].get('timing_scope', '')}", "",
              "Runtime versions: `" + json.dumps(result["runtime_versions"], sort_keys=True) + "`.", "",
              "Raw class scores for every partition are saved in `predictions.json`; the complete metrics, validation trials, ablations, and low-margin test cases are saved in `experiments.json`. No external-domain or explanation-faithfulness measurement is claimed."]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


class RealVerificationEngine:
    """UI-compatible inference for the separately trained real-media profile."""
    def __init__(self, root, encoder=None):
        self.root = Path(root)
        self.encoder = encoder
        self._bundle = None
        self._load_lock = threading.Lock()

    def _load(self):
        if self._bundle is not None:
            return
        with self._load_lock:
            if self._bundle is not None:
                return
            path = self.root / "models/real/bundle.json"
            if not path.exists():
                raise FileNotFoundError("Real-data model missing. Run python scripts/train_real.py first.")
            bundle = json.loads(path.read_text(encoding="utf-8"))
            if bundle.get("profile") != "real" or bundle.get("feature_order") != FEATURE_ORDER:
                raise ValueError("Unsupported real-data model bundle.")
            self.models = {name: LinearHead(value) for name, value in bundle["models"].items()}
            self.text_vectorizer = load_text_vectorizer(bundle["text_vectorizer"])
            self.alpha = float(bundle["late_fusion_alpha"])
            if not 0 <= self.alpha <= 1:
                raise ValueError("Invalid late-fusion weight.")
            if self.encoder is None:
                encoder_path = self.root / bundle["encoder"].get("local_path", "models/real/clip")
                if not (encoder_path / "config.json").is_file() or not any((encoder_path / name).is_file() for name in ("pytorch_model.bin", "model.safetensors")):
                    raise FileNotFoundError("Local CLIP encoder missing. Run python scripts/download_clip.py first.")
                self.encoder = ClipEncoder(encoder_path)
            self._bundle = bundle

    def scores_from_embeddings(self, image, text):
        self._load()
        matrices = feature_matrices(image, text)
        scores = {name: self.models[name].predict_proba(matrix) for name, matrix in matrices.items()}
        scores["late_fusion"] = self.alpha * scores["text_only"] + (1 - self.alpha) * scores["image_only"]
        return scores

    def predict(self, image, text, model="feature_fusion"):
        started = time.perf_counter()
        self._load()
        if model not in MODEL_NAMES:
            raise ValueError("Choose a supported model.")
        if not isinstance(text, str) or len(text) > 10000:
            raise ValueError("Caption must be text, up to 10,000 characters.")
        limits = ["Experimental score for the real corpus's inherited labels; it is not proof of authenticity or factual truth.",
                  "Text uses TF-IDF plus logistic regression; images use frozen CLIP vision features. Arbitrary uploads and external domains have not been validated.",
                  "Scores are uncalibrated. Token and image-region explanations are not available for this profile."]
        result = {"profile": "real", "dataset_name": self._bundle["dataset_name"], "label_definition": self._bundle["label_definition"],
                  "prediction": "UNCERTAIN", "fake_probability": None, "confidence": None, "model": model,
                  "explanation": "Provide both an image and a caption for this corpus-trained model.", "evidence": [],
                  "modality_scores": {"text": None, "image": None}, "warnings": limits, "in_domain": None,
                  "token_attributions": [], "image_regions": []}
        if image is None or not text.strip():
            limits.append("A modality is missing; no class score is produced.")
        else:
            image_features = self.encoder.encode_images([image])
            text_features = self.text_vectorizer.transform([text]).toarray()
            scores = {name: float(values[0]) for name, values in self.scores_from_embeddings(image_features, text_features).items()}
            p = scores[model]
            result.update({"prediction": "FAKE" if p >= .5 else "REAL", "fake_probability": p, "confidence": max(p, 1 - p),
                           "explanation": f"The {model.replace('_', ' ')} classifier estimates {p:.1%} for the dataset's FAKE label. The image is represented by frozen CLIP vision features and the caption by TF-IDF features. This prediction is a learned corpus classification, not a verified factual verdict.",
                           "modality_scores": {"text": scores["text_only"], "image": scores["image_only"]},
                           "evidence": [{"type": "text", "title": "Caption-only TF-IDF classifier", "impact": 0.0,
                                         "detail": f"Dataset FAKE score from caption features alone: {scores['text_only']:.1%}."},
                                        {"type": "image", "title": "CLIP image-only classifier", "impact": 0.0,
                                         "detail": f"Dataset FAKE score from image features alone: {scores['image_only']:.1%}."}]})
        result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return result
