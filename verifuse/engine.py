"""Local inference with explicit support boundaries and perturbation explanations."""
from __future__ import annotations
import json
import re
import threading
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from .features import COLORS, SHAPES, COUNTS, COLOR_ALIASES, SHAPE_ALIASES, Tfidf, cross_features, image_attributes, parse_caption, tokenize
from .learning import LogisticModel

MODEL_NAMES = ("text_only", "image_only", "feature_fusion", "late_fusion")
FILLER = {"this", "image", "shows", "the", "scene", "contains", "on", "a", "an", "light", "background", "picture", "of", "visible", "in", "there", "is", "are", "with", "and", "object", "objects"}


def caption_supported(text: str) -> bool:
    cap = parse_caption(text)
    known = set(COLORS) | set(SHAPES) | set(COUNTS) | set(COLOR_ALIASES) | set(SHAPE_ALIASES) | FILLER
    return bool(text.strip() and len(tokenize(text)) <= 28 and all(cap[k] is not None for k in ("color", "shape", "count")) and not cap["negated"] and not cap["ambiguous"] and all(t in known for t in tokenize(text)))


class VerificationEngine:
    def __init__(self, root: Path):
        self.root = Path(root)
        self._bundle = None
        self._load_lock = threading.Lock()

    def _load(self):
        if self._bundle is None:
            with self._load_lock:
                if self._bundle is None:
                    path = self.root / "models" / "bundle.json"
                    if not path.exists():
                        raise FileNotFoundError("Trained model artifacts are missing. Run: python scripts/generate_dataset.py && python scripts/run_experiments.py")
                    bundle = json.loads(path.read_text(encoding="utf-8"))
                    self.tfidf = Tfidf(**bundle["tfidf"])
                    self.models = {name: LogisticModel.from_dict(value) for name, value in bundle["models"].items()}
                    self.alpha = float(bundle["late_fusion_alpha"])
                    self._bundle = bundle

    def raw_scores(self, image: Image.Image | None, text: str, visual: dict | None = None) -> dict:
        self._load()
        visual = image_attributes(image) if visual is None else visual
        tx = self.tfidf.transform([text])
        ix = np.asarray(visual["features"])[None, :]
        cx = cross_features(parse_caption(text), visual)[None, :]
        text_p = float(self.models["text_only"].predict_proba(tx)[0])
        image_p = float(self.models["image_only"].predict_proba(ix)[0])
        fusion_p = float(self.models["feature_fusion"].predict_proba(np.hstack([tx, ix, cx]))[0])
        return {"text_only": text_p, "image_only": image_p, "feature_fusion": fusion_p, "late_fusion": self.alpha * text_p + (1 - self.alpha) * image_p}

    def predict(self, image: Image.Image | None, text: str, model: str = "feature_fusion") -> dict:
        started = time.perf_counter()
        if model not in MODEL_NAMES:
            raise ValueError("Unknown model. Choose " + ", ".join(MODEL_NAMES))
        if not isinstance(text, str):
            raise ValueError("Text must be a string.")
        if len(text) > 10000:
            raise ValueError("Please use at most 10,000 characters.")
        visual, cap = image_attributes(image), parse_caption(text)
        scores = self.raw_scores(image, text, visual)
        p = scores[model]
        in_domain = bool(visual["in_domain"] and caption_supported(text))
        confidence = float(max(p, 1 - p))
        prediction = ("FAKE" if p >= .5 else "REAL") if in_domain and confidence >= .58 else "UNCERTAIN"
        warnings = ["Experimental synthetic-scene estimate. REAL means caption/scene consistency here; it does not establish real-world news truth."]
        if not in_domain:
            warnings.append("Unsupported input: use a simple generated scene containing one to three same-color circles, squares, or triangles and a literal caption. External photographs and general news require a separately validated model.")
        if model in ("text_only", "image_only", "late_fusion"):
            warnings.append("This baseline has no joint attribute comparison and is not reliable for detecting semantic image-caption contradictions.")
        if in_domain and confidence < .58:
            warnings.append("The learned score is too close to the decision boundary for a decisive demo label.")
        evidence = []
        mismatches = []
        if image is not None:
            evidence.append({"type": "image", "title": "Observed image attributes", "detail": f"Pixel analysis estimates {visual['count']} {visual['color']} {visual['shape']}(s). {visual['reason']}", "impact": 0.0})
        evidence.append({"type": "text", "title": "Caption attributes", "detail": f"Caption parser found count={cap['count']}, color={cap['color']}, shape={cap['shape']}.", "impact": 0.0})
        if in_domain:
            for attr in ("color", "shape", "count"):
                agrees = cap[attr] == visual[attr]
                if not agrees:
                    mismatches.append(f"{attr}: caption says {cap[attr]}, image shows {visual[attr]}")
                changed = dict(cap)
                if agrees:
                    values = list(COLORS) if attr == "color" else list(SHAPES) if attr == "shape" else [1, 2, 3]
                    changed[attr] = values[(values.index(cap[attr]) + 1) % len(values)]
                else:
                    changed[attr] = visual[attr]
                # Isolate the cross-modal feature family; text and image encoders stay fixed.
                tx, ix = self.tfidf.transform([text]), visual["features"][None, :]
                altered = np.hstack([tx, ix, cross_features(changed, visual)[None, :]])
                changed_p = float(self.models["feature_fusion"].predict_proba(altered)[0])
                impact = p - changed_p if model == "feature_fusion" else 0.0
                evidence.append({"type": "cross_modal", "title": attr.title() + (" agrees" if agrees else " conflicts"), "detail": f"Caption: {cap[attr]}; observed image: {visual[attr]}. Impact is the probability change when only this comparison is counterfactually flipped.", "impact": round(float(impact), 6)})
        if not in_domain:
            explanation = "UNCERTAIN: this input is outside the validated synthetic-scene task. The raw model score is shown for inspection and must not be interpreted as a factual verdict."
        elif model != "feature_fusion":
            explanation = "This learned baseline scores " + ("caption wording" if model == "text_only" else "image attributes" if model == "image_only" else "a validation-selected blend of independent text and image scores") + ". It does not compare the two modalities directly. " + ("Observed conflicts: " + "; ".join(mismatches) + "." if mismatches else "The parsed attributes agree.")
        elif mismatches:
            explanation = "The feature-fusion model finds a caption/scene contradiction: " + "; ".join(mismatches) + ". This supports FAKE only under the controlled dataset's consistency definition."
        else:
            explanation = "The caption and rendered scene agree on color, shape, and count. The feature-fusion model supports REAL under the controlled dataset's consistency definition."
        token_attributions = []
        if text.strip() and in_domain:
            for match in list(re.finditer(r"[A-Za-z]+|\d+", text))[:28]:
                reduced = text[:match.start()] + text[match.end():]
                delta = p - self.raw_scores(image, reduced, visual)[model]
                token_attributions.append({"token": match.group(), "impact": round(float(delta), 6)})
        regions = []
        if image is not None and in_domain:
            for region in visual["regions"][:3]:
                occluded = image.convert("RGB").copy()
                box = (int(region["x"] * image.width) - 2, int(region["y"] * image.height) - 2, int((region["x"] + region["width"]) * image.width) + 2, int((region["y"] + region["height"]) * image.height) + 2)
                ImageDraw.Draw(occluded).rectangle(box, fill=(246, 245, 240))
                delta = p - self.raw_scores(occluded, text)[model]
                regions.append({**{k: region[k] for k in ("x", "y", "width", "height")}, "impact": round(float(delta), 6)})
        return {"prediction": prediction, "fake_probability": float(p), "confidence": confidence, "model": model, "explanation": explanation, "evidence": evidence, "modality_scores": {"text": scores["text_only"] if text.strip() else None, "image": scores["image_only"] if image is not None else None}, "warnings": warnings, "in_domain": in_domain, "elapsed_ms": round((time.perf_counter() - started) * 1000, 2), "token_attributions": token_attributions, "image_regions": regions}
