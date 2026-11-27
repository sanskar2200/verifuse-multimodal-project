"""Training, grouped evaluation, ablation and explanation diagnostics."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .dataset import SEED, caption, validate_records
from .features import CROSS_NAMES, IMAGE_NAMES, Tfidf, cross_features, image_attributes, parse_caption, tokenize
from .learning import LogisticModel, metrics
from .engine import VerificationEngine


def prepare(root: Path, records: list[dict], tfidf: Tfidf) -> dict:
    cache, visuals = {}, []
    for row in records:
        key = row["image_path"]
        if key not in cache:
            with Image.open(root / key) as image:
                cache[key] = image_attributes(image)
        visuals.append(cache[key])
    tx = tfidf.transform([r["text"] for r in records])
    ix = np.asarray([v["features"] for v in visuals])
    cx = np.asarray([cross_features(parse_caption(r["text"]), v) for r, v in zip(records, visuals)])
    return {"text_only": tx, "image_only": ix, "feature_fusion": np.hstack([tx, ix, cx]), "visuals": visuals}


def train_bundle(root: Path, manifest: dict, output: Path, controlled: bool = True) -> tuple[dict, dict]:
    records = manifest["records"]
    validate_records(records)
    pixel_splits, hashes = {}, {}
    for row in records:
        if row["image_path"] not in hashes:
            with Image.open(root / row["image_path"]) as image:
                decoded = image.convert("RGB")
                hashes[row["image_path"]] = hashlib.sha256(str(decoded.size).encode() + decoded.tobytes()).hexdigest()
        digest = hashes[row["image_path"]]
        if digest in pixel_splits and pixel_splits[digest] != row["split"]:
            raise ValueError("Identical image pixels cross dataset splits: " + row["image_path"])
        pixel_splits[digest] = row["split"]
    train = [r for r in records if r["split"] == "train"]
    val = [r for r in records if r["split"] == "val"]
    test = [r for r in records if r["split"] == "test"]
    for name, subset in (("train", train), ("val", val), ("test", test)):
        if len({r["label"] for r in subset}) != 2:
            raise ValueError(f"{name} split requires both REAL and FAKE examples.")
    tfidf = Tfidf().fit([r["text"] for r in train])
    matrices = prepare(root, records, tfidf)
    y = np.asarray([r["label"] == "FAKE" for r in records], dtype=float)
    train_mask = np.array([r["split"] == "train" for r in records])
    val_mask = np.array([r["split"] == "val" for r in records])
    models, scores = {}, {}
    for name in ("text_only", "image_only", "feature_fusion"):
        print(f"Fitting {name} on {int(train_mask.sum())} records...", flush=True)
        fitted = LogisticModel().fit(matrices[name][train_mask], y[train_mask])
        models[name] = fitted.to_dict()
        scores[name] = fitted.predict_proba(matrices[name])
    candidates = np.linspace(0, 1, 21)
    def log_loss(alpha):
        p = np.clip(alpha * scores["text_only"][val_mask] + (1 - alpha) * scores["image_only"][val_mask], 1e-8, 1 - 1e-8)
        return float(-np.mean(y[val_mask] * np.log(p) + (1 - y[val_mask]) * np.log(1 - p)))
    alpha = float(min(candidates, key=log_loss))
    scores["late_fusion"] = alpha * scores["text_only"] + (1 - alpha) * scores["image_only"]
    digest = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    scope = "controlled colored-shape image-caption consistency" if controlled else "imported corpus labels with an engineered colored-shape encoder; external validity unverified"
    bundle = {"version": 1, "seed": manifest.get("seed", SEED), "dataset_sha256": digest, "scope": scope, "tfidf": tfidf.to_dict(), "models": models, "late_fusion_alpha": alpha, "feature_names": {"text_only": list(tfidf.vocabulary), "image_only": IMAGE_NAMES, "feature_fusion": list(tfidf.vocabulary) + IMAGE_NAMES + CROSS_NAMES}, "decision_threshold": .5, "demo_uncertainty_threshold": .58 if controlled else None, "fit": {"regularization": .03, "steps": 1400, "optimizer": "full-batch Adam", "learning_rate": .07, "text_fit_split": "train", "late_fusion_selection_split": "val"}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    return bundle, {"matrices": matrices, "scores": scores, "y": y}


def run(root: Path, manifest: dict, output: Path | None = None, controlled: bool = True) -> dict:
    root = Path(root)
    output = output or root / "results"
    output.mkdir(parents=True, exist_ok=True)
    model_path = root / "models" / "bundle.json" if controlled else output / "bundle.json"
    bundle, fitted = train_bundle(root, manifest, model_path, controlled=controlled)
    records, scores, matrices, y = manifest["records"], fitted["scores"], fitted["matrices"], fitted["y"]
    mask = np.array([r["split"] == "test" for r in records])
    ood_mask = np.array([r["split"] == "ood_test" for r in records])
    experiments = []
    for eid, name, model, description in [
        ("text_baseline", "Text-only baseline", "text_only", "Train-only unigram/bigram TF-IDF followed by L2 logistic regression."),
        ("image_baseline", "Image-only baseline", "image_only", "Pixel-derived color, shape, count, foreground fraction, fill and RGB features followed by L2 logistic regression."),
        ("feature_fusion", "Feature-level fusion", "feature_fusion", "Concatenated TF-IDF, image attributes and explicit cross-modal contradiction features followed by L2 logistic regression."),
        ("late_fusion", "Decision-level fusion", "late_fusion", "Independent text/image probabilities blended with a weight selected by validation log loss.")]:
        experiments.append({"id": eid, "name": name, "description": description, "model": model, "evaluation_split": "test", "metrics": metrics(y[mask], scores[model][mask])})
    fusion = LogisticModel.from_dict(bundle["models"]["feature_fusion"])
    ntext, nimage = len(bundle["tfidf"]["vocabulary"]), len(IMAGE_NAMES)
    ablation = []
    for name, columns, description in [
        ("without_text", list(range(ntext)) + list(range(ntext + nimage, len(fusion.mean))), "Replace text and cross-modal features with their training means."),
        ("without_image", list(range(ntext, len(fusion.mean))), "Replace image and cross-modal features with their training means."),
        ("without_cross_modal", list(range(ntext + nimage, len(fusion.mean))), "Replace only joint contradiction features with their training means.")]:
        x = matrices["feature_fusion"][mask].copy()
        x[:, columns] = fusion.mean[columns]
        ablation.append({"id": name, "name": name.replace("_", " ").title(), "description": description, "metrics": metrics(y[mask], fusion.predict_proba(x))})
    experiments.append({"id": "modality_ablation", "name": "Modality ablation", "description": "Fixed trained fusion model; missing feature families replaced by training means. This is an inference ablation, not retraining.", "model": "feature_fusion", "evaluation_split": "test", "metrics": ablation[0]["metrics"], "variants": ablation})
    variants = [{"id": name, "name": name, "metrics": metrics(y[ood_mask], values[ood_mask])} for name, values in scores.items()] if ood_mask.any() else []
    if controlled:
        ood_description = "Count alterations never occur among train/validation/test negatives. OOD has entirely new scene groups and matching REAL controls. This is manipulation OOD within synthetic shapes, not real-photo OOD."
    elif ood_mask.any():
        ood_description = "Evaluation on the corpus author's ood_test split. Provided group IDs and exact image pixels are disjoint from other splits. This does not establish an unseen manipulation, a shifted source distribution, or factual validity; those properties require a separately documented corpus protocol."
    else:
        ood_description = "Not evaluated: the imported corpus supplies no ood_test records. No out-of-distribution or held-out-manipulation result is claimed."
    experiments.append({"id": "ood_generalization", "name": "Held-out manipulation" if controlled else "Author-designated OOD split", "description": ood_description, "model": "feature_fusion", "evaluation_split": "ood_test", "metrics": metrics(y[ood_mask], scores["feature_fusion"][ood_mask]) if ood_mask.any() else None, "variants": variants, "status": "evaluated" if ood_mask.any() else "not_available"})
    splits = {}
    for name in ("train", "val", "test", "ood_test"):
        rows = [r for r in records if r["split"] == name]
        splits[name] = {"records": len(rows), "groups": len({r["group_id"] for r in rows}), "REAL": sum(r["label"] == "REAL" for r in rows), "FAKE": sum(r["label"] == "FAKE" for r in rows)}
    result = {"project": "VeriFuse", "benchmark": manifest["name"], "seed": manifest.get("seed", SEED), "dataset_sha256": bundle["dataset_sha256"], "dataset": {"total_records": len(records), "scene_groups": len({r["group_id"] for r in records}), "splits": splits, "label_definition": manifest["label_definition"], "group_disjoint": True, "pixel_hash_disjoint": True, "limitations": ["Controlled synthetic benchmark; no accuracy claim on real-world news or photos.", "Matching captions and procedural source scenes define REAL.", "Explicit attribute extraction makes this task much easier than general multimodal misinformation.", "Labels, paths, ids, manipulation names, source and split are never input features."]}, "experiments": experiments, "models": {"features": {k: len(v) for k, v in bundle["feature_names"].items()}, "training": bundle["fit"], "late_fusion_alpha": bundle["late_fusion_alpha"], "threshold": .5}, "difficult_cases": [], "faithfulness": {}, "sanity_checks": [], "notes": ["FAKE is positive class; confusion matrices have true rows and predicted columns in order REAL, FAKE.", "PR-AUC is average precision (step integration) with tied scores grouped.", "Reported test metrics force a binary label at 0.5; interactive demo additionally abstains outside its domain or below 0.58 confidence."]}
    if not controlled:
        result["dataset"]["limitations"] = ["Labels, source provenance and the intended meaning of REAL/FAKE are supplied by the corpus author and are not independently verified.", "The image encoder and joint comparisons remain designed for colored geometric shapes; importing arbitrary images does not validate these features for that domain.", "Provided group IDs and exact decoded pixels are checked across splits, but event/source identity and distribution shift require separate corpus auditing.", "Labels, paths, ids, manipulation names, source and split are never input features."]
        result["notes"][-1] = "Reported metrics force a binary label at 0.5. No confidence calibration, abstention threshold or demo deployment is validated for this imported model."
    if controlled:
        engine = VerificationEngine(root)
        test_indices = np.flatnonzero(mask)
        selected = []
        for label in (0, 1):
            candidates = [i for i in test_indices if y[i] == label]
            selected.extend(sorted(candidates, key=lambda i: abs(scores["feature_fusion"][i] - .5))[:8])
        print("Evaluating 16 low-margin cases and explanation faithfulness...", flush=True)
        for index in sorted(selected, key=lambda i: abs(scores["feature_fusion"][i] - .5)):
            row = records[index]
            with Image.open(root / row["image_path"]) as image:
                pred = engine.predict(image, row["text"])
            result["difficult_cases"].append({"id": row["id"], "text": row["text"], "image_path": row["image_path"], "label": row["label"], "prediction": pred["prediction"], "fake_probability": pred["fake_probability"], "correct": pred["prediction"] == row["label"], "explanation": pred["explanation"], "manipulation": row["manipulation"], "evidence": pred["evidence"]})
        result["difficult_case_selection"] = "The 8 smallest-margin test pairs from each class (16 total), selected after all model fitting. These are relative low-confidence cases, not necessarily errors."
        relevant, filler, correction_drops, introduced_rises = [], [], [], []
        for index in test_indices[:48]:
            row, visual, p = records[index], matrices["visuals"][index], scores["feature_fusion"][index]
            text = row["text"]
            for word in dict.fromkeys(tokenize(text)):
                reduced = " ".join(w for w in tokenize(text) if w != word)
                delta = abs(float(p) - engine.raw_scores(None, reduced, visual)["feature_fusion"])
                (relevant if word in {"red", "blue", "green", "yellow", "circle", "circles", "square", "squares", "triangle", "triangles", "one", "two", "three"} else filler).append(delta)
            if row["label"] == "FAKE":
                corrected = caption({k: visual[k] for k in ("color", "shape", "count")}, 0)
                correction_drops.append(float(p - engine.raw_scores(None, corrected, visual)["feature_fusion"]))
            else:
                conflict = {k: visual[k] for k in ("color", "shape", "count")}
                colors = ["red", "blue", "green", "yellow"]
                conflict["color"] = colors[(colors.index(conflict["color"]) + 1) % 4]
                introduced_rises.append(float(engine.raw_scores(None, caption(conflict, 0), visual)["feature_fusion"] - p))
        result["faithfulness"] = {"protocol": "On the first 48 fixed test records, delete each distinct caption token and recompute predictions; compare attribute-token effects with filler-token effects. Correct false captions and introduce a color conflict into matching captions. These measure local perturbation sensitivity, not causal truth or human explanation quality.", "evaluated_records": 48, "mean_absolute_attribute_token_effect": float(np.mean(relevant)), "mean_absolute_filler_token_effect": float(np.mean(filler)), "relevance_pass": bool(np.mean(relevant) > np.mean(filler)), "mean_fake_probability_drop_after_correction": float(np.mean(correction_drops)), "correction_direction_pass_rate": float(np.mean(np.asarray(correction_drops) > 0)), "mean_fake_probability_rise_after_conflict": float(np.mean(introduced_rises)), "conflict_direction_pass_rate": float(np.mean(np.asarray(introduced_rises) > 0)), "image_region_method": "Each detected object rectangle is painted with the light background and rescored. Signed deltas are returned; occlusion can move an example off the training distribution.", "cross_modal_method": "Hold encoders fixed and counterfactually flip one attribute comparison; impact equals original minus changed FAKE probability. Baseline cross-modal impacts are zero."}
        original = next(r for r in records if r["split"] == "test" and r["label"] == "REAL")
        with Image.open(root / original["image_path"]) as im:
            supported = im.copy()
        rng = np.random.default_rng(SEED)
        unsupported = Image.fromarray(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8))
        cases = [("supported_matching_pair", supported, original["text"], "REAL"), ("no_image", None, original["text"], "UNCERTAIN"), ("blank_image", Image.new("RGB", (256, 256), "white"), original["text"], "UNCERTAIN"), ("noise_image", unsupported, original["text"], "UNCERTAIN"), ("missing_caption", supported, "", "UNCERTAIN"), ("negation_not_supported", supported, "This is not " + original["text"], "UNCERTAIN"), ("general_news_unsupported", supported, "A government confirmed a major economic announcement today.", "UNCERTAIN")]
        for name, image, text, expected in cases:
            predicted = engine.predict(image, text)
            result["sanity_checks"].append({"id": name, "expected": expected, "actual": predicted["prediction"], "passed": predicted["prediction"] == expected, "in_domain": predicted["in_domain"]})
        for modality in ("text", "image"):
            altered = next(r for r in records if r["split"] == "test" and r["label"] == "FAKE" and r["altered_modality"] == modality)
            with Image.open(root / altered["image_path"]) as image:
                predicted = engine.predict(image, altered["text"])
            result["sanity_checks"].append({"id": "deliberate_caption_mismatch" if modality == "text" else "manipulated_image_pair", "expected": "FAKE", "actual": predicted["prediction"], "passed": predicted["prediction"] == "FAKE", "in_domain": predicted["in_domain"]})
        result["domain_support"] = {"supported_test_images": sum(matrices["visuals"][i]["in_domain"] for i in test_indices), "total_test_images": len(test_indices)}
    result["controlled_benchmark"] = controlled
    (output / "experiments.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_report(result, output)
    draw_charts(result, output)
    return result


def write_report(result, output):
    scope = "This is a **controlled synthetic scene-consistency benchmark**, not evidence of real-world fake-news detection accuracy." if result.get("controlled_benchmark", True) else "This is an **imported-corpus experiment using the existing engineered shape encoder**. The labels and provenance were supplied by the CSV author. No real-world validity or news verification claim has been established."
    grouping = "source groups. Train/validation/test/OOD groups and decoded image pixels are disjoint across splits." if result.get("controlled_benchmark", True) else "provided group IDs. Provided group IDs and decoded image pixels are disjoint across the populated splits; source/event identity is not independently audited."
    lines = ["# VeriFuse reproducible experiment report", "", scope, "", result["dataset"]["label_definition"], "", "## Dataset and protocol", "", f"Seed: {result['seed']}. {result['dataset']['total_records']} pairs, {result['dataset']['scene_groups']} {grouping} TF-IDF and standardization fit only on train; validation chooses only the late-fusion blend. Test labels are never used in fitting.", "", "| Split | Pairs | Groups | REAL | FAKE |", "|---|---:|---:|---:|---:|"]
    for name, split in result["dataset"]["splits"].items():
        lines.append(f"| {name} | {split['records']} | {split['groups']} | {split['REAL']} | {split['FAKE']} |")
    lines += ["", "## Six experiments", "", "FAKE is the positive class. PR-AUC uses average precision with ties grouped. Binary metrics use 0.5; UI uncertainty handling is separate.", "", "| Experiment | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |", "|---|---:|---:|---:|---:|---:|---:|"]
    for exp in result["experiments"]:
        m = exp["metrics"]
        if m:
            lines.append("| " + exp["name"] + " | " + " | ".join(f"{m[k]:.4f}" if m[k] is not None else "N/A" for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")) + " |")
        else:
            lines.append("| " + exp["name"] + " (not evaluated) | " + " | ".join(["N/A"] * 6) + " |")
    for exp in result["experiments"]:
        lines += ["", "### " + exp["name"], "", exp["description"]]
        if exp.get("metrics"):
            lines += ["", f"Confusion matrix [[TN, FP], [FN, TP]]: `{exp['metrics']['confusion_matrix']}`."]
        for variant in exp.get("variants", []):
            lines += ["", f"- {variant['name']}: accuracy {variant['metrics']['accuracy']:.4f}, F1 {variant['metrics']['f1']:.4f}, matrix {variant['metrics']['confusion_matrix']}."]
    f = result["faithfulness"]
    if f:
        lines += ["", "## Explanation checks", "", f["protocol"], "", f"Mean absolute attribute-token effect: {f['mean_absolute_attribute_token_effect']:.4f}; filler-token effect: {f['mean_absolute_filler_token_effect']:.4f}. Correcting false captions reduced FAKE probability by {f['mean_fake_probability_drop_after_correction']:.4f} on average; expected direction rate {f['correction_direction_pass_rate']:.1%}. Introducing color conflicts increased FAKE probability by {f['mean_fake_probability_rise_after_conflict']:.4f}; direction rate {f['conflict_direction_pass_rate']:.1%}.", "", f["image_region_method"], "", f["cross_modal_method"]]
    lines += ["", "## Low-margin case analysis", "", result.get("difficult_case_selection", "Not evaluated for an imported corpus."), ""]
    for i, case in enumerate(result["difficult_cases"], 1):
        lines += [f"{i}. **{case['id']}** — truth {case['label']}, predicted {case['prediction']}, FAKE probability {case['fake_probability']:.4f}. Caption: {case['text']} {case['explanation']}", ""]
    lines += ["## Sanity checks", ""]
    for check in result["sanity_checks"]:
        lines.append(f"- {check['id']}: expected {check['expected']}; observed {check['actual']}; {'PASS' if check['passed'] else 'FAIL'}.")
    if result.get("controlled_benchmark", True):
        limits = "The fusion model receives explicit comparisons derived from image pixels and caption words. Its success demonstrates learning a narrow consistency task. OOD count manipulation remains within the same drawing domain. No pretrained vision-language network, external news dataset, human explanation study, probability calibration claim, or news benchmark result is implied. Single-modality baselines cannot reliably solve relational inconsistencies."
        reproduction = "Run `python scripts/generate_dataset.py` and `python scripts/run_experiments.py`. All training hyperparameters, threshold curves, per-class metrics, group metadata, weights and raw results are saved locally. `docs/DATASET.md` documents importing an external CSV."
    else:
        limits = "The reported scores compare predictions against supplied corpus labels. Label semantics, source provenance, event identity, domain shift and factual validity have not been independently established. The encoder still estimates colored-shape attributes; it may be unsuitable for this corpus. The name ood_test is an author-provided split designation and does not prove any particular held-out manipulation or distribution shift. No probability calibration, explanation validation or real-world deployment claim is made."
        reproduction = "Re-run `python scripts/run_experiments.py --csv path/to/the-same-corpus.csv --output results/imported` with the same CSV and image files. Saved settings, threshold curves, per-class metrics, weights and metadata hash describe this run. `docs/DATASET.md` documents the schema and grouping requirements."
    lines += ["", "## Limits and reproducibility", "", limits, "", reproduction, "", f"Dataset metadata SHA-256: `{result['dataset_sha256']}`."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def draw_charts(result, output):
    image = Image.new("RGB", (1200, 660), "#f7f8fc")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 24)
        small = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)
        title = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 32)
    except OSError:
        font = small = title = ImageFont.load_default()
    draw.text((48, 30), "VeriFuse / " + ("controlled synthetic benchmark" if result.get("controlled_benchmark", True) else "imported corpus experiment"), fill="#1b2341", font=title)
    draw.text((48, 82), "Held-out test accuracy. These results do not measure real-world news truth.", fill="#52617a", font=small)
    exps = result["experiments"][:4]
    for i, exp in enumerate(exps):
        y = 150 + i * 100
        draw.text((48, y + 12), exp["name"], fill="#243451", font=font)
        draw.rounded_rectangle((380, y, 1090, y + 56), radius=10, fill="#e5e9f2")
        width = int(710 * exp["metrics"]["accuracy"])
        draw.rounded_rectangle((380, y, 380 + max(width, 2), y + 56), radius=10, fill="#7565d8" if i == 2 else "#339fa0")
        draw.text((390, y + 14), f"{exp['metrics']['accuracy']:.1%}", fill="white", font=font)
    footer = "Grouped scene split | balanced REAL / FAKE | decision threshold 0.5" if result.get("controlled_benchmark", True) else "Provided group IDs | corpus-author labels | decision threshold 0.5"
    draw.text((48, 594), footer, fill="#52617a", font=small)
    image.save(output / "model_comparison.png")
    matrix = result["experiments"][2]["metrics"]["confusion_matrix"]
    image = Image.new("RGB", (740, 680), "#f7f8fc")
    draw = ImageDraw.Draw(image)
    draw.text((38, 28), "Feature fusion / confusion matrix", fill="#1b2341", font=title)
    draw.text((210, 98), "Predicted REAL", fill="#52617a", font=small)
    draw.text((455, 98), "Predicted FAKE", fill="#52617a", font=small)
    for r in range(2):
        draw.text((28, 230 + r * 220), "True " + ("REAL" if r == 0 else "FAKE"), fill="#52617a", font=small)
        for c in range(2):
            x, y = 190 + c * 240, 140 + r * 220
            draw.rounded_rectangle((x, y, x + 216, y + 195), radius=16, fill="#7565d8" if r == c else "#e5e9f2")
            draw.text((x + 87, y + 74), str(matrix[r][c]), fill="white" if r == c else "#25354f", font=title)
    footer = "Synthetic scene consistency, grouped held-out test" if result.get("controlled_benchmark", True) else "Imported labels, author-provided test split"
    draw.text((38, 622), footer, fill="#52617a", font=small)
    image.save(output / "confusion_matrix.png")
