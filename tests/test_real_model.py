"""Checks for source isolation, real-model fitting, caching and inference contracts."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from verifuse.real_model import (FEATURE_ORDER, LinearHead, RealVerificationEngine, binary_metrics,
                                 encode_manifest, feature_matrices, fit_head, load_embeddings,
                                 normalize_embeddings, train, validate_manifest)

ROOT = Path(__file__).resolve().parents[1]
HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None


def fixture_records():
    return [{"id": f"{split}-{label}", "split": split, "label": label, "text": f"caption {split} {label}",
             "image_path": f"{split}-{label}.png", "source_image_id": f"{split}-{label}"}
            for split in ("train", "val", "test") for label in ("REAL", "FAKE")]


class RealRepresentationTests(unittest.TestCase):
    def test_normalization_rejects_zero_and_nonfinite_values(self):
        np.testing.assert_allclose(normalize_embeddings([[3, 4]]), [[.6, .8]])
        for value in ([[0, 0]], [[float("nan"), 1]], [1, 2]):
            with self.assertRaises(ValueError):
                normalize_embeddings(value)

    def test_fusion_contains_relational_features_in_saved_order(self):
        features = feature_matrices([[2, 0]], [[0, 4, 1]])
        np.testing.assert_allclose(features["feature_fusion"], [[1, 0, 0, 4, 1]])
        self.assertEqual(FEATURE_ORDER, ["clip_image_embedding", "tfidf_text_features"])
        with self.assertRaisesRegex(ValueError, "rows must align"):
            feature_matrices([[1, 0]], [[1, 0], [0, 1]])

    def test_shared_source_identifiers_and_pixels_crossing_splits_fail(self):
        for field in ("source_image_id", "source_text_id", "pixel_sha256", "group_id"):
            records = fixture_records()
            records[0][field] = records[-1][field] = "shared"
            with self.assertRaisesRegex(ValueError, "Source leakage"):
                validate_manifest({"records": records})
        records = fixture_records()
        records[0]["image_id"] = "aliased-id"
        records[-1]["source_image_id"] = "aliased-id"
        with self.assertRaisesRegex(ValueError, "Source leakage"):
            validate_manifest({"records": records})

    def test_identifier_absence_is_not_reported_as_audit_success(self):
        records = fixture_records()
        for row in records:
            row.pop("source_image_id")
        audit = validate_manifest({"records": records})
        self.assertFalse(audit["supplied_source_identifiers_disjoint"])
        self.assertFalse(audit["pixel_hash_disjoint"])
        self.assertEqual(audit["splits"]["test"]["records"], 2)
        records[-1]["label"] = "REAL"
        with self.assertRaisesRegex(ValueError, "both REAL and FAKE"):
            validate_manifest({"records": records})

    def test_external_embeddings_align_by_id_and_reject_missing_ids(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as folder:
            path = Path(folder) / "features.npz"
            np.savez(path, ids=np.array(["b", "a"]), image=[[0, 2], [3, 0]], text=[[1, 0], [0, 1]])
            image, text, meta = load_embeddings(path, [{"id": "a"}, {"id": "b"}])
            np.testing.assert_allclose(image, [[1, 0], [0, 1]])
            np.testing.assert_allclose(text, [[0, 1], [1, 0]])
            self.assertIn("embedding_file_sha256", meta)
            with self.assertRaisesRegex(ValueError, "Embeddings missing"):
                load_embeddings(path, [{"id": "unknown"}])

    def test_head_rejects_invalid_parameters_and_inputs(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            LinearHead({"weights": [1], "mean": [0], "scale": [0], "bias": 0})
        head = LinearHead({"weights": [1], "mean": [0], "scale": [1], "bias": 0})
        with self.assertRaisesRegex(ValueError, "Invalid features"):
            head.predict_proba([[np.inf]])


class RecordingEncoder:
    """Deterministic stand-in for verifying cache behavior, never dataset training."""
    def __init__(self, model_path):
        self.model_path = model_path
        self.batch_size = 2
        self.truncated_texts = 0
        self.images_encoded = self.texts_encoded = 0

    def encode_images(self, images):
        self.images_encoded += len(images)
        return normalize_embeddings([[float(np.asarray(im).mean()) + 1, 1] for im in images])

    def encode_texts(self, texts):
        self.texts_encoded += len(texts)
        return normalize_embeddings([[len(text) + 1, 1] for text in texts])


class FeatureCacheTests(unittest.TestCase):
    def test_resume_reuses_inputs_but_invalidates_text_pixels_and_encoder(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as folder:
            root = Path(folder)
            encoder_path = root / "clip"
            encoder_path.mkdir()
            (encoder_path / "config.json").write_text("{}", encoding="utf-8")
            Image.new("RGB", (8, 8), "red").save(root / "a.png")
            records = [{"id": "a", "image_path": "a.png", "text": "one caption", "label": "REAL", "split": "train"}]
            encoder = RecordingEncoder(encoder_path)
            output = root / "embeddings.npz"
            encode_manifest(root, {"records": records}, encoder, output)
            self.assertEqual((encoder.images_encoded, encoder.texts_encoded), (1, 1))
            records[0]["label"], records[0]["split"] = "FAKE", "test"
            encode_manifest(root, {"records": records}, encoder, output)
            self.assertEqual((encoder.images_encoded, encoder.texts_encoded), (1, 1))
            records[0]["text"] = "a changed caption"
            encode_manifest(root, {"records": records}, encoder, output)
            self.assertEqual((encoder.images_encoded, encoder.texts_encoded), (1, 2))
            Image.new("RGB", (8, 8), "blue").save(root / "a.png")
            encode_manifest(root, {"records": records}, encoder, output)
            self.assertEqual((encoder.images_encoded, encoder.texts_encoded), (2, 2))
            (encoder_path / "config.json").write_text('{"revision":2}', encoding="utf-8")
            encode_manifest(root, {"records": records}, encoder, output)
            self.assertEqual((encoder.images_encoded, encoder.texts_encoded), (3, 3))


@unittest.skipUnless(HAS_SKLEARN, "Real-data training dependencies are optional for the synthetic demo.")
class RealFittingTests(unittest.TestCase):
    def test_scaling_uses_training_only_and_roundtrips(self):
        x = np.array([[-2.0], [-1.0], [1.0], [2.0]])
        head = fit_head(x, [0, 0, 1, 1], [[-100.0], [100.0]], [0, 1], candidates=[.1, 1])
        self.assertEqual(head["mean"], [0.0])
        self.assertAlmostEqual(head["scale"][0], np.std(x))
        np.testing.assert_array_equal(LinearHead(head).predict_proba(x) >= .5, [False, False, True, True])
        self.assertEqual(head["selection_split"], "val")

    def test_metrics_have_correct_class_support_and_matrix(self):
        result = binary_metrics([0, 0, 0, 1], [.1, .2, .8, .9])
        self.assertEqual(result["confusion_matrix"], [[2, 1], [0, 1]])
        self.assertEqual(result["per_class"]["REAL"]["support"], 3)
        self.assertEqual(result["per_class"]["FAKE"]["support"], 1)
        self.assertAlmostEqual(result["balanced_accuracy"], 5 / 6)

    def test_test_labels_do_not_change_fitted_heads_or_selection(self):
        records = fixture_records()
        image = np.pad(np.array([[1, -.2], [1, .2], [1, -.3], [1, .3], [1, -.4], [1, .4]]), ((0, 0), (0, 510)))
        text = np.pad(np.array([[1, -.4], [1, .4], [1, -.5], [1, .5], [1, -.6], [1, .6]]), ((0, 0), (0, 510)))
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as folder:
            root = Path(folder)
            (root / "models").mkdir()
            original = root / "models/bundle.json"
            original.write_text("synthetic preserved", encoding="utf-8")
            first = train(root, {"dataset": "unit fixture", "records": records}, image, text, candidates=[.01, .1])
            first_bundle = json.loads((root / "models/real/bundle.json").read_text(encoding="utf-8"))
            for row in records:
                if row["split"] == "test":
                    row["label"] = "FAKE" if row["label"] == "REAL" else "REAL"
            second = train(root, {"name": "unit fixture", "records": records}, image, text, candidates=[.01, .1])
            second_bundle = json.loads((root / "models/real/bundle.json").read_text(encoding="utf-8"))
            self.assertEqual(first_bundle["models"], second_bundle["models"])
            self.assertEqual(first_bundle["late_fusion_alpha"], second_bundle["late_fusion_alpha"])
            self.assertEqual(first["selected_model"], second["selected_model"])
            self.assertNotEqual(first["experiments"][0]["metrics"]["accuracy"], second["experiments"][0]["metrics"]["accuracy"])
            self.assertEqual(original.read_text(), "synthetic preserved")
            self.assertFalse(first_bundle["fit"]["clip_fine_tuned"])
            self.assertEqual(first["experiments"][-1]["status"], "not_available")
            self.assertEqual(first["faithfulness"], {})
            self.assertEqual(first["benchmark"], "unit fixture")
            self.assertEqual(first["majority_baseline"]["test_accuracy"], .5)
            self.assertEqual(first["majority_baseline"]["class"], "FAKE")


class RealInferenceTests(unittest.TestCase):
    def test_missing_modality_abstains_and_explanations_are_not_invented(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as folder:
            root = Path(folder)
            (root / "models/real").mkdir(parents=True)
            def head(size):
                return {"weights": [0] * size, "mean": [0] * size, "scale": [1] * size, "bias": 1.0}
            bundle = {"profile": "real", "feature_order": FEATURE_ORDER, "dataset_name": "fixture",
                      "label_definition": "Inherited labels", "late_fusion_alpha": .5,
                      "text_vectorizer": {"vocabulary": {"caption": 0, "real": 1}, "idf": [1, 1],
                                          "ngram_range": [1, 2], "min_df": 2, "max_features": 2, "sublinear_tf": True},
                      "models": {"image_only": head(2), "text_only": head(2), "feature_fusion": head(4)}}
            (root / "models/real/bundle.json").write_text(json.dumps(bundle), encoding="utf-8")
            class Encoder:
                def encode_images(self, images):
                    return np.array([[1, 0]])
            engine = RealVerificationEngine(root, encoder=Encoder())
            missing = engine.predict(None, "caption")
            self.assertEqual(missing["prediction"], "UNCERTAIN")
            self.assertIsNone(missing["fake_probability"])
            output = engine.predict(Image.new("RGB", (8, 8)), "caption")
            self.assertEqual(output["prediction"], "FAKE")
            self.assertIsNone(output["in_domain"])
            self.assertEqual(output["token_attributions"], [])
            self.assertEqual(output["image_regions"], [])
            json.dumps(output, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
