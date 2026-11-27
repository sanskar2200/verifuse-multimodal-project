"""Behavioral checks for learning, splitting, metrics and domain boundaries."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from verifuse.dataset import draw_scene, import_csv, load_manifest, validate_records
from verifuse.engine import VerificationEngine, caption_supported
from verifuse.experiments import run
from verifuse.features import Tfidf, image_attributes, parse_caption
from verifuse.learning import LogisticModel, metrics

ROOT = Path(__file__).resolve().parents[1]
(ROOT / "tmp").mkdir(exist_ok=True)


class MetricsAndLearningTests(unittest.TestCase):
    def test_tied_scores_have_chance_auc_and_prevalence_ap(self):
        result = metrics([0, 1, 0, 1], [.5, .5, .5, .5])
        self.assertEqual(result["roc_auc"], .5)
        self.assertEqual(result["pr_auc"], .5)
        self.assertEqual(result["confusion_matrix"], [[0, 2], [0, 2]])

    def test_known_confusion_matrix_and_reversed_ranking(self):
        result = metrics([0, 0, 1, 1], [.1, .8, .7, .9])
        self.assertEqual(result["confusion_matrix"], [[1, 1], [0, 2]])
        self.assertEqual(result["accuracy"], .75)
        self.assertAlmostEqual(result["precision"], 2 / 3)
        self.assertEqual(metrics([0, 0, 1, 1], [.9, .8, .2, .1])["roc_auc"], 0)

    def test_train_only_vocabulary(self):
        encoder = Tfidf().fit(["red circle", "red circle", "blue square", "blue square"])
        self.assertNotIn("unseen", encoder.vocabulary)
        self.assertTrue(np.isfinite(encoder.transform(["unseen", ""])).all())
        self.assertEqual(float(encoder.transform(["unseen"]).sum()), 0)

    def test_logistic_learns_and_roundtrips(self):
        x = np.array([[-2.], [-1.], [1.], [2.]])
        model = LogisticModel(steps=250).fit(x, np.array([0., 0., 1., 1.]))
        p = model.predict_proba(x)
        self.assertTrue(np.all((p >= .5) == [0, 0, 1, 1]))
        np.testing.assert_allclose(p, LogisticModel.from_dict(model.to_dict()).predict_proba(x))


class DataAndDomainTests(unittest.TestCase):
    def test_pixel_extraction_reads_all_attribute_combinations(self):
        for color in ("red", "blue", "green", "yellow"):
            for shape in ("circle", "square", "triangle"):
                for count in (1, 2, 3):
                    scene = {"color": color, "shape": shape, "count": count}
                    extracted = image_attributes(draw_scene(scene, 51))
                    self.assertTrue(extracted["in_domain"])
                    self.assertEqual({k: extracted[k] for k in scene}, scene)

    def test_caption_ambiguity_and_negation_abstain(self):
        self.assertEqual(parse_caption("A picture of two red circles.")["count"], 2)
        for text in ("No red circles", "one red and blue circle", "one red circle proves the election was stolen", "one red circle and two squares"):
            self.assertFalse(caption_supported(text))

    def test_group_leakage_rejected(self):
        rows = [{"id": "a", "group_id": "same", "split": "train", "label": "REAL"}, {"id": "b", "group_id": "same", "split": "test", "label": "FAKE"}]
        with self.assertRaisesRegex(ValueError, "Group leakage"):
            validate_records(rows)

    def test_import_csv_resolves_images_and_rejects_invalid_labels(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as name:
            folder = Path(name)
            draw_scene({"color": "red", "shape": "circle", "count": 1}, 5).save(folder / "sample.png")
            path = folder / "input.csv"
            path.write_text("image_path,text,label,group_id\nsample.png,one red circle,REAL,g1\n", encoding="utf-8")
            row = import_csv(path)["records"][0]
            self.assertEqual(Path(row["image_path"]), folder / "sample.png")
            path.write_text("image_path,text,label,group_id\nsample.png,one red circle,UNKNOWN,g1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Labels"):
                import_csv(path)


class ImportedExperimentTests(unittest.TestCase):
    def run_imported(self, folder, include_ood):
        root = Path(folder)
        (root / "models").mkdir()
        demo = root / "models" / "bundle.json"
        demo.write_text("existing demo artifact", encoding="utf-8")
        records = []
        splits = ["train", "val", "test"] + (["ood_test"] if include_ood else [])
        for index, split in enumerate(splits):
            image_path = f"{split}.png"
            draw_scene({"color": "red", "shape": "circle", "count": 1}, 900 + index).save(root / image_path)
            for suffix, label, text in (("real", "REAL", "one red circle"), ("fake", "FAKE", "one blue circle")):
                records.append({"id": f"{split}-{suffix}", "group_id": split, "split": split, "label": label, "text": text, "image_path": image_path})
            if split == "ood_test":
                records.append({"id": "ood-extra", "group_id": split, "split": split, "label": "REAL", "text": "This image shows one red circle.", "image_path": image_path})
        csv_path = root / "external.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        manifest = import_csv(csv_path)
        output = root / "imported"
        result = run(root, manifest, output, controlled=False)
        bundle = json.loads((output / "bundle.json").read_text(encoding="utf-8"))
        report = (output / "REPORT.md").read_text(encoding="utf-8")
        self.assertEqual(demo.read_text(encoding="utf-8"), "existing demo artifact")
        self.assertFalse(result["controlled_benchmark"])
        self.assertEqual(result["dataset"]["label_definition"], manifest["label_definition"])
        self.assertIn("imported corpus", bundle["scope"])
        self.assertIsNone(bundle["demo_uncertainty_threshold"])
        self.assertTrue(any("corpus author" in item for item in result["dataset"]["limitations"]))
        self.assertFalse(any("Matching captions and procedural" in item for item in result["dataset"]["limitations"]))
        self.assertIn("imported-corpus experiment", report)
        self.assertNotIn("OOD count manipulation remains", report)
        self.assertEqual(result["difficult_cases"], [])
        self.assertEqual(result["faithfulness"], {})
        for filename in ("model_comparison.png", "confusion_matrix.png"):
            with Image.open(output / filename) as chart:
                chart.verify()
        return result["experiments"][-1], report

    def test_imported_run_without_ood_has_no_ood_claim(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as folder:
            ood, report = self.run_imported(folder, include_ood=False)
            self.assertEqual(ood["status"], "not_available")
            self.assertIsNone(ood["metrics"])
            self.assertEqual(ood["variants"], [])
            self.assertIn("no ood_test records", ood["description"])
            self.assertIn("Author-designated OOD split (not evaluated)", report)

    def test_imported_run_with_ood_uses_only_author_split_claim(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as folder:
            ood, report = self.run_imported(folder, include_ood=True)
            self.assertEqual(ood["status"], "evaluated")
            self.assertIsNotNone(ood["metrics"])
            self.assertEqual(sum(map(sum, ood["metrics"]["confusion_matrix"])), 3)
            self.assertEqual(len(ood["variants"]), 4)
            self.assertIn("corpus author's ood_test split", ood["description"])
            self.assertNotIn("Count alterations", ood["description"])
            self.assertNotIn("matching REAL controls", ood["description"])
            self.assertIn("| ood_test | 3 | 1 | 2 | 1 |", report)


@unittest.skipUnless((ROOT / "models" / "bundle.json").exists(), "Generate and train artifacts first.")
class TrainedArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = VerificationEngine(ROOT)
        cls.manifest = load_manifest(ROOT / "data" / "manifest.json")
        cls.result = json.loads((ROOT / "results" / "experiments.json").read_text())

    def test_split_balance_group_isolation_and_pixel_hashes(self):
        rows = self.manifest["records"]
        self.assertGreaterEqual(len(rows), 400)
        groups, pixels = {}, {}
        for row in rows:
            self.assertEqual(groups.setdefault(row["group_id"], row["split"]), row["split"])
            with Image.open(ROOT / row["image_path"]) as image:
                digest = hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
            self.assertEqual(pixels.setdefault(digest, row["split"]), row["split"])
        for split in ("train", "val", "test", "ood_test"):
            subset = [r for r in rows if r["split"] == split]
            self.assertEqual(sum(r["label"] == "REAL" for r in subset), sum(r["label"] == "FAKE" for r in subset))
        self.assertFalse(any(r["manipulation"] == "count" for r in rows if r["split"] != "ood_test"))

    def test_fusion_responds_to_paired_evidence_and_attributions(self):
        image = draw_scene({"color": "blue", "shape": "square", "count": 2}, 199)
        real = self.engine.predict(image, "This image shows two blue squares.")
        fake = self.engine.predict(image, "This image shows two red squares.")
        self.assertEqual(real["prediction"], "REAL")
        self.assertEqual(fake["prediction"], "FAKE")
        self.assertGreater(fake["fake_probability"] - real["fake_probability"], .5)
        self.assertTrue(any(e["title"] == "Color conflicts" and e["impact"] > 0 for e in fake["evidence"]))
        self.assertTrue(any(t["token"] == "red" and t["impact"] > 0 for t in fake["token_attributions"]))
        self.assertEqual(len(fake["image_regions"]), 2)
        json.dumps(fake, allow_nan=False)

    def test_unsupported_input_and_absent_artifact(self):
        image = draw_scene({"color": "blue", "shape": "square", "count": 2}, 199)
        for candidate, text in ((None, "two blue squares"), (Image.new("RGB", (200, 200)), "two blue squares"), (image, "not two blue squares"), (image, "A political announcement today")):
            self.assertEqual(self.engine.predict(candidate, text)["prediction"], "UNCERTAIN")
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as folder:
            with self.assertRaisesRegex(FileNotFoundError, "run_experiments"):
                VerificationEngine(Path(folder)).predict(image, "two blue squares")

    def test_six_experiments_ablation_and_case_analysis_are_present(self):
        self.assertEqual(len(self.result["experiments"]), 6)
        self.assertGreaterEqual(len(self.result["difficult_cases"]), 15)
        fusion = self.result["experiments"][2]["metrics"]["accuracy"]
        without_cross = self.result["experiments"][4]["variants"][2]["metrics"]["accuracy"]
        self.assertGreater(fusion, without_cross + .2)
        self.assertTrue(all(check["passed"] for check in self.result["sanity_checks"]))
        self.assertTrue(self.result["faithfulness"]["relevance_pass"])


if __name__ == "__main__":
    unittest.main()
