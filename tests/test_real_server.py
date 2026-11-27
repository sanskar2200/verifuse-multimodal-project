"""Profile isolation and real-media HTTP routing, without downloading model weights."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PIL import Image
from verifuse.server import VerificationServer


class CorpusEngine:
    def predict(self, image, text, model):
        return {"profile": "real", "prediction": "UNCERTAIN", "image_size": list(image.size) if image else None,
                "text": text, "model": model}


class RealProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        for directory in ["data/real/images", "data/images", "models/real", "results/real"]:
            (cls.root / directory).mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (48, 32), "green").save(cls.root / "data/real/images/post.jpg")
        Image.new("RGB", (8, 8), "red").save(cls.root / "data/images/shape.png")
        cls.record = {"id": "real-post", "text": "A real source caption", "image_path": "data/real/images/post.jpg",
                      "label": "REAL", "split": "test", "manipulation": "dataset_label"}
        (cls.root / "data/real/manifest.json").write_text(json.dumps({"records": [cls.record]}))
        (cls.root / "data/manifest.json").write_text(json.dumps({"records": [{"id": "synthetic-post", "text": "red square",
            "image_path": "data/images/shape.png", "label": "FAKE", "split": "test"}]}))
        (cls.root / "models/real/bundle.json").write_text("{}")
        (cls.root / "results/real/experiments.json").write_text(json.dumps({"dataset": {"name": "real-corpus"}, "experiments": []}))
        (cls.root / "results/experiments.json").write_text(json.dumps({"dataset": {"name": "synthetic-corpus"}, "experiments": []}))
        (cls.root / "data/real/private.txt").write_text("private")
        cls.server = VerificationServer(("127.0.0.1", 0), cls.root, CorpusEngine(), profile="auto")
        cls.worker = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.worker.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.worker.join()
        cls.temp.cleanup()

    def request(self, route, payload=None):
        body = None if payload is None else json.dumps(payload).encode()
        request = Request(self.base + route, data=body, headers={"Content-Type": "application/json"})
        try:
            response = urlopen(request, timeout=5)
        except HTTPError as error:
            response = error
        with response:
            return response.status, response.read()

    def test_real_profile_exposes_only_matching_samples_and_results(self):
        health = json.loads(self.request("/api/health")[1])
        self.assertEqual(health["profile"], "real")
        self.assertIn("weak dataset labels", health["scope"])
        samples = json.loads(self.request("/api/samples")[1])
        self.assertEqual(samples["profile"], "real")
        self.assertEqual([sample["id"] for sample in samples["samples"]], ["real-post"])
        result = json.loads(self.request("/api/experiments")[1])
        self.assertEqual(result["profile"], "real")
        self.assertEqual(result["dataset"]["name"], "real-corpus")

    def test_sample_prediction_uses_real_image_and_rejects_other_profile_id(self):
        status, body = self.request("/api/predict", {"sample_id": "real-post", "text": "caption"})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["image_size"], [48, 32])
        self.assertEqual(self.request("/api/predict", {"sample_id": "synthetic-post", "text": "caption"})[0], 400)

    def test_real_images_are_served_without_traversal(self):
        self.assertEqual(self.request("/data/real/images/post.jpg")[0], 200)
        for route in ["/data/real/images/../private.txt", "/data/real/images/%2e%2e/private.txt",
                      "/data/real/images/../../manifest.json"]:
            with self.subTest(route=route):
                self.assertEqual(self.request(route)[0], 404)

    def test_explicit_synthetic_profile_overrides_existing_real_model(self):
        with VerificationServer(("127.0.0.1", 0), self.root, CorpusEngine(), profile="synthetic") as server:
            self.assertEqual(server.profile, "synthetic")
            self.assertEqual(server.records()[0]["id"], "synthetic-post")

    def test_auto_falls_back_only_when_real_bundle_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            with VerificationServer(("127.0.0.1", 0), directory, CorpusEngine(), profile="auto") as server:
                self.assertEqual(server.profile, "synthetic")
            with VerificationServer(("127.0.0.1", 0), directory, CorpusEngine(), profile="real") as server:
                self.assertEqual(server.profile, "real")
                self.assertEqual(server.records(), [])


if __name__ == "__main__":
    unittest.main()
