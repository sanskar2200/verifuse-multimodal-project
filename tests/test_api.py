"""Exercise the local HTTP boundary without loading model artifacts."""
import base64
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PIL import Image
from verifuse.server import VerificationServer


class EchoEngine:
    def predict(self, image, text, model):
        return {"prediction": "UNCERTAIN", "text": text, "model": model,
                "image_size": list(image.size) if image else None}


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        (root / "static").mkdir()
        (root / "static" / "index.html").write_text("<h1>VeriFuse</h1>")
        (root / "private.txt").write_text("private")
        cls.server = VerificationServer(("127.0.0.1", 0), root, EchoEngine())
        cls.worker = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.worker.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.worker.join()
        cls.temp.cleanup()

    def request(self, path, payload=None, headers=None):
        data = None if payload is None else json.dumps(payload).encode()
        request = Request(self.base + path, data=data, headers=headers or {"Content-Type": "application/json"})
        try:
            response = urlopen(request, timeout=5)
        except HTTPError as error:
            response = error
        with response:
            return response.status, response.read(), response.headers

    def test_health_and_static(self):
        status, body, headers = self.request("/api/health")
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(body)["model_ready"])
        self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
        self.assertEqual(self.request("/")[0], 200)

    def test_text_only_input_is_accepted(self):
        status, body, _ = self.request("/api/predict", {"text": "one red circle"})
        self.assertEqual(status, 200)
        self.assertIsNone(json.loads(body)["image_size"])

    def test_image_decoding(self):
        buffer = io.BytesIO()
        Image.new("RGB", (32, 40), "red").save(buffer, "PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode()
        status, body, _ = self.request("/api/predict", {"text": "one red square", "image_base64": encoded})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["image_size"], [32, 40])

    def test_invalid_inputs(self):
        for payload in [{}, {"text": 7}, {"text": "x" * 3001}, {"text": "x", "model": "unknown"}, {"text": "x", "image_base64": "invalid!"}, {"text": "x", "sample_id": "../../private.txt"}, {"text": "x", "image_base64": "data:image/svg+xml;base64,AA=="}, []]:
            with self.subTest(payload=str(payload)[:60]):
                self.assertEqual(self.request("/api/predict", payload)[0], 400)

    def test_origin_and_host_protection(self):
        self.assertEqual(self.request("/api/predict", {"text": "x"}, {"Content-Type": "application/json", "Origin": "https://example.com"})[0], 403)
        self.assertEqual(self.request("/api/health", headers={"Host": "example.com"})[0], 403)

    def test_path_traversal(self):
        for path in ["/../private.txt", "/%2e%2e/private.txt", "/data/images/../../private.txt", "/downloads/../private.txt"]:
            self.assertEqual(self.request(path)[0], 404)

    def test_missing_artifacts_have_useful_status(self):
        self.assertEqual(self.request("/api/experiments")[0], 503)
        self.assertEqual(json.loads(self.request("/api/samples")[1]), {"samples": []})


if __name__ == "__main__":
    unittest.main()
