"""Small local HTTP API. Uploads are processed in memory and never saved."""
from __future__ import annotations

import argparse
import base64
import binascii
import io
import json
import logging
import mimetypes
import os
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
MAX_BODY = 12 * 1024 * 1024
MAX_IMAGE = 8 * 1024 * 1024
MAX_PIXELS = 16_000_000
MODELS = {"feature_fusion", "late_fusion", "text_only", "image_only"}
Image.MAX_IMAGE_PIXELS = MAX_PIXELS
LOG = logging.getLogger("verifuse")


class VerificationServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, root=ROOT, engine=None, profile="synthetic"):
        self.root = Path(root).resolve()
        if profile not in {"auto", "real", "synthetic"}:
            raise ValueError("Choose an auto, real, or synthetic model profile.")
        self.profile = ("real" if (self.root / "models" / "real" / "bundle.json").is_file() else "synthetic") if profile == "auto" else profile
        self.engine = engine
        self.engine_lock = threading.Lock()
        super().__init__(address, Handler)

    def get_engine(self):
        if self.engine is None:
            if self.profile == "real":
                from verifuse.real_model import RealVerificationEngine
                self.engine = RealVerificationEngine(self.root)
            else:
                from verifuse.engine import VerificationEngine
                self.engine = VerificationEngine(self.root)
        loader = getattr(self.engine, "_load", None)
        if loader is not None:
            loader()
        return self.engine

    def artifact_path(self, directory, name):
        base = self.root / directory
        if self.profile == "real":
            base = base / "real"
        return base / name

    @property
    def scope(self):
        if self.profile == "real":
            return "Fakeddit real-media post classification with weak dataset labels; not forensic authenticity verification"
        return "controlled text-image consistency benchmark"

    def records(self):
        path = self.artifact_path("data", "manifest.json")
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        return data.get("records", data.get("samples", []))


class Handler(BaseHTTPRequestHandler):
    server: VerificationServer
    protocol_version = "HTTP/1.0"

    def setup(self):
        super().setup()
        self.connection.settimeout(20)

    def log_message(self, fmt, *args):
        # Do not put uploaded captions or images in logs.
        LOG.info("%s %s", self.command, urlsplit(self.path).path)

    def respond(self, status, body, content_type="application/json; charset=utf-8", filename=None):
        if not isinstance(body, bytes):
            body = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def error(self, status, message):
        self.respond(status, {"error": message})

    def trusted_request(self):
        # Keep a localhost research tool isolated from browser requests from other origins.
        host = self.headers.get("Host", "")
        try:
            hostname = urlsplit("http://" + host).hostname
        except ValueError:
            hostname = None
        allowed = {"localhost", "127.0.0.1", "::1"}
        if hostname not in allowed:
            self.error(403, "This application accepts localhost requests only.")
            return False
        origin = self.headers.get("Origin")
        if origin and origin != "http://" + host:
            self.error(403, "Use the application from its local browser address.")
            return False
        return True

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        if not self.trusted_request():
            return
        route = unquote(urlsplit(self.path).path)
        try:
            if route == "/api/health":
                try:
                    with self.server.engine_lock:
                        self.server.get_engine()
                    ready = True
                except Exception as exc:
                    LOG.exception("Model readiness check failed: %s", type(exc).__name__)
                    ready = False
                self.respond(200, {"status": "ok", "model_ready": ready, "project": "VeriFuse", "profile": self.server.profile, "scope": self.server.scope})
            elif route == "/api/samples":
                records = self.server.records()
                # Prefer evaluation samples and include multiple manipulation types.
                candidates = [r for r in records if r.get("split") in {"test", "ood_test"}] or records
                selected, buckets = [], {}
                for record in candidates:
                    key = (record.get("split"), record.get("manipulation", record.get("source_label")), record.get("label"))
                    buckets.setdefault(key, []).append(record)
                for index in range(3):
                    for bucket in buckets.values():
                        if index < len(bucket):
                            selected.append(bucket[index])
                samples = [{"id": r["id"], "caption": r.get("text", r.get("caption", "")), "image_url": "/" + str(r["image_path"]).replace("\\", "/"), "label": r["label"], "manipulation": r.get("manipulation", r.get("source_label", "unknown")), "split": r.get("split", "unknown")} for r in selected[:24]]
                response = {"samples": samples}
                if self.server.profile == "real":
                    response["profile"] = "real"
                self.respond(200, response)
            elif route == "/api/experiments":
                path = self.server.artifact_path("results", "experiments.json")
                if not path.exists():
                    self.error(503, f"The {self.server.profile} evaluation results are missing. Run the matching training command in the README.")
                else:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    data["profile"] = self.server.profile
                    self.respond(200, data)
            elif route.startswith("/data/real/images/"):
                self.serve_file(self.server.root / "data" / "real" / "images", route[len("/data/real/images/"):])
            elif route.startswith("/data/images/"):
                self.serve_file(self.server.root / "data" / "images", route[len("/data/images/"):])
            elif route.startswith("/downloads/"):
                self.serve_file(self.server.root / "output", route[len("/downloads/"):], attachment=True)
            elif route.startswith("/api/"):
                self.error(404, "API route not found.")
            else:
                relative = "index.html" if route == "/" else route.removeprefix("/static/").lstrip("/")
                self.serve_file(self.server.root / "static", relative)
        except (OSError, ValueError, KeyError) as exc:
            LOG.error("Could not serve resource: %s", type(exc).__name__)
            self.error(500, "A project artifact could not be loaded. Re-run the experiment pipeline.")

    def serve_file(self, base, relative, attachment=False):
        base = base.resolve()
        path = (base / relative).resolve()
        if not path.is_relative_to(base) or not path.is_file():
            self.error(404, "File not found.")
            return
        if attachment and (path.suffix.lower() not in {".pdf", ".docx", ".pptx", ".zip"} or not path.name.startswith("VeriFuse_")):
            self.error(404, "File not found.")
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if path.suffix in {".html", ".css", ".js"}:
            content_type += "; charset=utf-8"
        self.respond(200, path.read_bytes(), content_type, path.name if attachment else None)

    def do_POST(self):
        if not self.trusted_request():
            return
        if urlsplit(self.path).path != "/api/predict":
            self.error(404, "API route not found.")
            return
        if self.headers.get_content_type() != "application/json":
            self.error(415, "Send application/json.")
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.error(400, "Invalid request length.")
            return
        if not 0 < size <= MAX_BODY:
            self.error(413, "The request is empty or exceeds 12 MB.")
            return
        try:
            payload = json.loads(self.rfile.read(size))
            if not isinstance(payload, dict):
                raise ValueError("Expected a JSON object.")
            text = payload.get("text", "")
            if not isinstance(text, str) or len(text) > 3000:
                raise ValueError("Caption must be text, up to 3,000 characters.")
            model = payload.get("model", "feature_fusion")
            if not isinstance(model, str) or model not in MODELS:
                raise ValueError("Choose a supported model.")
            image = self.read_image(payload)
            if not text.strip() and image is None:
                raise ValueError("Add a caption or an image before analyzing.")
            with self.server.engine_lock:
                result = self.server.get_engine().predict(image=image, text=text, model=model)
            self.respond(200, result)
        except (json.JSONDecodeError, UnicodeDecodeError):
            self.error(400, "The request is not valid JSON.")
        except (ValueError, binascii.Error, UnidentifiedImageError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            self.error(400, str(exc) or "Use a valid PNG, JPEG, or WebP image.")
        except FileNotFoundError:
            self.error(503, f"The {self.server.profile} model artifacts are missing. Run the matching training command in the README.")
        except Exception as exc:
            LOG.exception("Prediction failed: %s", type(exc).__name__)
            self.error(500, "Prediction could not be completed. Check the local terminal for setup errors.")

    def read_image(self, payload):
        sample_id = payload.get("sample_id")
        encoded = payload.get("image_base64")
        if sample_id is not None and encoded:
            raise ValueError("Choose a sample or upload an image, not both.")
        if sample_id is not None:
            if not isinstance(sample_id, str):
                raise ValueError("Sample identifier must be text.")
            record = next((r for r in self.server.records() if r["id"] == sample_id), None)
            if record is None:
                raise ValueError("Sample not found. Choose a sample from the gallery.")
            path = (self.server.root / record["image_path"]).resolve()
            if not path.is_relative_to((self.server.root / "data").resolve()):
                raise ValueError("Invalid sample image path.")
            raw = path.read_bytes()
        elif encoded:
            if not isinstance(encoded, str):
                raise ValueError("Image must be a base64-encoded string.")
            if encoded.startswith("data:"):
                header, encoded = encoded.split(",", 1)
                if header not in {"data:image/png;base64", "data:image/jpeg;base64", "data:image/webp;base64"}:
                    raise ValueError("Use a PNG, JPEG, or WebP image.")
            raw = base64.b64decode(encoded, validate=True)
        else:
            return None
        if len(raw) > MAX_IMAGE:
            raise ValueError("Image must be 8 MB or smaller.")
        with Image.open(io.BytesIO(raw)) as image:
            if image.format not in {"PNG", "JPEG", "WEBP"}:
                raise ValueError("Use a PNG, JPEG, or WebP image.")
            if image.width * image.height > MAX_PIXELS:
                raise ValueError("Image must contain at most 16 million pixels.")
            image.load()
            return ImageOps.exif_transpose(image).convert("RGB")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--browser", choices=("none", "brave"), default="none", help="Optionally open the demo in Brave.")
    parser.add_argument("--profile", choices=("auto", "real", "synthetic"), default="auto", help="Auto uses the real-media model when its trained bundle exists.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        server = VerificationServer(("127.0.0.1", args.port), profile=args.profile)
    except OSError as exc:
        parser.exit(1, f"Cannot start on port {args.port}: {exc}\nTry --port 8766.\n")
    print(f"VeriFuse is running at http://127.0.0.1:{args.port}", flush=True)
    print(f"Profile: {server.profile}. {server.scope}. Press Ctrl+C to stop.", flush=True)
    if args.browser == "brave":
        candidates = [
            Path(os.environ.get("PROGRAMFILES", "C:/Program Files")) / "BraveSoftware/Brave-Browser/Application/brave.exe",
            Path(os.environ.get("PROGRAMFILES(X86)", "C:/Program Files (x86)")) / "BraveSoftware/Brave-Browser/Application/brave.exe",
            Path(os.environ.get("LOCALAPPDATA", ".")) / "BraveSoftware/Brave-Browser/Application/brave.exe",
            Path("/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"),
        ]
        executable = next((str(path) for path in candidates if path.is_file()), None)
        executable = executable or shutil.which("brave-browser") or shutil.which("brave")
        if executable:
            try:
                subprocess.Popen([executable, f"http://127.0.0.1:{args.port}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except OSError:
                print("Brave could not be opened. Open the local URL in Brave manually.", flush=True)
        else:
            print("Brave was not found. Open the local URL in your browser manually.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
