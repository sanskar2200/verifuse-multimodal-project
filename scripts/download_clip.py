"""Download only the pinned official CLIP files required for local inference."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
# SHA-256 of the pinned upstream weight, recorded by the verified local run.
WEIGHT_SHA256 = "a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f"
FILES = ["config.json", "merges.txt", "preprocessor_config.json", "pytorch_model.bin",
         "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "README.md"]


def fetch(name):
    path = ROOT / "models/real/clip" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/openai/clip-vit-base-patch32/resolve/{REVISION}/{name}"
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".part")
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as out:
                    count = 0
                    while block := response.read(4 * 1024 * 1024):
                        out.write(block)
                        count += len(block)
                        if count % (64 * 1024 * 1024) == 0:
                            print(f"{name}: {count // 1024 // 1024} MiB", flush=True)
                temporary.replace(path)
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 * (attempt + 1))
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if name == "pytorch_model.bin" and digest != WEIGHT_SHA256:
        raise ValueError("CLIP weight checksum differs from the pinned reference; download is not accepted.")
    print(f"Ready: {name} ({path.stat().st_size:,} bytes)", flush=True)
    return {"file": name, "url": url, "bytes": path.stat().st_size, "sha256": digest}


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=3) as pool:
        receipts = list(pool.map(fetch, FILES))
    target = ROOT / "models/real/clip/download_receipt.json"
    target.write_text(json.dumps({"model_id": "openai/clip-vit-base-patch32", "revision": REVISION,
                                  "files": receipts}, indent=2) + "\n", encoding="utf-8")
    print("Pinned CLIP encoder download complete.", flush=True)
