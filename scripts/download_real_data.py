"""Download a pinned public Fakeddit mirror and build an audited local manifest.

The mirror partitions are retained, not advertised as the original paper's split.
Copyright/license terms for the mirror are unspecified; downloaded media and text
must stay out of the distributable project ZIP. No model-generated labels or text.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import struct
import time
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen
from zipfile import ZipFile

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "real"
REPO = "ams-99/fakeddit_9k"
REVISION = "2ba956a67d2bab34de97b25d71af520ab3fe1e6e"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}"
FILES = {
    "train_balanced.csv": {"size": 1288311, "sha256": "a423d40de37602581e7991efeeab1e329b1db10af5460afdb270c44edd7bfef4"},
    "val_balanced.csv": {"size": 645194, "sha256": "68bbc58e3713775799f0fea925d1bcf530cbbe964e4c6f56a6fc1b47485a06dd"},
    "test_balanced.csv": {"size": 386500, "sha256": "3d359bfb69af1f6d0e6c3aedc60c886b9878fd9e6dcaa8f59fe512aaa811013f"},
    "image_cache.zip": {
        "size": 496335250,
        "sha256": "5ca5324c8424d6e6b8b461beafc6846404e36a3ce9eebd6f9bf509eb7f59e176",
    },
}
LABELS = {
    0: "true", 1: "satire", 2: "false connection", 3: "imposter content",
    4: "manipulated content", 5: "misleading content",
}
LABEL_SOURCE = "https://github.com/entitize/Fakeddit/issues/14#issuecomment-815355653"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str) -> dict:
    target = DATA / "raw" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    spec = FILES[name]
    url = f"{BASE}/{name}?download=true"
    good = target.exists() and target.stat().st_size == spec["size"]
    if good and spec.get("sha256"):
        good = sha256(target) == spec["sha256"]
    if not good:
        partial = target.with_name(target.name + ".part")
        for attempt in range(1, 5):
            try:
                offset = partial.stat().st_size if partial.exists() else 0
                headers = {"User-Agent": "VeriFuse-academic-project/1.0"}
                if offset:
                    headers["Range"] = f"bytes={offset}-"
                with urlopen(Request(url, headers=headers), timeout=60) as response:
                    if response.status != 206:
                        offset = 0
                    with partial.open("ab" if offset else "wb") as file:
                        total, logged = offset, time.monotonic()
                        while True:
                            chunk = response.read(1024 * 1024)
                            if not chunk:
                                break
                            file.write(chunk)
                            total += len(chunk)
                            if total > spec["size"]:
                                raise ValueError("Download exceeded pinned file size")
                            if time.monotonic() - logged > 8:
                                print(f"{name}: {total / 1e6:.1f}/{spec['size'] / 1e6:.1f} MB", flush=True)
                                logged = time.monotonic()
                if partial.stat().st_size != spec["size"]:
                    raise ValueError(f"Incomplete download for {name}")
                digest = sha256(partial)
                if spec.get("sha256") and digest != spec["sha256"]:
                    raise ValueError(f"SHA256 mismatch for {name}")
                partial.replace(target)
                break
            except Exception as error:
                if attempt == 4:
                    raise
                print(f"Retry {attempt}/4 for {name}: {type(error).__name__}", flush=True)
                time.sleep(min(2 * attempt, 8))
    digest = sha256(target)
    print(f"Verified {name}: {target.stat().st_size:,} bytes", flush=True)
    return {"name": name, "url": url, "size": target.stat().st_size, "sha256": digest}


def extract_images() -> None:
    destination = DATA / "images"
    destination.mkdir(parents=True, exist_ok=True)
    with ZipFile(DATA / "raw" / "image_cache.zip") as archive:
        files = [entry for entry in archive.infolist() if not entry.is_dir()]
        if sum(entry.file_size for entry in files) > 2_000_000_000:
            raise ValueError("Uncompressed image archive exceeds expected limit")
        names = set()
        for entry in files:
            path = PurePosixPath(entry.filename.replace("\\", "/"))
            if path.is_absolute() or ".." in path.parts or ":" in entry.filename:
                raise ValueError(f"Unsafe archive path: {entry.filename}")
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            if path.name in names:
                raise ValueError(f"Ambiguous image basename: {path.name}")
            names.add(path.name)
            if entry.file_size > 20_000_000 or entry.external_attr >> 16 & 0o170000 == 0o120000:
                raise ValueError(f"Unexpected image entry: {entry.filename}")
            target = (destination / path.name).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError("Image destination escapes data directory")
            if target.exists() and target.stat().st_size == entry.file_size:
                continue
            with archive.open(entry) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
    print(f"Images available: {len(names):,}", flush=True)


def canonical_url(value: str) -> str:
    parsed = urlsplit(value.strip())
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path, "", ""))


def build_manifest(receipts: list[dict]) -> dict:
    records, dropped = [], []
    image_cache = {}
    original_counts = {}
    for split in ("train", "val", "test"):
        with (DATA / "raw" / f"{split}_balanced.csv").open(encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        original_counts[split] = len(rows)
        for index, row in enumerate(rows):
            item_id = f"fakeddit9k_{split}_{index:05d}"
            text = row["text"].strip()
            numeric = int(row["6_way_label"])
            if numeric not in LABELS:
                raise ValueError(f"Unknown source label {numeric}")
            path = DATA / "images" / PurePosixPath(row["image_path"].replace("\\", "/")).name
            try:
                if not text or not path.is_file():
                    raise ValueError("missing image or text")
                if path.name not in image_cache:
                    with Image.open(path) as image:
                        image = ImageOps.exif_transpose(image).convert("RGB")
                        image.load()
                        width, height = image.size
                        digest = hashlib.sha256(struct.pack(">II", width, height) + image.tobytes()).hexdigest()
                    image_cache[path.name] = (sha256(path), digest, width, height)
                file_digest, pixel_digest, width, height = image_cache[path.name]
            except Exception as error:
                dropped.append({"id": item_id, "original_split": split, "reason": f"invalid_media:{type(error).__name__}"})
                continue
            normalized = re.sub(r"\s+", " ", text.casefold()).strip()
            url = canonical_url(row["original_url"])
            records.append({
                "id": item_id, "split": split, "original_split": split,
                "label": "REAL" if numeric == 0 else "FAKE",
                "source_label": LABELS[numeric], "source_label_id": numeric,
                "text": text, "image_path": path.relative_to(ROOT).as_posix(),
                "source_image_id": url or file_digest,
                "source_text_id": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
                "image_sha256": file_digest, "pixel_sha256": pixel_digest,
                "original_url": row["original_url"], "width": width, "height": height,
                "mirror_csv_row": index + 2,
            })
        print(f"Read {split}: {len(rows):,} original records", flush=True)

    # Connected components stop transitive image/text overlaps crossing splits.
    parents = list(range(len(records)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parents[max(a, b)] = min(a, b)
    first = {}
    keys = ("source_image_id", "source_text_id", "image_sha256", "pixel_sha256")
    for i, record in enumerate(records):
        for field in keys:
            key = (field, record[field])
            if key in first:
                union(i, first[key])
            else:
                first[key] = i
    groups = defaultdict(list)
    for i in range(len(records)):
        groups[find(i)].append(i)
    priority = {"train": 0, "val": 1, "test": 2}
    kept = []
    for members in groups.values():
        owner = min((records[i]["split"] for i in members), key=priority.get)
        group_id = "fakeddit_group_" + min(records[i]["id"] for i in members)
        seen_pairs = set()
        for i in members:
            record = records[i]
            record["group_id"] = group_id
            pair = (record["pixel_sha256"], record["source_text_id"], record["label"])
            reason = None
            if record["split"] != owner:
                reason = f"cross_split_component_owned_by_{owner}"
            elif pair in seen_pairs:
                reason = "duplicate_pair_within_split"
            if reason:
                dropped.append({"id": record["id"], "original_split": record["split"], "reason": reason, "group_id": group_id})
            else:
                kept.append(record)
                seen_pairs.add(pair)
    kept.sort(key=lambda r: (priority[r["split"]], r["id"]))
    summary = {}
    for split in priority:
        subset = [r for r in kept if r["split"] == split]
        summary[split] = {"count": len(subset), "labels": dict(Counter(r["label"] for r in subset)), "source_labels": dict(Counter(r["source_label"] for r in subset))}
    manifest = {
        "schema_version": 1,
        "name": "Fakeddit real Reddit image-text subset (community mirror)",
        "dataset": {"name": "Fakeddit real Reddit image-text subset (community mirror)", "id": REPO},
        "label_definition": "Source six-way label 0 (true) maps to REAL; labels 1-5 (satire, false connection, imposter, manipulated, misleading content) map to FAKE. Labels are inherited distant supervision, not new fact-checks or image-authenticity annotations.",
        "provenance": {"dataset_url": f"https://huggingface.co/datasets/{REPO}", "revision": REVISION, "original_dataset_url": "https://github.com/entitize/Fakeddit", "label_source_url": LABEL_SOURCE, "split_origin": "Community mirror; original paper split provenance unverified"},
        "dataset_id": REPO, "revision": REVISION, "source_url": f"https://huggingface.co/datasets/{REPO}",
        "original_dataset_url": "https://github.com/entitize/Fakeddit",
        "paper_url": "https://aclanthology.org/2020.lrec-1.755/",
        "label_source_url": LABEL_SOURCE, "source_label_mapping": LABELS,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Real Reddit images and original post text with supplied distant-supervision misinformation labels; not a general image-authenticity/deepfake detector.",
        "split_policy": "Preserve mirror train/val/test partitions; connected components sharing canonical image URL, exact file hash, decoded RGB hash, or normalized text are assigned priority train > val > test. Remove lower-priority component rows and exact within-split duplicate pairs. Original paper split provenance is unverified.",
        "license_status": "Neither the mirror nor the original repository specifies a dataset license. Downloaded text/media are for this local academic experiment; do not include them in the redistributable project ZIP. Original creators/rightsholders retain their rights.",
        "limitations": [
            "Community mirror lacks a dataset card, original Reddit post IDs and documented sampling/split provenance.",
            "Fakeddit labels are distant supervision from subreddit categories, not independent fact-checks for every post.",
            "FAKE includes satire, false connections, imposter, manipulated and misleading content; REAL means source label true, not a new truth verification.",
            "Image/text exact duplicates are audited, but near-duplicate events and pretraining overlap are not ruled out.",
        ],
        "original_counts": original_counts, "splits": summary,
        "removed_records": len(dropped), "download_receipts": receipts, "records": kept,
    }
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (DATA / "exclusions.json").write_text(json.dumps(dropped, indent=2), encoding="utf-8")
    (DATA / "download_receipts.json").write_text(json.dumps({"dataset_id": REPO, "revision": REVISION, "files": receipts}, indent=2), encoding="utf-8")
    print(json.dumps({"splits": summary, "excluded": len(dropped)}, indent=2), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="Use existing pinned local files without network")
    args = parser.parse_args()
    if args.prepare_only:
        receipts = []
        for name, spec in FILES.items():
            path = DATA / "raw" / name
            if not path.is_file() or path.stat().st_size != spec["size"]:
                raise ValueError(f"Missing or wrong-size pinned source: {name}")
            digest = sha256(path)
            if spec.get("sha256") and digest != spec["sha256"]:
                raise ValueError(f"Hash mismatch: {name}")
            receipts.append({"name": name, "url": f"{BASE}/{name}?download=true", "size": spec["size"], "sha256": digest})
    else:
        receipts = [fetch(name) for name in FILES]
    if not args.download_only:
        extract_images()
        build_manifest(receipts)


if __name__ == "__main__":
    main()
