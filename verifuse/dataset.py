"""Procedurally rendered, grouped multimodal consistency benchmark."""
from __future__ import annotations
import csv
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from .features import COLORS, SHAPES

SEED = 7031
NUMBER_WORDS = {1: "one", 2: "two", 3: "three"}
SPLIT_GROUPS = {"train": 216, "val": 72, "test": 72, "ood_test": 72}


def draw_scene(scene: dict, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    size, upscale = 256, 3
    background = tuple(int(v) for v in np.array([246, 245, 240]) + rng.integers(-3, 4))
    image = Image.new("RGB", (size * upscale, size * upscale), background)
    draw = ImageDraw.Draw(image)
    n = scene["count"]
    centers = {1: [(128, 126)], 2: [(77, 129), (180, 127)], 3: [(68, 90), (187, 93), (127, 190)]}[n]
    radius = int(rng.integers(25, 33))
    rgb = tuple(int(v) for v in np.clip(np.array(COLORS[scene["color"]]) + rng.integers(-10, 11, 3), 0, 255))
    for x, y in centers:
        x, y = int(x + rng.integers(-6, 7)), int(y + rng.integers(-6, 7))
        box = [(x - radius) * upscale, (y - radius) * upscale, (x + radius) * upscale, (y + radius) * upscale]
        if scene["shape"] == "circle":
            draw.ellipse(box, fill=rgb)
        elif scene["shape"] == "square":
            draw.rectangle(box, fill=rgb)
        else:
            draw.polygon([(x * upscale, (y - radius) * upscale), ((x + radius) * upscale, (y + radius) * upscale), ((x - radius) * upscale, (y + radius) * upscale)], fill=rgb)
    return image.resize((size, size), Image.Resampling.LANCZOS)


def caption(scene: dict, template: int) -> str:
    phrase = f"{NUMBER_WORDS[scene['count']]} {scene['color']} {scene['shape']}{'s' if scene['count'] != 1 else ''}"
    return [f"This image shows {phrase}.", f"The scene contains {phrase} on a light background.", f"A picture of {phrase}.", f"Visible in the image: {phrase}."][template % 4]


def generate(root: Path, seed: int = SEED) -> dict:
    root = Path(root)
    folder = root / "data" / "images"
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    combinations = list(itertools.product(COLORS, SHAPES, (1, 2, 3)))
    records = []
    serial = 0
    for split, groups in SPLIT_GROUPS.items():
        settings = combinations * (groups // len(combinations))
        rng.shuffle(settings)
        for index, (color, shape, count) in enumerate(settings):
            group = f"scene-{serial:04d}"
            scene = {"color": color, "shape": shape, "count": count}
            kind = "count" if split == "ood_test" else ("color" if index % 2 == 0 else "shape")
            changed = dict(scene)
            choices = list(COLORS) if kind == "color" else list(SHAPES) if kind == "shape" else [1, 2, 3]
            offset = int(rng.integers(1, len(choices)))
            changed[kind] = choices[(choices.index(scene[kind]) + offset) % len(choices)]
            altered_modality = "text" if (index // 2) % 2 == 0 else "image"
            scene_seed = seed + serial * 31
            image_path = f"data/images/{group}.png"
            draw_scene(scene, scene_seed).save(root / image_path)
            template = int(rng.integers(0, 4))
            base = {"group_id": group, "split": split, "source": "procedural-original", "scene_seed": scene_seed}
            records.append({**base, "id": group + "-real", "label": "REAL", "text": caption(scene, template), "image_path": image_path, "manipulation": "none", "altered_modality": "none", "scene": scene, "caption": scene})
            if altered_modality == "text":
                fake_path, fake_scene, fake_caption = image_path, scene, changed
            else:
                fake_path, fake_scene, fake_caption = f"data/images/{group}-altered.png", changed, scene
                draw_scene(changed, scene_seed).save(root / fake_path)
            records.append({**base, "id": group + "-fake", "label": "FAKE", "text": caption(fake_caption, template), "image_path": fake_path, "manipulation": kind, "altered_modality": altered_modality, "scene": fake_scene, "caption": fake_caption})
            serial += 1
    manifest = {"version": 1, "seed": seed, "name": "VeriFuse Controlled Scene Consistency", "label_definition": "REAL is an unaltered procedural scene paired with its matching caption; FAKE is its paired version with one caption or image attribute altered to contradict the other modality. These labels do not assert real-world news truth.", "records": records}
    (root / "data" / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with (root / "data" / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "group_id", "split", "label", "text", "image_path", "manipulation", "altered_modality"])
        writer.writeheader()
        writer.writerows({k: row[k] for k in writer.fieldnames} for row in records)
    return manifest


def load_manifest(path: Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_records(data["records"])
    return data


def validate_records(records: list[dict]):
    if not records:
        raise ValueError("The dataset has no records.")
    groups, ids = {}, set()
    for row in records:
        if row["label"] not in ("REAL", "FAKE"):
            raise ValueError("Labels must be REAL or FAKE.")
        if row["split"] not in SPLIT_GROUPS:
            raise ValueError("split must be train, val, test, or ood_test")
        if row["id"] in ids:
            raise ValueError("Duplicate record id: " + row["id"])
        ids.add(row["id"])
        if row["group_id"] in groups and groups[row["group_id"]] != row["split"]:
            raise ValueError("Group leakage across splits: " + row["group_id"])
        groups[row["group_id"]] = row["split"]


def import_csv(path: Path) -> dict:
    """Import documented external CSV, assigning groups before fitting features."""
    path = Path(path).resolve()
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    records = []
    for i, row in enumerate(rows):
        if not all(row.get(k) for k in ("image_path", "text", "label", "group_id")):
            raise ValueError("CSV requires nonempty image_path,text,label,group_id columns.")
        bucket = int(hashlib.sha256(row["group_id"].encode()).hexdigest()[:8], 16) % 10
        split = row.get("split") or ("train" if bucket < 7 else "val" if bucket == 7 else "test")
        image_path = (path.parent / row["image_path"]).resolve()
        if not image_path.is_file():
            raise ValueError(f"Missing image on CSV row {i + 2}: {image_path}")
        with Image.open(image_path) as image:
            image.verify()
        records.append({"id": row.get("id") or f"external-{i:06d}", "group_id": row["group_id"], "split": split, "label": row["label"].strip().upper(), "text": row["text"], "image_path": str(image_path), "manipulation": row.get("manipulation") or "unspecified", "altered_modality": row.get("altered_modality") or "unspecified", "source": "external-csv"})
    validate_records(records)
    return {"version": 1, "seed": SEED, "name": path.stem, "label_definition": "Labels supplied by CSV author; provenance and real-world validity have not been verified.", "records": records}
