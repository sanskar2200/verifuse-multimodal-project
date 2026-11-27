"""Transparent image attributes, caption parsing, and train-only TF-IDF."""
from __future__ import annotations

import re
from collections import Counter
import numpy as np
from PIL import Image

COLORS = {"red": (210, 66, 74), "blue": (53, 119, 209), "green": (45, 159, 111), "yellow": (232, 182, 40)}
SHAPES = ("circle", "square", "triangle")
COUNTS = {"one": 1, "two": 2, "three": 3, "1": 1, "2": 2, "3": 3, "a": 1, "an": 1}
COLOR_ALIASES = {"crimson": "red", "scarlet": "red", "azure": "blue", "emerald": "green", "golden": "yellow"}
SHAPE_ALIASES = {"circular": "circle", "round": "circle", "squares": "square", "circles": "circle", "triangles": "triangle", "triangular": "triangle"}
IMAGE_NAMES = ["image_color_" + c for c in COLORS] + ["image_shape_" + s for s in SHAPES] + ["image_count_" + str(i) for i in (1, 2, 3)] + ["foreground_fraction", "mean_red", "mean_green", "mean_blue", "component_fill"]
CROSS_NAMES = ["color_contradiction", "shape_contradiction", "count_contradiction", "contradiction_fraction", "all_attributes_match", "known_attribute_fraction"]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+|\d+", text.lower())


def parse_caption(text: str) -> dict:
    tokens = tokenize(text)
    colors = {COLOR_ALIASES.get(t, t) for t in tokens if t in COLORS or t in COLOR_ALIASES}
    shapes = {SHAPE_ALIASES.get(t, t) for t in tokens if t in SHAPES or t in SHAPE_ALIASES}
    # Articles in phrases such as 'a picture of two ...' are not numeric evidence.
    explicit = {COUNTS[t] for t in tokens if t in COUNTS and t not in ("a", "an")}
    if not explicit and any(t in ("a", "an") for t in tokens):
        explicit = {1}
    return {"color": next(iter(colors)) if len(colors) == 1 else None,
            "shape": next(iter(shapes)) if len(shapes) == 1 else None,
            "count": next(iter(explicit)) if len(explicit) == 1 else None,
            "negated": any(t in {"no", "not", "never", "without", "neither"} for t in tokens),
            "ambiguous": len(colors) > 1 or len(shapes) > 1 or len(explicit) > 1}


def image_attributes(image: Image.Image | None) -> dict:
    """Estimate attributes from actual pixels; never consult manifest labels."""
    if image is None:
        return {"color": None, "shape": None, "count": None, "in_domain": False, "regions": [], "features": np.zeros(len(IMAGE_NAMES)), "reason": "No image was supplied."}
    rgb = np.asarray(image.convert("RGB").resize((128, 128)), dtype=np.float64)
    saturation = rgb.max(2) - rgb.min(2)
    mask = (saturation > 58) & (rgb.min(2) < 190)
    seen = np.zeros(mask.shape, dtype=bool)
    components = []
    for y, x in zip(*np.where(mask)):
        if seen[y, x]:
            continue
        todo = [(int(y), int(x))]
        seen[y, x] = True
        points = []
        while todo:
            cy, cx = todo.pop()
            points.append((cy, cx))
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < 128 and 0 <= nx < 128 and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    todo.append((ny, nx))
        if len(points) >= 25:
            arr = np.asarray(points)
            y0, x0 = arr.min(0)
            y1, x1 = arr.max(0) + 1
            fill = len(points) / ((y1 - y0) * (x1 - x0))
            components.append({"x": float(x0 / 128), "y": float(y0 / 128), "width": float((x1 - x0) / 128), "height": float((y1 - y0) / 128), "fill": float(fill), "area": len(points)})
    fg = rgb[mask]
    mean = fg.mean(0) if len(fg) else np.zeros(3)
    color = min(COLORS, key=lambda k: float(np.linalg.norm(mean - COLORS[k]))) if len(fg) else None
    color_distance = float(np.linalg.norm(mean - COLORS[color])) if color else 999.0
    fill = float(np.median([c["fill"] for c in components])) if components else 0.0
    shape = ("square" if fill > .89 else "circle" if fill > .66 else "triangle") if components else None
    count = len(components)
    background = rgb[~mask]
    background_fraction = float(np.mean((rgb.min(2) > 214) & (saturation < 35)))
    background_std = float(background.std(0).mean()) if len(background) else 999
    equal_fill = max([abs(c["fill"] - fill) for c in components], default=999)
    geometry_ok = all(.07 < c["width"] < .5 and .07 < c["height"] < .5 and .55 < c["width"] / c["height"] < 1.55 for c in components)
    in_domain = bool(1 <= count <= 3 and color_distance < 42 and background_fraction > .64 and background_std < 24 and equal_fill < .16 and geometry_ok)
    features = np.array([float(color == c) for c in COLORS] + [float(shape == s) for s in SHAPES] + [float(count == i) for i in (1, 2, 3)] + [float(mask.mean()), *list(mean / 255), fill])
    return {"color": color, "shape": shape, "count": count if count else None, "in_domain": in_domain, "features": features, "regions": components, "reason": "Supported isolated colored shapes." if in_domain else "Image does not match the controlled colored-shape domain.", "background_fraction": background_fraction, "color_distance": color_distance}


def cross_features(caption: dict, visual: dict) -> np.ndarray:
    known = [caption.get(k) is not None and visual.get(k) is not None for k in ("color", "shape", "count")]
    mismatch = [float(ok and caption[k] != visual[k]) for ok, k in zip(known, ("color", "shape", "count"))]
    return np.array(mismatch + [sum(mismatch) / max(sum(known), 1), float(all(known) and sum(mismatch) == 0), sum(known) / 3])


class Tfidf:
    def __init__(self, vocabulary=None, idf=None):
        self.vocabulary = vocabulary or {}
        self.idf = np.asarray(idf, dtype=float) if idf is not None else np.array([])

    @staticmethod
    def terms(text):
        words = tokenize(text)
        return words + [words[i] + " " + words[i + 1] for i in range(len(words) - 1)]

    def fit(self, texts):
        counts = Counter(t for text in texts for t in set(self.terms(text)))
        self.vocabulary = {term: i for i, term in enumerate(sorted(t for t, n in counts.items() if n >= 2))}
        self.idf = np.array([np.log((1 + len(texts)) / (1 + counts[t])) + 1 for t in self.vocabulary])
        return self

    def transform(self, texts):
        result = np.zeros((len(texts), len(self.vocabulary)))
        for i, text in enumerate(texts):
            for term, count in Counter(self.terms(text)).items():
                if term in self.vocabulary:
                    result[i, self.vocabulary[term]] = (1 + np.log(count)) * self.idf[self.vocabulary[term]]
        return result / np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-12)

    def to_dict(self):
        return {"vocabulary": self.vocabulary, "idf": self.idf.tolist()}
