"""Train VeriFuse classifier heads on the separately documented real-data corpus."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from verifuse.real_model import ClipEncoder, encode_manifest, load_embeddings, train, validate_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/real/manifest.json")
    parser.add_argument("--encoder", type=Path, default=ROOT / "models/real/clip")
    parser.add_argument("--embeddings", type=Path, help="Documented precomputed NPZ with ids,image,text; otherwise extract from local files with resumable caching.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--c-values", type=float, nargs="+", default=[.001, .01, .1, 1.0, 10.0])
    parser.add_argument("--encode-only", action="store_true", help="Prepare/resume CLIP features without fitting heads.")
    args = parser.parse_args()
    if not args.manifest.is_file():
        parser.error("Prepare the real-data manifest before training: " + str(args.manifest))
    if any(c <= 0 for c in args.c_values):
        parser.error("Regularization C values must be positive.")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    audit = validate_manifest(manifest)
    print("Dataset splits: " + json.dumps(audit["splits"]), flush=True)
    if args.embeddings:
        image, text, metadata = load_embeddings(args.embeddings, manifest["records"])
    else:
        encoder = ClipEncoder(args.encoder, batch_size=args.batch_size, threads=args.threads)
        image, text, metadata = encode_manifest(ROOT, manifest, encoder)
    if not args.encode_only:
        result = train(ROOT, manifest, image, text, metadata, candidates=args.c_values)
        for experiment in result["experiments"][:4]:
            metric = experiment["metrics"]
            print(f"{experiment['name']}: test accuracy={metric['accuracy']:.4f}, F1={metric['f1']:.4f}, ROC-AUC={metric['roc_auc']:.4f}", flush=True)
        print("Saved model: models/real/bundle.json; report: results/real/report.md", flush=True)


if __name__ == "__main__":
    main()
