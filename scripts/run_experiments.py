"""Train real lightweight models and save all six reproducible evaluations."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from verifuse.dataset import generate, import_csv, load_manifest
from verifuse.experiments import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, help="Import an external corpus CSV; writes a separate model/report.")
    parser.add_argument("--output", type=Path, help="Result directory. Required with --csv.")
    args = parser.parse_args()
    if args.csv and not args.output:
        parser.error("--csv requires --output to preserve the validated demo model")
    if args.csv:
        manifest = import_csv(args.csv)
    else:
        path = ROOT / "data" / "manifest.json"
        manifest = load_manifest(path) if path.exists() else generate(ROOT)
    results = run(ROOT, manifest, args.output, controlled=not bool(args.csv))
    print(f"Saved six experiments. Feature fusion test accuracy: {results['experiments'][2]['metrics']['accuracy']:.4f}")
