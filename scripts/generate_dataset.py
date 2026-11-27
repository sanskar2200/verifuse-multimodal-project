"""Generate the complete local synthetic corpus without downloads."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from verifuse.dataset import generate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7031)
    args = parser.parse_args()
    manifest = generate(ROOT, seed=args.seed)
    print(f"Generated {len(manifest['records'])} records in {ROOT / 'data' / 'manifest.json'}")
