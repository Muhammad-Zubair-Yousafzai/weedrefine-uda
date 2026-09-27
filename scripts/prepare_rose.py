"""Index the raw ROSE download into a JSON file of image/mask pairs."""
import argparse
import json
from collections import Counter

from weedrefine.data.rose import build_index
from weedrefine.utils.io import load_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="root of the unzipped ROSE data")
    ap.add_argument("--cfg", default="configs/rose.yaml")
    ap.add_argument("--out", default="data/rose_index.json")
    args = ap.parse_args()

    cfg = load_yaml(args.cfg)
    entries = build_index(args.raw, cfg)
    with open(args.out, "w") as f:
        json.dump(entries, f, indent=1)

    counts = Counter((e["team"], e["crop"], e["year"]) for e in entries)
    print(f"{len(entries)} image/mask pairs written to {args.out}")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")
    if not entries:
        print("No pairs found. Inspect the folder tree and fix image_glob / mask_glob in the config.")


if __name__ == "__main__":
    main()
