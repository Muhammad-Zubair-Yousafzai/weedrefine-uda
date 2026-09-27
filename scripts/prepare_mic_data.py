"""Convert ROSE domains used by the pair configs into an MMSegmentation layout for MIC.

For each domain (team, crop, year) in the pairs:
  {out}/{domain}/img/{stem}.{jpg|png}   copied image
  {out}/{domain}/ann/{stem}.png         uint8 label: 0 background, 1 crop, 2 weed, 255 ignore
  {out}/{domain}/splits/all.txt         every stem (source domains train on all)
Target domains also get one random split per seed in the pair config:
  {out}/{domain}/splits/seed{s}_train.txt, seed{s}_val.txt
Every domain gets MIC rare class sampling stats:
  {out}/{domain}/sample_class_stats.json, samples_with_class.json
"""
import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from weedrefine.data.mic_stats import label_class_stats, samples_with_class
from weedrefine.data.rose import decode_mask
from weedrefine.data.splits import domain_name, split_stems
from weedrefine.utils.io import ensure_dir, load_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True, help="pair configs, e.g. configs/pairs/*.yaml")
    ap.add_argument("--rose_cfg", default="configs/rose.yaml")
    ap.add_argument("--index", default="data/rose_index.json")
    ap.add_argument("--out", default="data/mic")
    ap.add_argument("--val_frac", type=float, default=0.2)
    args = ap.parse_args()

    rose = load_yaml(args.rose_cfg)
    with open(args.index) as f:
        index = json.load(f)

    # domain -> set of seeds it needs splits for (empty set = source only)
    seeds = defaultdict(set)
    for p in args.pairs:
        cfg = load_yaml(p)
        seeds[tuple(cfg["source"][k] for k in ("team", "crop", "year"))] |= set()
        seeds[tuple(cfg["target"][k] for k in ("team", "crop", "year"))] |= set(cfg["seeds"])

    for (team, crop, year), dom_seeds in sorted(seeds.items()):
        name = domain_name(team, crop, year)
        entries = [e for e in index if (e["team"], e["crop"], e["year"]) == (team, crop, year)]
        img_dir = ensure_dir(Path(args.out) / name / "img")
        ann_dir = ensure_dir(Path(args.out) / name / "ann")
        split_dir = ensure_dir(Path(args.out) / name / "splits")

        counts = np.zeros(256, np.int64)
        class_stats = []
        for e in tqdm(entries, desc=name):
            img = Path(e["image"])
            shutil.copy2(img, img_dir / img.name)
            label = decode_mask(cv2.imread(e["mask"]), rose["mask_colors"],
                                rose["class_names"], rose["ignore_index"])
            cv2.imwrite(str(ann_dir / f"{img.stem}.png"), label)
            counts += np.bincount(label.ravel(), minlength=256)
            class_stats.append({**label_class_stats(label, rose["num_classes"]),
                                "file": f"ann/{img.stem}.png"})
        with open(Path(args.out) / name / "sample_class_stats.json", "w") as f:
            json.dump(class_stats, f, indent=1)
        with open(Path(args.out) / name / "samples_with_class.json", "w") as f:
            json.dump(samples_with_class(class_stats), f, indent=1)

        stems = sorted(Path(e["image"]).stem for e in entries)
        (split_dir / "all.txt").write_text("\n".join(stems) + "\n")
        for s in sorted(dom_seeds):
            train, val = split_stems(stems, args.val_frac, s)
            (split_dir / f"seed{s}_train.txt").write_text("\n".join(train) + "\n")
            (split_dir / f"seed{s}_val.txt").write_text("\n".join(val) + "\n")

        total = counts.sum()
        shares = " ".join(f"{n} {100 * counts[i] / total:.2f}%" for i, n in enumerate(rose["class_names"]))
        suffixes = sorted({Path(e["image"]).suffix for e in entries})
        role = f"target, splits for seeds {sorted(dom_seeds)}" if dom_seeds else "source"
        print(f"{name}: {len(entries)} images {suffixes}, {role} | {shares} "
              f"ignore {100 * counts[rose['ignore_index']] / total:.3f}%")


if __name__ == "__main__":
    main()
