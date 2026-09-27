"""Teacher-free check: can SAM regions (and DINO merged regions) represent weeds at all?

For a few target images, compares regions with ground truth:
  coverage  = weed pixels inside any region
  majority  = weed pixels in regions where weed is the majority (survive a vote)
  oracle    = IoU of the map where each region takes its majority GT class
The oracle weed IoU is the ceiling for region voting with a perfect teacher.
Region maps are saved as .npy so later steps can reuse them.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from weedrefine.data.rose import load_pair
from weedrefine.eval.metrics import IoUMeter
from weedrefine.eval.region_quality import class_region_stats, region_oracle
from weedrefine.refine.merge import merge_fragments
from weedrefine.refine.sam_regions import SamRegionGenerator
from weedrefine.utils.io import ensure_dir, load_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="pair config; its target and sam/merge settings are used")
    ap.add_argument("--rose_cfg", default="configs/rose.yaml")
    ap.add_argument("--index", default="data/rose_index.json")
    ap.add_argument("--checkpoint", help="override sam.checkpoint (e.g. tiny on CPU)")
    ap.add_argument("--model_cfg", help="override sam.model_cfg")
    ap.add_argument("--n", type=int, default=5, help="images to sample (only images with weed)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="outputs/sam_check")
    args = ap.parse_args()

    cfg, rose = load_yaml(args.config), load_yaml(args.rose_cfg)
    tgt, s, m = cfg["target"], cfg["sam"], cfg["merge"]
    with open(args.index) as f:
        entries = [e for e in json.load(f) if e["team"] == tgt["team"]
                   and e["crop"] == tgt["crop"] and e["year"] == tgt["year"]]
    order = np.random.default_rng(args.seed).permutation(len(entries))

    sam = SamRegionGenerator(args.checkpoint or s["checkpoint"], args.model_cfg or s["model_cfg"],
                             device=args.device, points_per_side=s["points_per_side"],
                             pred_iou_thresh=s["pred_iou_thresh"],
                             stability_score_thresh=s["stability_score_thresh"],
                             min_mask_region_area=s["min_mask_region_area"])
    from weedrefine.refine.features import DenseFeatureExtractor
    feat = DenseFeatureExtractor(m["feature_model"], device=args.device,
                                 long_side=m.get("feature_long_side", 1024))

    n, names, weed = rose["num_classes"], rose["class_names"], rose["class_names"].index("weed")
    out_dir = ensure_dir(Path(args.out) / cfg["name"])
    meters = {k: IoUMeter(n, rose["ignore_index"]) for k in ("sam", "sam_dino")}
    stats = {k: dict(total=0, covered=0, majority=0) for k in meters}
    used = []

    for i in order:
        if len(used) == args.n:
            break
        e = entries[i]
        img, gt = load_pair(e, rose)
        if not (gt == weed).any():
            continue
        t = time.time()
        regions = sam(img)
        merged = merge_fragments(regions, feat(img), m["sim_thresh"], m["dilate_px"])
        stem = Path(e["image"]).stem
        np.save(out_dir / f"{stem}_sam.npy", regions)
        np.save(out_dir / f"{stem}_sam_dino.npy", merged)
        line = [f"{stem}: {time.time() - t:.0f}s, weed px {int((gt == weed).sum())}"]
        for k, rm in (("sam", regions), ("sam_dino", merged)):
            meters[k].update(region_oracle(rm, gt, n, rose["ignore_index"]), gt)
            st = class_region_stats(rm, gt, weed, n, rose["ignore_index"])
            for f in st:
                stats[k][f] += st[f]
            line.append(f"{k}: {len(np.unique(rm[rm >= 0]))} regions, "
                        f"weed majority {st['majority'] / st['total']:.0%}")
        print(" | ".join(line), flush=True)
        used.append(stem)

    print(f"\n{cfg['name']} target {tgt}, {len(used)} images with weed (seed {args.seed})")
    for k in meters:
        sm, st = meters[k].summary(names), stats[k]
        print(f"{k:9s} weed coverage {st['covered'] / st['total']:.1%}  "
              f"weed in weed-majority regions {st['majority'] / st['total']:.1%}  "
              f"oracle IoU bg {sm['IoU_background']:.3f} crop {sm['IoU_crop']:.3f} "
              f"weed {sm['IoU_weed']:.3f} mIoU {sm['mIoU']:.3f}")


if __name__ == "__main__":
    main()
