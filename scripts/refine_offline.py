"""Phase A experiment: does SAM + DINO refinement improve teacher pseudo-labels?

Expects one saved teacher softmax per target image at
  {teacher_pred_dir}/{image_stem}.npy   shape (C, H, W), float16 or float32
Reports pseudo-label IoU against target ground truth for:
  teacher argmax, SAM vote only, SAM + DINO merge + vote.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from weedrefine.data.rose import load_pair
from weedrefine.eval.metrics import IoUMeter
from weedrefine.refine.merge import merge_fragments
from weedrefine.refine.sam_regions import SamRegionGenerator
from weedrefine.refine.vote import refine_pseudo_label
from weedrefine.utils.io import ensure_dir, load_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--rose_cfg", default="configs/rose.yaml")
    ap.add_argument("--index", default="data/rose_index.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    rose = load_yaml(args.rose_cfg)
    with open(args.index) as f:
        index = json.load(f)
    tgt = cfg["target"]
    entries = [e for e in index if e["team"] == tgt["team"]
               and e["crop"] == tgt["crop"] and e["year"] == tgt["year"]]
    print(f"{len(entries)} target images")

    sam = SamRegionGenerator(cfg["sam"]["checkpoint"], cfg["sam"]["model_cfg"],
                             device=args.device,
                             points_per_side=cfg["sam"]["points_per_side"],
                             pred_iou_thresh=cfg["sam"]["pred_iou_thresh"],
                             stability_score_thresh=cfg["sam"]["stability_score_thresh"],
                             min_mask_region_area=cfg["sam"]["min_mask_region_area"])
    feat = None
    if cfg["merge"]["enabled"]:
        from weedrefine.refine.features import DenseFeatureExtractor
        feat = DenseFeatureExtractor(cfg["merge"]["feature_model"], device=args.device,
                                     long_side=cfg["merge"].get("feature_long_side", 1024))

    n, names = rose["num_classes"], rose["class_names"]
    meters = {k: IoUMeter(n, rose["ignore_index"]) for k in ["teacher", "sam_vote", "sam_dino_vote"]}
    out_dir = ensure_dir(cfg["out_dir"])
    v = cfg["vote"]

    for e in tqdm(entries):
        stem = Path(e["image"]).stem
        pred_path = Path(cfg["teacher_pred_dir"]) / f"{stem}.npy"
        if not pred_path.exists():
            continue
        img, gt = load_pair(e, rose)
        prob = np.load(pred_path).astype(np.float32)

        regions = sam(img)
        np.save(out_dir / f"{stem}_regions.npy", regions)

        meters["teacher"].update(prob.argmax(0).astype(np.uint8), gt)
        meters["sam_vote"].update(
            refine_pseudo_label(prob, regions, v["conf_thresh"], v["min_purity"],
                                v["min_confident_pixels"]), gt)
        if feat is not None:
            merged = merge_fragments(regions, feat(img), cfg["merge"]["sim_thresh"],
                                     cfg["merge"]["dilate_px"])
            meters["sam_dino_vote"].update(
                refine_pseudo_label(prob, merged, v["conf_thresh"], v["min_purity"],
                                    v["min_confident_pixels"]), gt)

    results = {k: m.summary(names) for k, m in meters.items() if m.cm.sum() > 0}
    print(json.dumps(results, indent=2))
    with open(out_dir / "pseudo_label_iou.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
