"""Indexing and loading for the ROSE Challenge crop/weed data.

Layout and mask colours come from configs/rose.yaml.
"""
from pathlib import Path

import cv2
import numpy as np


def decode_mask(mask_bgr, class_colors, class_names, ignore_index=255):
    """Map an RGB colour mask to class ids.

    class_colors maps each class name to a list of [r, g, b] colours.
    Unmatched colours become ignore_index.
    """
    rgb = mask_bgr[..., ::-1]
    out = np.full(rgb.shape[:2], ignore_index, dtype=np.uint8)
    for cid, name in enumerate(class_names):
        for color in np.atleast_2d(np.array(class_colors[name], dtype=np.uint8)):
            out[np.all(rgb == color, axis=-1)] = cid
    return out


def build_index(raw_root, cfg):
    """List of {team, crop, year, image, mask} dicts matched by file stem per folder.

    crop is the folder name without its year suffix (Haricot_2021 -> Haricot).
    """
    root = Path(raw_root)
    exts = {e.lower() for e in cfg["image_exts"]}
    entries = []
    for year, spec in cfg["folders"].items():
        for team in spec["teams"]:
            for folder in spec["crops"]:
                fmt = dict(team=team, crop=folder, year=year)
                img_dir = root / cfg["image_dir"].format(**fmt)
                mask_dir = root / cfg["mask_dir"].format(**fmt)
                masks = {m.stem: m for m in mask_dir.glob("*.png")}
                crop = folder.removesuffix(f"_{year}")
                for img in sorted(img_dir.iterdir() if img_dir.is_dir() else []):
                    m = masks.get(img.stem)
                    if img.suffix.lower() in exts and m is not None:
                        entries.append(dict(team=team, crop=crop, year=year,
                                            image=str(img), mask=str(m)))
    return entries


def load_pair(entry, cfg):
    """Load (image_rgb uint8 HxWx3, label uint8 HxW)."""
    img = cv2.imread(entry["image"])[..., ::-1].copy()
    mask = cv2.imread(entry["mask"])
    label = decode_mask(mask, cfg["mask_colors"], cfg["class_names"], cfg["ignore_index"])
    return img, label
