from pathlib import Path

import numpy as np

from weedrefine.data.rose import decode_mask
from weedrefine.utils.io import load_yaml

CFG = load_yaml(Path(__file__).resolve().parents[1] / "configs" / "rose.yaml")


def test_decode_mask_all_colors():
    rgb = np.array([[[0, 0, 0], [254, 124, 18]],
                    [[255, 255, 255], [216, 67, 82]],
                    [[0, 255, 0], [216, 67, 83]]], dtype=np.uint8)
    bgr = rgb[..., ::-1]
    label = decode_mask(bgr, CFG["mask_colors"], CFG["class_names"], CFG["ignore_index"])
    expected = np.array([[0, 0],     # black, orange -> background
                         [1, 2],     # white -> crop, red -> weed
                         [255, 255]  # unknown colours -> ignore
                         ], dtype=np.uint8)
    assert label.dtype == np.uint8
    assert np.array_equal(label, expected)
