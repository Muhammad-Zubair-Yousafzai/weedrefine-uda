import numpy as np

from weedrefine.eval.region_quality import class_region_stats, region_oracle


def _setup():
    gt = np.zeros((4, 6), np.uint8)
    gt[:, 0:2] = 1           # crop
    gt[0, 4] = 2             # weed pixel inside a mostly background region
    gt[3, 5] = 2             # weed pixel in no region
    rm = np.full((4, 6), -1, np.int32)
    rm[:, 0:2] = 0           # pure crop region
    rm[0:2, 2:5] = 1         # background region holding one weed pixel
    return gt, rm


def test_region_oracle_majority_and_fill():
    gt, rm = _setup()
    o = region_oracle(rm, gt, 3)
    assert (o[:, 0:2] == 1).all()
    assert o[0, 4] == 0      # outvoted weed pixel is lost
    assert o[3, 5] == 0      # uncovered pixel gets fill


def test_region_oracle_ignores_ignore_pixels():
    gt = np.array([[255, 255, 2]], np.uint8)
    rm = np.zeros((1, 3), np.int32)
    assert (region_oracle(rm, gt, 3) == 2).all()


def test_class_region_stats_weed():
    gt, rm = _setup()
    s = class_region_stats(rm, gt, 2, 3)
    assert s == dict(total=2, covered=1, majority=0)
