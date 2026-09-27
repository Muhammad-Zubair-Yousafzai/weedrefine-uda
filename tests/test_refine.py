import numpy as np

from weedrefine.refine.features import grid_size
from weedrefine.refine.merge import merge_fragments, region_mean_features
from weedrefine.refine.sam_regions import masks_to_region_map
from weedrefine.refine.vote import refine_pseudo_label


def _prob_from_label(label, n=3, conf=0.9):
    prob = np.full((n,) + label.shape, (1 - conf) / (n - 1), dtype=np.float32)
    for c in range(n):
        prob[c][label == c] = conf
    return prob


def test_vote_fixes_noisy_pixels_inside_region():
    label = np.zeros((10, 10), np.uint8)
    label[2:8, 2:8] = 2            # weed region
    label[4, 4] = 1                # one wrong crop pixel inside the weed
    prob = _prob_from_label(label)
    regions = np.full((10, 10), -1, np.int32)
    regions[2:8, 2:8] = 0
    out = refine_pseudo_label(prob, regions, min_confident_pixels=5)
    assert out[4, 4] == 2
    assert (out[2:8, 2:8] == 2).all()


def test_impure_region_untouched():
    label = np.zeros((10, 10), np.uint8)
    label[:, 5:] = 1
    prob = _prob_from_label(label)
    regions = np.zeros((10, 10), np.int32)   # one region split 50/50
    out = refine_pseudo_label(prob, regions, min_purity=0.6, min_confident_pixels=5)
    assert (out == label).all()


def test_low_confidence_pixels_do_not_vote():
    label = np.full((10, 10), 1, np.uint8)
    prob = _prob_from_label(label, conf=0.5)  # everything below threshold
    regions = np.zeros((10, 10), np.int32)
    out = refine_pseudo_label(prob, regions, conf_thresh=0.7)
    assert (out == label).all()


def test_small_masks_painted_on_top():
    big = np.zeros((6, 6), bool); big[:, :] = True
    small = np.zeros((6, 6), bool); small[2:4, 2:4] = True
    rm = masks_to_region_map([{"segmentation": small, "area": 4},
                              {"segmentation": big, "area": 36}], (6, 6))
    assert rm[2, 2] != rm[0, 0]


def test_merge_joins_similar_neighbours_only():
    rm = np.full((6, 12), -1, np.int32)
    rm[:, 0:4] = 0
    rm[:, 4:8] = 1
    rm[:, 8:12] = 2
    feats = np.zeros((6, 12, 2), np.float32)
    feats[:, 0:8] = [1, 0]          # regions 0 and 1 look the same
    feats[:, 8:12] = [0, 1]         # region 2 differs
    merged = merge_fragments(rm, feats, sim_thresh=0.9, dilate_px=1)
    assert merged[0, 0] == merged[0, 5]
    assert merged[0, 5] != merged[0, 10]


def test_merge_with_coarse_feature_grid():
    # 32x48 region map, 2x3 feature grid (16 px cells), like DINO patches.
    rm = np.full((32, 48), -1, np.int32)
    rm[:, 0:16] = 0
    rm[:, 16:32] = 1
    rm[:, 32:48] = 2
    grid = np.zeros((2, 3, 2), np.float32)
    grid[:, 0:2] = [1, 0]
    grid[:, 2] = [0, 1]
    merged = merge_fragments(rm, grid, sim_thresh=0.9, dilate_px=1)
    assert merged[0, 0] == merged[0, 20]
    assert merged[0, 20] != merged[0, 40]


def test_tiny_region_gets_feature_of_its_cell():
    # A 3x3 px weed inside one 16 px cell must not vanish on the coarse grid.
    rm = np.zeros((32, 32), np.int32)
    rm[20:23, 20:23] = 1
    grid = np.zeros((2, 2, 2), np.float32)
    grid[:, :] = [1, 0]
    grid[1, 1] = [0, 1]
    means = region_mean_features(rm, grid)
    assert set(means) == {0, 1}
    assert np.allclose(means[1], [0, 1], atol=1e-5)


def test_grid_size_keeps_aspect_and_patch_multiple():
    assert grid_size(768, 1024, 1024, 16) == (768, 1024)
    assert grid_size(1536, 2048, 1024, 16) == (768, 1024)
    h, w = grid_size(3456, 5184, 1024, 16)
    assert w == 1024 and h % 16 == 0 and abs(h / w - 3456 / 5184) < 0.02
