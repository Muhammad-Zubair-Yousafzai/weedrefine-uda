"""Region-level majority voting on teacher pseudo-labels."""
import numpy as np


def refine_pseudo_label(prob, region_map, conf_thresh=0.7, min_purity=0.6,
                        min_confident_pixels=20, ignore_index=255,
                        mark_low_conf_ignore=False):
    """Refine a teacher pseudo-label using region masks.

    Args:
        prob: (C, H, W) teacher softmax probabilities.
        region_map: (H, W) int array, region id per pixel, -1 where no region.
        conf_thresh: only teacher pixels at or above this confidence vote.
        min_purity: a region is relabelled only if its majority class holds at
            least this fraction of the confident votes.
        min_confident_pixels: regions with fewer confident votes are left as is.
        mark_low_conf_ignore: if True, low confidence pixels that no region
            fixed are set to ignore_index (excluded from training).

    Returns:
        (H, W) uint8 refined label map.
    """
    num_classes = prob.shape[0]
    label = prob.argmax(0).astype(np.uint8)
    confident = prob.max(0) >= conf_thresh
    refined = label.copy()
    fixed = np.zeros(label.shape, dtype=bool)

    for rid in np.unique(region_map):
        if rid < 0:
            continue
        region = region_map == rid
        votes = label[region & confident]
        if votes.size < min_confident_pixels:
            continue
        counts = np.bincount(votes, minlength=num_classes)
        top = int(counts.argmax())
        if counts[top] / votes.size >= min_purity:
            refined[region] = top
            fixed |= region

    if mark_low_conf_ignore:
        refined[(~confident) & (~fixed)] = ignore_index
    return refined
