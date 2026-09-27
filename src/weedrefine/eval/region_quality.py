"""How well a region map fits the ground truth, independent of any teacher."""
import numpy as np


def region_oracle(region_map, gt, num_classes, ignore_index=255, fill=0):
    """Label each region with its majority ground-truth class.

    This is the best a region majority vote could do with a perfect teacher.
    Pixels in no region get `fill` (background), so missed weeds count as errors.
    Ignore pixels in gt do not vote.
    """
    out = np.full(gt.shape, fill, dtype=np.uint8)
    valid = gt != ignore_index
    for rid in np.unique(region_map):
        if rid < 0:
            continue
        region = region_map == rid
        votes = gt[region & valid]
        if votes.size:
            out[region] = np.bincount(votes, minlength=num_classes).argmax()
    return out


def class_region_stats(region_map, gt, cls, num_classes, ignore_index=255):
    """Pixel counts for one class: total, covered by any region, in regions where it is the majority."""
    oracle = region_oracle(region_map, gt, num_classes, ignore_index)
    is_cls = gt == cls
    return dict(total=int(is_cls.sum()),
                covered=int((is_cls & (region_map >= 0)).sum()),
                majority=int((is_cls & (region_map >= 0) & (oracle == cls)).sum()))
