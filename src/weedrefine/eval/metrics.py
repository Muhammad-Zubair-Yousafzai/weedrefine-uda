"""Per-class IoU and mIoU for semantic segmentation."""
import numpy as np


def confusion_matrix(pred, gt, num_classes, ignore_index=255):
    """num_classes x num_classes matrix, rows = ground truth, cols = prediction.

    Pixels where gt or pred equals ignore_index are skipped.
    """
    valid = (gt != ignore_index) & (pred != ignore_index)
    g = gt[valid].astype(np.int64)
    p = pred[valid].astype(np.int64)
    return np.bincount(g * num_classes + p,
                       minlength=num_classes ** 2).reshape(num_classes, num_classes)


def iou_from_confusion(cm):
    """Per-class IoU. Classes absent from both gt and pred get NaN."""
    tp = np.diag(cm).astype(np.float64)
    denom = cm.sum(0) + cm.sum(1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 0, tp / denom, np.nan)


class IoUMeter:
    """Accumulates a confusion matrix over a dataset, then reports IoU."""

    def __init__(self, num_classes, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred, gt):
        self.cm += confusion_matrix(pred, gt, self.num_classes, self.ignore_index)

    def summary(self, class_names=None):
        ious = iou_from_confusion(self.cm)
        names = class_names or [str(i) for i in range(self.num_classes)]
        out = {f"IoU_{n}": float(v) for n, v in zip(names, ious)}
        out["mIoU"] = float(np.nanmean(ious))
        return out
