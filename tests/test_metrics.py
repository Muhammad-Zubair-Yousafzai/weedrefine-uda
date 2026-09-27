import numpy as np

from weedrefine.eval.metrics import IoUMeter, confusion_matrix, iou_from_confusion


def test_perfect_prediction():
    gt = np.array([[0, 1], [2, 2]], dtype=np.uint8)
    ious = iou_from_confusion(confusion_matrix(gt, gt, 3))
    assert np.allclose(ious, 1.0)


def test_ignore_index_skipped():
    gt = np.array([[0, 255], [1, 1]], dtype=np.uint8)
    pred = np.array([[0, 1], [1, 1]], dtype=np.uint8)
    cm = confusion_matrix(pred, gt, 2)
    assert cm.sum() == 3


def test_known_iou():
    gt = np.array([0, 0, 1, 1], dtype=np.uint8)
    pred = np.array([0, 1, 1, 1], dtype=np.uint8)
    m = IoUMeter(2)
    m.update(pred, gt)
    s = m.summary(["a", "b"])
    assert np.isclose(s["IoU_a"], 0.5)
    assert np.isclose(s["IoU_b"], 2 / 3)
