import numpy as np

from weedrefine.data.mic_stats import label_class_stats, samples_with_class


def test_label_class_stats_skips_absent_and_ignore():
    label = np.array([[0, 0, 2], [255, 2, 2]], np.uint8)
    assert label_class_stats(label, 3) == {0: 2, 2: 3}


def test_samples_with_class():
    stats = [{0: 5, 2: 1, "file": "a.png"}, {0: 3, "file": "b.png"}]
    assert samples_with_class(stats) == {0: [["a.png", 5], ["b.png", 3]], 2: [["a.png", 1]]}
