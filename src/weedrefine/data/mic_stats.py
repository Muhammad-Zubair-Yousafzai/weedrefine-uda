"""Rare class sampling stats in the format MIC reads (see MIC tools/convert_datasets/gta.py).

MIC's UDADataset loads {source data_root}/sample_class_stats.json and
samples_with_class.json, and matches file.split('/')[-1] to the seg map file name.
"""
import numpy as np


def label_class_stats(label, num_classes):
    """{class_id: pixel count} for classes present in label. Ignore pixels are skipped."""
    counts = np.bincount(label.ravel(), minlength=256)[:num_classes]
    return {c: int(n) for c, n in enumerate(counts) if n > 0}


def samples_with_class(sample_stats):
    """[{class: n, ..., 'file': f}] -> {class: [[file, n], ...]}."""
    out = {}
    for s in sample_stats:
        for c, n in s.items():
            if c != "file":
                out.setdefault(c, []).append([s["file"], n])
    return out
