"""Deterministic train/val splits of a domain's image stems."""
import numpy as np


def split_stems(stems, val_frac, seed):
    """Shuffle sorted stems with `seed` and cut off round(val_frac * n) for validation.

    Returns (train, val), each sorted. Same inputs always give the same split.
    """
    stems = sorted(stems)
    n_val = int(round(val_frac * len(stems)))
    order = np.random.default_rng(seed).permutation(len(stems))
    val = sorted(stems[i] for i in order[:n_val])
    train = sorted(stems[i] for i in order[n_val:])
    return train, val


def domain_name(team, crop, year):
    return f"{team}_{crop}_{year}".lower()
