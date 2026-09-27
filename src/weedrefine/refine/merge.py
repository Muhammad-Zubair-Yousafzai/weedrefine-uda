"""Merge SAM leaf fragments into whole plant regions using feature similarity."""
import cv2
import numpy as np


class _UnionFind:
    def __init__(self, items):
        self.parent = {i: i for i in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def region_mean_features(region_map, features):
    """Mean L2-normalised feature per region.

    features: (h, w, D), either the size of region_map or a coarser patch grid covering
    the same image. Each region is area-downsampled onto the grid and its mean feature is
    weighted by how much of each cell it covers, so tiny regions still get a feature.
    """
    gh, gw = features.shape[:2]
    flat = features.reshape(gh * gw, -1)
    means = {}
    for rid in np.unique(region_map):
        if rid < 0:
            continue
        wgt = cv2.resize((region_map == rid).astype(np.float32), (gw, gh),
                         interpolation=cv2.INTER_AREA).reshape(-1)
        f = wgt @ flat / (wgt.sum() + 1e-8)
        means[int(rid)] = f / (np.linalg.norm(f) + 1e-8)
    return means


def adjacent_pairs(region_map, dilate_px=3):
    """Pairs of region ids that touch after dilating each region by dilate_px."""
    kernel = np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), np.uint8)
    pairs = set()
    for rid in np.unique(region_map):
        if rid < 0:
            continue
        grown = cv2.dilate((region_map == rid).astype(np.uint8), kernel) > 0
        for other in np.unique(region_map[grown]):
            if other >= 0 and other != rid:
                pairs.add((min(int(rid), int(other)), max(int(rid), int(other))))
    return pairs


def merge_fragments(region_map, features, sim_thresh=0.85, dilate_px=3):
    """Union adjacent regions whose mean features have cosine similarity >= sim_thresh."""
    means = region_mean_features(region_map, features)
    if not means:
        return region_map.copy()
    uf = _UnionFind(means.keys())
    for a, b in adjacent_pairs(region_map, dilate_px):
        if float(means[a] @ means[b]) >= sim_thresh:
            uf.union(a, b)
    merged = region_map.copy()
    for rid in means:
        root = uf.find(rid)
        if root != rid:
            merged[region_map == rid] = root
    return merged
