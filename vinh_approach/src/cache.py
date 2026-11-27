"""Compute-once cache for SIFT features and per-frame pairwise matches.

Features: data/<seq>/cache/feat_<cam>.npz  (keypoints + uint8 descriptors per frame)
Matches:  data/<seq>/cache/match_<a>_<b>.npz (index pairs per frame)

SIFT descriptor values are integers 0..255 stored as float, so uint8 storage is
lossless and 4x smaller.
"""
import itertools
import os
from multiprocessing import Pool

import cv2
import numpy as np

from data_loader import DATA_DIR, load_frame

SIFT_CONTRAST = 0.01   # default 0.04; lower finds more corners on the plain dome walls
SIFT_MAX = 8000


def _cache_dir(seq):
    d = os.path.join(DATA_DIR, seq, "cache")
    os.makedirs(d, exist_ok=True)
    return d


def _detect_cam(args):
    seq, cam, ids = args
    path = os.path.join(_cache_dir(seq), f"feat_{cam}.npz")
    if os.path.exists(path):
        return path
    sift = cv2.SIFT_create(nfeatures=SIFT_MAX, contrastThreshold=SIFT_CONTRAST)
    out = {}
    for f in ids:
        kps, des = sift.detectAndCompute(load_frame(seq, cam, f), None)
        out[f"p{f}"] = np.array([k.pt for k in kps], np.float32).reshape(-1, 2)
        out[f"d{f}"] = (des if des is not None else np.zeros((0, 128))).astype(np.uint8)
    np.savez(path, **out)
    return path


def load_features(seq, cams, ids, workers=8):
    """Return {(cam, frame): (pts float64 (N,2), des float32 (N,128))}."""
    with Pool(min(workers, len(cams))) as pool:
        paths = pool.map(_detect_cam, [(seq, c, ids) for c in cams])
    feats = {}
    for cam, path in zip(cams, paths):
        z = np.load(path)
        for f in ids:
            feats[(cam, f)] = (z[f"p{f}"].astype(np.float64), z[f"d{f}"].astype(np.float32))
    return feats


def _match_pair(args):
    seq, a, b, ids = args
    from features import match  # imported here so workers get a fresh matcher
    path = os.path.join(_cache_dir(seq), f"match_{a}_{b}.npz")
    if os.path.exists(path):
        return path
    za = np.load(os.path.join(_cache_dir(seq), f"feat_{a}.npz"))
    zb = np.load(os.path.join(_cache_dir(seq), f"feat_{b}.npz"))
    out = {f"m{f}": match(za[f"d{f}"].astype(np.float32), zb[f"d{f}"].astype(np.float32))
           for f in ids}
    np.savez(path, **out)
    return path


def load_matches(seq, cams, ids, workers=12):
    """Return {(a, b): {frame: (M,2) index pairs}} for every camera pair a<b."""
    pairs = list(itertools.combinations(sorted(cams), 2))
    with Pool(workers) as pool:
        paths = pool.map(_match_pair, [(seq, a, b, ids) for a, b in pairs])
    out = {}
    for (a, b), path in zip(pairs, paths):
        z = np.load(path)
        out[(a, b)] = {f: z[f"m{f}"] for f in ids}
    return out
