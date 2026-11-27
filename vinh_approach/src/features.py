"""Step 2 — SIFT features and matching between two images."""
import cv2
import numpy as np

_sift = None


def detect(img, n_features=8000):
    """SIFT keypoints -> (pts (N,2) pixels, descriptors (N,128))."""
    global _sift
    if _sift is None:
        _sift = cv2.SIFT_create(nfeatures=n_features)
    kps, des = _sift.detectAndCompute(img, None)
    pts = np.array([k.pt for k in kps], dtype=np.float64).reshape(-1, 2)
    return pts, des


def match(des1, des2, ratio=0.75, mutual=True):
    """Return index pairs (M,2) into des1/des2.

    Lowe ratio test: keep a match only if the best neighbour is clearly better
    than the second best. Mutual check: the match must also be best in the
    reverse direction. Both remove ambiguous matches (repeated wall texture).
    """
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return np.empty((0, 2), dtype=int)
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))

    def ratio_matches(a, b):
        out = {}
        for pair in flann.knnMatch(a, b, k=2):
            if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
                out[pair[0].queryIdx] = pair[0].trainIdx
        return out

    fwd = ratio_matches(des1, des2)
    if mutual:
        bwd = ratio_matches(des2, des1)
        pairs = [(i, j) for i, j in fwd.items() if bwd.get(j) == i]
    else:
        pairs = list(fwd.items())
    return np.array(pairs, dtype=int).reshape(-1, 2)
