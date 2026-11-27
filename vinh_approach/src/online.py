"""Step 4 — "On the fly": accumulate matches over time and re-estimate.

A single synchronized frame gives only tens of reliable matches between two
widely separated cameras (plain dome walls, large viewpoint change). As new
frames arrive, the moving person provides NEW correspondences at new 3D
positions, so pooling matches over frames makes the essential matrix much
better constrained.

Each update: RANSAC (5-point) on all pooled matches -> non-linear refinement
on the inliers -> keep whichever of {new, previous} estimate explains the
pooled matches better (truncated Sampson cost). Without the last two steps the
estimate jumps around between updates.

Finding (see README): plain pooling of all frames works better than removing
duplicate static-background matches.
"""
import numpy as np

from data_loader import normalize_points
from two_view import msac_cost, refine_pose, relative_pose, sampson_px as _sampson_rt


class OnlinePair:
    """Pose between cameras a -> b, updated as frames arrive."""

    def __init__(self, cam_a, cam_b, window=None):
        self.cam_a, self.cam_b = cam_a, cam_b
        self.window = window            # None = use all frames seen so far
        self.frames = []                # list of (frame_id, xa, xb, ia, ib)
        self.R = self.t = None
        self.n_inliers = 0
        self.n_matches = 0

    def add_frame(self, frame_id, pts_a, pts_b, idx_a, idx_b):
        """pts_*: matched pixel coords; idx_*: keypoint indices (for tracks)."""
        xa = normalize_points(pts_a, self.cam_a)
        xb = normalize_points(pts_b, self.cam_b)
        self.frames.append((frame_id, xa, xb, idx_a, idx_b))
        if self.window is not None:
            self.frames = self.frames[-self.window:]

    def update(self):
        """Re-estimate (R, t) from all matches in the window."""
        if not self.frames:
            return None
        xa = np.vstack([f[1] for f in self.frames])
        xb = np.vstack([f[2] for f in self.frames])
        self.n_matches = len(xa)
        f_px = 0.5 * (self.cam_a.K[0, 0] + self.cam_b.K[0, 0])
        res = relative_pose(xa, xb, f_px=f_px)
        if res is None and self.R is None:
            self.n_inliers = 0
            return None
        if res is None:
            res = (self.R, self.t, None)
        R, t, _ = res
        R, t = refine_pose(R, t, xa, xb, f_px)
        # Keep the previous estimate if it still explains the pooled matches
        # better than the new one (RANSAC is random; this stops jumps).
        if self.R is not None:
            Rp, tp = refine_pose(self.R, self.t, xa, xb, f_px)
            if msac_cost(Rp, tp, xa, xb, f_px) < msac_cost(R, t, xa, xb, f_px):
                R, t = Rp, tp
        self.R, self.t = R, t
        inl = _sampson_rt(R, t, xa, xb, f_px) < 2.0
        self.n_inliers = int(inl.sum())
        return R, t, inl

    @property
    def inlier_ratio(self):
        return self.n_inliers / max(self.n_matches, 1)

    def verified_matches(self, max_px=2.0):
        """Per frame, the keypoint index pairs consistent with the current E.

        Returns {frame_id: (M,2) int array}. Used to build multi-camera tracks.
        """
        if self.R is None:
            return {}
        E = skew(self.t) @ self.R
        f_px = 0.5 * (self.cam_a.K[0, 0] + self.cam_b.K[0, 0])
        out = {}
        for fid, xa, xb, ia, ib in self.frames:
            err = sampson_px(E, xa, xb, f_px)
            keep = err < max_px
            out[fid] = np.stack([ia[keep], ib[keep]], axis=1)
        return out


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def sampson_px(E, xa, xb, f_px):
    """Sampson (first-order geometric) epipolar error in approx. pixels."""
    ha = np.hstack([xa, np.ones((len(xa), 1))])
    hb = np.hstack([xb, np.ones((len(xb), 1))])
    Ea = ha @ E.T            # E x_a
    Etb = hb @ E             # E^T x_b
    num = np.sum(hb * Ea, axis=1) ** 2
    den = Ea[:, 0] ** 2 + Ea[:, 1] ** 2 + Etb[:, 0] ** 2 + Etb[:, 1] ** 2
    return np.sqrt(num / np.maximum(den, 1e-18)) * f_px
