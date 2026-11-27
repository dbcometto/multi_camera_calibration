"""Step 3 — Relative pose between two cameras from point matches.

Uses Nistér's five-point algorithm inside RANSAC (cv2.findEssentialMat) on
undistorted normalized coordinates, then cv2.recoverPose to pick the one
(R, t) of the four candidates that puts the points in front of both cameras.

Convention: x_j = R @ x_i + t  (camera i -> camera j), ||t|| = 1.
"""
import cv2
import numpy as np


def relative_pose(xn_i, xn_j, f_px=1400.0, thresh_px=1.0, prob=0.9999):
    """xn_i, xn_j: (N,2) normalized coordinates of matched points.

    f_px converts the pixel threshold to normalized units (Panoptic HD focal
    length is ~1400-1600 px).
    Returns R (3,3), t (3,), inlier mask (N,) bool; or None if it fails.
    """
    if len(xn_i) < 8:
        return None
    E, mask = cv2.findEssentialMat(xn_i, xn_j, np.eye(3), method=cv2.RANSAC,
                                   prob=prob, threshold=thresh_px / f_px)
    if E is None:
        return None
    if E.shape[0] > 3:          # several solutions returned -> take the first
        E = E[:3]
    n_good, R, t, mask_pose = cv2.recoverPose(E, xn_i, xn_j, np.eye(3), mask=mask.copy())
    inliers = (mask_pose.ravel() > 0)
    return R, t.ravel(), inliers


def gt_relative_pose(cam_i, cam_j):
    """Ground-truth relative pose in the same convention (t in cm, not unit)."""
    R = cam_j.R_gt @ cam_i.R_gt.T
    t = cam_j.t_gt - R @ cam_i.t_gt
    return R, t


# ------------------------------------------------------------------ refinement
def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def sampson_px(R, t, xa, xb, f_px):
    """Sampson epipolar error (approx. pixels) of matches under pose (R, t)."""
    E = _skew(t) @ R
    ha = np.hstack([xa, np.ones((len(xa), 1))])
    hb = np.hstack([xb, np.ones((len(xb), 1))])
    Ea, Etb = ha @ E.T, hb @ E
    num = np.sum(hb * Ea, axis=1) ** 2
    den = Ea[:, 0] ** 2 + Ea[:, 1] ** 2 + Etb[:, 0] ** 2 + Etb[:, 1] ** 2
    return np.sqrt(num / np.maximum(den, 1e-18)) * f_px


def msac_cost(R, t, xa, xb, f_px, thr_px=2.0):
    """Truncated squared error: how well a pose explains ALL pooled matches."""
    return float(np.sum(np.minimum(sampson_px(R, t, xa, xb, f_px), thr_px) ** 2))


def refine_pose(R, t, xa, xb, f_px, thr_px=2.0):
    """Non-linear refinement of (R, t) on the inliers (minimizes Sampson error).

    RANSAC returns the model from the best 5-point minimal sample; refining it
    on all inliers makes it more accurate and far less jumpy between updates.
    t is kept unit length via spherical angles (5 DOF in total).
    """
    from scipy.optimize import least_squares
    inl = sampson_px(R, t, xa, xb, f_px) < thr_px
    if inl.sum() < 8:
        return R, t
    xa, xb = xa[inl], xb[inl]

    def unit(p):
        return np.array([np.cos(p[0]) * np.cos(p[1]), np.cos(p[0]) * np.sin(p[1]), np.sin(p[0])])

    p0 = np.array([0, 0, 0, np.arcsin(np.clip(t[2], -1, 1)), np.arctan2(t[1], t[0])])
    fun = lambda p: sampson_px(cv2.Rodrigues(p[:3])[0] @ R, unit(p[3:]), xa, xb, f_px)  # noqa: E731
    s = least_squares(fun, p0, loss="huber", f_scale=1.0)
    return cv2.Rodrigues(s.x[:3])[0] @ R, unit(s.x[3:])
