"""Step 6 — Bundle adjustment with scipy.optimize.least_squares.

Minimizes the sum over all observations of || pi(R_c X_p + t_c) - x_cp ||^2
(in pixels), over all camera poses (axis-angle + translation) and 3D points.
Two stages: plain least squares, then remove observations with error >
outlier_px (leftover wrong matches) and solve again. One camera is held fixed
to remove the gauge freedom; overall scale is restored afterwards with the
known baseline (sfm.rescale).
"""
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


def _rotate(rvecs, pts):
    """Rodrigues rotation of pts (N,3) by per-row axis-angle rvecs (N,3)."""
    theta = np.linalg.norm(rvecs, axis=1)[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        k = np.where(theta > 1e-12, rvecs / theta, 0.0)
    cos, sin = np.cos(theta), np.sin(theta)
    dot = np.sum(pts * k, axis=1)[:, None]
    return cos * pts + sin * np.cross(k, pts) + dot * (1 - cos) * k


def reprojection_errors(poses, points, tracks, f_px):
    """Pixel reprojection error of every observation (for reporting)."""
    errs = []
    for tid, X in points.items():
        for c, (xn, _) in tracks[tid].items():
            if c in poses:
                R, t = poses[c]
                Xc = R @ X + t
                errs.append(np.linalg.norm(Xc[:2] / Xc[2] - xn) * f_px[c])
    return np.array(errs)


def bundle_adjust(poses, points, tracks, f_px, fixed, max_points=20000,
                  outlier_px=3.0, verbose=1, seed=0):
    cams = [c for c in poses if c != fixed]
    cidx = {c: i for i, c in enumerate(cams)}
    tids = list(points)
    if len(tids) > max_points:  # subsample for speed; poses are what we need
        rng = np.random.default_rng(seed)
        tids = list(rng.choice(tids, size=max_points, replace=False))
    pidx = {t: i for i, t in enumerate(tids)}

    obs_c, obs_p, obs_x, obs_f = [], [], [], []
    for t in tids:
        for c, (xn, _) in tracks[t].items():
            if c in poses:
                obs_c.append(cidx.get(c, -1))
                obs_p.append(pidx[t])
                obs_x.append(xn)
                obs_f.append(f_px[c])
    obs_c = np.array(obs_c)
    obs_p = np.array(obs_p)
    obs_x = np.array(obs_x)
    obs_f = np.array(obs_f)[:, None]

    R0, t0 = poses[fixed]
    r0 = cv2.Rodrigues(R0)[0].ravel()
    x0 = np.concatenate([
        np.concatenate([np.concatenate([cv2.Rodrigues(poses[c][0])[0].ravel(), poses[c][1]])
                        for c in cams]) if cams else np.zeros(0),
        np.concatenate([points[t] for t in tids])])
    nc = len(cams)

    def unpack(x):
        cp = x[:6 * nc].reshape(nc, 6)
        P = x[6 * nc:].reshape(-1, 3)
        # append the fixed camera as the last row (index -1)
        cp = np.vstack([cp, np.concatenate([r0, t0])[None]])
        return cp, P

    def residuals(x):
        cp, P = unpack(x)
        c = cp[obs_c]
        Xc = _rotate(c[:, :3], P[obs_p]) + c[:, 3:]
        return ((Xc[:, :2] / Xc[:, 2:3] - obs_x) * obs_f).ravel()

    def sparsity(oc, op):
        n = len(oc)
        A = lil_matrix((2 * n, 6 * nc + 3 * len(tids)), dtype=int)
        rows = np.arange(n)
        m = oc >= 0
        for k in range(6):
            A[2 * rows[m], 6 * oc[m] + k] = 1
            A[2 * rows[m] + 1, 6 * oc[m] + k] = 1
        for k in range(3):
            A[2 * rows, 6 * nc + 3 * op + k] = 1
            A[2 * rows + 1, 6 * nc + 3 * op + k] = 1
        return A

    all_c, all_p, all_x, all_f = obs_c, obs_p, obs_x, obs_f
    r_before = residuals(x0).reshape(-1, 2)
    x = x0
    # Stage 1: plain least squares (converges fast from a rough start).
    # Stage 2: drop observations with error > outlier_px, solve again.
    # (A Huber loss from the start converged very slowly in tests — see README.)
    for stage in range(2):
        if stage == 1:
            err = np.linalg.norm(residuals(x).reshape(-1, 2), axis=1)
            keep = err < outlier_px
            obs_c, obs_p, obs_x, obs_f = all_c[keep], all_p[keep], all_x[keep], all_f[keep]
        res = least_squares(residuals, x, jac_sparsity=sparsity(obs_c, obs_p), loss="linear",
                            x_scale="jac", method="trf", max_nfev=200, verbose=verbose)
        x = res.x
    n_inl = len(obs_c)
    obs_c, obs_p, obs_x, obs_f = all_c, all_p, all_x, all_f
    r_after = residuals(x).reshape(-1, 2)
    res.x = x
    cp, P = unpack(res.x)
    new_poses = {fixed: poses[fixed]}
    for c in cams:
        v = cp[cidx[c]]
        new_poses[c] = (cv2.Rodrigues(v[:3])[0], v[3:].copy())
    new_points = dict(points)
    for t in tids:
        new_points[t] = P[pidx[t]]
    stats = dict(n_obs=len(all_c), n_inlier_obs=n_inl, n_points=len(tids),
                 rms_before=float(np.sqrt(np.mean(np.sum(r_before ** 2, 1)))),
                 rms_after=float(np.sqrt(np.mean(np.sum(r_after ** 2, 1)))),
                 med_before=float(np.median(np.linalg.norm(r_before, axis=1))),
                 med_after=float(np.median(np.linalg.norm(r_after, axis=1))))
    return new_poses, new_points, stats
