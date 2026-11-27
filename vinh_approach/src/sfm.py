"""Steps 5-6 — Register all cameras into one frame (incremental, SfM-style).

1. Tracks: within each synchronized frame, verified pairwise matches are
   joined (union-find) into multi-camera tracks = one 3D point seen by
   several cameras at the same instant.
2. Seed: the best graph edge (most inliers). Camera A = world origin,
   camera B = two-view pose with |t| set to ONE known baseline (the only
   metric information used -> fixes the global scale).
3. Triangulate tracks seen by registered cameras.
4. Repeat: register the camera with the most 2D-3D correspondences by PnP +
   RANSAC (scale comes automatically from the 3D points), triangulate more.
5. Bundle adjustment over all cameras and points (bundle_adjustment.py).

Pose convention: x_cam = R X + t (world = seed camera A frame).
"""
import cv2
import numpy as np

from bundle_adjustment import bundle_adjust
from data_loader import normalize_points


# ---------------------------------------------------------------- tracks
def build_tracks(feats, verified, cams_obj, frame_ids):
    """verified: {(a,b): {frame: (M,2) kp index pairs}}.

    Returns list of tracks; each track = {cam: (xn (2,), px (2,))}.
    """
    tracks = []
    for f in frame_ids:
        parent = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for (a, b), per_frame in verified.items():
            m = per_frame.get(f)
            if m is None:
                continue
            for i, j in m:
                ra, rb = find((a, int(i))), find((b, int(j)))
                if ra != rb:
                    parent[ra] = rb
        groups = {}
        for node in list(parent):
            groups.setdefault(find(node), []).append(node)
        for nodes in groups.values():
            cams_in = [c for c, _ in nodes]
            if len(nodes) < 2 or len(set(cams_in)) != len(cams_in):
                continue  # inconsistent track (two keypoints in one image) -> drop
            tr = {}
            for c, k in nodes:
                px = feats[(c, f)][0][k]
                tr[c] = (normalize_points(px[None], cams_obj[c])[0], px)
            tracks.append(tr)
    return tracks


# ---------------------------------------------------------- triangulation
def triangulate(obs, poses):
    """Linear (DLT) triangulation from >=2 views. obs: {cam: xn}."""
    A = []
    for c, x in obs.items():
        R, t = poses[c]
        P = np.hstack([R, t[:, None]])
        A.append(x[0] * P[2] - P[0])
        A.append(x[1] * P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.array(A))
    X = Vt[-1]
    return X[:3] / X[3] if abs(X[3]) > 1e-12 else None


def check_point(X, obs, poses, f_px, max_px=4.0, min_angle_deg=1.0):
    centers = []
    for c, x in obs.items():
        R, t = poses[c]
        Xc = R @ X + t
        if Xc[2] <= 0:
            return False
        if np.linalg.norm(Xc[:2] / Xc[2] - x) * f_px[c] > max_px:
            return False
        centers.append(-R.T @ t)
    # largest angle between viewing rays (small angle = unstable depth)
    rays = [(X - C) / np.linalg.norm(X - C) for C in centers]
    best = max(np.degrees(np.arccos(np.clip(r1 @ r2, -1, 1)))
               for i, r1 in enumerate(rays) for r2 in rays[i + 1:])
    return best >= min_angle_deg


def triangulate_all(tracks, poses, f_px):
    points = {}
    for tid, tr in enumerate(tracks):
        obs = {c: v[0] for c, v in tr.items() if c in poses}
        if len(obs) < 2:
            continue
        X = triangulate(obs, poses)
        if X is not None and check_point(X, obs, poses, f_px):
            points[tid] = X
    return points


# ---------------------------------------------------------- registration
def register_pnp(cam, tracks, points, f_px, min_inliers=30):
    obj, img = [], []
    for tid, X in points.items():
        if cam in tracks[tid]:
            obj.append(X)
            img.append(tracks[tid][cam][0])
    if len(obj) < max(min_inliers, 6):
        return None, len(obj)
    obj = np.array(obj, np.float64)
    img = np.array(img, np.float64)
    ok, rvec, tvec, inl = cv2.solvePnPRansac(
        obj, img, np.eye(3), None, reprojectionError=4.0 / f_px[cam],
        iterationsCount=2000, confidence=0.9999, flags=cv2.SOLVEPNP_EPNP)
    if not ok or inl is None or len(inl) < min_inliers:
        return None, len(obj)
    inl = inl.ravel()
    rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img[inl], np.eye(3), None, rvec, tvec)
    R, _ = cv2.Rodrigues(rvec)
    return (R, tvec.ravel()), len(inl)


def incremental_sfm(graph, pairs, tracks, cams_obj, baseline_cm, ba_every=True, log=print):
    f_px = {c: cams_obj[c].K[0, 0] for c in cams_obj}
    a, b = max(graph["kept"], key=lambda e: pairs[e].n_inliers)
    p = pairs[(a, b)]
    poses = {a: (np.eye(3), np.zeros(3)),
             b: (p.R, p.t / np.linalg.norm(p.t) * baseline_cm)}
    log(f"  seed pair {a}-{b} ({p.n_inliers} inliers), scale from baseline {baseline_cm:.1f} cm")
    points = triangulate_all(tracks, poses, f_px)
    log(f"  triangulated {len(points)} points")
    order = [a, b]
    connected = {c for e in graph["kept"] for c in e}
    while True:
        todo = [c for c in connected if c not in poses]
        if not todo:
            break
        counts = {c: sum(1 for t in points if c in tracks[t]) for c in todo}
        c = max(counts, key=counts.get)
        pose, n = register_pnp(c, tracks, points, f_px)
        if pose is None:
            log(f"  could not register {c} ({counts[c]} 2D-3D matches); stopping")
            break
        poses[c] = pose
        order.append(c)
        points = triangulate_all(tracks, poses, f_px)
        log(f"  + {c}: PnP inliers {n}, points now {len(points)}")
        if ba_every and len(poses) >= 3:
            poses, points, _ = bundle_adjust(poses, points, tracks, f_px, fixed=a,
                                             max_points=8000, verbose=0)
    return poses, points, order, (a, b)


def rescale(poses, ref, other, baseline_cm):
    """BA leaves the overall scale free; re-apply the one known baseline."""
    Ca = -poses[ref][0].T @ poses[ref][1]
    Cb = -poses[other][0].T @ poses[other][1]
    s = baseline_cm / np.linalg.norm(Cb - Ca)
    return {c: (R, t * s) for c, (R, t) in poses.items()}, s
