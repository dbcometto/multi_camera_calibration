"""Step 7 — Error metrics against ground truth."""
import numpy as np


def rotation_error_deg(R_est, R_gt):
    """Angle of the rotation R_est^T R_gt (0 = perfect)."""
    c = (np.trace(R_est.T @ R_gt) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def translation_dir_error_deg(t_est, t_gt):
    """Angle between translation directions (scale-free, since two-view t has no scale)."""
    a = t_est / np.linalg.norm(t_est)
    b = t_gt / np.linalg.norm(t_gt)
    return float(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0))))


def gt_in_ref_frame(cams_obj, ref):
    """Ground-truth poses re-expressed with camera `ref` as the world origin
    (same gauge as the estimate, whose world frame is the seed camera)."""
    Rr, tr = cams_obj[ref].R_gt, cams_obj[ref].t_gt
    out = {}
    for c, cam in cams_obj.items():
        R = cam.R_gt @ Rr.T
        out[c] = (R, cam.t_gt - R @ tr)
    return out


def evaluate_poses(poses, cams_obj, ref):
    """Per camera: rotation error (deg), camera-centre position error (cm)."""
    gt = gt_in_ref_frame(cams_obj, ref)
    rows = {}
    for c, (R, t) in poses.items():
        Rg, tg = gt[c]
        C, Cg = -R.T @ t, -Rg.T @ tg
        rows[c] = dict(rot_deg=rotation_error_deg(R, Rg),
                       pos_cm=float(np.linalg.norm(C - Cg)))
    return rows
