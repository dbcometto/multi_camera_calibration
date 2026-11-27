"""Step 1 — Load CMU Panoptic frames and calibration.

Only intrinsics (K, distortion) are used as inputs to the pipeline.
Ground-truth extrinsics (R, t) are loaded separately and used only for evaluation.
"""
import glob
import json
import os

import cv2
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


class Camera:
    def __init__(self, entry):
        self.name = entry["name"]                       # e.g. "00_03"
        self.K = np.array(entry["K"], dtype=np.float64)
        self.dist = np.array(entry["distCoef"], dtype=np.float64).ravel()
        # Ground truth (world -> camera): x_cam = R X + t, t in centimetres.
        self.R_gt = np.array(entry["R"], dtype=np.float64)
        self.t_gt = np.array(entry["t"], dtype=np.float64).reshape(3)

    @property
    def center_gt(self):
        return -self.R_gt.T @ self.t_gt


def load_cameras(seq="171026_pose1"):
    """Return {name: Camera} for all HD cameras."""
    with open(os.path.join(DATA_DIR, seq, f"calibration_{seq}.json")) as f:
        cal = json.load(f)
    return {c["name"]: Camera(c) for c in cal["cameras"] if c["type"] == "hd"}


def frame_ids(seq, cam_names):
    """Frame indices that exist for every camera (synchronized by index)."""
    sets = []
    for name in cam_names:
        files = glob.glob(os.path.join(DATA_DIR, seq, "frames", f"hd_{name}", "*.jpg"))
        sets.append({int(os.path.basename(p)[:-4]) for p in files})
    return sorted(set.intersection(*sets))


def load_frame(seq, cam_name, frame_id, gray=True):
    path = os.path.join(DATA_DIR, seq, "frames", f"hd_{cam_name}", f"{frame_id:06d}.jpg")
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def normalize_points(pts, cam):
    """Pixel points (N,2) -> undistorted normalized image coordinates (N,2).

    Undistorting the points (instead of the whole image) is faster and exact.
    In normalized coordinates K = I, so one essential matrix works for two
    cameras with different intrinsics.
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.undistortPoints(pts, cam.K, cam.dist).reshape(-1, 2)
