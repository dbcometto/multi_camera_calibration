"""Demo for steps 1-3: two-camera relative pose from ONE synchronized frame.

For every camera pair: SIFT -> ratio test -> essential matrix (5-point + RANSAC)
-> R, t; compared with ground truth. Saves a match picture and a results table.

Usage (from vinh_approach/):
    python3 scripts/run_two_view.py
    python3 scripts/run_two_view.py --frame 600 --cams 00_01 00_03
"""
import argparse
import itertools
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from data_loader import frame_ids, load_cameras, load_frame, normalize_points  # noqa: E402
from features import detect, match  # noqa: E402
from metrics import rotation_error_deg, translation_dir_error_deg  # noqa: E402
from two_view import gt_relative_pose, relative_pose  # noqa: E402

RESULTS = os.path.join(HERE, "..", "results")


def draw_inliers(img_i, img_j, p_i, p_j, out_path, max_lines=150):
    h = max(img_i.shape[0], img_j.shape[0])
    canvas = np.zeros((h, img_i.shape[1] + img_j.shape[1], 3), np.uint8)
    canvas[:img_i.shape[0], :img_i.shape[1]] = img_i
    canvas[:img_j.shape[0], img_i.shape[1]:] = img_j
    rng = np.random.default_rng(0)
    idx = rng.choice(len(p_i), size=min(max_lines, len(p_i)), replace=False)
    for k in idx:
        a = tuple(int(v) for v in p_i[k])
        b = (int(p_j[k][0] + img_i.shape[1]), int(p_j[k][1]))
        color = tuple(int(c) for c in rng.integers(64, 255, 3))
        cv2.line(canvas, a, b, color, 2)
        cv2.circle(canvas, a, 6, color, 2)
        cv2.circle(canvas, b, 6, color, 2)
    cv2.imwrite(out_path, cv2.resize(canvas, None, fx=0.5, fy=0.5))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq", default="171026_pose1")
    p.add_argument("--cams", nargs="+", default=["00_00", "00_01", "00_02", "00_03", "00_04"])
    p.add_argument("--frame", type=int, default=None, help="frame index (default: middle)")
    args = p.parse_args()

    cams = load_cameras(args.seq)
    ids = frame_ids(args.seq, args.cams)
    frame = args.frame if args.frame is not None else ids[len(ids) // 2]
    print(f"Sequence {args.seq}, frame {frame}, cameras {args.cams}\n")

    feats = {}
    for name in args.cams:
        feats[name] = detect(load_frame(args.seq, name, frame))
        print(f"  {name}: {len(feats[name][0])} SIFT keypoints")
    print()

    os.makedirs(os.path.join(RESULTS, "two_view"), exist_ok=True)
    rows = []
    header = f"{'pair':<13}{'matches':>8}{'inliers':>8}{'rot err°':>10}{'t dir err°':>11}"
    print(header)
    print("-" * len(header))
    for a, b in itertools.combinations(args.cams, 2):
        (pa, da), (pb, db) = feats[a], feats[b]
        m = match(da, db)
        res = None
        if len(m) >= 8:
            xa = normalize_points(pa[m[:, 0]], cams[a])
            xb = normalize_points(pb[m[:, 1]], cams[b])
            f_px = 0.5 * (cams[a].K[0, 0] + cams[b].K[0, 0])
            res = relative_pose(xa, xb, f_px=f_px)
        if res is None:
            print(f"{a + '-' + b:<13}{len(m):>8}{'-':>8}{'failed':>10}")
            rows.append((a, b, len(m), 0, np.nan, np.nan))
            continue
        R, t, inl = res
        R_gt, t_gt = gt_relative_pose(cams[a], cams[b])
        er, et = rotation_error_deg(R, R_gt), translation_dir_error_deg(t, t_gt)
        print(f"{a + '-' + b:<13}{len(m):>8}{inl.sum():>8}{er:>10.2f}{et:>11.2f}")
        rows.append((a, b, len(m), int(inl.sum()), er, et))
        draw_inliers(load_frame(args.seq, a, frame, gray=False),
                     load_frame(args.seq, b, frame, gray=False),
                     pa[m[inl, 0]], pb[m[inl, 1]],
                     os.path.join(RESULTS, "two_view", f"matches_{a}_{b}_f{frame}.jpg"))

    out = os.path.join(RESULTS, "two_view", f"two_view_f{frame}.csv")
    with open(out, "w") as f:
        f.write("cam_i,cam_j,matches,inliers,rot_err_deg,t_dir_err_deg\n")
        for r in rows:
            f.write(",".join(str(v) for v in r) + "\n")
    print(f"\nSaved {out} and match images in results/two_view/")


if __name__ == "__main__":
    main()
