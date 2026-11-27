"""Full pipeline (steps 1-7) on n cameras, using the first --frames frames.

Usage (from vinh_approach/):
    python3 scripts/run_pipeline.py                 # all 12 cameras, all frames
    python3 scripts/run_pipeline.py --frames 10     # only the first 10 frames
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from bundle_adjustment import bundle_adjust, reprojection_errors  # noqa: E402
from cache import load_features, load_matches  # noqa: E402
from data_loader import frame_ids, load_cameras  # noqa: E402
from metrics import evaluate_poses, rotation_error_deg, translation_dir_error_deg  # noqa: E402
from two_view import gt_relative_pose  # noqa: E402
from online import OnlinePair  # noqa: E402
from pose_graph import build_graph  # noqa: E402
from sfm import build_tracks, incremental_sfm, rescale  # noqa: E402

CAMS = ["00_00", "00_01", "00_02", "00_03", "00_04", "00_07",
        "00_13", "00_18", "00_21", "00_24", "00_28", "00_29"]
RESULTS = os.path.join(HERE, "..", "results")


def run(seq, cams, n_frames=None, log=print):
    C = load_cameras(seq)
    C = {c: C[c] for c in cams}
    ids = frame_ids(seq, cams)
    if n_frames:
        ids = ids[:n_frames]
    feats = load_features(seq, cams, ids)
    matches = load_matches(seq, cams, ids)

    # Step 2-4: pairwise poses, pooled over all frames so far
    pairs = {}
    for (a, b), per_frame in matches.items():
        p = OnlinePair(C[a], C[b])
        for f in ids:
            m = per_frame[f]
            if len(m):
                p.add_frame(f, feats[(a, f)][0][m[:, 0]], feats[(b, f)][0][m[:, 1]],
                            m[:, 0], m[:, 1])
        p.update()
        pairs[(a, b)] = p

    # Step 5: graph with outlier-edge rejection (no ground truth used)
    g = build_graph(pairs)
    log(f"  graph: {len(g['candidates'])} candidate edges, "
        f"{len(g['triangle_ok'])} pass triangle check, {len(g['kept'])} kept")

    verified = {e: pairs[e].verified_matches() for e in g["kept"]}
    tracks = build_tracks(feats, verified, C, ids)
    log(f"  {len(tracks)} multi-camera tracks")

    # Step 5-6: incremental registration; the only metric input is ONE baseline
    a, b = max(g["kept"], key=lambda e: pairs[e].n_inliers)
    baseline = float(np.linalg.norm(C[a].center_gt - C[b].center_gt))
    f_px = {c: C[c].K[0, 0] for c in C}
    # Baseline for comparison: same registration but with NO bundle adjustment
    noba_poses, _, _, (ref, other) = incremental_sfm(g, pairs, tracks, C, baseline,
                                                     ba_every=False, log=lambda *_: None)
    pre = evaluate_poses(rescale(noba_poses, ref, other, baseline)[0], C, ref)
    # Full method: bundle adjustment after each new camera + final BA
    poses, points, order, (ref, other) = incremental_sfm(g, pairs, tracks, C, baseline, log=log)
    poses, points, ba = bundle_adjust(poses, points, tracks, f_px, fixed=ref, verbose=0)
    poses, _ = rescale(poses, ref, other, baseline)
    post = evaluate_poses(poses, C, ref)
    reproj = reprojection_errors(poses, {k: points[k] for k in list(points)[:20000]}, tracks, f_px)
    return dict(ids=ids, pairs=pairs, graph=g, poses=poses, points=points, order=order,
                ref=ref, other=other, baseline=baseline, ba=ba, pre=pre, post=post,
                reproj_median=float(np.median(reproj)), cams=C, tracks=tracks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="171026_pose1")
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--cams", nargs="+", default=CAMS)
    args = ap.parse_args()
    t0 = time.time()
    r = run(args.seq, args.cams, args.frames)
    print(f"\nRegistered {len(r['poses'])}/{len(args.cams)} cameras using {len(r['ids'])} frames "
          f"({time.time() - t0:.0f} s). Reference camera {r['ref']}.")
    print(f"Bundle adjustment: RMS {r['ba']['rms_before']:.2f} -> {r['ba']['rms_after']:.2f} px, "
          f"median {r['ba']['med_before']:.2f} -> {r['ba']['med_after']:.2f} px "
          f"({r['ba']['n_points']} points, {r['ba']['n_obs']} observations)\n")
    print(f"{'camera':8}{'rot° no BA':>12}{'cm no BA':>10}{'rot° BA':>10}{'cm BA':>8}")
    for c in sorted(r["post"]):
        b, a = r["pre"][c], r["post"][c]
        print(f"{c:8}{b['rot_deg']:12.3f}{b['pos_cm']:10.2f}{a['rot_deg']:10.3f}{a['pos_cm']:8.2f}")
    rot = [v["rot_deg"] for v in r["post"].values()]
    pos = [v["pos_cm"] for v in r["post"].values()]
    rot0 = [v["rot_deg"] for v in r["pre"].values()]
    pos0 = [v["pos_cm"] for v in r["pre"].values()]
    print(f"{'mean':8}{np.mean(rot0):12.3f}{np.mean(pos0):10.2f}{np.mean(rot):10.3f}{np.mean(pos):8.2f}")
    # Evaluation only (ground truth is NOT used by the graph step itself):
    # did the triangle/loop checks keep the right camera pairs?
    def pair_err(e):
        Rg, tg = gt_relative_pose(r["cams"][e[0]], r["cams"][e[1]])
        p = r["pairs"][e]
        return max(rotation_error_deg(p.R, Rg), translation_dir_error_deg(p.t, tg))
    g = r["graph"]
    bad = {e for e, p in r["pairs"].items() if p.R is not None and pair_err(e) > 5}
    cand, kept = set(g["candidates"]), set(g["kept"])
    print(f"\nGraph check vs ground truth: {len(r['pairs'])} pairs, {len(bad)} truly wrong (>5°).")
    print(f"  passed inlier gates: {len(cand)} ({len(cand & bad)} wrong) -> "
          f"kept {len(kept)} ({len(kept & bad)} wrong), rejected {len(cand - kept)} "
          f"({len((cand - kept) & bad)} wrong, {len((cand - kept) - bad)} good)")
    print(f"  worst kept pair: {max(pair_err(e) for e in kept):.2f}°")
    graph_eval = dict(pairs=len(r["pairs"]), truly_wrong=len(bad), candidates=len(cand),
                      wrong_candidates=len(cand & bad), kept=len(kept), wrong_kept=len(kept & bad),
                      rejected=len(cand - kept), good_rejected=len((cand - kept) - bad),
                      worst_kept_deg=max(pair_err(e) for e in kept))
    missing = sorted(set(args.cams) - set(r["poses"]))
    if missing:
        print("Not registered:", missing)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "pipeline_result.json"), "w") as f:
        json.dump(dict(frames=len(r["ids"]), ref=r["ref"], baseline_cm=r["baseline"],
                       order=r["order"], bundle_adjustment=r["ba"], graph_eval=graph_eval,
                       kept_edges=[list(e) for e in r["graph"]["kept"]],
                       rejected_edges={f"{a}-{b}": e for (a, b), e in r["graph"]["rejected"].items()},
                       no_ba=r["pre"], with_ba=r["post"],
                       poses={c: dict(R=R.tolist(), t=t.tolist()) for c, (R, t) in r["poses"].items()}),
                  f, indent=1)


if __name__ == "__main__":
    main()
