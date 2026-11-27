"""Step 4 demo — "calibration on the fly": accuracy vs number of frames seen.

Part A (pairs): for 4 camera pairs, re-estimate the two-view pose after every
new frame and plot rotation / translation-direction error.
Part B (all cameras): run the whole n-camera pipeline using only the first K
frames, for growing K, and plot mean camera error and #cameras registered.

Usage (from vinh_approach/):
    python3 scripts/run_online.py            # both parts (~8 min)
    python3 scripts/run_online.py --part A   # pairs only (~30 s)
    python3 scripts/run_online.py --part plotB  # redraw part B figure from CSV
"""
import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

from cache import load_features, load_matches  # noqa: E402
from data_loader import frame_ids, load_cameras  # noqa: E402
from metrics import rotation_error_deg, translation_dir_error_deg  # noqa: E402
from online import OnlinePair  # noqa: E402
from plots import INK2, SERIES, setup  # noqa: E402
from run_pipeline import CAMS, run  # noqa: E402
from two_view import gt_relative_pose  # noqa: E402

SEQ = "171026_pose1"
RESULTS = os.path.join(HERE, "..", "results")
PAIRS = [("00_01", "00_04"), ("00_00", "00_29"), ("00_02", "00_03"), ("00_01", "00_03")]
K_LIST = [1, 2, 3, 5, 8, 13, 20, 30, 53]


def part_a():
    C = load_cameras(SEQ)
    ids = frame_ids(SEQ, CAMS)
    feats = load_features(SEQ, CAMS, ids)
    matches = load_matches(SEQ, CAMS, ids)
    curves = {}
    for a, b in PAIRS:
        p = OnlinePair(C[a], C[b])
        Rg, tg = gt_relative_pose(C[a], C[b])
        rot, tdir = [], []
        for f in ids:
            m = matches[(a, b)][f]
            if len(m):
                p.add_frame(f, feats[(a, f)][0][m[:, 0]], feats[(b, f)][0][m[:, 1]],
                            m[:, 0], m[:, 1])
            res = p.update()
            rot.append(rotation_error_deg(res[0], Rg) if res else np.nan)
            tdir.append(translation_dir_error_deg(res[1], tg) if res else np.nan)
        curves[(a, b)] = (np.array(rot), np.array(tdir))
        print(f"  {a}-{b}: frame 1 -> {rot[0]:.2f}°/{tdir[0]:.2f}°, "
              f"all {len(ids)} frames -> {rot[-1]:.2f}°/{tdir[-1]:.2f}°")

    with open(os.path.join(RESULTS, "online_pairs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair", "n_frames", "rot_err_deg", "t_dir_err_deg"])
        for (a, b), (r, t) in curves.items():
            for k in range(len(r)):
                w.writerow([f"{a}-{b}", k + 1, r[k], t[k]])

    plt = setup()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    x = np.arange(1, len(ids) + 1)
    for ax, j, title in [(axes[0], 0, "Rotation error"), (axes[1], 1, "Translation direction error")]:
        for i, (pair, cv) in enumerate(curves.items()):
            ax.plot(x, np.maximum(cv[j], 1e-3), color=SERIES[i], label=f"{pair[0]} – {pair[1]}")
        ax.set_yscale("log")
        ax.set_xlabel("frames seen (1 frame per second)")
        ax.set_title(title, loc="left")
        ax.axhline(1.0, color=INK2, lw=1, ls=":")
    axes[0].set_ylabel("error vs ground truth (degrees, log)")
    axes[1].legend(loc="upper right", title="camera pair")
    fig.suptitle("Two-camera pose improves as frames arrive", x=0.01, ha="left",
                 fontweight="bold", fontsize=14)
    fig.savefig(os.path.join(RESULTS, "online_pairs.png"))
    print("  saved results/online_pairs.png")


def part_b():
    rows = []
    for k in K_LIST:
        try:
            r = run(SEQ, CAMS, k, log=lambda *_: None)
            row = dict(frames=k, registered=len(r["post"]),
                       rot_noba=np.mean([v["rot_deg"] for v in r["pre"].values()]),
                       pos_noba=np.mean([v["pos_cm"] for v in r["pre"].values()]),
                       rot_ba=np.mean([v["rot_deg"] for v in r["post"].values()]),
                       pos_ba=np.mean([v["pos_cm"] for v in r["post"].values()]))
        except Exception as e:  # too few matches to build a graph
            print(f"  K={k}: failed ({type(e).__name__}: {e})")
            row = dict(frames=k, registered=0, rot_noba=np.nan, pos_noba=np.nan,
                       rot_ba=np.nan, pos_ba=np.nan)
        rows.append(row)
        print(f"  K={k:2d} frames: {row['registered']}/{len(CAMS)} cameras, "
              f"no BA {row['rot_noba']:.3f}° {row['pos_noba']:.2f} cm | "
              f"BA {row['rot_ba']:.3f}° {row['pos_ba']:.2f} cm", flush=True)

    with open(os.path.join(RESULTS, "online_ncam.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    plot_b(rows)


def plot_b(rows=None):
    if rows is None:  # re-plot from the saved CSV without re-running
        with open(os.path.join(RESULTS, "online_ncam.csv")) as f:
            rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    plt = setup()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    k = np.array([r["frames"] for r in rows])
    reg = np.array([r["registered"] for r in rows])

    ax = axes[0]
    ax.plot(k, reg, "o-", color=SERIES[0], ms=6)
    for x, y in zip(k, reg):
        ax.annotate(f"{int(y)}", (x, y), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=10, color=INK2)
    ax.set_ylim(-0.5, len(CAMS) + 1.5)
    ax.set_ylabel("cameras calibrated")
    ax.set_title(f"Cameras calibrated (of {len(CAMS)})", loc="left")

    ax = axes[1]
    ok = reg > 0
    ax.plot(k[ok], [r["rot_noba"] for r, o in zip(rows, ok) if o], "o-", color=SERIES[1],
            ms=6, label="pose graph + PnP (no BA)")
    ax.plot(k[ok], [r["rot_ba"] for r, o in zip(rows, ok) if o], "o-", color=SERIES[0],
            ms=6, label="+ bundle adjustment")
    ax.set_ylim(0, None)
    ax.set_ylabel("degrees")
    ax.set_title("Mean rotation error of calibrated cameras", loc="left")
    ax.legend(loc="upper left")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(k)
        ax.set_xticklabels([int(v) for v in k])
        ax.minorticks_off()
        ax.set_xlabel("frames seen (1 frame per second)")
    fig.suptitle("n cameras on the fly: more frames -> more cameras, each already < 0.1° with BA",
                 x=0.01, ha="left", fontweight="bold", fontsize=14)
    fig.savefig(os.path.join(RESULTS, "online_ncam.png"))
    print("  saved results/online_ncam.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["A", "B", "both", "plotB"], default="both")
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    if args.part in ("A", "both"):
        print("Part A: camera pairs")
        part_a()
    if args.part in ("B", "both"):
        print("Part B: all cameras, growing number of frames")
        part_b()
    if args.part == "plotB":
        plot_b()
