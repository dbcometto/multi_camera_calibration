"""Figures for the presentation from one full pipeline run (all frames).

  results/camera_graph.png   top view: cameras, kept vs rejected camera pairs
  results/cameras_3d.png     estimated vs ground-truth cameras + 3D points
  results/per_camera_error.png  error per camera, with / without bundle adjustment
  results/matches_00_00_00_29.jpg  verified matches for one pair (one frame)

Usage (from vinh_approach/):  python3 scripts/make_figures.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

from data_loader import load_frame  # noqa: E402
from plots import CRITICAL, INK, INK2, SERIES, setup  # noqa: E402
from run_pipeline import CAMS, run  # noqa: E402
from run_two_view import draw_inliers  # noqa: E402

SEQ = "171026_pose1"
RESULTS = os.path.join(HERE, "..", "results")


def to_world(r, X):
    """Estimate lives in the reference camera frame -> map to Panoptic world
    (using the reference camera's GT pose) only so we can draw both together."""
    cam = r["cams"][r["ref"]]
    return (np.asarray(X) - cam.t_gt) @ cam.R_gt  # = R^T (X - t)


def main():
    plt = setup()
    r = run(SEQ, CAMS, None, log=print)
    C = r["cams"]
    names = sorted(C)
    gt_c = {c: C[c].center_gt for c in names}
    est_c = {c: to_world(r, -R.T @ t) for c, (R, t) in r["poses"].items()}
    # Panoptic world: y points down -> top view uses (x, z), height = -y.

    # ---------- 1. camera graph (top view)
    fig, ax = plt.subplots(figsize=(7.5, 7))
    g = r["graph"]
    for (a, b), _ in g["rejected"].items():
        pa, pb = gt_c[a], gt_c[b]
        ax.plot([pa[0], pb[0]], [pa[2], pb[2]], color=CRITICAL, lw=1, ls="--", alpha=0.7, zorder=1)
    for a, b in g["kept"]:
        pa, pb = gt_c[a], gt_c[b]
        ax.plot([pa[0], pb[0]], [pa[2], pb[2]], color=SERIES[0], lw=2, zorder=2)
    for c in names:
        p = gt_c[c]
        d = C[c].R_gt[2]
        ax.scatter(p[0], p[2], s=90, color=INK, zorder=3, edgecolor="white", linewidth=2)
        ax.annotate("", xy=(p[0] + 40 * d[0], p[2] + 40 * d[2]), xytext=(p[0], p[2]),
                    arrowprops=dict(arrowstyle="->", color=INK2, lw=1.2), zorder=3)
        ax.annotate(c[-2:], (p[0], p[2]), textcoords="offset points", xytext=(8, 6),
                    fontsize=11, color=INK, fontweight="bold")
    ax.plot([], [], color=SERIES[0], lw=2, label=f"kept pair ({len(g['kept'])})")
    ax.plot([], [], color=CRITICAL, lw=1, ls="--",
            label=f"rejected by triangle/loop check ({len(g['rejected'])})")
    ax.set_aspect("equal")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("z (cm)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.1), ncol=2)
    ax.set_title("Camera graph (top view, arrows = viewing direction)", loc="left")
    fig.savefig(os.path.join(RESULTS, "camera_graph.png"))
    plt.close(fig)

    # ---------- 2. 3D cameras + points
    pts = np.array(list(r["points"].values()))
    rng = np.random.default_rng(0)
    pts = to_world(r, pts[rng.choice(len(pts), size=min(8000, len(pts)), replace=False)])
    G0 = np.array([gt_c[c] for c in names])
    center = G0.mean(axis=0)
    pts = pts[np.linalg.norm(pts - center, axis=1) < 450]  # drop far outliers for display
    fig = plt.figure(figsize=(10, 7.5))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(pts[:, 0], pts[:, 2], -pts[:, 1], s=0.5, color=INK2, alpha=0.35)
    G = np.array([gt_c[c] for c in names])
    E = np.array([est_c[c] for c in names if c in est_c])
    ax.scatter(G[:, 0], G[:, 2], -G[:, 1], s=140, facecolor="none", edgecolor=SERIES[1],
               linewidth=2, label="ground truth")
    ax.scatter(E[:, 0], E[:, 2], -E[:, 1], s=30, color=SERIES[0], label="our estimate")
    for c in names:
        if c in r["poses"]:
            R = r["poses"][c][0] @ C[r["ref"]].R_gt   # world -> cam rotation
            d = R[2] * 60
            p = est_c[c]
            ax.plot([p[0], p[0] + d[0]], [p[2], p[2] + d[2]], [-p[1], -p[1] - d[1]],
                    color=SERIES[0], lw=1.5)
            ax.text(p[0], p[2], -p[1] + 15, c[-2:], fontsize=9, color=INK)
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("z (cm)")
    ax.set_zlabel("height (cm)")
    lo, hi = G.min(axis=0) - 60, G.max(axis=0) + 60
    lo[1] = min(lo[1], pts[:, 1].min())
    hi[1] = max(hi[1], pts[:, 1].max())
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[2], hi[2])
    ax.set_zlim(-hi[1], -lo[1])
    ax.set_box_aspect((hi[0] - lo[0], hi[2] - lo[2], hi[1] - lo[1]))
    ax.view_init(elev=28, azim=-60)
    ax.legend(loc="upper left")
    ax.set_title(f"{len(r['poses'])} cameras + {len(r['points'])} triangulated points", loc="left")
    fig.savefig(os.path.join(RESULTS, "cameras_3d.png"))
    plt.close(fig)

    # ---------- 3. per-camera error, no BA vs BA
    cams_sorted = [c for c in names if c != r["ref"]]
    x = np.arange(len(cams_sorted))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, key, title, unit in [(axes[0], "rot_deg", "Rotation error", "degrees"),
                                 (axes[1], "pos_cm", "Camera position error", "cm")]:
        ax.bar(x - 0.2, [r["pre"][c][key] for c in cams_sorted], 0.38, color=SERIES[1],
               label="no bundle adjustment")
        ax.bar(x + 0.2, [r["post"][c][key] for c in cams_sorted], 0.38, color=SERIES[0],
               label="with bundle adjustment")
        ax.set_xticks(x)
        ax.set_xticklabels([c[-2:] for c in cams_sorted])
        ax.set_xlabel(f"camera (reference = {r['ref'][-2:]})")
        ax.set_ylabel(unit)
        ax.set_title(title, loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].legend(loc="upper right")
    fig.savefig(os.path.join(RESULTS, "per_camera_error.png"))
    plt.close(fig)

    # ---------- 4. verified matches for one neighbouring pair (one frame)
    a, b = "00_00", "00_29"
    from cache import load_features
    f = r["ids"][len(r["ids"]) // 2]
    feats = load_features(SEQ, [a, b], [f])
    key = (a, b)
    vm = r["pairs"][key].verified_matches()[f]
    draw_inliers(load_frame(SEQ, a, f, gray=False), load_frame(SEQ, b, f, gray=False),
                 feats[(a, f)][0][vm[:, 0]], feats[(b, f)][0][vm[:, 1]],
                 os.path.join(RESULTS, f"matches_{a}_{b}.jpg"), max_lines=120)
    print("saved figures in results/")


if __name__ == "__main__":
    main()
