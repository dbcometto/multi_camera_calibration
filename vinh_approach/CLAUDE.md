# CLAUDE.md — Vinh's approach (EECE 7150 AFR Group 1, Project 3)

Context for Claude Code. Humans: start with `README.md` in this folder.

## What this project is

Course final project, Northeastern EECE 7150 (Autonomous Field Robotics), Group 1.
Team: Dante Cometto, Daniel Pitsch, Vinh Vu, Hraday Dasadia.

| Project | Topic | Presentation |
|---|---|---|
| **3 (this folder)** | Calibration on the fly — **extrinsics**, Camera–Camera → nCamera | 12/02/2026 |
| 1 | Calibration on the fly — **timing** (synchronization), Camera–Camera | 12/09/2026 |

Instructor guidance: "attempt and solve it, using a paper or yourself", then demo and present.
Each teammate may have their own approach folder; **this folder is Vinh's**.
Do not edit the team's top-level files (`../README.md`, `../requirements.txt`) without asking.

## The problem in one paragraph

Find the rotation R and translation t of every camera relative to the others
**without a calibration target** (no checkerboard), using only natural scene
features, improving the estimate as frames arrive. Intrinsics (K, distortion)
are given. Ground-truth extrinsics are used **only for evaluation**, plus
**one** known baseline length to fix the global scale.

## The approach (classical, no learning)

1. Load synchronized frames (1 fps), undistort keypoints — `src/data_loader.py`
2. SIFT (contrastThreshold 0.01) + FLANN, Lowe ratio 0.75 + mutual check — `src/features.py`, cached by `src/cache.py`
3. Two-view pose: 5-point essential matrix + RANSAC (Nistér) → recoverPose → Sampson-error refinement — `src/two_view.py`
4. On the fly: pool matches over frames, re-estimate each frame, keep the previous estimate if it fits the pool better — `src/online.py` (`OnlinePair`)
5. Camera graph: inlier gates → triangle rotation check → max spanning tree → loop check, rejects wrong pairs without ground truth — `src/pose_graph.py`
6. Union-find tracks per frame → seed pair scaled by 1 baseline → PnP+RANSAC to add cameras → DLT triangulation → bundle adjustment — `src/sfm.py`, `src/bundle_adjustment.py`
7. Errors vs ground truth (rotation °, camera position cm) — `src/metrics.py`

Main paper: Nistér 2003/2004 (five-point). Related work: Carrera, Angeli, Davison, ICRA 2011. PDFs in `papers/` (local only).

## Data

CMU Panoptic, sequence `171026_pose1`, 12 HD cameras:
`00_00 01 02 03 04 07 13 18 21 24 28 29` (a connected arc of the 31-camera dome ring).
Only the first 150 MB of each video is downloaded (≈55 s; the MP4 index is at the start, so partial files decode).
Calibration JSON: `x_cam = R·X_world + t`, **t in cm**, world **y-axis points down**, camera centre = `−Rᵀt`.
`data/` is git-ignored (~2.5 GB with cache); recreate with `python3 scripts/download_panoptic.py`.

## Commands (run from this folder)

```bash
pip install -r requirements.txt            # opencv-python>=4.5, numpy, scipy, matplotlib
python3 scripts/download_panoptic.py       # data (~2 min)
python3 scripts/run_pipeline.py            # full 12-camera run; first run builds cache (~4 min), then ~55 s
python3 scripts/run_pipeline.py --frames 10
python3 scripts/run_two_view.py            # single-frame 2-camera demo (5 cameras)
python3 scripts/run_online.py --part A     # pair convergence plot (~30 s)
python3 scripts/run_online.py --part B     # n-camera vs #frames (~8 min); --part plotB redraws from CSV
python3 scripts/make_figures.py            # graph / 3D / per-camera / matches figures
```

Outputs go to `results/` (figures + CSV/JSON, committed).
If SIFT/matching settings change, delete `data/171026_pose1/cache/` — the cache does not detect setting changes.

## Current results (all 53 frames)

- 12/12 cameras registered; mean rotation error **0.050°** with BA (0.178° without);
  camera position error **0.56 cm** with BA (0.37 cm without); median reprojection 0.33 px.
- Graph check: 66 pairs, 37 truly wrong; 54 pass gates; 30 kept (1 wrong, 5.3°), 24 rejected (all truly wrong).
- On the fly: 0 cameras at 1–3 frames, 3 at 5, 5 at 8, **all 12 by 13 frames**.

## Lessons learned — don't undo these without re-testing

- **One frame is not enough** for wide-baseline pairs (plain, repetitive dome walls). Pooling frames is the core idea.
- **Plain pooling beats deduplicating** repeated static-background matches.
- **RANSAC alone jumps** between updates; refinement + "keep the better estimate" is what makes it stable.
- **Inlier count/ratio cannot filter wrong pairs** (repeated texture gives confident wrong E). The triangle/loop rotation check does.
- **Bundle adjustment with a Huber loss from the start barely converged** (32 → 23 px from a perturbed start). Use plain least squares, drop residuals > 3 px, solve again (32 → 0.7 px).
- BA improves rotation clearly; sub-cm position differences are likely within ground-truth accuracy — don't claim BA improves position.
- Cameras facing each other across the dome can't be paired directly; the camera set must form a connected chain of neighbours.

## Known limitations

One sequence only; known intrinsics assumed; needs a moving scene; not real-time
(the n-camera step re-runs from scratch; truly live would need incremental BA).

## Working with teammates

- Teammates may be new to multi-view geometry. Explain plainly and briefly, with
  the one-line idea first, then the code location.
- AI use is allowed in this course, but everyone must understand and be able to
  present the method. Prefer explaining over just changing code.
- Commit on a branch and open a PR; the repo (github.com/dbcometto/multi_camera_calibration) is public and owned by Dante.
- Never commit `data/`, paper PDFs, or `__pycache__/` (already in `.gitignore`).
