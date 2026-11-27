# Vinh's Approach — Targetless Extrinsic Calibration (Project 3)

EECE 7150 AFR Group 1 — Project 3: *Calibration on the fly (extrinsics), Camera–Camera → nCamera*
Presentation: **12/02/2026**

## Purpose

Find where every camera is (rotation **R**, translation **t**) relative to the
others **without a calibration target** (no checkerboard), using only what the
cameras see, and keep improving the answer as new frames arrive ("on the fly").
Start with 2 cameras, then extend to n cameras.

## Results at a glance

CMU Panoptic `171026_pose1`, **12 HD cameras** (1920×1080), 1 frame per second,
compared with the dataset's ground-truth calibration.

| What | Result |
|---|---|
| Cameras calibrated | **12 / 12** (all of them) |
| Mean rotation error | **0.050°** (with bundle adjustment), 0.178° without |
| Mean camera position error | **0.56 cm** with BA, 0.37 cm without (cameras are 1.2–5.4 m apart) |
| Reprojection error after BA | median **0.33 px** |
| Frames needed | 5 s of video → first 3 cameras; **13 s → all 12 cameras** |
| Metric information used | **one** known distance between two cameras (sets the scale) |
| Runtime | ~4 min once (SIFT + matching, cached), then ~55 s per full run on a laptop CPU |

Two-camera, one frame only vs. pooled over time (rotation error vs. ground truth):

| Camera pair | 1 frame | all 53 frames |
|---|---|---|
| 00_01 – 00_04 | 0.13° | **0.01°** |
| 00_00 – 00_29 | 0.07° | **0.01°** |
| 00_01 – 00_03 | 1.18° | **0.06°** |
| 00_02 – 00_03 (hard) | 93° (wrong) | **0.31°** (correct from frame 16 on) |

## Pipeline

```
 CMU Panoptic: synced videos + K, distortion (inputs) + R, t (ground truth, evaluation only)
        │
 1. Load frames (1 per second), undistort keypoints with K + distortion
 2. SIFT (contrast threshold 0.01) + FLANN, Lowe ratio test 0.75 + mutual check
 3. Two-view pose: 5-point essential matrix + RANSAC (Nistér), recoverPose
        → non-linear refinement of R, t on inliers (Sampson error)
 4. ON THE FLY: every new frame adds matches to the pool → re-estimate;
        keep the previous estimate if it still explains the pool better
 5. Camera graph: inlier gates → triangle check → max spanning tree → loop check
        (removes wrong camera pairs without using ground truth)
 6. Multi-camera tracks (union-find, per frame) → seed pair (scale = 1 baseline)
        → add cameras one by one with PnP + RANSAC → triangulate
        → bundle adjustment after each camera and at the end
 7. Evaluation against ground truth: rotation (°), position (cm), reprojection (px)
```

### Why each step is there (what we found)

1. **One frame is not enough.** The dome walls are plain and repetitive and the
   cameras are far apart, so one frame gives only ~10–70 good matches per pair.
   Only cameras next to each other work from a single frame
   (`results/two_view/`).
2. **Pooling frames over time fixes it.** The person moves, so every frame adds
   new 3D points. Plain pooling beat removing repeated background matches.
3. **Refinement + "keep the better estimate" make it stable.** RANSAC alone gave
   answers that jumped by up to 100° between updates; with these two steps the
   curves in `online_pairs.png` settle smoothly.
4. **Wrong camera pairs must be removed.** 37 of the 66 pairs are wrong
   (> 5° vs ground truth), mostly cameras that face each other and see different
   walls. Some wrong pairs still have many inliers (repeated wall texture), so an
   inlier count alone can't filter them. The **triangle/loop rotation check**
   (pair rotations must agree around triangles of cameras) does:
   - 54 pairs pass the inlier gates (25 of them wrong)
   - **30 kept (1 wrong, only 5.3° off), 24 rejected (all 24 truly wrong, 0 good ones lost)**
5. **Bundle adjustment** lowers rotation error ~3.5× (0.18° → 0.05°).
   Position error stays below 1 cm either way; the differences (0.37 vs 0.56 cm)
   are likely at the accuracy level of the ground truth itself, so we do not
   claim BA improves position.
   *Bug found on the way:* a Huber loss from the start barely moved the
   solution (32 → 23 px). Plain least squares, then dropping observations > 3 px
   and solving again, converges properly (32 → 0.7 px from a perturbed start).
6. **Scale.** Images alone can't tell size, so one known camera-to-camera
   distance is used (the seed pair). Everything else is in cm automatically
   through PnP.

## Figures (in `results/`)

| File | Shows |
|---|---|
| `matches_00_00_00_29.jpg` | verified SIFT matches between two neighbouring cameras (one frame) |
| `online_pairs.png` | **two-camera on the fly**: error vs frames seen for 4 pairs |
| `online_ncam.png` | **n cameras on the fly**: cameras calibrated and error vs frames seen |
| `camera_graph.png` | top view of the 12 cameras, kept pairs (blue) vs rejected (red dashed) |
| `cameras_3d.png` | estimated cameras inside the ground-truth circles + 37k triangulated points (dome walls + person) |
| `per_camera_error.png` | rotation / position error per camera, with vs without bundle adjustment |
| `two_view/` | single-frame demo for 5 cameras (shows why one frame isn't enough) |
| `pipeline_result.json`, `*.csv` | the numbers behind everything |

## How to run

From the `vinh_approach/` folder (Python 3, see `requirements.txt`):

```bash
# 0. Data: 12 cameras, first ~55 s each, + ground truth (~2 GB, ~2 min)
python3 scripts/download_panoptic.py          # default = the 12 cameras used here

# 1-3. Single-frame two-camera demo (5 cameras, ~3 s)
python3 scripts/run_two_view.py

# Full n-camera pipeline (first run also builds the SIFT/match cache, ~4 min)
python3 scripts/run_pipeline.py
python3 scripts/run_pipeline.py --frames 10     # use only the first 10 frames

# On-the-fly experiments + figures
python3 scripts/run_online.py --part A          # pairs, ~30 s
python3 scripts/run_online.py --part B          # all cameras vs #frames, ~8 min
python3 scripts/make_figures.py                 # graph, 3D, per-camera, matches
```

## Code map

| Step | File | Main tools |
|---|---|---|
| 0 | `scripts/download_panoptic.py` | partial HTTP download, frame extraction |
| 1 | `src/data_loader.py` | calibration JSON, `cv2.undistortPoints` |
| 2 | `src/features.py`, `src/cache.py` | `cv2.SIFT_create`, FLANN, ratio + mutual check, multiprocessing cache |
| 3 | `src/two_view.py` | `cv2.findEssentialMat` (5-point RANSAC), `cv2.recoverPose`, Sampson refinement |
| 4 | `src/online.py` | `OnlinePair`: add frame → update → keep better |
| 5 | `src/pose_graph.py` | inlier gates, triangle check, spanning tree, loop check |
| 6 | `src/sfm.py`, `src/bundle_adjustment.py` | union-find tracks, DLT triangulation, `cv2.solvePnPRansac`, `scipy.optimize.least_squares` |
| 7 | `src/metrics.py` | rotation / position / translation-direction errors |
| plots | `src/plots.py`, `scripts/make_figures.py`, `scripts/run_online.py` | matplotlib |

## Limitations / honest notes

- **Ground truth is used only for evaluation** — plus one baseline length for
  scale. Intrinsics (K, distortion) are assumed known, as the project allows.
- Cameras with no overlapping view of anything (facing each other across the
  dome) can't be paired directly; they are connected through neighbours.
  This is why we picked 12 cameras forming a connected ring.
- Needs a few seconds of a moving scene; a fully static scene would give the
  same matches every frame and pooling would add little.
- One sequence only; the Panoptic ground truth is itself an estimate, so
  sub-centimetre differences are not meaningful.
- Not real-time as written (full re-run per update); the online pair update is
  cheap, the n-camera step would need incremental BA to be truly live.

## Folder layout

```
vinh_approach/
│
├── README.md            📖 for humans: full explanation (this file)
├── CLAUDE.md            🤖 for Claude Code: teammates' Claude gets the project instantly
├── requirements.txt     libraries to install
├── .gitignore           keeps data, PDFs and cache files off GitHub
│
├── papers/              📄 reading (PDFs local only)
│   ├── Nister2003_five_point_relative_pose.pdf       main method
│   ├── Carrera2011_SLAM_multicamera_calibration.pdf  related work
│   └── README.md
│
├── data/                💾 dataset (local only, 2.5 GB)
│   ├── README.md
│   └── 171026_pose1/    calibration JSON + videos/ + frames/ + cache/
│
├── src/                 🧠 HOW it works (one file per step)
│   ├── data_loader.py        1  load frames + calibration
│   ├── features.py           2  SIFT + matching
│   ├── cache.py              2  save features, reuse
│   ├── two_view.py           3  pose between 2 cameras
│   ├── online.py             4  on the fly
│   ├── pose_graph.py         5  remove wrong pairs
│   ├── sfm.py                6  join all cameras
│   ├── bundle_adjustment.py  6  fine-tune
│   ├── metrics.py            7  compare with ground truth
│   └── plots.py                 figure style
│
├── scripts/             ▶️ WHAT you run
│   ├── download_panoptic.py  get the data
│   ├── run_two_view.py       2-camera, single-frame demo
│   ├── run_pipeline.py       full 12-camera calibration
│   ├── run_online.py         accuracy vs. frames experiments
│   └── make_figures.py       figures for the slides
│
└── results/             📊 WHAT you show
    ├── online_pairs.png        on the fly, 2 cameras
    ├── online_ncam.png         on the fly, 12 cameras
    ├── camera_graph.png        kept vs. rejected pairs
    ├── cameras_3d.png          estimate vs. ground truth in 3D
    ├── per_camera_error.png    error per camera
    ├── matches_00_00_00_29.jpg matched points example
    ├── *.csv, pipeline_result.json   the numbers
    └── two_view/               single-frame demo (10 images + CSV)
```

## References

Main method and comparison (PDFs in `papers/`):
- D. Nistér, "An Efficient Solution to the Five-Point Relative Pose Problem", CVPR 2003 / IEEE TPAMI 2004 — two-view pose (step 3).
- G. Carrera, A. Angeli, A. J. Davison, "SLAM-Based Automatic Extrinsic Calibration of a Multi-Camera Rig", ICRA 2011 — related work: on-the-fly multi-camera calibration via SLAM.

Tools used:
- D. Lowe, "Distinctive Image Features from Scale-Invariant Keypoints" (SIFT), IJCV 2004.
- M. Fischler, R. Bolles, "Random Sample Consensus" (RANSAC), 1981.
- B. Triggs et al., "Bundle Adjustment — A Modern Synthesis", 2000.
- R. Hartley, A. Zisserman, *Multiple View Geometry in Computer Vision*.
- H. Joo et al., "Panoptic Studio", ICCV 2015 — dataset.
