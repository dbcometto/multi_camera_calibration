# data/ — CMU Panoptic subset

Everything in this folder except this README is ignored by git (too large).
Anyone can recreate it with one command (from `vinh_approach/`):

```bash
python3 scripts/download_panoptic.py      # default = these 12 cameras
```

## What gets downloaded

| Item | Details |
|---|---|
| Sequence | `171026_pose1` (one person doing poses in the Panoptic dome) |
| Cameras | 12 HD cameras: `00_00 01 02 03 04 07 13 18 21 24 28 29` (1920×1080, 29.97 fps) |
| Video | First **150 MB** of each camera's video ≈ 55 s. The full file is ~2.3 GB per camera, but the MP4 index is at the start, so a partial download still plays. |
| Frames | Every 30th frame (1 per second) saved as JPG → 53 frames common to all cameras |
| Ground truth | `calibration_171026_pose1.json`: K, distortion, R, t for all 31 HD cameras |
| Total size | ~2.5 GB including the ~0.4 GB feature cache |

Why these 12: the 31 HD cameras sit on a ring around the dome. Cameras facing
each other see different walls and can't be matched, so we picked an arc of
neighbouring cameras that is connected all the way round
(00 → 29/24/21 → 28/13 → 02 → 07 → 04/01 → 18 → 03).

## Folder layout after download

```
data/171026_pose1/
├── calibration_171026_pose1.json     ground truth for all 31 HD cameras
├── videos/hd_00_XX.mp4               partial videos
├── frames/hd_00_XX/000000.jpg …      frame index in the filename
└── cache/                            SIFT features + matches (built by the pipeline)
```

## Important facts about the data

- **Synchronized:** HD cameras are hardware-synced, so frame `000600.jpg` in
  every camera folder was taken at the same moment.
- **Calibration format:** each camera has `K` (3×3), `distCoef` (5 values),
  `R` (3×3) and `t` (3×1, **centimetres**). They map world → camera:
  `x_cam = R · X_world + t`. Camera centre in the world = `−Rᵀ t`.
  The world y-axis points **down**.
- **How we use the ground truth:** only `K` and `distCoef` are inputs.
  `R` and `t` are used **only to check** our answer (plus one baseline length
  to fix the scale).
