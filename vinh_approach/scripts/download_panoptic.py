"""Download a small subset of CMU Panoptic HD data for extrinsic calibration.

Only the first part of each HD video is downloaded (the full files are ~2.3 GB
each). Panoptic MP4s store their index ("moov") at the start, so a partial
download still decodes from frame 0.

Then frames are extracted every `--step` frames to JPG, so the rest of the
pipeline works on images.

Usage (from vinh_approach/):
    python3 scripts/download_panoptic.py
    python3 scripts/download_panoptic.py --seq 171026_pose1 --cams 0 1 2 --mb 150 --step 30
"""
import argparse
import os
import urllib.request

import cv2

BASE = "http://domedb.perception.cs.cmu.edu/webdata/dataset"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")


def download(url, out_path, max_bytes=None):
    if os.path.exists(out_path):
        print(f"  exists, skip: {out_path}")
        return
    req = urllib.request.Request(url)
    if max_bytes:
        req.add_header("Range", f"bytes=0-{max_bytes - 1}")
    tmp = out_path + ".part"
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        got = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            print(f"\r  {got / 1e6:7.1f} MB", end="", flush=True)
    print()
    os.replace(tmp, out_path)


def extract_frames(video_path, out_dir, step):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    idx = saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:  # end of the partial file
            break
        if idx % step == 0:
            cv2.imwrite(os.path.join(out_dir, f"{idx:06d}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1
        idx += 1
    cap.release()
    return idx, saved


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq", default="171026_pose1")
    p.add_argument("--cams", type=int, nargs="+", default=[0, 1, 2, 3, 4, 7, 13, 18, 21, 24, 28, 29],
                   help="HD camera indices (0-30)")
    p.add_argument("--mb", type=int, default=150,
                   help="MB to download per camera (~30 MB = 11 s of video)")
    p.add_argument("--step", type=int, default=30,
                   help="keep every Nth frame (30 = 1 frame per second)")
    args = p.parse_args()

    seq_dir = os.path.join(DATA_DIR, args.seq)
    os.makedirs(os.path.join(seq_dir, "videos"), exist_ok=True)

    print("Calibration (ground truth):")
    download(f"{BASE}/{args.seq}/calibration_{args.seq}.json",
             os.path.join(seq_dir, f"calibration_{args.seq}.json"))

    frame_counts = []
    for c in args.cams:
        name = f"hd_00_{c:02d}"
        print(f"Camera {name}:")
        video = os.path.join(seq_dir, "videos", f"{name}.mp4")
        download(f"{BASE}/{args.seq}/videos/hd_shared_crf20/{name}.mp4",
                 video, max_bytes=args.mb * 1024 * 1024)
        n, saved = extract_frames(video, os.path.join(seq_dir, "frames", name),
                                  args.step)
        frame_counts.append(n)
        print(f"  decoded {n} frames, saved {saved} JPGs")

    # Cameras are synchronized by frame index; only frames that exist in
    # every camera can be used together.
    print(f"Common frame range: 0..{min(frame_counts) - 1}")


if __name__ == "__main__":
    main()
