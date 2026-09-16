"""Extract ball positions from video using WASB detector + Kalman tracker.

Outputs a JSON with:
  - detected: per-frame ball positions from validated tracks (actual detections)
  - predicted: per-frame Kalman-predicted positions (fills gaps between detections)

Usage (remote GPU):
    python -u scripts/extract_ball_positions.py \
        --video /root/personal/samples/sample_short.mp4 \
        --config configs/remote_inpaint.yaml \
        --device cuda \
        --out /root/personal/results/ball_positions.json

Usage (Kaggle):
    Paste into a notebook cell. Upload wasb_tennis_best.pth.tar to dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tennisvision.config import load_config
from tennisvision.ball.wasb import WASBBallDetector, WASBConfig
from tennisvision.ball.tracker import MultiTrackManager, TrackerConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    bcfg = cfg["ball"]
    tcfg = cfg["tracker"]
    device = args.device or bcfg.get("device", "cpu")

    det = WASBBallDetector(WASBConfig(
        weights=bcfg["weights"],
        device=device,
        runtime=bcfg.get("runtime", "auto"),
        score_threshold=bcfg.get("score_threshold", 0.5),
    ))

    tracker = MultiTrackManager(TrackerConfig(
        gate_px=tcfg.get("gate_px", 120),
        max_gap_frames=tcfg.get("max_gap_frames", 8),
        min_len=tcfg.get("min_len", 3),
        min_speed=tcfg.get("min_speed", 5.0),
        max_speed=tcfg.get("max_speed", 200.0),
    ))

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {W}x{H} fps={fps:.1f} frames={total}")
    print(f"Device: {device}")

    # Per-frame: champion detected position + predicted position
    detected = {}     # frame -> [x, y]  (actual detection)
    predicted = {}    # frame -> [x, y]  (Kalman prediction, every frame with champion)
    fi = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        det.push_frame(frame)
        cand = det.detect()

        cand_xys = []
        if cand is not None:
            if isinstance(cand, list):
                cand_xys = [(int(c[0]), int(c[1])) for c in cand]
            elif isinstance(cand, tuple) and len(cand) == 2:
                cand_xys = [(int(cand[0]), int(cand[1]))]

        tracker.update(cand_xys, fi)

        # Get champion track's position
        champ = tracker.champion(fi)
        if champ is not None:
            # Predicted position (Kalman, available every frame)
            px, py = champ.pred_xy
            predicted[fi] = [float(px), float(py)]

            # Detected position (only if champion was actually detected this frame)
            if champ.last_det_frame == fi and champ.pts:
                pt = champ.pts[-1]
                detected[fi] = [float(pt.x), float(pt.y)]

        fi += 1
        if fi % 500 == 0:
            elapsed = time.time() - t0
            eta = (total - fi) / (fi / elapsed)
            print(f"  frame {fi}/{total}  {fi/elapsed:.1f} fps  eta {eta:.0f}s")

    cap.release()
    elapsed = time.time() - t0
    print(f"\nDone: {fi} frames in {elapsed:.1f}s ({fi/elapsed:.1f} fps)")
    print(f"Detected positions: {len(detected)} frames")
    print(f"Predicted positions: {len(predicted)} frames (includes Kalman fill)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    output = {
        "video": os.path.basename(args.video),
        "fps": fps,
        "total_frames": total,
        "width": W,
        "height": H,
        "n_detected": len(detected),
        "n_predicted": len(predicted),
        "detected": {str(k): v for k, v in sorted(detected.items())},
        "predicted": {str(k): v for k, v in sorted(predicted.items())},
    }
    with open(args.out, "w") as f:
        json.dump(output, f)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
