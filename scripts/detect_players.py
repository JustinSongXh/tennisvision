#!/usr/bin/env python3
"""Detect and track players per frame using YOLO.

Outputs bbox + foot position per detection, no pose/keypoints.
Fast pass — runs on every frame to provide input for scene readiness
and slot mapping in downstream steps.

Usage:
    python -u scripts/detect_players.py \
        --video samples/sample3.mp4 \
        --out results/sample3/player_detections.json

    # Limit frames
    python -u scripts/detect_players.py \
        --video samples/sample3.mp4 \
        --out results/sample3/player_detections.json \
        --max-frames 1800
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-frames", type=int, default=0, help="Limit frames (0=all)")
    parser.add_argument("--no-half", action="store_true", help="Disable FP16")
    args = parser.parse_args()

    from ultralytics import YOLO

    device = 0 if torch.cuda.is_available() else "cpu"
    quantize = "fp16" if (torch.cuda.is_available() and not args.no_half) else None

    print(f"Device: {device} quantize={quantize}")
    det_model = YOLO("yolo11m.pt")

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    TOTAL = min(total, args.max_frames) if args.max_frames > 0 else total
    print(f"Video: {W}x{H} fps={fps:.1f} processing {TOTAL} frames")

    frame_data = []
    fi, t0 = 0, time.time()

    while fi < TOTAL:
        ret, frame = cap.read()
        if not ret:
            break
        detections = []
        dr = det_model.track(frame, persist=True, verbose=False, classes=[0],
                             conf=0.3, imgsz=1280, device=device, quantize=quantize)[0]

        if dr.boxes is not None and dr.boxes.id is not None:
            boxes = dr.boxes.xyxy.cpu().numpy().astype(int)
            ids = dr.boxes.id.cpu().numpy().astype(int)
            confs = dr.boxes.conf.cpu().numpy()

            for i in range(len(boxes)):
                x0, y0, x1, y1 = boxes[i]
                detections.append({
                    "tid": int(ids[i]),
                    "det_conf": round(float(confs[i]), 3),
                    "bbox": [int(x0), int(y0), int(x1), int(y1)],
                    "foot_x": round(float((x0 + x1) / 2), 1),
                    "foot_y": int(y1),
                })

        frame_data.append({"frame": fi, "detections": detections})
        fi += 1
        if fi % 500 == 0:
            e = time.time() - t0
            print(f"  frame {fi}/{TOTAL}  {fi/e:.1f} fps  eta {(TOTAL-fi)/(fi/e):.0f}s")

    cap.release()
    elapsed = time.time() - t0
    print(f"\nDone: {fi} frames in {elapsed:.0f}s ({fi/max(elapsed,0.1):.1f} fps)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"fps": fps, "width": W, "height": H,
                    "total_frames": fi, "frames": frame_data}, f)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
