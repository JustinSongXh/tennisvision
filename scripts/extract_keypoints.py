#!/usr/bin/env python3
"""Extract player keypoints from video using two-stage YOLO pipeline.

Stage 1: yolo11m full-frame person detection + tracking
Stage 2: yolo26s-pose on each bbox crop for keypoints

Outputs keypoints.json with per-frame detections including normalized keypoints.

Usage:
    python -u scripts/extract_keypoints.py \
        --video samples/sample_short.mp4 \
        --out results/sample_short/keypoints.json

    # Limit frames
    python -u scripts/extract_keypoints.py \
        --video samples/sample_short.mp4 \
        --out results/sample_short/keypoints.json \
        --max-frames 1800

    # Disable FP16
    python -u scripts/extract_keypoints.py \
        --video samples/sample_short.mp4 \
        --out results/sample_short/keypoints.json \
        --no-half
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
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

    # Hardware auto-detection
    if torch.cuda.device_count() >= 2:
        det_device, pose_device = 0, 1
    elif torch.cuda.is_available():
        det_device, pose_device = 0, 0
    else:
        det_device, pose_device = "cpu", "cpu"
    use_half = torch.cuda.is_available() and not args.no_half

    print(f"Device: det={det_device} pose={pose_device} half={use_half}")
    det_model = YOLO("yolo11m.pt")
    pose_model = YOLO("yolo26s-pose.pt")
    CROP_PAD = 0.15

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
                             conf=0.3, imgsz=1280, device=det_device, half=use_half)[0]

        if dr.boxes is not None and dr.boxes.id is not None:
            boxes = dr.boxes.xyxy.cpu().numpy().astype(int)
            ids = dr.boxes.id.cpu().numpy().astype(int)
            confs = dr.boxes.conf.cpu().numpy()

            crops, offsets, kept = [], [], []
            for i in range(len(boxes)):
                x0, y0, x1, y1 = boxes[i]
                pad = int(CROP_PAD * max(x1 - x0, y1 - y0))
                cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
                cx1, cy1 = min(W, x1 + pad), min(H, y1 + pad)
                crops.append(frame[cy0:cy1, cx0:cx1])
                offsets.append((cx0, cy0))
                kept.append(i)

            prs = pose_model.predict(crops, verbose=False, classes=[0],
                                     conf=0.15, imgsz=640, device=pose_device,
                                     half=use_half) if crops else []

            for j, i in enumerate(kept):
                if j >= len(prs):
                    continue
                pr = prs[j]
                if pr.boxes is None or len(pr.boxes) == 0 or pr.keypoints is None:
                    continue
                best = int(pr.boxes.conf.cpu().numpy().argmax())
                kp = pr.keypoints.data[best].cpu().numpy()
                kp_x = kp[:, 0] + offsets[j][0]
                kp_y = kp[:, 1] + offsets[j][1]
                kp_v = kp[:, 2]
                valid = kp_v >= 0.3
                if valid.sum() < 3:
                    continue
                xn, xx = kp_x[valid].min(), kp_x[valid].max()
                yn, yx = kp_y[valid].min(), kp_y[valid].max()
                bw, bh = max(xx - xn, 1), max(yx - yn, 1)
                detections.append({
                    "tid": int(ids[i]), "det_conf": round(float(confs[i]), 3),
                    "bbox": [int(boxes[i][0]), int(boxes[i][1]),
                             int(boxes[i][2]), int(boxes[i][3])],
                    "foot_x": round(float((boxes[i][0] + boxes[i][2]) / 2), 1),
                    "foot_y": int(boxes[i][3]),
                    "kp_x": [round(float(v), 4) for v in kp_x],
                    "kp_y": [round(float(v), 4) for v in kp_y],
                    "kp_v": [round(float(v), 3) for v in kp_v],
                    "kp_norm_x": [round(float((kp_x[k] - xn) / bw), 4) for k in range(17)],
                    "kp_norm_y": [round(float((kp_y[k] - yn) / bh), 4) for k in range(17)],
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
