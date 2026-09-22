"""Debug pipeline: output every stage's result per frame to JSON.

Runs two-stage detect + pose + stroke on a short clip, saves:
  - per-frame: all detected persons (bbox, on_court)
  - per-frame: pose results (n_keypoints visible)
  - per-frame: stroke RNN output (all 4 class probabilities)

Usage:
    python scripts/test_debug_pipeline.py --video /tmp/sample_short_20s.mp4
"""

import argparse
import json
import os
import sys
import time
import warnings
import logging
from collections import deque

import cv2
import numpy as np

warnings.filterwarnings("ignore", message=".*GMC failed.*")
logging.getLogger("ultralytics").setLevel(logging.ERROR)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO

# ---- CONFIG ----
DETECT_WEIGHTS = "yolo11n.pt"
POSE_WEIGHTS = "weights/yolo26s-pose.pt"
RNN_WEIGHTS = "weights/tennis_rnn.h5"
DETECT_CONF = 0.3
DETECT_IMGSZ = 1280
POSE_CONF = 0.15
POSE_IMGSZ = 640
CROP_PAD = 0.15

WINDOW = 30
STRIDE = 5
LABELS = ("backhand", "forehand", "neutral", "serve")
_KEEP_IDX = np.array([0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])

# Calib (sample_short.mp4)
COURT_W_M, COURT_L_M = 10.97, 23.77
H_IMG_TO_REAL = np.array([
    [-0.004212704126665825, -0.00958000348568168, 8.066176934954633],
    [9.370094579164494e-05, 0.01803202259785551, -15.976691289361806],
    [1.1053780957617317e-07, -0.0019774218879326623, 1.0],
], dtype=np.float64)
COURT_MARGIN = 3.0


def on_court(bbox):
    x0, y0, x1, y1 = bbox
    foot_x, foot_y = 0.5 * (x0 + x1), y1
    p = H_IMG_TO_REAL @ [foot_x, foot_y, 1.0]
    rx, ry = p[0] / p[2], p[1] / p[2]
    return (-COURT_MARGIN <= rx <= COURT_W_M + COURT_MARGIN
            and -COURT_MARGIN <= ry <= COURT_L_M + COURT_MARGIN)


def kp_to_feature(kp_xy, bbox):
    x0, y0, x1, y1 = bbox
    bw, bh = max(x1 - x0, 1), max(y1 - y0, 1)
    kp = kp_xy[_KEEP_IDX]
    yx = np.stack([(kp[:, 1] - y0) / bh, (kp[:, 0] - x0) / bw], axis=1)
    return yx.reshape(-1).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", default="/tmp/debug_pipeline.json")
    args = parser.parse_args()

    det_model = YOLO(DETECT_WEIGHTS)
    pose_model = YOLO(POSE_WEIGHTS)

    try:
        import tf_keras as keras
    except ImportError:
        from tensorflow import keras
    rnn = keras.models.load_model(RNN_WEIGHTS, compile=False)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {W}x{H} fps={fps:.1f} frames={total}")

    track_windows = {}
    track_last_infer = {}
    frame_log = []

    fi = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_data = {"frame": fi, "time": round(fi / fps, 2), "detections": [], "strokes": []}

        # Stage 1: detect
        dr = det_model.track(frame, persist=True, verbose=False,
                             classes=[0], conf=DETECT_CONF, imgsz=DETECT_IMGSZ)[0]

        if dr.boxes is not None and dr.boxes.id is not None:
            det_boxes = dr.boxes.xyxy.cpu().numpy()
            det_ids = dr.boxes.id.cpu().numpy().astype(int)
            det_confs = dr.boxes.conf.cpu().numpy()

            for i in range(len(det_boxes)):
                x0, y0, x1, y1 = det_boxes[i].astype(int)
                tid = int(det_ids[i])
                dc = float(det_confs[i])
                court = on_court((x0, y0, x1, y1))

                det_info = {
                    "player": tid,
                    "bbox": [int(x0), int(y0), int(x1), int(y1)],
                    "det_conf": round(dc, 3),
                    "on_court": bool(court),
                    "foot_x": round((x0 + x1) / 2.0, 1),
                    "foot_y": int(y1),
                    "bbox_h": int(y1 - y0),
                }

                # Stage 2: pose (only on-court)
                if court:
                    bw, bh = x1 - x0, y1 - y0
                    pad = int(CROP_PAD * max(bw, bh))
                    cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
                    cx1, cy1 = min(W, x1 + pad), min(H, y1 + pad)
                    crop = frame[cy0:cy1, cx0:cx1]

                    pr = pose_model.predict(crop, verbose=False, classes=[0],
                                            conf=POSE_CONF, imgsz=POSE_IMGSZ)[0]

                    if pr.boxes is not None and len(pr.boxes) > 0 and pr.keypoints is not None:
                        best = int(pr.boxes.conf.cpu().numpy().argmax())
                        kp_crop = pr.keypoints.data[best].cpu().numpy()
                        pc = float(pr.boxes.conf[best])
                        kp_frame = kp_crop.copy()
                        kp_frame[:, 0] += cx0
                        kp_frame[:, 1] += cy0
                        n_visible = int((kp_crop[:, 2] >= 0.3).sum())

                        det_info["pose_conf"] = round(pc, 3)
                        det_info["visible_kp"] = n_visible

                        # Stage 3: stroke RNN
                        bbox = (x0, y0, x1, y1)
                        if tid not in track_windows:
                            track_windows[tid] = deque(maxlen=WINDOW)
                            track_last_infer[tid] = -999
                        track_windows[tid].append(kp_to_feature(kp_frame, bbox))

                        if (len(track_windows[tid]) >= WINDOW
                                and fi - track_last_infer[tid] >= STRIDE):
                            track_last_infer[tid] = fi
                            feats = np.stack(list(track_windows[tid]), axis=0)
                            probs = rnn.predict(feats[None, ...], verbose=0)[0]
                            probs_dict = {LABELS[j]: round(float(probs[j]), 3) for j in range(4)}
                            k = int(np.argmax(probs))
                            det_info["rnn_probs"] = probs_dict
                            det_info["rnn_label"] = LABELS[k]
                            det_info["rnn_conf"] = round(float(probs[k]), 3)

                            frame_data["strokes"].append({
                                "player": tid,
                                "label": LABELS[k],
                                "conf": round(float(probs[k]), 3),
                                "probs": probs_dict,
                                "foot_x": det_info["foot_x"],
                                "foot_y": det_info["foot_y"],
                            })
                    else:
                        det_info["pose_conf"] = None
                        det_info["visible_kp"] = 0

                frame_data["detections"].append(det_info)

        frame_log.append(frame_data)
        fi += 1
        if fi % 100 == 0:
            print(f"  frame {fi}/{total} {fi/(time.time()-t0):.1f} fps")

    cap.release()
    print(f"Done: {fi} frames in {time.time()-t0:.1f}s")

    with open(args.out, "w") as f:
        json.dump(frame_log, f, indent=2)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
