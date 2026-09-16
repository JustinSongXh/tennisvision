"""Kaggle GPU test: two-stage detection + pose + stroke classification.

Stage 1: yolo11n full-frame detection (finds all players including far-side)
       + on-court homography filter + max 4 players
Stage 2: yolo26s-pose on each bbox crop (extracts keypoints)
Stage 3: GRU stroke classifier on 30-frame sliding window per track
"""

import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "tf-keras"])

import cv2
import time
import warnings
import logging
from collections import deque

import numpy as np

warnings.filterwarnings("ignore", message=".*GMC failed.*")
logging.getLogger("ultralytics").setLevel(logging.ERROR)

from ultralytics import YOLO
import tf_keras as keras  # Keras 2 compat — tennis_rnn.h5 uses time_major kwarg

# ---- CONFIG ----
INPUT_VIDEO = "/kaggle/input/datasets/pinesquirrel/tennisvision-test/sample_short.mp4"
OUTPUT_VIDEO = "/kaggle/working/sample_short_pose_detect.mp4"
DETECT_WEIGHTS = "yolo11n.pt"                 # Stage 1: person detection (nano, fast)
POSE_WEIGHTS = "yolo26s-pose.pt"              # Stage 2: pose on crop (small, fast)
RNN_WEIGHTS = "/kaggle/input/models/pinesquirrel/tennis-rnn/keras/default/1/tennis_rnn.h5"

DETECT_DEVICE = "cuda:0"
POSE_DEVICE = "cuda:1"
DETECT_CONF = 0.3
DETECT_IMGSZ = 1280
POSE_CONF = 0.15
POSE_IMGSZ = 640
CROP_PAD = 0.15           # bbox padding ratio before pose crop
COURT_MARGIN_M = 3.0      # on-court filter tolerance in metres
MAX_PERSONS = 4           # doubles: keep top 4 by confidence

# ---- Court calibration (sample_short.mp4) ----
COURT_W_M, COURT_L_M = 10.97, 23.77
H_IMG_TO_REAL = np.array([
    [-0.004212704126665825, -0.00958000348568168,   8.066176934954633],
    [ 9.370094579164494e-05, 0.01803202259785551, -15.976691289361806],
    [ 1.1053780957617317e-07,-0.0019774218879326623, 1.0],
], dtype=np.float64)

def on_court(bbox):
    """Check if bbox foot point projects inside court + margin."""
    x0, y0, x1, y1 = bbox
    foot_x, foot_y = 0.5 * (x0 + x1), y1
    p = H_IMG_TO_REAL @ [foot_x, foot_y, 1.0]
    rx, ry = p[0] / p[2], p[1] / p[2]
    m = COURT_MARGIN_M
    return (-m <= rx <= COURT_W_M + m and -m <= ry <= COURT_L_M + m)

# Write custom ByteTrack config for stable IDs
import tempfile, os, yaml
_TRACKER_CFG = {
    "tracker_type": "bytetrack",
    "track_high_thresh": 0.3,     # default 0.5 — lower to keep far-side in high-conf pool
    "track_low_thresh": 0.05,     # default 0.1
    "new_track_thresh": 0.4,      # default 0.6 — harder to create new tracks
    "track_buffer": 90,           # default 30 — keep lost tracks for 3s @ 30fps
    "match_thresh": 0.85,         # default 0.8
    "fuse_score": True,
}
_TRACKER_PATH = os.path.join(tempfile.gettempdir(), "bytetrack_stable.yaml")
with open(_TRACKER_PATH, "w") as _f:
    yaml.dump(_TRACKER_CFG, _f)
print(f"Tracker config: {_TRACKER_PATH}")

# Stroke classifier settings
WINDOW = 30               # frames for sliding window
STRIDE = 5                # run RNN every N frames per track
MIN_STROKE_CONF = 0.9
MERGE_GAP = 30            # same player+label within N frames → merge into one event
LABELS = ("backhand", "forehand", "neutral", "serve")

# Drop eyes+ears (indices 1-4), keep 13 keypoints
_KEEP_IDX = np.array([0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])

# ---- HELPERS ----
SKELETON = [(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
COLORS = [(0,255,0), (255,0,0), (0,255,255), (255,0,255), (0,165,255), (255,255,0)]


def draw_person(frame, bbox, tid, det_conf, kp_frame, pose_conf, color, stroke_label=None):
    x0, y0, x1, y1 = [int(v) for v in bbox]
    cv2.rectangle(frame, (x0,y0), (x1,y1), color, 2)
    text = f"id={tid} det={det_conf:.2f} pose={pose_conf:.2f}"
    if stroke_label:
        text += f" [{stroke_label}]"
    cv2.putText(frame, text, (x0, y0-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    for j in range(17):
        if kp_frame[j][2] >= 0.3:
            cv2.circle(frame, (int(kp_frame[j][0]), int(kp_frame[j][1])), 4, color, -1)
    for j1, j2 in SKELETON:
        if kp_frame[j1][2] >= 0.3 and kp_frame[j2][2] >= 0.3:
            cv2.line(frame, (int(kp_frame[j1][0]),int(kp_frame[j1][1])),
                     (int(kp_frame[j2][0]),int(kp_frame[j2][1])), color, 2)


def kp_to_feature(kp_xy, bbox):
    """YOLO keypoints (17,3) in (x,y,conf) -> (26,) feature for RNN.

    Normalize to bbox (matching upstream training convention).
    Output order: (y, x) per kept keypoint, flattened.
    """
    x0, y0, x1, y1 = bbox
    bw, bh = max(x1 - x0, 1), max(y1 - y0, 1)
    kp = kp_xy[_KEEP_IDX]            # (13, 3)
    yx = np.stack([
        (kp[:, 1] - y0) / bh,        # y normalised
        (kp[:, 0] - x0) / bw,        # x normalised
    ], axis=1)                        # (13, 2)
    return yx.reshape(-1).astype(np.float32)  # (26,)


# ---- LOAD MODELS ----
det_model = YOLO(DETECT_WEIGHTS)
pose_model = YOLO(POSE_WEIGHTS)
rnn = keras.models.load_model(RNN_WEIGHTS, compile=False)
print(f"Loaded: detect={DETECT_WEIGHTS}, pose={POSE_WEIGHTS}")
print(f"Loaded stroke RNN: input={rnn.input_shape}, output={rnn.output_shape}")

cap = cv2.VideoCapture(INPUT_VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {W}x{H}  fps={fps:.1f}  frames={total}")

writer = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

# Per-track state for stroke classification
track_windows = {}    # tid -> deque of (26,) features
track_last_infer = {} # tid -> last frame_idx we ran RNN
track_labels = {}     # tid -> latest stroke label string

fi = 0
t0 = time.time()
stroke_events = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ---- Stage 1: detect all persons ----
    det_results = det_model.track(
        frame, persist=True, verbose=False,
        classes=[0], conf=DETECT_CONF, imgsz=DETECT_IMGSZ, device=DETECT_DEVICE,
        tracker=_TRACKER_PATH,
    )
    dr = det_results[0]

    if dr.boxes is not None and dr.boxes.id is not None:
        det_boxes = dr.boxes.xyxy.cpu().numpy()
        det_ids = dr.boxes.id.cpu().numpy().astype(int)
        det_confs = dr.boxes.conf.cpu().numpy()
        n_det = len(det_boxes)

        # Filter: on-court only, top MAX_PERSONS by confidence
        keep = [i for i in range(n_det) if on_court(det_boxes[i])]
        keep.sort(key=lambda i: -det_confs[i])
        keep = keep[:MAX_PERSONS]

        # ---- Stage 2: batch pose on on-court crops ----
        crops = []
        crop_offsets = []
        kept_indices = []
        for i in keep:
            x0, y0, x1, y1 = det_boxes[i].astype(int)
            bw, bh = x1 - x0, y1 - y0
            pad = int(CROP_PAD * max(bw, bh))
            cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
            cx1, cy1 = min(W, x1 + pad), min(H, y1 + pad)
            crops.append(frame[cy0:cy1, cx0:cx1])
            crop_offsets.append((cx0, cy0))
            kept_indices.append(i)

        pose_results = pose_model.predict(
            crops, verbose=False, classes=[0],
            conf=POSE_CONF, imgsz=POSE_IMGSZ, device=POSE_DEVICE,
        ) if crops else []

        for j, i in enumerate(kept_indices):
            x0, y0, x1, y1 = det_boxes[i].astype(int)
            tid = det_ids[i]
            dc = det_confs[i]
            cx0, cy0 = crop_offsets[j]
            pr = pose_results[j]

            if pr.boxes is None or len(pr.boxes) == 0 or pr.keypoints is None:
                continue  # no pose → skip (likely off-court person)

            best = int(pr.boxes.conf.cpu().numpy().argmax())
            kp_crop = pr.keypoints.data[best].cpu().numpy()  # (17, 3) in crop coords
            pc = float(pr.boxes.conf[best])

            # Map keypoints to full-frame coords
            kp_frame = kp_crop.copy()
            kp_frame[:, 0] += cx0  # x
            kp_frame[:, 1] += cy0  # y

            # ---- Stage 3: stroke classification ----
            bbox = (x0, y0, x1, y1)
            if tid not in track_windows:
                track_windows[tid] = deque(maxlen=WINDOW)
                track_last_infer[tid] = -999
            track_windows[tid].append(kp_to_feature(kp_frame, bbox))

            if (len(track_windows[tid]) >= WINDOW
                    and fi - track_last_infer[tid] >= STRIDE):
                track_last_infer[tid] = fi
                feats = np.stack(list(track_windows[tid]), axis=0)  # (30, 26)
                probs = rnn.predict(feats[None, ...], verbose=0)[0]
                k = int(np.argmax(probs))
                conf_stroke = float(probs[k])
                label = LABELS[k]
                if conf_stroke >= MIN_STROKE_CONF and label != "neutral":
                    track_labels[tid] = f"{label} {conf_stroke:.2f}"
                    # bbox foot center: (cx, y1) — y1 is bbox bottom
                    foot_x = (x0 + x1) / 2.0
                    foot_y = float(y1)
                    stroke_events.append((fi, tid, label, conf_stroke, foot_x, foot_y))
                else:
                    track_labels[tid] = None

            # ---- Draw ----
            c = COLORS[abs(tid) % len(COLORS)]
            slabel = track_labels.get(tid)
            draw_person(frame, bbox, tid, dc, kp_frame, pc, c, stroke_label=slabel)

    writer.write(frame)
    fi += 1
    if fi % 500 == 0:
        elapsed = time.time() - t0
        print(f"  frame {fi}/{total}  {fi/elapsed:.1f} fps  eta {(total-fi)/(fi/elapsed):.0f}s")

cap.release()
writer.release()
print(f"\nDone: {fi} frames in {time.time()-t0:.1f}s ({fi/(time.time()-t0):.1f} fps)")
print(f"Saved: {OUTPUT_VIDEO}")
# ---- Post-process: merge consecutive same player+label events ----
def merge_strokes(events, gap):
    """Merge runs of same (player, label) within `gap` frames → keep peak conf."""
    if not events:
        return []
    merged = []
    cur_frame, cur_tid, cur_label, cur_conf, cur_fx, cur_fy = events[0]
    for frame_i, tid, label, conf_s, fx, fy in events[1:]:
        if tid == cur_tid and label == cur_label and frame_i - cur_frame <= gap:
            if conf_s > cur_conf:
                cur_frame, cur_conf, cur_fx, cur_fy = frame_i, conf_s, fx, fy
        else:
            merged.append((cur_frame, cur_tid, cur_label, cur_conf, cur_fx, cur_fy))
            cur_frame, cur_tid, cur_label, cur_conf, cur_fx, cur_fy = frame_i, tid, label, conf_s, fx, fy
    merged.append((cur_frame, cur_tid, cur_label, cur_conf, cur_fx, cur_fy))
    return merged

merged_events = merge_strokes(stroke_events, MERGE_GAP)

unique_ids = sorted(set(tid for _, tid, _, _, _, _ in merged_events))
print(f"\nUnique player IDs with strokes: {unique_ids} ({len(unique_ids)} players)")
print(f"\nStroke events: {len(stroke_events)} raw → {len(merged_events)} merged")
print(f"{'frame':>7s}  {'player':>6s}  {'label':>10s}  {'conf':>6s}  {'foot_x':>7s}  {'foot_y':>7s}")
for frame_i, tid, label, conf_s, fx, fy in merged_events:
    print(f"  {frame_i:5d}  {tid:6d}  {label:10s}  {conf_s:.3f}  {fx:7.1f}  {fy:7.1f}")
