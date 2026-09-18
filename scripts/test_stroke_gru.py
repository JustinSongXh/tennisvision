"""Test stroke GRU on sample_short.mp4 — DEPRECATED.

Use test_serve_detection.py instead (works with pre-extracted keypoints JSON,
no need to re-run YOLO inference locally).
"""

import cv2, time, warnings, logging
from collections import deque
import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings('ignore')
logging.getLogger('ultralytics').setLevel(logging.ERROR)
from ultralytics import YOLO

# ---- Config ----
VIDEO = '/Users/justinsong/WorkSpace/solo/samples/sample_short.mp4'
OUT = '/Users/justinsong/WorkSpace/solo/results/sample_short_gru_check.mp4'
MAX_FRAMES = 300
CROP_PAD = 0.15
SEQ_LEN = 30
STRIDE = 5
LABELS = {0: 'backhand', 1: 'forehand', 2: 'serve'}
COLORS = {'backhand': (255, 0, 0), 'forehand': (0, 255, 0), 'serve': (0, 0, 255), None: (180, 180, 180)}
SKELETON = [(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]


class StrokeGRU(nn.Module):
    def __init__(self, input_dim=34, hidden=128, n_layers=2, n_classes=3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, n_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, n_classes))

    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h[-1])


# ---- Load models ----
det_model = YOLO('yolo11n.pt')
pose_model = YOLO('weights/yolo26s-pose.pt')

gru = StrokeGRU()
gru.load_state_dict(torch.load('weights/stroke_gru_best.pt', map_location='cpu'))
gru.eval()

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

# Per-track state
track_windows = {}     # tid -> deque of (34,) features
track_last_infer = {}  # tid -> last frame we ran GRU
track_labels = {}      # tid -> current label string

fi = 0
t0 = time.time()

while fi < MAX_FRAMES:
    ret, frame = cap.read()
    if not ret:
        break

    dr = det_model.track(frame, persist=True, verbose=False, classes=[0], conf=0.3, imgsz=1280)[0]

    if dr.boxes is not None and dr.boxes.id is not None:
        det_boxes = dr.boxes.xyxy.cpu().numpy().astype(int)
        det_ids = dr.boxes.id.cpu().numpy().astype(int)
        det_confs = dr.boxes.conf.cpu().numpy()

        crops, offsets, kept = [], [], []
        for i in range(len(det_boxes)):
            x0, y0, x1, y1 = det_boxes[i]
            pad = int(CROP_PAD * max(x1 - x0, y1 - y0))
            cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
            cx1, cy1 = min(W, x1 + pad), min(H, y1 + pad)
            crops.append(frame[cy0:cy1, cx0:cx1])
            offsets.append((cx0, cy0))
            kept.append(i)

        pose_results = pose_model.predict(crops, verbose=False, classes=[0], conf=0.15, imgsz=640) if crops else []

        for j, i in enumerate(kept):
            x0, y0, x1, y1 = det_boxes[i]
            tid = int(det_ids[i])
            dc = det_confs[i]
            cx0, cy0 = offsets[j]
            pr = pose_results[j]

            if pr.boxes is None or len(pr.boxes) == 0 or pr.keypoints is None:
                continue

            best = int(pr.boxes.conf.cpu().numpy().argmax())
            kp = pr.keypoints.data[best].cpu().numpy()

            # Map to frame coords
            kp_x = kp[:, 0] + cx0
            kp_y = kp[:, 1] + cy0
            kp_v = kp[:, 2]

            # Bbox-relative normalization
            valid = kp_v >= 0.3
            if valid.sum() < 3:
                continue
            xmin, xmax = kp_x[valid].min(), kp_x[valid].max()
            ymin, ymax = kp_y[valid].min(), kp_y[valid].max()
            bw, bh = max(xmax - xmin, 1), max(ymax - ymin, 1)
            xs_rel = (kp_x - xmin) / bw
            ys_rel = (kp_y - ymin) / bh

            # Interleave x,y per keypoint: [kp0_x, kp0_y, kp1_x, kp1_y, ...]
            feat = np.stack([xs_rel, ys_rel], axis=1).reshape(-1).astype(np.float32)  # (34,)

            # Accumulate per track
            if tid not in track_windows:
                track_windows[tid] = deque(maxlen=SEQ_LEN)
                track_last_infer[tid] = -999
            track_windows[tid].append(feat)

            # Run GRU every STRIDE frames
            if len(track_windows[tid]) >= SEQ_LEN and fi - track_last_infer[tid] >= STRIDE:
                track_last_infer[tid] = fi
                seq = np.stack(list(track_windows[tid]), axis=0)  # (30, 34)
                with torch.no_grad():
                    logits = gru(torch.FloatTensor(seq).unsqueeze(0))
                    probs = torch.softmax(logits, dim=1)[0].numpy()
                pred = int(np.argmax(probs))
                conf = float(probs[pred])
                label = LABELS[pred]
                if conf >= 0.7:
                    track_labels[tid] = f"{label} {conf:.2f}"
                else:
                    track_labels[tid] = None

            # Draw
            slabel = track_labels.get(tid)
            color = COLORS.get(slabel.split()[0] if slabel else None, (180, 180, 180))
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
            text = f"id={tid} det={dc:.2f}"
            if slabel:
                text += f" [{slabel}]"
            cv2.putText(frame, text, (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw skeleton
            for k in range(17):
                if kp_v[k] >= 0.3:
                    cv2.circle(frame, (int(kp_x[k]), int(kp_y[k])), 3, color, -1)
            for j1, j2 in SKELETON:
                if kp_v[j1] >= 0.3 and kp_v[j2] >= 0.3:
                    cv2.line(frame, (int(kp_x[j1]), int(kp_y[j1])), (int(kp_x[j2]), int(kp_y[j2])), color, 2)

    writer.write(frame)
    fi += 1
    if fi % 100 == 0:
        print(f"  frame {fi}/{MAX_FRAMES} {fi/(time.time()-t0):.1f} fps")

cap.release()
writer.release()
print(f"Done: {fi} frames in {time.time()-t0:.1f}s")
print(f"Saved: {OUT}")
