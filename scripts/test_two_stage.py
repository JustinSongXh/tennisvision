"""Test two-stage detection: yolo11m detect -> yolo26m-pose on crop.

Runs on a short segment of sample_short.mp4 and writes an annotated video
to /tmp/two_stage_test.mp4 for visual inspection.
"""

import cv2
import time
import warnings
import logging

import numpy as np

warnings.filterwarnings("ignore", message=".*GMC failed.*")
logging.getLogger("ultralytics").setLevel(logging.ERROR)

from ultralytics import YOLO

# ---- CONFIG ----
INPUT_VIDEO = "/Users/justinsong/WorkSpace/solo/samples/sample_short.mp4"
OUTPUT_VIDEO = "/tmp/two_stage_test.mp4"
DETECT_WEIGHTS = "yolo11m.pt"
POSE_WEIGHTS = "weights/yolo26m-pose.pt"
DETECT_CONF = 0.3
DETECT_IMGSZ = 1280
POSE_CONF = 0.15
POSE_IMGSZ = 640
CROP_PAD = 0.15          # bbox padding ratio before pose crop
MAX_FRAMES = 300          # test on first N frames (0 = all)

SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]
COLORS = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (0, 165, 255), (255, 255, 0)]


def draw_person(frame, bbox, tid, det_conf, kp, pose_conf, color):
    x0, y0, x1, y1 = bbox
    cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
    label = f"id={tid} det={det_conf:.2f} pose={pose_conf:.2f}"
    cv2.putText(frame, label, (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    for j in range(17):
        if kp[j][2] >= 0.3:
            cv2.circle(frame, (int(kp[j][0]), int(kp[j][1])), 4, color, -1)
    for j1, j2 in SKELETON:
        if kp[j1][2] >= 0.3 and kp[j2][2] >= 0.3:
            p1 = (int(kp[j1][0]), int(kp[j1][1]))
            p2 = (int(kp[j2][0]), int(kp[j2][1]))
            cv2.line(frame, p1, p2, color, 2)


# ---- LOAD ----
det_model = YOLO(DETECT_WEIGHTS)
pose_model = YOLO(POSE_WEIGHTS)

cap = cv2.VideoCapture(INPUT_VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
n_frames = min(total, MAX_FRAMES) if MAX_FRAMES > 0 else total
print(f"Video: {w}x{h} fps={fps:.1f} frames={total}, testing {n_frames}")

writer = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

fi = 0
t0 = time.time()
stats = {"det_total": 0, "pose_ok": 0, "pose_fail": 0}

while fi < n_frames:
    ret, frame = cap.read()
    if not ret:
        break

    # Stage 1: detect all persons
    det_results = det_model.track(
        frame, persist=True, verbose=False,
        classes=[0], conf=DETECT_CONF, imgsz=DETECT_IMGSZ,
    )
    dr = det_results[0]

    if dr.boxes is not None and dr.boxes.id is not None:
        boxes = dr.boxes.xyxy.cpu().numpy()
        ids = dr.boxes.id.cpu().numpy().astype(int)
        det_confs = dr.boxes.conf.cpu().numpy()

        for i in range(len(boxes)):
            stats["det_total"] += 1
            x0, y0, x1, y1 = boxes[i].astype(int)
            tid = ids[i]
            dc = det_confs[i]

            # Pad bbox for pose crop
            bw, bh = x1 - x0, y1 - y0
            pad = int(CROP_PAD * max(bw, bh))
            cx0 = max(0, x0 - pad)
            cy0 = max(0, y0 - pad)
            cx1 = min(w, x1 + pad)
            cy1 = min(h, y1 + pad)
            crop = frame[cy0:cy1, cx0:cx1]

            # Stage 2: pose on crop
            pr = pose_model.predict(crop, verbose=False, classes=[0], conf=POSE_CONF, imgsz=POSE_IMGSZ)
            rp = pr[0]

            if rp.boxes is not None and len(rp.boxes) > 0 and rp.keypoints is not None:
                best = int(rp.boxes.conf.cpu().numpy().argmax())
                kp = rp.keypoints.data[best].cpu().numpy()  # (17, 3) in crop coords
                pc = float(rp.boxes.conf[best])

                # Map keypoints back to full frame
                kp_frame = kp.copy()
                kp_frame[:, 0] += cx0  # x
                kp_frame[:, 1] += cy0  # y

                color = COLORS[abs(tid) % len(COLORS)]
                draw_person(frame, (x0, y0, x1, y1), tid, dc, kp_frame, pc, color)
                stats["pose_ok"] += 1
            else:
                # No pose — draw grey bbox
                cv2.rectangle(frame, (x0, y0), (x1, y1), (128, 128, 128), 1)
                cv2.putText(frame, f"id={tid} det={dc:.2f} NO_POSE",
                            (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
                stats["pose_fail"] += 1

    writer.write(frame)
    fi += 1
    if fi % 100 == 0:
        elapsed = time.time() - t0
        print(f"  frame {fi}/{n_frames} {fi/elapsed:.1f} fps")

cap.release()
writer.release()
elapsed = time.time() - t0
print(f"\nDone: {fi} frames in {elapsed:.1f}s ({fi/elapsed:.1f} fps)")
print(f"Detections: {stats['det_total']}  pose_ok: {stats['pose_ok']}  pose_fail: {stats['pose_fail']}")
print(f"Saved: {OUTPUT_VIDEO}")
