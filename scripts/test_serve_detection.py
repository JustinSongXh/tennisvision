"""Serve detection test — final validated pipeline.

Uses pre-extracted keypoints JSON (from Kaggle two-stage pipeline) to test
serve detection locally without re-running YOLO inference.

Pipeline:
  1. Load keypoints_all_frames.json (yolo11m + yolo26s-pose output)
  2. Homography on-court filter
  3. 4-slot spatial mapping (NEAR_L/R, FAR_L/R)
  4. Wrist speed trigger (threshold=0.15)
  5. GRU v4 inference during active periods (every frame)
  6. Serve marking (prob > 0.8, merge consecutive frames)

Usage:
    python scripts/test_serve_detection.py \
        --keypoints results/keypoints_all_frames_v6.json \
        --gru weights/stroke_gru_v4_best.pt \
        --out results/serve_events.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---- Court calibration (sample_short.mp4) ----
H_IMG_TO_REAL = np.array([
    [-0.004212704126665825, -0.00958000348568168, 8.066176934954633],
    [9.370094579164494e-05, 0.01803202259785551, -15.976691289361806],
    [1.1053780957617317e-07, -0.0019774218879326623, 1.0],
], dtype=np.float64)
COURT_W, COURT_L, MARGIN = 10.97, 23.77, 3.0
NET_Y_M = COURT_L / 2


def to_court(fx, fy):
    p = H_IMG_TO_REAL @ [fx, fy, 1.0]
    if abs(p[2]) < 1e-9:
        return None, None
    return p[0] / p[2], p[1] / p[2]


def on_court(rx, ry):
    return (-MARGIN <= rx <= COURT_W + MARGIN and -MARGIN <= ry <= COURT_L + MARGIN)


def get_slot(rx, ry):
    mid_x = COURT_W / 2
    return (0 if rx < mid_x else 1) if ry < NET_Y_M else (2 if rx < mid_x else 3)


SLOT_NAMES = {0: "NEAR_L", 1: "NEAR_R", 2: "FAR_L", 3: "FAR_R"}


# ---- GRU model ----
class StrokeGRU(nn.Module):
    def __init__(self, input_dim=34, hidden=128, n_layers=2, n_classes=4):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, n_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, n_classes)
        )

    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keypoints", default="results/keypoints_all_frames_v6.json")
    parser.add_argument("--gru", default="weights/stroke_gru_v4_best.pt")
    parser.add_argument("--out", default="results/serve_events.json")
    parser.add_argument("--speed-threshold", type=float, default=0.15)
    parser.add_argument("--serve-threshold", type=float, default=0.8)
    parser.add_argument("--gap-tolerance", type=int, default=5)
    parser.add_argument("--seq-len", type=int, default=30)
    args = parser.parse_args()

    # Load data
    print(f"Loading keypoints: {args.keypoints}")
    with open(args.keypoints) as f:
        data = json.load(f)
    print(f"  {data['total_frames']} frames, fps={data['fps']:.1f}")

    # Load GRU
    print(f"Loading GRU: {args.gru}")
    gru = StrokeGRU(n_classes=4)
    gru.load_state_dict(torch.load(args.gru, map_location="cpu"))
    gru.eval()

    LABELS = {0: "backhand", 1: "forehand", 2: "serve", 3: "background"}
    SEQ_LEN = args.seq_len

    # Per-slot state
    kp_buffers = {s: [] for s in range(4)}
    prev_kps = {s: None for s in range(4)}
    active = {s: False for s in range(4)}
    gap_count = {s: 0 for s in range(4)}

    serve_hits = []
    all_events = []  # all GRU outputs for debugging
    gru_count = 0

    for fd in data["frames"]:
        fi = fd["frame"]

        # Assign detections to slots
        slot_dets = {}
        for det in fd["detections"]:
            rx, ry = to_court(det["foot_x"], det["foot_y"])
            if rx is None or not on_court(rx, ry):
                continue
            slot = get_slot(rx, ry)
            cx = COURT_W * 0.25 if slot % 2 == 0 else COURT_W * 0.75
            cy = COURT_L * 0.15 if slot < 2 else COURT_L * 0.85
            dist = np.sqrt((rx - cx) ** 2 + (ry - cy) ** 2)
            if slot not in slot_dets or dist < slot_dets[slot][0]:
                slot_dets[slot] = (dist, det)

        for slot in range(4):
            if slot not in slot_dets:
                if active[slot]:
                    gap_count[slot] += 1
                    if gap_count[slot] > args.gap_tolerance:
                        active[slot] = False
                continue

            det = slot_dets[slot][1]
            xs = np.array(det["kp_norm_x"])
            ys = np.array(det["kp_norm_y"])
            kp = np.stack([xs, ys], axis=1)

            kp_buffers[slot].append(kp)
            if len(kp_buffers[slot]) > SEQ_LEN + 30:
                kp_buffers[slot] = kp_buffers[slot][-(SEQ_LEN + 30):]

            # Wrist speed
            speed = 0.0
            if prev_kps[slot] is not None:
                for wi in [9, 10]:
                    dx = kp[wi, 0] - prev_kps[slot][wi, 0]
                    dy = kp[wi, 1] - prev_kps[slot][wi, 1]
                    speed += np.sqrt(dx ** 2 + dy ** 2)
            prev_kps[slot] = kp

            if speed > args.speed_threshold:
                active[slot] = True
                gap_count[slot] = 0
            elif active[slot]:
                gap_count[slot] += 1
                if gap_count[slot] > args.gap_tolerance:
                    active[slot] = False

            # GRU during active period
            if active[slot] and len(kp_buffers[slot]) >= SEQ_LEN:
                feat = np.array(
                    [k[:, :2].reshape(-1) for k in kp_buffers[slot][-SEQ_LEN:]],
                    dtype=np.float32,
                )
                with torch.no_grad():
                    probs = torch.softmax(
                        gru(torch.FloatTensor(feat).unsqueeze(0)), dim=1
                    )[0].numpy()
                gru_count += 1
                pred = int(np.argmax(probs))

                event = {
                    "frame": fi,
                    "time": round(fi / data["fps"], 2),
                    "slot": slot,
                    "slot_name": SLOT_NAMES[slot],
                    "label": LABELS[pred],
                    "conf": round(float(probs[pred]), 3),
                    "probs": {LABELS[i]: round(float(probs[i]), 3) for i in range(4)},
                    "wrist_speed": round(speed, 3),
                    "foot_x": det["foot_x"],
                    "foot_y": det["foot_y"],
                }
                all_events.append(event)

                if probs[2] > args.serve_threshold:
                    serve_hits.append((fi, slot, float(probs[2])))

    # Merge consecutive serve hits
    merged_serves = []
    if serve_hits:
        cur_start, cur_end, cur_slot, cur_conf = (
            serve_hits[0][0], serve_hits[0][0], serve_hits[0][1], serve_hits[0][2],
        )
        for fi, slot, conf in serve_hits[1:]:
            if slot == cur_slot and fi - cur_end <= 10:
                cur_end = fi
                cur_conf = max(cur_conf, conf)
            else:
                merged_serves.append({
                    "start_frame": cur_start,
                    "end_frame": cur_end,
                    "start_time": round(cur_start / data["fps"], 2),
                    "end_time": round(cur_end / data["fps"], 2),
                    "slot": cur_slot,
                    "slot_name": SLOT_NAMES[cur_slot],
                    "conf": round(cur_conf, 3),
                })
                cur_start, cur_end, cur_slot, cur_conf = fi, fi, slot, conf
        merged_serves.append({
            "start_frame": cur_start,
            "end_frame": cur_end,
            "start_time": round(cur_start / data["fps"], 2),
            "end_time": round(cur_end / data["fps"], 2),
            "slot": cur_slot,
            "slot_name": SLOT_NAMES[cur_slot],
            "conf": round(cur_conf, 3),
        })

    # Print results
    print(f"\nGRU invocations: {gru_count}")
    print(f"Total events: {len(all_events)}")
    print(f"  backhand: {sum(1 for e in all_events if e['label'] == 'backhand')}")
    print(f"  forehand: {sum(1 for e in all_events if e['label'] == 'forehand')}")
    print(f"  serve:    {sum(1 for e in all_events if e['label'] == 'serve')}")
    print(f"  background: {sum(1 for e in all_events if e['label'] == 'background')}")

    print(f"\nServe events (merged): {len(merged_serves)}")
    for s in merged_serves:
        print(f"  {s['start_time']}s-{s['end_time']}s {s['slot_name']} conf={s['conf']}")

    # Save
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    output = {
        "params": {
            "speed_threshold": args.speed_threshold,
            "serve_threshold": args.serve_threshold,
            "gap_tolerance": args.gap_tolerance,
            "seq_len": args.seq_len,
        },
        "stats": {
            "total_frames": data["total_frames"],
            "fps": data["fps"],
            "gru_invocations": gru_count,
            "total_events": len(all_events),
        },
        "serve_events": merged_serves,
        "all_events": all_events,
    }
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
