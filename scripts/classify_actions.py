#!/usr/bin/env python3
"""Classify player actions within rally windows (Step 6, optional).

For each rally, opens the video, extracts keypoints on-demand,
runs GRU classification, and merges short-duration same-action events.

Inputs:
  --players   player_detections.json (Step 3)
  --serves    serve_events.json (Step 4)
  --rallies   rally_events.json (Step 5)
  --calib     calib.json (Step 1)
  --ball      ball_positions.json (Step 2, optional)
  --video     original video (for on-demand pose extraction)

Outputs action_events.json with per-rally action classifications.

Usage:
    python -u scripts/classify_actions.py \
        --players results/sample3/player_detections.json \
        --serves results/sample3/serve_events.json \
        --rallies results/sample3/rally_events.json \
        --calib results/sample3/calib.json \
        --ball results/sample3/ball_positions.json \
        --video samples/sample3.mp4 \
        --out results/sample3/action_events.json
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

from tennisvision.action.slot_mapper import SlotMapper, SLOT_NAMES
from tennisvision.action.gru_classifier import StrokeGRU, LABELS as GRU_LABELS


# =====================================================================
# On-demand pose extraction (same as detect_serves.py)
# =====================================================================

class PoseExtractor:
    def __init__(self, video_path, device, quantize):
        from ultralytics import YOLO
        self.cap = cv2.VideoCapture(video_path)
        self.W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.pose_model = YOLO("yolo26s-pose.pt")
        self.device = device
        self.quantize = quantize
        self.crop_pad = 0.15
        self._last_fi = -1
        self._last_frame = None

    def _read_frame(self, fi):
        if fi == self._last_fi:
            return self._last_frame
        if fi != self._last_fi + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = self.cap.read()
        self._last_fi = fi
        self._last_frame = frame if ret else None
        return self._last_frame

    def extract(self, fi, detections):
        """Extract keypoints for detections in frame fi.

        Returns list of dicts with tid, kp_norm_x, kp_norm_y added.
        """
        frame = self._read_frame(fi)
        if frame is None:
            return []

        W, H = self.W, self.H
        crops, offsets, kept = [], [], []
        for i, det in enumerate(detections):
            bbox = det["bbox"]
            x0, y0, x1, y1 = bbox
            pad = int(self.crop_pad * max(x1 - x0, y1 - y0))
            cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
            cx1, cy1 = min(W, x1 + pad), min(H, y1 + pad)
            crops.append(frame[cy0:cy1, cx0:cx1])
            offsets.append((cx0, cy0))
            kept.append(i)

        if not crops:
            return []

        prs = self.pose_model.predict(crops, verbose=False, classes=[0],
                                       conf=0.15, imgsz=640, device=self.device,
                                       quantize=self.quantize)
        results = []
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
            det_out = dict(detections[i])
            det_out["kp_norm_x"] = [round(float((kp_x[k] - xn) / bw), 4) for k in range(17)]
            det_out["kp_norm_y"] = [round(float((kp_y[k] - yn) / bh), 4) for k in range(17)]
            results.append(det_out)
        return results

    def close(self):
        self.cap.release()


# =====================================================================
# Buffer with interpolation
# =====================================================================

def build_buffer(slot_buf, fi, seq_len, max_missing_ratio=0.2):
    """Build SEQ_LEN buffer with linear interpolation for small gaps."""
    if not slot_buf:
        return None, False

    start_fi = fi - seq_len + 1
    frame_map = {f: kp for f, kp in slot_buf if start_fi <= f <= fi}

    if len(frame_map) < seq_len * (1 - max_missing_ratio):
        return None, False

    result = []
    for t in range(start_fi, fi + 1):
        if t in frame_map:
            result.append(frame_map[t])
        else:
            prev_f = prev_kp = next_f = next_kp = None
            for dt in range(1, seq_len):
                if prev_kp is None and (t - dt) in frame_map:
                    prev_f, prev_kp = t - dt, frame_map[t - dt]
                if next_kp is None and (t + dt) in frame_map:
                    next_f, next_kp = t + dt, frame_map[t + dt]
                if prev_kp is not None and next_kp is not None:
                    break
            if prev_kp is not None and next_kp is not None:
                alpha = (t - prev_f) / (next_f - prev_f)
                result.append(prev_kp * (1 - alpha) + next_kp * alpha)
            elif prev_kp is not None:
                result.append(prev_kp)
            elif next_kp is not None:
                result.append(next_kp)
            else:
                return None, False

    return np.array(result, dtype=np.float32), True


# =====================================================================
# Merge same-action events within a time window
# =====================================================================

def merge_actions(events, fps, merge_window=1.0):
    """Merge consecutive events with the same label+slot within merge_window seconds.

    Keeps the highest confidence and extends the time range.
    """
    if not events:
        return []

    merged = []
    cur = dict(events[0])
    cur["end_frame"] = cur["frame"]
    cur["end_time"] = cur["time"]

    for ev in events[1:]:
        same_slot = ev["slot"] == cur["slot"]
        same_label = ev["label"] == cur["label"]
        close = (ev["time"] - cur["end_time"]) <= merge_window

        if same_slot and same_label and close:
            cur["end_frame"] = ev["frame"]
            cur["end_time"] = ev["time"]
            cur["conf"] = max(cur["conf"], ev["conf"])
        else:
            merged.append(cur)
            cur = dict(ev)
            cur["end_frame"] = cur["frame"]
            cur["end_time"] = cur["time"]

    merged.append(cur)

    # Filter out background events and very short events
    result = []
    for ev in merged:
        if ev["label"] == "background":
            continue
        result.append({
            "start_frame": ev["frame"],
            "end_frame": ev["end_frame"],
            "start_time": ev["time"],
            "end_time": ev["end_time"],
            "slot": ev["slot"],
            "slot_name": ev["slot_name"],
            "label": ev["label"],
            "conf": ev["conf"],
        })

    return result


# =====================================================================
# Assign action to near/far side using ball position
# =====================================================================

def assign_action_side(action, ball_det, ball_pred, mapper):
    """Use ball position at action time to determine which side hit the ball.

    If ball is on the same side as the slot (near/far), the action is likely
    the hitter. Otherwise it might be a preparation movement.
    """
    mid_frame = (action["start_frame"] + action["end_frame"]) // 2
    k = str(mid_frame)
    bp = ball_det.get(k) or ball_pred.get(k)
    if bp is None:
        action["ball_side"] = "unknown"
        return

    bx, by = bp[0], bp[1]
    rx, ry = mapper.to_court(bx, by)
    if rx is None:
        action["ball_side"] = "unknown"
        return

    slot = action["slot"]
    # Near side = ry < net, Far side = ry > net
    ball_near = ry < mapper.net_y_m
    player_near = slot < 2
    action["ball_side"] = "same" if ball_near == player_near else "opposite"


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--players", required=True,
                        help="player_detections.json from detect_players.py")
    parser.add_argument("--rallies", required=True,
                        help="rally_events.json from detect_rallies.py")
    parser.add_argument("--calib", required=True, help="calib.json")
    parser.add_argument("--video", required=True,
                        help="Original video for on-demand pose extraction")
    parser.add_argument("--gru", default="weights/stroke_gru_v4_best.pt")
    parser.add_argument("--ball", default=None,
                        help="ball_positions.json for ball-side assignment")
    parser.add_argument("--out", required=True)
    parser.add_argument("--speed-threshold", type=float, default=0.15)
    parser.add_argument("--min-conf", type=float, default=0.5)
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--gap-tolerance", type=int, default=5)
    parser.add_argument("--merge-window", type=float, default=1.0,
                        help="Seconds within which same slot+label events are merged")
    parser.add_argument("--no-half", action="store_true", help="Disable FP16")
    args = parser.parse_args()

    # ---- Load inputs ----
    with open(args.calib) as f:
        calib = json.load(f)
    H_img_to_real = np.array(calib["H_img_to_real"], dtype=np.float64)
    mapper = SlotMapper(H_img_to_real)

    print(f"Loading players: {args.players}")
    with open(args.players) as f:
        player_data = json.load(f)
    fps = player_data["fps"]
    player_frames = player_data["frames"]
    # Build frame index for fast lookup
    frame_index = {fd["frame"]: fd for fd in player_frames}

    print(f"Loading rallies: {args.rallies}")
    with open(args.rallies) as f:
        rally_data = json.load(f)
    rallies = rally_data.get("rallies", rally_data.get("rally_events", []))
    print(f"  {len(rallies)} rallies")

    ball_det, ball_pred = {}, {}
    if args.ball:
        print(f"Loading ball: {args.ball}")
        with open(args.ball) as f:
            ball_data = json.load(f)
        ball_det = ball_data.get("detected", {})
        ball_pred = ball_data.get("predicted", {})
        print(f"  {len(ball_det)} detected, {len(ball_pred)} predicted")

    # ---- Load GRU ----
    print(f"Loading GRU: {args.gru}")
    gru = StrokeGRU(n_classes=4)
    gru.load_state_dict(torch.load(args.gru, map_location="cpu"))
    gru.eval()

    # ---- Init pose extractor ----
    device = 0 if torch.cuda.is_available() else "cpu"
    quantize = "fp16" if (torch.cuda.is_available() and not args.no_half) else None
    pose = PoseExtractor(args.video, device, quantize)

    SEQ_LEN = args.seq_len
    BUF_MAX = SEQ_LEN + 30
    t0 = time.time()
    total_pose_frames = 0
    total_gru_calls = 0

    all_rally_actions = []

    for ri, rally in enumerate(rallies):
        start_frame = rally.get("start_frame", int(rally.get("start_time", 0) * fps))
        end_frame = rally.get("end_frame", int(rally.get("end_time", 0) * fps))

        # Reset per-slot state for each rally
        slot_kp_buf = {s: [] for s in range(4)}
        prev_kps = {s: None for s in range(4)}
        active = {s: False for s in range(4)}
        gap_count = {s: 0 for s in range(4)}
        raw_events = []

        for fi in range(start_frame, end_frame + 1):
            fd = frame_index.get(fi)
            if fd is None:
                continue

            slot_dets = mapper.assign(fd["detections"])

            # On-demand pose for assigned slots
            slot_det_list = [slot_dets[s] for s in sorted(slot_dets.keys())]
            if slot_det_list:
                kp_results = pose.extract(fi, slot_det_list)
                total_pose_frames += 1
                kp_by_tid = {r["tid"]: r for r in kp_results}
            else:
                kp_by_tid = {}

            for slot in range(4):
                if slot not in slot_dets:
                    if active[slot]:
                        gap_count[slot] += 1
                        if gap_count[slot] > args.gap_tolerance:
                            active[slot] = False
                    continue

                det = slot_dets[slot]
                tid = det["tid"]
                if tid not in kp_by_tid:
                    continue

                kp_det = kp_by_tid[tid]
                kp = np.stack([kp_det["kp_norm_x"], kp_det["kp_norm_y"]],
                              axis=1).astype(np.float32)

                slot_kp_buf[slot].append((fi, kp))
                if len(slot_kp_buf[slot]) > BUF_MAX:
                    slot_kp_buf[slot] = slot_kp_buf[slot][-BUF_MAX:]

                # Wrist speed trigger
                speed = 0.0
                if prev_kps[slot] is not None:
                    for wi in [9, 10]:
                        speed += np.sqrt((kp[wi, 0] - prev_kps[slot][wi, 0]) ** 2 +
                                         (kp[wi, 1] - prev_kps[slot][wi, 1]) ** 2)
                prev_kps[slot] = kp

                if speed > args.speed_threshold:
                    active[slot] = True
                    gap_count[slot] = 0
                elif active[slot]:
                    gap_count[slot] += 1
                    if gap_count[slot] > args.gap_tolerance:
                        active[slot] = False

                if not active[slot]:
                    continue

                buf, ok = build_buffer(slot_kp_buf[slot], fi, SEQ_LEN)
                if not ok:
                    continue

                feat = buf[:, :, :2].reshape(SEQ_LEN, -1)
                with torch.no_grad():
                    probs = torch.softmax(
                        gru(torch.FloatTensor(feat).unsqueeze(0)), dim=1
                    )[0].numpy()
                total_gru_calls += 1
                pred = int(np.argmax(probs))
                conf = float(probs[pred])

                if conf >= args.min_conf:
                    raw_events.append({
                        "frame": fi,
                        "time": round(fi / fps, 2),
                        "slot": slot,
                        "slot_name": SLOT_NAMES[slot],
                        "label": GRU_LABELS[pred],
                        "conf": round(conf, 3),
                    })

        # Merge same-action events
        merged = merge_actions(raw_events, fps, args.merge_window)

        # Assign ball side
        for action in merged:
            assign_action_side(action, ball_det, ball_pred, mapper)

        rally_entry = {
            "rally_index": ri,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_time": round(start_frame / fps, 2),
            "end_time": round(end_frame / fps, 2),
            "actions": merged,
            "raw_event_count": len(raw_events),
        }
        all_rally_actions.append(rally_entry)

        if (ri + 1) % 5 == 0 or ri == len(rallies) - 1:
            print(f"  Rally {ri+1}/{len(rallies)}: "
                  f"{len(merged)} actions from {len(raw_events)} raw events")

    pose.close()

    # ---- Summary ----
    elapsed = time.time() - t0
    total_actions = sum(len(r["actions"]) for r in all_rally_actions)
    print(f"\nDone in {elapsed:.0f}s")
    print(f"  Pose frames: {total_pose_frames}")
    print(f"  GRU calls: {total_gru_calls}")
    print(f"  Total actions: {total_actions} across {len(rallies)} rallies")

    # Per-label breakdown
    from collections import Counter
    label_counts = Counter()
    for r in all_rally_actions:
        for a in r["actions"]:
            label_counts[a["label"]] += 1
    for label in ["backhand", "forehand", "serve"]:
        if label_counts[label]:
            print(f"    {label}: {label_counts[label]}")

    # ---- Save ----
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    output = {
        "params": {
            "speed_threshold": args.speed_threshold,
            "min_conf": args.min_conf,
            "seq_len": args.seq_len,
            "merge_window": args.merge_window,
        },
        "stats": {
            "fps": fps,
            "total_rallies": len(rallies),
            "pose_frames": total_pose_frames,
            "gru_calls": total_gru_calls,
            "total_actions": total_actions,
        },
        "rallies": all_rally_actions,
    }
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
