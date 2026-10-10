#!/usr/bin/env python3
"""Multi-signal serve detection (v6).

Pipeline:
  1. Load player detections with raw keypoints (from detect_players.py)
  2. Slot mapping via homography (auto-detect singles/doubles)
  3. Scene readiness: 3+ slots stable for 0.5s+
  4. Hand-above-head trigger on baseline players
  5. Track to peak (last highest wrist) → asymmetric 15-frame window @15fps
  6. v4 normalization → GRU v6 confirmation
  7. Post-filtering: net distance, ball toss, frame edge

No on-demand pose extraction — all keypoints come from Step 3.

Usage:
    python -u scripts/detect_serves.py \
        --players results/sample3/player_detections.json \
        --calib results/sample3/calib.json \
        --ball results/sample3/ball_positions.json \
        --out results/sample3/serve_events.json \
        --video samples/sample3.mp4
"""

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
from tennisvision.action.gru_classifier import StrokeGRU
from tennisvision.action.serve_traj import check_serve_toss, compute_net_direction

TARGET_FPS = 15
SEQ_LEN = 15
BEFORE_PEAK = 9
AFTER_PEAK = 5
WRIST = [9, 10]
NOSE = 0


# =====================================================================
# Auto-detect singles vs doubles
# =====================================================================

def detect_match_type(frames, mapper, sample_size=3000):
    slot_fill_counts = []
    for fd in frames[:sample_size]:
        slot_dets = mapper.assign(fd["detections"])
        slot_fill_counts.append(len(slot_dets))
    if not slot_fill_counts:
        return 4
    busy = [c for c in slot_fill_counts if c >= 2]
    if not busy:
        return 4
    p75 = sorted(busy)[int(len(busy) * 0.75)]
    return 2 if p75 <= 2 else 4


# =====================================================================
# Scene readiness
# =====================================================================

class SceneReadiness:
    def __init__(self, ready_frames, min_slots=3):
        self.ready_frames = ready_frames
        self.min_slots = min_slots
        self.slot_foot = {s: [] for s in range(4)}

    def update(self, fi, slot_dets):
        for slot, det in slot_dets.items():
            self.slot_foot[slot].append((fi, det["foot_x"], det["foot_y"]))
            cutoff = fi - self.ready_frames - 30
            while self.slot_foot[slot] and self.slot_foot[slot][0][0] < cutoff:
                self.slot_foot[slot].pop(0)

    def is_ready(self, fi):
        check_start = fi - self.ready_frames
        if check_start < 0:
            return False
        stable = 0
        for s in range(4):
            hist = [(f, fx, fy) for f, fx, fy in self.slot_foot[s]
                    if check_start <= f <= fi]
            if len(hist) < self.ready_frames * 0.8:
                continue
            fxs = [fx for _, fx, _ in hist]
            fys = [fy for _, _, fy in hist]
            if np.std(fxs) < 30 and np.std(fys) < 20:
                stable += 1
        return stable >= self.min_slots


# =====================================================================
# Normalization
# =====================================================================

def normalize_kp(kp_x, kp_y, kp_v):
    """v4-style normalization: x/bw, y/bh (stretch to square)."""
    kp_x, kp_y, kp_v = np.array(kp_x), np.array(kp_y), np.array(kp_v)
    valid = kp_v >= 0.3
    if valid.sum() < 3:
        return None
    xn, xx = kp_x[valid].min(), kp_x[valid].max()
    yn, yx = kp_y[valid].min(), kp_y[valid].max()
    bw, bh = max(xx - xn, 1), max(yx - yn, 1)
    norm_x = (kp_x - xn) / bw
    norm_y = (kp_y - yn) / bh
    return np.stack([norm_x, norm_y], axis=1).astype(np.float32)


# =====================================================================
# Serve detector state machine
# =====================================================================

class ServeDetector:
    """Hand-above-head trigger → track peak → emit candidate.

    Per-slot states: IDLE → TRACKING → peak found → emit → COOLDOWN → IDLE
    """

    def __init__(self, fps, cooldown_seconds=2.0):
        self.fps = fps
        self.cooldown_frames = int(round(cooldown_seconds * fps))
        self.ds_interval = max(1, int(round(fps / TARGET_FPS)))
        self.state = {s: "IDLE" for s in range(4)}
        self.tracking_start = {s: 0 for s in range(4)}
        self.min_wrist_y = {s: float('inf') for s in range(4)}
        self.peak_frame = {s: 0 for s in range(4)}
        self.cooldown_until = {s: 0 for s in range(4)}
        # Raw keypoint buffer: (fi, kp_x, kp_y, kp_v)
        self.raw_buf = {s: [] for s in range(4)}
        self.RAW_BUF_MAX = int(round(3 * fps))

    def _is_above_head(self, kp_x, kp_y, kp_v):
        if kp_v[NOSE] < 0.3:
            return False
        nose_y = kp_y[NOSE]
        for w in WRIST:
            if kp_v[w] > 0.3 and kp_y[w] < nose_y:
                return True
        return False

    def _get_wrist_min_y(self, kp_y, kp_v):
        min_y = float('inf')
        for w in WRIST:
            if kp_v[w] > 0.3 and kp_y[w] < min_y:
                min_y = kp_y[w]
        return min_y

    def update(self, fi, slot, kp_x, kp_y, kp_v):
        """Returns peak_frame if serve candidate detected, else None."""
        self.raw_buf[slot].append((fi, kp_x, kp_y, kp_v))
        if len(self.raw_buf[slot]) > self.RAW_BUF_MAX:
            self.raw_buf[slot] = self.raw_buf[slot][-self.RAW_BUF_MAX:]

        state = self.state[slot]
        above = self._is_above_head(kp_x, kp_y, kp_v)

        if state == "IDLE":
            if fi < self.cooldown_until[slot]:
                return None
            if above:
                self.state[slot] = "TRACKING"
                self.tracking_start[slot] = fi
                self.min_wrist_y[slot] = self._get_wrist_min_y(kp_y, kp_v)
                self.peak_frame[slot] = fi
            return None

        elif state == "TRACKING":
            if above:
                wy = self._get_wrist_min_y(kp_y, kp_v)
                if wy < self.min_wrist_y[slot]:
                    self.min_wrist_y[slot] = wy
                    self.peak_frame[slot] = fi
                if fi - self.tracking_start[slot] > 2 * self.fps:
                    self.state[slot] = "IDLE"
                return None
            else:
                peak = self.peak_frame[slot]
                self.state[slot] = "IDLE"
                self.cooldown_until[slot] = fi + self.cooldown_frames
                return peak

        return None

    def build_window(self, slot, peak_frame):
        """Build 15-frame window around peak, downsampled to 15fps.

        Returns (np.array of shape (15, 17, 2), ok).
        """
        buf = self.raw_buf[slot]
        if not buf:
            return None, False

        before_frames = int(BEFORE_PEAK * self.ds_interval)
        after_frames = int(AFTER_PEAK * self.ds_interval)
        window_start = peak_frame - before_frames
        window_end = peak_frame + after_frames

        # Normalize and collect available frames
        frame_map = {}
        for fi, kx, ky, kv in buf:
            if window_start <= fi <= window_end:
                kp_norm = normalize_kp(kx, ky, kv)
                if kp_norm is not None:
                    frame_map[fi] = kp_norm

        if len(frame_map) < 5:
            return None, False

        # Sample SEQ_LEN evenly spaced frames
        sample_frames = np.round(np.linspace(window_start, window_end, SEQ_LEN)).astype(int)
        sorted_keys = sorted(frame_map.keys())

        result = []
        for target in sample_frames:
            if target in frame_map:
                result.append(frame_map[target])
            else:
                best_fi = min(sorted_keys, key=lambda f: abs(f - target))
                if abs(best_fi - target) <= self.ds_interval * 2:
                    result.append(frame_map[best_fi])
                else:
                    before = [f for f in sorted_keys if f <= target]
                    after = [f for f in sorted_keys if f >= target]
                    if before and after:
                        bf, af = before[-1], after[0]
                        alpha = (target - bf) / max(af - bf, 1)
                        result.append(frame_map[bf] * (1 - alpha) + frame_map[af] * alpha)
                    elif before:
                        result.append(frame_map[before[-1]])
                    elif after:
                        result.append(frame_map[after[0]])
                    else:
                        return None, False

        return np.array(result, dtype=np.float32), True

    def reset_slot(self, slot):
        self.state[slot] = "IDLE"
        self.raw_buf[slot].clear()


# =====================================================================
# Serve review video
# =====================================================================

def write_serve_review(src_video, serves, fps, dst_video, pad_seconds=0.5):
    cap = cv2.VideoCapture(src_video)
    if not cap.isOpened():
        raise SystemExit("cannot open " + src_video)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pad = int(round(pad_seconds * fps))

    os.makedirs(os.path.dirname(dst_video) or ".", exist_ok=True)
    writer = cv2.VideoWriter(dst_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not writer.isOpened():
        cap.release()
        raise SystemExit("cannot open writer for " + dst_video)

    clips = [(i, ev, max(0, ev["start_frame"] - pad), min(total - 1, ev["end_frame"] + pad))
             for i, ev in enumerate(serves)]

    max_needed = max(ce for _, _, _, ce in clips)
    clip_idx, fi = 0, 0
    while fi <= max_needed and clip_idx < len(clips):
        ret, frame = cap.read()
        if not ret:
            break
        while clip_idx < len(clips) and fi > clips[clip_idx][3]:
            clip_idx += 1
        if clip_idx >= len(clips):
            break
        idx, ev, cs, ce = clips[clip_idx]
        if cs <= fi <= ce:
            label = "Serve %d/%d  %s  conf=%.2f  t=%.1fs" % (
                idx + 1, len(serves), ev["slot_name"], ev["conf"], fi / fps)
            cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            writer.write(frame)
            if fi == ce:
                clip_idx += 1
        fi += 1

    cap.release()
    writer.release()
    print(f"  Saved: {dst_video} ({len(clips)} clips)")


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--players", required=True)
    parser.add_argument("--calib", required=True)
    parser.add_argument("--gru", default="weights/stroke_gru_v6_best.pt")
    parser.add_argument("--ball", default=None)
    parser.add_argument("--video", default=None, help="For serve_review.mp4 generation")
    parser.add_argument("--out", required=True)
    parser.add_argument("--ready-duration", type=float, default=0.5)
    parser.add_argument("--serve-threshold", type=float, default=0.8)
    parser.add_argument("--min-net-dist", type=float, default=8.0)
    args = parser.parse_args()

    # ---- Load inputs ----
    with open(args.calib) as f:
        calib = json.load(f)
    H_img_to_real = np.array(calib["H_img_to_real"], dtype=np.float64)
    mapper = SlotMapper(H_img_to_real)
    net_dir = compute_net_direction(H_img_to_real)

    with open(args.players) as f:
        data = json.load(f)
    fps = data["fps"]
    frames = data["frames"]
    img_w = data.get("width", 1920)
    print(f"Players: {data['total_frames']} frames, fps={fps:.1f}")

    ball_det, ball_pred = {}, {}
    if args.ball:
        with open(args.ball) as f:
            ball_data = json.load(f)
        ball_det = ball_data.get("detected", {})
        ball_pred = ball_data.get("predicted", {})
        print(f"Ball: {len(ball_det)} detected, {len(ball_pred)} predicted")

    # ---- Auto-detect singles/doubles ----
    n_players = detect_match_type(frames, mapper)
    min_ready_slots = max(n_players - 1, 2)
    print(f"Match: {'doubles' if n_players == 4 else 'singles'} "
          f"(need {min_ready_slots}+ slots)")

    # ---- Load GRU v6 ----
    print(f"GRU: {args.gru}")
    gru = StrokeGRU(n_classes=4)
    gru.load_state_dict(torch.load(args.gru, map_location="cpu"))
    gru.eval()

    # ---- Processing ----
    READY_FRAMES = int(round(args.ready_duration * fps))
    readiness = SceneReadiness(READY_FRAMES, min_slots=min_ready_slots)
    detector = ServeDetector(fps)

    scene_was_ready = False
    serve_candidates = []
    gru_count = 0
    trigger_count = 0
    t0 = time.time()

    for fd in frames:
        fi = fd["frame"]
        slot_dets = mapper.assign(fd["detections"])
        readiness.update(fi, slot_dets)
        ready = readiness.is_ready(fi)

        if not ready:
            if scene_was_ready:
                for s in range(4):
                    detector.reset_slot(s)
            scene_was_ready = False
            continue

        if not scene_was_ready:
            for s in range(4):
                detector.reset_slot(s)
        scene_was_ready = True

        # Process each slot's keypoints
        for slot in sorted(slot_dets.keys()):
            det = slot_dets[slot]
            if "kp_x" not in det:
                continue

            kp_x = np.array(det["kp_x"])
            kp_y = np.array(det["kp_y"])
            kp_v = np.array(det["kp_v"])

            peak = detector.update(fi, slot, kp_x, kp_y, kp_v)
            if peak is None:
                continue

            trigger_count += 1
            window, ok = detector.build_window(slot, peak)
            if not ok:
                continue

            feat = window[:, :, :2].reshape(SEQ_LEN, -1)
            with torch.no_grad():
                probs = torch.softmax(
                    gru(torch.FloatTensor(feat).unsqueeze(0)), dim=1
                )[0].numpy()
            gru_count += 1

            serve_p = float(probs[2])
            if serve_p > args.serve_threshold:
                ds = detector.ds_interval
                serve_candidates.append({
                    "peak_frame": peak,
                    "start_frame": peak - int(BEFORE_PEAK * ds),
                    "end_frame": peak + int(AFTER_PEAK * ds),
                    "start_time": round((peak - int(BEFORE_PEAK * ds)) / fps, 2),
                    "end_time": round((peak + int(AFTER_PEAK * ds)) / fps, 2),
                    "slot": slot,
                    "slot_name": SLOT_NAMES[slot],
                    "conf": round(serve_p, 3),
                })

        if fi > 0 and fi % 5000 == 0:
            e = time.time() - t0
            print(f"  frame {fi}/{len(frames)}  {fi/e:.1f} fps  "
                  f"triggers={trigger_count}  gru={gru_count}  "
                  f"candidates={len(serve_candidates)}")

    # ---- Post-filtering ----
    def _get_ball(frame_idx):
        k = str(frame_idx)
        return ball_det.get(k) or ball_pred.get(k)

    def _filter_serve(ev):
        mid = ev["peak_frame"]
        if mid >= len(frames):
            return True, ""
        fd = frames[mid]
        slot = ev["slot"]
        slot_dets = mapper.assign(fd["detections"])
        if slot not in slot_dets:
            return True, ""
        det = slot_dets[slot]
        foot_x, foot_y = det["foot_x"], det["foot_y"]

        rx, ry = mapper.to_court(foot_x, foot_y)
        if rx is not None and ry is not None:
            if abs(ry - mapper.net_y_m) < args.min_net_dist:
                return False, f"too close to net ({abs(ry - mapper.net_y_m):.1f}m)"
        if rx is not None:
            if rx < -1.0 or rx > mapper.cfg.court_width_m + 1.0:
                return False, f"outside sideline (rx={rx:.1f}m)"

        bbox = det["bbox"]
        if bbox[0] <= 1 or bbox[2] >= img_w - 1:
            return False, "frame edge"

        if ball_det or ball_pred:
            search_start = max(0, ev["start_frame"])
            search_end = ev["end_frame"] + 1
            min_x_diff = float("inf")
            for check_fi in range(search_start, search_end):
                bp = _get_ball(check_fi)
                if bp is not None:
                    diff = abs(bp[0] - foot_x)
                    if diff < min_x_diff:
                        min_x_diff = diff
            if min_x_diff < float("inf"):
                bbox_w = bbox[2] - bbox[0]
                if min_x_diff > max(bbox_w * 5, 300):
                    return False, f"ball x too far ({min_x_diff:.0f}px)"

        if ball_det or ball_pred:
            toss_ok, reason = check_serve_toss(
                _get_ball, ev["start_frame"], ev["end_frame"],
                foot_x, foot_y, bbox, fps, net_dir=net_dir)
            if not toss_ok:
                return False, reason

        return True, ""

    filtered = []
    for ev in serve_candidates:
        keep, reason = _filter_serve(ev)
        if keep:
            filtered.append(ev)
        else:
            print(f"  Filtered: {ev['start_time']}s {ev['slot_name']} - {reason}")

    # ---- Results ----
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")
    print(f"  Hand-above-head triggers: {trigger_count}")
    print(f"  GRU calls: {gru_count}")
    print(f"  Candidates (pre-filter): {len(serve_candidates)}")
    print(f"  Serves (post-filter): {len(filtered)}")

    from collections import Counter
    slots = Counter(s["slot_name"] for s in filtered)
    for slot in ["NEAR_L", "NEAR_R", "FAR_L", "FAR_R"]:
        if slots[slot]:
            print(f"    {slot}: {slots[slot]}")

    for s in filtered:
        print(f"  {s['start_time']}s-{s['end_time']}s {s['slot_name']} conf={s['conf']}")

    # ---- Save ----
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    output = {
        "match_type": "doubles" if n_players == 4 else "singles",
        "params": {
            "ready_duration": args.ready_duration,
            "serve_threshold": args.serve_threshold,
            "min_net_dist": args.min_net_dist,
            "target_fps": TARGET_FPS,
            "seq_len": SEQ_LEN,
        },
        "stats": {
            "total_frames": len(frames),
            "fps": fps,
            "triggers": trigger_count,
            "gru_calls": gru_count,
        },
        "serve_events": filtered,
    }
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {args.out}")

    if args.video and filtered:
        review_path = os.path.join(os.path.dirname(args.out) or ".", "serve_review.mp4")
        print(f"\nGenerating serve review: {review_path}")
        write_serve_review(args.video, filtered, fps, review_path)


if __name__ == "__main__":
    main()
