#!/usr/bin/env python3
"""Multi-signal serve detection.

Pipeline:
  1. Load player detections (bbox only, from detect_players.py)
  2. Slot mapping via homography (auto-detect singles/doubles)
  3. Scene readiness: 3+ slots stable for 0.5s+
  4. On-demand keypoint extraction (YOLO-pose, only during ready windows)
  5. Wrist speed trigger → GRU classification
  6. Post-merge filtering: net distance, ball toss, frame edge

Inputs:
  --players   player_detections.json (Step 3)
  --calib     calib.json (Step 1)
  --ball      ball_positions.json (Step 2, optional)
  --video     original video (for on-demand pose extraction + serve review)

Usage:
    python -u scripts/detect_serves.py \
        --players results/sample3/player_detections.json \
        --calib results/sample3/calib.json \
        --ball results/sample3/ball_positions.json \
        --video samples/sample3.mp4 \
        --out results/sample3/serve_events.json
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
from tennisvision.action.gru_classifier import StrokeGRU, LABELS as GRU_LABELS
from tennisvision.action.serve_traj import check_serve_toss, compute_net_direction


# =====================================================================
# Auto-detect singles vs doubles
# =====================================================================

def detect_match_type(frames, mapper, sample_size=3000):
    """Scan initial frames to determine singles (2) or doubles (4)."""
    slot_fill_counts = []
    for fd in frames[:sample_size]:
        slot_dets = mapper.assign(fd["detections"])
        slot_fill_counts.append(len(slot_dets))
    if not slot_fill_counts:
        return 4  # default doubles
    # Look at the 75th percentile of slot fill during "busy" frames
    busy = [c for c in slot_fill_counts if c >= 2]
    if not busy:
        return 4
    p75 = sorted(busy)[int(len(busy) * 0.75)]
    match_type = 2 if p75 <= 2 else 4
    return match_type


# =====================================================================
# Scene readiness
# =====================================================================

class SceneReadiness:
    """Track slot occupancy and position stability."""

    def __init__(self, ready_frames, min_slots=3):
        self.ready_frames = ready_frames
        self.min_slots = min_slots
        self.slot_foot = {s: [] for s in range(4)}  # (fi, fx, fy)

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
# On-demand keypoint extraction
# =====================================================================

class PoseExtractor:
    """Extract keypoints on demand from video crops."""

    def __init__(self, video_path, device, quantize, no_half=False):
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
        # Sequential reads are fine since we process in order
        if fi != self._last_fi + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = self.cap.read()
        self._last_fi = fi
        self._last_frame = frame if ret else None
        return self._last_frame

    def extract(self, fi, detections):
        """Extract keypoints for detections in frame fi.

        Returns list of dicts with kp_norm_x, kp_norm_y added.
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
            scale = max(bw, bh)
            det_out = dict(detections[i])
            det_out["kp_norm_x"] = [round(float((kp_x[k] - xn) / scale), 4) for k in range(17)]
            det_out["kp_norm_y"] = [round(float((kp_y[k] - yn) / scale), 4) for k in range(17)]
            results.append(det_out)

        return results

    def close(self):
        self.cap.release()


# =====================================================================
# Buffer with interpolation
# =====================================================================

def build_buffer(slot_buf, fi, seq_len, max_missing_ratio=0.2):
    """Build SEQ_LEN buffer with interpolation for small gaps.

    slot_buf: list of (frame_idx, kp_array)
    Returns (np.array, ok).
    """
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
    writer = cv2.VideoWriter(dst_video, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (W, H))
    if not writer.isOpened():
        cap.release()
        raise SystemExit("cannot open writer for " + dst_video)

    clips = []
    for i, ev in enumerate(serves):
        clip_start = max(0, ev["start_frame"] - pad)
        clip_end = min(total - 1, ev["end_frame"] + pad)
        clips.append((i, ev, clip_start, clip_end))

    max_needed = max(ce for _, _, _, ce in clips)
    clip_idx = 0
    fi = 0
    while fi <= max_needed and clip_idx < len(clips):
        ret, frame = cap.read()
        if not ret:
            break
        while clip_idx < len(clips) and fi > clips[clip_idx][3]:
            clip_idx += 1
        if clip_idx >= len(clips):
            break
        idx, ev, clip_start, clip_end = clips[clip_idx]
        if clip_start <= fi <= clip_end:
            label = "Serve %d/%d  %s  conf=%.2f  t=%.1fs" % (
                idx + 1, len(serves), ev["slot_name"], ev["conf"], fi / fps)
            cv2.putText(frame, label, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            writer.write(frame)
            if fi == clip_end:
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
    parser.add_argument("--players", required=True,
                        help="player_detections.json from detect_players.py")
    parser.add_argument("--calib", required=True, help="calib.json")
    parser.add_argument("--video", required=True,
                        help="Original video (for on-demand pose extraction)")
    parser.add_argument("--gru", default="weights/stroke_gru_v5_best.pt")
    parser.add_argument("--ball", default=None,
                        help="ball_positions.json for toss filtering")
    parser.add_argument("--out", required=True)
    # Thresholds (seconds at API surface)
    parser.add_argument("--ready-duration", type=float, default=0.5,
                        help="Seconds of scene stability for readiness")
    parser.add_argument("--speed-threshold", type=float, default=0.15)
    parser.add_argument("--serve-threshold", type=float, default=0.8)
    parser.add_argument("--gap-tolerance", type=int, default=5)
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--min-net-dist", type=float, default=8.0,
                        help="Min distance from net (metres) to accept serve")
    parser.add_argument("--no-half", action="store_true", help="Disable FP16")
    args = parser.parse_args()

    # ---- Load inputs ----
    with open(args.calib) as f:
        calib = json.load(f)
    H_img_to_real = np.array(calib["H_img_to_real"], dtype=np.float64)
    mapper = SlotMapper(H_img_to_real)
    net_dir = compute_net_direction(H_img_to_real)
    print(f"Calibration: {args.calib}")

    print(f"Loading players: {args.players}")
    with open(args.players) as f:
        data = json.load(f)
    fps = data["fps"]
    frames = data["frames"]
    img_w = data.get("width", 1920)
    print(f"  {data['total_frames']} frames, fps={fps:.1f}")

    ball_det, ball_pred = {}, {}
    if args.ball:
        print(f"Loading ball: {args.ball}")
        with open(args.ball) as f:
            ball_data = json.load(f)
        ball_det = ball_data.get("detected", {})
        ball_pred = ball_data.get("predicted", {})
        print(f"  {len(ball_det)} detected, {len(ball_pred)} predicted")

    # ---- Auto-detect singles/doubles ----
    n_players = detect_match_type(frames, mapper)
    min_ready_slots = max(n_players - 1, 2)  # 3 for doubles, 2 for singles
    print(f"Match type: {'doubles' if n_players == 4 else 'singles'} "
          f"({n_players} players, need {min_ready_slots}+ slots for readiness)")

    # ---- Load GRU ----
    print(f"Loading GRU: {args.gru}")
    gru = StrokeGRU(n_classes=4)
    gru.load_state_dict(torch.load(args.gru, map_location="cpu"))
    gru.eval()

    # ---- Init pose extractor ----
    device = 0 if torch.cuda.is_available() else "cpu"
    quantize = "fp16" if (torch.cuda.is_available() and not args.no_half) else None
    pose = PoseExtractor(args.video, device, quantize, args.no_half)

    # ---- Processing ----
    SEQ_LEN = args.seq_len
    READY_FRAMES = int(round(args.ready_duration * fps))
    readiness = SceneReadiness(READY_FRAMES, min_slots=min_ready_slots)

    # Per-slot state (only active during ready periods)
    slot_kp_buf = {s: [] for s in range(4)}  # (fi, kp_array)
    BUF_MAX = SEQ_LEN + 30
    prev_kps = {s: None for s in range(4)}
    active = {s: False for s in range(4)}
    gap_count = {s: 0 for s in range(4)}
    scene_was_ready = False

    serve_hits = []
    all_events = []
    gru_count = 0
    pose_frames = 0
    t0 = time.time()

    for fd in frames:
        fi = fd["frame"]
        slot_dets = mapper.assign(fd["detections"])

        # Always update readiness tracker
        readiness.update(fi, slot_dets)

        ready = readiness.is_ready(fi)
        if not ready:
            if scene_was_ready:
                # Transition ready → not ready: reset buffers
                for s in range(4):
                    slot_kp_buf[s].clear()
                    prev_kps[s] = None
                    active[s] = False
                    gap_count[s] = 0
            scene_was_ready = False
            continue

        if not scene_was_ready:
            # Transition not ready → ready: fresh start
            for s in range(4):
                slot_kp_buf[s].clear()
                prev_kps[s] = None
                active[s] = False
                gap_count[s] = 0
        scene_was_ready = True

        # ---- On-demand pose extraction ----
        # Only extract for detections assigned to slots
        slot_det_list = []
        for slot in sorted(slot_dets.keys()):
            slot_det_list.append(slot_dets[slot])

        if slot_det_list:
            kp_results = pose.extract(fi, slot_det_list)
            pose_frames += 1
            # Re-map results back to slots by tid
            kp_by_tid = {r["tid"]: r for r in kp_results}
        else:
            kp_by_tid = {}

        # ---- Per-slot GRU processing ----
        for slot in range(4):
            if slot not in slot_dets:
                if active[slot]:
                    gap_count[slot] += 1
                    if gap_count[slot] > args.gap_tolerance:
                        active[slot] = False
                continue

            det = slot_dets[slot]
            tid = det["tid"]

            # Check if pose was extracted for this detection
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
            gru_count += 1
            pred = int(np.argmax(probs))

            event = {
                "frame": fi,
                "time": round(fi / fps, 2),
                "slot": slot,
                "slot_name": SLOT_NAMES[slot],
                "label": GRU_LABELS[pred],
                "conf": round(float(probs[pred]), 3),
                "probs": {GRU_LABELS[i]: round(float(probs[i]), 3) for i in range(4)},
            }
            all_events.append(event)

            if probs[2] > args.serve_threshold:
                serve_hits.append((fi, slot, float(probs[2])))

        if fi > 0 and fi % 1000 == 0:
            e = time.time() - t0
            print(f"  frame {fi}/{len(frames)}  {fi/e:.1f} fps  "
                  f"pose={pose_frames}  gru={gru_count}")

    pose.close()

    # ---- Merge consecutive serve hits ----
    merged_serves = []
    if serve_hits:
        cs, ce, cslot, cconf = (
            serve_hits[0][0], serve_hits[0][0], serve_hits[0][1], serve_hits[0][2],
        )
        for fi, slot, conf in serve_hits[1:]:
            if slot == cslot and fi - ce <= 10:
                ce = fi
                cconf = max(cconf, conf)
            else:
                merged_serves.append({
                    "start_frame": cs, "end_frame": ce,
                    "start_time": round(cs / fps, 2),
                    "end_time": round(ce / fps, 2),
                    "slot": cslot, "slot_name": SLOT_NAMES[cslot],
                    "conf": round(cconf, 3),
                })
                cs, ce, cslot, cconf = fi, fi, slot, conf
        merged_serves.append({
            "start_frame": cs, "end_frame": ce,
            "start_time": round(cs / fps, 2),
            "end_time": round(ce / fps, 2),
            "slot": cslot, "slot_name": SLOT_NAMES[cslot],
            "conf": round(cconf, 3),
        })

    # ---- Post-merge filtering ----
    def _get_ball(frame_idx):
        k = str(frame_idx)
        return ball_det.get(k) or ball_pred.get(k)

    def _filter_serve(ev):
        mid_frame = (ev["start_frame"] + ev["end_frame"]) // 2
        if mid_frame >= len(frames):
            return True, ""
        fd = frames[mid_frame]
        slot = ev["slot"]

        slot_dets = mapper.assign(fd["detections"])
        if slot not in slot_dets:
            return True, ""
        det = slot_dets[slot]
        foot_x, foot_y = det["foot_x"], det["foot_y"]

        # 1. Net proximity
        rx, ry = mapper.to_court(foot_x, foot_y)
        if rx is not None and ry is not None:
            dist_net = abs(ry - mapper.net_y_m)
            if dist_net < args.min_net_dist:
                return False, f"too close to net ({dist_net:.1f}m)"

        # 2. Sideline
        if rx is not None:
            if rx < -1.0 or rx > mapper.cfg.court_width_m + 1.0:
                return False, f"outside sideline (rx={rx:.1f}m)"

        # 3. Frame edge
        bbox = det["bbox"]
        if bbox[0] <= 1 or bbox[2] >= img_w - 1:
            return False, "frame edge (incomplete body)"

        # 4. Ball x-proximity
        if ball_det or ball_pred:
            search_start = max(0, ev["start_frame"] - int(fps))
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
                    return False, f"ball x too far (min_diff={min_x_diff:.0f}px)"

        # 5. Ball toss
        if ball_det or ball_pred:
            toss_ok, toss_reason = check_serve_toss(
                _get_ball, ev["start_frame"], ev["end_frame"],
                foot_x, foot_y, bbox, fps, net_dir=net_dir,
            )
            if not toss_ok:
                return False, toss_reason

        return True, ""

    filtered_serves = []
    for ev in merged_serves:
        keep, reason = _filter_serve(ev)
        if keep:
            filtered_serves.append(ev)
        else:
            print(f"  Filtered: {ev['start_time']}s {ev['slot_name']} - {reason}")
    merged_serves = filtered_serves

    # ---- Print results ----
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")
    print(f"  Pose frames: {pose_frames}/{len(frames)} "
          f"({pose_frames/max(len(frames),1)*100:.0f}%)")
    print(f"  GRU invocations: {gru_count}")
    print(f"  Total events: {len(all_events)}")
    print(f"    backhand: {sum(1 for e in all_events if e['label'] == 'backhand')}")
    print(f"    forehand: {sum(1 for e in all_events if e['label'] == 'forehand')}")
    print(f"    serve:    {sum(1 for e in all_events if e['label'] == 'serve')}")
    print(f"    background: {sum(1 for e in all_events if e['label'] == 'background')}")

    print(f"\nServe events (merged): {len(merged_serves)}")
    for s in merged_serves:
        print(f"  {s['start_time']}s-{s['end_time']}s {s['slot_name']} conf={s['conf']}")

    # ---- Save ----
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    output = {
        "match_type": "doubles" if n_players == 4 else "singles",
        "params": {
            "ready_duration": args.ready_duration,
            "speed_threshold": args.speed_threshold,
            "serve_threshold": args.serve_threshold,
            "gap_tolerance": args.gap_tolerance,
            "seq_len": args.seq_len,
            "min_net_dist": args.min_net_dist,
        },
        "stats": {
            "total_frames": len(frames),
            "fps": fps,
            "pose_frames": pose_frames,
            "gru_invocations": gru_count,
            "total_events": len(all_events),
        },
        "serve_events": merged_serves,
        "all_events": all_events,
    }
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {args.out}")

    # ---- Serve review video ----
    if merged_serves:
        review_path = os.path.join(os.path.dirname(args.out) or ".", "serve_review.mp4")
        print(f"\nGenerating serve review: {review_path}")
        write_serve_review(args.video, merged_serves, fps, review_path)


if __name__ == "__main__":
    main()
