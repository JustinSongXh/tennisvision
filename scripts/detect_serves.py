"""Serve detection using GRU stroke classifier on pre-extracted keypoints.

Pipeline:
  1. Load keypoints JSON (yolo11m + yolo26s-pose output)
  2. Homography on-court filter (from --calib or hardcoded fallback)
  3. 4-slot spatial mapping (NEAR_L/R, FAR_L/R)
  4. Wrist speed trigger (threshold=0.15)
  5. GRU v4 inference during active periods (every frame)
  6. Serve marking (prob > 0.8, merge consecutive frames)

Usage:
    # With calibration file (recommended)
    python -u scripts/detect_serves.py \
        --keypoints results/sample2/keypoints.json \
        --calib results/sample2/calib.json \
        --gru weights/stroke_gru_v4_best.pt \
        --out results/sample2/serve_events.json

    # Without calib (uses hardcoded fallback for sample_short.mp4)
    python -u scripts/detect_serves.py \
        --keypoints results/sample_short/keypoints.json \
        --gru weights/stroke_gru_v4_best.pt \
        --out results/sample_short/serve_events.json
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tennisvision.action.slot_mapper import SlotMapper, SLOT_NAMES
from tennisvision.action.gru_classifier import StrokeGRU, LABELS as GRU_LABELS
from tennisvision.action.serve_traj import check_serve_toss, compute_net_direction

# ---- Hardcoded fallback (sample_short.mp4) ----
_DEFAULT_H_IMG_TO_REAL = np.array([
    [-0.004212704126665825, -0.00958000348568168, 8.066176934954633],
    [9.370094579164494e-05, 0.01803202259785551, -15.976691289361806],
    [1.1053780957617317e-07, -0.0019774218879326623, 1.0],
], dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--keypoints", required=True)
    parser.add_argument("--gru", default="weights/stroke_gru_v4_best.pt")
    parser.add_argument("--calib", default=None,
                        help="calib.json with H_img_to_real (omit to use hardcoded fallback)")
    parser.add_argument("--video", default=None,
                        help="Original video for serve_review.mp4 clip generation")
    parser.add_argument("--out", required=True)
    parser.add_argument("--speed-threshold", type=float, default=0.15)
    parser.add_argument("--serve-threshold", type=float, default=0.8)
    parser.add_argument("--gap-tolerance", type=int, default=5)
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--serve-far-only", action="store_true", default=False,
                        help="Only accept serve from FAR slots (baseline)")
    parser.add_argument("--ball", default=None,
                        help="ball_positions.json for ball-player distance filter")
    parser.add_argument("--min-net-dist", type=float, default=8.0,
                        help="Min distance from net (metres) to accept serve")
    args = parser.parse_args()

    # Load homography
    if args.calib:
        with open(args.calib) as f:
            calib = json.load(f)
        H_img_to_real = np.array(calib["H_img_to_real"], dtype=np.float64)
        print(f"Calibration: {args.calib}")
    else:
        H_img_to_real = _DEFAULT_H_IMG_TO_REAL
        print("Calibration: hardcoded fallback (sample_short.mp4)")
    mapper = SlotMapper(H_img_to_real)
    net_dir = compute_net_direction(H_img_to_real)

    # Load data
    print(f"Loading keypoints: {args.keypoints}")
    with open(args.keypoints) as f:
        data = json.load(f)
    print(f"  {data['total_frames']} frames, fps={data['fps']:.1f}")

    # Load ball positions (optional)
    ball_det, ball_pred = {}, {}
    if args.ball:
        print(f"Loading ball: {args.ball}")
        with open(args.ball) as f:
            ball_data = json.load(f)
        ball_det = ball_data.get("detected", {})
        ball_pred = ball_data.get("predicted", {})
        print(f"  {len(ball_det)} detected, {len(ball_pred)} predicted")

    img_w = data.get("width", 1920)

    # Load GRU
    print(f"Loading GRU: {args.gru}")
    gru = StrokeGRU(n_classes=4)
    gru.load_state_dict(torch.load(args.gru, map_location="cpu"))
    gru.eval()

    SEQ_LEN = args.seq_len

    # Per-slot state
    kp_buffers = {s: [] for s in range(4)}
    prev_kps = {s: None for s in range(4)}
    active = {s: False for s in range(4)}
    gap_count = {s: 0 for s in range(4)}

    serve_hits = []
    all_events = []
    gru_count = 0

    for fd in data["frames"]:
        fi = fd["frame"]
        slot_dets = mapper.assign(fd["detections"])

        for slot in range(4):
            if slot not in slot_dets:
                if active[slot]:
                    gap_count[slot] += 1
                    if gap_count[slot] > args.gap_tolerance:
                        active[slot] = False
                continue

            det = slot_dets[slot]
            kp = np.stack([det["kp_norm_x"], det["kp_norm_y"]], axis=1).astype(np.float32)

            kp_buffers[slot].append(kp)
            if len(kp_buffers[slot]) > SEQ_LEN + 30:
                kp_buffers[slot] = kp_buffers[slot][-(SEQ_LEN + 30):]

            # Wrist speed
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
                    "label": GRU_LABELS[pred],
                    "conf": round(float(probs[pred]), 3),
                    "probs": {GRU_LABELS[i]: round(float(probs[i]), 3) for i in range(4)},
                }
                all_events.append(event)

                if probs[2] > args.serve_threshold:
                    if args.serve_far_only and slot < 2:
                        continue
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

    # Post-merge filtering
    def _get_ball(frame_idx):
        k = str(frame_idx)
        pos = ball_det.get(k) or ball_pred.get(k)
        return pos

    def _filter_serve(ev):
        mid_frame = (ev["start_frame"] + ev["end_frame"]) // 2
        fd = data["frames"][mid_frame]
        slot = ev["slot"]

        # Find the player assigned to this slot
        slot_dets = mapper.assign(fd["detections"])
        if slot not in slot_dets:
            return True, ""  # can't verify, keep it
        det = slot_dets[slot]
        foot_x, foot_y = det["foot_x"], det["foot_y"]

        # 1. Net proximity: too close to net → not a serve
        rx, ry = mapper.to_court(foot_x, foot_y)
        if rx is not None and ry is not None:
            dist_net = abs(ry - mapper.net_y_m)
            if dist_net < args.min_net_dist:
                return False, f"too close to net ({dist_net:.1f}m)"

        # 2. Sideline: outside court width → not a serve
        if rx is not None:
            if rx < -1.0 or rx > mapper.cfg.court_width_m + 1.0:
                return False, f"outside sideline (rx={rx:.1f}m)"

        # 3. Frame edge: bbox touching left/right edge → incomplete body
        bbox = det["bbox"]
        if bbox[0] <= 1 or bbox[2] >= img_w - 1:
            return False, "frame edge (incomplete body)"

        # 4. Ball x-proximity: check a window around the event,
        #    if ball is detected in any frame near the player → keep
        #    (GRU reports serve after the motion, so ball may have left by mid_frame)
        if ball_det or ball_pred:
            search_start = max(0, ev["start_frame"] - int(data["fps"]))
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

        # 5. Ball toss check: verify toss-hit trajectory pattern
        #    (x-stationary segment with upward motion + hit departure)
        if ball_det or ball_pred:
            toss_ok, toss_reason = check_serve_toss(
                _get_ball, ev["start_frame"], ev["end_frame"],
                foot_x, foot_y, bbox, data["fps"],
                net_dir=net_dir,
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

    # Generate serve_review.mp4
    if args.video and merged_serves:
        review_path = os.path.join(os.path.dirname(args.out) or ".", "serve_review.mp4")
        print(f"\nGenerating serve review: {review_path}")
        _write_serve_review(args.video, merged_serves, data["fps"], review_path)


def _write_serve_review(src_video, serves, fps, dst_video, pad_seconds=0.5):
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

    # Build clip ranges
    clips = []
    for i, ev in enumerate(serves):
        clip_start = max(0, ev["start_frame"] - pad)
        clip_end = min(total - 1, ev["end_frame"] + pad)
        clips.append((i, ev, clip_start, clip_end))

    # Sequential read
    max_needed = max(ce for _, _, _, ce in clips)
    clip_idx = 0
    fi = 0
    while fi <= max_needed and clip_idx < len(clips):
        ret, frame = cap.read()
        if not ret:
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


if __name__ == "__main__":
    main()
