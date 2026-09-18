#!/usr/bin/env python3
"""TennisVision — Full rally detection pipeline.

Constraints / Prerequisites:
  - Fixed camera (single viewpoint, no camera movement)
  - Court must be mostly visible in frame
  - Doubles match (4 players expected)
  - Overhand serve only (underhand not trained)
  - Video resolution >= 720p (1080p recommended)
  - Frame rate 30fps (other rates auto-adjusted)

Pipeline steps:
  Step 1: Court calibration      → calib.json + court_overlay.jpg
  Step 2: Keypoints extraction   → keypoints.json (requires Kaggle GPU)
  Step 3: Ball trajectory        → ball_positions.json
  Step 4: Serve detection        → serve_events.json
  Step 5: Rally detection        → rally_events.json

All outputs saved to results/<video_name>/.

Usage:
    # Full local run (with pre-extracted keypoints from Kaggle):
    python scripts/run_full_pipeline.py \\
        --video samples/sample_short.mp4 \\
        --keypoints results/keypoints_all_frames_v6.json

    # Skip ball trajectory (serve detection only):
    python scripts/run_full_pipeline.py \\
        --video samples/sample_short.mp4 \\
        --keypoints results/keypoints_all_frames_v6.json \\
        --skip-ball

    # With existing calibration:
    python scripts/run_full_pipeline.py \\
        --video samples/sample_short.mp4 \\
        --calib samples/calib_sample1.json \\
        --keypoints results/keypoints_all_frames_v6.json
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# Step 1: Court calibration
# ============================================================

def step1_calibration(video_path, calib_path, out_dir):
    """Calibrate court → homography matrix + visualization."""
    print("\n" + "=" * 60)
    print("Step 1: Court Calibration")
    print("=" * 60)

    if calib_path and os.path.exists(calib_path):
        print(f"  Loading existing calibration: {calib_path}")
        with open(calib_path) as f:
            calib = json.load(f)
        H = np.array(calib["H_img_to_real"], dtype=np.float64)
        H_inv = np.array(calib["H_real_to_img"], dtype=np.float64)
    else:
        print(f"  Running auto-calibration on {video_path}...")
        from tennisvision.court.calibration import load as load_calib

        calib_out = os.path.join(out_dir, "calib.json")
        os.system(
            f"{sys.executable} scripts/calibrate.py blue-resnet "
            f"--video {video_path} --out {calib_out} --vis {os.path.join(out_dir, 'court_overlay.jpg')}"
        )
        if not os.path.exists(calib_out):
            print("  ❌ Calibration failed! Please provide --calib manually.")
            sys.exit(1)
        with open(calib_out) as f:
            calib = json.load(f)
        H = np.array(calib["H_img_to_real"], dtype=np.float64)
        H_inv = np.array(calib["H_real_to_img"], dtype=np.float64)
        calib_path = calib_out

    # Generate court overlay visualization
    vis_path = os.path.join(out_dir, "court_overlay.jpg")
    if not os.path.exists(vis_path):
        _draw_court_overlay(video_path, H_inv, calib.get("keypoints_img", {}), vis_path)

    print(f"  ✅ Calibration loaded")
    print(f"  📷 Court overlay: {vis_path}")
    print(f"  ⚠️  Please verify the court lines match the video!")

    return H, H_inv, calib_path


def _draw_court_overlay(video_path, H_inv, keypoints_img, out_path):
    """Draw court lines on a video frame for verification."""
    from tennisvision.court import reference as ref

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 300)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return

    # Draw court lines
    for (x0, y0), (x1, y1) in ref.court_line_segments_m():
        p0 = H_inv @ [x0, y0, 1.0]
        p1 = H_inv @ [x1, y1, 1.0]
        pt0 = (int(p0[0] / p0[2]), int(p0[1] / p0[2]))
        pt1 = (int(p1[0] / p1[2]), int(p1[1] / p1[2]))
        cv2.line(frame, pt0, pt1, (0, 255, 0), 3)

    # Draw keypoints
    for kid, (ix, iy) in keypoints_img.items():
        pt = (int(round(float(ix))), int(round(float(iy))))
        label = ref.LABELS.get(int(kid), str(kid))
        cv2.circle(frame, pt, 8, (0, 0, 255), -1)
        cv2.putText(frame, label, (pt[0] + 12, pt[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imwrite(out_path, frame)


# ============================================================
# Step 2: Keypoints extraction (Kaggle GPU)
# ============================================================

def step2_check_keypoints(keypoints_path, out_dir):
    """Check keypoints file exists, print instructions if not."""
    print("\n" + "=" * 60)
    print("Step 2: Keypoints Extraction")
    print("=" * 60)

    if keypoints_path and os.path.exists(keypoints_path):
        with open(keypoints_path) as f:
            data = json.load(f)
        total = data["total_frames"]
        fps = data["fps"]
        n_det = sum(len(fd["detections"]) for fd in data["frames"])
        tids = set()
        for fd in data["frames"]:
            for d in fd["detections"]:
                tids.add(d["tid"])
        print(f"  ✅ Keypoints loaded: {total} frames, {n_det} detections, {len(tids)} track IDs")
        return data

    print("  ❌ Keypoints file not found!")
    print()
    print("  This step requires GPU. Run on Kaggle:")
    print("    1. Upload video to Kaggle dataset")
    print("    2. Run infer-stroke-gru kernel (uses yolo11m + yolo26s-pose)")
    print("    3. Download keypoints_all_frames.json")
    print(f"    4. Re-run with: --keypoints <path_to_json>")
    sys.exit(1)


# ============================================================
# Step 3: Ball trajectory
# ============================================================

def step3_ball_trajectory(video_path, config_path, out_dir):
    """Extract ball positions with WASB + Kalman tracker."""
    print("\n" + "=" * 60)
    print("Step 3: Ball Trajectory")
    print("=" * 60)

    out_path = os.path.join(out_dir, "ball_positions.json")
    if os.path.exists(out_path):
        print(f"  Loading existing: {out_path}")
        with open(out_path) as f:
            bp = json.load(f)
        print(f"  ✅ {bp.get('n_detected', '?')} detected, {bp.get('n_predicted', '?')} predicted")
        return out_path

    print(f"  Extracting from {video_path} (CPU, ~13 fps with ONNX)...")

    from tennisvision.config import load_config
    from tennisvision.ball.wasb import WASBBallDetector, WASBConfig
    from tennisvision.ball.tracker import MultiTrackManager, TrackerConfig

    cfg = load_config(config_path) if config_path else load_config()
    bcfg, tcfg = cfg["ball"], cfg["tracker"]

    det = WASBBallDetector(WASBConfig(
        weights=bcfg["weights"], device="cpu",
        runtime=bcfg.get("runtime", "auto"),
        score_threshold=bcfg.get("score_threshold", 0.5),
    ))
    tracker = MultiTrackManager(TrackerConfig(
        gate_px=tcfg.get("gate_px", 120),
        max_gap_frames=tcfg.get("max_gap_frames", 8),
        min_len=tcfg.get("min_len", 3),
        min_speed=tcfg.get("min_speed", 5.0),
        max_speed=tcfg.get("max_speed", 200.0),
    ))

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H_px = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    detected, predicted = {}, {}
    fi, t0 = 0, time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        det.push_frame(frame)
        cand = det.detect()
        cand_xys = []
        if cand is not None:
            if isinstance(cand, list):
                cand_xys = [(int(c[0]), int(c[1])) for c in cand]
            elif isinstance(cand, tuple) and len(cand) == 2:
                cand_xys = [(int(cand[0]), int(cand[1]))]
        tracker.update(cand_xys, fi)
        champ = tracker.champion(fi)
        if champ is not None:
            px, py = champ.pred_xy
            predicted[fi] = [float(px), float(py)]
            if champ.last_det_frame == fi and champ.pts:
                pt = champ.pts[-1]
                detected[fi] = [float(pt.x), float(pt.y)]
        fi += 1
        if fi % 500 == 0:
            elapsed = time.time() - t0
            eta = (total - fi) / max(fi / elapsed, 0.1)
            print(f"    frame {fi}/{total}  {fi/elapsed:.1f} fps  eta {eta:.0f}s")
    cap.release()

    output = {
        "video": os.path.basename(video_path),
        "fps": fps, "total_frames": fi, "width": W, "height": H_px,
        "n_detected": len(detected), "n_predicted": len(predicted),
        "detected": {str(k): v for k, v in sorted(detected.items())},
        "predicted": {str(k): v for k, v in sorted(predicted.items())},
    }
    with open(out_path, "w") as f:
        json.dump(output, f)
    print(f"  ✅ {len(detected)} detected, {len(predicted)} predicted → {out_path}")
    return out_path


# ============================================================
# Step 4: Serve detection
# ============================================================

def step4_serve_detection(kp_data, gru_path, H_img_to_real, out_dir):
    """Detect serve events: slot mapping + wrist trigger + GRU."""
    print("\n" + "=" * 60)
    print("Step 4: Serve Detection")
    print("=" * 60)

    import torch
    from tennisvision.action.slot_mapper import SlotMapper, SLOT_NAMES
    from tennisvision.action.gru_classifier import StrokeGRU, LABELS

    fps = kp_data["fps"]
    mapper = SlotMapper(H_img_to_real)

    gru = StrokeGRU(n_classes=4)
    gru.load_state_dict(torch.load(gru_path, map_location="cpu"))
    gru.eval()

    SEQ_LEN, SPEED_THR, SERVE_THR, GAP_TOL = 30, 0.15, 0.8, 5

    kp_buf = {s: [] for s in range(4)}
    prev_kp = {s: None for s in range(4)}
    act = {s: False for s in range(4)}
    gap_ct = {s: 0 for s in range(4)}
    serve_hits, all_events = [], []

    for fd in kp_data["frames"]:
        fi = fd["frame"]
        slot_dets = mapper.assign(fd["detections"])

        for slot in range(4):
            if slot not in slot_dets:
                if act[slot]:
                    gap_ct[slot] += 1
                    if gap_ct[slot] > GAP_TOL:
                        act[slot] = False
                continue

            det = slot_dets[slot]
            kp = np.stack([det["kp_norm_x"], det["kp_norm_y"]], axis=1).astype(np.float32)
            kp_buf[slot].append(kp)
            if len(kp_buf[slot]) > SEQ_LEN + 30:
                kp_buf[slot] = kp_buf[slot][-(SEQ_LEN + 30):]

            speed = 0.0
            if prev_kp[slot] is not None:
                for wi in [9, 10]:
                    speed += np.sqrt((kp[wi, 0] - prev_kp[slot][wi, 0]) ** 2 +
                                     (kp[wi, 1] - prev_kp[slot][wi, 1]) ** 2)
            prev_kp[slot] = kp

            if speed > SPEED_THR:
                act[slot] = True
                gap_ct[slot] = 0
            elif act[slot]:
                gap_ct[slot] += 1
                if gap_ct[slot] > GAP_TOL:
                    act[slot] = False

            if act[slot] and len(kp_buf[slot]) >= SEQ_LEN:
                feat = np.array([k[:, :2].reshape(-1) for k in kp_buf[slot][-SEQ_LEN:]], dtype=np.float32)
                with torch.no_grad():
                    probs = torch.softmax(gru(torch.FloatTensor(feat).unsqueeze(0)), dim=1)[0].numpy()
                pred = int(np.argmax(probs))
                all_events.append({
                    "frame": fi, "slot": slot, "slot_name": SLOT_NAMES[slot],
                    "label": LABELS[pred], "conf": round(float(probs[pred]), 3),
                })
                if probs[2] > SERVE_THR and slot >= 2:
                    serve_hits.append((fi, slot, float(probs[2])))

    # Merge
    merged = []
    if serve_hits:
        cs, ce, csl, cc = serve_hits[0][0], serve_hits[0][0], serve_hits[0][1], serve_hits[0][2]
        for fi, sl, c in serve_hits[1:]:
            if sl == csl and fi - ce <= 10:
                ce, cc = fi, max(cc, c)
            else:
                merged.append({"frame": cs, "end_frame": ce, "slot": csl,
                               "slot_name": SLOT_NAMES[csl], "conf": round(cc, 3),
                               "time": round(cs / fps, 2)})
                cs, ce, csl, cc = fi, fi, sl, c
        merged.append({"frame": cs, "end_frame": ce, "slot": csl,
                        "slot_name": SLOT_NAMES[csl], "conf": round(cc, 3),
                        "time": round(cs / fps, 2)})

    out_path = os.path.join(out_dir, "serve_events.json")
    with open(out_path, "w") as f:
        json.dump({"serve_events": merged, "all_events": all_events}, f, indent=2)

    print(f"  ✅ {len(merged)} serve events detected")
    for s in merged:
        print(f"    🎾 {s['time']}s {s['slot_name']} conf={s['conf']}")
    print(f"  Saved: {out_path}")
    return merged


# ============================================================
# Step 5: Rally detection
# ============================================================

def step5_rally_detection(serve_events, ball_path, H_img_to_real, fps, total_frames, video_path, out_dir):
    """Detect rallies from serve events + ball trajectory."""
    print("\n" + "=" * 60)
    print("Step 5: Rally Detection")
    print("=" * 60)

    from tennisvision.pipeline.rally_fusion import MultiSignalRallyDetector, FusionRallyConfig
    from tennisvision.pipeline.rally import write_rally_video, save_rally_json

    ball_positions = {}
    if ball_path and os.path.exists(ball_path):
        with open(ball_path) as f:
            bp = json.load(f)
        ball_positions = {int(k): tuple(v) for k, v in bp.get("predicted", {}).items()}

    # Net Y pixel from homography (court midpoint projection)
    mid = [10.97 / 2, 23.77 / 2, 1.0]
    H_inv = np.linalg.inv(H_img_to_real)
    p = H_inv @ mid
    net_y_px = float(p[1] / p[2]) if abs(p[2]) > 1e-9 else 625.0

    detector = MultiSignalRallyDetector(FusionRallyConfig(no_cross_timeout_s=5.0, min_duration_s=2.0))
    rallies = detector.detect(
        total_frames, fps,
        serve_events=serve_events,
        ball_positions=ball_positions,
        net_y_px=net_y_px,
    )

    # Save JSON
    rally_json_path = os.path.join(out_dir, "rally_events.json")
    save_rally_json(rallies, rally_json_path, fps=fps)

    # Cut rally video
    rally_video_path = os.path.join(out_dir, "rally_cuts.mp4")
    if rallies:
        write_rally_video(video_path, rallies, rally_video_path,
                          separator_seconds=1.0, extra_tail_seconds=2.0)

    print(f"  ✅ {len(rallies)} rallies detected")
    for r in rallies:
        dur = (r.end_frame - r.start_frame) / fps
        print(f"    📹 Rally {r.idx+1}: {r.start_frame/fps:.1f}s-{r.end_frame/fps:.1f}s ({dur:.1f}s)")
    print(f"  Saved: {rally_json_path}")
    if rallies:
        print(f"  Video: {rally_video_path}")
    return rallies


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--calib", default=None, help="Court calibration JSON (auto-generate if not provided)")
    parser.add_argument("--keypoints", required=True, help="Keypoints JSON from Kaggle")
    parser.add_argument("--gru", default="weights/stroke_gru_v4_best.pt")
    parser.add_argument("--config", default=None, help="YAML config for ball detector")
    parser.add_argument("--skip-ball", action="store_true")
    args = parser.parse_args()

    # Create output directory: results/<video_name>/
    base = os.path.splitext(os.path.basename(args.video))[0]
    out_dir = os.path.join("results", base)
    os.makedirs(out_dir, exist_ok=True)

    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + "TennisVision — Rally Detection Pipeline".center(58) + "║")
    print("╠" + "═" * 58 + "╣")
    print(f"║  Video:     {args.video:<44s} ║")
    print(f"║  Output:    {out_dir + '/':<44s} ║")
    print("╠" + "═" * 58 + "╣")
    print("║  Constraints:                                           ║")
    print("║    · Fixed camera (no movement)                         ║")
    print("║    · Court visible in frame                             ║")
    print("║    · Doubles match (4 players)                          ║")
    print("║    · Overhand serve only                                ║")
    print("║    · Resolution >= 720p, 30fps recommended              ║")
    print("╚" + "═" * 58 + "╝")

    # Step 1
    H, H_inv, calib_path = step1_calibration(args.video, args.calib, out_dir)

    # Step 2
    kp_data = step2_check_keypoints(args.keypoints, out_dir)
    fps = kp_data["fps"]
    total_frames = kp_data["total_frames"]

    # Step 3
    ball_path = None
    if not args.skip_ball:
        ball_path = step3_ball_trajectory(args.video, args.config, out_dir)
    else:
        existing = os.path.join(out_dir, "ball_positions.json")
        if os.path.exists(existing):
            ball_path = existing
            print(f"\n  [ball] Using existing: {ball_path}")
        else:
            print(f"\n  [ball] Skipped (no ball trajectory)")

    # Step 4
    serves = step4_serve_detection(kp_data, args.gru, H, out_dir)

    # Step 5
    rallies = step5_rally_detection(serves, ball_path, H, fps, total_frames, args.video, out_dir)

    # Summary
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + "Pipeline Complete!".center(58) + "║")
    print("╠" + "═" * 58 + "╣")
    print(f"║  Serves detected:  {len(serves):<38d} ║")
    print(f"║  Rallies detected: {len(rallies):<38d} ║")
    print("╠" + "═" * 58 + "╣")
    print(f"║  Output files in {out_dir + '/:':<40s} ║")
    print("║    court_overlay.jpg    — verify court lines            ║")
    print("║    serve_events.json    — serve timestamps              ║")
    print("║    rally_events.json    — rally boundaries              ║")
    if rallies:
        print("║    rally_cuts.mp4       — rally highlight video         ║")
    if ball_path:
        print("║    ball_positions.json  — ball trajectory               ║")
    print("╚" + "═" * 58 + "╝")


if __name__ == "__main__":
    main()
