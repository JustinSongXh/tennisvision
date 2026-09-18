#!/usr/bin/env python3
"""TennisVision — Full rally detection pipeline.

Constraints:
  · Fixed camera (no movement)
  · Court visible in frame
  · Doubles match (4 players)
  · Overhand serve only
  · Resolution >= 720p, 30fps recommended

Steps:
  1. Court calibration      → calib.json + court_overlay.jpg
  2. Keypoints extraction   → keypoints.json (yolo11m + yolo26s-pose)
  3. Ball trajectory        → ball_positions.json (WASB + Kalman)
  4. Serve detection        → serve_events.json (GRU v4 + wrist trigger)
  5. Rally detection        → rally_events.json + rally_cuts.mp4

All outputs saved to results/<video_name>/.

Usage:
    # Full run
    python scripts/run_full_pipeline.py --video samples/sample_short.mp4

    # Single step
    python scripts/run_full_pipeline.py --video samples/sample_short.mp4 --step 1

    # With existing keypoints (skip step 2)
    python scripts/run_full_pipeline.py --video samples/sample_short.mp4 \\
        --keypoints results/keypoints.json

    # Limit frames (for testing)
    python scripts/run_full_pipeline.py --video samples/sample_short.mp4 --max-frames 1800

    # Disable FP16
    python scripts/run_full_pipeline.py --video samples/sample_short.mp4 --no-half
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

# ---- Hardware auto-detection ----
if torch.cuda.device_count() >= 2:
    DET_DEVICE, POSE_DEVICE = 0, 1
elif torch.cuda.is_available():
    DET_DEVICE, POSE_DEVICE = 0, 0
else:
    DET_DEVICE, POSE_DEVICE = "cpu", "cpu"

USE_HALF = torch.cuda.is_available()
BALL_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BALL_RUNTIME = "torch" if torch.cuda.is_available() else "auto"


def get_out_dir(video_path):
    base = os.path.splitext(os.path.basename(video_path))[0]
    d = os.path.join("results", base)
    os.makedirs(d, exist_ok=True)
    return d


def print_banner(step_num, title):
    print(f"\n{'='*60}")
    print(f"  Step {step_num}: {title}")
    print(f"{'='*60}")


def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return info


# ============================================================
# Step 1: Court calibration
# ============================================================

def step1(video_path, out_dir):
    print_banner(1, "Court Calibration")
    calib_path = os.path.join(out_dir, "calib.json")
    vis_path = os.path.join(out_dir, "court_overlay.jpg")

    if os.path.exists(calib_path):
        print(f"  ⏭️  Already exists: {calib_path}")
    else:
        print(f"  Running auto-calibration...")
        import subprocess
        ret = subprocess.run([
            sys.executable, "scripts/calibrate.py", "blue-resnet",
            "--video", video_path, "--out", calib_path, "--vis", vis_path
        ])
        if ret.returncode != 0 or not os.path.exists(calib_path):
            print(f"  ❌ Calibration failed!")
            return False

    if not os.path.exists(vis_path) and os.path.exists(calib_path):
        _draw_court_overlay(video_path, calib_path, vis_path)

    print(f"  ✅ Calibration: {calib_path}")
    print(f"  📷 Court overlay: {vis_path}")
    print(f"  ⚠️  请检查 {vis_path}，确认球场线与实际画面对齐!")
    return True


def _draw_court_overlay(video_path, calib_path, vis_path):
    from tennisvision.court import reference as ref
    with open(calib_path) as f:
        calib = json.load(f)
    H_inv = np.array(calib["H_real_to_img"], dtype=np.float64)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 300)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return
    for (x0, y0), (x1, y1) in ref.court_line_segments_m():
        p0 = H_inv @ [x0, y0, 1.0]
        p1 = H_inv @ [x1, y1, 1.0]
        cv2.line(frame,
                 (int(p0[0]/p0[2]), int(p0[1]/p0[2])),
                 (int(p1[0]/p1[2]), int(p1[1]/p1[2])),
                 (0, 255, 0), 3)
    for kid, (ix, iy) in calib.get("keypoints_img", {}).items():
        pt = (int(round(float(ix))), int(round(float(iy))))
        label = ref.LABELS.get(int(kid), str(kid))
        cv2.circle(frame, pt, 8, (0, 0, 255), -1)
        cv2.putText(frame, label, (pt[0]+12, pt[1]+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imwrite(vis_path, frame)


# ============================================================
# Step 2: Keypoints extraction
# ============================================================

def step2(video_path, out_dir, max_frames=0):
    print_banner(2, "Keypoints Extraction")
    kp_path = os.path.join(out_dir, "keypoints.json")

    if os.path.exists(kp_path):
        with open(kp_path) as f:
            data = json.load(f)
        if data["total_frames"] > 0:
            print(f"  ⏭️  Already exists: {kp_path} ({data['total_frames']} frames)")
            return True

    from ultralytics import YOLO

    print(f"  Device: det={DET_DEVICE} pose={POSE_DEVICE} half={USE_HALF}")
    det_model = YOLO("yolo11m.pt")
    pose_model = YOLO("yolo26s-pose.pt")
    CROP_PAD = 0.15

    vinfo = get_video_info(video_path)
    FPS, W, H = vinfo["fps"], vinfo["width"], vinfo["height"]
    TOTAL = min(vinfo["total_frames"], max_frames) if max_frames > 0 else vinfo["total_frames"]
    print(f"  Video: {W}x{H} fps={FPS:.1f} processing {TOTAL} frames")

    frame_data = []
    fi, t0 = 0, time.time()
    cap = cv2.VideoCapture(video_path)

    while fi < TOTAL:
        ret, frame = cap.read()
        if not ret:
            break
        detections = []
        dr = det_model.track(frame, persist=True, verbose=False, classes=[0],
                             conf=0.3, imgsz=1280, device=DET_DEVICE, half=USE_HALF)[0]

        if dr.boxes is not None and dr.boxes.id is not None:
            boxes = dr.boxes.xyxy.cpu().numpy().astype(int)
            ids = dr.boxes.id.cpu().numpy().astype(int)
            confs = dr.boxes.conf.cpu().numpy()

            crops, offsets, kept = [], [], []
            for i in range(len(boxes)):
                x0, y0, x1, y1 = boxes[i]
                pad = int(CROP_PAD * max(x1-x0, y1-y0))
                cx0, cy0 = max(0, x0-pad), max(0, y0-pad)
                cx1, cy1 = min(W, x1+pad), min(H, y1+pad)
                crops.append(frame[cy0:cy1, cx0:cx1])
                offsets.append((cx0, cy0))
                kept.append(i)

            prs = pose_model.predict(crops, verbose=False, classes=[0],
                                     conf=0.15, imgsz=640, device=POSE_DEVICE,
                                     half=USE_HALF) if crops else []

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
                bw, bh = max(xx-xn, 1), max(yx-yn, 1)
                detections.append({
                    "tid": int(ids[i]), "det_conf": round(float(confs[i]), 3),
                    "bbox": [int(boxes[i][0]), int(boxes[i][1]), int(boxes[i][2]), int(boxes[i][3])],
                    "foot_x": round(float((boxes[i][0]+boxes[i][2])/2), 1),
                    "foot_y": int(boxes[i][3]),
                    "kp_x": [round(float(v), 4) for v in kp_x],
                    "kp_y": [round(float(v), 4) for v in kp_y],
                    "kp_v": [round(float(v), 3) for v in kp_v],
                    "kp_norm_x": [round(float((kp_x[k]-xn)/bw), 4) for k in range(17)],
                    "kp_norm_y": [round(float((kp_y[k]-yn)/bh), 4) for k in range(17)],
                })

        frame_data.append({"frame": fi, "detections": detections})
        fi += 1
        if fi % 500 == 0:
            e = time.time() - t0
            print(f"    frame {fi}/{TOTAL}  {fi/e:.1f} fps  eta {(TOTAL-fi)/(fi/e):.0f}s")

    cap.release()
    elapsed = time.time() - t0
    print(f"  ✅ {fi} frames in {elapsed:.0f}s ({fi/max(elapsed,0.1):.1f} fps)")

    with open(kp_path, "w") as f:
        json.dump({"fps": FPS, "width": W, "height": H,
                    "total_frames": fi, "frames": frame_data}, f)
    print(f"  Saved: {kp_path}")
    return True


# ============================================================
# Step 3: Ball trajectory
# ============================================================

def step3(video_path, out_dir, config_path=None, max_frames=0):
    print_banner(3, "Ball Trajectory")
    out_path = os.path.join(out_dir, "ball_positions.json")

    if os.path.exists(out_path):
        with open(out_path) as f:
            bp = json.load(f)
        if bp.get("total_frames", 0) > 0:
            print(f"  ⏭️  Already exists: {out_path}")
            print(f"  {bp.get('n_detected','?')} detected, {bp.get('n_predicted','?')} predicted")
            return True

    print(f"  Device: {BALL_DEVICE} runtime={BALL_RUNTIME}")

    from tennisvision.config import load_config
    from tennisvision.ball.wasb import WASBBallDetector, WASBConfig
    from tennisvision.ball.tracker import MultiTrackManager, TrackerConfig

    cfg = load_config(config_path) if config_path else load_config()
    bcfg, tcfg = cfg["ball"], cfg["tracker"]

    det = WASBBallDetector(WASBConfig(
        weights=bcfg["weights"], device=BALL_DEVICE, runtime=BALL_RUNTIME,
        score_threshold=bcfg.get("score_threshold", 0.5),
    ))
    tracker = MultiTrackManager(TrackerConfig(
        gate_px=tcfg.get("gate_px", 120), max_gap_frames=tcfg.get("max_gap_frames", 8),
        min_len=tcfg.get("min_len", 3), min_speed=tcfg.get("min_speed", 5.0),
        max_speed=tcfg.get("max_speed", 200.0),
    ))

    vinfo = get_video_info(video_path)
    FPS, W, H_px = vinfo["fps"], vinfo["width"], vinfo["height"]
    TOTAL = min(vinfo["total_frames"], max_frames) if max_frames > 0 else vinfo["total_frames"]

    detected, predicted = {}, {}
    fi, t0 = 0, time.time()
    cap = cv2.VideoCapture(video_path)
    while fi < TOTAL:
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
            e = time.time() - t0
            print(f"    frame {fi}/{TOTAL}  {fi/e:.1f} fps  eta {(TOTAL-fi)/(fi/e):.0f}s")
    cap.release()

    elapsed = time.time() - t0
    print(f"  ✅ {len(detected)} detected, {len(predicted)} predicted in {elapsed:.0f}s")

    with open(out_path, "w") as f:
        json.dump({"video": os.path.basename(video_path),
                    "fps": FPS, "total_frames": fi, "width": W, "height": H_px,
                    "n_detected": len(detected), "n_predicted": len(predicted),
                    "detected": {str(k): v for k, v in sorted(detected.items())},
                    "predicted": {str(k): v for k, v in sorted(predicted.items())}}, f)
    print(f"  Saved: {out_path}")
    return True


# ============================================================
# Step 4: Serve detection
# ============================================================

def step4(out_dir, gru_path="weights/stroke_gru_v4_best.pt"):
    print_banner(4, "Serve Detection")
    from tennisvision.action.slot_mapper import SlotMapper, SLOT_NAMES
    from tennisvision.action.gru_classifier import StrokeGRU, LABELS

    calib_path = os.path.join(out_dir, "calib.json")
    with open(calib_path) as f:
        calib = json.load(f)
    H = np.array(calib["H_img_to_real"], dtype=np.float64)
    mapper = SlotMapper(H)

    kp_path = os.path.join(out_dir, "keypoints.json")
    with open(kp_path) as f:
        kp_data = json.load(f)
    fps = kp_data["fps"]
    print(f"  Keypoints: {kp_data['total_frames']} frames")

    gru = StrokeGRU(n_classes=4)
    gru.load_state_dict(torch.load(gru_path, map_location="cpu"))
    gru.eval()

    SEQ_LEN, SPEED_THR, SERVE_THR, GAP_TOL = 30, 0.15, 0.8, 5
    kp_buf = {s: [] for s in range(4)}
    prev_kp = {s: None for s in range(4)}
    act = {s: False for s in range(4)}
    gap_ct = {s: 0 for s in range(4)}
    serve_hits, all_events = [], []
    gru_count = 0

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
                    speed += np.sqrt((kp[wi,0]-prev_kp[slot][wi,0])**2 +
                                     (kp[wi,1]-prev_kp[slot][wi,1])**2)
            prev_kp[slot] = kp
            if speed > SPEED_THR:
                act[slot] = True; gap_ct[slot] = 0
            elif act[slot]:
                gap_ct[slot] += 1
                if gap_ct[slot] > GAP_TOL:
                    act[slot] = False
            if act[slot] and len(kp_buf[slot]) >= SEQ_LEN:
                feat = np.array([k[:,:2].reshape(-1) for k in kp_buf[slot][-SEQ_LEN:]], dtype=np.float32)
                with torch.no_grad():
                    probs = torch.softmax(gru(torch.FloatTensor(feat).unsqueeze(0)), dim=1)[0].numpy()
                gru_count += 1
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
                               "time": round(cs/fps, 2)})
                cs, ce, csl, cc = fi, fi, sl, c
        merged.append({"frame": cs, "end_frame": ce, "slot": csl,
                        "slot_name": SLOT_NAMES[csl], "conf": round(cc, 3),
                        "time": round(cs/fps, 2)})

    out_path = os.path.join(out_dir, "serve_events.json")
    with open(out_path, "w") as f:
        json.dump({"serve_events": merged, "all_events": all_events,
                    "stats": {"gru_invocations": gru_count}}, f, indent=2)

    print(f"  GRU invocations: {gru_count}")
    print(f"  ✅ {len(merged)} serve events:")
    for s in merged:
        print(f"    🎾 {s['time']}s {s['slot_name']} conf={s['conf']}")
    print(f"  Saved: {out_path}")
    return merged


# ============================================================
# Step 5: Rally detection
# ============================================================

def step5(video_path, out_dir):
    print_banner(5, "Rally Detection")
    from tennisvision.pipeline.rally_fusion import MultiSignalRallyDetector, FusionRallyConfig
    from tennisvision.pipeline.rally import write_rally_video, save_rally_json

    calib_path = os.path.join(out_dir, "calib.json")
    with open(calib_path) as f:
        calib = json.load(f)
    H = np.array(calib["H_img_to_real"], dtype=np.float64)

    serve_path = os.path.join(out_dir, "serve_events.json")
    with open(serve_path) as f:
        serves = json.load(f)["serve_events"]

    kp_path = os.path.join(out_dir, "keypoints.json")
    with open(kp_path) as f:
        kp_meta = json.load(f)
    fps, total_frames = kp_meta["fps"], kp_meta["total_frames"]

    ball_path = os.path.join(out_dir, "ball_positions.json")
    ball_positions = {}
    if os.path.exists(ball_path):
        with open(ball_path) as f:
            bp = json.load(f)
        ball_positions = {int(k): tuple(v) for k, v in bp.get("predicted", {}).items()}
        print(f"  Ball positions: {len(ball_positions)} frames")
    else:
        print(f"  ⚠️  No ball trajectory")

    H_inv = np.linalg.inv(H)
    p = H_inv @ [10.97/2, 23.77/2, 1.0]
    net_y_px = float(p[1]/p[2]) if abs(p[2]) > 1e-9 else 625.0

    detector = MultiSignalRallyDetector(FusionRallyConfig(no_cross_timeout_s=5.0, min_duration_s=2.0))
    rallies = detector.detect(total_frames, fps, serve_events=serves,
                              ball_positions=ball_positions, net_y_px=net_y_px)

    rally_json = os.path.join(out_dir, "rally_events.json")
    save_rally_json(rallies, rally_json, fps=fps)

    rally_video = os.path.join(out_dir, "rally_cuts.mp4")
    if rallies:
        print(f"  Writing rally video...")
        write_rally_video(video_path, rallies, rally_video,
                          separator_seconds=1.0, extra_tail_seconds=2.0)

    print(f"  ✅ {len(rallies)} rallies:")
    for r in rallies:
        dur = (r.end_frame - r.start_frame) / fps
        print(f"    📹 Rally {r.idx+1}: {r.start_frame/fps:.1f}s-{r.end_frame/fps:.1f}s ({dur:.1f}s)")
    print(f"  Saved: {rally_json}")
    if rallies:
        print(f"  Video: {rally_video}")
    return rallies


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True)
    parser.add_argument("--step", type=int, default=0, help="Single step (1-5). 0=all.")
    parser.add_argument("--keypoints", default=None, help="Pre-extracted keypoints JSON")
    parser.add_argument("--config", default=None, help="YAML config for ball detector")
    parser.add_argument("--gru", default="weights/stroke_gru_v4_best.pt")
    parser.add_argument("--max-frames", type=int, default=0, help="Limit frames (0=all)")
    parser.add_argument("--no-half", action="store_true", help="Disable FP16")
    args = parser.parse_args()

    global USE_HALF
    if args.no_half:
        USE_HALF = False

    out_dir = get_out_dir(args.video)

    if args.keypoints and os.path.exists(args.keypoints):
        dst = os.path.join(out_dir, "keypoints.json")
        if not os.path.exists(dst):
            import shutil
            shutil.copy2(args.keypoints, dst)
            print(f"Copied keypoints to {dst}")

    print()
    print("╔" + "═"*58 + "╗")
    print("║" + "TennisVision — Rally Detection Pipeline".center(58) + "║")
    print("╠" + "═"*58 + "╣")
    print(f"║  Video:   {args.video:<47s} ║")
    print(f"║  Output:  {out_dir+'/':<47s} ║")
    print(f"║  Device:  det={DET_DEVICE} pose={POSE_DEVICE} half={USE_HALF!s:<23s} ║")
    if args.max_frames > 0:
        print(f"║  Limit:   {args.max_frames} frames{' '*(40-len(str(args.max_frames)))} ║")
    print("╠" + "═"*58 + "╣")
    print("║  Constraints:                                           ║")
    print("║    · Fixed camera · Court visible · Doubles (4 players) ║")
    print("║    · Overhand serve · >= 720p · 30fps recommended       ║")
    print("╚" + "═"*58 + "╝")

    run_all = args.step == 0

    if run_all or args.step == 1:
        if not step1(args.video, out_dir): return

    if run_all or args.step == 2:
        if not step2(args.video, out_dir, args.max_frames): return

    if run_all or args.step == 3:
        step3(args.video, out_dir, args.config, args.max_frames)

    if run_all or args.step == 4:
        step4(out_dir, gru_path=args.gru)

    if run_all or args.step == 5:
        step5(args.video, out_dir)

    if run_all:
        print()
        print("╔" + "═"*58 + "╗")
        print("║" + "✅ Pipeline Complete!".center(58) + "║")
        print("╠" + "═"*58 + "╣")
        print(f"║  Results in: {out_dir+'/':<44s} ║")
        for f in sorted(os.listdir(out_dir)):
            sz = os.path.getsize(os.path.join(out_dir, f))
            print(f"║    {f:<30s} {sz//1024:>6d} KB     ║")
        print("╚" + "═"*58 + "╝")


if __name__ == "__main__":
    main()
