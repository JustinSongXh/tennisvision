#!/usr/bin/env python3
"""TennisVision — Rally detection pipeline.

Full run (串行执行, 每步保存中间结果):
    python scripts/run_full_pipeline.py --video samples/sample_short.mp4

Single step (单独跑某一步):
    python scripts/run_full_pipeline.py --video samples/sample_short.mp4 --step 1
    python scripts/run_full_pipeline.py --video samples/sample_short.mp4 --step 3
    python scripts/run_full_pipeline.py --video samples/sample_short.mp4 --step 4
    python scripts/run_full_pipeline.py --video samples/sample_short.mp4 --step 5

Constraints:
  · Fixed camera (no movement)
  · Court visible in frame
  · Doubles match (4 players)
  · Overhand serve only
  · Resolution >= 720p, 30fps recommended

Steps:
  1. Court calibration      → calib.json + court_overlay.jpg
  2. Keypoints extraction   → keypoints.json (Kaggle GPU, manual)
  3. Ball trajectory        → ball_positions.json
  4. Serve detection        → serve_events.json
  5. Rally detection        → rally_events.json + rally_cuts.mp4

All outputs saved to results/<video_name>/.
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


# ---- Inference config (auto-detect hardware) ----
if torch.cuda.device_count() >= 2:
    DET_DEVICE, POSE_DEVICE = 0, 1
elif torch.cuda.is_available():
    DET_DEVICE, POSE_DEVICE = 0, 0
else:
    DET_DEVICE, POSE_DEVICE = "cpu", "cpu"

USE_HALF = torch.cuda.is_available()  # FP16 only on GPU
BALL_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BALL_RUNTIME = "torch" if torch.cuda.is_available() else "auto"  # GPU needs torch, CPU uses ONNX


def get_out_dir(video_path):
    base = os.path.splitext(os.path.basename(video_path))[0]
    d = os.path.join("results", base)
    os.makedirs(d, exist_ok=True)
    return d


def print_banner(step_num, title):
    print(f"\n{'='*60}")
    print(f"  Step {step_num}: {title}")
    print(f"{'='*60}")


# ============================================================
# Step 1: Court calibration
# ============================================================

def step1(video_path, out_dir):
    print_banner(1, "Court Calibration")

    calib_path = os.path.join(out_dir, "calib.json")
    vis_path = os.path.join(out_dir, "court_overlay.jpg")

    if os.path.exists(calib_path):
        print(f"  ⏭️  Already exists: {calib_path}")
        print(f"  Delete it to re-run calibration.")
    else:
        print(f"  Running auto-calibration...")
        ret = os.system(
            f"{sys.executable} scripts/calibrate.py blue-resnet "
            f"--video {video_path} --out {calib_path} --vis {vis_path}"
        )
        if ret != 0 or not os.path.exists(calib_path):
            print(f"  ❌ Calibration failed!")
            print(f"  Try manual calibration: python scripts/calibrate.py mark --image <marked_image> --out {calib_path}")
            return False

    # 画 court overlay
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
                 (int(p0[0] / p0[2]), int(p0[1] / p0[2])),
                 (int(p1[0] / p1[2]), int(p1[1] / p1[2])),
                 (0, 255, 0), 3)

    for kid, (ix, iy) in calib.get("keypoints_img", {}).items():
        pt = (int(round(float(ix))), int(round(float(iy))))
        label = ref.LABELS.get(int(kid), str(kid))
        cv2.circle(frame, pt, 8, (0, 0, 255), -1)
        cv2.putText(frame, label, (pt[0] + 12, pt[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imwrite(vis_path, frame)


# ============================================================
# Step 2: Keypoints extraction (Kaggle GPU)
# ============================================================

def step2(video_path, out_dir):
    print_banner(2, "Keypoints Extraction (Kaggle GPU)")

    kp_path = os.path.join(out_dir, "keypoints.json")

    if os.path.exists(kp_path):
        with open(kp_path) as f:
            data = json.load(f)
        n = data["total_frames"]
        print(f"  ✅ Already exists: {kp_path} ({n} frames)")
        return True

    print(f"  ❌ Keypoints file not found: {kp_path}")
    print()
    print(f"  这一步需要 GPU，请在 Kaggle 上执行:")
    print(f"    1. 上传视频到 Kaggle dataset")
    print(f"    2. 运行 infer-stroke-gru kernel (yolo11m + yolo26s-pose)")
    print(f"    3. 下载 keypoints_all_frames.json")
    print(f"    4. 重命名并放到: {kp_path}")
    print()
    print(f"  或指定已有的 keypoints 文件:")
    print(f"    python scripts/run_full_pipeline.py --video {video_path} --keypoints <path>")
    return False


# ============================================================
# Step 3: Ball trajectory
# ============================================================

def step3(video_path, out_dir, config_path=None):
    print_banner(3, "Ball Trajectory")

    out_path = os.path.join(out_dir, "ball_positions.json")

    if os.path.exists(out_path):
        with open(out_path) as f:
            bp = json.load(f)
        print(f"  ⏭️  Already exists: {out_path}")
        print(f"  {bp.get('n_detected', '?')} detected, {bp.get('n_predicted', '?')} predicted")
        return True

    print(f"  Extracting ball positions ({BALL_DEVICE}, runtime={BALL_RUNTIME})...")
    print(f"  This may take a while for long videos.")

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
        gate_px=tcfg.get("gate_px", 120),
        max_gap_frames=tcfg.get("max_gap_frames", 8),
        min_len=tcfg.get("min_len", 3),
        min_speed=tcfg.get("min_speed", 5.0),
        max_speed=tcfg.get("max_speed", 200.0),
    ))

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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
            print(f"    frame {fi}/{total}  {fi / elapsed:.1f} fps  eta {eta:.0f}s")
    cap.release()

    output = {
        "video": os.path.basename(video_path),
        "fps": fps, "total_frames": fi, "width": W, "height": H,
        "n_detected": len(detected), "n_predicted": len(predicted),
        "detected": {str(k): v for k, v in sorted(detected.items())},
        "predicted": {str(k): v for k, v in sorted(predicted.items())},
    }
    with open(out_path, "w") as f:
        json.dump(output, f)
    print(f"  ✅ {len(detected)} detected, {len(predicted)} predicted")
    print(f"  Saved: {out_path}")
    return True


# ============================================================
# Step 4: Serve detection
# ============================================================

def step4(out_dir, keypoints_path=None, gru_path="weights/stroke_gru_v4_best.pt"):
    print_banner(4, "Serve Detection")

    import torch
    from tennisvision.action.slot_mapper import SlotMapper, SLOT_NAMES
    from tennisvision.action.gru_classifier import StrokeGRU, LABELS

    # Load calib
    calib_path = os.path.join(out_dir, "calib.json")
    with open(calib_path) as f:
        calib = json.load(f)
    H = np.array(calib["H_img_to_real"], dtype=np.float64)

    # Load keypoints
    kp_path = keypoints_path or os.path.join(out_dir, "keypoints.json")
    with open(kp_path) as f:
        kp_data = json.load(f)
    fps = kp_data["fps"]
    print(f"  Keypoints: {kp_data['total_frames']} frames")

    mapper = SlotMapper(H)
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
                gru_count += 1
                pred = int(np.argmax(probs))
                all_events.append({
                    "frame": fi, "slot": slot, "slot_name": SLOT_NAMES[slot],
                    "label": LABELS[pred], "conf": round(float(probs[pred]), 3),
                })
                if probs[2] > SERVE_THR and slot >= 2:
                    serve_hits.append((fi, slot, float(probs[2])))

    # Merge consecutive serve hits
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
        json.dump({"serve_events": merged, "all_events": all_events,
                    "stats": {"gru_invocations": gru_count}}, f, indent=2)

    print(f"  GRU invocations: {gru_count}")
    print(f"  ✅ {len(merged)} serve events detected:")
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

    # Load calib
    calib_path = os.path.join(out_dir, "calib.json")
    with open(calib_path) as f:
        calib = json.load(f)
    H = np.array(calib["H_img_to_real"], dtype=np.float64)

    # Load serves
    serve_path = os.path.join(out_dir, "serve_events.json")
    with open(serve_path) as f:
        serve_data = json.load(f)
    serves = serve_data["serve_events"]

    # Load keypoints meta for fps/total_frames
    kp_path = os.path.join(out_dir, "keypoints.json")
    if not os.path.exists(kp_path):
        # Try to find any keypoints file
        for f in os.listdir(out_dir):
            if "keypoints" in f and f.endswith(".json"):
                kp_path = os.path.join(out_dir, f)
                break
    with open(kp_path) as f:
        kp_meta = json.load(f)
    fps = kp_meta["fps"]
    total_frames = kp_meta["total_frames"]

    # Load ball positions
    ball_path = os.path.join(out_dir, "ball_positions.json")
    ball_positions = {}
    if os.path.exists(ball_path):
        with open(ball_path) as f:
            bp = json.load(f)
        ball_positions = {int(k): tuple(v) for k, v in bp.get("predicted", {}).items()}
        print(f"  Ball positions: {len(ball_positions)} frames")
    else:
        print(f"  ⚠️  No ball trajectory, rally end detection may be less accurate")

    # Net Y pixel
    H_inv = np.linalg.inv(H)
    p = H_inv @ [10.97 / 2, 23.77 / 2, 1.0]
    net_y_px = float(p[1] / p[2]) if abs(p[2]) > 1e-9 else 625.0

    detector = MultiSignalRallyDetector(FusionRallyConfig(no_cross_timeout_s=5.0, min_duration_s=2.0))
    rallies = detector.detect(
        total_frames, fps,
        serve_events=serves,
        ball_positions=ball_positions,
        net_y_px=net_y_px,
    )

    rally_json = os.path.join(out_dir, "rally_events.json")
    save_rally_json(rallies, rally_json, fps=fps)

    rally_video = os.path.join(out_dir, "rally_cuts.mp4")
    if rallies:
        print(f"  Writing rally cut video...")
        write_rally_video(video_path, rallies, rally_video,
                          separator_seconds=1.0, extra_tail_seconds=2.0)

    print(f"  ✅ {len(rallies)} rallies detected:")
    for r in rallies:
        dur = (r.end_frame - r.start_frame) / fps
        print(f"    📹 Rally {r.idx + 1}: {r.start_frame / fps:.1f}s - {r.end_frame / fps:.1f}s ({dur:.1f}s)")
    print(f"  Saved: {rally_json}")
    if rallies:
        print(f"  Video: {rally_video}")
    return rallies


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--step", type=int, default=0,
                        help="Run single step (1-5). 0 = run all.")
    parser.add_argument("--keypoints", default=None,
                        help="Pre-extracted keypoints JSON (copies to output dir)")
    parser.add_argument("--config", default=None, help="YAML config for ball detector")
    parser.add_argument("--gru", default="weights/stroke_gru_v4_best.pt")
    parser.add_argument("--no-half", action="store_true",
                        help="Disable FP16 (use FP32 even on GPU)")
    args = parser.parse_args()

    # Override global config if --no-half
    global USE_HALF
    if args.no_half:
        USE_HALF = False

    out_dir = get_out_dir(args.video)

    # If keypoints provided, copy/link to output dir
    if args.keypoints and os.path.exists(args.keypoints):
        dst = os.path.join(out_dir, "keypoints.json")
        if not os.path.exists(dst):
            import shutil
            shutil.copy2(args.keypoints, dst)
            print(f"Copied keypoints to {dst}")

    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + "TennisVision — Rally Detection Pipeline".center(58) + "║")
    print("╠" + "═" * 58 + "╣")
    print(f"║  Video:  {args.video:<48s} ║")
    print(f"║  Output: {out_dir + '/':<48s} ║")
    print("╠" + "═" * 58 + "╣")
    print("║  Constraints:                                           ║")
    print("║    · Fixed camera (no movement)                         ║")
    print("║    · Court visible in frame                             ║")
    print("║    · Doubles match (4 players)                          ║")
    print("║    · Overhand serve only                                ║")
    print("║    · Resolution >= 720p, 30fps recommended              ║")
    print("╚" + "═" * 58 + "╝")

    run_all = args.step == 0

    # Step 1
    if run_all or args.step == 1:
        if not step1(args.video, out_dir):
            return

    # Step 2
    if run_all or args.step == 2:
        if not step2(args.video, out_dir):
            if run_all:
                print("\n  ⛔ Pipeline stopped. Complete step 2 manually, then re-run.")
                return

    # Step 3
    if run_all or args.step == 3:
        step3(args.video, out_dir, args.config)

    # Step 4
    if run_all or args.step == 4:
        step4(out_dir, gru_path=args.gru)

    # Step 5
    if run_all or args.step == 5:
        step5(args.video, out_dir)

    if run_all:
        print()
        print("╔" + "═" * 58 + "╗")
        print("║" + "✅ Pipeline Complete!".center(58) + "║")
        print("╠" + "═" * 58 + "╣")
        print(f"║  Results in: {out_dir + '/':<44s} ║")
        print("║    court_overlay.jpg  — verify court lines              ║")
        print("║    serve_events.json  — serve timestamps                ║")
        print("║    rally_events.json  — rally boundaries                ║")
        print("║    rally_cuts.mp4     — rally highlight video           ║")
        print("║    ball_positions.json — ball trajectory                ║")
        print("╚" + "═" * 58 + "╝")


if __name__ == "__main__":
    main()
