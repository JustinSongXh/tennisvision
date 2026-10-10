#!/usr/bin/env python3
"""TennisVision — Full rally detection pipeline.

Orchestrates six steps, each calling a standalone script via subprocess:
  1. Court calibration      → calib.json + court_overlay.jpg   (calibrate.py)
  2. Ball trajectory        → ball_positions.json               (extract_ball_positions.py)
  3. Player detection       → player_detections.json            (detect_players.py)
  4. Serve detection        → serve_events.json                 (detect_serves.py)
     (scene readiness + on-demand pose + GRU + toss filter)
  5. Rally detection        → rally_events.json + rally_cuts.mp4 (detect_rallies.py)
  6. Action classification  → action_events.json                (classify_actions.py) [optional]

Steps 2 and 3 are independent and could run in parallel.
Step 4 consumes Steps 1+2+3. Step 5 consumes Step 4. Step 6 is optional.

All outputs saved to results/<video_name>/.

Usage:
    # Full run (steps 1-5, skip step 6)
    python scripts/run_full_pipeline.py --video samples/sample3.mp4

    # Single step
    python scripts/run_full_pipeline.py --video samples/sample3.mp4 --step 1

    # Include optional action classification
    python scripts/run_full_pipeline.py --video samples/sample3.mp4 --with-actions

    # Custom weights directory (e.g. Kaggle)
    python scripts/run_full_pipeline.py --video video.mp4 --weights-dir /path/to/weights

    # Limit frames (for testing)
    python scripts/run_full_pipeline.py --video samples/sample3.mp4 --max-frames 1800
"""

import argparse
import os
import subprocess
import sys

try:
    import torch
    _HAS_CUDA = torch.cuda.is_available()
except ImportError:
    _HAS_CUDA = False


SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))


def get_out_dir(video_path):
    base = os.path.splitext(os.path.basename(video_path))[0]
    d = os.path.join("results", base)
    os.makedirs(d, exist_ok=True)
    return d


def print_banner(step_num, title):
    print(f"\n{'='*60}")
    print(f"  Step {step_num}: {title}")
    print(f"{'='*60}")


def run_script(name, args_list):
    """Run a script via subprocess, streaming stdout/stderr."""
    cmd = [sys.executable, "-u", os.path.join(SCRIPTS_DIR, name)] + args_list
    print(f"  $ {' '.join(cmd)}")
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print(f"  FAILED (exit code {ret.returncode})")
        return False
    return True


# ============================================================

def step1_calibrate(video_path, out_dir, weights_dir):
    print_banner(1, "Court Calibration")
    calib_path = os.path.join(out_dir, "calib.json")
    vis_path = os.path.join(out_dir, "court_overlay.jpg")

    if os.path.exists(calib_path):
        print(f"  Already exists: {calib_path}")
        return True

    return run_script("calibrate.py", [
        "blue-resnet",
        "--video", video_path,
        "--out", calib_path,
        "--vis", vis_path,
        "--weights", os.path.join(weights_dir, "court_resnet.pth"),
    ])


def step2_ball(video_path, out_dir, weights_dir, config_path, max_frames):
    print_banner(2, "Ball Trajectory")
    out_path = os.path.join(out_dir, "ball_positions.json")

    if os.path.exists(out_path):
        print(f"  Already exists: {out_path}")
        return True

    args = [
        "--video", video_path,
        "--out", out_path,
        "--weights", os.path.join(weights_dir, "wasb_tennis_best.pth.tar"),
        "--device", "cuda" if _HAS_CUDA else "cpu",
    ]
    if config_path:
        args += ["--config", config_path]
    if max_frames > 0:
        args += ["--max-frames", str(max_frames)]
    return run_script("extract_ball_positions.py", args)


def step3_players(video_path, out_dir, max_frames, no_half):
    print_banner(3, "Player Detection")
    out_path = os.path.join(out_dir, "player_detections.json")

    if os.path.exists(out_path):
        print(f"  Already exists: {out_path}")
        return True

    args = [
        "--video", video_path,
        "--out", out_path,
    ]
    if max_frames > 0:
        args += ["--max-frames", str(max_frames)]
    if no_half:
        args += ["--no-half"]
    return run_script("detect_players.py", args)


def step4_serves(video_path, out_dir, weights_dir, no_half):
    print_banner(4, "Serve Detection")
    out_path = os.path.join(out_dir, "serve_events.json")

    if os.path.exists(out_path):
        print(f"  Already exists: {out_path}")
        return True

    args = [
        "--players", os.path.join(out_dir, "player_detections.json"),
        "--calib", os.path.join(out_dir, "calib.json"),
        "--video", video_path,
        "--gru", os.path.join(weights_dir, "stroke_gru_v4_best.pt"),
        "--ball", os.path.join(out_dir, "ball_positions.json"),
        "--out", out_path,
    ]
    if no_half:
        args += ["--no-half"]
    return run_script("detect_serves.py", args)


def step5_rallies(video_path, out_dir):
    print_banner(5, "Rally Detection")
    rally_json = os.path.join(out_dir, "rally_events.json")

    if os.path.exists(rally_json):
        print(f"  Already exists: {rally_json}")
        return True

    return run_script("detect_rallies.py", [
        "--video", video_path,
        "--results-dir", out_dir,
    ])


def step6_actions(video_path, out_dir, weights_dir, no_half):
    print_banner(6, "Action Classification (optional)")
    out_path = os.path.join(out_dir, "action_events.json")

    if os.path.exists(out_path):
        print(f"  Already exists: {out_path}")
        return True

    args = [
        "--players", os.path.join(out_dir, "player_detections.json"),
        "--rallies", os.path.join(out_dir, "rally_events.json"),
        "--calib", os.path.join(out_dir, "calib.json"),
        "--video", video_path,
        "--gru", os.path.join(weights_dir, "stroke_gru_v4_best.pt"),
        "--ball", os.path.join(out_dir, "ball_positions.json"),
        "--out", out_path,
    ]
    if no_half:
        args += ["--no-half"]
    return run_script("classify_actions.py", args)


# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True)
    parser.add_argument("--step", type=int, default=0, help="Single step (1-6). 0=all.")
    parser.add_argument("--config", default=None, help="YAML config for ball detector")
    parser.add_argument("--weights-dir", default="weights",
                        help="Directory containing model weights")
    parser.add_argument("--max-frames", type=int, default=0, help="Limit frames (0=all)")
    parser.add_argument("--no-half", action="store_true", help="Disable FP16")
    parser.add_argument("--with-actions", action="store_true",
                        help="Run optional Step 6 (action classification)")
    args = parser.parse_args()

    out_dir = get_out_dir(args.video)
    wdir = args.weights_dir

    print()
    print("=" * 60)
    print("  TennisVision - Rally Detection Pipeline")
    print("=" * 60)
    print(f"  Video:       {args.video}")
    print(f"  Output:      {out_dir}/")
    print(f"  Weights:     {wdir}/")
    if args.max_frames > 0:
        print(f"  Max frames:  {args.max_frames}")
    print("=" * 60)

    run_all = args.step == 0

    if run_all or args.step == 1:
        if not step1_calibrate(args.video, out_dir, wdir):
            return

    if run_all or args.step == 2:
        if not step2_ball(args.video, out_dir, wdir, args.config, args.max_frames):
            return

    if run_all or args.step == 3:
        if not step3_players(args.video, out_dir, args.max_frames, args.no_half):
            return

    if run_all or args.step == 4:
        if not step4_serves(args.video, out_dir, wdir, args.no_half):
            return

    if run_all or args.step == 5:
        if not step5_rallies(args.video, out_dir):
            return

    if args.with_actions or args.step == 6:
        step6_actions(args.video, out_dir, wdir, args.no_half)

    if run_all:
        print()
        print("=" * 60)
        print("  Pipeline Complete!")
        print("=" * 60)
        print(f"  Results in: {out_dir}/")
        for f in sorted(os.listdir(out_dir)):
            sz = os.path.getsize(os.path.join(out_dir, f))
            print(f"    {f:<30s} {sz//1024:>6d} KB")
        print("=" * 60)


if __name__ == "__main__":
    main()
