#!/usr/bin/env python3
"""TennisVision — Full rally detection pipeline.

Orchestrates five steps, each calling a standalone script via subprocess:
  1. Court calibration      → calib.json + court_overlay.jpg   (calibrate.py)
  2. Keypoints extraction   → keypoints.json                   (extract_keypoints.py)
  3. Ball trajectory        → ball_positions.json               (extract_ball_positions.py)
  4. Serve detection        → serve_events.json                 (detect_serves.py)
  5. Rally detection        → rally_events.json + rally_cuts.mp4 (detect_rallies.py)

All outputs saved to results/<video_name>/.

Usage:
    # Full run
    python scripts/run_full_pipeline.py --video samples/sample_short.mp4

    # Single step
    python scripts/run_full_pipeline.py --video samples/sample_short.mp4 --step 1

    # Custom weights directory (e.g. Kaggle)
    python scripts/run_full_pipeline.py --video video.mp4 --weights-dir /path/to/weights

    # Limit frames (for testing)
    python scripts/run_full_pipeline.py --video samples/sample_short.mp4 --max-frames 1800

    # Disable FP16
    python scripts/run_full_pipeline.py --video samples/sample_short.mp4 --no-half
"""

import argparse
import os
import shutil
import subprocess
import sys


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

def step1(video_path, out_dir, weights_dir):
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


def step2(video_path, out_dir, max_frames, no_half):
    print_banner(2, "Keypoints Extraction")
    kp_path = os.path.join(out_dir, "keypoints.json")

    if os.path.exists(kp_path):
        print(f"  Already exists: {kp_path}")
        return True

    args = [
        "--video", video_path,
        "--out", kp_path,
    ]
    if max_frames > 0:
        args += ["--max-frames", str(max_frames)]
    if no_half:
        args += ["--no-half"]
    return run_script("extract_keypoints.py", args)


def step3(video_path, out_dir, weights_dir, config_path, max_frames):
    print_banner(3, "Ball Trajectory")
    out_path = os.path.join(out_dir, "ball_positions.json")

    if os.path.exists(out_path):
        print(f"  Already exists: {out_path}")
        return True

    args = [
        "--video", video_path,
        "--out", out_path,
        "--weights", os.path.join(weights_dir, "wasb_tennis_best.pth.tar"),
    ]
    if config_path:
        args += ["--config", config_path]
    if max_frames > 0:
        args += ["--max-frames", str(max_frames)]
    return run_script("extract_ball_positions.py", args)


def step4(out_dir, weights_dir):
    print_banner(4, "Serve Detection")
    out_path = os.path.join(out_dir, "serve_events.json")

    if os.path.exists(out_path):
        print(f"  Already exists: {out_path}")
        return True

    return run_script("detect_serves.py", [
        "--keypoints", os.path.join(out_dir, "keypoints.json"),
        "--calib", os.path.join(out_dir, "calib.json"),
        "--gru", os.path.join(weights_dir, "stroke_gru_v4_best.pt"),
        "--out", out_path,
    ])


def step5(video_path, out_dir):
    print_banner(5, "Rally Detection")
    rally_json = os.path.join(out_dir, "rally_events.json")

    if os.path.exists(rally_json):
        print(f"  Already exists: {rally_json}")
        return True

    return run_script("detect_rallies.py", [
        "--video", video_path,
        "--results-dir", out_dir,
    ])


# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True)
    parser.add_argument("--step", type=int, default=0, help="Single step (1-5). 0=all.")
    parser.add_argument("--keypoints", default=None, help="Pre-extracted keypoints JSON")
    parser.add_argument("--config", default=None, help="YAML config for ball detector")
    parser.add_argument("--weights-dir", default="weights",
                        help="Directory containing model weights")
    parser.add_argument("--max-frames", type=int, default=0, help="Limit frames (0=all)")
    parser.add_argument("--no-half", action="store_true", help="Disable FP16")
    args = parser.parse_args()

    out_dir = get_out_dir(args.video)
    wdir = args.weights_dir

    # Copy pre-extracted keypoints if provided
    if args.keypoints and os.path.exists(args.keypoints):
        dst = os.path.join(out_dir, "keypoints.json")
        if not os.path.exists(dst):
            shutil.copy2(args.keypoints, dst)
            print(f"Copied keypoints to {dst}")

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
        if not step1(args.video, out_dir, wdir):
            return

    if run_all or args.step == 2:
        if not step2(args.video, out_dir, args.max_frames, args.no_half):
            return

    if run_all or args.step == 3:
        step3(args.video, out_dir, wdir, args.config, args.max_frames)

    if run_all or args.step == 4:
        step4(out_dir, wdir)

    if run_all or args.step == 5:
        step5(args.video, out_dir)

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
