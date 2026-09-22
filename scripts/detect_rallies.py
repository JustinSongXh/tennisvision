#!/usr/bin/env python3
"""Detect rallies from serve events + ball trajectory.

Uses three rally end conditions:
  1. No net crossing timeout (5s default)
  2. Next serve event
  3. Triple bounce in same half court (within 2s window)

Usage:
    python -u scripts/detect_rallies.py \
        --video samples/sample_short.mp4 \
        --results-dir results/sample_short

    # Custom thresholds
    python -u scripts/detect_rallies.py \
        --video samples/sample_short.mp4 \
        --results-dir results/sample_short \
        --no-cross-timeout 5.0 \
        --min-duration 2.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tennisvision.pipeline.rally_fusion import MultiSignalRallyDetector, FusionRallyConfig
from tennisvision.pipeline.rally import write_rally_video, save_rally_json


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True, help="Original video (for rally_cuts.mp4)")
    parser.add_argument("--results-dir", required=True,
                        help="Directory with calib.json, keypoints.json, ball_positions.json, serve_events.json")
    parser.add_argument("--no-cross-timeout", type=float, default=5.0,
                        help="Seconds without net crossing to end rally (default: 5.0)")
    parser.add_argument("--min-duration", type=float, default=2.0,
                        help="Minimum rally duration in seconds (default: 2.0)")
    args = parser.parse_args()

    rdir = args.results_dir

    # Load calib
    calib_path = os.path.join(rdir, "calib.json")
    with open(calib_path) as f:
        calib = json.load(f)
    H = np.array(calib["H_img_to_real"], dtype=np.float64)

    # Load serves
    serve_path = os.path.join(rdir, "serve_events.json")
    with open(serve_path) as f:
        serves = json.load(f)["serve_events"]

    # Load keypoints (for fps + total_frames)
    kp_path = os.path.join(rdir, "keypoints.json")
    with open(kp_path) as f:
        kp_meta = json.load(f)
    fps, total_frames = kp_meta["fps"], kp_meta["total_frames"]

    # Load ball positions
    ball_path = os.path.join(rdir, "ball_positions.json")
    ball_all = {}
    ball_det = {}
    if os.path.exists(ball_path):
        with open(ball_path) as f:
            bp = json.load(f)
        ball_all = {int(k): tuple(v) for k, v in bp.get("predicted", {}).items()}
        # Merge detected into all (detected takes priority)
        for k, v in bp.get("detected", {}).items():
            ball_all[int(k)] = tuple(v)
        ball_det = {int(k): tuple(v) for k, v in bp.get("detected", {}).items()}
        print(f"Ball positions: {len(ball_all)} total, {len(ball_det)} detected")
    else:
        print("No ball trajectory found, continuing without")

    # Net y in pixel space
    H_inv = np.linalg.inv(H)
    p = H_inv @ [10.97 / 2, 23.77 / 2, 1.0]
    net_y_px = float(p[1] / p[2]) if abs(p[2]) > 1e-9 else 625.0

    # Detect rallies
    detector = MultiSignalRallyDetector(FusionRallyConfig(
        no_cross_timeout_s=args.no_cross_timeout,
        min_duration_s=args.min_duration,
    ))
    rallies = detector.detect(total_frames, fps, serve_events=serves,
                              ball_positions=ball_all, ball_detected=ball_det,
                              net_y_px=net_y_px, H_img_to_real=H)

    # Save rally JSON
    rally_json = os.path.join(rdir, "rally_events.json")
    save_rally_json(rallies, rally_json, fps=fps)

    # Write rally video
    rally_video = os.path.join(rdir, "rally_cuts.mp4")
    if rallies:
        print("Writing rally video...")
        # Bounces as (rx, ry, frame) for minimap
        bounce_dots = [(rx, ry, f) for f, _, rx, ry in detector.bounces]
        write_rally_video(args.video, rallies, rally_video,
                          separator_seconds=1.0, extra_tail_seconds=2.0,
                          bounces=bounce_dots)

    print(f"\n{len(rallies)} rallies:")
    for r in rallies:
        dur = (r.end_frame - r.start_frame) / fps
        print(f"  Rally {r.idx+1}: {r.start_frame/fps:.1f}s-{r.end_frame/fps:.1f}s ({dur:.1f}s)")
    print(f"Saved: {rally_json}")
    if rallies:
        print(f"Video: {rally_video}")


if __name__ == "__main__":
    main()
