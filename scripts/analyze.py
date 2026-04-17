#!/usr/bin/env python3
"""Full pipeline CLI: video + calib.json -> annotated video."""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tennisvision import pipeline
from tennisvision.config import load_config
from tennisvision.court.calibration import load as load_calib


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video",  required=True)
    p.add_argument("--out",    required=True, help="output annotated .mp4")
    p.add_argument("--calib",  required=True, help="calib.json")
    p.add_argument("--config", default=None, help="YAML config (optional)")
    p.add_argument("--progress-every", type=int, default=100)
    args = p.parse_args()

    cfg = load_config(args.config)
    calib = load_calib(args.calib)

    t0 = time.time()
    res = pipeline.run(
        video_path=args.video,
        output_path=args.out,
        calib=calib,
        cfg=cfg,
        progress_every=args.progress_every,
    )
    dt = time.time() - t0
    print("done -> %s  (%.1fs, %d frames, %d bounces)" %
          (args.out, dt, res.total_frames, res.bounces))


if __name__ == "__main__":
    main()
