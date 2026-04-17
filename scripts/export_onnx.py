#!/usr/bin/env python3
"""Export WASB weights to ONNX.

The WASBBallDetector also does this lazily on first use — this script
is just the explicit entrypoint for CI / weight pre-baking.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tennisvision.ball.wasb import _export_onnx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="weights/wasb_tennis_best.pth.tar")
    p.add_argument("--out",     default=None)
    p.add_argument("--input-width",  type=int, default=512)
    p.add_argument("--input-height", type=int, default=288)
    args = p.parse_args()
    out = args.out or (os.path.splitext(args.weights)[0] + ".onnx")
    _export_onnx(args.weights, out, args.input_height, args.input_width)


if __name__ == "__main__":
    main()
