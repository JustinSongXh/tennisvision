#!/usr/bin/env python3
"""Court calibration CLI.

Supported methods:
  mark   : extract 4 doubles corners from a user-drawn red-line annotation image.
  tcd    : (V1) run yastrebksv/TennisCourtDetector CNN on a video frame.
  resnet : (V1) run CourtCheck-style ResNet50 + Linear(28).

Examples:
  # From a hand-marked reference image
  python scripts/calibrate.py mark --image court_mark.jpg --out calib.json

  # From a video frame using the CNN (V1)
  python scripts/calibrate.py tcd --video tennis.mp4 --frame 0 --out calib.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

# make `tennisvision` importable when run from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tennisvision.court import BlueContourDetector, RedMarkExtractor, reference as ref
from tennisvision.court.calibration import Calibration, save
from tennisvision.court.homography import homography_from_4_corners


def _project(H, pts):
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    h = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    p = (H @ h.T).T
    w = p[:, 2:3]
    w[np.abs(w) < 1e-12] = 1e-12
    return p[:, :2] / w


def _draw_verification(frame, calib: Calibration, out_path: str):
    vis = frame.copy()
    H_inv = calib.H_real_to_img
    for p0_m, p1_m in ref.court_line_segments_m():
        p = _project(H_inv, [p0_m, p1_m])
        q0 = (int(round(p[0, 0])), int(round(p[0, 1])))
        q1 = (int(round(p[1, 0])), int(round(p[1, 1])))
        cv2.line(vis, q0, q1, (0, 255, 0), 2)
    # label detected keypoints
    for kid, (x, y) in calib.keypoints_img.items():
        pt = (int(round(x)), int(round(y)))
        cv2.circle(vis, pt, 8, (0, 0, 255), -1)
        cv2.putText(vis, ref.LABELS.get(kid, str(kid)),
                    (pt[0] + 10, pt[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    cv2.imwrite(out_path, vis)


def cmd_mark(args):
    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit("cannot read " + args.image)
    det = RedMarkExtractor()
    hr = det.calibrate(img)
    if hr is None:
        raise SystemExit("RedMarkExtractor could not identify 4 court sides")

    # keypoints: we only have the 4 doubles corners explicitly, plus we
    # can project the full set back via H_inv for reference.
    kps: dict[int, tuple[float, float]] = {}
    H_inv = hr.H_real_to_img
    for kid, (xm, ym) in ref.KEYPOINTS_M.items():
        v = np.array([xm, ym, 1.0]); p = H_inv @ v
        kps[kid] = (float(p[0] / p[2]), float(p[1] / p[2]))

    calib = Calibration.from_homography_result(
        hr, image_size=(img.shape[1], img.shape[0]),
        keypoints_img=kps, source="mark",
    )
    save(calib, args.out)
    print("wrote", args.out)
    if args.vis:
        _draw_verification(img, calib, args.vis)
        print("wrote", args.vis)


def cmd_blue(args):
    det = BlueContourDetector()
    sample, corners = det.calibrate_from_video(args.video)
    if sample is None or len(corners) != 4:
        raise SystemExit("blue-contour calibration failed "
                         "(got %d corners)" % len(corners))
    # corners is dict with ids 0..3 = FL, FR, NR, NL
    four = [corners[0], corners[1], corners[2], corners[3]]
    hr = homography_from_4_corners(four)

    # Fill in all 14 projected keypoints for the calibration file
    import numpy as np
    H_inv = hr.H_real_to_img
    kps = {}
    for kid, (xm, ym) in ref.KEYPOINTS_M.items():
        v = np.array([xm, ym, 1.0]); p = H_inv @ v
        kps[kid] = (float(p[0] / p[2]), float(p[1] / p[2]))

    calib = Calibration.from_homography_result(
        hr, image_size=(sample.shape[1], sample.shape[0]),
        keypoints_img=kps, source="blue-contour",
    )
    save(calib, args.out)
    print("wrote", args.out)
    if args.vis:
        _draw_verification(sample, calib, args.vis)
        print("wrote", args.vis)


def cmd_tcd(args):
    from tennisvision.court.tcd_model import HeatmapDetector  # noqa: import at call time
    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("cannot read frame %d of %s" % (args.frame, args.video))
    det = HeatmapDetector(weights=args.weights, device=args.device)
    kps = det.detect(frame)
    if len(kps) < 6:
        raise SystemExit("only %d keypoints detected; need >=6" % len(kps))
    from tennisvision.court.homography import estimate_homography
    hr = estimate_homography(kps)
    if hr is None:
        raise SystemExit("could not fit homography from detected keypoints")
    calib = Calibration.from_homography_result(
        hr, image_size=(frame.shape[1], frame.shape[0]),
        keypoints_img=kps, source="tcd",
    )
    save(calib, args.out)
    print("wrote %s  (used subset %s, reprojection err %.2f m)" %
          (args.out, hr.used_keypoints, hr.reprojection_error_m))
    if args.vis:
        _draw_verification(frame, calib, args.vis)
        print("wrote", args.vis)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="method", required=True)

    m = sub.add_parser("mark", help="4-corner calibration from user-drawn red-line image")
    m.add_argument("--image", required=True)
    m.add_argument("--out",   default="calib.json")
    m.add_argument("--vis",   default="calib_vis.jpg")
    m.set_defaults(func=cmd_mark)

    b = sub.add_parser("blue", help="blue-contour 4-corner calibration (court fully in frame)")
    b.add_argument("--video", required=True)
    b.add_argument("--out",   default="calib.json")
    b.add_argument("--vis",   default="calib_vis.jpg")
    b.set_defaults(func=cmd_blue)

    t = sub.add_parser("tcd", help="TennisCourtDetector CNN (V1 — requires weights)")
    t.add_argument("--video",   required=True)
    t.add_argument("--frame",   type=int, default=0)
    t.add_argument("--weights", default="weights/court_tcd.pt")
    t.add_argument("--device",  default="cpu")
    t.add_argument("--out",     default="calib.json")
    t.add_argument("--vis",     default="calib_vis.jpg")
    t.set_defaults(func=cmd_tcd)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
