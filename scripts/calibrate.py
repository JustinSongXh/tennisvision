#!/usr/bin/env python3
"""Court calibration CLI.

Three methods are supported:

  mark         Manual: extract 4 doubles corners from a user-drawn red-line
               annotation image. No ML weights needed.

  blue-resnet  Automatic (recommended for amateur/ground-level footage):
               detect blue court surface via HSV → perspective-warp to
               rectified view → ResNet50 regression for all 14 keypoints.
               Requires weights/court_resnet.pth.

  tcd          Automatic (broadcast/elevated footage): run the
               yastrebksv/TennisCourtDetector heatmap CNN directly on video
               frames. Requires weights/court_tcd.pt.

  blue         Automatic (no ML): blue-contour corner detection only.
               Lower accuracy; useful when no ML weights are available.

Examples:
  # Recommended: automatic from video (amateur blue courts)
  python scripts/calibrate.py blue-resnet --video tennis.mp4 --out calib.json

  # Manual fallback from a hand-marked reference image
  python scripts/calibrate.py mark --image court_mark.jpg --out calib.json

  # Broadcast footage with TCD heatmap CNN
  python scripts/calibrate.py tcd --video tennis.mp4 --out calib.json
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



def _enclosing_trapezoid(mask, pad_frac=0.03, pad_const=10):
    """Fit a tight trapezoid around the non-zero region in mask, then expand outward.

    For each non-empty row, record the leftmost and rightmost non-zero pixel.
    Fit a line to each set via linear regression (the court sidelines are straight
    in perspective). The four corners are the line values at y_min / y_max.
    Expansion: pad = trap_height * pad_frac + pad_const  (proportional + constant).

    Returns (TL, TR, BR, BL) as integer pixel tuples, or None on failure.
    """
    H, W = mask.shape[:2]
    rows_with_content = np.where(np.any(mask > 0, axis=1))[0]
    if len(rows_with_content) == 0:
        return None
    y_min = int(rows_with_content[0])
    y_max = int(rows_with_content[-1])
    trap_h = y_max - y_min
    if trap_h < 10:
        return None

    min_row_width = max(5, int(W * 0.01))
    left_xs, right_xs, ys = [], [], []
    for y in rows_with_content:
        nz = np.where(mask[y] > 0)[0]
        if len(nz) < min_row_width:
            continue
        left_xs.append(float(nz[0]))
        right_xs.append(float(nz[-1]))
        ys.append(float(y))

    if len(ys) < 4:
        return None

    ya = np.array(ys)
    lfit = np.polyfit(ya, np.array(left_xs),  1)
    rfit = np.polyfit(ya, np.array(right_xs), 1)

    pad = int(trap_h * pad_frac + pad_const)

    def cl(x, y):
        return (int(np.clip(x, 0, W - 1)), int(np.clip(y, 0, H - 1)))

    tl = cl(np.polyval(lfit, y_min) - pad, y_min - pad)
    tr = cl(np.polyval(rfit, y_min) + pad, y_min - pad)
    br = cl(np.polyval(rfit, y_max) + pad, y_max + pad)
    bl = cl(np.polyval(lfit, y_max) - pad, y_max + pad)
    return tl, tr, br, bl



def cmd_blue_resnet(args):
    """ResNet50 regression court keypoint detector with perspective warp.

    1. Build blue mask → fit enclosing trapezoid → perspective-warp to rectangle.
    2. Run ResNet50 on N sampled frames in warp space, median-aggregate 14 keypoints.
    3. Inverse-warp keypoints back to original image coordinates.
    4. Estimate homography and save calib.

    Pass --no-warp to skip perspective rectification (broadcast/elevated footage).
    """
    from tennisvision.court.tcd_model import ResNet50CourtDetector
    from tennisvision.court.homography import estimate_homography

    det_blue = BlueContourDetector()
    sample, mask = det_blue.court_mask_from_video(args.video)
    if sample is None:
        raise SystemExit("could not read any frames from " + args.video)

    H_img, W_img = sample.shape[:2]
    det = ResNet50CourtDetector(weights=args.weights, device=args.device)

    M = M_inv = None
    out_w, out_h = args.warp_size

    if not args.no_warp:
        if mask is None:
            raise SystemExit("blue-contour mask failed; cannot build warp")
        quad = _enclosing_trapezoid(mask, pad_frac=0.0, pad_const=0)
        if quad is None:
            raise SystemExit("could not fit enclosing trapezoid from blue mask")
        tl, tr, _br, _bl = quad
        bl = (0, H_img - 1)
        br = (W_img - 1, H_img - 1)
        src = np.float32([tl, tr, br, bl])
        dst = np.float32([[0, 0], [out_w - 1, 0],
                          [out_w - 1, out_h - 1], [0, out_h - 1]])
        M = cv2.getPerspectiveTransform(src, dst)
        M_inv = np.linalg.inv(M)

        if args.debug:
            warped_sample = cv2.warpPerspective(sample, M, (out_w, out_h))
            cv2.imwrite(args.debug, warped_sample)
            trap_vis = sample.copy()
            pts_vis = np.array([tl, tr, br, bl], dtype=np.int32)
            cv2.polylines(trap_vis, [pts_vis], True, (0, 255, 0), 3)
            for pt, name in zip([tl, tr, br, bl], ["TL", "TR", "BR", "BL"]):
                cv2.circle(trap_vis, pt, 10, (0, 0, 255), -1)
                cv2.putText(trap_vis, name, (pt[0] + 10, pt[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imwrite(args.debug.replace(".jpg", "_trap.jpg"), trap_vis)
            print("debug images written:", args.debug,
                  "and", args.debug.replace(".jpg", "_trap.jpg"))

    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idxs = np.linspace(5, max(total - 5, 6), args.frames).astype(int)

    all_dets: dict = {}
    for fi in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, f = cap.read()
        if not ok:
            continue
        if M is not None:
            f = cv2.warpPerspective(f, M, (out_w, out_h))
        for kid, (x, y) in det.detect(f).items():
            all_dets.setdefault(kid, []).append((x, y))
    cap.release()

    # Median in detection space
    kps_med: dict = {}
    for kid, pts in all_dets.items():
        xm = float(np.median([p[0] for p in pts]))
        ym = float(np.median([p[1] for p in pts]))
        if M_inv is not None:
            pt_back = cv2.perspectiveTransform(
                np.array([[[xm, ym]]], dtype=np.float32), M_inv)[0][0]
            xm, ym = float(pt_back[0]), float(pt_back[1])
        kps_med[kid] = (xm, ym)

    print("ResNet50 detected %d keypoints over %d frames: %s" %
          (len(kps_med), len(frame_idxs), sorted(kps_med.keys())))

    hr = estimate_homography(kps_med, min_conf_count=4)
    if hr is None:
        raise SystemExit("could not fit homography from detected keypoints")

    calib = Calibration.from_homography_result(
        hr, image_size=(sample.shape[1], sample.shape[0]),
        keypoints_img=kps_med, source="blue-resnet",
    )
    save(calib, args.out)
    print("wrote %s  (used subset %s, reprojection err %.2f m)" %
          (args.out, hr.used_keypoints, hr.reprojection_error_m))
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

    br = sub.add_parser("blue-resnet", help="ResNet50 regression court keypoint detector")
    br.add_argument("--video",      required=True)
    br.add_argument("--weights",    default="weights/court_resnet.pth")
    br.add_argument("--device",     default="cpu")
    br.add_argument("--frames",     type=int, default=20)
    br.add_argument("--no-warp",    action="store_true", dest="no_warp",
                    help="skip perspective warp; run ResNet50 on original frames directly")
    br.add_argument("--warp-size",  type=int, nargs=2, default=[640, 360],
                    dest="warp_size", metavar=("W", "H"))
    br.add_argument("--out",        default="calib.json")
    br.add_argument("--vis",        default="calib_vis.jpg")
    br.add_argument("--debug",      default=None)
    br.set_defaults(func=cmd_blue_resnet)

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
