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


def cmd_blue_warp(args):
    """Perspective-warp the court trapezoid to a rectangle, run TCD, then unproject.

    1. Build accumulated blue mask → fit enclosing trapezoid (no padding,
       bottom corners at frame edges so near players don't clip the region).
    2. cv2.getPerspectiveTransform: trapezoid → rectangle (TCD input size).
    3. Warp each sampled frame, run TCD, median-aggregate keypoints in warp space.
    4. Apply inverse warp to bring keypoints back to original image coordinates.
    5. Estimate homography and save calib.
    """
    from tennisvision.court.tcd_model import HeatmapDetector
    from tennisvision.court.homography import estimate_homography

    det_blue = BlueContourDetector()
    sample, mask = det_blue.court_mask_from_video(args.video)
    if sample is None or mask is None:
        raise SystemExit("blue-contour mask failed")

    H, W = sample.shape[:2]

    # Fit trapezoid with NO padding — raw line-fit corners only
    quad = _enclosing_trapezoid(mask, pad_frac=0.0, pad_const=0)
    if quad is None:
        raise SystemExit("could not fit enclosing trapezoid from blue mask")

    tl, tr, _br, _bl = quad
    bl = (0, H - 1)
    br = (W - 1, H - 1)

    # Perspective warp: trapezoid → rectangle matching TCD's native input size
    out_w, out_h = args.warp_size
    src = np.float32([tl, tr, br, bl])
    dst = np.float32([[0, 0], [out_w - 1, 0],
                      [out_w - 1, out_h - 1], [0, out_h - 1]])
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = np.linalg.inv(M)

    if args.debug:
        warped_sample = cv2.warpPerspective(sample, M, (out_w, out_h))
        cv2.imwrite(args.debug, warped_sample)
        # Also save the trapezoid outline on the original frame
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

    # Multi-frame TCD aggregation in warped space
    det_tcd = HeatmapDetector(weights=args.weights, device=args.device,
                              low_thresh=args.tcd_thresh)
    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idxs = np.linspace(5, max(total - 5, 6), args.frames).astype(int)

    all_dets: dict = {}
    for fi in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, f = cap.read()
        if not ok:
            continue
        warped_f = cv2.warpPerspective(f, M, (out_w, out_h))
        for kid, (xw, yw) in det_tcd.detect(warped_f).items():
            all_dets.setdefault(kid, []).append((xw, yw))
    cap.release()

    # Median in warp space, then inverse-project back to original
    kps = {}
    for kid, pts in all_dets.items():
        xw = float(np.median([p[0] for p in pts]))
        yw = float(np.median([p[1] for p in pts]))
        pt_back = cv2.perspectiveTransform(
            np.array([[[xw, yw]]], dtype=np.float32), M_inv)[0][0]
        kps[kid] = (float(pt_back[0]), float(pt_back[1]))

    print("TCD detected %d keypoints over %d frames: %s" %
          (len(kps), len(frame_idxs), sorted(kps.keys())))

    if len(kps) < 4:
        raise SystemExit("only %d keypoints detected; need >=4" % len(kps))

    hr = estimate_homography(kps, min_conf_count=4)
    if hr is None:
        raise SystemExit("could not fit homography from detected keypoints")

    calib = Calibration.from_homography_result(
        hr, image_size=(sample.shape[1], sample.shape[0]),
        keypoints_img=kps, source="blue-warp-tcd",
    )
    save(calib, args.out)
    print("wrote %s  (used subset %s, reprojection err %.2f m)" %
          (args.out, hr.used_keypoints, hr.reprojection_error_m))
    if args.vis:
        _draw_verification(sample, calib, args.vis)
        print("wrote", args.vis)


def cmd_blue_trap(args):
    """Blue-contour calibration with enclosing-trapezoid pre-filter.

    Step 1: build accumulated blue mask (same as `blue` command).
    Step 2: fit enclosing trapezoid to that mask; bottom corners extend
            to frame edges so near-baseline players are excluded.
    Step 3: AND the trapezoid mask with the accumulated blue mask.
    Step 4: run corner detection on the filtered mask.
    """
    det = BlueContourDetector()
    sample, mask = det.court_mask_from_video(args.video)
    if sample is None or mask is None:
        raise SystemExit("blue-contour mask failed")

    H, W = sample.shape[:2]
    quad = _enclosing_trapezoid(mask, pad_frac=args.pad_frac,
                                pad_const=args.pad_const)
    if quad is None:
        raise SystemExit("could not fit enclosing trapezoid from blue mask")

    tl, tr, _br, _bl = quad
    pts = np.array([tl, tr, (W - 1, H - 1), (0, H - 1)], dtype=np.int32)
    trap_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(trap_mask, [pts], 255)

    # Apply trapezoid to the accumulated blue mask
    filtered = cv2.bitwise_and(mask, mask, mask=trap_mask)

    if args.debug:
        dbg = sample.copy()
        cv2.polylines(dbg, [pts], True, (0, 255, 0), 3)
        for pt, name in zip([tl, tr, (W-1, H-1), (0, H-1)],
                            ["TL", "TR", "BR", "BL"]):
            cv2.circle(dbg, pt, 10, (0, 0, 255), -1)
            cv2.putText(dbg, name, (pt[0] + 10, pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imwrite(args.debug, dbg)
        print("debug image written to", args.debug)

    corners = det._corners_from_mask(filtered, sample.shape)
    if len(corners) != 4:
        raise SystemExit("corner detection failed (got %d corners)" % len(corners))

    four = [corners[0], corners[1], corners[2], corners[3]]
    hr = homography_from_4_corners(four)

    kps = {}
    H_inv = hr.H_real_to_img
    for kid, (xm, ym) in ref.KEYPOINTS_M.items():
        v = np.array([xm, ym, 1.0]); p = H_inv @ v
        kps[kid] = (float(p[0] / p[2]), float(p[1] / p[2]))

    calib = Calibration.from_homography_result(
        hr, image_size=(sample.shape[1], sample.shape[0]),
        keypoints_img=kps, source="blue-trap",
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


def cmd_blue_tcd(args):
    """Hybrid: blue-contour mask → fit enclosing trapezoid → expand → TCD."""
    from tennisvision.court.tcd_model import HeatmapDetector  # noqa
    from tennisvision.court.homography import estimate_homography

    det_blue = BlueContourDetector()
    sample, mask = det_blue.court_mask_from_video(args.video)
    if sample is None:
        raise SystemExit("could not read any frames from " + args.video)

    H, W = sample.shape[:2]

    if mask is not None:
        quad = _enclosing_trapezoid(mask, pad_frac=args.pad_frac,
                                    pad_const=args.pad_const)
        if quad is not None:
            tl, tr, _br, _bl = quad
            # Bottom corners: extend to frame edges so near-baseline players
            # don't occlude the court region from TCD's view.
            bl = (0, H - 1)
            br = (W - 1, H - 1)
            pts = np.array([tl, tr, br, bl], dtype=np.int32)
            trap_mask = np.zeros((H, W), dtype=np.uint8)
            cv2.fillPoly(trap_mask, [pts], 255)
            masked_frame = cv2.bitwise_and(sample, sample, mask=trap_mask)
            if args.debug:
                dbg = sample.copy()
                cv2.polylines(dbg, [pts], True, (0, 255, 0), 3)
                for pt, name in zip([tl, tr, br, bl], ["TL", "TR", "BR", "BL"]):
                    cv2.circle(dbg, pt, 10, (0, 0, 255), -1)
                    cv2.putText(dbg, name, (pt[0] + 10, pt[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imwrite(args.debug, dbg)
                cv2.imwrite(args.debug.replace(".jpg", "_masked.jpg"), masked_frame)
                print("debug images written:", args.debug,
                      "and", args.debug.replace(".jpg", "_masked.jpg"))
        else:
            print("WARNING: trapezoid fit failed; using raw blue mask")
            masked_frame = cv2.bitwise_and(sample, sample, mask=mask)
    else:
        print("WARNING: blue-contour mask failed; running TCD on unmasked frame")
        masked_frame = sample

    det_tcd = HeatmapDetector(weights=args.weights, device=args.device,
                              low_thresh=args.tcd_thresh)

    # Multi-frame aggregation: sample N frames across the video, apply the same
    # trapezoid mask to each, run TCD, and median-aggregate positions per keypoint.
    # Players move between frames so different court keypoints become visible.
    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idxs = np.linspace(5, max(total - 5, 6), args.frames).astype(int)

    all_dets: dict = {}
    for fi in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, f = cap.read()
        if not ok:
            continue
        mf = cv2.bitwise_and(f, f, mask=trap_mask if mask is not None else np.full(
            (H, W), 255, np.uint8))
        for kid, (x, y) in det_tcd.detect(mf).items():
            all_dets.setdefault(kid, []).append((x, y))
    cap.release()

    kps = {kid: (float(np.median([p[0] for p in pts])),
                 float(np.median([p[1] for p in pts])))
           for kid, pts in all_dets.items()}
    print("TCD aggregated %d keypoints over %d frames: %s" %
          (len(kps), len(frame_idxs), sorted(kps.keys())))

    if len(kps) < 4:
        raise SystemExit("only %d keypoints detected; need >=4" % len(kps))

    hr = estimate_homography(kps, min_conf_count=4)
    if hr is None:
        raise SystemExit("could not fit homography from detected keypoints")

    calib = Calibration.from_homography_result(
        hr, image_size=(sample.shape[1], sample.shape[0]),
        keypoints_img=kps, source="blue-tcd",
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

    bw = sub.add_parser("blue-warp", help="perspective-warp court to rectangle, run TCD, unproject back")
    bw.add_argument("--video",      required=True)
    bw.add_argument("--weights",    default="weights/court_tcd.pt")
    bw.add_argument("--device",     default="cpu")
    bw.add_argument("--warp-size",  type=int, nargs=2, default=[640, 360],
                    dest="warp_size", metavar=("W", "H"),
                    help="output rectangle size for TCD (default 640 360)")
    bw.add_argument("--frames",     type=int, default=20)
    bw.add_argument("--tcd-thresh", type=int, default=150, dest="tcd_thresh")
    bw.add_argument("--out",        default="calib.json")
    bw.add_argument("--vis",        default="calib_vis.jpg")
    bw.add_argument("--debug",      default=None)
    bw.set_defaults(func=cmd_blue_warp)

    btr = sub.add_parser("blue-trap", help="blue-contour + enclosing-trapezoid pre-filter (no TCD needed)")
    btr.add_argument("--video",      required=True)
    btr.add_argument("--pad-frac",   type=float, default=0.03, dest="pad_frac")
    btr.add_argument("--pad-const",  type=int,   default=10,   dest="pad_const")
    btr.add_argument("--out",        default="calib.json")
    btr.add_argument("--vis",        default="calib_vis.jpg")
    btr.add_argument("--debug",      default=None)
    btr.set_defaults(func=cmd_blue_trap)

    bt = sub.add_parser("blue-tcd", help="hybrid: blue-contour mask + TCD CNN (handles partial courts)")
    bt.add_argument("--video",      required=True)
    bt.add_argument("--weights",    default="weights/court_tcd.pt")
    bt.add_argument("--device",     default="cpu")
    bt.add_argument("--pad-frac",   type=float, default=0.03,
                    dest="pad_frac",  help="expand trapezoid by trap_height*frac (default 0.03)")
    bt.add_argument("--pad-const",  type=int, default=10,
                    dest="pad_const", help="additional constant pixel expansion (default 10)")
    bt.add_argument("--frames",     type=int, default=20,
                    help="number of video frames to aggregate TCD over (default 20)")
    bt.add_argument("--tcd-thresh", type=int, default=170,
                    dest="tcd_thresh", help="TCD heatmap threshold 0-255 (default 170, lower=more detections)")
    bt.add_argument("--out",        default="calib.json")
    bt.add_argument("--vis",        default="calib_vis.jpg")
    bt.add_argument("--debug",      default=None, help="save trapezoid debug image to this path")
    bt.set_defaults(func=cmd_blue_tcd)

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
