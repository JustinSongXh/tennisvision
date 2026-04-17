"""Homography estimation with best-of-4-subsets voting (after CourtCheck).

Given possibly-noisy 14 keypoint detections, try every predefined 4-point
subset in reference.COURT_CONF and pick the homography whose REMAINING
(unused) keypoints reproject to the reference with the smallest mean
error.  This gives RANSAC-like robustness without ever needing RANSAC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from . import reference as ref


@dataclass
class HomographyResult:
    H_img_to_real: np.ndarray       # 3x3 image pixel -> real meters
    H_real_to_img: np.ndarray       # 3x3 inverse
    subset: tuple[int, int, int, int]
    reprojection_error_m: float     # mean error over the unused keypoints
    used_keypoints: tuple[int, int, int, int]


def _project(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply 3x3 homography H to an Nx2 array; returns Nx2."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    h = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    p = (H @ h.T).T
    w = p[:, 2:3]
    w[np.abs(w) < 1e-12] = 1e-12
    return p[:, :2] / w


def project_image_to_court(H, pt):
    out = _project(H, np.array([pt]))[0]
    return float(out[0].item()), float(out[1].item())


def project_court_to_image(H_inv, pt_m):
    out = _project(H_inv, np.array([pt_m]))[0]
    return float(out[0].item()), float(out[1].item())


def estimate_homography(
    keypoints_img: dict[int, tuple[float, float]],
    *,
    min_conf_count: int = 6,
) -> Optional[HomographyResult]:
    """Pick the best homography from candidate 4-point subsets.

    keypoints_img: dict mapping keypoint_id -> (x, y) in image pixels.
                   Only ids present in the dict are considered detected.
    min_conf_count: minimum total number of detected keypoints required.
    """
    detected_ids = set(keypoints_img.keys())
    if len(detected_ids) < min_conf_count:
        return None

    best: Optional[HomographyResult] = None

    for subset in ref.COURT_CONF:
        if not all(kid in detected_ids for kid in subset):
            continue

        src = np.array([keypoints_img[k] for k in subset], dtype=np.float32)
        dst = np.array([ref.KEYPOINTS_M[k]  for k in subset], dtype=np.float32)
        H, _ = cv2.findHomography(src, dst, method=0)
        if H is None:
            continue

        # Score on the OTHER detected keypoints (not in the subset).
        other_ids = [k for k in detected_ids if k not in subset]
        if not other_ids:
            # Degenerate: can't validate.  Accept only if nothing else
            # has been found yet.
            err = float("inf")
        else:
            img_pts  = np.array([keypoints_img[k]   for k in other_ids], dtype=np.float64)
            real_pts = np.array([ref.KEYPOINTS_M[k] for k in other_ids], dtype=np.float64)
            projected = _project(H, img_pts)
            err = float(np.mean(np.linalg.norm(projected - real_pts, axis=1)))

        if best is None or err < best.reprojection_error_m:
            try:
                H_inv = np.linalg.inv(H)
            except np.linalg.LinAlgError:
                continue
            best = HomographyResult(
                H_img_to_real=H,
                H_real_to_img=H_inv,
                subset=tuple(subset),
                reprojection_error_m=err,
                used_keypoints=tuple(subset),
            )

    return best


def homography_from_4_corners(
    corners_img: list[tuple[float, float]],
    corners_real: Optional[list[tuple[float, float]]] = None,
) -> HomographyResult:
    """Convenience: build a HomographyResult from exactly 4 given image
    corners in FL/FR/NR/NL order (doubles).  Used by the red-mark
    calibration path where we've already computed corners manually.
    """
    if corners_real is None:
        corners_real = [ref.KEYPOINTS_M[kid] for kid in ref.doubles_corner_ids()]
    src = np.array(corners_img,  dtype=np.float32)
    dst = np.array(corners_real, dtype=np.float32)
    H, _ = cv2.findHomography(src, dst, method=0)
    if H is None:
        raise ValueError("findHomography failed — are the 4 points non-collinear?")
    H_inv = np.linalg.inv(H)
    return HomographyResult(
        H_img_to_real=H,
        H_real_to_img=H_inv,
        subset=ref.doubles_corner_ids(),
        reprojection_error_m=0.0,   # no independent validation points
        used_keypoints=ref.doubles_corner_ids(),
    )
