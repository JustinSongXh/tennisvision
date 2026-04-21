"""Serialization and caching of court calibration."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .homography import HomographyResult


@dataclass
class Calibration:
    H_img_to_real: np.ndarray           # 3x3
    H_real_to_img: np.ndarray           # 3x3
    image_size: tuple[int, int]          # (W, H)
    keypoints_img: dict[int, tuple[float, float]] = field(default_factory=dict)
    source: str = ""                     # auto | mark | manual
    reprojection_error_m: float = 0.0

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "image_size": list(self.image_size),
            "reprojection_error_m": self.reprojection_error_m,
            "keypoints_img": {int(k): list(v) for k, v in self.keypoints_img.items()},
            "H_img_to_real": self.H_img_to_real.tolist(),
            "H_real_to_img": self.H_real_to_img.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Calibration":
        return cls(
            H_img_to_real=np.array(d["H_img_to_real"], dtype=np.float64),
            H_real_to_img=np.array(d["H_real_to_img"], dtype=np.float64),
            image_size=tuple(d.get("image_size", [0, 0])),
            keypoints_img={int(k): tuple(v) for k, v in d.get("keypoints_img", {}).items()},
            source=d.get("source", ""),
            reprojection_error_m=float(d.get("reprojection_error_m", 0.0)),
        )

    @classmethod
    def from_homography_result(
        cls, hr: HomographyResult, image_size: tuple[int, int],
        keypoints_img: Optional[dict[int, tuple[float, float]]] = None,
        source: str = "",
    ) -> "Calibration":
        return cls(
            H_img_to_real=hr.H_img_to_real,
            H_real_to_img=hr.H_real_to_img,
            image_size=image_size,
            keypoints_img=keypoints_img or {},
            source=source,
            reprojection_error_m=hr.reprojection_error_m,
        )


    def court_h_strip_mask(self, W: int, H: int, margin_px: int = 30) -> np.ndarray:
        """Binary mask (H×W uint8=255) that keeps only the horizontal band
        between the left and right doubles sidelines projected onto the image.

        Vertical extent is NOT constrained — the ball can fly above the
        court baseline or out of frame.  Only the left/right sidelines are
        used, eliminating adjacent-court interference.

        The mask is computed once per calibration and can be cached.
        """
        from .reference import COURT_WIDTH_M, COURT_LENGTH_M

        def _proj_m(xm: float, ym: float) -> tuple[float, float]:
            v = np.array([xm, ym, 1.0])
            p = self.H_real_to_img @ v
            return float(p[0] / p[2]), float(p[1] / p[2])

        # Two points on each sideline span the full court depth
        left_pts  = [_proj_m(0.0,           0.0),
                     _proj_m(0.0,           COURT_LENGTH_M)]
        right_pts = [_proj_m(COURT_WIDTH_M, 0.0),
                     _proj_m(COURT_WIDTH_M, COURT_LENGTH_M)]

        # Fit x = a*y + b for each sideline
        lfit = np.polyfit([p[1] for p in left_pts],  [p[0] for p in left_pts],  1)
        rfit = np.polyfit([p[1] for p in right_pts], [p[0] for p in right_pts], 1)

        ys = np.arange(H, dtype=np.float32)
        xl = (np.polyval(lfit, ys) - margin_px).astype(int).clip(0, W - 1)
        xr = (np.polyval(rfit, ys) + margin_px).astype(int).clip(0, W - 1)

        mask = np.zeros((H, W), dtype=np.uint8)
        for y in range(H):
            if xr[y] > xl[y]:
                mask[y, xl[y]: xr[y] + 1] = 255
        return mask


def save(calib: Calibration, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(calib.to_dict(), f, indent=2)


def load(path: str) -> Calibration:
    with open(path) as f:
        d = json.load(f)
    return Calibration.from_dict(d)
