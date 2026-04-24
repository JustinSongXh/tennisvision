"""Serialization and caching of court calibration."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import reference as ref
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

    def on_court_polygon_img(self, margin_m: float = 0.0) -> np.ndarray:
        """Image-space quadrilateral covering the court plus `margin_m`
        of court-meter padding outside every side.

        Built by projecting 4 real-world corners through H_real_to_img,
        so perspective is baked in.  Airborne balls above the court
        still project INSIDE this polygon (they're above court ground,
        not above an adjacent court), so this is resilient to the z=0
        projection error that makes court-coord X-gates unreliable for
        high balls.

        Returns a (4, 2) float32 array of image pixels in the order
        near-left, near-right, far-right, far-left (CCW in court space).
        """
        m = float(margin_m)
        corners_real = np.array([
            [-m,                        -m,                         1.0],
            [ref.COURT_WIDTH_M + m,     -m,                         1.0],
            [ref.COURT_WIDTH_M + m,     ref.COURT_LENGTH_M + m,     1.0],
            [-m,                        ref.COURT_LENGTH_M + m,     1.0],
        ], dtype=np.float64)
        H = self.H_real_to_img
        corners_img = (H @ corners_real.T).T           # (4, 3)
        corners_img = corners_img[:, :2] / corners_img[:, 2:3]
        return corners_img.astype(np.float32)


def save(calib: Calibration, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(calib.to_dict(), f, indent=2)


def load(path: str) -> Calibration:
    with open(path) as f:
        d = json.load(f)
    return Calibration.from_dict(d)
