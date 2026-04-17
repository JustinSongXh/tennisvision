"""Right-side mini-map: court template + accumulated bounce dots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import cv2
import numpy as np

from ..court import reference as ref


@dataclass
class MinimapConfig:
    width_px: int = 150
    margin_px: int = 15
    fade_frames: int = 600


def _coord_mapper(width_px: int, height_px: int):
    W, L = ref.COURT_WIDTH_M, ref.COURT_LENGTH_M

    def rw_to_mm(x_m: float, y_m: float) -> tuple[int, int]:
        mx = int(round(x_m / W * (width_px - 1)))
        my = int(round((L - y_m) / L * (height_px - 1)))
        return mx, my
    return rw_to_mm


class Minimap:
    def __init__(self, cfg: MinimapConfig):
        self.cfg = cfg
        self._template, self._rw_to_mm, self._h = self._build()

    def _build(self):
        W_m, L_m = ref.COURT_WIDTH_M, ref.COURT_LENGTH_M
        mm_w = self.cfg.width_px
        mm_h = int(round(mm_w * (L_m / W_m)))
        img = np.full((mm_h, mm_w, 3), 240, dtype=np.uint8)

        rw_to_mm = _coord_mapper(mm_w, mm_h)
        BLACK = (30, 30, 30)
        NET   = (180, 180, 0)

        # doubles rectangle
        cv2.rectangle(img, rw_to_mm(0, 0), rw_to_mm(W_m, L_m), BLACK, 1)
        # singles
        s = ref.SINGLES_INSET
        cv2.line(img, rw_to_mm(s, 0),     rw_to_mm(s, L_m), BLACK, 1)
        cv2.line(img, rw_to_mm(W_m - s, 0), rw_to_mm(W_m - s, L_m), BLACK, 1)
        # service lines
        cv2.line(img, rw_to_mm(s, ref.SERVICE_Y),
                       rw_to_mm(W_m - s, ref.SERVICE_Y), BLACK, 1)
        cv2.line(img, rw_to_mm(s, L_m - ref.SERVICE_Y),
                       rw_to_mm(W_m - s, L_m - ref.SERVICE_Y), BLACK, 1)
        # center service line
        cv2.line(img, rw_to_mm(W_m / 2.0, ref.SERVICE_Y),
                       rw_to_mm(W_m / 2.0, L_m - ref.SERVICE_Y), BLACK, 1)
        # net
        cv2.line(img, rw_to_mm(0, ref.NET_Y), rw_to_mm(W_m, ref.NET_Y), NET, 2)
        return img, rw_to_mm, mm_h

    def render(self, bounces: Iterable[tuple[float, float, int]],
               frame_idx: int) -> np.ndarray:
        mm = self._template.copy()
        for x_m, y_m, f in bounces:
            age = max(0, frame_idx - f)
            fade = max(0.3, 1.0 - age / float(self.cfg.fade_frames))
            col = (int(30 * (1 - fade) + 30),
                   int(30 * (1 - fade) + 30),
                   int(255 * fade + 30 * (1 - fade)))
            pt = self._rw_to_mm(x_m, y_m)
            cv2.circle(mm, pt, 4, col, -1)
            cv2.circle(mm, pt, 4, (0, 0, 0), 1)
        return mm

    def overlay(self, frame: np.ndarray, bounces, frame_idx: int) -> None:
        mm = self.render(bounces, frame_idx)
        Hf, Wf = frame.shape[:2]
        mm_h, mm_w = mm.shape[:2]
        x0 = Wf - mm_w - self.cfg.margin_px
        y0 = self.cfg.margin_px
        cv2.rectangle(frame, (x0 - 3, y0 - 3),
                              (x0 + mm_w + 2, y0 + mm_h + 2),
                              (255, 255, 255), 2)
        frame[y0:y0 + mm_h, x0:x0 + mm_w] = mm
