"""Classical CV ball detector — HSV yellow-green + MOG2 motion.

This is the CPU-only fallback.  It's cheap, zero-dep beyond OpenCV, but
misses the ball at trajectory apex (low motion) and can be fooled by
wind-blown tree / fence regions.  Use tracknet.TrackNetDetector for
higher recall when GPU/ONNX is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np


@dataclass
class Candidate:
    x: int
    y: int
    area: float
    circularity: float


class HSVMotionDetector:
    """Per-frame ball candidate extractor.

    Usage:
        det = HSVMotionDetector(cfg)
        for frame in video:
            cands = det.detect(frame)
    """

    def __init__(
        self,
        hsv_low: Iterable[int] = (25, 60, 120),
        hsv_high: Iterable[int] = (50, 255, 255),
        min_area: int = 3,
        max_area: int = 400,
        min_circularity: float = 0.55,
        mog_var_threshold: float = 25,
        mog_history: int = 500,
        player_min_area: int = 1500,
        player_max_area: int = 60000,
        player_max_w_frac: float = 0.55,
        player_max_h_frac: float = 0.75,
        player_bbox_pad: int = 0,
    ):
        self.hsv_low  = np.array(list(hsv_low),  dtype=np.uint8)
        self.hsv_high = np.array(list(hsv_high), dtype=np.uint8)
        self.min_area = min_area
        self.max_area = max_area
        self.min_circularity = min_circularity
        self.player_min_area = player_min_area
        self.player_max_area = player_max_area
        self.player_max_w_frac = player_max_w_frac
        self.player_max_h_frac = player_max_h_frac
        self.player_bbox_pad = player_bbox_pad

        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=mog_history, varThreshold=mog_var_threshold, detectShadows=False,
        )
        self._k_open = np.ones((2, 2), np.uint8)
        self._k_dil  = np.ones((3, 3), np.uint8)

    def detect(self, frame: np.ndarray) -> list[Candidate]:
        H, W = frame.shape[:2]

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv, self.hsv_low, self.hsv_high)

        fg = self._bg.apply(frame)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

        player_bboxes = self._player_bboxes(fg, W, H)

        mask = cv2.bitwise_and(color_mask, fg)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._k_open)
        mask = cv2.dilate(mask, self._k_dil, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out: list[Candidate] = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area or area > self.max_area:
                continue
            perim = cv2.arcLength(c, True)
            if perim <= 0:
                continue
            circ = 4.0 * np.pi * area / (perim * perim)
            if circ < self.min_circularity:
                continue
            (cx, cy), _ = cv2.minEnclosingCircle(c)
            cx, cy = int(cx), int(cy)
            if _point_in_any(cx, cy, player_bboxes):
                continue
            out.append(Candidate(cx, cy, float(area), float(circ)))
        return out

    def _player_bboxes(self, fg: np.ndarray, W: int, H: int):
        out = []
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            a = cv2.contourArea(c)
            if a < self.player_min_area or a > self.player_max_area:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            if bw > self.player_max_w_frac * W or bh > self.player_max_h_frac * H:
                continue
            p = self.player_bbox_pad
            out.append((x - p, y - p, x + bw + p, y + bh + p))
        return out


def _point_in_any(x: int, y: int, boxes) -> bool:
    for x1, y1, x2, y2 in boxes:
        if x1 <= x <= x2 and y1 <= y <= y2:
            return True
    return False
