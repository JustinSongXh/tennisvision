"""Multi-object Kalman tracker with champion selection.

One Track per candidate cluster; per-frame greedy nearest-neighbour
association; Kalman-predicted position drives the gating.  Tracks get
validated once they have >= MIN_LEN observations whose average image
speed is in [MIN_SPEED, MAX_SPEED] px/frame — this simultaneously
rejects (a) vibrating-pixel noise and (b) teleporting false matches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


@dataclass
class TrackPoint:
    x: int
    y: int
    frame: int


class Track:
    _next_id = 0

    def __init__(self, x: int, y: int, frame: int, min_len: int, min_speed: float, max_speed: float):
        self.id = Track._next_id
        Track._next_id += 1
        self.pts: list[TrackPoint] = [TrackPoint(x, y, frame)]
        self.last_det_frame = frame
        self.validated = False
        self.pred_xy: tuple[float, float] = (float(x), float(y))
        self._min_len = min_len
        self._min_speed = min_speed
        self._max_speed = max_speed

        kf = cv2.KalmanFilter(4, 2)
        kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]], dtype=np.float32)
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]], dtype=np.float32)
        kf.processNoiseCov     = np.diag([1.0, 1.0, 4.0, 4.0]).astype(np.float32)
        kf.measurementNoiseCov = (np.eye(2, dtype=np.float32) * 2.0)
        kf.errorCovPost        = (np.eye(4, dtype=np.float32) * 10.0)
        kf.statePost = np.array([[x], [y], [0.0], [0.0]], dtype=np.float32)
        self.kf = kf

    def predict(self):
        p = self.kf.predict().ravel()
        self.pred_xy = (float(p[0]), float(p[1]))
        return self.pred_xy

    def correct(self, x: int, y: int, frame: int) -> None:
        self.kf.correct(np.array([[np.float32(x)], [np.float32(y)]], dtype=np.float32))
        self.pts.append(TrackPoint(x, y, frame))
        self.last_det_frame = frame
        if not self.validated and len(self.pts) >= self._min_len:
            s = self.avg_speed()
            if self._min_speed <= s <= self._max_speed:
                self.validated = True

    def avg_speed(self) -> float:
        if len(self.pts) < 2:
            return 0.0
        d = 0.0; t = 0
        for i in range(1, len(self.pts)):
            p1 = self.pts[i]; p0 = self.pts[i - 1]
            d += float(np.hypot(p1.x - p0.x, p1.y - p0.y))
            t += (p1.frame - p0.frame)
        return d / max(t, 1)


@dataclass
class TrackerConfig:
    gate_px: float = 80.0
    max_gap_frames: int = 5
    min_len: int = 3
    min_speed: float = 10.0
    max_speed: float = 150.0
    render_tail: int = 45


class MultiTrackManager:
    def __init__(self, cfg: TrackerConfig):
        self.cfg = cfg
        self.tracks: list[Track] = []

    def update(self, candidates: list[tuple[int, int]], frame_idx: int) -> None:
        active = [t for t in self.tracks
                  if frame_idx - t.last_det_frame <= self.cfg.max_gap_frames]
        for t in active:
            t.predict()

        active.sort(key=lambda t: -len(t.pts))
        used: set[int] = set()
        for t in active:
            px, py = t.pred_xy
            best_i, best_d = -1, self.cfg.gate_px
            for i, (cx, cy) in enumerate(candidates):
                if i in used:
                    continue
                d = float(np.hypot(cx - px, cy - py))
                if d < best_d:
                    best_d, best_i = d, i
            if best_i >= 0:
                cx, cy = candidates[best_i]
                t.correct(cx, cy, frame_idx)
                used.add(best_i)

        for i, (cx, cy) in enumerate(candidates):
            if i not in used:
                self.tracks.append(Track(
                    cx, cy, frame_idx,
                    self.cfg.min_len, self.cfg.min_speed, self.cfg.max_speed,
                ))

        # Pruning: validated tracks stay for render_tail frames so the
        # trail fades gracefully; unvalidated drops after max_gap.
        fresh = []
        for t in self.tracks:
            age = frame_idx - t.last_det_frame
            if t.validated:
                if age <= self.cfg.render_tail:
                    fresh.append(t)
            else:
                if age <= self.cfg.max_gap_frames:
                    fresh.append(t)
        self.tracks = fresh

    def champion(self, frame_idx: int) -> Optional[Track]:
        cands = [t for t in self.tracks
                 if t.validated and frame_idx - t.last_det_frame <= self.cfg.max_gap_frames]
        if not cands:
            return None
        cands.sort(key=lambda t: (len(t.pts), -(frame_idx - t.last_det_frame)), reverse=True)
        return cands[0]
