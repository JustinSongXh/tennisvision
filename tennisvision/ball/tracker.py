"""Multi-object Kalman tracker with champion selection.

One Track per candidate cluster; per-frame greedy nearest-neighbour
association; Kalman-predicted position drives the gating.  Tracks get
validated once they have >= MIN_LEN observations whose average speed
passes a threshold check.  Champion selection uses projectile-residual
scoring: score = n_points / (1 + residual), so noise tracks with high
residual lose even if they have many points.

When a homography (H_img_to_real) is provided, speed validation operates
in court-coordinate meters instead of pixels, eliminating perspective
distortion between near and far court.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class TrackPoint:
    x: int
    y: int
    frame: int


def _px_to_court(x: float, y: float, H: np.ndarray) -> tuple[float, float]:
    p = H @ np.array([x, y, 1.0], dtype=np.float64)
    return float(p[0] / p[2]), float(p[1] / p[2])


@dataclass
class TrackerConfig:
    gate_px: float = 80.0
    max_gap_frames: int = 5
    min_len: int = 3
    # Court-coordinate speed thresholds (m/s), used when homography available
    min_speed_ms: float = 3.0
    max_speed_ms: float = 90.0
    # Pixel-space fallback (px/frame), used when no homography
    min_speed_px: float = 5.0
    max_speed_px: float = 200.0
    # Displacement below this (meters) marks track as static
    static_threshold_m: float = 0.5
    static_window: int = 10
    render_tail: int = 45


class Track:
    _next_id = 0

    def __init__(self, x: int, y: int, frame: int, cfg: TrackerConfig,
                 H: Optional[np.ndarray] = None, fps: float = 30.0):
        self.id = Track._next_id
        Track._next_id += 1
        self.pts: list[TrackPoint] = [TrackPoint(x, y, frame)]
        self.last_det_frame = frame
        self.validated = False
        self.is_static = False
        self.pred_xy: tuple[float, float] = (float(x), float(y))
        self._cfg = cfg
        self._H = H
        self._fps = fps

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
        if not self.validated and len(self.pts) >= self._cfg.min_len:
            if self._H is not None:
                s = self.avg_speed_ms()
                lo, hi = self._cfg.min_speed_ms, self._cfg.max_speed_ms
            else:
                s = self.avg_speed_px()
                lo, hi = self._cfg.min_speed_px, self._cfg.max_speed_px
            if lo <= s <= hi:
                self.validated = True
        self._update_static()

    def _update_static(self) -> None:
        """Mark track static if recent displacement < threshold."""
        w = self._cfg.static_window
        recent = self.pts[-w:]
        if len(recent) < 3:
            self.is_static = False
            return
        p0, p1 = recent[0], recent[-1]
        if self._H is not None:
            c0 = _px_to_court(p0.x, p0.y, self._H)
            c1 = _px_to_court(p1.x, p1.y, self._H)
            disp = float(np.hypot(c1[0] - c0[0], c1[1] - c0[1]))
            self.is_static = disp < self._cfg.static_threshold_m
        else:
            disp = float(np.hypot(p1.x - p0.x, p1.y - p0.y))
            self.is_static = disp < 5.0

    def avg_speed_px(self) -> float:
        """Average speed in px/frame."""
        if len(self.pts) < 2:
            return 0.0
        d = 0.0; t = 0
        for i in range(1, len(self.pts)):
            p1 = self.pts[i]; p0 = self.pts[i - 1]
            d += float(np.hypot(p1.x - p0.x, p1.y - p0.y))
            t += (p1.frame - p0.frame)
        return d / max(t, 1)

    def avg_speed_ms(self) -> float:
        """Average speed in m/s using court-coordinate projection."""
        if len(self.pts) < 2 or self._H is None:
            return 0.0
        d = 0.0; t = 0
        for i in range(1, len(self.pts)):
            p1 = self.pts[i]; p0 = self.pts[i - 1]
            cx1, cy1 = _px_to_court(p1.x, p1.y, self._H)
            cx0, cy0 = _px_to_court(p0.x, p0.y, self._H)
            d += float(np.hypot(cx1 - cx0, cy1 - cy0))
            t += (p1.frame - p0.frame)
        return d / max(t, 1) * self._fps

    def projectile_residual(self, window: int = 15) -> float:
        """Mean residual of a parabolic fit on recent points.

        Fits x(t) = a0 + a1*t (linear) and y(t) = b0 + b1*t + b2*t^2
        (quadratic, gravity) and returns the mean Euclidean residual.
        Returns 0.0 if there are too few points for a reliable fit.
        """
        pts = self.pts[-window:]
        if len(pts) < 5:
            return 0.0
        t0 = pts[0].frame
        ts = np.array([p.frame - t0 for p in pts], dtype=np.float64)
        xs = np.array([p.x for p in pts], dtype=np.float64)
        ys = np.array([p.y for p in pts], dtype=np.float64)
        cx = np.polyfit(ts, xs, 1)
        x_fit = np.polyval(cx, ts)
        cy = np.polyfit(ts, ys, 2)
        y_fit = np.polyval(cy, ts)
        return float(np.mean(np.hypot(x_fit - xs, y_fit - ys)))


class MultiTrackManager:
    def __init__(self, cfg: TrackerConfig, *,
                 H_img_to_real: Optional[np.ndarray] = None,
                 fps: float = 30.0):
        self.cfg = cfg
        self.tracks: list[Track] = []
        self._H = H_img_to_real
        self._fps = fps

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
                    cx, cy, frame_idx, self.cfg,
                    H=self._H, fps=self._fps,
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
                 if t.validated and not t.is_static
                 and frame_idx - t.last_det_frame <= self.cfg.max_gap_frames]
        if not cands:
            return None
        cands.sort(key=lambda t: self._champion_score(t, frame_idx), reverse=True)
        return cands[0]

    @staticmethod
    def _champion_score(t: Track, frame_idx: int) -> float:
        residual = t.projectile_residual()
        n = len(t.pts)
        score = n / (1.0 + residual)
        age = frame_idx - t.last_det_frame
        return score - age * 0.01
