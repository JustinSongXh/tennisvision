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
    # Tracklet stitching (offline post-pass)
    stitch_max_gap_s: float = 0.5       # max gap in seconds
    stitch_pos_threshold: float = 80.0   # max position error in pixels
    stitch_vel_angle_deg: float = 45.0   # max velocity angle difference
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

    def _exit_state(self, window: int = 8) -> Optional[tuple]:
        """Parabolic coefficients and exit velocity from last `window` points.

        Returns (coeffs, velocity) where:
          coeffs = (cx_poly1, cy_poly2) for extrapolation
          velocity = (vx, vy) in px/frame at the last point
        Returns None if too few points.
        """
        pts = self.pts[-window:]
        if len(pts) < 4:
            return None
        t0 = pts[0].frame
        ts = np.array([p.frame - t0 for p in pts], dtype=np.float64)
        xs = np.array([p.x for p in pts], dtype=np.float64)
        ys = np.array([p.y for p in pts], dtype=np.float64)
        cx = np.polyfit(ts, xs, 1)
        cy = np.polyfit(ts, ys, 2)
        t_last = ts[-1]
        vx = cx[0]
        vy = 2 * cy[0] * t_last + cy[1]
        return (cx, cy), np.array([vx, vy])

    def _entry_velocity(self, window: int = 4) -> Optional[np.ndarray]:
        """Velocity (vx, vy) in px/frame at the first point."""
        pts = self.pts[:window]
        if len(pts) < 2:
            return None
        dt = pts[-1].frame - pts[0].frame
        if dt == 0:
            return None
        vx = (pts[-1].x - pts[0].x) / dt
        vy = (pts[-1].y - pts[0].y) / dt
        return np.array([vx, vy])


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

    def stitch_tracklets(self) -> int:
        """Offline post-pass: merge broken tracklets via projectile extrapolation.

        For each gap between validated tracklets A (ends) and B (starts),
        extrapolate A's parabolic arc to B's start frame. If position error
        and velocity angle are within thresholds, merge B into A.

        Returns the number of merges performed.
        """
        max_gap = int(round(self.cfg.stitch_max_gap_s * self._fps))
        pos_thr = self.cfg.stitch_pos_threshold
        ang_thr = self.cfg.stitch_vel_angle_deg

        validated = [t for t in self.tracks if t.validated and len(t.pts) >= 4]
        if len(validated) < 2:
            return 0

        validated.sort(key=lambda t: t.pts[-1].frame)

        merged_ids: set[int] = set()
        n_merges = 0

        for i, a in enumerate(validated):
            if a.id in merged_ids:
                continue
            for j in range(i + 1, len(validated)):
                b = validated[j]
                if b.id in merged_ids:
                    continue

                a_end = a.pts[-1].frame
                b_start = b.pts[0].frame
                gap = b_start - a_end
                if gap <= 0 or gap > max_gap:
                    continue

                exit_state = a._exit_state()
                if exit_state is None:
                    continue
                (cx, cy), vel_a = exit_state

                vel_b = b._entry_velocity()
                if vel_b is None:
                    continue

                # Velocity angle check
                dot = float(np.dot(vel_a, vel_b))
                na = float(np.linalg.norm(vel_a))
                nb = float(np.linalg.norm(vel_b))
                if na < 1e-6 or nb < 1e-6:
                    continue
                cos_ang = np.clip(dot / (na * nb), -1.0, 1.0)
                angle_deg = float(np.degrees(np.arccos(cos_ang)))
                if angle_deg > ang_thr:
                    continue

                # Position extrapolation check
                t0_a = a.pts[-len(a.pts[-8:]):][0].frame
                dt = b_start - t0_a
                x_pred = float(np.polyval(cx, dt))
                y_pred = float(np.polyval(cy, dt))
                bx, by = b.pts[0].x, b.pts[0].y
                pos_err = float(np.hypot(x_pred - bx, y_pred - by))
                if pos_err > pos_thr:
                    continue

                # Merge B into A
                a.pts.extend(b.pts)
                a.last_det_frame = b.last_det_frame
                merged_ids.add(b.id)
                n_merges += 1
                break  # move to next A candidate after successful merge

        if merged_ids:
            self.tracks = [t for t in self.tracks if t.id not in merged_ids]

        return n_merges
