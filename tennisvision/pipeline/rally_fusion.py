"""Serve-anchored rally detection with three end conditions.

Rally start: serve event (deduped: consecutive serves within 10s → take last)
Rally end (first to trigger):
  1. No net crossing for N seconds
  2. Next serve event
  3. Double bounce in same half court

Inputs:
  - serve_events: from detect_serves.py
  - ball_positions: from WASB + Kalman tracker (detected + predicted)
  - ball_detected: detected-only positions (for bounce detection)
  - net_y_px: net line pixel Y coordinate (from calibration)
  - H_img_to_real: homography matrix (for court coordinate mapping)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .rally import Rally


@dataclass
class FusionRallyConfig:
    no_cross_timeout_s: float = 5.0
    no_cross_buffer_s: float = 4.0       # buffer after last crossing for timeout end
    pre_roll_s: float = 1.0
    min_duration_s: float = 2.0
    serve_dedup_s: float = 10.0          # merge consecutive serves within this window
    triple_bounce_window_s: float = 2.0  # 3 bounces on same half within this window
    bounce_skip_start_s: float = 1.0     # skip bounces in first N seconds of rally
    # Bounce scoring (Good-Tennis style)
    bounce_window: int = 20
    bounce_center_offset: int = 10
    bounce_min_score: float = 0.34
    bounce_min_gap_s: float = 0.45
    bounce_max_interp_gap: int = 12


class MultiSignalRallyDetector:
    """Detect rallies from serve events + ball trajectory."""

    def __init__(self, cfg: Optional[FusionRallyConfig] = None):
        self.cfg = cfg or FusionRallyConfig()

    def detect(
        self,
        total_frames: int,
        fps: float,
        *,
        serve_events: List[dict],
        ball_positions: Optional[Dict[int, Tuple[float, float]]] = None,
        ball_detected: Optional[Dict[int, Tuple[float, float]]] = None,
        net_y_px: Optional[float] = None,
        H_img_to_real: Optional[np.ndarray] = None,
    ) -> List[Rally]:
        cfg = self.cfg
        ball = ball_positions or {}
        ball_det = ball_detected or {}

        # --- Dedup serves ---
        deduped = self._dedup_serves(serve_events, cfg.serve_dedup_s)
        print(f"Serves: {len(serve_events)} total, {len(deduped)} deduped")

        # --- Detect bounces + triple bounces ---
        triple_bounce_set = set()
        bounces = []
        if ball_det and H_img_to_real is not None:
            bounces = self._detect_bounces(ball, ball_det, total_frames, fps,
                                           H_img_to_real, cfg)
            triple_bounce_set = self._find_triple_bounces(bounces, fps, cfg)
            print(f"Bounces: {len(bounces)}, triple bounces: {len(triple_bounce_set)}")
        # Store for caller: list of (frame, half, rx, ry)
        self.bounces = bounces

        # --- Net x-range in pixels (for filtering off-court crossings) ---
        net_x_range = None
        if H_img_to_real is not None:
            H_inv = np.linalg.inv(H_img_to_real)
            margin = 2.0  # meters outside court to allow (for outside-in shots)
            p_left = H_inv @ [-margin, 23.77 / 2, 1.0]
            p_right = H_inv @ [10.97 + margin, 23.77 / 2, 1.0]
            net_x_range = (p_left[0] / p_left[2], p_right[0] / p_right[2])

        # --- Build rallies ---
        rallies = []
        pre_roll = int(round(cfg.pre_roll_s * fps))
        min_dur = int(round(cfg.min_duration_s * fps))

        for si, ev in enumerate(deduped):
            serve_frame = ev["start_frame"]
            start_frame = max(0, serve_frame - pre_roll)

            # Don't overlap with previous rally
            if rallies and start_frame < rallies[-1].end_frame:
                start_frame = rallies[-1].end_frame + 1

            # Next serve = hard deadline
            next_serve_frame = total_frames
            if si + 1 < len(deduped):
                next_serve_frame = deduped[si + 1]["start_frame"]

            # Find rally end
            rally_end, end_reason = self._find_rally_end(
                serve_frame, next_serve_frame, total_frames, fps,
                ball, net_y_px, triple_bounce_set, cfg,
                ball_detected=ball_det, net_x_range=net_x_range,
            )

            rally_end = min(rally_end, total_frames - 1)
            duration = rally_end - start_frame

            if duration < min_dur:
                continue

            rallies.append(Rally(
                idx=len(rallies),
                start_frame=start_frame,
                end_frame=rally_end,
                n_events=0,
                net_crossings=0,
            ))

        print(f"Rallies: {len(rallies)}")
        return rallies

    # ------------------------------------------------------------------
    # Serve dedup
    # ------------------------------------------------------------------

    @staticmethod
    def _dedup_serves(serve_events: List[dict], window_s: float) -> List[dict]:
        deduped = []
        for ev in serve_events:
            if deduped and ev["start_time"] - deduped[-1]["end_time"] < window_s:
                deduped[-1] = ev  # replace with later serve
            else:
                deduped.append(ev)
        return deduped

    # ------------------------------------------------------------------
    # Rally end detection
    # ------------------------------------------------------------------

    def _find_rally_end(
        self,
        serve_frame: int,
        next_serve_frame: int,
        total_frames: int,
        fps: float,
        ball: Dict[int, Tuple[float, float]],
        net_y_px: Optional[float],
        triple_bounce_set: set,
        cfg: FusionRallyConfig,
        ball_detected: Optional[Dict[int, Tuple[float, float]]] = None,
        net_x_range: Optional[Tuple[float, float]] = None,
    ) -> Tuple[int, str]:
        timeout_frames = int(round(cfg.no_cross_timeout_s * fps))
        buffer_frames = int(round(cfg.no_cross_buffer_s * fps))
        skip_start = serve_frame + int(cfg.bounce_skip_start_s * fps)
        det = ball_detected or {}
        x_lo, x_hi = net_x_range if net_x_range else (0.0, 1e9)

        prev_det_y = None
        last_crossing = serve_frame

        for fi in range(serve_frame, min(total_frames, next_serve_frame)):
            # Rule 3: triple bounce (same half, 3 bounces within window)
            if fi in triple_bounce_set and fi > skip_start:
                return fi + int(fps), "triple bounce"

            # Ball presence check (detected + predicted)
            pos = ball.get(fi)
            if pos is None:
                if (fi - last_crossing) > timeout_frames:
                    return last_crossing + buffer_frames, "no crossing timeout"
                continue

            # Net crossing: only from detected positions (not Kalman predictions)
            det_pos = det.get(fi)
            if det_pos is not None and net_y_px is not None and prev_det_y is not None:
                bx, by = det_pos[0], det_pos[1]
                if (prev_det_y < net_y_px and by >= net_y_px) or \
                   (prev_det_y >= net_y_px and by < net_y_px):
                    if x_lo <= bx <= x_hi:  # ball within court x-range
                        last_crossing = fi
            if det_pos is not None:
                prev_det_y = det_pos[1]

            # Rule 1: no crossing timeout
            if (fi - last_crossing) > timeout_frames:
                return last_crossing + buffer_frames, "no crossing timeout"

        # Rule 2: next serve
        return min(next_serve_frame - 1, total_frames - 1), "next serve"

    # ------------------------------------------------------------------
    # Bounce detection (Good-Tennis style scoring)
    # ------------------------------------------------------------------

    def _detect_bounces(
        self,
        ball_all: Dict[int, Tuple[float, float]],
        ball_det: Dict[int, Tuple[float, float]],
        total_frames: int,
        fps: float,
        H: np.ndarray,
        cfg: FusionRallyConfig,
    ) -> List[Tuple[int, str]]:
        """Returns list of (frame, half_court) bounce events."""
        net_y_court = 23.77 / 2

        # Build coordinate array with outlier removal + interpolation
        coords = np.full((total_frames, 2), np.nan, dtype=np.float32)
        for fi in range(total_frames):
            pos = ball_det.get(fi) or ball_all.get(fi)
            if pos is not None:
                coords[fi] = pos

        coords = self._remove_outliers(coords)
        coords = self._interpolate(coords, cfg.bounce_max_interp_gap)

        # Velocity
        velocity = np.full(total_frames, np.nan, dtype=np.float32)
        for fi in range(1, total_frames):
            if np.isnan(coords[fi]).any() or np.isnan(coords[fi - 1]).any():
                continue
            velocity[fi] = np.linalg.norm(coords[fi] - coords[fi - 1]) * fps
        if total_frames > 1 and not np.isnan(velocity[1]):
            velocity[0] = velocity[1]

        # Sliding window scoring
        W = cfg.bounce_window
        CO = cfg.bounce_center_offset
        min_gap_frames = max(1, int(cfg.bounce_min_gap_s * fps))

        raw = []
        for end_i in range(W - 1, total_frames):
            start_i = end_i - W + 1
            ci_abs = end_i - CO  # center index in absolute frames
            if ci_abs <= 0 or ci_abs >= total_frames - 1:
                continue

            window = coords[start_i:end_i + 1]
            window_v = velocity[start_i:end_i + 1]
            if np.isnan(window).any() or np.isnan(window_v).any():
                continue

            ci = W - CO - 1  # center index in window
            score = self._score_bounce_window(window, window_v, ci)
            if score < cfg.bounce_min_score:
                continue

            # Court coordinate check
            hp = H @ [window[ci, 0], window[ci, 1], 1.0]
            if abs(hp[2]) < 1e-9:
                continue
            rx, ry = hp[0] / hp[2], hp[1] / hp[2]
            if not (-0.9 <= rx <= 10.97 + 0.9 and -0.9 <= ry <= 23.77 + 0.9):
                continue

            half = "NEAR" if ry < net_y_court else "FAR"
            raw.append((ci_abs, score, half, rx, ry))

        # Dedupe by confidence
        bounces = []
        for frame, score, half, rx, ry in sorted(raw, key=lambda x: -x[1]):
            if any(abs(frame - f) < min_gap_frames for f, _, _, _ in bounces):
                continue
            bounces.append((frame, half, rx, ry))
        bounces.sort(key=lambda x: x[0])
        return bounces

    @staticmethod
    def _score_bounce_window(window: np.ndarray, velocity: np.ndarray,
                             ci: int) -> float:
        """Score a window for bounce likelihood."""
        centered = window - np.nanmean(window, axis=0)
        scale = max(float(np.nanstd(centered)), 1.0)
        normalized = centered / scale

        # Smooth
        sm = window.copy()
        for i in range(1, len(sm) - 1):
            sm[i] = (window[i - 1] + window[i] * 2 + window[i + 1]) / 4.0

        center_pt = sm[ci]
        before = sm[max(0, ci - 5):ci]
        after = sm[ci + 1:min(len(sm), ci + 6)]
        if len(before) < 3 or len(after) < 3:
            return 0.0

        before_c = np.mean(before, axis=0)
        after_c = np.mean(after, axis=0)

        # Turn angle
        v_in = center_pt - before_c
        v_out = after_c - center_pt
        denom = np.linalg.norm(v_in) * np.linalg.norm(v_out)
        turn_deg = 0.0
        if denom > 1e-6:
            turn_deg = np.degrees(np.arccos(np.clip(np.dot(v_in, v_out) / denom, -1, 1)))

        # Deviation from straight line
        line = after_c - before_c
        line_len = np.linalg.norm(line)
        deviation = 0.0
        if line_len > 1e-6:
            deviation = abs(np.cross(line, center_pt - before_c)) / line_len

        # Speed ratio
        local_v = velocity[max(0, ci - 4):min(len(velocity), ci + 5)]
        median_v = float(np.nanmedian(local_v))
        peak_v = float(np.nanmax(local_v))
        v_center = float(velocity[ci])
        speed_ratio = peak_v / max(median_v, 1.0)
        if v_center > 2500 or speed_ratio > 12:
            return 0.0

        # Y reversal
        y = normalized[:, 1]
        y_slope_in = np.polyfit(np.arange(ci + 1, dtype=np.float32),
                                y[:ci + 1], 1)[0] if ci >= 1 else 0.0
        y_slope_out = np.polyfit(np.arange(len(y) - ci, dtype=np.float32),
                                 y[ci:], 1)[0] if ci < len(y) - 1 else 0.0
        y_reversal = y_slope_in > 0.05 and y_slope_out < -0.05

        local_y = window[max(0, ci - 5):min(len(window), ci + 6), 1]
        y_peak = window[ci, 1] >= np.max(local_y) - 4.0
        y_valley = window[ci, 1] <= np.min(local_y) + 4.0
        y_extreme = y_peak or y_valley

        if not (y_reversal or y_extreme):
            return 0.0

        # Weighted score
        return (0.28 * min(1.0, turn_deg / 95.0)
                + 0.24 * min(1.0, deviation / 18.0)
                + 0.12 * min(1.0, max(0.0, speed_ratio - 1.0) / 2.0)
                + 0.16 * (1.0 if y_reversal else 0.0)
                + 0.10 * (1.0 if y_extreme else 0.0))

    @staticmethod
    def _find_triple_bounces(bounces: List[Tuple[int, str]], fps: float,
                             cfg: FusionRallyConfig) -> set:
        window_frames = cfg.triple_bounce_window_s * fps
        triple_set = set()
        for i in range(2, len(bounces)):
            f0, h0 = bounces[i - 2][0], bounces[i - 2][1]
            f1, h1 = bounces[i - 1][0], bounces[i - 1][1]
            f2, h2 = bounces[i][0], bounces[i][1]
            if h0 == h1 == h2 and (f2 - f0) < window_frames:
                triple_set.add(f2)
        return triple_set

    # ------------------------------------------------------------------
    # Trajectory preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def _remove_outliers(coords: np.ndarray) -> np.ndarray:
        cleaned = coords.copy()
        valid = np.where(~np.isnan(cleaned[:, 0]))[0]
        if len(valid) < 5:
            return cleaned
        steps = []
        for a, b in zip(valid[:-1], valid[1:]):
            steps.append(np.linalg.norm(cleaned[b] - cleaned[a]) / max(1, b - a))
        steps = np.array(steps)
        med = np.median(steps)
        mad = np.median(np.abs(steps - med))
        threshold = max(90.0, med + 6.0 * max(mad, 1.0))

        for idx in valid[1:-1]:
            prev_idx = next((c for c in range(idx - 1, -1, -1)
                             if not np.isnan(cleaned[c, 0])), None)
            next_idx = next((c for c in range(idx + 1, len(cleaned))
                             if not np.isnan(cleaned[c, 0])), None)
            if prev_idx is None or next_idx is None:
                continue
            pd = np.linalg.norm(cleaned[idx] - cleaned[prev_idx]) / max(1, idx - prev_idx)
            nd = np.linalg.norm(cleaned[next_idx] - cleaned[idx]) / max(1, next_idx - idx)
            bd = np.linalg.norm(cleaned[next_idx] - cleaned[prev_idx]) / max(1, next_idx - prev_idx)
            if pd > threshold and nd > threshold and bd < threshold:
                cleaned[idx] = [np.nan, np.nan]
        return cleaned

    @staticmethod
    def _interpolate(coords: np.ndarray, max_gap: int) -> np.ndarray:
        result = coords.copy()
        valid = np.where(~np.isnan(result[:, 0]))[0]
        for a, b in zip(valid[:-1], valid[1:]):
            gap = b - a
            if gap <= 1 or gap > max_gap + 1:
                continue
            for fi in range(a + 1, b):
                alpha = (fi - a) / gap
                result[fi] = result[a] * (1 - alpha) + result[b] * alpha
        return result
