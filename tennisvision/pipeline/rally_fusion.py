"""Serve-anchored rally detection with net-crossing timeout.

Rally boundaries:
  - START: serve event (from stroke classifier)
  - END:   ball trajectory has not crossed the net for N seconds

Simple state machine, no scoring or thresholds to tune.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set

import numpy as np

from .rally import Rally


@dataclass
class FusionRallyConfig:
    # Net crossing timeout: end rally after this many seconds without a crossing
    no_cross_timeout_s: float = 5.0
    # Padding
    pre_roll_s: float = 1.0
    post_roll_s: float = 1.0
    # Minimum rally duration (filter noise)
    min_duration_s: float = 2.0


class MultiSignalRallyDetector:
    """Detect rallies using serve events + ball net-crossing timeout.

    State machine:
      WAITING → serve detected → RALLY
      RALLY   → ball crosses net → reset timeout
      RALLY   → timeout expires (no crossing for N seconds) → WAITING
      RALLY   → next serve detected → close current rally, start new one
    """

    def __init__(self, cfg: Optional[FusionRallyConfig] = None):
        self.cfg = cfg or FusionRallyConfig()

    def detect(
        self,
        total_frames: int,
        fps: float,
        *,
        serve_frames: List[int],
        ball_positions: Optional[dict] = None,
        net_y_px: Optional[float] = None,
    ) -> List[Rally]:
        """Run rally detection.

        Args:
            total_frames: total video frames
            fps: video frame rate
            serve_frames: frame indices where serve was detected
            ball_positions: {frame_idx: (x, y)} ball position per frame
                           (from ball tracker / Kalman filter)
            net_y_px: y pixel coordinate of the net line
                     (from court calibration)

        Returns:
            list of Rally objects
        """
        cfg = self.cfg
        timeout_frames = int(round(cfg.no_cross_timeout_s * fps))
        pre_roll = int(round(cfg.pre_roll_s * fps))
        post_roll = int(round(cfg.post_roll_s * fps))
        min_dur = int(round(cfg.min_duration_s * fps))

        serves = sorted(serve_frames)
        ball = ball_positions or {}

        # Precompute per-frame net side: -1 (far), +1 (near), None (no ball)
        sides = {}
        if net_y_px is not None:
            for f, (x, y) in ball.items():
                sides[f] = -1 if y < net_y_px else 1

        # Precompute crossing frames
        crossing_frames: List[int] = []
        if sides:
            sorted_ball_frames = sorted(sides.keys())
            prev_side = None
            for f in sorted_ball_frames:
                s = sides[f]
                if prev_side is not None and s != prev_side:
                    crossing_frames.append(f)
                prev_side = s

        # State machine
        raw_rallies: List[tuple] = []  # (start_frame, end_frame)
        rally_start: Optional[int] = None
        last_crossing: Optional[int] = None

        for serve_f in serves:
            # Close previous rally if open
            if rally_start is not None:
                end = self._find_end(
                    rally_start, serve_f - 1, crossing_frames,
                    timeout_frames, total_frames,
                )
                raw_rallies.append((rally_start, end))

            rally_start = serve_f
            last_crossing = serve_f  # treat serve as initial "crossing"

        # Close last rally
        if rally_start is not None:
            end = self._find_end(
                rally_start, total_frames - 1, crossing_frames,
                timeout_frames, total_frames,
            )
            raw_rallies.append((rally_start, end))

        # Apply timeout-based end within each rally
        rallies: List[Rally] = []
        for start, end in raw_rallies:
            # Find actual end: last crossing + timeout, or rally boundary
            relevant_crossings = [
                f for f in crossing_frames if start <= f <= end
            ]
            if relevant_crossings:
                actual_end = min(
                    relevant_crossings[-1] + timeout_frames,
                    end,
                )
            else:
                # No crossings at all — rally ends at timeout after serve
                actual_end = min(start + timeout_frames, end)

            dur = actual_end - start
            if dur < min_dur:
                continue

            padded_start = max(0, start - pre_roll)
            padded_end = min(total_frames - 1, actual_end + post_roll)

            rallies.append(Rally(
                idx=len(rallies),
                start_frame=padded_start,
                end_frame=padded_end,
                n_events=len(relevant_crossings),
                net_crossings=len(relevant_crossings),
            ))

        return rallies

    @staticmethod
    def _find_end(
        rally_start: int,
        boundary: int,
        crossing_frames: List[int],
        timeout: int,
        total_frames: int,
    ) -> int:
        """Find rally end: last crossing + timeout, capped at boundary."""
        relevant = [f for f in crossing_frames if rally_start <= f <= boundary]
        if relevant:
            return min(relevant[-1] + timeout, boundary, total_frames - 1)
        return min(rally_start + timeout, boundary, total_frames - 1)
