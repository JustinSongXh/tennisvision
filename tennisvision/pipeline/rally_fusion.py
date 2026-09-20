"""Serve-anchored rally detection with net-crossing timeout.

State machine:
  IDLE  → serve event (FAR slot, near baseline) → RALLY
  RALLY → ball crosses net → reset timeout
  RALLY → timeout expires (N seconds no crossing) → IDLE
  RALLY → new serve → close current rally, start new

Inputs:
  - serve_events: from GRUActionClassifier (is_serve=True events)
  - ball_positions: from WASB + Kalman tracker (detected positions)
  - net_y_px: net line pixel Y coordinate (from calibration)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .rally import Rally


@dataclass
class FusionRallyConfig:
    no_cross_timeout_s: float = 5.0       # end rally after N seconds without crossing
    pre_roll_s: float = 1.0               # pad before rally start
    post_roll_s: float = 1.0              # pad after rally end
    min_duration_s: float = 2.0           # minimum rally duration
    max_cross_gap: int = 5                # max frame gap for crossing detection
    serve_far_only: bool = True           # only accept serve from FAR slots


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
        net_y_px: Optional[float] = None,
    ) -> List[Rally]:
        """Run rally detection.

        Args:
            total_frames: total video frames
            fps: video frame rate
            serve_events: list of serve event dicts with 'frame', 'slot', 'conf'
            ball_positions: {frame: (x, y)} ball positions for crossing detection
            net_y_px: net Y pixel coordinate

        Returns:
            list of Rally objects
        """
        cfg = self.cfg
        timeout_frames = int(round(cfg.no_cross_timeout_s * fps))
        pre_roll = int(round(cfg.pre_roll_s * fps))
        post_roll = int(round(cfg.post_roll_s * fps))
        min_dur = int(round(cfg.min_duration_s * fps))
        ball = ball_positions or {}

        # Filter serves: only FAR slots if configured
        valid_serves = []
        for ev in serve_events:
            if cfg.serve_far_only and ev.get("slot", -1) < 2:
                continue  # skip NEAR slots
            valid_serves.append(ev.get("start_frame", ev.get("frame")))
        valid_serves = sorted(set(valid_serves))

        # Compute net crossings from ball positions
        crossing_frames = self._compute_crossings(ball, net_y_px, cfg.max_cross_gap)

        print(f"Serves: {len(serve_events)} total, {len(valid_serves)} valid")
        print(f"Ball crossings: {len(crossing_frames)}")

        # State machine
        raw_rallies: List[Tuple[int, int, int]] = []
        rally_start: Optional[int] = None

        for serve_f in valid_serves:
            if rally_start is not None:
                end = self._find_end(
                    rally_start, serve_f - 1, crossing_frames,
                    timeout_frames, total_frames,
                )
                raw_rallies.append((rally_start, end))
            rally_start = serve_f

        # Close last rally
        if rally_start is not None:
            end = self._find_end(
                rally_start, total_frames - 1, crossing_frames,
                timeout_frames, total_frames,
            )
            raw_rallies.append((rally_start, end))

        # Compute actual end + crossings per rally
        timed_rallies = []
        for start, end in raw_rallies:
            crossings = [f for f in crossing_frames if start <= f <= end]
            if crossings:
                actual_end = min(crossings[-1] + timeout_frames, end)
            else:
                actual_end = min(start + timeout_frames, end)
            timed_rallies.append((start, actual_end, len(crossings)))

        # Filter + pad, prevent overlaps
        rallies: List[Rally] = []
        prev_end = -1
        for start, actual_end, nc in timed_rallies:
            if (actual_end - start) < min_dur:
                continue
            padded_start = max(0, start - pre_roll)
            padded_end = min(total_frames - 1, actual_end + post_roll)
            if padded_start <= prev_end:
                padded_start = prev_end + 1
            if padded_start >= padded_end:
                continue
            rallies.append(Rally(
                idx=len(rallies),
                start_frame=padded_start,
                end_frame=padded_end,
                n_events=nc,
                net_crossings=nc,
            ))
            prev_end = padded_end

        return rallies

    @staticmethod
    def _find_end(rally_start, boundary, crossing_frames, timeout, total_frames):
        relevant = [f for f in crossing_frames if rally_start <= f <= boundary]
        if relevant:
            return min(relevant[-1] + timeout, boundary, total_frames - 1)
        return min(rally_start + timeout, boundary, total_frames - 1)

    @staticmethod
    def _compute_crossings(
        ball_positions: Dict[int, Tuple[float, float]],
        net_y_px: Optional[float],
        max_gap: int,
    ) -> List[int]:
        if not ball_positions or net_y_px is None:
            return []
        sorted_frames = sorted(ball_positions.keys())
        crossings = []
        prev_side: Optional[int] = None
        prev_frame = -999
        for f in sorted_frames:
            y = ball_positions[f][1]
            side = -1 if y < net_y_px else 1
            if (prev_side is not None
                    and side != prev_side
                    and (f - prev_frame) <= max_gap):
                crossings.append(f)
            prev_side = side
            prev_frame = f
        return crossings
