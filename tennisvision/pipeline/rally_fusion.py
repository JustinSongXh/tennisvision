"""Serve-anchored rally detection with net-crossing timeout.

State machine:
  IDLE  → valid serve (near baseline) → RALLY
  RALLY → ball crosses net (detected only) → reset timeout
  RALLY → timeout expires (no crossing for N seconds) → IDLE
  RALLY → new valid serve → close current rally, start new one
  IDLE  → any other signal → ignore

Only WASB-detected ball positions (not Kalman-predicted) are used for
net-crossing detection to avoid false crossings from prediction jumps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .rally import Rally


@dataclass
class FusionRallyConfig:
    no_cross_timeout_s: float = 5.0     # end rally after N seconds without crossing
    serve_confirm_s: float = 5.0        # serve must have crossing within N seconds
    serve_baseline_margin_m: float = 4.0  # serve must be within N metres of a baseline
    pre_roll_s: float = 1.0
    post_roll_s: float = 1.0
    min_duration_s: float = 2.0
    max_cross_gap: int = 5              # max frame gap for a crossing to count


class MultiSignalRallyDetector:
    """State-machine rally detector: serve starts, net-crossing timeout ends."""

    def __init__(self, cfg: Optional[FusionRallyConfig] = None):
        self.cfg = cfg or FusionRallyConfig()

    def detect(
        self,
        total_frames: int,
        fps: float,
        *,
        serve_frames: List[int],
        serve_positions: Optional[Dict[int, Tuple[float, float]]] = None,
        ball_detected: Optional[Dict[int, Tuple[float, float]]] = None,
        ball_predicted: Optional[Dict[int, Tuple[float, float]]] = None,
        net_y_px: Optional[float] = None,
        H_img_to_real: Optional[np.ndarray] = None,
        court_length_m: float = 23.77,
    ) -> List[Rally]:
        """Run state machine.

        Args:
            total_frames: total video frames
            fps: video frame rate
            serve_frames: frame indices where serve was classified
            serve_positions: {frame: (foot_x, foot_y)} player foot position at serve
            ball_detected: {frame: (x, y)} WASB detected positions only
                          (used for serve confirmation — avoids Kalman false crossings)
            ball_predicted: {frame: (x, y)} Kalman predicted positions
                           (used for in-rally crossing detection — better coverage)
            net_y_px: net y pixel coordinate
            H_img_to_real: 3x3 homography for baseline check
            court_length_m: court length in metres
        """
        cfg = self.cfg
        timeout_frames = int(round(cfg.no_cross_timeout_s * fps))
        confirm_frames = int(round(cfg.serve_confirm_s * fps))
        pre_roll = int(round(cfg.pre_roll_s * fps))
        post_roll = int(round(cfg.post_roll_s * fps))
        min_dur = int(round(cfg.min_duration_s * fps))
        det_ball = ball_detected or {}
        pred_ball = ball_predicted or {}

        # --- Validate serves: must be near baseline ---
        valid_serves = self._filter_serves(
            serve_frames, serve_positions,
            H_img_to_real, court_length_m, cfg.serve_baseline_margin_m,
        )

        # --- Crossings from detected positions (for serve confirmation) ---
        det_crossings = self._compute_crossings(det_ball, net_y_px, cfg.max_cross_gap)

        # --- Crossings from predicted positions (for in-rally tracking) ---
        pred_crossings = self._compute_crossings(pred_ball, net_y_px, cfg.max_cross_gap)

        # --- Confirm serves: must have predicted crossing within confirm window ---
        confirmed_serves = []
        for sf in valid_serves:
            if any(sf <= cf <= sf + confirm_frames for cf in pred_crossings):
                confirmed_serves.append(sf)

        print(f"Serves: {len(serve_frames)} total, {len(valid_serves)} near baseline, "
              f"{len(confirmed_serves)} confirmed with crossing")

        # Use predicted crossings for in-rally tracking
        crossing_frames = pred_crossings

        # --- State machine ---
        rallies_raw: List[Tuple[int, int, int]] = []  # (start, end, n_crossings)
        state = "IDLE"
        rally_start = 0
        last_crossing = 0

        # Process all events in time order
        all_events: List[Tuple[int, str]] = []
        for f in confirmed_serves:
            all_events.append((f, "serve"))
        for f in crossing_frames:
            all_events.append((f, "crossing"))
        all_events.sort(key=lambda e: e[0])

        n_crossings = 0

        for frame, event_type in all_events:
            if state == "IDLE":
                if event_type == "serve":
                    state = "RALLY"
                    rally_start = frame
                    last_crossing = frame
                    n_crossings = 0
                # ignore crossings/strokes in IDLE

            elif state == "RALLY":
                if event_type == "crossing":
                    last_crossing = frame
                    n_crossings += 1

                elif event_type == "serve":
                    # New serve → close current rally, start new one
                    rally_end = frame - 1
                    rallies_raw.append((rally_start, rally_end, n_crossings))
                    rally_start = frame
                    last_crossing = frame
                    n_crossings = 0

        # Check timeout for open rally
        if state == "RALLY":
            rally_end = min(last_crossing + timeout_frames, total_frames - 1)
            rallies_raw.append((rally_start, rally_end, n_crossings))

        # Also retroactively apply timeout within rallies
        # (a rally might have a long gap without crossings in the middle)
        final_rallies: List[Tuple[int, int, int]] = []
        for start, end, nc in rallies_raw:
            relevant = [f for f in crossing_frames if start <= f <= end]
            if relevant:
                # Find actual end: last crossing + timeout
                actual_end = min(relevant[-1] + timeout_frames, end)
            else:
                actual_end = min(start + timeout_frames, end)
            final_rallies.append((start, actual_end, len(relevant)))

        # --- Filter + pad, prevent overlaps ---
        rallies: List[Rally] = []
        prev_end = -1
        for start, actual_end, nc in final_rallies:
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

    def _filter_serves(
        self,
        serve_frames: List[int],
        serve_positions: Optional[Dict[int, Tuple[float, float]]],
        H_img_to_real: Optional[np.ndarray],
        court_length_m: float,
        margin_m: float,
    ) -> List[int]:
        """Keep only serves near a baseline."""
        if H_img_to_real is None or serve_positions is None:
            print(f"Serves: {len(serve_frames)} (no baseline filter — missing homography or positions)")
            return sorted(serve_frames)

        H = np.asarray(H_img_to_real, dtype=np.float64)
        valid = []
        for f in sorted(serve_frames):
            pos = serve_positions.get(f)
            if pos is None:
                continue
            # Project player foot position to court coordinates
            px, py = pos
            p = H @ [px, py, 1.0]
            if abs(p[2]) < 1e-9:
                continue
            court_y = float(p[1] / p[2])
            # Near either baseline?
            near_near = abs(court_y) <= margin_m
            near_far = abs(court_y - court_length_m) <= margin_m
            if near_near or near_far:
                valid.append(f)

        print(f"Serves: {len(serve_frames)} total, {len(valid)} near baseline "
              f"(dropped {len(serve_frames) - len(valid)})")
        return valid

    @staticmethod
    def _compute_crossings(
        ball_detected: Dict[int, Tuple[float, float]],
        net_y_px: Optional[float],
        max_gap: int,
    ) -> List[int]:
        """Compute net crossing frames from detected-only ball positions."""
        if not ball_detected or net_y_px is None:
            return []

        sorted_frames = sorted(ball_detected.keys())
        crossings: List[int] = []
        prev_side: Optional[int] = None
        prev_frame = -999

        for f in sorted_frames:
            y = ball_detected[f][1]
            side = -1 if y < net_y_px else 1
            if (prev_side is not None
                    and side != prev_side
                    and (f - prev_frame) <= max_gap):
                crossings.append(f)
            prev_side = side
            prev_frame = f

        return crossings
