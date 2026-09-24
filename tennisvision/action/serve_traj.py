"""Ball-trajectory-based serve toss verification.

Provides ``check_serve_toss`` — a filter function that examines the ball
trajectory around a GRU serve candidate and decides whether a genuine
toss pattern exists.

Serve toss signature:
  1. Ball moves along the court-vertical axis (perpendicular to the net)
     with minimal lateral (net-parallel) drift.
  2. The vertical motion follows a parabolic trajectory consistent with
     gravity (free flight under g ≈ 9.8 m/s²).
  3. After the toss, the ball departs laterally (the hit).

Camera tilt is corrected by projecting ball displacement onto the net
direction (lateral) and its perpendicular (vertical).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np


@dataclass
class TossFilterConfig:
    # Search window around the GRU event
    search_before_s: float = 2.0
    search_after_s: float = 0.5

    # Ball-player proximity (to select relevant ball detections)
    max_ball_player_x_px: float = 300.0
    max_ball_player_y_px: float = 300.0

    # Lateral stationarity: lat_std / v_range must be below this ratio
    max_lateral_ratio: float = 0.25

    # Parabolic fit quality
    min_r2: float = 0.95               # min R² for quadratic fit
    min_points: int = 5                 # min data points for fit

    # Gap tolerance
    max_gap_s: float = 0.1             # max gap between consecutive ball detections

    # Hit confirmation
    hit_search_s: float = 0.5
    min_hit_lateral_px: float = 7.0    # min lateral jump per frame


def compute_net_direction(H_img_to_real: np.ndarray) -> np.ndarray:
    """Compute the net line direction in image space from the homography.

    Returns a unit vector along the net in image coordinates.
    """
    H_inv = np.linalg.inv(H_img_to_real)
    court_half_y = 23.77 / 2  # net at midpoint

    def to_img(cx, cy):
        p = H_inv @ np.array([cx, cy, 1.0])
        return np.array([p[0] / p[2], p[1] / p[2]])

    nl = to_img(0.0, court_half_y)
    nr = to_img(10.97, court_half_y)
    d = nr - nl
    return d / np.linalg.norm(d)


def check_serve_toss(
    get_ball: Callable[[int], Optional[list]],
    ev_start_frame: int,
    ev_end_frame: int,
    foot_x: float,
    foot_y: float,
    bbox: List[int],
    fps: float,
    net_dir: Optional[np.ndarray] = None,
    cfg: Optional[TossFilterConfig] = None,
) -> Tuple[bool, str]:
    """Check whether a GRU serve candidate has a valid ball toss pattern.

    Parameters
    ----------
    get_ball : callable(frame_idx) -> [x, y] or None
    ev_start_frame, ev_end_frame : int
        GRU event frame range.
    foot_x, foot_y : float
        Player foot position (image pixels).
    bbox : list [x0, y0, x1, y1]
        Player bounding box.
    fps : float
    net_dir : ndarray, optional
        Unit vector along the net in image space.  If None, assumes
        horizontal net (no camera tilt).
    cfg : TossFilterConfig, optional

    Returns
    -------
    (keep, reason) : (bool, str)
    """
    cfg = cfg or TossFilterConfig()

    if net_dir is None:
        net_dir = np.array([1.0, 0.0])
    # Court-vertical = perpendicular to net direction
    vert_dir = np.array([-net_dir[1], net_dir[0]])

    search_before = int(round(cfg.search_before_s * fps))
    search_after = int(round(cfg.search_after_s * fps))
    win_start = max(0, ev_start_frame - search_before)
    win_end = ev_end_frame + search_after

    bbox_w = bbox[2] - bbox[0]
    bbox_h = bbox[3] - bbox[1]
    max_dx_player = max(cfg.max_ball_player_x_px, bbox_w * 3)
    max_dy_player = max(cfg.max_ball_player_y_px, bbox_h * 3)

    # Collect ball positions near the player
    ball_frames: List[Tuple[int, float, float]] = []
    for fi in range(win_start, win_end + 1):
        bp = get_ball(fi)
        if bp is None:
            continue
        if abs(bp[0] - foot_x) > max_dx_player:
            continue
        if abs(bp[1] - foot_y) > max_dy_player:
            continue
        ball_frames.append((fi, bp[0], bp[1]))

    if not ball_frames:
        return False, "no ball near player in toss window"

    # Split into continuous runs (break on gaps)
    max_gap_frames = int(round(cfg.max_gap_s * fps))
    runs = _split_into_runs(ball_frames, max_gap_frames)

    # Try each run for toss pattern
    for run in runs:
        if len(run) < cfg.min_points:
            continue

        ok, detail, toss_end_idx = _check_toss_parabola(
            run, fps, net_dir, vert_dir, cfg,
        )
        if ok:
            # Hit confirmation: either the toss was cut at a lateral spike
            # (hit already found within the run), or look after the run.
            hit = toss_end_idx < len(run)
            if not hit:
                run_last_frame = run[-1][0]
                hit = _check_hit(get_ball, run_last_frame, fps, net_dir, cfg)
            return True, f"toss OK ({detail} hit={'yes' if hit else 'no'})"

    return False, "no valid toss pattern"


def _split_into_runs(
    ball_frames: List[Tuple[int, float, float]],
    max_gap: int,
) -> List[List[Tuple[int, float, float]]]:
    """Split ball_frames into continuous runs, breaking on large gaps."""
    runs: List[List[Tuple[int, float, float]]] = []
    cur: List[Tuple[int, float, float]] = [ball_frames[0]]

    for i in range(1, len(ball_frames)):
        fi, x, y = ball_frames[i]
        prev_fi = cur[-1][0]
        if fi - prev_fi - 1 > max_gap:
            runs.append(cur)
            cur = [(fi, x, y)]
        else:
            cur.append((fi, x, y))

    runs.append(cur)
    return runs


def _check_toss_parabola(
    run: List[Tuple[int, float, float]],
    fps: float,
    net_dir: np.ndarray,
    vert_dir: np.ndarray,
    cfg: TossFilterConfig,
) -> Tuple[bool, str, int]:
    """Check if a run contains a toss segment that fits a gravity parabola.

    First finds the hit point (lateral velocity spike) and extracts the
    pre-hit sub-segment.  Then fits y_v = a*t² + b*t + c to verify
    gravity-consistent free flight.

    Works uniformly for near-court and far-court:
      - Near-court: run may already be just the toss (detection gap after
        ball moves too fast), so no cut needed.
      - Far-court: run includes toss + flight, cut at the lateral spike.
    """
    xs = np.array([x for _, x, _ in run])
    ys_img = np.array([y for _, _, y in run])

    # Project onto lateral (net-parallel) and vertical (perpendicular) axes
    dx = xs - xs[0]
    dy = ys_img - ys_img[0]
    lateral = dx * net_dir[0] + dy * net_dir[1]
    vertical = dx * vert_dir[0] + dy * vert_dir[1]

    # Find hit point: where lateral velocity suddenly jumps.
    # Compute per-frame lateral displacement.
    d_lat = np.diff(lateral)
    toss_end = len(run)  # default: entire run is toss

    # Median of |d_lat| during quiet phase (first few frames)
    quiet_n = min(5, len(d_lat))
    quiet_med = float(np.median(np.abs(d_lat[:quiet_n])))
    hit_thresh = max(cfg.min_hit_lateral_px, quiet_med * 3)

    for i in range(len(d_lat)):
        if abs(d_lat[i]) > hit_thresh:
            toss_end = i + 1  # include the frame before the jump
            break

    # Extract toss sub-segment
    if toss_end < cfg.min_points:
        return False, "", 0

    toss_frames = [fi for fi, _, _ in run[:toss_end]]
    toss_lat = lateral[:toss_end]
    toss_vert = vertical[:toss_end]

    # Vertical range
    v_range = float(np.max(toss_vert) - np.min(toss_vert))
    if v_range < 20:
        return False, "", 0

    # Check lateral stationarity relative to vertical range
    lat_std = float(np.std(toss_lat))
    if lat_std / v_range > cfg.max_lateral_ratio:
        return False, "", 0

    # Fit parabola: v(t) = a*t² + b*t + c
    t = np.array([(fi - toss_frames[0]) / fps for fi in toss_frames])
    coeffs = np.polyfit(t, toss_vert, 2)
    a = coeffs[0]

    # R²
    v_pred = np.polyval(coeffs, t)
    ss_res = float(np.sum((toss_vert - v_pred) ** 2))
    ss_tot = float(np.sum((toss_vert - np.mean(toss_vert)) ** 2))
    if ss_tot < 1e-6:
        return False, "", 0
    r2 = 1.0 - ss_res / ss_tot

    if r2 < cfg.min_r2:
        return False, "", 0

    detail = (
        f"r2={r2:.3f} lat_std={lat_std:.1f} "
        f"v_range={v_range:.0f} a={2*a:.0f}px/s² "
        f"n={toss_end}/{len(run)}"
    )
    return True, detail, toss_end


def _check_hit(
    get_ball: Callable[[int], Optional[list]],
    seg_last_frame: int,
    fps: float,
    net_dir: np.ndarray,
    cfg: TossFilterConfig,
) -> bool:
    """Look for a sudden lateral jump after the toss segment."""
    hit_window = int(round(cfg.hit_search_s * fps))
    prev_pos = get_ball(seg_last_frame)
    for fi in range(seg_last_frame + 1, seg_last_frame + hit_window + 1):
        pos = get_ball(fi)
        if pos is not None and prev_pos is not None:
            dx = pos[0] - prev_pos[0]
            dy = pos[1] - prev_pos[1]
            lateral_move = abs(dx * net_dir[0] + dy * net_dir[1])
            if lateral_move >= cfg.min_hit_lateral_px:
                return True
        if pos is not None:
            prev_pos = pos
    return False
