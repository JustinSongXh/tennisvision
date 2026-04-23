"""Debug rendering for player detection + stroke classification.

Two helpers:
  - `draw_players`  — colored bbox per YOLOv8 track_id with "#id" tag
  - `draw_stroke_labels` — shows each player's most recent stroke event
    (within `ttl_frames`) above their bbox, fading linearly with age.

Colors are stable per track_id via golden-ratio HSV hashing, so the same
player keeps the same color across frames even when the track list
changes.  Kept in `render/` next to trail/minimap/hud for symmetry.
"""

from __future__ import annotations

import bisect
import colorsys
from typing import Dict, Iterable, Sequence, Tuple

import cv2
import numpy as np


def color_for(track_id: int) -> Tuple[int, int, int]:
    """Stable BGR color for a YOLOv8 track ID.  Hue advances by golden
    ratio so neighbouring IDs are visually distinct."""
    h = ((int(track_id) * 0.61803398875) % 1.0)
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


def draw_players(frame: np.ndarray,
                 bboxes: Dict[int, Tuple[int, int, int, int]],
                 thickness: int = 2) -> None:
    """Draw colored bounding rectangles over `frame` (in place).  Each
    player gets its own color determined by `track_id`."""
    for tid, (x0, y0, x1, y1) in bboxes.items():
        col = color_for(tid)
        cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)),
                      col, thickness)
        tag = "#%d" % tid
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(th + 4, int(y0) - 4)
        cv2.rectangle(frame, (int(x0), ty - th - 3),
                      (int(x0) + tw + 4, ty + 3), col, -1)
        cv2.putText(frame, tag, (int(x0) + 2, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def _latest_event(events_sorted: Sequence, frame_idx: int, ttl_frames: int):
    """Return the latest event with `frame <= frame_idx` and age <= ttl,
    or None.  `events_sorted` must be ascending by .frame."""
    if not events_sorted:
        return None
    # bisect_right by frame
    frames = [e.frame for e in events_sorted]
    i = bisect.bisect_right(frames, frame_idx) - 1
    if i < 0:
        return None
    ev = events_sorted[i]
    if frame_idx - ev.frame > ttl_frames:
        return None
    return ev


def draw_stroke_labels(frame: np.ndarray,
                       bboxes: Dict[int, Tuple[int, int, int, int]],
                       events_by_player: Dict[int, Sequence],
                       frame_idx: int,
                       ttl_frames: int = 60) -> None:
    """Above each visible player bbox, draw "LABEL PROB" of their most
    recent stroke event.  Fades linearly to ~25% intensity over ttl_frames
    so older events remain visible but recede behind newer ones."""
    for tid, (x0, y0, x1, y1) in bboxes.items():
        ev = _latest_event(events_by_player.get(tid, ()), frame_idx, ttl_frames)
        if ev is None:
            continue
        age = frame_idx - ev.frame
        fade = max(0.25, 1.0 - age / float(ttl_frames))
        col = color_for(tid)
        col_faded = tuple(int(c * fade) for c in col)
        text = "%s %.2f" % (ev.label, ev.confidence)
        # Place label ABOVE the "#id" tag — add a bit of headroom.
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        ly = max(th + 22, int(y0) - 22)
        bg = (0, 0, 0)
        cv2.rectangle(frame, (int(x0), ly - th - 4),
                      (int(x0) + tw + 6, ly + 4), bg, -1)
        cv2.putText(frame, text, (int(x0) + 3, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col_faded, 2, cv2.LINE_AA)


def events_by_player_sorted(events: Iterable) -> Dict[int, list]:
    """Bucket stroke events by player_id and sort each bucket by frame.
    Events with player_id == None (single-player fallback) are dropped."""
    out: Dict[int, list] = {}
    for ev in events:
        pid = getattr(ev, "player_id", None)
        if pid is None:
            continue
        out.setdefault(int(pid), []).append(ev)
    for pid in out:
        out[pid].sort(key=lambda e: e.frame)
    return out
