"""Champion-track trail overlay."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np


CHAMPION_COLOR = (0, 255, 255)


def draw_trail(
    frame: np.ndarray,
    points: Sequence,              # objects with .x .y .frame
    current_frame: int,
    tail: int = 45,
    is_current_det: bool = False,
) -> None:
    recent = [(p.x, p.y, p.frame) for p in points if current_frame - p.frame <= tail]
    if len(recent) >= 2:
        arr = np.array([(x, y) for x, y, _ in recent], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [arr], False, CHAMPION_COLOR, 2)
    for x, y, f in recent:
        age = current_frame - f
        alpha = 1.0 - age / float(tail + 1)
        r = 3 if age > 0 else 6
        col = tuple(int(c * alpha) for c in CHAMPION_COLOR)
        cv2.circle(frame, (x, y), r, col, -1)
    if is_current_det and recent:
        lx, ly, _ = recent[-1]
        cv2.circle(frame, (lx, ly), 14, CHAMPION_COLOR, 2)
