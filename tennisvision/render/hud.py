"""Per-frame text HUD."""

from __future__ import annotations

import cv2
import numpy as np


def draw_hud(frame: np.ndarray, text: str,
             origin: tuple[int, int] = (10, 30),
             scale: float = 0.55,
             color: tuple[int, int, int] = (255, 255, 255),
             thickness: int = 2) -> None:
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness)
