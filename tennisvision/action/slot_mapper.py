"""4-slot spatial mapping for doubles tennis.

Maps detected players to fixed court positions using homography,
solving ByteTrack ID fragmentation (40+ IDs → 4 stable slots).

Slots:
  0 = NEAR_L (近端左)
  1 = NEAR_R (近端右)
  2 = FAR_L  (远端左)
  3 = FAR_R  (远端右)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


SLOT_NAMES = {0: "NEAR_L", 1: "NEAR_R", 2: "FAR_L", 3: "FAR_R"}


@dataclass
class SlotMapperConfig:
    court_width_m: float = 10.97
    court_length_m: float = 23.77
    court_margin_m: float = 3.0


class SlotMapper:
    """Map detections to 4 fixed court slots via homography."""

    def __init__(self, H_img_to_real: np.ndarray, cfg: Optional[SlotMapperConfig] = None):
        self.H = np.asarray(H_img_to_real, dtype=np.float64)
        self.cfg = cfg or SlotMapperConfig()
        self.net_y_m = self.cfg.court_length_m / 2

    def to_court(self, foot_x: float, foot_y: float) -> Tuple[Optional[float], Optional[float]]:
        """Project image foot point to court coordinates (metres)."""
        p = self.H @ [foot_x, foot_y, 1.0]
        if abs(p[2]) < 1e-9:
            return None, None
        return float(p[0] / p[2]), float(p[1] / p[2])

    def on_court(self, rx: float, ry: float) -> bool:
        m = self.cfg.court_margin_m
        w, l = self.cfg.court_width_m, self.cfg.court_length_m
        return (-m <= rx <= w + m and -m <= ry <= l + m)

    def get_slot(self, rx: float, ry: float) -> int:
        """Return slot index (0-3) for a court coordinate."""
        mid_x = self.cfg.court_width_m / 2
        if ry < self.net_y_m:
            return 0 if rx < mid_x else 1
        else:
            return 2 if rx < mid_x else 3

    def assign(self, detections: List[dict]) -> Dict[int, dict]:
        """Assign detections to slots. Each slot gets the closest on-court detection.

        Args:
            detections: list of dicts with 'foot_x', 'foot_y' and other fields

        Returns:
            {slot_id: detection_dict} for occupied slots
        """
        w = self.cfg.court_width_m
        l = self.cfg.court_length_m
        slot_centers = {
            0: (w * 0.25, l * 0.15),
            1: (w * 0.75, l * 0.15),
            2: (w * 0.25, l * 0.85),
            3: (w * 0.75, l * 0.85),
        }

        candidates = []
        for det in detections:
            rx, ry = self.to_court(det["foot_x"], det["foot_y"])
            if rx is None or not self.on_court(rx, ry):
                continue
            slot = self.get_slot(rx, ry)
            cx, cy = slot_centers[slot]
            dist = np.sqrt((rx - cx) ** 2 + (ry - cy) ** 2)
            candidates.append((slot, dist, det))

        # Each slot takes the closest detection
        candidates.sort(key=lambda x: x[1])
        assigned = {}
        for slot, dist, det in candidates:
            if slot not in assigned:
                assigned[slot] = det
        return assigned
