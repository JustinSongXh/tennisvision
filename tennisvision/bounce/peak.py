"""y-peak bounce detector (baseline / fallback).

Scans a `(x, y, frame)` trajectory for local maxima of image-y.  The
cheapest possible signal — good enough for a first pass but fires on
every hit-apex; upgrade to CatBoost (bounce.catboost) when weights are
available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class BounceEvent:
    frame: int
    x_img: float
    y_img: float


@dataclass
class PeakBounceConfig:
    lookback: int = 3           # frames on either side
    min_dy: float = 4.0         # peak height threshold (px)
    cooldown: int = 10          # frames between successive bounces


class PeakBounceDetector:
    """Online peak detector.  Call `check()` after every new observation
    appended to the champion track; returns a BounceEvent or None.
    """

    def __init__(self, cfg: PeakBounceConfig):
        self.cfg = cfg
        self._last_bounce_frame = -1 << 30

    def check(self, points: Sequence) -> "BounceEvent | None":
        """`points` is a sequence of objects with x, y, frame attributes
        (typically tracker.TrackPoint).
        """
        lb = self.cfg.lookback
        n = len(points)
        if n < 2 * lb + 1:
            return None
        i = n - lb - 1          # candidate peak index
        p_peak = points[i]
        if p_peak.frame - self._last_bounce_frame < self.cfg.cooldown:
            return None
        y_peak = p_peak.y
        left  = [points[j].y for j in range(i - lb, i)]
        right = [points[j].y for j in range(i + 1, i + lb + 1)]
        if not (all(y_peak >= y for y in left) and all(y_peak >= y for y in right)):
            return None
        if (y_peak - max(min(left), min(right))) < self.cfg.min_dy:
            return None
        self._last_bounce_frame = p_peak.frame
        return BounceEvent(frame=p_peak.frame,
                           x_img=float(p_peak.x), y_img=float(p_peak.y))


# Batch variant for use after Pass-1 tracking is complete.
def detect_all(points: Sequence, cfg: PeakBounceConfig) -> list[BounceEvent]:
    det = PeakBounceDetector(cfg)
    out: list[BounceEvent] = []
    # simulate online feeding: the detector inspects a sliding window
    # each time, so we just call check() at each prefix length.
    for end in range(2 * cfg.lookback + 1, len(points) + 1):
        ev = det.check(points[:end])
        if ev is not None:
            out.append(ev)
    return out
