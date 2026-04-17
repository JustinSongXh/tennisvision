"""Standard tennis-court geometry (ITF doubles) and 14 keypoint labels.

The 14 keypoints follow yastrebksv/TennisCourtDetector convention so that
its pretrained weights are directly usable.  Origin is the near-baseline
x left-DOUBLES-sideline corner; +x goes right along the baseline, +y
goes toward the far baseline.  Units are meters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


# --- court dimensions (ITF doubles) ---
COURT_WIDTH_M  = 10.97          # doubles baseline
COURT_LENGTH_M = 23.77          # baseline to baseline
SINGLES_INSET  = 1.37           # alley width
SERVICE_Y      = 6.40           # near service line y
NET_Y          = COURT_LENGTH_M / 2.0   # 11.885
NET_OVERHANG   = 0.914

# 14 keypoints (id -> (x_m, y_m))
# Indices chosen so the first 4 are the 4 doubles corners — handy for
# legacy code that only wants those.  Remaining indices are the singles
# corners + service-line intersections + T-points, matching the
# TennisCourtDetector convention.
KEYPOINTS_M: dict[int, tuple[float, float]] = {
    # doubles corners (outer rectangle)
    0:  (0.0,                COURT_LENGTH_M),   # FL  far-left doubles
    1:  (COURT_WIDTH_M,      COURT_LENGTH_M),   # FR  far-right doubles
    2:  (COURT_WIDTH_M,      0.0),              # NR  near-right doubles
    3:  (0.0,                0.0),              # NL  near-left doubles
    # singles corners (inner rectangle)
    4:  (SINGLES_INSET,                 COURT_LENGTH_M),   # far-left singles
    5:  (COURT_WIDTH_M - SINGLES_INSET, COURT_LENGTH_M),   # far-right singles
    6:  (COURT_WIDTH_M - SINGLES_INSET, 0.0),              # near-right singles
    7:  (SINGLES_INSET,                 0.0),              # near-left singles
    # near service-line endpoints (between singles sidelines)
    8:  (SINGLES_INSET,                 SERVICE_Y),
    9:  (COURT_WIDTH_M - SINGLES_INSET, SERVICE_Y),
    # far service-line endpoints
    10: (SINGLES_INSET,                 COURT_LENGTH_M - SERVICE_Y),
    11: (COURT_WIDTH_M - SINGLES_INSET, COURT_LENGTH_M - SERVICE_Y),
    # service T-points (center-line ∩ service lines)
    12: (COURT_WIDTH_M / 2.0,           SERVICE_Y),
    13: (COURT_WIDTH_M / 2.0,           COURT_LENGTH_M - SERVICE_Y),
}

NUM_KEYPOINTS = len(KEYPOINTS_M)

LABELS = {
    0: "FL",  1: "FR",  2: "NR",  3: "NL",
    4: "FLs", 5: "FRs", 6: "NRs", 7: "NLs",
    8: "NLv", 9: "NRv", 10: "FLv", 11: "FRv",
    12: "Tn", 13: "Tf",
}


# --- 4-point subsets used for best-of homography estimation ---
# Each tuple is 4 keypoint ids that, if all detected, can yield a valid
# planar homography.  Ordered roughly from "most informative" (doubles
# corners span the whole court) to "local" (service box corners span
# just the near half).  The homography estimator tries each in turn.
COURT_CONF: list[tuple[int, int, int, int]] = [
    # full doubles rectangle — best conditioning
    (0, 1, 2, 3),
    # full singles rectangle
    (4, 5, 6, 7),
    # near half: doubles corners + far service endpoints
    (3, 2, 11, 10),
    (7, 6, 11, 10),
    # near service box (singles side)
    (7, 6, 9, 8),
    # near right service box
    (12, 9, 6, 13),  # uses service T, mixed
    # 2 baselines + far service line + sidelines (top of court)
    (0, 1, 11, 10),
    (4, 5, 11, 10),
    # near baseline + near service line
    (3, 2, 9, 8),
    (7, 6, 9, 8),
]


@dataclass(frozen=True)
class Keypoint:
    id: int
    x: float          # image pixels
    y: float          # image pixels
    confidence: float = 1.0


def doubles_corner_ids() -> tuple[int, int, int, int]:
    return (0, 1, 2, 3)


def real_coord(kp_id: int) -> tuple[float, float]:
    return KEYPOINTS_M[kp_id]


def court_line_segments_m() -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Line segments (in meters) that together trace the full court.
    Used to project the full court overlay onto an image via H_inv.
    """
    W, L, s = COURT_WIDTH_M, COURT_LENGTH_M, SINGLES_INSET
    return [
        # outer doubles rectangle
        ((0, 0),   (W, 0)),
        ((W, 0),   (W, L)),
        ((W, L),   (0, L)),
        ((0, L),   (0, 0)),
        # singles sidelines
        ((s, 0),     (s, L)),
        ((W - s, 0), (W - s, L)),
        # service lines (between singles)
        ((s, SERVICE_Y),     (W - s, SERVICE_Y)),
        ((s, L - SERVICE_Y), (W - s, L - SERVICE_Y)),
        # center service line
        ((W / 2.0, SERVICE_Y),      (W / 2.0, L - SERVICE_Y)),
        # net
        ((-NET_OVERHANG, NET_Y), (W + NET_OVERHANG, NET_Y)),
    ]
