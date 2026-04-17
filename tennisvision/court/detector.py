"""Court-keypoint detectors.

Three implementations share a common Protocol so the pipeline can swap
one for another via config:

    - RedMarkExtractor:  extract 4 doubles corners from a user-drawn
                         red-line annotation image.  Offline, fast, no
                         dependencies beyond OpenCV.  Good for a single
                         reference frame the user has marked.
    - HeatmapDetector:   yastrebksv TennisCourtDetector CNN (stub here;
                         implemented in tennisvision.court.tcd_model).
    - ResNet50Regressor: CourtCheck-style ResNet50 + Linear(28) (stub).
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import cv2
import numpy as np

from . import reference as ref
from .homography import (
    HomographyResult, estimate_homography, homography_from_4_corners,
)


@runtime_checkable
class CourtDetector(Protocol):
    def detect(self, frame: np.ndarray) -> dict:
        """Return a dict mapping keypoint_id -> (x, y) image-pixel coords.
        Only detected keypoints are in the result; missing ones are
        absent (not None).
        """


# ---------------------------------------------------------------------------
# Blue-surface contour extractor (for fully in-frame courts)
# ---------------------------------------------------------------------------

class BlueContourDetector:
    """For videos where the entire blue doubles court is inside the frame
    and clearly bounded by green surround.  Unions per-frame largest blue
    connected-components across many frames (to heal player occlusions),
    then fits a 4-vertex polygon to the blob.

    Only returns the 4 doubles corners (keypoint ids 0..3 in FL/FR/NR/NL
    order).  Fails (returns {}) if the blob isn't contiguous or can't be
    approximated by a quadrilateral.
    """

    BLUE_LOW  = (95, 60, 60)
    BLUE_HIGH = (125, 255, 230)
    SAMPLE_FRAMES = 40       # how many frames to union (per-call API)
    CLOSE_KERNEL  = 31       # bridge near+far halves across the net gap
    CENTER_BIAS   = True     # prefer CC closest to image center over just largest

    def __init__(self, blue_low=None, blue_high=None):
        if blue_low is not None:
            self.BLUE_LOW = tuple(blue_low)
        if blue_high is not None:
            self.BLUE_HIGH = tuple(blue_high)

    # Single-frame API (satisfies the CourtDetector Protocol but is
    # inferior to `calibrate_from_video` for occluded scenes).
    def detect(self, frame: np.ndarray) -> dict:
        mask = self._single_frame_mask(frame)
        return self._corners_from_mask(mask)

    # Preferred API: run on a video path, unioning many frames.
    def calibrate_from_video(self, video_path: str):
        import cv2
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = np.linspace(10, max(total - 10, 11), self.SAMPLE_FRAMES).astype(int)
        accum = None
        sample = None
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, f = cap.read()
            if not ok:
                continue
            if sample is None:
                sample = f
            m = self._single_frame_mask(f, keep_contour=True)
            accum = m if accum is None else cv2.bitwise_or(accum, m)
        cap.release()
        if accum is None:
            return sample, {}
        accum = cv2.morphologyEx(accum, cv2.MORPH_CLOSE,
                                 np.ones((self.CLOSE_KERNEL,) * 2, np.uint8))
        return sample, self._corners_from_mask(accum)

    def _single_frame_mask(self, frame: np.ndarray, keep_contour=False) -> np.ndarray:
        import cv2
        H, W = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        m = cv2.inRange(hsv, np.array(self.BLUE_LOW, np.uint8),
                              np.array(self.BLUE_HIGH, np.uint8))
        if keep_contour:
            # Close the net-sized gap first so near+far halves fuse
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                                 np.ones((self.CLOSE_KERNEL,) * 2, np.uint8))
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return m
            if self.CENTER_BIAS:
                cx, cy = W / 2.0, H / 2.0
                def score(c):
                    M = cv2.moments(c)
                    if M["m00"] < 1e-6:
                        return 1e9
                    x, y = M["m10"] / M["m00"], M["m01"] / M["m00"]
                    # penalize distance from image center, reward area
                    return ((x - cx) ** 2 + (y - cy) ** 2) / (cv2.contourArea(c) + 1.0)
                chosen = min(contours, key=score)
            else:
                chosen = max(contours, key=cv2.contourArea)
            out = np.zeros_like(m)
            cv2.drawContours(out, [chosen], -1, 255, -1)
            return out
        return m

    def _corners_from_mask(self, mask) -> dict:
        import cv2
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return {}
        cnt = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(cnt)
        peri = cv2.arcLength(hull, True)
        quad = None
        for eps in np.linspace(0.005, 0.10, 40):
            approx = cv2.approxPolyDP(hull, eps * peri, True)
            if len(approx) == 4:
                quad = approx.reshape(-1, 2)
                break
        if quad is None:
            return {}

        # Order as [FL, FR, NR, NL] — "far" = smaller y (image top),
        # "left" = smaller x.
        pts = np.asarray(quad, dtype=np.float32)
        idx_sort_y = np.argsort(pts[:, 1])
        far = pts[idx_sort_y[:2]]
        near = pts[idx_sort_y[2:]]
        fl, fr = far[np.argsort(far[:, 0])]
        nl, nr = near[np.argsort(near[:, 0])]
        return {
            0: (float(fl[0]), float(fl[1])),
            1: (float(fr[0]), float(fr[1])),
            2: (float(nr[0]), float(nr[1])),
            3: (float(nl[0]), float(nl[1])),
        }


# ---------------------------------------------------------------------------
# Red-marked reference image extractor
# ---------------------------------------------------------------------------

class RedMarkExtractor:
    """Extract 4 doubles-corner keypoints from a user-annotated reference
    image in which the court-boundary lines are drawn in red.

    Pipeline: red HSV mask -> Canny -> HoughLinesP -> angle+extent
    clustering -> pick 2 horizontal baselines + 2 sloped sidelines ->
    intersect for corners.
    """

    HSV_RED_RANGES = (
        ((0,   110, 110), (12,  255, 255)),
        ((168, 110, 110), (180, 255, 255)),
    )
    HOUGH_MIN_LENGTH = 80
    HOUGH_MAX_GAP    = 25
    HOUGH_THRESHOLD  = 40
    LINE_ANGLE_TOL_DEG = 4.0
    LINE_OFFSET_TOL_PX = 10.0
    MIN_CLUSTER_LEN = 200
    HORIZ_ANGLE_MAX = 12.0      # |angle| <= this AND wide x-span => horizontal
    HORIZ_X_SPAN_FRAC = 0.40     # fraction of image width
    LEFT_ANGLE_MAX = -3.0        # left sideline: angle < this
    RIGHT_ANGLE_MIN = 10.0

    def detect(self, frame: np.ndarray) -> dict[int, tuple[float, float]]:
        _, _, corners = self._detect_corners(frame)
        if corners is None:
            return {}
        fl, fr, nr, nl = corners
        return {0: fl, 1: fr, 2: nr, 3: nl}

    def calibrate(self, frame: np.ndarray) -> Optional[HomographyResult]:
        _, _, corners = self._detect_corners(frame)
        if corners is None:
            return None
        return homography_from_4_corners(list(corners))

    # --- internals ---

    def _red_mask(self, img: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        (lo1, hi1), (lo2, hi2) = self.HSV_RED_RANGES
        m1 = cv2.inRange(hsv, np.array(lo1, np.uint8), np.array(hi1, np.uint8))
        m2 = cv2.inRange(hsv, np.array(lo2, np.uint8), np.array(hi2, np.uint8))
        m = cv2.bitwise_or(m1, m2)
        return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    def _seg_to_line(self, x1, y1, x2, y2):
        a = float(y2 - y1); b = float(x1 - x2); c = -(a * x1 + b * y1)
        n = float(np.hypot(a, b))
        if n < 1e-9: return None
        return a / n, b / n, c / n

    def _line_angle_deg(self, a, b):
        theta = np.degrees(np.arctan2(a, -b))
        if theta > 90:  theta -= 180
        if theta < -90: theta += 180
        return theta

    def _lines_same(self, l1, l2) -> bool:
        a1, b1, c1 = l1
        a2, b2, c2 = l2
        if a1 * a2 + b1 * b2 < 0:
            a2, b2, c2 = -a2, -b2, -c2
        dot = max(-1.0, min(1.0, a1 * a2 + b1 * b2))
        if np.degrees(np.arccos(dot)) > self.LINE_ANGLE_TOL_DEG:
            return False
        return abs(c1 - c2) < self.LINE_OFFSET_TOL_PX

    def _fit_line(self, pts):
        pts = np.asarray(pts, dtype=np.float32)
        fit = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        vx, vy, x0, y0 = float(fit[0]), float(fit[1]), float(fit[2]), float(fit[3])
        a, b = vy, -vx
        c = -(a * x0 + b * y0)
        n = float(np.hypot(a, b))
        return a / n, b / n, c / n

    def _intersect(self, l1, l2) -> Optional[tuple[float, float]]:
        a1, b1, c1 = l1; a2, b2, c2 = l2
        d = a1 * b2 - a2 * b1
        if abs(d) < 1e-6: return None
        x = (b1 * c2 - b2 * c1) / d
        y = (a2 * c1 - a1 * c2) / d
        return x, y

    def _cluster(self, segs):
        clusters = []
        for s in segs:
            line = self._seg_to_line(*s)
            if line is None: continue
            merged = False
            for cl in clusters:
                if self._lines_same(line, cl["line"]):
                    cl["segs"].append(s)
                    pts = []
                    for ss in cl["segs"]:
                        pts += [(ss[0], ss[1]), (ss[2], ss[3])]
                    cl["line"] = self._fit_line(pts)
                    merged = True
                    break
            if not merged:
                clusters.append({"segs": [s], "line": line})
        for cl in clusters:
            cl["length"] = sum(
                float(np.hypot(s[2] - s[0], s[3] - s[1])) for s in cl["segs"])
            xs = []
            for s in cl["segs"]:
                xs += [s[0], s[2]]
            cl["x_span"] = max(xs) - min(xs)
            cl["cx"] = float(np.mean([0.5 * (s[0] + s[2]) for s in cl["segs"]]))
            cl["cy"] = float(np.mean([0.5 * (s[1] + s[3]) for s in cl["segs"]]))
            a, b, _ = cl["line"]
            cl["angle"] = self._line_angle_deg(a, b)
        return clusters

    def _detect_corners(self, frame):
        H_img, W_img = frame.shape[:2]
        mask = self._red_mask(frame)
        edges = cv2.Canny(mask, 40, 120)
        raw = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=self.HOUGH_THRESHOLD,
            minLineLength=self.HOUGH_MIN_LENGTH,
            maxLineGap=self.HOUGH_MAX_GAP,
        )
        segs = []
        if raw is not None:
            for s in raw:
                x1, y1, x2, y2 = [int(v) for v in s[0]]
                segs.append((x1, y1, x2, y2))
        if not segs:
            return mask, [], None

        clusters = [c for c in self._cluster(segs) if c["length"] >= self.MIN_CLUSTER_LEN]
        horiz = [c for c in clusters
                 if c["x_span"] >= self.HORIZ_X_SPAN_FRAC * W_img
                 and abs(c["angle"]) < self.HORIZ_ANGLE_MAX]
        lefts = [c for c in clusters
                 if c["cx"] < W_img / 2.0
                 and c["angle"] < self.LEFT_ANGLE_MAX
                 and c["length"] >= 300]
        rights = [c for c in clusters
                  if c["cx"] > W_img / 2.0
                  and c["angle"] > self.RIGHT_ANGLE_MIN
                  and c["length"] >= 300]
        if len(horiz) < 2 or not lefts or not rights:
            return mask, clusters, None

        horiz.sort(key=lambda c: c["cy"])       # top to bottom
        service_h = horiz[-2]                   # 2nd from bottom
        base_h    = horiz[-1]                   # bottommost

        y_ref = base_h["cy"]
        def x_at_y(line, y):
            a, b, c = line
            if abs(a) < 1e-9: return None
            return -(b * y + c) / a

        lefts_scored  = sorted(
            [(x_at_y(c["line"], y_ref), c) for c in lefts  if x_at_y(c["line"], y_ref) is not None],
            key=lambda xc: xc[0],
        )
        rights_scored = sorted(
            [(x_at_y(c["line"], y_ref), c) for c in rights if x_at_y(c["line"], y_ref) is not None],
            key=lambda xc: -xc[0],
        )
        if not lefts_scored or not rights_scored:
            return mask, clusters, None
        left_s  = lefts_scored[0][1]
        right_s = rights_scored[0][1]

        BL = self._intersect(base_h["line"],    left_s["line"])
        BR = self._intersect(base_h["line"],    right_s["line"])
        SL = self._intersect(service_h["line"], left_s["line"])
        SR = self._intersect(service_h["line"], right_s["line"])
        if None in (BL, BR, SL, SR):
            return mask, clusters, None

        # Anchor (BL, BR, SR, SL) -> real (0,0)/(W,0)/(W,SERVICE_Y)/(0,SERVICE_Y)
        anchor_img  = np.array([BL, BR, SR, SL], dtype=np.float32)
        anchor_real = np.array([
            (0,                  0),
            (ref.COURT_WIDTH_M,  0),
            (ref.COURT_WIDTH_M,  ref.SERVICE_Y),
            (0,                  ref.SERVICE_Y),
        ], dtype=np.float32)
        H, _ = cv2.findHomography(anchor_img, anchor_real, method=0)
        if H is None:
            return mask, clusters, None
        H_inv = np.linalg.inv(H)

        # Project 4 doubles corners back to image
        def _proj(pt):
            v = np.array([pt[0], pt[1], 1.0])
            p = H_inv @ v
            return float(p[0] / p[2]), float(p[1] / p[2])
        FL = _proj((0,                  ref.COURT_LENGTH_M))
        FR = _proj((ref.COURT_WIDTH_M,  ref.COURT_LENGTH_M))
        NR = _proj((ref.COURT_WIDTH_M,  0))
        NL = _proj((0,                  0))
        return mask, clusters, (FL, FR, NR, NL)
