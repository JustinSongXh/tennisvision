"""YOLO26-pose tracker: player detection, tracking, and pose in one model call.

Replaces the old MoveNet (pose) + YOLOv8n (detection) pair.  A single
`yolo26n-pose.pt` forward pass produces bounding boxes, ByteTrack IDs, and
17 COCO keypoints for every player simultaneously.

`YOLOPoseTracker.push_frame()` is called once per frame and returns
`{track_id: (bbox, Pose)}`.  The pipeline no longer needs a separate
`PlayerDetector`.

Output `Pose.keypoints` is (17, 3) float32 in **(y, x, score)** order,
matching the convention expected by `_pose_to_feature` in stroke_classifier.

Two-stage tracking when a `Calibration` is supplied:
  * main tracker runs on the full frame at `imgsz` (typically 640) and
    catches near-side players with stable positive ByteTrack IDs.
  * far tracker runs on a tight crop around the far-court corners (plus
    margin) so far-side players fill more of the `imgsz` input — they get
    detected where the full-frame pass misses them.  IDs are negated to
    avoid collision with main-tracker IDs.
  * detections from the far tracker that overlap a main-tracker bbox
    (IoU > 0.5) are dropped so the same player isn't tracked twice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from ..court import reference as _ref
from ..court.homography import project_image_to_court as _proj


KEYPOINT_NAMES = (
    "nose",
    "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
)


@dataclass
class Pose:
    """One frame's pose: (17, 3) array of (y_px, x_px, score) in ORIGINAL image coords."""
    keypoints: np.ndarray   # shape (17, 3), dtype float32
    frame_idx: int

    def visible_mask(self, thr: float) -> np.ndarray:
        return self.keypoints[:, 2] >= thr


@dataclass
class YOLOPoseTrackerConfig:
    weights: str = "weights/yolo26n-pose.pt"
    device: str = "cpu"           # "cpu" | "cuda" | "cuda:0"
    conf: float = 0.3
    iou: float = 0.5
    imgsz: int = 640              # standard; small-target accuracy comes from court filter, not imgsz
    tracker: str = "bytetrack.yaml"
    score_threshold: float = 0.2  # per-keypoint visibility threshold (downstream)
    min_bbox_h: int = 60          # px — drop tiny / far detections
    max_persons: int = 4          # cap: 2 singles, 4 doubles
    court_margin_m: float = 3.0   # on-court filter tolerance in metres
    far_crop_margin_px: int = 40  # extra pixels around far-court corners
    far_merge_contained: float = 0.6  # above this fraction of overlap with a main bbox, drop far detection
    far_conf: float = 0.15        # lower conf for far-crop pass (net mesh cuts score)


# Convenience alias so pipeline code can import either name.
YOLOPoseConfig = YOLOPoseTrackerConfig


class YOLOPoseTracker:
    """Single-model player tracker + pose extractor.

    Call `push_frame(frame, frame_idx)` each frame to get
    `{track_id: (bbox, Pose)}` where bbox is (x0, y0, x1, y1) in pixels.

    On-court filtering is applied when a `Calibration` is supplied at
    construction; otherwise every detected person is kept.
    """

    def __init__(self, cfg: YOLOPoseTrackerConfig, calib=None):
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "YOLOPoseTracker needs ultralytics: `pip install ultralytics`"
            ) from e
        self.cfg = cfg
        self.calib = calib
        self._model_main = YOLO(cfg.weights)
        # Second YOLO instance holds its own ByteTrack state for the far-crop
        # pass — sharing one instance would cross-contaminate trackers.
        self._model_far = YOLO(cfg.weights) if calib is not None else None
        self._far_crop = self._compute_far_crop(calib) if calib is not None else None

    def reset(self) -> None:
        for m in (self._model_main, self._model_far):
            if m is None:
                continue
            try:
                m.predictor = None  # type: ignore[attr-defined]
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_far_crop(self, calib) -> Optional[Tuple[int, int, int, int]]:
        """Bbox around the far half of the court, projected to image px.

        Spans the far doubles corners down to the net line, plus
        `far_crop_margin_px` outward on sides/bottom.  The top extends
        further — a player standing on the far baseline has their feet
        at the projected corner and head well above, so we need extra
        headroom.  Estimate: ~2× the baseline-to-net image span
        (vertical compression squeezes the far half, but standing-player
        height is a fixed ~1.8m regardless of perspective).
        """
        from ..court.reference import COURT_WIDTH_M, COURT_LENGTH_M, NET_Y
        pts_m = [
            (0.0,           COURT_LENGTH_M),  # FL doubles
            (COURT_WIDTH_M, COURT_LENGTH_M),  # FR doubles
            (0.0,           NET_Y),           # net-left
            (COURT_WIDTH_M, NET_Y),           # net-right
        ]
        xs, ys = [], []
        for xm, ym in pts_m:
            v = np.array([xm, ym, 1.0])
            p = calib.H_real_to_img @ v
            xs.append(p[0] / p[2])
            ys.append(p[1] / p[2])
        m = self.cfg.far_crop_margin_px
        baseline_to_net_px = max(ys) - min(ys)
        top_pad = int(max(150, 2.0 * baseline_to_net_px))
        return (
            int(min(xs) - m),
            int(min(ys) - top_pad),
            int(max(xs) + m),
            int(max(ys) + m),
        )

    def _on_court(self, bbox: Tuple[float, float, float, float]) -> bool:
        if self.calib is None:
            return True
        x0, y0, x1, y1 = bbox
        foot_x, foot_y = 0.5 * (x0 + x1), y1
        try:
            rx, ry = _proj(self.calib.H_img_to_real, (foot_x, foot_y))
        except Exception:
            return True
        m = self.cfg.court_margin_m
        h = y1 - y0
        # Small + far-side detections are usually adjacent-court players
        # (upper-left / upper-right of frame).  Apply a tight x-margin so
        # they only pass if they project near our own doubles lane.
        if h < 150 and ry < _ref.NET_Y:
            x_margin = 1.0
        else:
            x_margin = m
        return (
            -x_margin <= rx <= _ref.COURT_WIDTH_M + x_margin
            and -m <= ry <= _ref.COURT_LENGTH_M + m
        )

    @staticmethod
    def _kp_to_pose(kp_xy: np.ndarray, frame_idx: int) -> Pose:
        """Convert YOLO keypoints (17, 3) in (x, y, conf) → Pose in (y, x, score)."""
        kp_yx = np.stack(
            [kp_xy[:, 1], kp_xy[:, 0], kp_xy[:, 2]], axis=1
        ).astype(np.float32)
        return Pose(keypoints=kp_yx, frame_idx=frame_idx)

    @staticmethod
    def _contained_in(far_bbox, main_bbox) -> float:
        """Fraction of `far_bbox` that lies inside `main_bbox`.

        Needed instead of IoU because the far crop can truncate a tall
        near-side player to just their head/shoulders — IoU would be low
        (boxes differ a lot in size) but the truncated bbox sits entirely
        inside the main tracker's full-body bbox.
        """
        ax0, ay0, ax1, ay1 = far_bbox
        bx0, by0, bx1, by1 = main_bbox
        iw = max(0, min(ax1, bx1) - max(ax0, bx0))
        ih = max(0, min(ay1, by1) - max(ay0, by0))
        inter = iw * ih
        far_area = max(1, (ax1 - ax0) * (ay1 - ay0))
        return float(inter) / far_area

    def _run(self, model, img, conf):
        """Run one tracker call and return (boxes, ids, confs, kps) arrays."""
        results = model.track(
            img,
            persist=True,
            verbose=False,
            classes=[0],
            conf=conf,
            iou=self.cfg.iou,
            tracker=self.cfg.tracker,
            device=self.cfg.device,
            imgsz=self.cfg.imgsz,
        )
        r = results[0]
        if (r.boxes is None or r.boxes.id is None
                or r.keypoints is None or len(r.keypoints.data) == 0):
            return None
        return (
            r.boxes.xyxy.cpu().numpy(),
            r.boxes.id.cpu().numpy().astype(int),
            r.boxes.conf.cpu().numpy(),
            r.keypoints.data.cpu().numpy(),
        )

    def _track_far(
        self, frame: np.ndarray, frame_idx: int
    ) -> Dict[int, Tuple[Tuple[int, int, int, int], Pose]]:
        if self._model_far is None or self._far_crop is None:
            return {}
        H, W = frame.shape[:2]
        x0c, y0c, x1c, y1c = self._far_crop
        x0c = max(0, x0c); y0c = max(0, y0c)
        x1c = min(W, x1c); y1c = min(H, y1c)
        if (x1c - x0c) < 40 or (y1c - y0c) < 40:
            return {}
        crop = frame[y0c:y1c, x0c:x1c]
        packed = self._run(self._model_far, crop, self.cfg.far_conf)
        if packed is None:
            return {}
        boxes, ids, confs, kps = packed
        # Remap crop coords → original frame coords.
        boxes = boxes.copy();  boxes[:, 0::2] += x0c; boxes[:, 1::2] += y0c
        kps   = kps.copy();    kps[:, :, 0]   += x0c; kps[:, :, 1]   += y0c

        out: Dict[int, Tuple[Tuple[int, int, int, int], Pose]] = {}
        for idx in np.argsort(-confs):
            x0, y0, x1, y1 = boxes[idx]
            # Looser height gate — far-side players are genuinely smaller.
            if (y1 - y0) < self.cfg.min_bbox_h * 0.5:
                continue
            bbox = (int(x0), int(y0), int(x1), int(y1))
            if not self._on_court(bbox):
                continue
            pose = self._kp_to_pose(kps[idx], frame_idx)
            out[int(ids[idx])] = (bbox, pose)
        return out

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push_frame(
        self, frame: np.ndarray, frame_idx: int
    ) -> Dict[int, Tuple[Tuple[int, int, int, int], Pose]]:
        """Run detection + tracking + pose on one frame.

        Returns {track_id: (bbox, Pose)} filtered by height, court bounds,
        and max_persons.  Empty dict when no players are detected.
        """
        packed = self._run(self._model_main, frame, self.cfg.conf)
        out: Dict[int, Tuple[Tuple[int, int, int, int], Pose]] = {}
        if packed is not None:
            boxes, ids, confs, kps = packed
            for idx in np.argsort(-confs):
                if len(out) >= self.cfg.max_persons:
                    break
                x0, y0, x1, y1 = boxes[idx]
                if (y1 - y0) < self.cfg.min_bbox_h:
                    continue
                bbox = (int(x0), int(y0), int(x1), int(y1))
                if not self._on_court(bbox):
                    continue
                pose = self._kp_to_pose(kps[idx], frame_idx)
                out[int(ids[idx])] = (bbox, pose)

        # Far-crop pass — only adds players not already covered by main.
        far = self._track_far(frame, frame_idx)
        if far:
            main_boxes = [bbox for (bbox, _) in out.values()]
            for tid, (bbox, pose) in far.items():
                if len(out) >= self.cfg.max_persons:
                    break
                if any(self._contained_in(bbox, mb) > self.cfg.far_merge_contained for mb in main_boxes):
                    continue
                # Negate ID so it never collides with main's positive IDs.
                out[-int(tid)] = (bbox, pose)

        return out
