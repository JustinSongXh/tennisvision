"""YOLO26-pose tracker: player detection, tracking, and pose in one model call.

Replaces the old MoveNet (pose) + YOLOv8n (detection) pair.  A single
`yolo26n-pose.pt` forward pass produces bounding boxes, ByteTrack IDs, and
17 COCO keypoints for every player simultaneously.

`YOLOPoseTracker.push_frame()` is called once per frame and returns
`{track_id: (bbox, Pose)}`.  The pipeline no longer needs a separate
`PlayerDetector`.

Output `Pose.keypoints` is (17, 3) float32 in **(y, x, score)** order,
matching the convention expected by `_pose_to_feature` in stroke_classifier.
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
    imgsz: int = 640
    tracker: str = "bytetrack.yaml"
    score_threshold: float = 0.2  # per-keypoint visibility threshold (downstream)
    min_bbox_h: int = 60          # px — drop tiny / far detections
    max_persons: int = 4          # cap: 2 singles, 4 doubles
    court_margin_m: float = 3.0   # on-court filter tolerance in metres


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
        self._model = YOLO(cfg.weights)

    def reset(self) -> None:
        try:
            self._model.predictor = None  # type: ignore[attr-defined]
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _on_court(self, bbox: Tuple[float, float, float, float]) -> bool:
        if self.calib is None:
            return True
        x0, _, x1, y1 = bbox
        foot_x, foot_y = 0.5 * (x0 + x1), y1
        try:
            rx, ry = _proj(self.calib.H_img_to_real, (foot_x, foot_y))
        except Exception:
            return True
        m = self.cfg.court_margin_m
        return (
            -m <= rx <= _ref.COURT_WIDTH_M + m
            and -m <= ry <= _ref.COURT_LENGTH_M + m
        )

    @staticmethod
    def _kp_to_pose(kp_xy: np.ndarray, frame_idx: int) -> Pose:
        """Convert YOLO keypoints (17, 3) in (x, y, conf) → Pose in (y, x, score)."""
        kp_yx = np.stack(
            [kp_xy[:, 1], kp_xy[:, 0], kp_xy[:, 2]], axis=1
        ).astype(np.float32)
        return Pose(keypoints=kp_yx, frame_idx=frame_idx)

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
        results = self._model.track(
            frame,
            persist=True,
            verbose=False,
            classes=[0],
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            tracker=self.cfg.tracker,
            device=self.cfg.device,
            imgsz=self.cfg.imgsz,
        )
        r = results[0]

        if (r.boxes is None or r.boxes.id is None
                or r.keypoints is None or len(r.keypoints.data) == 0):
            return {}

        boxes = r.boxes.xyxy.cpu().numpy()          # (N, 4)
        ids   = r.boxes.id.cpu().numpy().astype(int)  # (N,)
        confs = r.boxes.conf.cpu().numpy()           # (N,)
        kps   = r.keypoints.data.cpu().numpy()       # (N, 17, 3): x, y, conf

        out: Dict[int, Tuple[Tuple[int, int, int, int], Pose]] = {}
        for idx in np.argsort(-confs):               # high-confidence first
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

        return out
