"""Per-frame player detection + tracking via YOLOv8.

We use ultralytics' built-in `.track(persist=True)` which pipes through
ByteTrack/BoTSORT and returns persistent track IDs across frames — good
enough for singles (2 IDs) and doubles (4 IDs) without any special
handling for "how many players are on court".

Bystanders (coach, ball-kid, spectator catching a stray ball) are the
main false positives on amateur footage.  Filtering layers:
  - min bbox height      — drops very-far / tiny detections
  - on-court check       — projects the bbox foot (bottom center) through
                           the calibration homography and keeps only
                           people whose feet land inside the court
                           (with a `court_margin_m` tolerance for
                           baseline overshoots and warm-up areas)
  - max N persons        — keeps the most confident, caps doubles at 4
                           (raise to 5–6 if you expect a coach in frame)

The on-court filter is only applied when a Calibration is provided; if
calib is None the detector falls back to "everyone counts", which is
useful for debugging without a calibrated video.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from ..court import reference as _ref
from ..court.homography import project_image_to_court as _proj_img_to_court


@dataclass
class PlayerDetectorConfig:
    weights: str = "weights/yolov8n.pt"      # auto-downloaded by ultralytics on first run
    device: str = "cpu"
    conf: float = 0.4
    iou: float = 0.5
    tracker: str = "bytetrack.yaml"          # built-in; "botsort.yaml" also valid
    min_bbox_h: int = 60                     # px; rejects very-far or tiny detections
    max_persons: int = 4                     # cap per-frame to N most-confident
    imgsz: int = 640
    # On-court filter: keep detection only if bbox-foot projects within
    # [-margin, COURT_X + margin] × [-margin, COURT_Y + margin] meters.
    # 3 m gives room for baseline follow-through and warm-up; lower for
    # stricter filtering if bystanders keep sneaking in.
    court_margin_m: float = 3.0


class PlayerDetector:
    """Streaming person detector.  `detect(frame, frame_idx)` returns
    `{track_id: (x0, y0, x1, y1)}` in original-frame pixel coordinates.

    Pass `calib=<Calibration>` to enable on-court filtering; pass
    `calib=None` (default) for unfiltered output.
    """

    def __init__(self, cfg: PlayerDetectorConfig, calib=None):
        try:
            from ultralytics import YOLO                        # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "PlayerDetector needs ultralytics: `pip install ultralytics`"
            ) from e
        self.cfg = cfg
        self.calib = calib
        self._model = YOLO(cfg.weights)

    def reset(self) -> None:
        # ultralytics internally keeps tracker state on the predictor; the
        # cleanest reset is to drop the predictor so the next .track() call
        # rebuilds it fresh.  Safe no-op if predictor hasn't been built yet.
        try:
            self._model.predictor = None                         # type: ignore[attr-defined]
        except Exception:
            pass

    def _on_court(self, bbox: Tuple[float, float, float, float]) -> bool:
        if self.calib is None:
            return True
        x0, _, x1, y1 = bbox
        foot_x, foot_y = 0.5 * (x0 + x1), y1                     # bottom-center
        try:
            rx, ry = _proj_img_to_court(self.calib.H_img_to_real,
                                         (foot_x, foot_y))
        except Exception:
            return True                                          # don't drop if homography is wonky
        m = self.cfg.court_margin_m
        return (-m <= rx <= _ref.COURT_WIDTH_M + m
                and -m <= ry <= _ref.COURT_LENGTH_M + m)

    def detect(self, frame: np.ndarray, frame_idx: int) -> Dict[int, Tuple[int, int, int, int]]:
        results = self._model.track(
            frame,
            persist=True,
            verbose=False,
            classes=[0],                                         # person only
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            tracker=self.cfg.tracker,
            device=self.cfg.device,
            imgsz=self.cfg.imgsz,
        )
        r = results[0]
        out: Dict[int, Tuple[int, int, int, int]] = {}
        if r.boxes is None or r.boxes.id is None:
            return out
        boxes = r.boxes.xyxy.cpu().numpy()                       # (N, 4)
        ids   = r.boxes.id.cpu().numpy().astype(int)             # (N,)
        confs = r.boxes.conf.cpu().numpy()                       # (N,)

        order = np.argsort(-confs)                               # high → low
        kept = 0
        for idx in order:
            if kept >= self.cfg.max_persons:
                break
            x0, y0, x1, y1 = boxes[idx]
            if (y1 - y0) < self.cfg.min_bbox_h:
                continue
            bbox = (int(x0), int(y0), int(x1), int(y1))
            if not self._on_court(bbox):
                continue
            out[int(ids[idx])] = bbox
            kept += 1
        return out
