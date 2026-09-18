"""Player action / stroke recognition.

Two pipelines:

1. **Legacy (single-pass)**: YOLOPoseTracker + RNN stroke classifier
2. **Two-stage + GRU v4 (validated)**: yolo11m detect → 4-slot mapping →
   yolo26s-pose keypoints → wrist trigger → GRU v4 → serve marking
   See docs/experiment_log.md for details.

Pretrained weights:
  - yolo26s-pose.pt           ultralytics (keypoint extraction)
  - stroke_gru_v4_best.pt     THETIS smart-sliced (4-class GRU)
  - action_mlp_bbox.pkl       Roboflow (single-frame MLP, backup)

Heavy deps imported lazily.
"""

from .pose import YOLOPoseTracker, YOLOPoseTrackerConfig, Pose
from .slot_mapper import SlotMapper, SlotMapperConfig, SLOT_NAMES
from .gru_classifier import (
    GRUActionClassifier,
    GRUClassifierConfig,
    StrokeGRU,
    ActionEvent,
    LABELS,
)
from .stroke_classifier import (
    MultiPlayerStrokeRecognizer,
    StrokeClassifier,
    StrokeClassifierConfig,
    StrokeEvent,
)

__all__ = [
    # New pipeline
    "GRUActionClassifier",
    "GRUClassifierConfig",
    "StrokeGRU",
    "ActionEvent",
    "SlotMapper",
    "SlotMapperConfig",
    "SLOT_NAMES",
    "LABELS",
    # Legacy
    "MultiPlayerStrokeRecognizer",
    "Pose",
    "StrokeClassifier",
    "StrokeClassifierConfig",
    "StrokeEvent",
    "YOLOPoseTracker",
    "YOLOPoseTrackerConfig",
]
