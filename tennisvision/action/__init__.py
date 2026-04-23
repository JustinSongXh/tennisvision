"""Player action / stroke recognition.

Three stages:
  1. Detection + tracking + pose — YOLOPoseTracker (YOLO26n-pose + ByteTrack)
     gives per-frame `{track_id: (bbox, Pose)}` in a single model call.
  2. Classifier — GRU over a 30-frame sliding window of (y, x) keypoints,
     emitting {backhand, forehand, neutral, serve} per tracked player.

Pretrained weights:
  - yolo26n-pose.pt     ultralytics (player detection + pose)
  - tennis_rnn.h5       antoinekeller/tennis_shot_recognition (stroke GRU)

Heavy deps (`tensorflow`, `ultralytics`) are imported lazily inside the
wrappers so this package stays importable without them.
"""

from .pose import YOLOPoseTracker, YOLOPoseTrackerConfig, Pose
from .stroke_classifier import (
    MultiPlayerStrokeRecognizer,
    StrokeClassifier,
    StrokeClassifierConfig,
    StrokeEvent,
)

__all__ = [
    "MultiPlayerStrokeRecognizer",
    "Pose",
    "StrokeClassifier",
    "StrokeClassifierConfig",
    "StrokeEvent",
    "YOLOPoseTracker",
    "YOLOPoseTrackerConfig",
]
