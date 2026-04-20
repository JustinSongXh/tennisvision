"""Player action / stroke recognition.

Three stages:
  1. Person detection — YOLOv8 + ByteTrack gives per-frame
     `{track_id: bbox}`.  Player count is not hard-coded — singles
     (2 tracks) and doubles (4 tracks) work the same way.
  2. Pose — per-person 2D keypoints via MoveNet SinglePose (TFLite),
     run on each player bbox crop.
  3. Classifier — GRU over a 30-frame sliding window of (y, x)
     keypoints, emitting {backhand, forehand, neutral, serve}
     per tracked player.

Pretrained weights (no training needed):
  - YOLOv8n                     auto-downloaded by ultralytics
  - MoveNet lightning fp16      TF Hub (tflite file, ~3 MB)
  - tennis_rnn.h5               antoinekeller/tennis_shot_recognition

Heavy deps (`tensorflow`, `ultralytics`) are imported lazily inside the
wrappers so this package stays importable on machines that only need the
ball / court pipeline.
"""

from .player import PlayerDetector, PlayerDetectorConfig
from .pose import MoveNetConfig, MoveNetPoseExtractor, Pose
from .stroke_classifier import (
    MultiPlayerStrokeRecognizer,
    StrokeClassifier,
    StrokeClassifierConfig,
    StrokeEvent,
)

__all__ = [
    "MoveNetConfig",
    "MoveNetPoseExtractor",
    "MultiPlayerStrokeRecognizer",
    "PlayerDetector",
    "PlayerDetectorConfig",
    "Pose",
    "StrokeClassifier",
    "StrokeClassifierConfig",
    "StrokeEvent",
]
