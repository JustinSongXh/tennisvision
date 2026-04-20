"""Sliding-window GRU stroke classifier (single- and multi-player).

Ports inference from antoinekeller/tennis_shot_recognition (upstream repo
has no license file; verify before redistributing weights).

Input per frame:   Pose (17 COCO keypoints with scores)
Window:            last N frames (default 30 ≈ 1 s @ 30 fps)
Feature per frame: 13 keypoints (drop eyes+ears) × (y_norm, x_norm) = 26-d
Model:             Keras GRU over (N, 26) → softmax over 4 classes
Classes (upstream order): 0=backhand, 1=forehand, 2=neutral, 3=serve

Two entry points:
  - `StrokeClassifier`               — single full-frame player (tests / single-player clips)
  - `MultiPlayerStrokeRecognizer`    — YOLOv8-tracked persons, one sliding window per track ID.
                                        Handles singles (2) and doubles (4) identically,
                                        and does not hard-code the player count.

Feature normalization: the upstream classifier was trained on keypoints
**relative to the player's bbox** — MoveNet after an ROI tracker that
shrinks around the subject.  For multi-player we reproduce that by
normalizing each keypoint (y, x) to the YOLOv8 bbox dims, not the full
frame — otherwise the feature would encode "where on the court the
player is" and the classifier would see noise.  The single-player
fallback keeps frame-size normalization (the ROI equals the frame).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from .pose import Pose


# MoveNet keypoints 1..4 are left_eye / right_eye / left_ear / right_ear.
# Upstream drops those four; keep the other 13 in original order.
_KEEP_IDX = np.array([0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], dtype=np.int32)


@dataclass
class StrokeClassifierConfig:
    weights: str = "weights/tennis_rnn.h5"
    window_frames: int = 30
    labels: Tuple[str, ...] = ("backhand", "forehand", "neutral", "serve")
    min_confidence: float = 0.9        # argmax prob below this → no event emitted
    stride: int = 5                    # run inference at most every `stride` frames per track
    emit_neutral: bool = False         # if False, suppress neutral-class events
    normalize_by_frame_size: bool = True  # divide (y, x) by (H, W) before buffering


@dataclass
class StrokeEvent:
    frame: int
    label: str
    confidence: float
    cx: float = 0.0
    cy: float = 0.0
    player_id: Optional[int] = None    # None = single-player / full-frame mode


def _pose_to_feature(
    p: Pose,
    norm_h: int,
    norm_w: int,
    origin_y: float = 0.0,
    origin_x: float = 0.0,
    normalize: bool = True,
) -> np.ndarray:
    """(17,3) pose → (26,) feature vector in (y, x) order.

    Coordinates in `p.keypoints` are absolute frame pixels.  Subtract
    `(origin_y, origin_x)` to make them relative to the ROI / bbox,
    then divide by `(norm_h, norm_w)` so the feature lies in [0, 1]
    within that ROI — matching how the upstream classifier was trained.
    For the full-frame single-player path pass `origin=0, norm=(H, W)`.
    """
    kp = p.keypoints[_KEEP_IDX]                 # (13, 3)
    yx = kp[:, :2].copy()                       # (13, 2)  → (y, x)
    if normalize:
        yx[:, 0] = (yx[:, 0] - origin_y) / max(norm_h, 1)
        yx[:, 1] = (yx[:, 1] - origin_x) / max(norm_w, 1)
    return yx.reshape(-1).astype(np.float32)    # (26,)


def _load_keras_model(weights: str):
    """Load a Keras `.h5` checkpoint.

    The upstream `tennis_rnn.h5` was saved with Keras 2 and uses the
    `time_major` kwarg on GRU, which Keras 3 (TF>=2.16) rejects.  Try
    `tf_keras` (Keras-2 compat package) first; fall back to Keras 3
    only if tf_keras isn't installed.
    """
    try:
        import tf_keras as keras   # type: ignore  # Keras 2 API
    except ImportError:
        try:
            from tensorflow import keras  # type: ignore  # Keras 3
        except ImportError as e:
            raise RuntimeError(
                "StrokeClassifier requires tensorflow + tf-keras: "
                "`pip install tensorflow tf-keras`"
            ) from e
    return keras.models.load_model(weights, compile=False)


# ----------------------------------------------------------------------
# Single-player (full-frame) — kept for tests and single-player clips.
# ----------------------------------------------------------------------

class StrokeClassifier:
    def __init__(self, cfg: StrokeClassifierConfig, pose_extractor):
        self.cfg = cfg
        self.pose = pose_extractor
        self._model = _load_keras_model(cfg.weights)
        self._window: deque = deque(maxlen=cfg.window_frames)
        self._last_infer_frame: int = -10**9

    def reset(self) -> None:
        self._window.clear()
        self._last_infer_frame = -10**9

    def push_frame(self, frame: np.ndarray, frame_idx: int, roi=None) -> list:
        """Returns a list of StrokeEvent (0 or 1 element) for API parity
        with MultiPlayerStrokeRecognizer."""
        H, W = frame.shape[:2]
        pose = self.pose.extract(frame, frame_idx, roi=roi)
        if roi is not None:
            x0, y0, x1, y1 = roi
            feat = _pose_to_feature(
                pose, norm_h=(y1 - y0), norm_w=(x1 - x0),
                origin_y=y0, origin_x=x0,
                normalize=self.cfg.normalize_by_frame_size,
            )
        else:
            feat = _pose_to_feature(
                pose, norm_h=H, norm_w=W,
                normalize=self.cfg.normalize_by_frame_size,
            )
        self._window.append((pose, feat))

        if len(self._window) < self.cfg.window_frames:
            return []
        if frame_idx - self._last_infer_frame < self.cfg.stride:
            return []
        self._last_infer_frame = frame_idx

        feats = np.stack([f for (_, f) in self._window], axis=0)     # (N, 26)
        probs = self._model.predict(feats[None, ...], verbose=0)[0]  # (4,)
        k = int(np.argmax(probs))
        conf = float(probs[k])
        label = self.cfg.labels[k]
        if conf < self.cfg.min_confidence:
            return []
        if label == "neutral" and not self.cfg.emit_neutral:
            return []

        last_pose = self._window[-1][0]
        vis = last_pose.visible_mask(self.pose.cfg.score_threshold)
        if vis.any():
            cy = float(last_pose.keypoints[vis, 0].mean())
            cx = float(last_pose.keypoints[vis, 1].mean())
        else:
            cx = cy = 0.0
        return [StrokeEvent(frame=frame_idx, label=label, confidence=conf,
                            cx=cx, cy=cy)]


# ----------------------------------------------------------------------
# Multi-player — one window per YOLO track ID.
# ----------------------------------------------------------------------

@dataclass
class _PlayerState:
    window: deque
    last_infer_frame: int = -10**9


class MultiPlayerStrokeRecognizer:
    """Runs per-player pose + classification.

    The player count is whatever YOLOv8 reports on each frame, filtered
    through `PlayerDetector` (bbox-size and conf thresholds).  Singles →
    typically 2 tracks; doubles → 4; warm-up rallies with a coach in
    frame → 3 — all handled the same way.

    Track state is garbage-collected when a track disappears for longer
    than `track_ttl_frames` frames.  That prevents a momentary miss (e.g.
    occlusion by the net) from resetting a player's sliding window.
    """

    def __init__(
        self,
        cfg: StrokeClassifierConfig,
        pose_extractor,
        player_detector,
        track_ttl_frames: int = 30,
    ):
        self.cfg = cfg
        self.pose = pose_extractor
        self.player_det = player_detector
        self.track_ttl_frames = track_ttl_frames
        self._model = _load_keras_model(cfg.weights)
        self._states: dict = {}      # track_id -> _PlayerState
        self._last_seen: dict = {}   # track_id -> frame_idx

    def reset(self) -> None:
        self._states.clear()
        self._last_seen.clear()
        self.player_det.reset()

    def push_frame(self, frame: np.ndarray, frame_idx: int) -> list:
        H, W = frame.shape[:2]
        detections = self.player_det.detect(frame, frame_idx)  # {tid: (x0,y0,x1,y1)}

        # GC tracks that have been absent for a while.
        for tid in list(self._states.keys()):
            if frame_idx - self._last_seen.get(tid, frame_idx) > self.track_ttl_frames:
                self._states.pop(tid, None)
                self._last_seen.pop(tid, None)

        events: list = []
        for tid, bbox in detections.items():
            self._last_seen[tid] = frame_idx
            st = self._states.get(tid)
            if st is None:
                st = _PlayerState(window=deque(maxlen=self.cfg.window_frames))
                self._states[tid] = st

            pose = self.pose.extract(frame, frame_idx, roi=bbox)
            bx0, by0, bx1, by1 = bbox
            feat = _pose_to_feature(
                pose, norm_h=(by1 - by0), norm_w=(bx1 - bx0),
                origin_y=by0, origin_x=bx0,
                normalize=self.cfg.normalize_by_frame_size,
            )
            st.window.append((pose, feat))

            if len(st.window) < self.cfg.window_frames:
                continue
            if frame_idx - st.last_infer_frame < self.cfg.stride:
                continue
            st.last_infer_frame = frame_idx

            feats = np.stack([f for (_, f) in st.window], axis=0)
            probs = self._model.predict(feats[None, ...], verbose=0)[0]
            k = int(np.argmax(probs))
            conf = float(probs[k])
            label = self.cfg.labels[k]
            if conf < self.cfg.min_confidence:
                continue
            if label == "neutral" and not self.cfg.emit_neutral:
                continue

            cx = 0.5 * (bbox[0] + bbox[2])
            cy = 0.5 * (bbox[1] + bbox[3])
            events.append(StrokeEvent(frame=frame_idx, label=label, confidence=conf,
                                      cx=float(cx), cy=float(cy), player_id=int(tid)))
        return events
