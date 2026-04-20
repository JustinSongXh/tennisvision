"""MoveNet SinglePose Lightning wrapper (TFLite fp16).

Weights: download once with
    wget -O weights/movenet_lightning_f16.tflite \
        'https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4?lite-format=tflite'

Preprocessing matches `tf.image.resize_with_pad` (aspect-preserving
letterbox centered on a black canvas, uint8).  Outputs are 17 COCO
keypoints with confidence, back-projected to original-frame pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

# TFLite interpreter — prefer the lightweight `tflite_runtime` package;
# fall back to the one bundled with full `tensorflow`.
_Interpreter = None
try:
    from tflite_runtime.interpreter import Interpreter as _Interpreter  # type: ignore
except ImportError:
    try:
        from tensorflow.lite.python.interpreter import Interpreter as _Interpreter  # type: ignore
    except ImportError:
        pass


# COCO-17 keypoint order that MoveNet outputs.
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
class MoveNetConfig:
    weights: str = "weights/movenet_lightning_f16.tflite"
    input_size: int = 192
    score_threshold: float = 0.2      # per-keypoint score below this → treated as missing


@dataclass
class Pose:
    """One frame's pose: (17, 3) array of (y_px, x_px, score) in ORIGINAL image coords."""
    keypoints: np.ndarray   # shape (17, 3), dtype float32
    frame_idx: int

    def visible_mask(self, thr: float) -> np.ndarray:
        return self.keypoints[:, 2] >= thr


def _letterbox(frame: np.ndarray, size: int) -> Tuple[np.ndarray, float, int, int]:
    """Resize preserving aspect ratio, pad with zeros, centered.

    Returns (padded uint8 HxWx3, scale, pad_top, pad_left).
    """
    H, W = frame.shape[:2]
    scale = size / max(H, W)
    nh, nw = int(round(H * scale)), int(round(W * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (size - nh) // 2
    left = (size - nw) // 2
    padded = np.zeros((size, size, 3), dtype=np.uint8)
    padded[top:top + nh, left:left + nw] = resized
    return padded, float(scale), top, left


class MoveNetPoseExtractor:
    """Stateless per-frame MoveNet wrapper.

    Call `extract(frame, roi=None)` to get a Pose.  `roi` (optional) is
    an (x0, y0, x1, y1) pixel box — when present, MoveNet runs on that
    crop and keypoints are back-projected into the full frame.  This is
    the hook for per-player tracking in multi-player scenes; leaving
    it None means "run on whole frame" (good only for single-player
    clips until a separate player detector is wired in).
    """

    def __init__(self, cfg: MoveNetConfig):
        if _Interpreter is None:
            raise RuntimeError(
                "MoveNet needs TFLite runtime: `pip install tflite-runtime` "
                "(or `tensorflow`).")
        self.cfg = cfg
        self._interp = _Interpreter(model_path=cfg.weights)
        self._interp.allocate_tensors()
        self._in = self._interp.get_input_details()[0]
        self._out = self._interp.get_output_details()[0]

    def extract(
        self,
        frame: np.ndarray,
        frame_idx: int,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> Pose:
        if roi is not None:
            x0, y0, x1, y1 = roi
            x0 = max(0, x0); y0 = max(0, y0)
            x1 = min(frame.shape[1], x1); y1 = min(frame.shape[0], y1)
            crop = frame[y0:y1, x0:x1]
            origin_x, origin_y = x0, y0
            crop_h, crop_w = (y1 - y0), (x1 - x0)
        else:
            crop = frame
            origin_x = origin_y = 0
            crop_h, crop_w = frame.shape[:2]
        if crop_h <= 0 or crop_w <= 0:
            return Pose(keypoints=np.zeros((17, 3), dtype=np.float32),
                        frame_idx=frame_idx)

        padded, scale, pad_top, pad_left = _letterbox(crop, self.cfg.input_size)
        # MoveNet expects RGB uint8
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        self._interp.set_tensor(self._in["index"], rgb[None, ...])
        self._interp.invoke()
        # Output shape: (1, 1, 17, 3) — (y, x, score) all normalized to [0,1]
        out = self._interp.get_tensor(self._out["index"])[0, 0]

        # Un-letterbox: out coords are fractions of input_size.
        in_s = float(self.cfg.input_size)
        ys_in = out[:, 0] * in_s
        xs_in = out[:, 1] * in_s
        ys_crop = (ys_in - pad_top) / scale
        xs_crop = (xs_in - pad_left) / scale
        kp = np.stack([ys_crop + origin_y, xs_crop + origin_x, out[:, 2]], axis=1)
        return Pose(keypoints=kp.astype(np.float32), frame_idx=frame_idx)
