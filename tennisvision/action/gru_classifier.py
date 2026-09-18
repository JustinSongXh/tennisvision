"""GRU-based action classifier with wrist-speed trigger.

Classifies tennis strokes from keypoint sequences:
  - backhand (0)
  - forehand (1)
  - serve (2)
  - background (3)

Architecture:
  Input: (batch, seq_len, 34) — seq_len frames × 17 keypoints × 2 coords
  GRU: 2 layers, hidden=128, dropout=0.3
  FC: 128 → 64 → 4 classes

Wrist-speed trigger:
  - Only runs GRU when wrist is actively moving (speed > threshold)
  - Serve events marked when serve_prob > serve_threshold
  - All intermediate results preserved for downstream fusion
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


LABELS = {0: "backhand", 1: "forehand", 2: "serve", 3: "background"}


@dataclass
class GRUClassifierConfig:
    weights: str = "weights/stroke_gru_v4_best.pt"
    seq_len: int = 30
    speed_threshold: float = 0.15     # wrist speed trigger
    serve_threshold: float = 0.8      # serve prob marking
    gap_tolerance: int = 5            # frames below threshold before deactivating
    wrist_indices: tuple = (9, 10)    # L_wrist, R_wrist in COCO 17


class StrokeGRU(nn.Module):
    """GRU network for stroke classification."""

    def __init__(self, input_dim: int = 34, hidden: int = 128,
                 n_layers: int = 2, n_classes: int = 4):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, n_layers,
                          batch_first=True, dropout=0.3)
        self.fc = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)
        return self.fc(h[-1])


class _SlotState:
    """Per-slot state for wrist trigger and keypoint buffer."""

    def __init__(self, maxlen: int):
        self.kp_buffer: list = []
        self.maxlen = maxlen
        self.prev_kp: Optional[np.ndarray] = None
        self.active: bool = False
        self.gap_count: int = 0

    def push_kp(self, kp: np.ndarray) -> None:
        self.kp_buffer.append(kp)
        if len(self.kp_buffer) > self.maxlen + 30:
            self.kp_buffer = self.kp_buffer[-(self.maxlen + 30):]

    def wrist_speed(self, kp: np.ndarray, wrist_indices: tuple) -> float:
        if self.prev_kp is None:
            self.prev_kp = kp
            return 0.0
        speed = 0.0
        for wi in wrist_indices:
            dx = kp[wi, 0] - self.prev_kp[wi, 0]
            dy = kp[wi, 1] - self.prev_kp[wi, 1]
            speed += np.sqrt(dx ** 2 + dy ** 2)
        self.prev_kp = kp
        return speed


@dataclass
class ActionEvent:
    """Single GRU classification output."""
    frame: int
    time: float
    slot: int
    slot_name: str
    label: str
    conf: float
    probs: Dict[str, float]
    wrist_speed: float
    is_serve: bool = False


class GRUActionClassifier:
    """Wrist-triggered GRU action classifier for 4 court slots.

    Usage:
        clf = GRUActionClassifier(cfg)
        for frame_data in video_frames:
            events = clf.push_frame(frame_idx, slot_detections)
    """

    def __init__(self, cfg: Optional[GRUClassifierConfig] = None):
        self.cfg = cfg or GRUClassifierConfig()
        self.model = StrokeGRU(n_classes=4)
        self.model.load_state_dict(
            torch.load(self.cfg.weights, map_location="cpu")
        )
        self.model.eval()
        self._slots: Dict[int, _SlotState] = {
            s: _SlotState(self.cfg.seq_len) for s in range(4)
        }

    def push_frame(
        self, frame_idx: int, fps: float, slot_kps: Dict[int, np.ndarray],
    ) -> List[ActionEvent]:
        """Process one frame's slot-assigned keypoints.

        Args:
            frame_idx: current frame index
            fps: video frame rate
            slot_kps: {slot_id: (17, 2) bbox-normalized keypoints}

        Returns:
            list of ActionEvent (may be empty)
        """
        cfg = self.cfg
        events = []

        for slot in range(4):
            ss = self._slots[slot]

            if slot not in slot_kps:
                if ss.active:
                    ss.gap_count += 1
                    if ss.gap_count > cfg.gap_tolerance:
                        ss.active = False
                continue

            kp = slot_kps[slot]
            ss.push_kp(kp)
            speed = ss.wrist_speed(kp, cfg.wrist_indices)

            if speed > cfg.speed_threshold:
                ss.active = True
                ss.gap_count = 0
            elif ss.active:
                ss.gap_count += 1
                if ss.gap_count > cfg.gap_tolerance:
                    ss.active = False

            # GRU inference during active period
            if ss.active and len(ss.kp_buffer) >= cfg.seq_len:
                feat = np.array(
                    [k[:, :2].reshape(-1) for k in ss.kp_buffer[-cfg.seq_len:]],
                    dtype=np.float32,
                )
                with torch.no_grad():
                    logits = self.model(torch.FloatTensor(feat).unsqueeze(0))
                    probs = torch.softmax(logits, dim=1)[0].numpy()

                pred = int(np.argmax(probs))
                conf = float(probs[pred])
                is_serve = float(probs[2]) > cfg.serve_threshold

                from .slot_mapper import SLOT_NAMES
                events.append(ActionEvent(
                    frame=frame_idx,
                    time=round(frame_idx / fps, 2),
                    slot=slot,
                    slot_name=SLOT_NAMES[slot],
                    label=LABELS[pred],
                    conf=round(conf, 3),
                    probs={LABELS[i]: round(float(probs[i]), 3) for i in range(4)},
                    wrist_speed=round(speed, 3),
                    is_serve=is_serve,
                ))

        return events

    def reset(self) -> None:
        for ss in self._slots.values():
            ss.kp_buffer.clear()
            ss.prev_kp = None
            ss.active = False
            ss.gap_count = 0
