"""GRU-based stroke classifier.

Classifies tennis strokes from keypoint sequences:
  - backhand (0)
  - forehand (1)
  - serve (2)
  - background (3)

Architecture:
  Input: (batch, seq_len, 34) — seq_len frames x 17 keypoints x 2 coords
  GRU: 2 layers, hidden=128, dropout=0.3
  FC: 128 -> 64 -> n_classes
"""

from __future__ import annotations

import torch
import torch.nn as nn


LABELS = {0: "backhand", 1: "forehand", 2: "serve", 3: "background"}


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
