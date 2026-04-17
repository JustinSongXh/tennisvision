"""Configuration loading and defaults."""

from __future__ import annotations

import copy
import os
from typing import Any, Optional

try:
    import yaml
except ImportError:
    yaml = None


DEFAULTS: dict = {
    "court": {
        "detector": "mark",
        "weights": "weights/court_tcd.pt",
        "input_size": [640, 360],
    },
    "ball": {
        "detector": "wasb",
        "weights": "weights/wasb_tennis_best.pth.tar",
        "device": "cpu",
        "runtime": "auto",
        "onnx_path": None,
        "score_threshold": 0.5,
        "max_disp": 300.0,
        "hsv_low":  [25, 60, 120],
        "hsv_high": [50, 255, 255],
        "min_area": 3,
        "max_area": 400,
        "min_circularity": 0.55,
        "mog_var_threshold": 25,
        "player_min_area": 1500,
        "player_max_area": 60000,
        "player_max_w_frac": 0.55,
        "player_max_h_frac": 0.75,
    },
    "tracker": {
        "gate_px": 120,
        "max_gap_frames": 8,
        "min_len": 3,
        "min_speed": 5.0,
        "max_speed": 200.0,
        "render_tail": 45,
    },
    "bounce": {
        "detector": "catboost",
        "weights": "weights/bounce_catboost.cbm",
        "lookback": 3,
        "min_dy": 4,
        "cooldown": 10,
        "catboost_threshold": 0.20,
        "catboost_nms_window": 1,
        "min_track_len": 8,
    },
    "render": {
        "minimap_width_px": 150,
        "minimap_margin_px": 15,
        "bounce_fade_frames": 600,
    },
    "debug": {
        "dir": None,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: Optional[str] = None) -> dict:
    if path is None:
        return copy.deepcopy(DEFAULTS)
    if yaml is None:
        raise RuntimeError("PyYAML is required to read config files")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path) as f:
        user = yaml.safe_load(f) or {}
    return _deep_merge(DEFAULTS, user)
