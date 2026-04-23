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
        "two_stage": True,             # run a second WASB pass on a far-court crop
                                       # (needs calib); off → single full-frame pass
        "two_stage_dedup_px": 60.0,    # merge main + far candidates within this distance
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
    "inpainter": {
        # TrackNetV3 InpaintNet: learned gap-filler run on the union of
        # validated tracks before bounce detection.  Default off —
        # shuttlecock-trained, empirical on tennis; A/B vs baseline on
        # your video before trusting.
        "enabled": False,
        "weights": "weights/InpaintNet_best.pt",
        "device": "cpu",
        # "single" — one forward pass over the whole series (fully-conv,
        # no seam artifacts).  "nonoverlap" — non-overlap windows of
        # seq_len (faithful to upstream eval).
        "window_mode": "single",
        "seq_len": None,                     # None = use ckpt's stored value (typ. 16)
        "max_gap_frames": 60,                # skip gaps longer than this
        "th_h_frac": 0.05,                   # out-of-view y-threshold as fraction of H
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
    "action": {
        "enabled": False,
        "pose_weights": "weights/yolo26n-pose.pt",
        "pose_device": "cpu",        # "cpu" | "cuda" | "cuda:0"
        "rnn_weights": "weights/tennis_rnn.h5",
        "window_frames": 30,
        "labels": ["backhand", "forehand", "neutral", "serve"],
        "min_confidence": 0.9,
        "stride": 5,
        "emit_neutral": False,
        "ball_proximity_px": 300,      # skip RNN when ball has NOT been within N px of player
                                       # in ball_proximity_window_frames recent frames; 0=off
        "ball_proximity_window_frames": 15,  # lookback frames for proximity check (~0.5s at 30fps)
        "score_threshold": 0.2,
        # Player filtering params (detection weights are now pose_weights above).
        "player": {
            "conf": 0.3,
            "iou": 0.5,
            "tracker": "bytetrack.yaml",
            "imgsz": 640,
            "min_bbox_h": 60,
            "max_persons": 4,             # 2 singles / 4 doubles / coach → clamp high
            "track_ttl_frames": 30,
            # On-court filter: keep persons whose foot (bbox bottom-center)
            # projects inside [-margin, court+margin] meters via calib
            # homography.  Lower to e.g. 1.0 if bystanders keep slipping
            # in; raise if you see real players being rejected.
            "court_margin_m": 3.0,
        },
    },
    "debug": {
        "dir": None,
    },
    "rally": {
        # Online rally detection runs inside Pass 1a (ball-only scan).
        # A rally is an activity burst of ball detections bounded by
        # `online_silence_seconds` of ball-free frames on both sides,
        # containing at least `online_min_net_crossings` transitions of
        # the ball across the net (net_y is the midpoint projection of
        # NET_Y via H_real_to_img — approximate but sufficient for a
        # binary "is this a real rally" decision over many frames).
        #
        # After detection, Pass 1b runs pose + stroke classifier ONLY
        # on the frames inside each rally.
        #
        # All temporal knobs are in SECONDS; multiplied by fps at runtime.
        "enabled": True,
        "online_silence_seconds": 3.0,     # ball-free gap that ends an activity burst
        "online_crossing_silence_seconds": 4.0,  # force-close when ball is still in play
                                                 # but no net crossing for this long
                                                 # (catches pickup / dribble / toss gaps
                                                 # between real rallies); 0 = disabled
        "online_min_activity_density": 0.30,  # fraction of the burst's frames that must
                                              # have a ball detection to count as a rally.
                                              # Real rallies run ~0.5-0.8; spotty warm-up
                                              # bursts are often < 0.2; 0 = disabled.
        "online_min_net_crossings": 3,     # crossings needed to confirm a rally
        "pre_roll_seconds": 1.0,           # lead-in before first ball motion
        "post_roll_seconds": 1.0,          # kept after last ball motion
        "separator_seconds": 1.0,          # "Rally N" title between cut clips
        # Output paths: None → derive from --out by replacing `.mp4`
        # with `_rally.mp4` / `_rallies.json`.
        "clip_path": None,
        "json_path": None,
        # Stricter filter applied ONLY to the cut highlight video; the
        # full _rallies.json still lists every rally Pass 1a detected.
        # Both thresholds at 0 → include every detected rally in the cut.
        "clip_min_net_crossings": 3,
        "clip_min_duration_seconds": 2.0,
        # Parallel Pass-1b: kick off a pose worker thread as soon as a
        # rally is confirmed in Pass 1a, so ball detection (main thread,
        # WASB on CoreML/CPU) overlaps with pose + RNN inference.
        # Requires enough CPU cores to avoid contention (8+ recommended).
        # When enabled, pose_torch_threads caps the worker's torch pool
        # so the main thread keeps its fair share of cores.
        "parallel_pose": False,
        "pose_torch_threads": 4,
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
