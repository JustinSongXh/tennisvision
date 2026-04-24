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
        # Distance / margin thresholds accept either a pixel value OR a
        # ratio (of the frame diagonal).  Ratio takes precedence when
        # set.  Ratios make the config portable across 720p / 1080p / 4K.
        "max_disp": 300.0,
        "max_disp_ratio": None,
        "two_stage": True,             # run a second WASB pass on a far-court crop
                                       # (needs calib); off → single full-frame pass
        "two_stage_dedup_px": 60.0,    # merge main + far candidates within this distance
        "two_stage_dedup_ratio": None,
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
        # All pixel thresholds accept an equivalent *_ratio (of frame
        # diagonal) that overrides the pixel value at runtime.  Speed
        # ratios are interpreted as "fraction of diagonal per frame".
        "gate_px": 120,
        "gate_ratio": None,
        "max_gap_frames": 8,
        "min_len": 3,
        "min_speed": 5.0,
        "min_speed_ratio": None,
        "max_speed": 200.0,
        "max_speed_ratio": None,
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
        # Lob handling: when the ball was moving UP, or was already in
        # the far-side airspace without clearly descending, at the
        # moment detection was lost, stretch `online_silence_seconds`
        # by this factor.  Covers 2-3s of no-detection while a lob is
        # airborne / occluded behind the net.  1.0 = disabled.
        "online_silence_lob_multiplier": 2.0,
        "online_silence_lob_up_speed_px": 2.0,   # min |vy| (image px/frame) that
                                                 # counts as "rising"; below is flat
        # Serve-toss based rally boundary (soft signal).  A confirmed
        # toss pattern (hold→rise→apex→descent, near a baseline, mostly
        # vertical) closes the current burst and starts a fresh rally
        # at the toss frame — but ONLY if the burst has been quiet for
        # `serve_soft_quiet_seconds` since its last net crossing.  That
        # gate avoids the false mid-rally splits that made the earlier
        # force-close implementation (reverted 6eb789f) drop real
        # rallies.  Pure ball signal so it stays reliable when pose /
        # player detection misses the server.  Default off; flip on in
        # your override yaml when between-points stretches keep
        # merging two real points into one burst.
        "serve_enabled": False,
        # Rise threshold adapts per-toss between near and far values by
        # interpolating on the toss origin's image-y between the near-
        # and far-baseline image projections — perspective compresses
        # far-side toss image extent ~4x vs. near-side, so a single
        # global ratio either misses far tosses (set too large) or
        # false-fires on near-side rise noise (set too small).
        "serve_toss_rise_ratio_near": 0.060,     # of frame diag (~132px @ 1080p)
        "serve_toss_rise_ratio_far":  0.015,     # of frame diag (~33px  @ 1080p)
        "serve_toss_min_frames": 6,              # ~0.2s at 30fps
        "serve_toss_max_horiz_ratio": 0.6,       # |dx|/rise cap — filters lobs
        "serve_baseline_margin_m": 4.0,          # court-y distance from baseline
                                                 # that toss origin must sit within
        "serve_pre_static_max_dy_px": 12.0,      # max |dy/df| (px/frame) for the
                                                 # pre-toss "hold" frames; WASB
                                                 # per-frame jitter sits around
                                                 # 7-8px so the gate needs headroom
        "serve_suppress_seconds": 2.0,           # re-detect cooldown per serve
        "serve_history_seconds": 2.0,            # ring buffer depth for toss pattern
        "serve_soft_quiet_seconds": 1.0,         # min time since last net crossing
                                                 # for a toss to be trusted; 0 off
                                                 # the soft gate (hard force-close)
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
        # Post-Pass-1b pose validation: drop rally candidates that have
        # too few non-neutral stroke events (forehand / backhand / serve).
        # Real rallies always show ≥2 strokes; warm-up / pickup / dribble
        # keeps the ball in play but produces nothing.  Works together
        # with the trajectory-based filters (crossings + density) as a
        # second sanity gate.  0 disables.
        "post_filter_min_strokes": 2,
        # Bounce-split validation: require the rally's bounces to span
        # both court halves.  Bounces sit on z=0 (the ground plane) so
        # their homography projection to court-y is accurate, unlike
        # airborne ball trajectory which can't tell a high pre-serve
        # toss apart from a real over-the-net shot.  Rally 0 of
        # sample_short — the server dribbling before serve — got
        # misclassified as a rally precisely because of this.
        "require_bounces_both_halves": True,
        # Rally-merge pass: if two adjacent rallies sit less than
        # `merge_gap_seconds` apart AND no bounce falls inside the gap,
        # merge them — a mid-rally tracking drop-out looks identical to
        # a between-point silence at the detector level, but real
        # between-point gaps contain pickup/dribble bounces whereas a
        # ball-airborne-out-of-frame gap doesn't.  0 disables.
        "merge_gap_seconds": 5.0,
        # Online mode: ball detection, tracker, rally detector AND
        # pose + stroke classifier all run together in a single pass,
        # so per-frame results are available live (for real-time
        # scoreboards, streaming overlays, etc.).  Rally boundaries in
        # this mode come entirely from OnlineRallyDetector — bounce-
        # based filters (require_bounces_both_halves, merge_gap_seconds)
        # are skipped because bounce detection is a batch post-pass
        # that can't run per-frame.  Default False keeps the original
        # two-pass offline pipeline (better rally quality, slower).
        "online": False,
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
