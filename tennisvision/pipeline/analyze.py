"""End-to-end analysis pipeline.

Two-pass by default:
  Pass 1 — headless ball detection + tracking; cache per-frame state
           and the full point history of every validated track that
           eventually retires.
  Post   — run the configured bounce detector on each track; union and
           project bounces into court coordinates via homography.
  Pass 2 — re-read the video, re-use Pass-1 state to render trails and
           the accumulated bounces on the mini-map.

For the simple y-peak detector single-pass would also work, but the
two-pass design lets CatBoost (which needs look-ahead features) plug
in with zero changes to the render code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from ..ball.classical import HSVMotionDetector
from ..ball.tracker import MultiTrackManager, Track, TrackerConfig, TrackPoint

try:
    from ..ball.wasb import WASBBallDetector, WASBConfig
    _HAS_WASB = True
except ImportError:
    _HAS_WASB = False
from ..bounce.peak import BounceEvent, PeakBounceConfig, detect_all as peak_detect_all
from ..court import reference as ref
from ..court.calibration import Calibration
from ..court.homography import project_image_to_court
from ..render.hud import draw_hud
from ..render.minimap import Minimap, MinimapConfig
from ..render.trail import draw_trail


@dataclass
class _FrameState:
    n_cands: int = 0
    n_tracks: int = 0
    champion_id: Optional[int] = None
    champion_trail: list = field(default_factory=list)   # list of (x, y, frame)
    is_current_det: bool = False


@dataclass
class AnalyzeResult:
    total_frames: int
    bounces: int
    validated_tracks: int
    stroke_events: list = field(default_factory=list)


def _build_stroke_recognizer(acfg, calib=None):
    """Return an object with `push_frame(frame, frame_idx) -> list[StrokeEvent]`,
    or None if action.enabled is false.

    When `action.player.enabled` (default) → MultiPlayerStrokeRecognizer,
    singles and doubles handled identically by YOLOv8 tracking.
    When false → single-player full-frame fallback (debug / single-player clips).

    All heavy deps (tensorflow, ultralytics) are imported lazily so
    ball-only runs don't pay for them.
    """
    if not acfg or not acfg.get("enabled"):
        return None
    from ..action.pose import MoveNetConfig, MoveNetPoseExtractor
    from ..action.stroke_classifier import (
        MultiPlayerStrokeRecognizer,
        StrokeClassifier,
        StrokeClassifierConfig,
    )

    pose = MoveNetPoseExtractor(MoveNetConfig(
        weights=acfg["movenet_tflite"],
        score_threshold=acfg.get("score_threshold", 0.2),
    ))
    cls_cfg = StrokeClassifierConfig(
        weights=acfg["rnn_weights"],
        window_frames=acfg["window_frames"],
        labels=tuple(acfg["labels"]),
        min_confidence=acfg["min_confidence"],
        stride=acfg.get("stride", 5),
        emit_neutral=acfg.get("emit_neutral", False),
    )

    pcfg = acfg.get("player", {})
    if pcfg.get("enabled", True):
        from ..action.player import PlayerDetector, PlayerDetectorConfig
        det = PlayerDetector(
            PlayerDetectorConfig(
                weights=pcfg["weights"],
                device=pcfg.get("device", "cpu"),
                conf=pcfg.get("conf", 0.4),
                iou=pcfg.get("iou", 0.5),
                tracker=pcfg.get("tracker", "bytetrack.yaml"),
                imgsz=pcfg.get("imgsz", 640),
                min_bbox_h=pcfg.get("min_bbox_h", 60),
                max_persons=pcfg.get("max_persons", 4),
                court_margin_m=pcfg.get("court_margin_m", 3.0),
            ),
            calib=calib,
        )
        return MultiPlayerStrokeRecognizer(
            cls_cfg, pose, det,
            track_ttl_frames=pcfg.get("track_ttl_frames", 30),
        )
    return StrokeClassifier(cls_cfg, pose)


def _build_ball_detector(bcfg):
    """Return (detector_obj, detect_fn). `detect_fn(frame) -> list[(x, y)]`."""
    name = bcfg["detector"]
    if name == "classical":
        det = HSVMotionDetector(
            hsv_low=bcfg["hsv_low"], hsv_high=bcfg["hsv_high"],
            min_area=bcfg["min_area"], max_area=bcfg["max_area"],
            min_circularity=bcfg["min_circularity"],
            mog_var_threshold=bcfg["mog_var_threshold"],
            player_min_area=bcfg["player_min_area"],
            player_max_area=bcfg["player_max_area"],
            player_max_w_frac=bcfg["player_max_w_frac"],
            player_max_h_frac=bcfg["player_max_h_frac"],
        )
        return det, lambda f: [(c.x, c.y) for c in det.detect(f)]
    if name == "wasb":
        if not _HAS_WASB:
            raise RuntimeError("WASB requires torch; `pip install torch`")
        det = WASBBallDetector(WASBConfig(
            weights=bcfg["weights"],
            device=bcfg.get("device", "cpu"),
            runtime=bcfg.get("runtime", "auto"),
            onnx_path=bcfg.get("onnx_path", None),
            score_threshold=bcfg.get("score_threshold", 0.5),
            max_disp=bcfg.get("max_disp", 300.0),
        ))
        def _run(frame):
            det.push_frame(frame)
            pt = det.detect()
            return [pt] if pt is not None else []
        return det, _run
    raise NotImplementedError("ball.detector=%r not supported" % name)


def _run_bounce_detector(tracks: list, cfg: dict) -> list:
    """Run the configured bounce detector on each validated track,
    return a merged list of BounceEvents (image coords), sorted by frame."""
    pcfg = cfg["bounce"]
    out: list = []
    if pcfg["detector"] == "peak":
        pk = PeakBounceConfig(
            lookback=pcfg["lookback"], min_dy=pcfg["min_dy"], cooldown=pcfg["cooldown"])
        for t in tracks:
            out.extend(peak_detect_all(t.pts, pk))
    elif pcfg["detector"] == "catboost":
        from ..bounce.catboost import CatBoostBounceDetector, CatBoostBounceConfig
        cb = CatBoostBounceDetector(CatBoostBounceConfig(
            weights=pcfg["weights"], threshold=pcfg["catboost_threshold"],
            nms_window=pcfg["catboost_nms_window"],
        ))
        # Merge every validated track into a single frame-indexed series;
        # missing frames get NaN (CatBoost cubic-spline smoothing fills them).
        # This matches the single-highlight-clip usage the upstream model
        # was trained on, instead of running per fragment.
        max_frame = max((p.frame for t in tracks for p in t.pts), default=0)
        if max_frame == 0:
            return out
        merged = [None] * (max_frame + 1)
        for t in tracks:
            for p in t.pts:
                merged[p.frame] = p
        # Build a sequence with None stand-ins for gaps.
        class _Gap:
            __slots__ = ("x", "y", "frame")
            def __init__(self, f):
                self.x = None; self.y = None; self.frame = f
        seq = [merged[f] if merged[f] is not None else _Gap(f)
               for f in range(max_frame + 1)]
        out.extend(cb.detect(seq))
    else:
        raise NotImplementedError(
            "bounce.detector=%r not supported" % pcfg["detector"])
    out.sort(key=lambda e: e.frame)
    return out


def _project_bounces(events: list, calib: Calibration) -> list:
    """Project image-space BounceEvents to court-meter space; filter to
    plausible on-court bounces.  Returns (x_m, y_m, frame) tuples."""
    out = []
    for e in events:
        rx, ry = project_image_to_court(
            calib.H_img_to_real, (e.x_img, e.y_img))
        if (-1.0 <= rx <= ref.COURT_WIDTH_M + 1.0
                and -1.0 <= ry <= ref.COURT_LENGTH_M + 1.0):
            out.append((rx, ry, e.frame))
    return out


def run(
    video_path: str,
    output_path: str,
    calib: Calibration,
    cfg: dict,
    *,
    progress_every: int = 100,
) -> AnalyzeResult:
    import time
    ball_det, ball_detect_fn = _build_ball_detector(cfg["ball"])
    tcfg = cfg["tracker"]
    tracker = MultiTrackManager(TrackerConfig(
        gate_px=tcfg["gate_px"], max_gap_frames=tcfg["max_gap_frames"],
        min_len=tcfg["min_len"], min_speed=tcfg["min_speed"],
        max_speed=tcfg["max_speed"], render_tail=tcfg["render_tail"],
    ))

    rcfg = cfg["render"]
    minimap = Minimap(MinimapConfig(
        width_px=rcfg["minimap_width_px"],
        margin_px=rcfg["minimap_margin_px"],
        fade_frames=rcfg["bounce_fade_frames"],
    ))

    stroke_rec = _build_stroke_recognizer(cfg.get("action"), calib=calib)
    stroke_events: list = []
    if stroke_rec is not None:
        mode = "multi-player" if cfg["action"].get("player", {}).get("enabled", True) \
            else "full-frame single-player"
        print("[action] stroke classifier enabled (%s, window=%d, min_conf=%.2f)"
              % (mode, cfg["action"]["window_frames"], cfg["action"]["min_confidence"]),
              flush=True)

    # ------------------------------------------------------------------
    # PASS 1 — tracking only; remember per-frame state + retired tracks
    # ------------------------------------------------------------------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit("cannot open " + video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS)
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_states: dict[int, _FrameState] = {}
    retired_tracks: dict[int, Track] = {}     # by track.id
    tail_len = tcfg["render_tail"]
    frame_idx = 0

    print("[pass 1] tracking %d frames ..." % total, flush=True)
    t0 = time.time()
    t_last = t0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # snapshot track ids before update so we can detect retired ones
        before_ids = {t.id: t for t in tracker.tracks}

        cand_xys = ball_detect_fn(frame)
        tracker.update(cand_xys, frame_idx)
        champion = tracker.champion(frame_idx)

        after_ids = {t.id for t in tracker.tracks}
        for tid, t in before_ids.items():
            if tid not in after_ids and t.validated:
                retired_tracks[tid] = t

        if stroke_rec is not None:
            stroke_events.extend(stroke_rec.push_frame(frame, frame_idx))

        fs = _FrameState(n_cands=len(cand_xys), n_tracks=len(tracker.tracks))
        if champion is not None:
            fs.champion_id = champion.id
            fs.is_current_det = (champion.last_det_frame == frame_idx)
            fs.champion_trail = [
                (p.x, p.y, p.frame) for p in champion.pts
                if frame_idx - p.frame <= tail_len
            ]
        frame_states[frame_idx] = fs

        if frame_idx % progress_every == 0:
            now = time.time()
            fps = progress_every / max(now - t_last, 1e-6)
            eta = (total - frame_idx) / max(fps, 1e-6)
            print("  frame %d / %d  %.1f fps  eta %.0fs  (%d retired tracks)"
                  % (frame_idx, total, fps, eta, len(retired_tracks)),
                  flush=True)
            t_last = now
    cap.release()
    print("[pass 1] done in %.1fs  (%d validated tracks)" %
          (time.time() - t0, sum(1 for t in retired_tracks.values() if t.validated)),
          flush=True)

    if stroke_rec is not None:
        by_label: dict = {}
        by_player: dict = {}
        for ev in stroke_events:
            by_label[ev.label] = by_label.get(ev.label, 0) + 1
            if ev.player_id is not None:
                by_player.setdefault(ev.player_id, {})
                by_player[ev.player_id][ev.label] = \
                    by_player[ev.player_id].get(ev.label, 0) + 1
        label_summary = ", ".join("%s=%d" % kv for kv in sorted(by_label.items())) or "(none)"
        print("[action] %d stroke events: %s" % (len(stroke_events), label_summary),
              flush=True)
        for pid in sorted(by_player.keys()):
            parts = ", ".join("%s=%d" % kv for kv in sorted(by_player[pid].items()))
            print("[action]   player #%d: %s" % (pid, parts), flush=True)

    # Also include tracks still alive after the last frame
    for t in tracker.tracks:
        if t.validated:
            retired_tracks.setdefault(t.id, t)
    all_tracks = list(retired_tracks.values())

    # ------------------------------------------------------------------
    # BOUNCE DETECTION
    # ------------------------------------------------------------------
    print("[bounces] detector=%s ..." % cfg["bounce"]["detector"], flush=True)
    events = _run_bounce_detector(all_tracks, cfg)
    bounces_court = _project_bounces(events, calib)
    print("[bounces] %d raw events -> %d on-court" %
          (len(events), len(bounces_court)), flush=True)

    # ------------------------------------------------------------------
    # PASS 2 — render
    # ------------------------------------------------------------------
    cap = cv2.VideoCapture(video_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    print("[pass 2] rendering %d frames ..." % total, flush=True)
    frame_idx = 0
    t0 = time.time()
    t_last = t0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        fs = frame_states.get(frame_idx, _FrameState())

        vis = frame.copy()
        if fs.champion_id is not None and fs.champion_trail:
            pts_obj = [TrackPoint(x, y, f) for (x, y, f) in fs.champion_trail]
            draw_trail(vis, pts_obj, frame_idx,
                       tail=tail_len, is_current_det=fs.is_current_det)
        minimap.overlay(vis, bounces_court, frame_idx)
        n_shown = sum(1 for _, _, f in bounces_court if f <= frame_idx)

        champ_info = "-" if fs.champion_id is None \
            else "#%d len=%d" % (fs.champion_id, len(fs.champion_trail))
        hud = "f=%d/%d cands=%d tracks=%d champ=%s bounces=%d/%d" % (
            frame_idx, total, fs.n_cands, fs.n_tracks, champ_info,
            n_shown, len(bounces_court))
        draw_hud(vis, hud)

        out.write(vis)
        if frame_idx % progress_every == 0:
            now = time.time()
            fps = progress_every / max(now - t_last, 1e-6)
            eta = (total - frame_idx) / max(fps, 1e-6)
            print("  frame %d / %d  %.1f fps  eta %.0fs"
                  % (frame_idx, total, fps, eta), flush=True)
            t_last = now

    cap.release()
    out.release()
    print("[pass 2] done in %.1fs" % (time.time() - t0), flush=True)

    return AnalyzeResult(
        total_frames=frame_idx,
        bounces=len(bounces_court),
        validated_tracks=len(all_tracks),
        stroke_events=stroke_events,
    )
