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
from ..render.action import (
    draw_players, draw_stroke_labels, events_by_player_sorted,
)
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
    # Present when action.player is enabled: {track_id: (x0, y0, x1, y1)}.
    # Captured from the stroke recognizer on every Pass-1 frame so Pass-2
    # can render per-player bboxes without re-running YOLOv8.
    player_bboxes: dict = field(default_factory=dict)


@dataclass
class AnalyzeResult:
    total_frames: int
    bounces: int
    validated_tracks: int
    stroke_events: list = field(default_factory=list)
    inpainted_frames: int = 0        # only set when ball.inpainter.enabled
    coverage_before: int = 0         # frames with an original detection
    coverage_after: int = 0          # frames with a point after inpainting


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
    from ..action.pose import YOLOPoseTrackerConfig, YOLOPoseTracker
    from ..action.stroke_classifier import (
        MultiPlayerStrokeRecognizer,
        StrokeClassifier,
        StrokeClassifierConfig,
    )

    cls_cfg = StrokeClassifierConfig(
        weights=acfg["rnn_weights"],
        window_frames=acfg["window_frames"],
        labels=tuple(acfg["labels"]),
        min_confidence=acfg["min_confidence"],
        stride=acfg.get("stride", 5),
        emit_neutral=acfg.get("emit_neutral", False),
        ball_proximity_px=float(acfg.get("ball_proximity_px", 300)),
        ball_proximity_window_frames=int(acfg.get("ball_proximity_window_frames", 15)),
    )

    pcfg = acfg.get("player", {})
    tracker = YOLOPoseTracker(
        YOLOPoseTrackerConfig(
            weights=acfg["pose_weights"],
            device=acfg.get("pose_device", "cpu"),
            conf=pcfg.get("conf", 0.3),
            iou=pcfg.get("iou", 0.5),
            tracker=pcfg.get("tracker", "bytetrack.yaml"),
            imgsz=pcfg.get("imgsz", 640),
            score_threshold=acfg.get("score_threshold", 0.2),
            min_bbox_h=pcfg.get("min_bbox_h", 60),
            max_persons=pcfg.get("max_persons", 4),
            court_margin_m=pcfg.get("court_margin_m", 3.0),
        ),
        calib=calib,
    )
    return MultiPlayerStrokeRecognizer(
        cls_cfg, tracker,
        track_ttl_frames=pcfg.get("track_ttl_frames", 30),
    )


def _build_ball_detector(bcfg, calib=None):
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
        from ..ball.wasb import TwoStageBallDetector
        wcfg = WASBConfig(
            weights=bcfg["weights"],
            device=bcfg.get("device", "cpu"),
            runtime=bcfg.get("runtime", "auto"),
            onnx_path=bcfg.get("onnx_path", None),
            score_threshold=bcfg.get("score_threshold", 0.5),
            max_disp=bcfg.get("max_disp", 300.0),
        )
        use_two_stage = bool(bcfg.get("two_stage", True)) and calib is not None
        det = TwoStageBallDetector(
            wcfg, calib=calib,
            dedup_px=float(bcfg.get("two_stage_dedup_px", 60.0)),
        ) if use_two_stage else WASBBallDetector(wcfg)
        if use_two_stage:
            print("[wasb] two-stage enabled (main + far-crop)")
        def _run(frame):
            det.push_frame(frame)
            result = det.detect()
            # Main detector returns Optional[(x,y)]; TwoStage returns list.
            if result is None:
                return []
            if isinstance(result, list):
                return result
            return [result]
        return det, _run
    raise NotImplementedError("ball.detector=%r not supported" % name)


def _maybe_inpaint(tracks: list, icfg: dict, frame_size: tuple) -> tuple:
    """If `ball.inpainter.enabled`, fill each validated Track's INTERNAL
    gaps with InpaintNet and return a new list of synthetic Tracks.
    Otherwise return the input tracks unchanged.

    Per-track, not merged: the tracker already decided whether two
    separated segments belong to the same trajectory (via its own
    max_gap_frames).  If it split them into two Tracks, we respect that
    and do NOT fabricate a bridge between them — bridging across rallies
    or across a ball leaving the frame was the failure mode of the
    earlier merged-series approach (false bounces on synthesized
    between-track motion).

    Second return value is a stats dict (or None) for logging.
    """
    if not icfg or not icfg.get("enabled"):
        return tracks, None
    from ..ball.inpainter import (
        InpaintNetConfig, TrajectoryInpainter, result_to_trackpoints,
    )
    inp = TrajectoryInpainter(
        InpaintNetConfig(
            weights=icfg["weights"], device=icfg.get("device", "cpu"),
            window_mode=icfg.get("window_mode", "single"),
            seq_len=icfg.get("seq_len"),
            max_gap_frames=icfg.get("max_gap_frames", 60),
            th_h_frac=icfg.get("th_h_frac", 0.05),
        ),
        frame_size=frame_size,
    )

    new_tracks: list = []
    total_series_len = 0
    total_coverage_before = 0
    total_coverage_after = 0
    total_filled = 0

    for t in tracks:
        res = inp.inpaint_tracks([t])     # single-track series, no cross-track merge
        pts = result_to_trackpoints(res)
        if not pts:
            new_tracks.append(t)
            total_series_len += len(t.pts)
            total_coverage_before += len(t.pts)
            total_coverage_after += len(t.pts)
            continue
        fake = Track(pts[0].x, pts[0].y, pts[0].frame,
                     min_len=1, min_speed=0.0, max_speed=1e9)
        fake.pts = list(pts)
        fake.validated = True
        new_tracks.append(fake)

        finite = ~np.isnan(res.xs)
        total_series_len += len(res.xs)
        total_coverage_before += int((finite & ~res.was_inpainted).sum())
        total_coverage_after += len(pts)
        total_filled += int(res.was_inpainted.sum())

    stats = {
        "orig_tracks": len(tracks),
        "series_len": total_series_len,
        "coverage_before": total_coverage_before,
        "coverage_after": total_coverage_after,
        "inpainted_frames": total_filled,
    }
    return new_tracks, stats


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
    ball_det, ball_detect_fn = _build_ball_detector(cfg["ball"], calib=calib)
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

    # Precompute horizontal court strip mask to reject adjacent-court balls.
    # Only left/right sidelines are enforced; vertical extent is unconstrained
    # so balls that fly high or arc above the baseline are still captured.
    _court_mask: Optional[np.ndarray] = None
    _court_mask_margin = cfg.get("ball", {}).get("court_mask_margin_px", 30)
    if _court_mask_margin > 0:
        _court_mask = calib.court_h_strip_mask(W, H, margin_px=_court_mask_margin)

    def _filter_cands(cands: list) -> list:
        if _court_mask is None:
            return cands
        return [(x, y) for x, y in cands
                if 0 <= int(y) < H and 0 <= int(x) < W
                and _court_mask[int(y), int(x)] > 0]

    frame_states: dict[int, _FrameState] = {}
    retired_tracks: dict[int, Track] = {}     # by track.id
    tail_len = tcfg["render_tail"]
    frame_idx = 0

    # Online gating: skip the expensive pose + stroke pass during obvious
    # non-rally stretches.  Ball detection still runs every frame so a
    # resuming rally is caught within one frame; once recovered we call
    # stroke_rec.reset() to flush any stale pose window from the old rally.
    _rcfg = cfg.get("rally") or {}
    online_gating = bool(_rcfg.get("online_gating", False)) and stroke_rec is not None
    silence_thresh = max(1, int(_rcfg.get("online_silence_frames", 90)))
    gate_in_rally = False
    silent_frames = 0
    if online_gating:
        print("[rally] online gating enabled (silence_thresh=%d frames)" %
              silence_thresh, flush=True)

    print("[pass 1] tracking %d frames ..." % total, flush=True)
    t0 = time.time()
    t_last = t0
    gated_skips = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # snapshot track ids before update so we can detect retired ones
        before_ids = {t.id: t for t in tracker.tracks}

        cand_xys = _filter_cands(ball_detect_fn(frame))
        tracker.update(cand_xys, frame_idx)
        champion = tracker.champion(frame_idx)

        after_ids = {t.id for t in tracker.tracks}
        for tid, t in before_ids.items():
            if tid not in after_ids and t.validated:
                retired_tracks[tid] = t

        # ------ online rally gating state machine ------
        if online_gating:
            has_activity = bool(cand_xys) or (champion is not None and champion.validated)
            if not gate_in_rally:
                if has_activity:
                    gate_in_rally = True
                    silent_frames = 0
                    stroke_rec.reset()     # flush stale pose window / tracks
            else:
                if has_activity:
                    silent_frames = 0
                else:
                    silent_frames += 1
                    if silent_frames >= silence_thresh:
                        gate_in_rally = False

        if stroke_rec is not None and (not online_gating or gate_in_rally):
            ball_xy: Optional[tuple] = None
            if champion is not None and champion.pts:
                last_pt = champion.pts[-1]
                ball_xy = (last_pt.x, last_pt.y)
            stroke_events.extend(stroke_rec.push_frame(frame, frame_idx,
                                                        ball_xy=ball_xy))
        elif online_gating and not gate_in_rally:
            gated_skips += 1

        fs = _FrameState(n_cands=len(cand_xys), n_tracks=len(tracker.tracks))
        if champion is not None:
            fs.champion_id = champion.id
            fs.is_current_det = (champion.last_det_frame == frame_idx)
            fs.champion_trail = [
                (p.x, p.y, p.frame) for p in champion.pts
                if frame_idx - p.frame <= tail_len
            ]
        if stroke_rec is not None:
            fs.player_bboxes = dict(getattr(stroke_rec, "last_detections", {}))
        frame_states[frame_idx] = fs

        if frame_idx % progress_every == 0:
            now = time.time()
            proc_fps = progress_every / max(now - t_last, 1e-6)
            eta = (total - frame_idx) / max(proc_fps, 1e-6)
            print("  frame %d / %d  %.1f fps  eta %.0fs  (%d retired tracks)"
                  % (frame_idx, total, proc_fps, eta, len(retired_tracks)),
                  flush=True)
            t_last = now
    cap.release()
    print("[pass 1] done in %.1fs  (%d validated tracks)" %
          (time.time() - t0, sum(1 for t in retired_tracks.values() if t.validated)),
          flush=True)
    if online_gating:
        pct = 100.0 * gated_skips / max(frame_idx, 1)
        print("[rally] online gating: %d / %d frames skipped pose+stroke (%.0f%%)"
              % (gated_skips, frame_idx, pct), flush=True)

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
    # TRAJECTORY INPAINTING (optional; V2)
    # ------------------------------------------------------------------
    icfg = cfg.get("inpainter", {})
    bounce_input_tracks, inpaint_stats = _maybe_inpaint(
        all_tracks, icfg, frame_size=(W, H))
    if inpaint_stats is not None:
        print("[inpaint] %d orig tracks -> series len %d; "
              "coverage %d -> %d (+%d filled frames)" % (
                  inpaint_stats["orig_tracks"],
                  inpaint_stats["series_len"],
                  inpaint_stats["coverage_before"],
                  inpaint_stats["coverage_after"],
                  inpaint_stats["inpainted_frames"],
              ), flush=True)

    # ------------------------------------------------------------------
    # BOUNCE DETECTION
    # ------------------------------------------------------------------
    print("[bounces] detector=%s ..." % cfg["bounce"]["detector"], flush=True)
    events = _run_bounce_detector(bounce_input_tracks, cfg)
    bounces_court = _project_bounces(events, calib)
    print("[bounces] %d raw events -> %d on-court" %
          (len(events), len(bounces_court)), flush=True)

    # ------------------------------------------------------------------
    # RALLY DETECTION (before Pass 2 so the renderer can clear HUD /
    # minimap at rally boundaries instead of accumulating across the
    # whole video).
    # ------------------------------------------------------------------
    rcfg_rally = cfg.get("rally") or {}
    rally_enabled = bool(rcfg_rally.get("enabled", True))
    rallies: list = []
    frame_to_rally: list = []     # frame_idx -> rally index, or -1
    if rally_enabled:
        from .rally import detect_rallies

        event_frames: list = []
        for t in all_tracks:
            event_frames.extend(p.frame for p in t.pts)
        event_frames.extend(f for (_, _, f) in bounces_court)
        event_frames.extend(ev.frame for ev in stroke_events)

        rallies = detect_rallies(
            event_frames,
            fps=fps,
            gap_seconds=float(rcfg_rally.get("gap_seconds", 3.0)),
            min_events=int(rcfg_rally.get("min_events", 3)),
            pre_roll_frames=int(rcfg_rally.get("pre_roll_frames", 30)),
            post_roll_frames=int(rcfg_rally.get("post_roll_frames", 30)),
            total_frames=total,
        )
        print("[rally] detected %d rallies" % len(rallies), flush=True)

        frame_to_rally = [-1] * (total + 2)
        for r in rallies:
            for f in range(r.start_frame, min(r.end_frame, total) + 1):
                frame_to_rally[f] = r.idx

    # ------------------------------------------------------------------
    # PASS 2 — render
    # ------------------------------------------------------------------
    cap = cv2.VideoCapture(video_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    # Per-player event bucket for the stroke-label overlay.  Sorted once
    # here so Pass 2's bisect lookup stays O(log n).
    events_by_player = events_by_player_sorted(stroke_events)
    # Keep stroke labels visible for 2 s past the event (upstream classifier
    # fires every `stride` frames per player, so at 30 fps the label is
    # refreshed long before this TTL expires during active play).
    stroke_label_ttl = int(round(fps * 2.0)) if fps else 60

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

        # Current rally (or -1 if this frame sits in a gap).  Used below
        # to reset HUD text and minimap bounces at rally boundaries.
        ridx = frame_to_rally[frame_idx] if frame_to_rally else -1
        r_cur = rallies[ridx] if ridx >= 0 else None
        if r_cur is not None:
            rally_bounces = [(x, y, f) for (x, y, f) in bounces_court
                             if r_cur.start_frame <= f <= frame_idx]
        else:
            rally_bounces = []

        if fs.champion_id is not None and fs.champion_trail:
            pts_obj = [TrackPoint(x, y, f) for (x, y, f) in fs.champion_trail]
            draw_trail(vis, pts_obj, frame_idx,
                       tail=tail_len, is_current_det=fs.is_current_det)
        if fs.player_bboxes:
            draw_players(vis, fs.player_bboxes)
            draw_stroke_labels(vis, fs.player_bboxes, events_by_player,
                               frame_idx, ttl_frames=stroke_label_ttl)

        if rally_enabled and r_cur is None:
            # Between rallies: no minimap, no per-rally counters.
            hud = "f=%d/%d  (between rallies)  rallies=%d" % (
                frame_idx, total, len(rallies))
        else:
            minimap.overlay(vis, rally_bounces, frame_idx)
            champ_info = "-" if fs.champion_id is None \
                else "#%d len=%d" % (fs.champion_id, len(fs.champion_trail))
            if r_cur is not None:
                hud = "rally %d  f=%d/%d  cands=%d tracks=%d champ=%s bounces=%d" % (
                    r_cur.idx + 1, frame_idx, total,
                    fs.n_cands, fs.n_tracks, champ_info, len(rally_bounces))
            else:
                # rally disabled → fall back to original cumulative HUD
                n_shown = sum(1 for _, _, f in bounces_court if f <= frame_idx)
                hud = "f=%d/%d cands=%d tracks=%d champ=%s bounces=%d/%d" % (
                    frame_idx, total, fs.n_cands, fs.n_tracks, champ_info,
                    n_shown, len(bounces_court))
        draw_hud(vis, hud)

        out.write(vis)
        if frame_idx % progress_every == 0:
            now = time.time()
            proc_fps = progress_every / max(now - t_last, 1e-6)
            eta = (total - frame_idx) / max(proc_fps, 1e-6)
            print("  frame %d / %d  %.1f fps  eta %.0fs"
                  % (frame_idx, total, proc_fps, eta), flush=True)
            t_last = now

    cap.release()
    out.release()
    print("[pass 2] done in %.1fs" % (time.time() - t0), flush=True)

    # ------------------------------------------------------------------
    # RALLY CUT VIDEO — detection already done before Pass 2, now cut
    # the rendered output into a highlight-only companion file.
    # ------------------------------------------------------------------
    if rally_enabled:
        from .rally import write_rally_video, save_rally_json

        stem, ext = os.path.splitext(output_path)
        clip_path = rcfg_rally.get("clip_path") or (stem + "_rally" + (ext or ".mp4"))
        json_path = rcfg_rally.get("json_path") or (stem + "_rallies.json")
        if rallies:
            t_cut = time.time()
            write_rally_video(
                output_path, rallies, clip_path,
                separator_seconds=float(rcfg_rally.get("separator_seconds", 1.0)),
            )
            print("[rally] wrote %s (%.1fs)" %
                  (clip_path, time.time() - t_cut), flush=True)
        save_rally_json(rallies, json_path, fps=fps)
        print("[rally] wrote %s" % json_path, flush=True)

    return AnalyzeResult(
        total_frames=frame_idx,
        bounces=len(bounces_court),
        validated_tracks=len(all_tracks),
        stroke_events=stroke_events,
        inpainted_frames=(inpaint_stats["inpainted_frames"] if inpaint_stats else 0),
        coverage_before=(inpaint_stats["coverage_before"] if inpaint_stats else 0),
        coverage_after=(inpaint_stats["coverage_after"] if inpaint_stats else 0),
    )
