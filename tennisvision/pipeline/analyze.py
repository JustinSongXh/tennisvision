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
import threading
import time
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


def _run_pose_for_rally(rally, video_path: str, stroke_rec, frame_states: dict):
    """Pose + stroke for one rally window.

    Re-opens the video, seeks to rally.start_frame, and feeds each frame
    through `stroke_rec` with the tracker's ball position pulled from
    `frame_states` (Pass 1a already populated it).  Returns
    (events, bboxes_by_frame, frames_done, wall_time_s).
    """
    stroke_rec.reset()
    cap = cv2.VideoCapture(video_path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, rally.start_frame - 1))
        events: list = []
        bboxes_by_frame: dict = {}
        t0 = time.time()
        done = 0
        for fi in range(rally.start_frame, rally.end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break
            fs = frame_states.get(fi)
            ball_xy: Optional[tuple] = None
            if fs is not None and fs.champion_trail:
                last_x, last_y, last_f = fs.champion_trail[-1]
                if fi - last_f <= 5:
                    ball_xy = (last_x, last_y)
            events.extend(stroke_rec.push_frame(frame, fi, ball_xy=ball_xy))
            bboxes_by_frame[fi] = dict(
                getattr(stroke_rec, "last_detections", {}))
            done += 1
    finally:
        cap.release()
    return events, bboxes_by_frame, done, time.time() - t0


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
    # Peek at frame dimensions so *_ratio config keys can be resolved to
    # pixels before any detector or tracker is built.  Ratios scale with
    # frame resolution — same config file works on 720p / 1080p / 4K.
    _cap_peek = cv2.VideoCapture(video_path)
    if not _cap_peek.isOpened():
        raise SystemExit("cannot open " + video_path)
    W = int(_cap_peek.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(_cap_peek.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = _cap_peek.get(cv2.CAP_PROP_FPS)
    total = int(_cap_peek.get(cv2.CAP_PROP_FRAME_COUNT))
    _cap_peek.release()
    frame_diag = float((W * W + H * H) ** 0.5)

    def _resolve_ratio(d: dict, ratio_key: str, pixel_key: str, basis: float):
        """Return the pixel value for `pixel_key`, preferring a ratio-based
        key (multiplied by `basis`) if it's set, otherwise the pixel key."""
        if ratio_key in d and d[ratio_key] is not None:
            return float(d[ratio_key]) * basis
        return d.get(pixel_key)

    # Ball + tracker config: resolve ratios → pixels so downstream
    # classes stay pixel-based (no API churn).
    bcfg = dict(cfg["ball"])
    v = _resolve_ratio(bcfg, "max_disp_ratio", "max_disp", frame_diag)
    if v is not None:
        bcfg["max_disp"] = v
    v = _resolve_ratio(bcfg, "two_stage_dedup_ratio", "two_stage_dedup_px", frame_diag)
    if v is not None:
        bcfg["two_stage_dedup_px"] = v
    tcfg = dict(cfg["tracker"])
    v = _resolve_ratio(tcfg, "gate_ratio", "gate_px", frame_diag)
    if v is not None:
        tcfg["gate_px"] = v
    v = _resolve_ratio(tcfg, "min_speed_ratio", "min_speed", frame_diag)
    if v is not None:
        tcfg["min_speed"] = v
    v = _resolve_ratio(tcfg, "max_speed_ratio", "max_speed", frame_diag)
    if v is not None:
        tcfg["max_speed"] = v

    print("[video] %dx%d (diag=%.0f px)  fps=%.2f  total=%d" %
          (W, H, frame_diag, fps, total), flush=True)
    print("[config] resolved ball.max_disp=%.0f  tracker.gate_px=%.0f  "
          "min_speed=%.1f  max_speed=%.1f"
          % (bcfg.get("max_disp", 0.0), tcfg.get("gate_px", 0.0),
             tcfg.get("min_speed", 0.0), tcfg.get("max_speed", 0.0)),
          flush=True)

    ball_det, ball_detect_fn = _build_ball_detector(bcfg, calib=calib)

    # Adjacent-court filter (track-level): built here so Pass 1a can
    # record the polygon once and Pass-1a-post can apply it to every
    # validated track.  A single stray candidate inside the polygon is
    # allowed through to the tracker — only whole tracks that sit
    # predominantly outside (fraction-inside < min_inside_fraction) get
    # dropped after tracks retire.  This is looser than a per-candidate
    # gate so near-sideline serves don't lose tracker continuity, but
    # still strips adjacent-court tracks that WASB grabs from the
    # neighboring court.
    on_court_margin_m = float(bcfg.get("on_court_margin_m", 0.0))
    on_court_min_inside = float(bcfg.get("on_court_min_inside_fraction", 0.0))
    on_court_poly: Optional[np.ndarray] = None
    if on_court_margin_m > 0.0 and calib is not None:
        try:
            on_court_poly = calib.on_court_polygon_img(on_court_margin_m)
            if not np.all(np.isfinite(on_court_poly)):
                print("[ball] on-court polygon has non-finite corners; "
                      "disabling adjacent-court filter", flush=True)
                on_court_poly = None
        except Exception as e:                      # noqa: BLE001
            print("[ball] on_court_polygon_img failed (%s); disabling "
                  "adjacent-court filter" % e, flush=True)
            on_court_poly = None
    if on_court_poly is not None:
        print("[ball] adjacent-court filter on (margin=%.1fm, "
              "min_inside=%.2f, polygon=%s)"
              % (on_court_margin_m, on_court_min_inside,
                 " ".join("(%.0f,%.0f)" % (x, y) for x, y in on_court_poly)),
              flush=True)

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

    frame_states: dict[int, _FrameState] = {}
    retired_tracks: dict[int, Track] = {}     # by track.id
    tail_len = tcfg["render_tail"]
    frame_idx = 0

    # ------------------------------------------------------------------
    # PASS 1 — ball detection + tracking + online rally detection.
    # Default (offline): pose is DEFERRED to Pass 1b so it only runs on
    # confirmed rally frames.  With `rally.online=True`, pose + stroke
    # classifier are inlined into this single pass so per-frame results
    # are available as the video is processed (for real-time overlays /
    # scoreboards); the bounce-based rally filters and rally-merge pass
    # are skipped in that mode because bounce detection is batch-only.
    # ------------------------------------------------------------------
    _rcfg = cfg.get("rally") or {}
    rally_enabled = bool(_rcfg.get("enabled", True))
    online_mode = bool(_rcfg.get("online", False)) and stroke_rec is not None
    if _rcfg.get("online", False) and stroke_rec is None:
        print("[rally] online=True but no stroke classifier available — "
              "falling back to offline two-pass flow", flush=True)

    net_center = np.array([ref.COURT_WIDTH_M / 2.0, ref.NET_Y, 1.0])
    p_net = calib.H_real_to_img @ net_center
    net_y_px = float(p_net[1] / p_net[2])
    silence_thresh = max(1, int(round(
        float(_rcfg.get("online_silence_seconds", 3.0)) * max(fps, 1.0))))
    pre_roll_frames = int(round(
        float(_rcfg.get("pre_roll_seconds", 1.0)) * max(fps, 1.0)))
    post_roll_frames = int(round(
        float(_rcfg.get("post_roll_seconds", 1.0)) * max(fps, 1.0)))

    online_det = None
    if rally_enabled:
        from .rally import OnlineRallyDetector
        crossing_silence_thresh = int(round(
            float(_rcfg.get("online_crossing_silence_seconds", 0.0)) * max(fps, 1.0)))
        online_det = OnlineRallyDetector(
            net_y_px=net_y_px,
            silence_thresh_frames=silence_thresh,
            crossing_silence_thresh_frames=crossing_silence_thresh,
            min_net_crossings=int(_rcfg.get("online_min_net_crossings", 3)),
            min_activity_density=float(
                _rcfg.get("online_min_activity_density", 0.0)),
            lob_silence_multiplier=float(
                _rcfg.get("online_silence_lob_multiplier", 1.0)),
            lob_up_speed_px=float(
                _rcfg.get("online_silence_lob_up_speed_px", 2.0)),
            pre_roll_frames=pre_roll_frames,
            post_roll_frames=post_roll_frames,
            total_frames=total,
        )
        print("[rally] online detector: silence=%.1fs (lob x%.1f)  "
              "crossing_silence=%.1fs  min_crossings=%d  min_density=%.2f  "
              "pre/post_roll=%.1fs/%.1fs  net_y_px=%.0f" % (
                  float(_rcfg.get("online_silence_seconds", 3.0)),
                  float(_rcfg.get("online_silence_lob_multiplier", 1.0)),
                  float(_rcfg.get("online_crossing_silence_seconds", 0.0)),
                  int(_rcfg.get("online_min_net_crossings", 3)),
                  float(_rcfg.get("online_min_activity_density", 0.0)),
                  float(_rcfg.get("pre_roll_seconds", 1.0)),
                  float(_rcfg.get("post_roll_seconds", 1.0)),
                  net_y_px), flush=True)

    pass_label = "pass 1 (online: ball+pose+tracker)" if online_mode \
        else "pass 1a (ball+tracker only)"
    print("[%s] %d frames ..." % (pass_label, total), flush=True)
    if online_mode:
        stroke_rec.reset()
    t0 = time.time()
    t_last = t0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        before_ids = {t.id: t for t in tracker.tracks}

        cand_xys = ball_detect_fn(frame)
        tracker.update(cand_xys, frame_idx)
        champion = tracker.champion(frame_idx)

        after_ids = {t.id for t in tracker.tracks}
        for tid, t in before_ids.items():
            if tid not in after_ids and t.validated:
                retired_tracks[tid] = t

        if online_det is not None:
            online_det.observe(frame_idx, cand_xys, champion)

        fs = _FrameState(n_cands=len(cand_xys), n_tracks=len(tracker.tracks))
        if champion is not None:
            fs.champion_id = champion.id
            fs.is_current_det = (champion.last_det_frame == frame_idx)
            fs.champion_trail = [
                (p.x, p.y, p.frame) for p in champion.pts
                if frame_idx - p.frame <= tail_len
            ]

        # Online mode: run pose + stroke classifier inline so downstream
        # consumers can read per-frame stroke events / player bboxes as
        # the frame lands, without waiting for a batch Pass 1b.
        if online_mode:
            ball_xy: Optional[tuple] = None
            if fs.champion_trail:
                last_x, last_y, last_f = fs.champion_trail[-1]
                if frame_idx - last_f <= 5:
                    ball_xy = (last_x, last_y)
            stroke_events.extend(
                stroke_rec.push_frame(frame, frame_idx, ball_xy=ball_xy))
            fs.player_bboxes = dict(
                getattr(stroke_rec, "last_detections", {}))

        frame_states[frame_idx] = fs

        if frame_idx % progress_every == 0:
            now = time.time()
            proc_fps = progress_every / max(now - t_last, 1e-6)
            eta = (total - frame_idx) / max(proc_fps, 1e-6)
            n_rallies = len(online_det.rallies) if online_det is not None else 0
            print("  [%s] frame %d / %d  %.1f fps  eta %.0fs  "
                  "(%d retired tracks, %d rallies, %d strokes)"
                  % (pass_label, frame_idx, total, proc_fps, eta,
                     len(retired_tracks), n_rallies, len(stroke_events)),
                  flush=True)
            t_last = now
    cap.release()
    if online_det is not None:
        online_det.finalize(frame_idx)
        rallies = online_det.rallies
    else:
        rallies = []
    print("[%s] done in %.1fs  (%d validated tracks, %d rallies, %d strokes)" %
          (pass_label, time.time() - t0,
           sum(1 for t in retired_tracks.values() if t.validated),
           len(rallies), len(stroke_events)),
          flush=True)

    # Also include tracks still alive after the last frame
    for t in tracker.tracks:
        if t.validated:
            retired_tracks.setdefault(t.id, t)
    all_tracks = list(retired_tracks.values())

    # ------------------------------------------------------------------
    # ADJACENT-COURT FILTER (track-level) — drop tracks whose fraction
    # of points inside the inflated court polygon falls below
    # `ball.on_court_min_inside_fraction`.  Runs before inpainting /
    # bounce detection so adjacent-court tracks never contribute to
    # either.  Also scrubs `frame_states` entries whose champion came
    # from a dropped track, so Pass 2 doesn't render a ghost trail for
    # it.  Per-track (not per-candidate) so a lone stray inside the
    # polygon never kills a near-sideline legitimate track.
    # ------------------------------------------------------------------
    if (on_court_poly is not None and all_tracks
            and on_court_min_inside > 0.0):
        kept_tracks: list = []
        kept_ids: set = set()
        dropped_ids: list = []
        for t in all_tracks:
            n_pts = len(t.pts)
            if n_pts == 0:
                continue
            n_inside = sum(
                1 for p in t.pts
                if cv2.pointPolygonTest(
                    on_court_poly, (float(p.x), float(p.y)), False) >= 0)
            frac = n_inside / n_pts
            if frac >= on_court_min_inside:
                kept_tracks.append(t)
                kept_ids.add(t.id)
            else:
                dropped_ids.append((t.id, n_pts, frac))
        if dropped_ids:
            print("[ball] adjacent-court filter dropped %d / %d tracks "
                  "(min_inside=%.2f)"
                  % (len(dropped_ids), len(all_tracks), on_court_min_inside),
                  flush=True)
            for tid, npts, frac in dropped_ids[:10]:
                print("  [drop] track %d  n_pts=%d  inside=%.2f"
                      % (tid, npts, frac), flush=True)
            # Scrub frame_states so Pass 2 doesn't render a trail from a
            # dropped adjacent-court champion.
            for fs in frame_states.values():
                if (fs.champion_id is not None
                        and fs.champion_id not in kept_ids):
                    fs.champion_id = None
                    fs.champion_trail = []
                    fs.is_current_det = False
        all_tracks = kept_tracks

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
    # RALLY-MERGE PASS (offline only) — glue back together rallies that
    # Pass 1a split at a mid-rally ball-tracking drop-out.  Runs before
    # bounce-filter so merged rallies are the ones evaluated for both-
    # halves coverage.  Needs bounces (gap without a bounce is the
    # "same rally" signal), so skipped in online mode.
    # ------------------------------------------------------------------
    if not online_mode:
        merge_gap = float(_rcfg.get("merge_gap_seconds", 0.0))
        if merge_gap > 0.0 and len(rallies) >= 2:
            from .rally import merge_close_rallies
            pre_n = len(rallies)
            rallies, n_merged = merge_close_rallies(
                rallies, bounces_court, fps, max_gap_seconds=merge_gap,
            )
            if n_merged:
                print("[rally] merge-filter combined %d pairs (%d -> %d "
                      "rallies; gap<=%.1fs, no bounce in gap)"
                      % (n_merged, pre_n, len(rallies), merge_gap),
                      flush=True)

    # ------------------------------------------------------------------
    # BOUNCE-FILTER (offline only) — require bounces in both court halves.
    # Online mode skips this: rally boundaries are committed when the
    # detector closes them, and bounce detection is a post-pass.
    # ------------------------------------------------------------------
    if (not online_mode
            and bool(_rcfg.get("require_bounces_both_halves", True))
            and rallies):
        kept: list = []
        dropped: list = []
        for r in rallies:
            near = 0
            far = 0
            for (_rx, ry, f) in bounces_court:
                if r.start_frame <= f <= r.end_frame:
                    if ry < ref.NET_Y:
                        near += 1
                    else:
                        far += 1
            if near >= 1 and far >= 1:
                r.idx = len(kept)
                kept.append(r)
            else:
                dropped.append((r, near, far))
        if dropped:
            print("[rally] bounce-filter dropped %d / %d (need bounces in both halves)"
                  % (len(dropped), len(dropped) + len(kept)), flush=True)
            for r, near, far in dropped:
                print("  [%d..%d] crossings=%d bounces=(near=%d,far=%d)"
                      % (r.start_frame, r.end_frame, r.net_crossings, near, far),
                      flush=True)
        rallies = kept

    # ------------------------------------------------------------------
    # PASS 1b — pose + stroke classifier, per rally (offline only).
    # Online mode already produced stroke_events inline during Pass 1.
    # ------------------------------------------------------------------
    if not online_mode and stroke_rec is not None and rallies:
        t_pass1b = time.time()
        total_rally_frames = 0
        print("[pass 1b] pose + stroke for %d rallies ..." % len(rallies),
              flush=True)
        for r in rallies:
            evs, bboxes, n_done, r_time = _run_pose_for_rally(
                r, video_path, stroke_rec, frame_states)
            stroke_events.extend(evs)
            for fi, bb in bboxes.items():
                fs = frame_states.get(fi)
                if fs is not None:
                    fs.player_bboxes = bb
            total_rally_frames += n_done
            print("  [pass 1b] rally %d/%d  frames [%d..%d] (%d f)  "
                  "%.1fs = %.1f fps  crossings=%d  strokes_so_far=%d"
                  % (r.idx + 1, len(rallies), r.start_frame, r.end_frame,
                     n_done, r_time, n_done / max(r_time, 1e-6),
                     r.net_crossings, len(stroke_events)),
                  flush=True)
        dt = time.time() - t_pass1b
        print("[pass 1b] done in %.1fs  (%d frames, %.1f fps, %d strokes)"
              % (dt, total_rally_frames,
                 total_rally_frames / max(dt, 1e-6),
                 len(stroke_events)), flush=True)

    # ------------------------------------------------------------------
    # Common post-processing (both modes): backfill n_strokes, optional
    # stroke-count filter, per-label / per-player summary.
    # ------------------------------------------------------------------
    if rallies and stroke_events:
        for r in rallies:
            r.n_strokes = sum(
                1 for ev in stroke_events
                if r.start_frame <= ev.frame <= r.end_frame)

        post_min_strokes = int(_rcfg.get("post_filter_min_strokes", 0))
        if post_min_strokes > 0:
            kept: list = []
            dropped: list = []
            for r in rallies:
                if r.n_strokes >= post_min_strokes:
                    r.idx = len(kept)
                    kept.append(r)
                else:
                    dropped.append(r)
            if dropped:
                print("[rally] stroke-filter dropped %d / %d (n_strokes<%d)"
                      % (len(dropped), len(dropped) + len(kept),
                         post_min_strokes), flush=True)
                for r in dropped:
                    print("  [%d..%d] crossings=%d n_strokes=%d"
                          % (r.start_frame, r.end_frame, r.net_crossings,
                             r.n_strokes), flush=True)
            rallies = kept

    if stroke_events:
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

    # Build the frame→rally lookup used in Pass 2 for HUD / minimap reset.
    frame_to_rally: list = [-1] * (total + 2)
    for r in rallies:
        for f in range(r.start_frame, min(r.end_frame, total) + 1):
            frame_to_rally[f] = r.idx

    rcfg_rally = cfg.get("rally") or {}

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
        from .rally import write_rally_video, save_rally_json, select_clip_rallies

        stem, ext = os.path.splitext(output_path)
        clip_path = rcfg_rally.get("clip_path") or (stem + "_rally" + (ext or ".mp4"))
        json_path = rcfg_rally.get("json_path") or (stem + "_rallies.json")

        # Full list stays in the json.  The cut video gets a stricter
        # subset — short rallies and ball-pickup blips are dropped.
        clip_rallies = select_clip_rallies(
            rallies, fps,
            min_net_crossings=int(rcfg_rally.get("clip_min_net_crossings", 3)),
            min_duration_seconds=float(rcfg_rally.get("clip_min_duration_seconds", 2.0)),
        )
        if len(clip_rallies) != len(rallies):
            print("[rally] clip filter: %d -> %d rallies "
                  "(clip_min_net_crossings=%d, clip_min_duration_seconds=%.1f)"
                  % (len(rallies), len(clip_rallies),
                     int(rcfg_rally.get("clip_min_net_crossings", 3)),
                     float(rcfg_rally.get("clip_min_duration_seconds", 2.0))),
                  flush=True)

        if clip_rallies:
            t_cut = time.time()
            write_rally_video(
                output_path, clip_rallies, clip_path,
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
