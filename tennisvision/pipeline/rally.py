"""Rally detection and cut-video writer.

A rally is a contiguous period of ball activity.  We take the union of
three event streams the main pipeline already produces — ball-track
frames, bounce frames, stroke-event frames — and group consecutive
events whose gap is smaller than `gap_seconds`.  Groups shorter than
`min_events` are dropped as noise (a stray ball detection, a single
practice swing).  Each kept group becomes a `Rally` with pre- and
post-roll margins applied.

`write_rally_video()` re-reads the already-annotated output video and
copies the frames inside each rally into a new file, inserting a short
black "Rally N" title between rallies.  No re-encoding of the analysis
pass is needed — this runs after Pass 2 on the final .mp4 the user
already has.
"""

from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import asdict, dataclass
from typing import Deque, Iterable, Optional

import cv2
import numpy as np


@dataclass
class Rally:
    idx: int
    start_frame: int
    end_frame: int
    n_events: int
    # Populated by the online detector / validate_rallies(); useful for
    # downstream selection (e.g. a stricter net-crossing threshold for
    # the clip video).
    net_crossings: int = 0
    n_strokes: int = 0


class OnlineRallyDetector:
    """State machine fed one frame at a time during Pass 1a.

    Consumes (frame_idx, cand_xys, champion) each frame and emits a
    `Rally` as soon as it confirms the previous activity burst was a
    real rally (≥ min_net_crossings net transitions).  Rally start is
    the FIRST frame of ball activity in that burst — not the frame of
    the 3rd crossing — because the serve toss and initial motion are
    part of the rally.

    Net-side is decided by `p.y < net_y_px` (midpoint approximation).
    Good enough for binary "did the ball end up on the other half" over
    many frames, even though it breaks down for airborne balls at the
    instant of crossing.
    """

    def __init__(self, *, net_y_px: float, silence_thresh_frames: int,
                 min_net_crossings: int, pre_roll_frames: int,
                 post_roll_frames: int, total_frames: int,
                 crossing_silence_thresh_frames: int = 0,
                 min_activity_density: float = 0.0,
                 lob_silence_multiplier: float = 1.0,
                 lob_up_speed_px: float = 2.0,
                 discontinuity_jump_px: float = 0.0,
                 discontinuity_no_cross_frames: int = 30,
                 serve_enabled: bool = False,
                 serve_toss_rise_px_near: float = 132.0,
                 serve_toss_rise_px_far: float = 33.0,
                 near_baseline_y_px: Optional[float] = None,
                 far_baseline_y_px: Optional[float] = None,
                 serve_toss_min_frames: int = 6,
                 serve_toss_max_horiz_ratio: float = 0.6,
                 serve_baseline_margin_m: float = 4.0,
                 serve_pre_static_max_dy_px: float = 12.0,
                 serve_suppress_frames: int = 60,
                 serve_history_frames: int = 60,
                 serve_soft_quiet_frames: int = 30,
                 H_img_to_real: Optional[np.ndarray] = None,
                 court_length_m: float = 23.77):
        self.net_y_px = float(net_y_px)
        self.silence_thresh = max(1, int(silence_thresh_frames))
        self.min_crossings = max(1, int(min_net_crossings))
        self.pre_roll = max(0, int(pre_roll_frames))
        self.post_roll = max(0, int(post_roll_frames))
        self.total = max(1, int(total_frames))
        # If > 0, force-close the current activity when no net crossing
        # has been observed for this many frames — catches ball-pickup /
        # dribble / toss between real rallies that would otherwise keep
        # the activity burst open (the ball is still being detected).
        self.crossing_silence_thresh = max(0, int(crossing_silence_thresh_frames))
        # Activity-density floor: fraction of the activity window that
        # must have had a ball detection to count as a real rally.
        # Spotty coverage (e.g. 12%) usually means the burst spans a
        # warm-up / scrappy exchange with long invisible stretches.
        self.min_activity_density = max(0.0, float(min_activity_density))
        # Lob handling: when the ball was heading up OR was already in
        # the far-side airspace at the moment of detection loss, stretch
        # the silence window by this multiplier.  A lob can stay airborne
        # 2-3s with no detection (small, far from camera, partially
        # occluded by net), which under a fixed silence_thresh would
        # force-close the rally mid-flight.  1.0 disables the stretch.
        self.lob_silence_multiplier = max(1.0, float(lob_silence_multiplier))
        self.lob_up_speed_px = max(0.0, float(lob_up_speed_px))
        # Discontinuity-close gate: force-close a burst when the ball
        # appears to teleport (jump + direction flip) in a stretch
        # without recent net crossings — signals a between-point ball
        # pickup & re-toss that bounce / crossing-silence counters
        # otherwise miss.  See config.py for full rationale.
        self.discontinuity_jump_px = max(0.0, float(discontinuity_jump_px))
        self.discontinuity_no_cross_frames = max(
            0, int(discontinuity_no_cross_frames))
        # Serve detection: a confirmed toss pattern
        #   hold → rise → apex → descent, near a baseline, mostly vertical
        # is a SOFT signal that a new point is starting.  To avoid the
        # false-positive mid-rally splits that made the previous
        # force-close implementation (reverted 6eb789f) drop real
        # rallies, we only act on a toss when the current burst also
        # shows evidence of being between points: the last net crossing
        # must be ≥ `serve_soft_quiet_frames` in the past.  Mid-rally
        # there is a crossing every ~0.5s, so a 1s quiet window
        # distinguishes "between points" from "mid-rally false rise".
        # Pure ball signal so it stays reliable when pose / player
        # detection misses the server (far side, occluded).
        self.serve_enabled = bool(serve_enabled)
        # Rise threshold is interpolated per-toss between near and far
        # values based on the toss origin's image-y position between
        # the two baselines (projected through H_real_to_img).  A toss
        # from the near baseline covers many more image pixels than an
        # identical-height toss from the far baseline because of
        # perspective compression; a single global threshold either
        # misses far tosses (too large) or false-fires on near noise
        # (too small).
        self.serve_toss_rise_px_near = max(1.0, float(serve_toss_rise_px_near))
        self.serve_toss_rise_px_far = max(1.0, float(serve_toss_rise_px_far))
        self.near_baseline_y_px = (
            float(near_baseline_y_px) if near_baseline_y_px is not None else None)
        self.far_baseline_y_px = (
            float(far_baseline_y_px) if far_baseline_y_px is not None else None)
        self.serve_toss_min_frames = max(1, int(serve_toss_min_frames))
        self.serve_toss_max_horiz_ratio = max(0.0, float(serve_toss_max_horiz_ratio))
        self.serve_baseline_margin_m = max(0.0, float(serve_baseline_margin_m))
        self.serve_pre_static_max_dy_px = max(0.0, float(serve_pre_static_max_dy_px))
        self.serve_suppress_frames = max(1, int(serve_suppress_frames))
        self.serve_history_frames = max(
            self.serve_toss_min_frames + 3, int(serve_history_frames))
        self.serve_soft_quiet_frames = max(0, int(serve_soft_quiet_frames))
        self.H_img_to_real = (
            np.asarray(H_img_to_real, dtype=np.float64)
            if H_img_to_real is not None else None)
        self.court_length_m = float(court_length_m)

        self._activity_start: Optional[int] = None
        self._crossings = 0
        self._last_side: Optional[int] = None
        self._last_crossing_frame: Optional[int] = None
        self._silent = 0
        self._activity_frames = 0
        # Last two observed ball-y samples — used to decide whether the
        # current silence is likely a lob (ball heading up or high when
        # detection was lost) and the silence window should be stretched.
        self._last_ball_frame: Optional[int] = None
        self._last_ball_x: Optional[float] = None
        self._last_ball_y: Optional[float] = None
        self._prev_ball_frame: Optional[int] = None
        self._prev_ball_x: Optional[float] = None
        self._prev_ball_y: Optional[float] = None
        self._discontinuity_closes = 0
        # Ball (frame, x, y) ring buffer for serve toss detection.
        self._ball_hist: Deque[tuple[int, float, float]] = deque()
        self._last_serve_frame: Optional[int] = None
        # Counters for the run summary: tosses we saw vs. tosses we
        # acted on (the rest were suppressed by the soft-quiet gate).
        self._serve_toss_detected = 0
        self._serve_toss_acted = 0
        self.rallies: list[Rally] = []

    # ------------------------------------------------------------------
    def observe(self, frame_idx: int, cand_xys: list, champion) -> None:
        has_ball = bool(cand_xys)
        ball_y: Optional[float] = None
        ball_x: Optional[float] = None

        if has_ball:
            # Prefer the tracker's current-frame ball position (cleaner),
            # fall back to the first raw candidate.
            if (champion is not None and champion.pts
                    and champion.last_det_frame == frame_idx):
                ball_x = float(champion.pts[-1].x)
                ball_y = float(champion.pts[-1].y)
            elif cand_xys:
                ball_x = float(cand_xys[0][0])
                ball_y = float(cand_xys[0][1])

            # Serve detection runs BEFORE activity_start bookkeeping so
            # a confirmed toss can close the current burst and start a
            # fresh one at the toss frame — but only if the burst looks
            # like it is genuinely between points (soft-quiet gate).
            toss_start: Optional[int] = None
            if (self.serve_enabled
                    and ball_x is not None and ball_y is not None):
                toss_start = self._detect_serve(frame_idx, ball_x, ball_y)
            if toss_start is not None:
                self._serve_toss_detected += 1
                # Soft gate: during a mid-rally burst, net crossings
                # happen every ~0.5s.  If the most recent *actual*
                # crossing was within `serve_soft_quiet_frames` of
                # this toss, the toss is almost certainly a false
                # rise (lob / high shot) rather than a real
                # between-points toss — act on it and we would slice
                # a legit rally in two, both halves likely failing
                # min_crossings.  A burst with zero crossings can not
                # become a rally anyway (min_crossings > 0), so we
                # pass the toss through — splitting a warm-up burst
                # costs nothing and lets the new point start cleanly.
                quiet_ok = (
                    self._activity_start is None
                    or self._crossings == 0
                    or (frame_idx - self._last_crossing_frame)
                        >= self.serve_soft_quiet_frames)
                self._last_serve_frame = frame_idx
                if quiet_ok:
                    self._serve_toss_acted += 1
                    if (self._activity_start is not None
                            and self._activity_start < toss_start):
                        self._close_at(toss_start - 1)
                        self._reset()
                    # Open a fresh burst at the toss frame, backfilling
                    # activity_frames from the ball history so the
                    # density gate still sees the toss window.
                    self._activity_start = toss_start
                    self._activity_frames = sum(
                        1 for (f, _x, _y) in self._ball_hist if f >= toss_start)
                    self._silent = 0
                    self._crossings = 0
                    self._last_side = None
                    self._last_crossing_frame = toss_start

            # Discontinuity close: the ball just teleported (big jump)
            # AND direction flipped AND we are in a quiet stretch
            # since the last net crossing.  In a real rally, bounces
            # and strokes are tracker-continuous (jump per frame
            # stays under ~80 px), and lobs flip direction but also
            # stay continuous — so only a between-point ball pickup
            # / re-toss, where the ball appears at a new position
            # in the image, produces large-jump + direction-flip.
            # Gate additionally on the gap since last observation so
            # long detector-gap re-acquisitions (handled elsewhere
            # by silence thresholds) do not mis-fire.
            if (self._activity_start is not None
                    and self.discontinuity_jump_px > 0
                    and ball_x is not None and ball_y is not None
                    and self._last_ball_frame is not None
                    and self._last_ball_x is not None
                    and self._prev_ball_x is not None
                    and self._last_ball_y is not None
                    and self._prev_ball_y is not None):
                gap = frame_idx - self._last_ball_frame
                if 1 <= gap <= 10:
                    dx_new = ball_x - self._last_ball_x
                    dy_new = ball_y - self._last_ball_y
                    jump = (dx_new * dx_new + dy_new * dy_new) ** 0.5
                    dx_old = self._last_ball_x - self._prev_ball_x
                    dy_old = self._last_ball_y - self._prev_ball_y
                    dot = dx_new * dx_old + dy_new * dy_old
                    time_since_cross = (
                        frame_idx - self._last_crossing_frame
                        if self._last_crossing_frame is not None
                        else 10 ** 9)
                    if (jump > self.discontinuity_jump_px
                            and dot < 0
                            and time_since_cross
                                >= self.discontinuity_no_cross_frames):
                        self._discontinuity_closes += 1
                        self._close_at(self._last_ball_frame)
                        self._reset()
                        return

            if self._activity_start is None:
                self._activity_start = frame_idx
                self._crossings = 0
                self._last_side = None
                self._last_crossing_frame = frame_idx
                self._activity_frames = 0
            self._silent = 0
            self._activity_frames += 1

            if ball_y is not None:
                side = -1 if ball_y < self.net_y_px else 1
                if self._last_side is not None and side != self._last_side:
                    self._crossings += 1
                    self._last_crossing_frame = frame_idx
                self._last_side = side
                # Record the two most recent ball (x, y) samples — y
                # for lob detection at the start of silence, x for
                # the discontinuity gate above.
                self._prev_ball_frame = self._last_ball_frame
                self._prev_ball_y = self._last_ball_y
                self._prev_ball_x = self._last_ball_x
                self._last_ball_frame = frame_idx
                self._last_ball_y = ball_y
                if ball_x is not None:
                    self._last_ball_x = ball_x
        else:
            if self._activity_start is not None:
                self._silent += 1
                if self._silent >= self._effective_silence_thresh():
                    self._close_at(frame_idx - self._silent)
                    self._reset()
                    return

        # Crossing-silence check: runs EVERY frame while an activity
        # burst is active, not only on has_ball frames.  If ball
        # detection is spotty between real rallies, the growing gap
        # since last_crossing_frame is what fires — otherwise spurious
        # brief ball appearances on the same side would mask it.
        if (self._activity_start is not None
                and self.crossing_silence_thresh > 0
                and self._crossings >= 1
                and self._last_crossing_frame is not None
                and (frame_idx - self._last_crossing_frame)
                    >= self.crossing_silence_thresh):
            last_cross = self._last_crossing_frame
            self._close_at(last_cross)
            if has_ball:
                # Continue tracking from this frame so play resuming
                # immediately can start a fresh rally.
                self._activity_start = frame_idx
                self._crossings = 0
                self._last_side = (
                    None if ball_y is None
                    else (-1 if ball_y < self.net_y_px else 1))
                self._last_crossing_frame = frame_idx
                self._activity_frames = 1
                self._silent = 0
            else:
                self._reset()

    def finalize(self, last_frame_idx: int) -> None:
        """Close any still-open activity when the video ends."""
        if self._activity_start is not None:
            self._close_at(last_frame_idx)
            self._reset()

    # ------------------------------------------------------------------
    def _close_at(self, activity_end_frame: int) -> None:
        """Emit a Rally ending at the given activity end frame (before
        post_roll is applied).  Does NOT reset state — callers are
        responsible for calling _reset() or setting up a new burst."""
        if self._crossings < self.min_crossings:
            return
        # Density gate: reject bursts where the ball was only visible
        # for a tiny fraction of the window (spotty detection, usually
        # warm-up / coaching mix).  Denom is the span from activity
        # start to activity end in frames.
        if self.min_activity_density > 0.0:
            span = activity_end_frame - (self._activity_start or activity_end_frame) + 1
            density = self._activity_frames / max(span, 1)
            if density < self.min_activity_density:
                return
        start = max(1, (self._activity_start or activity_end_frame) - self.pre_roll)
        # Don't let pre_roll back up into the previous rally's padded
        # tail — that would produce overlapping rallies.
        if self.rallies:
            start = max(start, self.rallies[-1].end_frame + 1)
        end = min(self.total, activity_end_frame + self.post_roll)
        if end >= start:
            self.rallies.append(Rally(
                idx=len(self.rallies),
                start_frame=start,
                end_frame=end,
                n_events=self._activity_frames,
                net_crossings=self._crossings,
            ))

    def _detect_serve(self, frame_idx: int, x: float, y: float) -> Optional[int]:
        """Look for a serve toss in the recent ball trajectory.

        Append (frame, x, y) to the history, then check for a
        "hold → rise → apex → descent" pattern:
        - apex (minimum y) sits inside the buffer, followed by ≥2
          descending samples — apex is confirmed, not a mid-rise
          reading
        - backward from the apex, find the toss origin: the last point
          preceded by ≥2 quasi-static frames
          (|dy/df| ≤ serve_pre_static_max_dy_px).  Distinguishes a real
          toss release from any rally rise — rally balls never come
          to rest
        - rise origin→apex ≥ `serve_toss_rise_px`, over ≥
          `serve_toss_min_frames`
        - horizontal drift ≤ `serve_toss_max_horiz_ratio × rise` —
          filters cross-court lobs
        - origin projects within `serve_baseline_margin_m` of either
          baseline (z=0 projection is fine at the release: ball is
          near hand-height, projection error small)

        Returns the origin's frame index when a serve is confirmed,
        else None.  Self-suppresses for `serve_suppress_frames` after
        a hit so the same toss does not re-fire as the apex slides
        through the buffer.
        """
        self._ball_hist.append((frame_idx, x, y))
        while (self._ball_hist
               and (frame_idx - self._ball_hist[0][0])
                   > self.serve_history_frames):
            self._ball_hist.popleft()

        if (self._last_serve_frame is not None
                and (frame_idx - self._last_serve_frame)
                    < self.serve_suppress_frames):
            return None
        if len(self._ball_hist) < self.serve_toss_min_frames + 3:
            return None

        # Apex = minimum y in the buffer.  Relaxed from strict monotonic
        # descent: the 2 post-apex samples must never dip BELOW apex
        # height (that would mean apex wasn't actually the peak), AND
        # at least one of them must be STRICTLY above apex (confirm
        # real descent, not a 3-frame plateau that never fell).
        # WASB detection noise can leave the image y flat for a frame
        # around the peak, so an apex-height hold at +1 is fine as
        # long as +2 shows the ball coming down.
        hist = list(self._ball_hist)
        apex_i = min(range(len(hist)), key=lambda i: hist[i][2])
        if apex_i == 0 or apex_i > len(hist) - 3:
            return None
        apex_y = hist[apex_i][2]
        post1_y = hist[apex_i + 1][2]
        post2_y = hist[apex_i + 2][2]
        if post1_y < apex_y or post2_y < apex_y:
            return None
        if post1_y == apex_y and post2_y == apex_y:
            return None

        # Locate the start of the rise: scan backward from the apex for
        # the last point where the ball was essentially at rest
        # (per-frame |dy| <= serve_pre_static_max_dy_px).  That point
        # is the toss origin — the moment the ball was released.
        # Require at least 2 static frames before the release so we do
        # not confuse a fast-moving rally-shot rise with a toss.
        static_max = self.serve_pre_static_max_dy_px
        origin_i: Optional[int] = None
        static_run = 0
        for i in range(apex_i - 1, 0, -1):
            df = max(1, hist[i][0] - hist[i - 1][0])
            dy = abs(hist[i][2] - hist[i - 1][2]) / df
            if dy <= static_max:
                static_run += 1
                if static_run >= 2:
                    origin_i = i
                    break
            else:
                static_run = 0
        if origin_i is None:
            return None

        origin_f, origin_x, origin_y = hist[origin_i]
        apex_f, apex_x, apex_y = hist[apex_i]
        rise_px = origin_y - apex_y
        duration = apex_f - origin_f
        horiz = abs(apex_x - origin_x)
        # Adaptive rise threshold: interpolate between the near- and
        # far-baseline settings based on the toss origin's image-y
        # between the two baseline projections.  Near baseline (large
        # image y) needs a large rise; far baseline (small image y)
        # needs a small one because perspective squashes the toss
        # image extent.  Falls back to the near value when baseline
        # pixel positions were not supplied at construction.
        if (self.near_baseline_y_px is not None
                and self.far_baseline_y_px is not None):
            span = self.near_baseline_y_px - self.far_baseline_y_px
            if span > 1e-6:
                t = max(0.0, min(1.0,
                    (origin_y - self.far_baseline_y_px) / span))
                rise_threshold = (
                    self.serve_toss_rise_px_far
                    + t * (self.serve_toss_rise_px_near
                           - self.serve_toss_rise_px_far))
            else:
                rise_threshold = self.serve_toss_rise_px_near
        else:
            rise_threshold = self.serve_toss_rise_px_near
        if rise_px < rise_threshold:
            return None
        if duration < self.serve_toss_min_frames:
            return None
        if horiz > self.serve_toss_max_horiz_ratio * rise_px:
            return None

        if self.H_img_to_real is not None and self.serve_baseline_margin_m > 0:
            p = self.H_img_to_real @ np.array([origin_x, origin_y, 1.0])
            if abs(p[2]) < 1e-9:
                return None
            court_y = float(p[1] / p[2])
            near_near = abs(court_y) <= self.serve_baseline_margin_m
            near_far = (
                abs(court_y - self.court_length_m)
                <= self.serve_baseline_margin_m)
            if not (near_near or near_far):
                return None

        return int(origin_f)

    def _effective_silence_thresh(self) -> int:
        """Silence threshold, stretched by `lob_silence_multiplier` when
        the last observed ball looked like a lob (heading upward or
        already above the net line).

        Upward motion is decided from the last two observed y samples —
        image y decreases toward the top of the frame, so `dy/df < 0`
        means the ball was rising.  Far-side position alone (y <
        net_y_px) is a weaker signal on its own (normal over-the-net
        shots also register there), so we only treat it as a lob
        indicator when the ball is ALSO not clearly descending.
        """
        if self.lob_silence_multiplier <= 1.0:
            return self.silence_thresh
        if self._last_ball_y is None:
            return self.silence_thresh
        going_up = False
        nearly_flat = True
        if (self._prev_ball_y is not None
                and self._prev_ball_frame is not None
                and self._last_ball_frame is not None):
            df = max(1, self._last_ball_frame - self._prev_ball_frame)
            vy = (self._last_ball_y - self._prev_ball_y) / df
            going_up = vy <= -self.lob_up_speed_px
            nearly_flat = abs(vy) < self.lob_up_speed_px
        was_high = self._last_ball_y < self.net_y_px
        if going_up or (was_high and nearly_flat):
            return int(round(self.silence_thresh * self.lob_silence_multiplier))
        return self.silence_thresh

    def _reset(self) -> None:
        self._activity_start = None
        self._crossings = 0
        self._last_side = None
        self._last_crossing_frame = None
        self._silent = 0
        self._activity_frames = 0
        self._last_ball_frame = None
        self._last_ball_x = None
        self._last_ball_y = None
        self._prev_ball_frame = None
        self._prev_ball_x = None
        self._prev_ball_y = None


def detect_rallies(
    event_frames: Iterable[int],
    fps: float,
    *,
    gap_seconds: float = 3.0,
    min_events: int = 3,
    pre_roll_frames: int = 30,
    post_roll_frames: int = 30,
    total_frames: int = 0,
) -> list[Rally]:
    """Group event frames into rallies.

    `event_frames` is the union of any signal that indicates the ball is
    in play: validated-track point frames, bounce frames, stroke-event
    frames.  Duplicates are fine; we de-dup and sort internally.
    """
    sorted_frames = sorted({int(f) for f in event_frames if f and f > 0})
    if not sorted_frames:
        return []

    gap_frames = max(1, int(round(gap_seconds * max(fps, 1.0))))
    groups: list[list[int]] = [[sorted_frames[0]]]
    for f in sorted_frames[1:]:
        if f - groups[-1][-1] <= gap_frames:
            groups[-1].append(f)
        else:
            groups.append([f])

    out: list[Rally] = []
    kept = 0
    for grp in groups:
        if len(grp) < min_events:
            continue
        s = max(1, grp[0] - pre_roll_frames)
        e = grp[-1] + post_roll_frames
        if total_frames > 0:
            e = min(e, total_frames)
        if e <= s:
            continue
        out.append(Rally(idx=kept, start_frame=s, end_frame=e, n_events=len(grp)))
        kept += 1
    return out


def validate_rallies(
    rallies: list[Rally],
    tracks: list,
    stroke_events: list,
    net_y_px: float,
    *,
    min_net_crossings: int = 0,
    min_stroke_events: int = 0,
    stroke_labels: tuple = ("forehand", "backhand", "serve"),
) -> list[Rally]:
    """Drop candidate rallies that don't show real cross-court play.

    Picking up balls / warm-up dribbling generates enough ball track
    points and bounces to fool the gap-based grouping in detect_rallies.
    But the ball never crosses the net and no real strokes fire, so a
    simple two-signal filter removes them:

    Keep a rally if AT LEAST ONE of:
      * the ball crosses the net line ≥ min_net_crossings times within
        [start_frame, end_frame] (any validated track counts)
      * there are ≥ min_stroke_events non-neutral RNN stroke events in
        the rally window

    Setting both thresholds to 0 disables filtering.  Rallies that pass
    are renumbered so idx is consecutive in the returned list.
    """
    if min_net_crossings <= 0 and min_stroke_events <= 0:
        return rallies

    label_set = set(stroke_labels)
    kept: list[Rally] = []
    dropped = 0
    for r in rallies:
        n_cross = 0
        if min_net_crossings > 0:
            for t in tracks:
                prev_sign = None
                for p in t.pts:
                    if not (r.start_frame <= p.frame <= r.end_frame):
                        continue
                    # In image coords, smaller y is farther from camera,
                    # so y < net_y_px means the ball is on the far side.
                    sign = -1 if p.y < net_y_px else 1
                    if prev_sign is not None and sign != prev_sign:
                        n_cross += 1
                    prev_sign = sign

        n_strokes = 0
        if min_stroke_events > 0:
            n_strokes = sum(
                1 for ev in stroke_events
                if r.start_frame <= ev.frame <= r.end_frame
                and getattr(ev, "label", None) in label_set
            )

        net_ok = min_net_crossings > 0 and n_cross >= min_net_crossings
        strokes_ok = min_stroke_events > 0 and n_strokes >= min_stroke_events
        # Keep if ANY enabled filter passes.  Net-crossing is the strong
        # signal (warm-up / ball pickup stays on one side); the stroke
        # count is a safety net for legit rallies where tracking was
        # patchy but the RNN still fired.
        if net_ok or strokes_ok:
            kept.append(Rally(
                idx=len(kept),
                start_frame=r.start_frame,
                end_frame=r.end_frame,
                n_events=r.n_events,
                net_crossings=n_cross,
                n_strokes=n_strokes,
            ))
        else:
            dropped += 1
    return kept


def merge_close_rallies(
    rallies: list[Rally],
    bounces_court: list,
    fps: float,
    *,
    max_gap_seconds: float = 5.0,
) -> tuple[list[Rally], int]:
    """Merge adjacent rallies if the gap between them is short AND
    contains no bounces.

    A mid-rally tracking drop-out (high lob goes out of frame, fast deep
    ball that WASB misses for a few seconds) produces a gap that looks
    identical to a between-point gap at the detector level.  The
    distinguishing signal is physical: real between-point gaps almost
    always contain pickup / dribbling bounces, whereas a mid-rally
    silence doesn't — the ball is airborne, out of frame, or simply
    missed by the detector, and hasn't touched the ground.

    `max_gap_seconds <= 0` disables the pass.
    """
    if max_gap_seconds <= 0 or len(rallies) < 2:
        return list(rallies), 0
    max_gap_frames = int(round(max_gap_seconds * max(fps, 1.0)))
    if max_gap_frames <= 0:
        return list(rallies), 0

    def _clone(r: Rally, idx: int) -> Rally:
        return Rally(
            idx=idx, start_frame=r.start_frame, end_frame=r.end_frame,
            n_events=r.n_events, net_crossings=r.net_crossings,
            n_strokes=r.n_strokes,
        )

    out: list[Rally] = [_clone(rallies[0], 0)]
    n_merged = 0
    for r in rallies[1:]:
        prev = out[-1]
        gap = r.start_frame - prev.end_frame - 1
        if 0 <= gap <= max_gap_frames:
            has_bounce_in_gap = any(
                prev.end_frame < f < r.start_frame
                for (_x, _y, f) in bounces_court
            )
            if not has_bounce_in_gap:
                prev.end_frame = r.end_frame
                prev.n_events += r.n_events
                prev.net_crossings += r.net_crossings
                prev.n_strokes += r.n_strokes
                n_merged += 1
                continue
        out.append(_clone(r, len(out)))
    return out, n_merged


def select_clip_rallies(
    rallies: list[Rally],
    fps: float,
    *,
    min_net_crossings: int = 0,
    min_duration_seconds: float = 0.0,
) -> list[Rally]:
    """Tighter subset of rallies for the highlight cut.

    Uses the `net_crossings` field that `validate_rallies` cached on
    each Rally, so this runs in O(n) without re-walking ball tracks.
    Filters cascade as AND: a rally must pass every enabled threshold.
    Both at 0 (default) returns the input list unchanged.
    """
    if min_net_crossings <= 0 and min_duration_seconds <= 0:
        return list(rallies)
    min_dur_frames = int(round(max(0.0, min_duration_seconds) * max(fps, 1.0)))
    out: list[Rally] = []
    for r in rallies:
        if min_net_crossings > 0 and r.net_crossings < min_net_crossings:
            continue
        if min_dur_frames > 0 and (r.end_frame - r.start_frame + 1) < min_dur_frames:
            continue
        out.append(Rally(
            idx=len(out),
            start_frame=r.start_frame,
            end_frame=r.end_frame,
            n_events=r.n_events,
            net_crossings=r.net_crossings,
            n_strokes=r.n_strokes,
        ))
    return out


def write_rally_video(
    src_video: str,
    rallies: list[Rally],
    dst_video: str,
    *,
    separator_seconds: float = 1.0,
    extra_tail_seconds: float = 0.0,
    fourcc: str = "mp4v",
) -> None:
    """Copy each rally's frames from `src_video` to `dst_video`, with a
    short black 'Rally N' title between them.

    `src_video` should be the already-rendered annotated output — the
    cut video inherits its overlays (trails, bboxes, stroke labels,
    minimap) without re-running analysis.

    `extra_tail_seconds` tacks that many extra frames past each
    rally's `end_frame` onto the cut video so the trailing bounces
    /ball-settling after the final stroke stay visible.  The JSON
    rally boundary is not touched — this only pads the visual cut.
    Successive rallies may overlap in the cut (rally N's tail can
    contain frames that rally N+1 later replays from its start);
    that is intentional and avoids merging distinct rallies just to
    keep the tail.
    """
    if not rallies:
        print("[rally] no rallies detected; skipping cut video", flush=True)
        return
    cap = cv2.VideoCapture(src_video)
    if not cap.isOpened():
        raise SystemExit("cannot open " + src_video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(os.path.dirname(dst_video) or ".", exist_ok=True)
    writer = cv2.VideoWriter(dst_video, cv2.VideoWriter_fourcc(*fourcc),
                             fps, (W, H))
    if not writer.isOpened():
        cap.release()
        raise SystemExit("cannot open writer for " + dst_video)

    sep_frames = max(1, int(round(separator_seconds * fps)))
    tail_frames = max(0, int(round(extra_tail_seconds * fps)))
    blank = np.zeros((H, W, 3), dtype=np.uint8)

    # Build frame ranges needed per rally (sorted by start)
    clips = []
    for r in rallies:
        clip_end = r.end_frame + tail_frames
        if total_src_frames > 0:
            clip_end = min(clip_end, total_src_frames - 1)
        clips.append((r, r.start_frame, clip_end))

    # Sequential read — no seeking, accurate on all codecs
    max_frame_needed = max(ce for _, _, ce in clips)
    clip_idx = 0
    fi = 0
    while fi <= max_frame_needed and clip_idx < len(clips):
        ret, frame = cap.read()
        if not ret:
            break
        # Write title card just before we reach the next rally
        r, clip_start, clip_end = clips[clip_idx]
        if fi == clip_start:
            text = "Rally %d  (%.1fs)" % (
                r.idx + 1, (r.end_frame - r.start_frame + 1) / max(fps, 1.0))
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 4)
            title = blank.copy()
            cv2.putText(title, text, ((W - tw) // 2, (H + th) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4)
            for _ in range(sep_frames):
                writer.write(title)
        if clip_start <= fi <= clip_end:
            writer.write(frame)
        if fi >= clip_end:
            clip_idx += 1
        fi += 1
    cap.release()
    writer.release()


def save_rally_json(rallies: list[Rally], path: str, *, fps: float = 0.0) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "fps": fps,
        "count": len(rallies),
        "rallies": [asdict(r) for r in rallies],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
