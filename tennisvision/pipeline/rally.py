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
from dataclasses import asdict, dataclass
from typing import Iterable

import cv2
import numpy as np


@dataclass
class Rally:
    idx: int
    start_frame: int
    end_frame: int
    n_events: int
    # Populated by validate_rallies(); useful for downstream selection
    # (e.g. a stricter net-crossing threshold for the clip video).
    net_crossings: int = 0
    n_strokes: int = 0


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
    fourcc: str = "mp4v",
) -> None:
    """Copy each rally's frames from `src_video` to `dst_video`, with a
    short black 'Rally N' title between them.

    `src_video` should be the already-rendered annotated output — the
    cut video inherits its overlays (trails, bboxes, stroke labels,
    minimap) without re-running analysis.
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

    os.makedirs(os.path.dirname(dst_video) or ".", exist_ok=True)
    writer = cv2.VideoWriter(dst_video, cv2.VideoWriter_fourcc(*fourcc),
                             fps, (W, H))
    if not writer.isOpened():
        cap.release()
        raise SystemExit("cannot open writer for " + dst_video)

    sep_frames = max(1, int(round(separator_seconds * fps)))
    blank = np.zeros((H, W, 3), dtype=np.uint8)

    for r in rallies:
        text = "Rally %d  (%.1fs)" % (
            r.idx + 1, (r.end_frame - r.start_frame + 1) / max(fps, 1.0))
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 4)
        title = blank.copy()
        cv2.putText(title, text, ((W - tw) // 2, (H + th) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4)
        for _ in range(sep_frames):
            writer.write(title)

        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, r.start_frame - 1))
        n_needed = r.end_frame - r.start_frame + 1
        n_read = 0
        while n_read < n_needed:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
            n_read += 1
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
