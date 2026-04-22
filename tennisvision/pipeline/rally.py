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
