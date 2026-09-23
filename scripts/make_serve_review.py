"""Generate serve_review.mp4 from raw + filtered serve events.

Shows ALL raw serve clips, with a green checkmark for kept events
and a red X for filtered events.

Usage:
    python scripts/make_serve_review.py \
        --video /Users/justinsong/WorkSpace/solo/samples/sample3.mp4 \
        --raw results/sample3/serve_events_raw.json \
        --filtered results/sample3/serve_events.json \
        --out results/sample3/serve_review.mp4
"""

import argparse
import json

import cv2
import numpy as np


def _draw_check(frame, x, y, size, color, thickness):
    """Draw a checkmark at (x, y)."""
    pts = np.array([
        [x, y],
        [x + size // 3, y + size // 3],
        [x + size, y - size // 2],
    ], dtype=np.int32)
    cv2.polylines(frame, [pts], False, color, thickness, cv2.LINE_AA)


def _draw_cross(frame, x, y, size, color, thickness):
    """Draw an X at (x, y)."""
    cv2.line(frame, (x, y - size // 2), (x + size, y + size // 2),
             color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x, y + size // 2), (x + size, y - size // 2),
             color, thickness, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--raw", required=True, help="Raw serve_events JSON (all events)")
    parser.add_argument("--filtered", required=True, help="Filtered serve_events JSON")
    parser.add_argument("--out", required=True)
    parser.add_argument("--ball", default=None, help="ball_positions.json")
    parser.add_argument("--pad", type=float, default=0.5, help="Seconds before/after each event")
    args = parser.parse_args()

    raw = json.load(open(args.raw))
    filt = json.load(open(args.filtered))
    raw_events = raw["serve_events"]
    filt_events = filt["serve_events"]
    fps = raw["stats"]["fps"]

    # Load ball positions
    ball_pos = {}
    if args.ball:
        bd = json.load(open(args.ball))
        for k, v in bd.get("detected", {}).items():
            ball_pos[int(k)] = v
        for k, v in bd.get("predicted", {}).items():
            ball_pos.setdefault(int(k), v)

    # Build set of kept events by (start_frame, slot)
    kept = {(e["start_frame"], e["slot"]) for e in filt_events}

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit("cannot open " + args.video)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pad = int(round(args.pad * fps))

    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not writer.isOpened():
        cap.release()
        raise SystemExit("cannot open writer for " + args.out)

    clips = []
    for i, ev in enumerate(raw_events):
        cs = max(0, ev["start_frame"] - pad)
        ce = min(total - 1, ev["end_frame"] + pad)
        is_kept = (ev["start_frame"], ev["slot"]) in kept
        clips.append((i, ev, cs, ce, is_kept))

    max_needed = max(ce for _, _, _, ce, _ in clips)
    clip_idx = 0
    fi = 0
    while fi <= max_needed and clip_idx < len(clips):
        ret, frame = cap.read()
        if not ret:
            break
        idx, ev, clip_start, clip_end, is_kept = clips[clip_idx]
        if clip_start <= fi <= clip_end:
            label = "Serve %d/%d  %s  conf=%.2f  t=%.1fs" % (
                idx + 1, len(raw_events), ev["slot_name"], ev["conf"], fi / fps)
            cv2.putText(frame, label, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            # Draw ball position
            bp = ball_pos.get(fi)
            if bp is not None:
                bx, by = int(bp[0]), int(bp[1])
                cv2.circle(frame, (bx, by), 10, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(frame, (bx, by), 3, (0, 255, 255), -1, cv2.LINE_AA)
            if is_kept:
                _draw_check(frame, W - 120, 40, 60, (0, 255, 0), 6)
            else:
                _draw_cross(frame, W - 120, 40, 60, (0, 0, 255), 6)
            writer.write(frame)
            if fi == clip_end:
                clip_idx += 1
        fi += 1

    cap.release()
    writer.release()
    n_kept = sum(1 for *_, k in clips if k)
    n_filt = len(clips) - n_kept
    print(f"Saved: {args.out} ({len(clips)} clips, {n_kept} kept, {n_filt} filtered)")


if __name__ == "__main__":
    main()
