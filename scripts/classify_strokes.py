#!/usr/bin/env python3
"""Classify player strokes from pre-extracted keypoints.

Runs GRU stroke classifier on keypoints.json, using the 4-slot spatial
mapping to assign detections to court positions. Outputs per-frame
classification events.

This is an auxiliary standalone script. Requires keypoints.json and calib.json
from the main pipeline (Steps 1-2).

Usage:
    python -u scripts/classify_strokes.py \
        --keypoints results/sample2/keypoints.json \
        --calib results/sample2/calib.json \
        --gru weights/stroke_gru_v4_best.pt \
        --out results/sample2/stroke_events.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tennisvision.action.slot_mapper import SlotMapper, SLOT_NAMES
from tennisvision.action.gru_classifier import StrokeGRU, LABELS


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--keypoints", required=True)
    parser.add_argument("--calib", required=True)
    parser.add_argument("--gru", default="weights/stroke_gru_v4_best.pt")
    parser.add_argument("--out", required=True)
    parser.add_argument("--speed-threshold", type=float, default=0.15,
                        help="Wrist speed threshold to activate GRU")
    parser.add_argument("--min-conf", type=float, default=0.5,
                        help="Minimum confidence to emit event")
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--gap-tolerance", type=int, default=5)
    args = parser.parse_args()

    # Load homography
    with open(args.calib) as f:
        calib = json.load(f)
    H_img_to_real = np.array(calib["H_img_to_real"], dtype=np.float64)
    mapper = SlotMapper(H_img_to_real)

    # Load keypoints
    print(f"Loading keypoints: {args.keypoints}")
    with open(args.keypoints) as f:
        data = json.load(f)
    fps = data["fps"]
    print(f"  {data['total_frames']} frames, fps={fps:.1f}")

    # Load GRU
    print(f"Loading GRU: {args.gru}")
    gru = StrokeGRU(n_classes=4)
    gru.load_state_dict(torch.load(args.gru, map_location="cpu"))
    gru.eval()

    SEQ_LEN = args.seq_len

    # Per-slot state
    kp_buffers = {s: [] for s in range(4)}
    prev_kps = {s: None for s in range(4)}
    active = {s: False for s in range(4)}
    gap_count = {s: 0 for s in range(4)}

    all_events = []
    gru_count = 0

    for fd in data["frames"]:
        fi = fd["frame"]
        slot_dets = mapper.assign(fd["detections"])

        for slot in range(4):
            if slot not in slot_dets:
                if active[slot]:
                    gap_count[slot] += 1
                    if gap_count[slot] > args.gap_tolerance:
                        active[slot] = False
                continue

            det = slot_dets[slot]
            kp = np.stack([det["kp_norm_x"], det["kp_norm_y"]], axis=1).astype(np.float32)

            kp_buffers[slot].append(kp)
            if len(kp_buffers[slot]) > SEQ_LEN + 30:
                kp_buffers[slot] = kp_buffers[slot][-(SEQ_LEN + 30):]

            # Wrist speed trigger
            speed = 0.0
            if prev_kps[slot] is not None:
                for wi in [9, 10]:
                    speed += np.sqrt((kp[wi, 0] - prev_kps[slot][wi, 0]) ** 2 +
                                     (kp[wi, 1] - prev_kps[slot][wi, 1]) ** 2)
            prev_kps[slot] = kp

            if speed > args.speed_threshold:
                active[slot] = True
                gap_count[slot] = 0
            elif active[slot]:
                gap_count[slot] += 1
                if gap_count[slot] > args.gap_tolerance:
                    active[slot] = False

            # GRU inference during active period
            if active[slot] and len(kp_buffers[slot]) >= SEQ_LEN:
                feat = np.array(
                    [k[:, :2].reshape(-1) for k in kp_buffers[slot][-SEQ_LEN:]],
                    dtype=np.float32,
                )
                with torch.no_grad():
                    probs = torch.softmax(
                        gru(torch.FloatTensor(feat).unsqueeze(0)), dim=1
                    )[0].numpy()
                gru_count += 1
                pred = int(np.argmax(probs))
                conf = float(probs[pred])

                if conf >= args.min_conf:
                    all_events.append({
                        "frame": fi,
                        "time": round(fi / fps, 2),
                        "slot": slot,
                        "slot_name": SLOT_NAMES[slot],
                        "label": LABELS[pred],
                        "conf": round(conf, 3),
                        "probs": {LABELS[i]: round(float(probs[i]), 3)
                                  for i in range(4)},
                    })

    # Print summary
    print(f"\nGRU invocations: {gru_count}")
    print(f"Events (conf >= {args.min_conf}): {len(all_events)}")
    for label in ["backhand", "forehand", "serve", "background"]:
        n = sum(1 for e in all_events if e["label"] == label)
        if n > 0:
            print(f"  {label}: {n}")

    # Per-slot breakdown
    for slot in range(4):
        slot_events = [e for e in all_events if e["slot"] == slot]
        if slot_events:
            print(f"  {SLOT_NAMES[slot]}: {len(slot_events)} events")

    # Save
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    output = {
        "params": {
            "speed_threshold": args.speed_threshold,
            "min_conf": args.min_conf,
            "seq_len": args.seq_len,
        },
        "stats": {
            "total_frames": data["total_frames"],
            "fps": fps,
            "gru_invocations": gru_count,
            "total_events": len(all_events),
        },
        "events": all_events,
    }
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
