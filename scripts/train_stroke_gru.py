#!/usr/bin/env python3
"""Train GRU stroke classifier on THETIS dataset (v6).

Downloads THETIS RGB videos, extracts keypoints with YOLO-pose,
and trains a 4-class GRU (backhand / forehand / serve / background).

v6 changes:
  - Unified 15fps sampling (THETIS ~17-19fps, match videos 30fps)
  - Peak = last wrist-above-head frame (racket hand contact point)
  - Asymmetric window: 9 frames before peak + peak + 5 frames after = 15 frames (1s @15fps)
  - v4-style normalization (x/bw, y/bh, stretch to square)

Usage:
    python -u scripts/train_stroke_gru.py \
        --out weights/stroke_gru_v6_best.pt --epochs 30

    # Resume from cached keypoints
    python -u scripts/train_stroke_gru.py \
        --keypoints /path/to/thetis_keypoints.pkl \
        --out weights/stroke_gru_v6_best.pt
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
import urllib.request

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tennisvision.action.gru_classifier import StrokeGRU

# THETIS 7 core categories → 3-class labels (background is synthetic)
THETIS_CATEGORIES = {
    'flat_service': 'serve',
    'kick_service': 'serve',
    'slice_service': 'serve',
    'forehand_flat': 'forehand',
    'forehand_openstands': 'forehand',
    'backhand': 'backhand',
    'backhand2hands': 'backhand',
}

LABEL2IDX = {'backhand': 0, 'forehand': 1, 'serve': 2, 'background': 3}
TARGET_FPS = 15
SEQ_LEN = 15  # 1 second @ 15fps
BEFORE_PEAK = 9  # frames before peak
AFTER_PEAK = 5   # frames after peak (BEFORE_PEAK + 1 + AFTER_PEAK = SEQ_LEN)


# ------------------------------------------------------------------
# Step 1: Download THETIS videos
# ------------------------------------------------------------------

def download_thetis(data_dir: str) -> None:
    """Download all THETIS RGB category videos from GitHub."""
    os.makedirs(data_dir, exist_ok=True)
    for cat in THETIS_CATEGORIES:
        cat_dir = os.path.join(data_dir, cat)
        os.makedirs(cat_dir, exist_ok=True)

        api_url = f"https://api.github.com/repos/THETIS-dataset/dataset/contents/VIDEO_RGB/{cat}"
        req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            files = json.loads(resp.read())

        avi_files = [f for f in files if f['name'].endswith('.avi')]
        print(f"{cat}: {len(avi_files)} files")

        for i, f in enumerate(avi_files):
            out_path = os.path.join(cat_dir, f['name'])
            if os.path.exists(out_path):
                continue
            try:
                urllib.request.urlretrieve(f['download_url'], out_path)
            except Exception as e:
                print(f"  Error: {f['name']}: {e}")
                continue
            if (i + 1) % 30 == 0:
                print(f"  {cat}: {i+1}/{len(avi_files)}")
        print(f"  {cat}: done ({len(os.listdir(cat_dir))} files)")

    total = sum(len(os.listdir(os.path.join(data_dir, c))) for c in THETIS_CATEGORIES)
    print(f"\nTotal videos: {total}")


# ------------------------------------------------------------------
# Step 2: Extract keypoints (with fps stored per sequence)
# ------------------------------------------------------------------

def extract_keypoints(data_dir: str, out_pkl: str) -> list:
    """Extract pose keypoints from all THETIS videos.

    Returns list of (kp_array, label, fps) tuples.
    """
    from ultralytics import YOLO
    model = YOLO("yolo26s-pose.pt")

    sequences = []
    for cat, label in THETIS_CATEGORIES.items():
        cat_dir = os.path.join(data_dir, cat)
        videos = sorted(glob.glob(f"{cat_dir}/*.avi"))
        print(f"\nProcessing {cat} -> {label}: {len(videos)} videos")

        for vi, vpath in enumerate(videos):
            cap = cv2.VideoCapture(vpath)
            src_fps = cap.get(cv2.CAP_PROP_FPS)
            kp_seq = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                results = model.predict(frame, verbose=False, conf=0.3, imgsz=320)
                r = results[0]
                if r.keypoints is not None and len(r.keypoints.data) > 0:
                    best = int(r.boxes.conf.cpu().numpy().argmax())
                    kp = r.keypoints.data[best].cpu().numpy()
                    kp_seq.append(kp)
                else:
                    kp_seq.append(np.zeros((17, 3), dtype=np.float32))
            cap.release()

            if len(kp_seq) >= 10:
                sequences.append((np.array(kp_seq), label, src_fps))
            if (vi + 1) % 30 == 0:
                print(f"  {vi+1}/{len(videos)}")

    print(f"\nTotal sequences: {len(sequences)}")
    for lbl in ['backhand', 'forehand', 'serve']:
        n = sum(1 for _, l, _ in sequences if l == lbl)
        print(f"  {lbl}: {n}")

    with open(out_pkl, "wb") as f:
        pickle.dump(sequences, f)
    print(f"Saved {out_pkl}")
    return sequences


# ------------------------------------------------------------------
# Step 3: Utilities
# ------------------------------------------------------------------

def _resample(kp_seq, src_fps, target_fps):
    """Resample keypoint sequence from src_fps to target_fps."""
    n = len(kp_seq)
    duration = n / src_fps
    n_target = max(1, int(round(duration * target_fps)))
    indices = np.linspace(0, n - 1, n_target).astype(int)
    return kp_seq[indices]


def _find_peak(kp_seq):
    """Find the peak frame: last wrist-above-head highest point.

    For serves: racket hand peaks last (contact point).
    For forehand/backhand: falls back to max wrist speed.
    """
    WRIST = [9, 10]
    NOSE = 0

    # Find frames where either wrist is above nose (y_wrist < y_nose)
    above_head = []
    for i in range(len(kp_seq)):
        if kp_seq[i][NOSE, 2] < 0.3:
            continue
        nose_y = kp_seq[i][NOSE, 1]
        for w in WRIST:
            if kp_seq[i][w, 2] > 0.3 and kp_seq[i][w, 1] < nose_y:
                above_head.append(i)
                break

    if above_head:
        # Find the minimum wrist y (highest point)
        min_y = float('inf')
        for i in above_head:
            for w in WRIST:
                if kp_seq[i][w, 2] > 0.3 and kp_seq[i][w, 1] < min_y:
                    min_y = kp_seq[i][w, 1]

        # Peak zone: frames within 10% of the highest point
        nose_y_ref = kp_seq[above_head[0]][NOSE, 1]
        threshold = min_y + (nose_y_ref - min_y) * 0.1
        peak_zone = [i for i in above_head
                     if any(kp_seq[i][w, 2] > 0.3 and kp_seq[i][w, 1] <= threshold
                            for w in WRIST)]
        # Last frame in peak zone = racket hand contact
        return peak_zone[-1] if peak_zone else above_head[-1]

    # Fallback: max wrist speed
    max_speed, peak = 0.0, len(kp_seq) // 2
    for i in range(1, len(kp_seq)):
        speed = 0.0
        for w in WRIST:
            if kp_seq[i][w, 2] > 0.3 and kp_seq[i - 1][w, 2] > 0.3:
                speed += np.sqrt((kp_seq[i][w, 0] - kp_seq[i - 1][w, 0]) ** 2 +
                                 (kp_seq[i][w, 1] - kp_seq[i - 1][w, 1]) ** 2)
        if speed > max_speed:
            max_speed = speed
            peak = i
    return peak


def _normalize_frame(kp):
    """Normalize a single frame's keypoints: x/bw, y/bh (v4 square stretch)."""
    valid = kp[:, 2] > 0.3
    if valid.sum() >= 3:
        xmin, xmax = kp[valid, 0].min(), kp[valid, 0].max()
        ymin, ymax = kp[valid, 1].min(), kp[valid, 1].max()
        bw = max(xmax - xmin, 1)
        bh = max(ymax - ymin, 1)
        kp = kp.copy()
        kp[:, 0] = (kp[:, 0] - xmin) / bw
        kp[:, 1] = (kp[:, 1] - ymin) / bh
    return kp


def _extract_window(kp_seq, peak, before=BEFORE_PEAK, after=AFTER_PEAK):
    """Extract asymmetric window around peak: before + peak + after = SEQ_LEN.

    Pads with edge frames if window extends beyond sequence bounds.
    """
    n = len(kp_seq)
    start = peak - before
    end = peak + after + 1  # exclusive

    # Pad if needed
    pad_before = max(0, -start)
    pad_after = max(0, end - n)
    start = max(0, start)
    end = min(n, end)

    window = kp_seq[start:end]
    if pad_before > 0:
        window = np.concatenate([np.tile(kp_seq[0:1], (pad_before, 1, 1)), window])
    if pad_after > 0:
        window = np.concatenate([window, np.tile(kp_seq[-1:], (pad_after, 1, 1))])

    return window


# ------------------------------------------------------------------
# Step 4: Dataset
# ------------------------------------------------------------------

class StrokeDataset(Dataset):
    """v6 dataset: 15fps resampled, peak-anchored asymmetric window."""

    def __init__(self, sequences: list):
        self.samples = []
        for item in sequences:
            # Handle both (kp_seq, label, fps) and (kp_seq, label) formats
            if len(item) == 3:
                kp_seq, label, src_fps = item
            else:
                kp_seq, label = item
                src_fps = 17.0  # default THETIS fps

            # Resample to TARGET_FPS
            kp_seq = _resample(kp_seq, src_fps, TARGET_FPS)

            # Find peak on raw pixel coordinates (before normalization)
            peak = _find_peak(kp_seq)

            # Normalize all frames
            norm_seq = np.array([_normalize_frame(kp) for kp in kp_seq])

            # Action sample: asymmetric window around peak
            window = _extract_window(norm_seq, peak)
            feat = window[:, :, :2].reshape(SEQ_LEN, -1)
            self.samples.append((feat.astype(np.float32), LABEL2IDX[label]))

            # Background samples: windows centered far from peak
            # Use _extract_window which handles edge padding
            n = len(norm_seq)
            # Background at video start (centered at frame BEFORE_PEAK//2)
            bg_center_start = min(BEFORE_PEAK // 2, max(0, peak - BEFORE_PEAK - SEQ_LEN // 2))
            if abs(bg_center_start - peak) > SEQ_LEN // 2:
                bg = _extract_window(norm_seq, bg_center_start)
                feat = bg[:, :, :2].reshape(SEQ_LEN, -1)
                self.samples.append((feat.astype(np.float32), LABEL2IDX['background']))
            # Background at video end
            bg_center_end = max(n - 1 - AFTER_PEAK // 2, min(n - 1, peak + AFTER_PEAK + SEQ_LEN // 2))
            if abs(bg_center_end - peak) > SEQ_LEN // 2:
                bg = _extract_window(norm_seq, bg_center_end)
                feat = bg[:, :, :2].reshape(SEQ_LEN, -1)
                self.samples.append((feat.astype(np.float32), LABEL2IDX['background']))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feat, label = self.samples[idx]
        return torch.FloatTensor(feat), torch.LongTensor([label])[0]


# ------------------------------------------------------------------
# Step 5: Training
# ------------------------------------------------------------------

def train(sequences: list, out_path: str, epochs: int = 30,
          lr: float = 1e-3, n_classes: int = 4) -> None:
    """Train GRU classifier and save best checkpoint."""
    from sklearn.model_selection import train_test_split

    train_seq, val_seq = train_test_split(sequences, test_size=0.2, random_state=42)
    train_ds = StrokeDataset(train_seq)
    val_ds = StrokeDataset(val_seq)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=64)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    from collections import Counter
    idx2label = {v: k for k, v in LABEL2IDX.items()}
    for ds, name in [(train_ds, "Train"), (val_ds, "Val")]:
        counts = Counter(label for _, label in ds.samples)
        parts = [f"{idx2label[k]}:{v}" for k, v in sorted(counts.items())]
        print(f"  {name}: {', '.join(parts)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StrokeGRU(n_classes=n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for X, y in train_dl:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (pred.argmax(1) == y).sum().item()
            total += len(y)

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for X, y in val_dl:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                val_correct += (pred.argmax(1) == y).sum().item()
                val_total += len(y)

        val_acc = val_correct / max(val_total, 1)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            torch.save(model.state_dict(), out_path)

        print(f"Epoch {epoch+1}/{epochs}: "
              f"loss={total_loss/len(train_dl):.3f} "
              f"train_acc={correct/total:.3f} "
              f"val_acc={val_acc:.3f}"
              f"{' *' if val_acc >= best_val_acc else ''}")

    print(f"\nBest val_acc: {best_val_acc:.3f}")
    print(f"Saved: {out_path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="Output weights path")
    parser.add_argument("--keypoints", default=None,
                        help="Pre-extracted keypoints pickle (skip download+extract)")
    parser.add_argument("--data-dir", default="/tmp/thetis_rgb",
                        help="Directory for THETIS videos (default: /tmp/thetis_rgb)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-classes", type=int, default=4,
                        help="Number of classes (3=no background, 4=with background)")
    args = parser.parse_args()

    if args.keypoints and os.path.exists(args.keypoints):
        print(f"Loading cached keypoints: {args.keypoints}")
        with open(args.keypoints, "rb") as f:
            sequences = pickle.load(f)
    else:
        print("Step 1: Downloading THETIS dataset...")
        download_thetis(args.data_dir)

        print("\nStep 2: Extracting keypoints...")
        pkl_path = os.path.join(args.data_dir, "thetis_keypoints.pkl")
        sequences = extract_keypoints(args.data_dir, pkl_path)

    print(f"\nStep 3: Training GRU v6 ({args.epochs} epochs, "
          f"SEQ_LEN={SEQ_LEN}, TARGET_FPS={TARGET_FPS})...")
    train(sequences, args.out, epochs=args.epochs,
          lr=args.lr, n_classes=args.n_classes)


if __name__ == "__main__":
    main()
