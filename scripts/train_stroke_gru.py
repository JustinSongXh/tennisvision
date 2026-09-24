#!/usr/bin/env python3
"""Train GRU stroke classifier on THETIS dataset.

Downloads THETIS RGB videos, extracts keypoints with YOLO-pose,
and trains a 4-class GRU (backhand / forehand / serve / background).

This is an auxiliary standalone script, not part of the main pipeline.
Requires GPU for reasonable speed.

Usage:
    # Full training (download + extract + train)
    python -u scripts/train_stroke_gru.py --out weights/stroke_gru_v4_best.pt

    # Resume from cached keypoints
    python -u scripts/train_stroke_gru.py \
        --keypoints /path/to/thetis_keypoints.pkl \
        --out weights/stroke_gru_v4_best.pt

    # Custom hyperparameters
    python -u scripts/train_stroke_gru.py \
        --out weights/stroke_gru_v4_best.pt \
        --epochs 80 --seq-len 30 --lr 1e-3
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

# THETIS categories → our 3-class labels (background is synthetic)
THETIS_CATEGORIES = {
    'flat_service': 'serve',
    'kick_service': 'serve',
    'slice_service': 'serve',
    'forehand_flat': 'forehand',
    'forehand_openstands': 'forehand',
    'forehand_slice': 'forehand',
    'forehand_volley': 'forehand',
    'backhand': 'backhand',
    'backhand2hands': 'backhand',
    'backhand_slice': 'backhand',
    'backhand_volley': 'backhand',
    'smash': 'forehand',
}

LABEL2IDX = {'backhand': 0, 'forehand': 1, 'serve': 2, 'background': 3}


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
# Step 2: Extract keypoints
# ------------------------------------------------------------------

def extract_keypoints(data_dir: str, out_pkl: str) -> list:
    """Extract pose keypoints from all THETIS videos."""
    from ultralytics import YOLO
    model = YOLO("yolo26s-pose.pt")

    sequences = []
    for cat, label in THETIS_CATEGORIES.items():
        cat_dir = os.path.join(data_dir, cat)
        videos = sorted(glob.glob(f"{cat_dir}/*.avi"))
        print(f"\nProcessing {cat} -> {label}: {len(videos)} videos")

        for vi, vpath in enumerate(videos):
            cap = cv2.VideoCapture(vpath)
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
                sequences.append((np.array(kp_seq), label))
            if (vi + 1) % 30 == 0:
                print(f"  {vi+1}/{len(videos)}")

    print(f"\nTotal sequences: {len(sequences)}")
    for lbl in ['backhand', 'forehand', 'serve']:
        n = sum(1 for _, l in sequences if l == lbl)
        print(f"  {lbl}: {n}")

    with open(out_pkl, "wb") as f:
        pickle.dump(sequences, f)
    print(f"Saved {out_pkl}")
    return sequences


# ------------------------------------------------------------------
# Step 3: Dataset + Training
# ------------------------------------------------------------------

class StrokeDataset(Dataset):
    """Sliding-window dataset from keypoint sequences."""

    def __init__(self, sequences: list, seq_len: int = 30):
        self.samples = []
        for kp_seq, label in sequences:
            kp_seq = kp_seq.copy()
            # Bbox-relative normalization per frame
            for i in range(len(kp_seq)):
                valid = kp_seq[i][:, 2] > 0.3
                if valid.sum() >= 3:
                    xmin = kp_seq[i][valid, 0].min()
                    xmax = kp_seq[i][valid, 0].max()
                    ymin = kp_seq[i][valid, 1].min()
                    ymax = kp_seq[i][valid, 1].max()
                    bw = max(xmax - xmin, 1)
                    bh = max(ymax - ymin, 1)
                    scale = max(bw, bh)
                    kp_seq[i][:, 0] = (kp_seq[i][:, 0] - xmin) / scale
                    kp_seq[i][:, 1] = (kp_seq[i][:, 1] - ymin) / scale

            if len(kp_seq) >= seq_len:
                # Sliding windows with 50% overlap
                for start in range(0, len(kp_seq) - seq_len + 1, seq_len // 2):
                    window = kp_seq[start:start + seq_len]
                    feat = window[:, :, :2].reshape(seq_len, -1)
                    self.samples.append((feat.astype(np.float32), LABEL2IDX[label]))
            else:
                # Pad short sequences
                padded = np.concatenate([
                    kp_seq,
                    np.tile(kp_seq[-1:], (seq_len - len(kp_seq), 1, 1))
                ])
                feat = padded[:seq_len, :, :2].reshape(seq_len, -1)
                self.samples.append((feat.astype(np.float32), LABEL2IDX[label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feat, label = self.samples[idx]
        return torch.FloatTensor(feat), torch.LongTensor([label])[0]


def train(sequences: list, out_path: str, epochs: int = 50,
          seq_len: int = 30, lr: float = 1e-3, n_classes: int = 4) -> None:
    """Train GRU classifier and save best checkpoint."""
    from sklearn.model_selection import train_test_split

    train_seq, val_seq = train_test_split(sequences, test_size=0.2, random_state=42)
    train_ds = StrokeDataset(train_seq, seq_len)
    val_ds = StrokeDataset(val_seq, seq_len)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=64)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

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
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seq-len", type=int, default=30)
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

    print(f"\nStep 3: Training GRU ({args.epochs} epochs, {args.n_classes} classes)...")
    train(sequences, args.out, epochs=args.epochs,
          seq_len=args.seq_len, lr=args.lr, n_classes=args.n_classes)


if __name__ == "__main__":
    main()
