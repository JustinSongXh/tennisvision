"""Wrapper around yastrebksv/TennisCourtDetector for court keypoint inference.

Architecture ported from tracknet.py at
https://github.com/yastrebksv/TennisCourtDetector (no explicit license
in the upstream repo, but the model is publicly released for research
use; include attribution in any derived artifact).

Pretrained weights (15-channel output): Drive id 1f-Co64ehgq4uddcQm1aFBDtbnyZhQvgG.
Expected location: weights/court_tcd.pt (or pass --weights).

Input:  BGR frame of any size
Output: dict[keypoint_id -> (x_img, y_img)] with ids 0..13 following
        tennisvision.court.reference numbering.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# Keypoint index mapping — TennisCourtDetector output channel index
# 0..13 vs our tennisvision.court.reference.KEYPOINTS_M indices.  The
# upstream dataset numbers the 14 corners in a specific order; below is
# the permutation that maps their channel N to our reference id.
#
# Upstream order (from TennisCourtDetector/README.md):
#   0  top-left (far-left doubles)      -> our 0  FL
#   1  top-right (far-right doubles)    -> our 1  FR
#   2  bottom-left (near-left doubles)  -> our 3  NL
#   3  bottom-right (near-right doubles)-> our 2  NR
#   4  top-left singles                 -> our 4  FLs
#   5  top-right singles                -> our 5  FRs
#   6  bottom-left singles              -> our 7  NLs
#   7  bottom-right singles             -> our 6  NRs
#   8  top inner-left T                 -> our 10 FLv
#   9  top inner-right T                -> our 11 FRv
#   10 bottom inner-left T              -> our 8  NLv
#   11 bottom inner-right T             -> our 9  NRv
#   12 top T                            -> our 13 Tf
#   13 bottom T                         -> our 12 Tn
UPSTREAM_TO_OURS: dict = {
    0: 0,  1: 1,  2: 3,  3: 2,        # doubles corners (FL, FR, NL, NR)
    4: 4,  5: 7,  6: 5,  7: 6,        # singles corners (FLs, NLs, FRs, NRs)
    8: 10, 9: 11, 10: 8, 11: 9,       # service-line x singles
    12: 13, 13: 12,                    # service Ts (Tf, Tn)
}


if _HAS_TORCH:

    class _ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3,
                     pad=1, stride=1, bias=True):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size,
                          stride=stride, padding=pad, bias=bias),
                nn.ReLU(),
                nn.BatchNorm2d(out_channels),
            )

        def forward(self, x):
            return self.block(x)


    class BallTrackerNet(nn.Module):
        """Encoder-decoder net used by yastrebksv for both ball tracking
        and court keypoint detection.  For courts it's instantiated with
        out_channels=15 (14 keypoints + 1 unused/aux)."""

        def __init__(self, out_channels: int = 15):
            super().__init__()
            self.out_channels = out_channels
            self.conv1  = _ConvBlock(3, 64)
            self.conv2  = _ConvBlock(64, 64)
            self.pool1  = nn.MaxPool2d(2, 2)
            self.conv3  = _ConvBlock(64, 128)
            self.conv4  = _ConvBlock(128, 128)
            self.pool2  = nn.MaxPool2d(2, 2)
            self.conv5  = _ConvBlock(128, 256)
            self.conv6  = _ConvBlock(256, 256)
            self.conv7  = _ConvBlock(256, 256)
            self.pool3  = nn.MaxPool2d(2, 2)
            self.conv8  = _ConvBlock(256, 512)
            self.conv9  = _ConvBlock(512, 512)
            self.conv10 = _ConvBlock(512, 512)
            self.ups1   = nn.Upsample(scale_factor=2)
            self.conv11 = _ConvBlock(512, 256)
            self.conv12 = _ConvBlock(256, 256)
            self.conv13 = _ConvBlock(256, 256)
            self.ups2   = nn.Upsample(scale_factor=2)
            self.conv14 = _ConvBlock(256, 128)
            self.conv15 = _ConvBlock(128, 128)
            self.ups3   = nn.Upsample(scale_factor=2)
            self.conv16 = _ConvBlock(128, 64)
            self.conv17 = _ConvBlock(64, 64)
            self.conv18 = _ConvBlock(64, out_channels)

        def forward(self, x):
            x = self.conv2(self.conv1(x));          x = self.pool1(x)
            x = self.conv4(self.conv3(x));          x = self.pool2(x)
            x = self.conv7(self.conv6(self.conv5(x))); x = self.pool3(x)
            x = self.conv10(self.conv9(self.conv8(x)))
            x = self.ups1(x)
            x = self.conv13(self.conv12(self.conv11(x)))
            x = self.ups2(x)
            x = self.conv15(self.conv14(x))
            x = self.ups3(x)
            x = self.conv17(self.conv16(x))
            x = self.conv18(x)
            return x


def _postprocess_heatmap(heatmap: np.ndarray, low_thresh: int = 170,
                         min_radius: int = 10, max_radius: int = 25):
    """Upstream postprocess: threshold + HoughCircles; return first
    circle center in heatmap pixel coords, or (None, None)."""
    import cv2
    _, hm = cv2.threshold(heatmap, low_thresh, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(
        hm, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
        param1=50, param2=2, minRadius=min_radius, maxRadius=max_radius,
    )
    if circles is None:
        return None, None
    return float(circles[0][0][0]), float(circles[0][0][1])


class HeatmapDetector:
    """Run yastrebksv/TennisCourtDetector on a single frame.

    Parameters
    ----------
    weights: path to the .pt state_dict checkpoint (default weights/court_tcd.pt).
    device : "cpu" or "cuda".
    input_size : (W, H) the network expects; default (640, 360).
    """

    def __init__(self, weights: str = "weights/court_tcd.pt",
                 device: str = "cpu",
                 input_size: tuple = (640, 360),
                 low_thresh: int = 170,
                 min_radius: int = 10,
                 max_radius: int = 25):
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch is required for HeatmapDetector.  "
                               "Install `torch` or use detector='mark'.")
        self.device = torch.device(device)
        self.input_size = input_size
        self.low_thresh = low_thresh
        self.min_radius = min_radius
        self.max_radius = max_radius

        self.model = BallTrackerNet(out_channels=15)
        state = torch.load(weights, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def detect(self, frame: np.ndarray) -> dict:
        import cv2
        H, W = frame.shape[:2]
        in_w, in_h = self.input_size
        img = cv2.resize(frame, (in_w, in_h))
        inp = img.astype(np.float32) / 255.0
        inp = np.transpose(inp, (2, 0, 1))                 # HWC -> CHW
        x = torch.from_numpy(inp).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            y = self.model(x)[0]
            pred = torch.sigmoid(y).cpu().numpy()          # (15, in_h, in_w)

        scale_x = W / float(in_w)
        scale_y = H / float(in_h)

        out = {}
        for ch in range(14):
            hm = (pred[ch] * 255).astype(np.uint8)
            x_pred, y_pred = _postprocess_heatmap(
                hm, low_thresh=self.low_thresh,
                min_radius=self.min_radius, max_radius=self.max_radius,
            )
            if x_pred is None or y_pred is None:
                continue
            our_id = UPSTREAM_TO_OURS[ch]
            out[our_id] = (x_pred * scale_x, y_pred * scale_y)
        return out
