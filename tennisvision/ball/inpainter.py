"""Learning-based trajectory gap-filling (TrackNetV3 InpaintNet).

Ports `InpaintNet` + `generate_inpaint_mask` from qaz812345/TrackNetV3
(MIT License).  Upstream trained on **shuttlecock**; reuse for tennis is
empirical — validate before trusting.

Pipeline position:

    WASB per-frame detect -> MultiTrackManager (Kalman)
                                      |
                          retired/alive validated tracks
                                      |
                           TrajectoryInpainter  <-- this module
                                      |
                              dense series of
                        (x, y, frame, was_inpainted)
                                      |
                              CatBoost bounce + render

We merge every Track into a single frame-indexed series (union of all
validated tracks), build the inpaint mask with the same rules as the
upstream reference, and run a 1D U-Net over the sequence to fill the
holes.  The model is tiny (~500k params, <2 MB weights) so CPU-only
torch inference is fast enough — no ONNX detour needed at this stage.

Mask semantics match upstream:
  mask[i] == 1  =>  replace this frame with the model's prediction
  mask[i] == 0  =>  keep the original detection (or leave as None)

The mask builder deliberately skips gaps where the ball exits the top
of frame (e.g. after a lob) — filling those would fabricate coordinates
for a ball that is physically out of view.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    nn = object  # type: ignore[assignment]  # so class defs below don't ImportError

from .tracker import Track, TrackPoint


# ----------------------------------------------------------------------
# Model (verbatim from qaz812345/TrackNetV3/model.py, MIT License).
# Kept inline because it's small (~30 LOC); upstream copyright applies.
# ----------------------------------------------------------------------

if _HAS_TORCH:
    class _Conv1DBlock(nn.Module):
        def __init__(self, in_dim: int, out_dim: int):
            super().__init__()
            self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=3,
                                  padding="same", bias=True)
            self.relu = nn.LeakyReLU()

        def forward(self, x):
            return self.relu(self.conv(x))

    class _Double1DConv(nn.Module):
        def __init__(self, in_dim: int, out_dim: int):
            super().__init__()
            self.conv_1 = _Conv1DBlock(in_dim, out_dim)
            self.conv_2 = _Conv1DBlock(out_dim, out_dim)

        def forward(self, x):
            return self.conv_2(self.conv_1(x))

    class InpaintNet(nn.Module):
        """1D U-Net over coordinate sequences.

        Forward takes `(x, m)`:
          x: (N, L, 2) float in [0, 1] — (x_norm, y_norm) per frame
          m: (N, L, 1) float {0, 1}    — 1 = hole to fill, 0 = observation
        Returns (N, L, 2) in [0, 1] via sigmoid.
        """

        def __init__(self):
            super().__init__()
            self.down_1 = _Conv1DBlock(3, 32)
            self.down_2 = _Conv1DBlock(32, 64)
            self.down_3 = _Conv1DBlock(64, 128)
            self.buttleneck = _Double1DConv(128, 256)   # [sic] upstream spelling
            self.up_1 = _Conv1DBlock(384, 128)
            self.up_2 = _Conv1DBlock(192, 64)
            self.up_3 = _Conv1DBlock(96, 32)
            self.predictor = nn.Conv1d(32, 2, 3, padding="same")
            self.sigmoid = nn.Sigmoid()

        def forward(self, x, m):
            x = torch.cat([x, m], dim=2)        # (N, L, 3)
            x = x.permute(0, 2, 1)              # (N, 3, L)
            x1 = self.down_1(x)
            x2 = self.down_2(x1)
            x3 = self.down_3(x2)
            x = self.buttleneck(x3)
            x = torch.cat([x, x3], dim=1)
            x = self.up_1(x)
            x = torch.cat([x, x2], dim=1)
            x = self.up_2(x)
            x = torch.cat([x, x1], dim=1)
            x = self.up_3(x)
            x = self.predictor(x)
            x = self.sigmoid(x)
            return x.permute(0, 2, 1)           # (N, L, 2)


# ----------------------------------------------------------------------
# Mask builder (ported from qaz812345/TrackNetV3/test.py:generate_inpaint_mask)
# ----------------------------------------------------------------------

def generate_inpaint_mask(vis: np.ndarray, y: np.ndarray, th_h: float) -> np.ndarray:
    """Mark internal detection gaps as fillable, skipping gaps where the
    ball plausibly left the top of the frame.

    Args:
        vis: (L,) float, 1 where a detection exists, 0 where missing
        y:   (L,) float, y-pixel coordinate (0 at top of image)
        th_h: pixel threshold; gaps whose bordering detections are within
            `th_h` of the top edge are NOT filled (ball is out of view)

    Returns:
        mask: (L,) float {0, 1}
    """
    vis = np.asarray(vis).astype(np.int32)
    y = np.asarray(y).astype(np.float32)
    L = len(vis)
    mask = np.zeros(L, dtype=np.float32)
    if L == 0:
        return mask

    # Walk the sequence: find every run of zeros bounded by ones (or the ends).
    i = 0
    while i < L:
        # advance through visible frames
        while i < L - 1 and vis[i] == 1:
            i += 1
        # i is now first-invisible (or last frame)
        j = i
        while j < L - 1 and vis[j] == 0:
            j += 1
        # j is first-visible after the gap (or last frame)
        if j == i:
            break

        if i == 0:
            # gap starts at frame 0: fill only if the ball reappears below the top edge
            if j < L and vis[j] == 1 and y[j] > th_h:
                mask[:j] = 1
        elif j < L - 1 or (j == L - 1 and vis[j] == 1):
            # internal gap with defined pre/post visibility
            if y[i - 1] > th_h and (vis[j] == 0 or y[j] > th_h):
                mask[i:j] = 1
        # else: trailing gap at end of sequence — leave as zero (upstream skips too)
        i = j
    return mask


# ----------------------------------------------------------------------
# Public wrapper
# ----------------------------------------------------------------------

@dataclass
class InpaintNetConfig:
    weights: str = "weights/inpaint_best.pt"
    device: str = "cpu"
    # Upstream ckpt stores seq_len in param_dict; we only use it for
    # windowed inference (single-pass ignores it since the model is
    # fully convolutional in the temporal axis).
    window_mode: str = "single"          # "single" | "nonoverlap"
    seq_len: Optional[int] = None        # override ckpt's stored seq_len
    # Skip gaps longer than this (frames) — beyond this the learned
    # interpolation is likely unreliable; leave as None so downstream
    # (CatBoost) can decide to ignore the track.
    max_gap_frames: int = 60
    # Upstream's `th_h` for mask building: y-pixel threshold (ball
    # treated as out-of-view if border detection is within this many
    # pixels of the top).  5% of frame height matches upstream default.
    th_h_frac: float = 0.05


@dataclass
class InpaintResult:
    """Dense per-frame result over [frame_lo, frame_hi].  Both `xs` and
    `ys` are length = frame_hi - frame_lo + 1; entries can be NaN where
    no observation existed and the frame was not filled (either gap was
    out-of-view or exceeded max_gap_frames)."""
    frame_lo: int
    xs: np.ndarray                      # (L,) float, NaN = no value
    ys: np.ndarray                      # (L,) float, NaN = no value
    was_inpainted: np.ndarray           # (L,) bool


class TrajectoryInpainter:
    """Wraps a loaded InpaintNet and a frame size for normalization."""

    def __init__(self, cfg: InpaintNetConfig, frame_size: Tuple[int, int]):
        """frame_size = (W, H) in pixels — used to normalize coordinates
        into [0, 1] before the model and denormalize after."""
        if not _HAS_TORCH:
            raise RuntimeError(
                "TrajectoryInpainter needs torch: `pip install torch`")
        self.cfg = cfg
        self.W, self.H = frame_size

        ckpt = torch.load(cfg.weights, map_location=cfg.device, weights_only=False)
        # Upstream format: {'model': state_dict, 'param_dict': {'seq_len': N, ...}, ...}
        # Be lenient to plain state_dict too.
        if isinstance(ckpt, dict) and "model" in ckpt:
            state = ckpt["model"]
            pd = ckpt.get("param_dict", {})
            self.seq_len = int(cfg.seq_len or pd.get("seq_len", 16))
        else:
            state = ckpt
            self.seq_len = int(cfg.seq_len or 16)

        self._model = InpaintNet().to(cfg.device)
        self._model.load_state_dict(state)
        self._model.eval()

    # -- Track list -> dense per-frame series (union) --------------------

    @staticmethod
    def merge_tracks(tracks: Sequence[Track]) -> Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
        """Build a per-frame dense series from the union of all points.

        When two tracks claim the same frame, keeps whichever point
        appears first (stable by iteration order).  The caller decides
        which tracks to pass in — typically just the validated ones.

        Returns (frame_lo, frame_hi, xs, ys, vis), all dense over
        `[frame_lo, frame_hi]`.
        """
        frames = [p.frame for t in tracks for p in t.pts]
        if not frames:
            return 0, -1, np.zeros(0), np.zeros(0), np.zeros(0)
        lo, hi = min(frames), max(frames)
        L = hi - lo + 1
        xs = np.full(L, np.nan, dtype=np.float32)
        ys = np.full(L, np.nan, dtype=np.float32)
        for t in tracks:
            for p in t.pts:
                idx = p.frame - lo
                if np.isnan(xs[idx]):    # first-writer wins
                    xs[idx] = float(p.x)
                    ys[idx] = float(p.y)
        vis = (~np.isnan(xs)).astype(np.float32)
        return lo, hi, xs, ys, vis

    # -- core inference --------------------------------------------------

    def _run_model(self, xs: np.ndarray, ys: np.ndarray, mask: np.ndarray
                   ) -> Tuple[np.ndarray, np.ndarray]:
        """Run the model over a full sequence and return model outputs
        in pixel space (shape-matched to inputs)."""
        coord = np.stack([xs / max(self.W, 1), ys / max(self.H, 1)], axis=-1)
        # Upstream feeds 0 for missing coords; we've already NaN-filled
        # those, so replace NaN -> 0 just before the model.
        coord = np.where(np.isnan(coord), 0.0, coord).astype(np.float32)
        m = mask.astype(np.float32)[:, None]

        x_t = torch.from_numpy(coord)[None, ...].to(self.cfg.device)   # (1, L, 2)
        m_t = torch.from_numpy(m)[None, ...].to(self.cfg.device)       # (1, L, 1)
        with torch.no_grad():
            out = self._model(x_t, m_t).cpu().numpy()[0]               # (L, 2)
        out_x = out[:, 0] * self.W
        out_y = out[:, 1] * self.H
        return out_x, out_y

    # -- public API ------------------------------------------------------

    def inpaint_series(self, lo: int, xs: np.ndarray, ys: np.ndarray,
                       vis: np.ndarray) -> InpaintResult:
        """Given a dense per-frame series (NaN where missing), return
        the same series with gaps filled where the learned mask allows."""
        L = len(xs)
        if L == 0:
            return InpaintResult(frame_lo=lo, xs=xs, ys=ys,
                                 was_inpainted=np.zeros(0, dtype=bool))

        th_h = self.cfg.th_h_frac * self.H
        # Replace NaNs with 0 for y in mask builder (upstream expects numeric
        # placeholders at gaps; mask itself decides what's valid).
        y_for_mask = np.where(np.isnan(ys), 0.0, ys)
        mask = generate_inpaint_mask(vis, y_for_mask, th_h=th_h)

        # Apply max_gap_frames ceiling — unmask runs longer than that.
        if self.cfg.max_gap_frames > 0:
            mask = _clip_long_runs(mask, max_len=self.cfg.max_gap_frames)

        if not mask.any():
            return InpaintResult(frame_lo=lo, xs=xs.copy(), ys=ys.copy(),
                                 was_inpainted=np.zeros(L, dtype=bool))

        # The model is fully convolutional in L; single-pass is equivalent
        # to non-overlap windowing up to boundary padding.  Offer both.
        if self.cfg.window_mode == "single":
            out_x, out_y = self._run_model(xs, ys, mask)
        elif self.cfg.window_mode == "nonoverlap":
            out_x, out_y = self._run_windowed(xs, ys, mask)
        else:
            raise ValueError("unknown window_mode: %r" % self.cfg.window_mode)

        filled_x = np.where(mask > 0, out_x, xs)
        filled_y = np.where(mask > 0, out_y, ys)
        return InpaintResult(frame_lo=lo, xs=filled_x.astype(np.float32),
                             ys=filled_y.astype(np.float32),
                             was_inpainted=(mask > 0))

    def _run_windowed(self, xs: np.ndarray, ys: np.ndarray, mask: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray]:
        """Non-overlap windowed inference at self.seq_len, zero-padding
        the tail window to length."""
        L = len(xs)
        S = self.seq_len
        out_x = np.zeros(L, dtype=np.float32)
        out_y = np.zeros(L, dtype=np.float32)
        for i in range(0, L, S):
            j = min(i + S, L)
            chunk_x = np.zeros(S, dtype=np.float32)
            chunk_y = np.zeros(S, dtype=np.float32)
            chunk_m = np.zeros(S, dtype=np.float32)
            chunk_x[: j - i] = np.nan_to_num(xs[i:j], nan=0.0)
            chunk_y[: j - i] = np.nan_to_num(ys[i:j], nan=0.0)
            chunk_m[: j - i] = mask[i:j]
            ox, oy = self._run_model(chunk_x, chunk_y, chunk_m)
            out_x[i:j] = ox[: j - i]
            out_y[i:j] = oy[: j - i]
        return out_x, out_y

    def inpaint_tracks(self, tracks: Sequence[Track]) -> InpaintResult:
        """Convenience: merge `tracks` into a dense series and inpaint."""
        lo, hi, xs, ys, vis = self.merge_tracks(tracks)
        if hi < lo:
            return InpaintResult(frame_lo=0, xs=np.zeros(0), ys=np.zeros(0),
                                 was_inpainted=np.zeros(0, dtype=bool))
        return self.inpaint_series(lo, xs, ys, vis)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _clip_long_runs(mask: np.ndarray, max_len: int) -> np.ndarray:
    """Zero out any contiguous run of 1s longer than `max_len`."""
    out = mask.copy()
    L = len(out)
    i = 0
    while i < L:
        if out[i] <= 0:
            i += 1
            continue
        j = i
        while j < L and out[j] > 0:
            j += 1
        if j - i > max_len:
            out[i:j] = 0
        i = j
    return out


def result_to_trackpoints(res: InpaintResult) -> List[TrackPoint]:
    """Convert an InpaintResult to a flat list of TrackPoints, dropping
    frames where both xs and ys are NaN (never observed, never filled)."""
    pts: List[TrackPoint] = []
    for i, (x, y) in enumerate(zip(res.xs, res.ys)):
        if np.isnan(x) or np.isnan(y):
            continue
        pts.append(TrackPoint(int(round(x)), int(round(y)), res.frame_lo + i))
    return pts
