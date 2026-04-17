"""WASB (Widely-Applicable Small-ball Ball Detection & Tracking) inference wrapper.

Wraps the HRNet variant defined in .hrnet and reproduces the tennis
preprocessing / postprocessing pipeline from nttcom/WASB-SBDT (MIT).

Two inference backends — selected via `WASBConfig.runtime`:
  - "torch": the reference PyTorch path.  Works on any device.
  - "onnx" : ONNX Runtime with OpenVINO EP when available.  CPU-only;
             typically 3–5x faster than "torch" on Intel CPUs.
  - "auto" : "onnx" when onnxruntime is importable AND device=="cpu",
             else "torch".

On first use of the ONNX path we lazily export `<weights>.onnx` next to
the .pt checkpoint — users don't have to run a separate tool.
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    import onnxruntime as ort
    _HAS_ORT = True
except ImportError:
    _HAS_ORT = False

from .hrnet import HRNet, WASB_CONFIG


# --- Letterbox-style affine transform (from WASB / CenterNet / HRNet-pose) ---

def _get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def _get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    return [src_point[0] * cs - src_point[1] * sn,
            src_point[0] * sn + src_point[1] * cs]


def _get_affine_transform(center, scale, rot, output_size, inv=0):
    if not isinstance(scale, (np.ndarray, list)):
        scale = np.array([scale, scale], dtype=np.float32)
    scale_tmp = np.asarray(scale, dtype=np.float32)
    src_w = scale_tmp[0]
    dst_w, dst_h = output_size

    rot_rad = np.pi * rot / 180
    src_dir = _get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center
    src[1, :] = center + src_dir
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = dst[0, :] + dst_dir
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        return cv2.getAffineTransform(np.float32(dst), np.float32(src))
    return cv2.getAffineTransform(np.float32(src), np.float32(dst))


def _apply_affine(pt, trans):
    v = np.array([pt[0], pt[1], 1.0], dtype=np.float32)
    return (trans @ v)[:2]


# --- Heatmap blob detector (concomp + weighted centroid) ---

def _detect_blobs(hm, score_threshold=0.5):
    """Returns list of (x, y, score) in heatmap coordinates."""
    out = []
    if hm.max() <= score_threshold:
        return out
    _, hm_th = cv2.threshold(hm, score_threshold, 1, cv2.THRESH_BINARY)
    n_labels, labels = cv2.connectedComponents(hm_th.astype(np.uint8))
    for m in range(1, n_labels):
        ys, xs = np.where(labels == m)
        ws = hm[ys, xs]
        sw = float(np.sum(ws))
        if sw <= 0:
            continue
        x = float(np.sum(xs * ws) / sw)
        y = float(np.sum(ys * ws) / sw)
        out.append((x, y, sw))
    return out


# --- Main detector ---

@dataclass
class WASBConfig:
    weights: str = "weights/wasb_tennis_best.pth.tar"
    device: str = "cpu"
    runtime: str = "auto"                # auto | torch | onnx
    onnx_path: Optional[str] = None      # default: <weights>.onnx
    score_threshold: float = 0.5
    input_width:  int = 512
    input_height: int = 288
    # Temporal gating: candidate must be within max_disp px of previous
    # observed ball (in ORIGINAL image coords).  0 disables gating.
    max_disp: float = 300.0


# Per-run ImageNet stats
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _resolve_runtime(cfg: "WASBConfig") -> str:
    rt = cfg.runtime
    if rt == "auto":
        return "onnx" if (_HAS_ORT and cfg.device == "cpu") else "torch"
    if rt == "onnx" and not _HAS_ORT:
        raise RuntimeError(
            "ball.runtime='onnx' requires onnxruntime — "
            "`pip install onnxruntime` (or `onnxruntime-openvino` for Intel CPUs).")
    if rt == "torch" and not _HAS_TORCH:
        raise RuntimeError("ball.runtime='torch' requires torch")
    return rt


def _export_onnx(pt_weights: str, onnx_path: str,
                 input_h: int, input_w: int) -> None:
    """One-off export of the WASB HRNet to ONNX (fixed batch=1)."""
    if not _HAS_TORCH:
        raise RuntimeError("exporting ONNX requires torch installed once")
    print("[wasb] exporting %s -> %s ..." % (pt_weights, onnx_path))
    import torch as _t
    model = HRNet(WASB_CONFIG)
    ckpt = _t.load(pt_weights, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()

    # Wrap to produce a single tensor (HRNet.forward returns a dict).
    class _SingleScale(_t.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
        def forward(self, x):
            y = self.inner(x)
            return y[0]

    wrapped = _SingleScale(model).eval()
    dummy = _t.zeros(1, 9, input_h, input_w, dtype=_t.float32)
    os.makedirs(os.path.dirname(onnx_path) or ".", exist_ok=True)
    _t.onnx.export(
        wrapped, dummy, onnx_path,
        input_names=["input"], output_names=["heatmap"],
        opset_version=13, do_constant_folding=True,
    )
    print("[wasb] wrote %s" % onnx_path)


def _build_ort_session(onnx_path: str):
    available = ort.get_available_providers()
    providers = []
    if "OpenVINOExecutionProvider" in available:
        providers.append(("OpenVINOExecutionProvider", {"device_type": "CPU_FP32"}))
    providers.append("CPUExecutionProvider")
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(onnx_path, sess_options=so, providers=providers)


class WASBBallDetector:
    """Streaming detector: call push_frame() per frame; after at least
    3 frames have been pushed, call detect() to get the current ball
    position (or None).
    """

    def __init__(self, cfg: WASBConfig):
        self.cfg = cfg
        self._runtime = _resolve_runtime(cfg)

        if self._runtime == "onnx":
            onnx_path = cfg.onnx_path or (os.path.splitext(cfg.weights)[0] + ".onnx")
            if not os.path.isfile(onnx_path):
                _export_onnx(cfg.weights, onnx_path,
                             cfg.input_height, cfg.input_width)
            self._session = _build_ort_session(onnx_path)
            self._input_name = self._session.get_inputs()[0].name
            eps = [p[0] if isinstance(p, tuple) else p
                   for p in self._session.get_providers()]
            print("[wasb] ONNX runtime providers: %s" % eps)
        else:
            if not _HAS_TORCH:
                raise RuntimeError("torch runtime requires `pip install torch`")
            self.device = torch.device(cfg.device)
            self.model = HRNet(WASB_CONFIG)
            ckpt = torch.load(cfg.weights, map_location=self.device)
            state = ckpt.get("model_state_dict", ckpt)
            self.model.load_state_dict(state)
            self.model.to(self.device)
            self.model.eval()

        self._buffer: deque = deque(maxlen=3)
        self._last_pos: Optional[tuple] = None

        # Cached affine transforms — recomputed when input resolution changes.
        self._cached_trans_fwd = None
        self._cached_trans_inv = None
        self._cached_shape = None

    # ------------------------------------------------------------------

    def push_frame(self, frame: np.ndarray) -> None:
        self._buffer.append(frame)

    def reset_tracking(self) -> None:
        self._last_pos = None

    def detect(self) -> Optional[tuple[int, int]]:
        if len(self._buffer) < 3:
            return None
        H, W = self._buffer[-1].shape[:2]
        if self._cached_shape != (W, H):
            center = np.array([W / 2.0, H / 2.0], dtype=np.float32)
            scale  = max(H, W) * 1.0
            out_size = (self.cfg.input_width, self.cfg.input_height)
            self._cached_trans_fwd = _get_affine_transform(center, scale, 0, out_size, inv=0)
            self._cached_trans_inv = _get_affine_transform(center, scale, 0, out_size, inv=1)
            self._cached_shape = (W, H)

        # Warp + normalize each of the 3 frames (oldest first).
        chans = []
        out_size = (self.cfg.input_width, self.cfg.input_height)
        for bgr in list(self._buffer):
            warped = cv2.warpAffine(bgr, self._cached_trans_fwd, out_size,
                                     flags=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            rgb = (rgb - _MEAN) / _STD
            chans.append(np.transpose(rgb, (2, 0, 1)))   # CHW
        inp = np.concatenate(chans, axis=0)[None, ...]    # (1, 9, H, W)

        if self._runtime == "onnx":
            raw = self._session.run(None, {self._input_name: inp})[0]  # (1, 3, H, W)
            hm = 1.0 / (1.0 + np.exp(-raw[0]))                          # sigmoid
        else:
            x = torch.from_numpy(inp).to(self.device)
            with torch.inference_mode():
                y = self.model(x)
                # y is dict {scale: (B, 3, H, W)} — tennis uses scale 0 only.
                scale = list(y.keys())[0]
                hm = torch.sigmoid(y[scale])[0].cpu().numpy()           # (3, H, W)

        # Channel 2 = heatmap for the CURRENT (most-recent) frame.
        blobs = _detect_blobs(hm[2], score_threshold=self.cfg.score_threshold)
        if not blobs:
            return None

        # Back-project each blob centroid to original-image coords.
        candidates = []
        for bx, by, bs in blobs:
            ox, oy = _apply_affine((bx, by), self._cached_trans_inv)
            candidates.append((float(ox), float(oy), bs))

        # Temporal gating: prefer candidates close to previous position.
        best = None
        if self._last_pos is not None and self.cfg.max_disp > 0:
            lx, ly = self._last_pos
            gated = [(x, y, s) for (x, y, s) in candidates
                     if (x - lx) ** 2 + (y - ly) ** 2 <= self.cfg.max_disp ** 2]
            if gated:
                best = max(gated, key=lambda t: t[2])
        if best is None:
            best = max(candidates, key=lambda t: t[2])

        self._last_pos = (best[0], best[1])
        return int(round(best[0])), int(round(best[1]))
