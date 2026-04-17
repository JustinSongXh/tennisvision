"""CatBoost-based bounce detector.

Port of yastrebksv/TennisProject's `BounceDetector` (see upstream at
https://github.com/yastrebksv/TennisProject/blob/main/bounce_detector.py).
Trained on trajectory features — specifically, joint reversal of
x and y velocities plus forward/backward velocity asymmetry — the
model distinguishes real court bounces from hit apexes that a naive
y-peak detector confuses them with.

Requires: catboost, pandas, scipy (see requirements-ml.txt).

The upstream `.cbm` weight file is committed directly in the
TennisProject repo; see scripts/README or download instructions in
the survey doc.  Place at weights/bounce_catboost.cbm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

try:
    import catboost as ctb
    import pandas as pd
    from scipy.interpolate import CubicSpline
    from scipy.spatial import distance as _sp_distance
    _HAS_DEPS = True
except ImportError:
    _HAS_DEPS = False

import numpy as np

from .peak import BounceEvent


@dataclass
class CatBoostBounceConfig:
    weights: str = "weights/bounce_catboost.cbm"
    threshold: float = 0.45          # upstream default; CourtCheck uses 0.20
    nms_window: int = 1               # merge consecutive peaks within this many frames


class CatBoostBounceDetector:
    """Post-hoc (batch) bounce detector.  Takes a full champion track
    and returns all bounces; pairs well with the 2-pass pipeline."""

    NUM = 3                           # lag 1 and 2 -> matches trained features

    def __init__(self, cfg: CatBoostBounceConfig):
        if not _HAS_DEPS:
            raise RuntimeError(
                "catboost + pandas + scipy required for CatBoostBounceDetector — "
                "`pip install -r requirements-ml.txt`.")
        self.cfg = cfg
        self.model = ctb.CatBoostRegressor()
        self.model.load_model(cfg.weights)

    def detect(self, points: Sequence) -> list:
        """Given a (x, y, frame) sequence (x or y may be None for gaps),
        return a list of BounceEvent in ascending frame order."""
        x_ball = [None if p.x is None else float(p.x) for p in points]
        y_ball = [None if p.y is None else float(p.y) for p in points]
        frames = [int(p.frame) for p in points]

        x_ball, y_ball = self._smooth(list(x_ball), list(y_ball))
        feats, kept_frames = self._prepare_features(x_ball, y_ball)
        if len(feats) == 0:
            return []
        preds = self.model.predict(feats)
        ind = np.where(preds > self.cfg.threshold)[0]
        if len(ind) == 0:
            return []
        ind = self._nms(list(ind), preds)

        # Build BounceEvent list — map feature-row idx -> original frame idx,
        # then back to the (x, y) at that frame.
        frame_to_xy = {f: (x, y) for f, x, y in zip(frames, x_ball, y_ball)}
        out = []
        for i in ind:
            f = kept_frames[i]
            xy = frame_to_xy.get(f)
            if xy is None or xy[0] is None:
                continue
            out.append(BounceEvent(frame=f, x_img=float(xy[0]), y_img=float(xy[1])))
        out.sort(key=lambda e: e.frame)
        return out

    # ---- helpers ported from yastrebksv ----

    def _prepare_features(self, x_ball, y_ball):
        num = self.NUM
        eps = 1e-15
        df = pd.DataFrame({
            "frame": list(range(len(x_ball))),
            "x": x_ball, "y": y_ball,
        })
        for i in range(1, num):
            df[f"x_lag_{i}"]     = df["x"].shift(i)
            df[f"x_lag_inv_{i}"] = df["x"].shift(-i)
            df[f"y_lag_{i}"]     = df["y"].shift(i)
            df[f"y_lag_inv_{i}"] = df["y"].shift(-i)
            df[f"x_diff_{i}"]     = (df[f"x_lag_{i}"]     - df["x"]).abs()
            df[f"y_diff_{i}"]     =  df[f"y_lag_{i}"]     - df["y"]
            df[f"x_diff_inv_{i}"] = (df[f"x_lag_inv_{i}"] - df["x"]).abs()
            df[f"y_diff_inv_{i}"] =  df[f"y_lag_inv_{i}"] - df["y"]
            df[f"x_div_{i}"] = (df[f"x_diff_{i}"] / (df[f"x_diff_inv_{i}"] + eps)).abs()
            df[f"y_div_{i}"] =  df[f"y_diff_{i}"] / (df[f"y_diff_inv_{i}"] + eps)

        for i in range(1, num):
            df = df[df[f"x_lag_{i}"].notna()]
            df = df[df[f"x_lag_inv_{i}"].notna()]
        df = df[df["x"].notna()]

        cols = (
            [f"x_diff_{i}"     for i in range(1, num)] +
            [f"x_diff_inv_{i}" for i in range(1, num)] +
            [f"x_div_{i}"      for i in range(1, num)] +
            [f"y_diff_{i}"     for i in range(1, num)] +
            [f"y_diff_inv_{i}" for i in range(1, num)] +
            [f"y_div_{i}"      for i in range(1, num)]
        )
        return df[cols], list(df["frame"])

    def _smooth(self, x_ball, y_ball):
        interp = 5
        counter = 0
        is_none = [int(x is None) for x in x_ball]
        for i in range(interp, len(x_ball) - 1):
            if not x_ball[i] and sum(is_none[i - interp:i]) == 0 and counter < 3:
                x_ext, y_ext = self._extrapolate(
                    x_ball[i - interp:i], y_ball[i - interp:i])
                x_ball[i] = x_ext
                y_ball[i] = y_ext
                is_none[i] = 0
                if x_ball[i + 1] is not None:
                    d = _sp_distance.euclidean(
                        (x_ext, y_ext), (x_ball[i + 1], y_ball[i + 1]))
                    if d > 80:
                        x_ball[i + 1] = None
                        y_ball[i + 1] = None
                        is_none[i + 1] = 1
                counter += 1
            else:
                counter = 0
        return x_ball, y_ball

    @staticmethod
    def _extrapolate(xs_in, ys_in):
        xs = list(range(len(xs_in)))
        fx = CubicSpline(xs, xs_in, bc_type="natural")
        fy = CubicSpline(xs, ys_in, bc_type="natural")
        return float(fx(len(xs_in))), float(fy(len(ys_in)))

    @staticmethod
    def _nms(idxs, preds):
        out = [idxs[0]]
        for i in range(1, len(idxs)):
            if idxs[i] - idxs[i - 1] != 1:
                out.append(idxs[i])
            elif preds[idxs[i]] > preds[idxs[i - 1]]:
                out[-1] = idxs[i]
        return out
