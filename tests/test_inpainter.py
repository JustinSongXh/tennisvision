"""Tests for tennisvision.ball.inpainter.

Focus on the pure-python helpers (mask generation, long-run clipping,
track merging) since those are deterministic.  The model inference
itself is smoke-tested with a random-init network — enough to guarantee
shape/dtype plumbing is correct, without requiring pretrained weights
that live outside git.

Runs under plain `python3 -m unittest tests.test_inpainter` or under
pytest — the test bodies use stdlib assertions only.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tennisvision.ball.inpainter import (
    InpaintNetConfig,
    InpaintResult,
    TrajectoryInpainter,
    _clip_long_runs,
    generate_inpaint_mask,
    result_to_trackpoints,
)
from tennisvision.ball.tracker import Track, TrackPoint


# ----------------------------------------------------------------------
# generate_inpaint_mask
# ----------------------------------------------------------------------

def test_mask_fills_internal_gap():
    # ball visible, disappears for 3 frames, reappears
    vis = np.array([1, 1, 0, 0, 0, 1, 1], dtype=np.float32)
    y   = np.array([100, 110, 0, 0, 0, 130, 140], dtype=np.float32)
    mask = generate_inpaint_mask(vis, y, th_h=20)
    np.testing.assert_array_equal(mask, [0, 0, 1, 1, 1, 0, 0])


def test_mask_skips_out_of_view_gap():
    # gap bordered by detections too close to top (y < th_h) — ball is
    # plausibly above the frame, do NOT fabricate coordinates
    vis = np.array([1, 1, 0, 0, 1, 1], dtype=np.float32)
    y   = np.array([100, 10, 0, 0, 8, 100], dtype=np.float32)
    mask = generate_inpaint_mask(vis, y, th_h=20)
    np.testing.assert_array_equal(mask, [0, 0, 0, 0, 0, 0])


def test_mask_skips_trailing_gap():
    # sequence ends with zeros — don't fill, we don't know where the ball went
    vis = np.array([1, 1, 1, 0, 0, 0], dtype=np.float32)
    y   = np.array([100, 110, 120, 0, 0, 0], dtype=np.float32)
    mask = generate_inpaint_mask(vis, y, th_h=20)
    np.testing.assert_array_equal(mask, [0, 0, 0, 0, 0, 0])


def test_mask_leading_gap_filled_if_ball_appears_below_edge():
    vis = np.array([0, 0, 1, 1], dtype=np.float32)
    y   = np.array([0, 0, 100, 110], dtype=np.float32)
    mask = generate_inpaint_mask(vis, y, th_h=20)
    np.testing.assert_array_equal(mask, [1, 1, 0, 0])


def test_mask_empty():
    assert generate_inpaint_mask(np.zeros(0), np.zeros(0), th_h=10).shape == (0,)


# ----------------------------------------------------------------------
# _clip_long_runs
# ----------------------------------------------------------------------

def test_clip_keeps_short_runs():
    m = np.array([0, 1, 1, 0, 1, 1, 1, 0], dtype=np.float32)
    np.testing.assert_array_equal(_clip_long_runs(m, max_len=3), m)


def test_clip_zeros_long_runs():
    m = np.array([0, 1, 1, 1, 1, 1, 0, 1, 1, 0], dtype=np.float32)
    out = _clip_long_runs(m, max_len=3)
    np.testing.assert_array_equal(out, [0, 0, 0, 0, 0, 0, 0, 1, 1, 0])


# ----------------------------------------------------------------------
# merge_tracks
# ----------------------------------------------------------------------

def _track(pts):
    t = Track(pts[0][0], pts[0][1], pts[0][2],
              min_len=3, min_speed=1.0, max_speed=200.0)
    for x, y, f in pts[1:]:
        t.pts.append(TrackPoint(x, y, f))
    return t


def test_merge_tracks_union():
    t1 = _track([(10, 10, 5), (11, 11, 6), (12, 12, 7)])
    t2 = _track([(50, 50, 9), (51, 51, 10)])
    lo, hi, xs, ys, vis = TrajectoryInpainter.merge_tracks([t1, t2])
    assert (lo, hi) == (5, 10)
    # frame 8 should be NaN (gap between the two tracks)
    np.testing.assert_array_equal(vis, [1, 1, 1, 0, 1, 1])
    assert np.isnan(xs[3]) and np.isnan(ys[3])
    assert xs[0] == 10 and xs[4] == 50


def test_merge_tracks_first_writer_wins():
    t1 = _track([(10, 10, 5), (11, 11, 6)])
    t2 = _track([(99, 99, 6), (100, 100, 7)])   # overlapping frame 6
    _, _, xs, ys, _ = TrajectoryInpainter.merge_tracks([t1, t2])
    # t1 wrote frame 6 first; t2 must not overwrite
    assert xs[1] == 11 and ys[1] == 11


def test_merge_empty():
    lo, hi, xs, ys, vis = TrajectoryInpainter.merge_tracks([])
    assert hi < lo and len(xs) == 0


# ----------------------------------------------------------------------
# result_to_trackpoints
# ----------------------------------------------------------------------

def test_result_to_trackpoints_drops_nan():
    xs = np.array([10, np.nan, 30], dtype=np.float32)
    ys = np.array([20, np.nan, 40], dtype=np.float32)
    res = InpaintResult(frame_lo=100, xs=xs, ys=ys,
                        was_inpainted=np.array([False, False, True]))
    pts = result_to_trackpoints(res)
    assert [(p.x, p.y, p.frame) for p in pts] == [(10, 20, 100), (30, 40, 102)]


# ----------------------------------------------------------------------
# Model smoke test (random-init weights — checks plumbing, not accuracy)
# ----------------------------------------------------------------------

def _make_random_init_inpainter(tmp_dir: Path, frame_size=(1280, 720), seq_len=16):
    try:
        import torch
    except ImportError:
        raise unittest.SkipTest("torch not installed")
    from tennisvision.ball.inpainter import InpaintNet
    model = InpaintNet()
    ckpt_path = tmp_dir / "rand_inpaint.pt"
    torch.save(
        {"model": model.state_dict(), "param_dict": {"seq_len": seq_len}},
        ckpt_path,
    )
    cfg = InpaintNetConfig(weights=str(ckpt_path), device="cpu",
                           window_mode="single")
    return TrajectoryInpainter(cfg, frame_size=frame_size)


def test_inpaint_shape_and_post_rule():
    """Masked positions must take the model's output; unmasked positions
    must preserve the original value exactly."""
    with tempfile.TemporaryDirectory() as td:
        inp = _make_random_init_inpainter(Path(td))
    # parabolic synthetic trajectory with a 4-frame hole in the middle
    L = 40
    t = np.arange(L, dtype=np.float32)
    xs = 50 + 20 * t
    ys = 200 + 0.5 * (t - 20) ** 2                 # bowl-shaped
    vis = np.ones(L, dtype=np.float32)
    hole = slice(18, 22)
    xs[hole] = np.nan
    ys[hole] = np.nan
    vis[hole] = 0

    res = inp.inpaint_series(lo=0, xs=xs, ys=ys, vis=vis)

    assert res.xs.shape == (L,) and res.ys.shape == (L,)
    # Unmasked positions preserved (no NaN outside the hole)
    non_hole = np.ones(L, dtype=bool); non_hole[hole] = False
    np.testing.assert_array_equal(res.xs[non_hole], xs[non_hole])
    np.testing.assert_array_equal(res.ys[non_hole], ys[non_hole])
    # Hole positions filled with finite numbers
    assert np.isfinite(res.xs[hole]).all()
    assert np.isfinite(res.ys[hole]).all()
    assert res.was_inpainted[hole].all()
    assert not res.was_inpainted[non_hole].any()


def test_inpaint_skips_out_of_view_hole():
    """If the hole is bordered by detections near the top of frame, the
    mask builder should leave the hole untouched (xs/ys stay NaN)."""
    with tempfile.TemporaryDirectory() as td:
        inp = _make_random_init_inpainter(Path(td), frame_size=(1280, 720))
    # th_h = 0.05 * 720 = 36 px; borders at y=10 and y=12 are "out of view"
    L = 10
    xs = np.array([100, 110, np.nan, np.nan, np.nan, 150, 160,
                   170, 180, 190], dtype=np.float32)
    ys = np.array([10, 12, np.nan, np.nan, np.nan, 14, 15,
                   100, 110, 120], dtype=np.float32)
    vis = (~np.isnan(xs)).astype(np.float32)
    res = inp.inpaint_series(lo=0, xs=xs, ys=ys, vis=vis)
    assert np.isnan(res.xs[2:5]).all() and np.isnan(res.ys[2:5]).all()
    assert not res.was_inpainted[2:5].any()


def test_long_gap_not_filled():
    """Gaps longer than max_gap_frames must be left untouched."""
    with tempfile.TemporaryDirectory() as td:
        inp = _make_random_init_inpainter(Path(td))
    inp.cfg.max_gap_frames = 3
    L = 20
    xs = np.arange(L, dtype=np.float32) * 5 + 100
    ys = np.arange(L, dtype=np.float32) * 3 + 200
    hole = slice(5, 12)                 # 7-frame hole > max_gap_frames=3
    xs[hole] = np.nan
    ys[hole] = np.nan
    vis = (~np.isnan(xs)).astype(np.float32)

    res = inp.inpaint_series(lo=0, xs=xs, ys=ys, vis=vis)
    assert not res.was_inpainted[hole].any()
    assert np.isnan(res.xs[hole]).all()


if __name__ == "__main__":
    # Minimal runner so this file works with plain `python3 tests/test_inpainter.py`
    # on environments without pytest.  Discovers module-level test_* functions.
    import sys
    import traceback
    ns = dict(globals())
    names = sorted(n for n in ns if n.startswith("test_") and callable(ns[n]))
    fails = 0
    for name in names:
        try:
            ns[name]()
            print("ok     " + name)
        except unittest.SkipTest as e:
            print("skip   %s (%s)" % (name, e))
        except AssertionError:
            fails += 1
            print("FAIL   " + name)
            traceback.print_exc()
        except Exception:
            fails += 1
            print("ERROR  " + name)
            traceback.print_exc()
    print("\n%d/%d passed" % (len(names) - fails, len(names)))
    sys.exit(1 if fails else 0)
