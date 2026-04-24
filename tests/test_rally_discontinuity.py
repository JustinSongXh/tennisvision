"""Smoke tests for the OnlineRallyDetector discontinuity-close gate.

Verify that a big position jump + motion-direction flip + no recent
net crossing fires the close, while tracker-continuous flips (bounces,
lobs) and jumps right after a real crossing do NOT.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)

from tennisvision.pipeline.rally import OnlineRallyDetector


# Test footage is synthetic, but we mirror sample_short.mp4's 30fps so
# the time-to-frames conversions below are readable.  Frame constants
# are NEVER hardcoded — the time constants below are the source of
# truth and every integer frame count is derived from them.
_TEST_FPS = 30.0
_DISCONTINUITY_NO_CROSS_SECONDS = 1.0       # shipping default
_CROSSING_SILENCE_SECONDS = 4.0


def _sec(seconds: float) -> int:
    return int(round(seconds * _TEST_FPS))


class _FakeChamp:
    def __init__(self, x: float, y: float, f: int) -> None:
        class _P:  # noqa: D401
            pass
        p = _P()
        p.x = x
        p.y = y
        p.frame = f
        self.pts = [p]
        self.last_det_frame = f


def _mk_det() -> OnlineRallyDetector:
    # net_y_px matches sample_short.mp4 (669).  Time constants come
    # from the shipping config in seconds — frames are derived via
    # _sec() so the intent stays readable and the test would still
    # express the right durations if `_TEST_FPS` changed.
    return OnlineRallyDetector(
        net_y_px=669,
        silence_thresh_frames=9000,
        min_net_crossings=3,
        pre_roll_frames=0,
        post_roll_frames=0,
        total_frames=100_000,
        crossing_silence_thresh_frames=_sec(_CROSSING_SILENCE_SECONDS),
        discontinuity_jump_px=120.0,
        discontinuity_no_cross_frames=_sec(_DISCONTINUITY_NO_CROSS_SECONDS),
    )


def _drive(det: OnlineRallyDetector, seq: list) -> None:
    for (f, x, y) in seq:
        if x is None:
            det.observe(f, [], None)
        else:
            det.observe(f, [(x, y)], _FakeChamp(x, y, f))


def test_real_rally_bounces_do_not_trigger() -> None:
    """Tracker-continuous rally (direction flips at bounces / apex
    but frame-to-frame position jumps stay under 80 px) must NOT
    fire the discontinuity close.
    """
    d = _mk_det()
    y_path = [
        750, 720, 690, 660, 630, 600, 570, 540, 510, 480, 460,
        440, 440, 460, 490, 530, 580, 640, 700, 770, 820,
        800, 760, 720, 680, 640, 600, 560, 520, 490, 470, 460,
        450, 470, 510, 560, 620, 700, 780, 830,
    ]
    seq = [
        (f, 500 + 30 * (f % 3), y_path[f])
        for f in range(len(y_path))
    ]
    _drive(d, seq)
    assert d._discontinuity_closes == 0


def test_lob_does_not_trigger() -> None:
    """A lob flips vy direction at apex but stays tracker-continuous
    (small frame-to-frame jumps).  Must NOT fire.
    """
    d = _mk_det()
    y_path = [
        800, 780, 755, 725, 690, 650, 610, 570, 530, 490, 450, 420,
        405, 405, 420, 450, 490, 540, 600, 670, 740, 800,
    ]
    seq = [(f, 960, y_path[f]) for f in range(len(y_path))]
    # Tail silence on our side, no crossings — stretches time since
    # last crossing past the no_cross gate, testing that direction
    # flip alone (with small jumps) still stays silent.  Long enough
    # (1.5s @ test fps) to clear `_DISCONTINUITY_NO_CROSS_SECONDS`.
    f_next = len(y_path)
    for i in range(_sec(1.5)):
        seq.append((f_next + i, 960, 830))
    _drive(d, seq)
    assert d._discontinuity_closes == 0


def test_pickup_toss_triggers() -> None:
    """A genuine between-point ball pickup: ball sits on our side
    after the rally ends, then a player throws it to a far position.
    Large jump + direction flip + time_since_cross > gate → fire.
    """
    d = _mk_det()
    seq: list = []
    rally_y = [800, 750, 700, 650, 600, 550, 500, 550, 600, 650, 700, 750, 800]
    f = 0
    for y_val in rally_y:
        seq.append((f, 500 + (f * 15) % 100, y_val))
        f += 1
    # Between-point quiet: ball hovers on our side, no crossings.
    # Long enough (1.5s @ test fps) to clear the no_cross gate.
    for i in range(_sec(1.5)):
        seq.append((f, 600, 820 + (i % 5)))
        f += 1
    # Teleport: ball jumps from ~x=600 to x=1400 (~800px jump), x
    # direction flips from +x to -x, and time_since_cross is already
    # past `_DISCONTINUITY_NO_CROSS_SECONDS`.
    seq.append((f, 1400, 700))
    f += 1
    for i in range(5):
        seq.append((f, 1400 - i * 20, 700 + i * 15))
        f += 1
    _drive(d, seq)
    assert d._discontinuity_closes == 1


def test_jump_right_after_crossing_does_not_trigger() -> None:
    """A big jump AND direction flip that happen within the
    no-cross window of a real crossing must NOT fire — real strokes
    can introduce local discontinuities right at the hit.
    """
    d = _mk_det()
    seq = []
    f = 0
    for (x, y) in [(500, 800), (520, 780), (540, 740),
                   (560, 680), (580, 600), (600, 500)]:
        seq.append((f, x, y))
        f += 1
    # Immediately after the crossing, inject a jump + direction flip.
    seq.append((f, 450, 600))
    _drive(d, seq)
    assert d._discontinuity_closes == 0


if __name__ == "__main__":
    for fn in (
        test_real_rally_bounces_do_not_trigger,
        test_lob_does_not_trigger,
        test_pickup_toss_triggers,
        test_jump_right_after_crossing_does_not_trigger,
    ):
        fn()
        print(fn.__name__, "ok")
