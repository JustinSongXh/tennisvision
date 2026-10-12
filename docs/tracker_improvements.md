# Ball Tracker Improvements

Issue: #14

## Problem

Current `MultiTrackManager` (`tennisvision/ball/tracker.py`) has three weaknesses:

1. **Champion selection only counts points** — noise tracks (LED ads, logos, ball boys) can accumulate many observations and win over the real ball track.
2. **Speed validation uses px/frame** — a single threshold can't work across near/far court due to perspective distortion. Far-court real balls appear slow in pixels and get rejected; near-court noise appears fast enough and passes.
3. **No physics model** — the tracker has no concept of projectile motion. A real tennis ball follows parabolic arcs between bounces/hits; noise tracks don't.

## Current Architecture

```
WASB HRNet (per-frame)
    → candidate (x, y) list
    → MultiTrackManager.update()
        → Kalman filter (constant-velocity, 4-state: x, y, vx, vy)
        → greedy nearest-neighbour association (gate_px threshold)
        → track validated when: len(pts) >= min_len AND min_speed <= avg_speed <= max_speed
    → champion = validated track with most points
```

### Current Kalman model

State: `[x, y, vx, vy]`, constant-velocity transition (no gravity term).
Measurement: `[x, y]` only.

### Current champion selection

```python
# tracker.py:142
cands.sort(key=lambda t: (len(t.pts), -(frame_idx - t.last_det_frame)), reverse=True)
```

Primary sort: most observation points. Tiebreaker: most recently detected.

## Proposed Changes

### 1. Projectile-residual champion scoring

**Goal:** Penalize tracks whose trajectory doesn't resemble projectile motion.

**Method:** For each validated track, fit a parabola on a sliding window of recent points and compute the mean residual:

```
For points (t_i, x_i, y_i) in window of last N observations (N ~ 15-20):
    Fit x(t) = a0 + a1*t           (linear in x, horizontal)
    Fit y(t) = b0 + b1*t + b2*t^2  (quadratic in y, gravity)
    residual = mean sqrt((x_fit - x_i)^2 + (y_fit - y_i)^2)
```

Champion scoring becomes:

```
score = len(pts) / (1 + residual)
```

- Real ball: low residual (follows parabola between bounces) → high score
- Noise track: high residual (random jitter) → low score even with many points
- Bounce/hit points break the parabola, so the window should only span the current arc — reset on direction change or use a short window

**Edge cases:**
- Less than 5-6 points in window: not enough for reliable fit, fall back to point count only
- Ball just after bounce: residual spikes briefly, but the window slides past quickly
- Spin effects: negligible at this resolution, parabola is sufficient

### 2. Court-coordinate speed validation

**Goal:** Replace px/frame speed thresholds with m/s using homography projection.

**Method:** Pass `H_img_to_real` (3x3 matrix from calib.json) into the tracker. Project each track point to court meters, compute speed in m/s.

```python
def px_to_court(x, y, H):
    p = H @ [x, y, 1.0]
    return p[0] / p[2], p[1] / p[2]

# Between consecutive detections:
dx_m = court_x1 - court_x0
dy_m = court_y1 - court_y0
dt_s = (frame1 - frame0) / fps
speed_ms = hypot(dx_m, dy_m) / dt_s
```

Validation threshold: `min_speed_ms=3.0, max_speed_ms=90.0` (serve ~60 m/s, rally ~20-40 m/s, noise ~0 m/s).

**Caveat:** Homography maps to the ground plane. A ball in flight projects to its "shadow" position — the higher the ball, the larger the positional error. For speed validation (distinguishing real ball from static noise) this is acceptable:
- Static noise: projected speed ~ 0 m/s regardless of projection error
- Real ball: projected speed >> 3 m/s even with height distortion
- Threshold is intentionally wide (3-90 m/s) to tolerate the error

**Fallback:** When no homography is provided (e.g. calibration failed), fall back to current px/frame logic.

**Interface change to TrackerConfig:**

```python
@dataclass
class TrackerConfig:
    gate_px: float = 80.0
    max_gap_frames: int = 5
    min_len: int = 3
    min_speed_ms: float = 3.0      # was min_speed (px/frame)
    max_speed_ms: float = 90.0     # was max_speed (px/frame)
    min_speed_px: float = 5.0      # fallback when no homography
    max_speed_px: float = 200.0    # fallback when no homography
    render_tail: int = 45
```

### 3. Static track filtering

**Goal:** Early exclusion of stationary false tracks from champion candidates.

**Method:** Over a sliding window (e.g. last 10 observations), compute total displacement in court coordinates. If displacement < threshold (e.g. 0.5m), mark track as `static`.

```python
class Track:
    ...
    self.is_static: bool = False

    def _check_static(self, threshold_m=0.5, window=10):
        recent = self.pts[-window:]
        if len(recent) < 3:
            return
        # Project first and last point to court coords
        d = court_distance(recent[0], recent[-1])
        self.is_static = (d < threshold_m)
```

Static tracks are excluded from the champion candidate pool but not deleted (they still absorb their detections so those false positives don't spawn new tracks).

**Depends on:** Change 2 (court-coordinate projection).

### 4. Post-pass tracklet stitching

**Goal:** Merge broken tracklets caused by occlusion or brief out-of-frame exits.

**When:** Offline, after the full tracking pass completes (not during streaming).

**Method:** For each pair of tracklets (A ends, B starts within gap_limit):

```
1. Extrapolate A's last parabolic arc forward to B's start time
2. Compare extrapolated position with B's first observation
3. Compare velocity continuity (A's exit velocity vs B's entry velocity)
4. If position error < threshold AND velocity angle difference < threshold:
     merge B into A
```

**Constraints:**
- Maximum gap: 0.5 seconds (15 frames at 30fps)
- Do not merge across bounce points (large vertical velocity reversal)
- Only merge validated tracklets
- Process gaps chronologically to avoid transitive merge conflicts

**Interface:** New method `MultiTrackManager.stitch_tracklets()` called once after all frames are processed, before champion selection for JSON output.

## Evaluation

Run `extract_ball_positions.py` before/after on existing test videos. Compare:

| Metric | How to compute | Good direction |
|--------|---------------|----------------|
| Detection coverage | `n_detected / total_frames` per rally segment | Higher |
| Track continuity | Max consecutive detected frames in champion | Higher |
| Champion switches | Count changes in champion track id across frames | Lower |
| Champion residual | Mean projectile residual of champion track | Lower |
| Downstream: serve detection | Compare serve_events.json before/after | Fewer false positives/negatives |
| Downstream: rally detection | Compare rally_events.json before/after | Better boundaries |

No manual annotation needed — all metrics are computable from JSON outputs.

## Implementation Order

1. Projectile residual scoring (tracker.py) — highest value, self-contained
2. Court-coordinate speed (tracker.py + extract_ball_positions.py) — needs calib passed through
3. Static filtering (tracker.py) — depends on 2
4. Tracklet stitching (tracker.py) — most complex, lowest priority

Changes 1-3 modify `Track` and `MultiTrackManager` in tracker.py.
Change 2 also touches `extract_ball_positions.py` (pass calib) and `run_full_pipeline.py` (pipe calib path).
Change 4 adds a new post-processing method.
