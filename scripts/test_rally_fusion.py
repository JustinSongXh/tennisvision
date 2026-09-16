"""Test multi-signal rally detection on sample_short.mp4.

Uses audio onset detector + stroke events from Kaggle run to test
the fusion rally detector. Ball trajectory is simulated from
the existing pipeline's ball detection frames.

Usage:
    python scripts/test_rally_fusion.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tennisvision.audio import AudioHitDetector
from tennisvision.pipeline.rally_fusion import MultiSignalRallyDetector, FusionRallyConfig

VIDEO = "/Users/justinsong/WorkSpace/solo/samples/sample_short.mp4"
FPS = 30.0
TOTAL_FRAMES = 4503

# Stroke events from Kaggle run (merged, frame indices)
STROKE_FRAMES = [
    29, 69, 94, 114, 181, 279, 309, 484, 584, 648, 671, 733, 749, 783, 814,
    920, 966, 984, 1019, 1036, 1157, 1167, 1214, 1288, 1384, 1394, 1459, 1488,
    1509, 1558, 1792, 1819, 1842, 1929, 2251, 2394, 2444, 2501, 2541, 2584,
    2666, 2758, 3011, 3061, 3276, 3315, 3461, 3546, 3595, 3653, 3708, 3805,
    3820, 3858, 3913, 4058, 4096, 4103, 4178, 4218, 4449, 4473,
]

print("=== Step 1: Audio onset detection ===")
audio_det = AudioHitDetector()
audio_events = audio_det.detect_from_video(VIDEO, fps=FPS)
audio_frames = [e.frame for e in audio_events]
print(f"Audio onsets: {len(audio_frames)}")

print(f"\n=== Step 2: Stroke events (from Kaggle) ===")
print(f"Stroke events: {len(STROKE_FRAMES)}")

print(f"\n=== Step 3: Multi-signal rally detection ===")
# No ball trajectory for now — test with audio + stroke only
det = MultiSignalRallyDetector(FusionRallyConfig(
    w_ball=1.0,
    w_audio=1.0,
    w_stroke=1.0,
    min_duration_s=3.0,
))

# Test 1: audio + stroke only (no ball)
rallies_no_ball = det.detect(
    TOTAL_FRAMES, FPS,
    audio_onset_frames=audio_frames,
    stroke_event_frames=STROKE_FRAMES,
)
print(f"\nAudio + Stroke only: {len(rallies_no_ball)} rallies")
for r in rallies_no_ball:
    dur = (r.end_frame - r.start_frame) / FPS
    print(f"  Rally {r.idx}: frames [{r.start_frame}-{r.end_frame}] "
          f"({r.start_frame/FPS:.1f}s - {r.end_frame/FPS:.1f}s) "
          f"dur={dur:.1f}s")

# Test 2: stroke only
rallies_stroke = det.detect(
    TOTAL_FRAMES, FPS,
    stroke_event_frames=STROKE_FRAMES,
)
print(f"\nStroke only: {len(rallies_stroke)} rallies")
for r in rallies_stroke:
    dur = (r.end_frame - r.start_frame) / FPS
    print(f"  Rally {r.idx}: frames [{r.start_frame}-{r.end_frame}] "
          f"({r.start_frame/FPS:.1f}s - {r.end_frame/FPS:.1f}s) "
          f"dur={dur:.1f}s")

# Test 3: audio only
rallies_audio = det.detect(
    TOTAL_FRAMES, FPS,
    audio_onset_frames=audio_frames,
)
print(f"\nAudio only: {len(rallies_audio)} rallies")
for r in rallies_audio:
    dur = (r.end_frame - r.start_frame) / FPS
    print(f"  Rally {r.idx}: frames [{r.start_frame}-{r.end_frame}] "
          f"({r.start_frame/FPS:.1f}s - {r.end_frame/FPS:.1f}s) "
          f"dur={dur:.1f}s")
