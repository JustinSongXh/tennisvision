"""Audio hit detection prototype — pure rule-based, no ML.

Extracts audio from video, detects impulsive events via spectral flux,
classifies hit vs bounce by high/low frequency energy ratio.

Usage:
    python scripts/test_audio_hits.py --video samples/sample_short.mp4
"""

import argparse
import subprocess
import tempfile
import os

import numpy as np
import librosa
from scipy.signal import butter, sosfilt

# ---- CONFIG ----
SR = 44100                # sample rate
BANDPASS_LO = 150         # Hz — removes wind, foot thuds
BANDPASS_HI = 8000        # Hz — removes phone mic noise
ONSET_HOP = 128           # ~2.9 ms resolution
ONSET_FFT = 256           # ~5.8 ms window
ONSET_FMIN = 200          # Hz
ONSET_FMAX = 8000         # Hz
ONSET_MELS = 64
ONSET_DELTA = 0.3         # prominence threshold (tune per video)
ONSET_WAIT = 70           # ~200 ms min gap at hop=128
ENERGY_PERCENTILE = 75    # keep only top N% energy events


def extract_audio(video_path):
    """Extract mono audio from video via ffmpeg."""
    tmp = tempfile.mktemp(suffix=".wav")
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-ac", "1", "-ar", str(SR), tmp,
    ], capture_output=True)
    y, _ = librosa.load(tmp, sr=SR, mono=True)
    os.unlink(tmp)
    return y


def bandpass_filter(y, lo, hi, sr, order=5):
    """Apply Butterworth bandpass filter."""
    sos = butter(order, [lo, hi], btype='band', fs=sr, output='sos')
    return sosfilt(sos, y)


def detect_onsets(y, sr):
    """Detect impulsive audio events via spectral flux onset detection."""
    onset_env = librosa.onset.onset_strength(
        y=y, sr=sr,
        hop_length=ONSET_HOP,
        n_fft=ONSET_FFT,
        fmin=ONSET_FMIN,
        fmax=ONSET_FMAX,
        n_mels=ONSET_MELS,
    )
    onsets = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=ONSET_HOP,
        pre_max=3,
        post_max=3,
        pre_avg=10,
        post_avg=10,
        delta=ONSET_DELTA,
        wait=ONSET_WAIT,
    )
    times = librosa.frames_to_time(onsets, sr=sr, hop_length=ONSET_HOP)
    return times, onset_env


def filter_by_energy(y, sr, times, percentile):
    """Keep only events above energy percentile threshold."""
    half_win = int(0.025 * sr)  # 25ms each side
    energies = []
    for t in times:
        center = int(t * sr)
        start, end = max(0, center - half_win), min(len(y), center + half_win)
        rms = np.sqrt(np.mean(y[start:end] ** 2))
        energies.append(rms)
    energies = np.array(energies)
    if len(energies) == 0:
        return times, energies
    thresh = np.percentile(energies, percentile)
    mask = energies >= thresh
    return times[mask], energies[mask]


def compute_event_energy(y, sr, times):
    """Compute RMS energy for each event."""
    half_win = int(0.025 * sr)
    results = []
    for t in times:
        center = int(t * sr)
        start, end = max(0, center - half_win), min(len(y), center + half_win)
        segment = y[start:end]
        if len(segment) < 64:
            continue
        rms = float(np.sqrt(np.mean(segment ** 2)))
        results.append({
            "time": float(t),
            "frame": int(round(t * 30)),  # assume 30fps
            "rms": rms,
        })
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--delta", type=float, default=ONSET_DELTA,
                        help="Onset prominence threshold (default 0.3)")
    parser.add_argument("--energy-pct", type=float, default=ENERGY_PERCENTILE,
                        help="Energy percentile filter (default 90)")
    args = parser.parse_args()

    print(f"Extracting audio from {args.video}...")
    y_raw = extract_audio(args.video)
    duration = len(y_raw) / SR
    print(f"Audio: {duration:.1f}s, {SR} Hz, {len(y_raw)} samples")

    # Filter
    y = bandpass_filter(y_raw, BANDPASS_LO, BANDPASS_HI, SR)

    # Detect onsets
    times, onset_env = detect_onsets(y, SR)
    print(f"Raw onsets: {len(times)}")

    # Energy filter
    times, energies = filter_by_energy(y, SR, times, args.energy_pct)
    print(f"After energy filter (top {100-args.energy_pct:.0f}%): {len(times)}")

    # Compute energy for each event
    events = compute_event_energy(y, SR, times)
    print(f"Events with energy: {len(events)}")

    # Print events
    print(f"\n{'time':>8s}  {'frame':>6s}  {'rms':>8s}")
    print("-" * 28)
    for e in events:
        print(f"{e['time']:8.2f}s  {e['frame']:6d}  {e['rms']:8.5f}")

    # Summary stats
    if len(events) > 1:
        intervals = np.diff([e["time"] for e in events])
        print(f"\nEvent intervals: mean={np.mean(intervals):.2f}s "
              f"min={np.min(intervals):.2f}s max={np.max(intervals):.2f}s")
        print(f"Events per minute: {len(events) / duration * 60:.1f}")


if __name__ == "__main__":
    main()
