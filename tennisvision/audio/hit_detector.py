"""Audio onset-based hit candidate detector.

Two-stage pipeline:
  1. Extract audio from video → bandpass filter (150-8000 Hz)
  2. Spectral-flux onset detection → energy filter → hit candidates

Output is a list of AudioEvent with timestamps and RMS energy.
Precision is low on its own (~50%); designed to be fused with visual
signals (ball trajectory, stroke classifier) for filtering.
Recall is high (~97% with default settings).
"""

from __future__ import annotations

import subprocess
import tempfile
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class AudioHitDetectorConfig:
    sample_rate: int = 44100
    bandpass_lo: int = 150       # Hz — removes wind, foot thuds
    bandpass_hi: int = 8000      # Hz — removes phone mic noise
    filter_order: int = 5        # Butterworth filter order
    onset_hop: int = 128         # ~2.9 ms resolution at 44.1kHz
    onset_fft: int = 256         # ~5.8 ms window
    onset_fmin: int = 200        # Hz — onset strength lower bound
    onset_fmax: int = 8000       # Hz — onset strength upper bound
    onset_mels: int = 64         # Mel bands for onset envelope
    onset_delta: float = 0.15    # prominence threshold (lower = more candidates)
    onset_wait: int = 70         # ~200 ms min gap between events at hop=128
    energy_percentile: float = 50.0  # keep top N% energy events (0 = keep all)


@dataclass
class AudioEvent:
    time: float          # seconds
    frame: int           # video frame index (at given fps)
    rms: float           # RMS energy of the event window


class AudioHitDetector:
    """Detect hit candidates from video audio track.

    Usage:
        det = AudioHitDetector(AudioHitDetectorConfig())
        events = det.detect_from_video("match.mp4", fps=30.0)
    """

    def __init__(self, cfg: Optional[AudioHitDetectorConfig] = None):
        self.cfg = cfg or AudioHitDetectorConfig()

    def extract_audio(self, video_path: str) -> np.ndarray:
        """Extract mono audio from video via ffmpeg."""
        try:
            import librosa  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "AudioHitDetector needs librosa: `pip install librosa`"
            ) from e

        fd, tmp = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", video_path,
                 "-vn", "-ac", "1", "-ar", str(self.cfg.sample_rate), tmp],
                capture_output=True, check=True,
            )
            y, _ = librosa.load(tmp, sr=self.cfg.sample_rate, mono=True)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        return y

    def bandpass_filter(self, y: np.ndarray) -> np.ndarray:
        """Apply Butterworth bandpass filter."""
        from scipy.signal import butter, sosfilt  # type: ignore

        sos = butter(
            self.cfg.filter_order,
            [self.cfg.bandpass_lo, self.cfg.bandpass_hi],
            btype="band", fs=self.cfg.sample_rate, output="sos",
        )
        return sosfilt(sos, y)

    def detect_onsets(self, y: np.ndarray) -> np.ndarray:
        """Detect impulsive events via spectral flux."""
        import librosa  # type: ignore

        onset_env = librosa.onset.onset_strength(
            y=y, sr=self.cfg.sample_rate,
            hop_length=self.cfg.onset_hop,
            n_fft=self.cfg.onset_fft,
            fmin=self.cfg.onset_fmin,
            fmax=self.cfg.onset_fmax,
            n_mels=self.cfg.onset_mels,
        )
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=self.cfg.sample_rate,
            hop_length=self.cfg.onset_hop,
            pre_max=3, post_max=3,
            pre_avg=10, post_avg=10,
            delta=self.cfg.onset_delta,
            wait=self.cfg.onset_wait,
        )
        return librosa.frames_to_time(
            onsets, sr=self.cfg.sample_rate, hop_length=self.cfg.onset_hop,
        )

    def filter_by_energy(
        self, y: np.ndarray, times: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Keep only events above energy percentile."""
        if self.cfg.energy_percentile <= 0 or len(times) == 0:
            rms = np.array([self._event_rms(y, t) for t in times])
            return times, rms

        rms = np.array([self._event_rms(y, t) for t in times])
        thresh = np.percentile(rms, self.cfg.energy_percentile)
        mask = rms >= thresh
        return times[mask], rms[mask]

    def _event_rms(self, y: np.ndarray, t: float) -> float:
        sr = self.cfg.sample_rate
        half_win = int(0.025 * sr)  # 25ms each side
        center = int(t * sr)
        start = max(0, center - half_win)
        end = min(len(y), center + half_win)
        segment = y[start:end]
        if len(segment) == 0:
            return 0.0
        return float(np.sqrt(np.mean(segment ** 2)))

    def detect(self, y_raw: np.ndarray, fps: float = 30.0) -> List[AudioEvent]:
        """Run full pipeline on raw audio waveform.

        Args:
            y_raw: mono audio at self.cfg.sample_rate
            fps: video frame rate for frame index calculation

        Returns:
            list of AudioEvent sorted by time
        """
        y = self.bandpass_filter(y_raw)
        times = self.detect_onsets(y)
        times, rms = self.filter_by_energy(y, times)
        return [
            AudioEvent(
                time=float(t),
                frame=int(round(t * fps)),
                rms=float(r),
            )
            for t, r in zip(times, rms)
        ]

    def detect_from_video(
        self, video_path: str, fps: float = 30.0,
    ) -> List[AudioEvent]:
        """End-to-end: extract audio from video and detect hits."""
        y_raw = self.extract_audio(video_path)
        return self.detect(y_raw, fps=fps)
