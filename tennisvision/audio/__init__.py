"""Audio analysis for tennis hit detection.

Extracts audio from video, detects impulsive events (ball hits, bounces)
via spectral-flux onset detection with bandpass filtering.

Heavy deps (librosa, scipy) are imported lazily.
"""

from .hit_detector import AudioHitDetector, AudioHitDetectorConfig, AudioEvent

__all__ = [
    "AudioHitDetector",
    "AudioHitDetectorConfig",
    "AudioEvent",
]
