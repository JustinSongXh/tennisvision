from .classical import HSVMotionDetector, Candidate
from .tracker import MultiTrackManager, Track, TrackerConfig, TrackPoint

# WASB pulls in torch — lazy-imported so the package still works on
# torch-less environments as long as the user sticks with 'classical'.
try:
    from .wasb import WASBBallDetector, WASBConfig
    _HAS_WASB = True
except Exception:
    _HAS_WASB = False
    WASBBallDetector = None
    WASBConfig = None

__all__ = [
    "HSVMotionDetector", "Candidate",
    "MultiTrackManager", "Track", "TrackerConfig", "TrackPoint",
    "WASBBallDetector", "WASBConfig",
]
