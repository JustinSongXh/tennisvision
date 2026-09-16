# Audio Hit Detection Design

Issue: #06

## Goal

Detect tennis ball hit events (racket contact) from video audio track, as a
complementary signal for rally detection — especially when ball tracking or
far-side player detection fails.

## Tennis Hit Acoustic Signature

| Property | Racket Hit | Court Bounce | Ambient/Noise |
|----------|-----------|--------------|---------------|
| Duration | 5-15ms impulse | 3-8ms impulse | Continuous |
| Frequency | 250-1100Hz primary; 500Hz string mode; 8-12kHz ball deformation | Below 500Hz, duller | Broadband < 2kHz |
| Energy | Sharp attack, fast decay | Duller attack | Sustained, no transients |

Key discriminator: racket hits have **stronger high-frequency content** (string
vibration + ball deformation) while bounces are duller with energy < 500Hz.

## Pipeline (3 stages)

### Stage 1: Preprocessing

```
Input:  video audio track (ffmpeg extract)
Output: mono waveform, filtered

- Sample rate: 44,100 Hz (or 16,000 Hz for lighter processing)
- Channels: mono
- Bandpass filter: 5th-order Butterworth, 150 Hz - 8000 Hz
  - 150 Hz HPF: removes wind, foot thuds, court rumble
  - 8 kHz LPF: removes phone mic noise
```

### Stage 2: Candidate Detection (energy-based, no ML)

```
- Compute onset strength envelope via librosa:
    onset_env = librosa.onset.onset_strength(
        y=y_filtered, sr=sr,
        hop_length=128,       # ~2.9 ms resolution
        n_fft=256,            # ~5.8 ms window
        fmin=200, fmax=8000,
        n_mels=64
    )

- Peak picking:
    onsets = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=128,
        pre_max=3, post_max=3,     # ~8.7 ms local max window
        pre_avg=10, post_avg=10,   # ~29 ms averaging window
        delta=0.3,                 # prominence threshold (tune per video)
        wait=70                    # ~200 ms min gap between events
    )

- Energy filter: keep only top N% energy events (e.g., 95th percentile)
  to reject weak ambient onsets
```

### Stage 3: Classification

**Option A — No ML (simplest, ~200 lines)**
```
For each candidate, extract 50ms window:
- Compute FFT
- lo_energy = mean(|FFT| in 100-500 Hz)
- hi_energy = mean(|FFT| in 1000-5000 Hz)
- ratio = hi / lo
- If ratio > 1.5 → "hit", else → "bounce"

Expected: ~70-85% precision, ~60-75% recall
```

**Option B — Small CNN on Mel spectrogram (best accuracy)**
```
For each candidate, extract 50ms clip:
- Mel spectrogram: 64 bands, n_fft=256, hop=128, fmin=100, fmax=12000
- Feed into 2-3 conv layer CNN → softmax(hit, bounce, noise)
- Reference: IBM US Open system F1=92.39%
- Training data: ~1000 labeled clips (see bootstrap below)
```

**Option C — Pretrained embeddings + linear head**
```
- Use PANNs CNN14 or YAMNet as feature extractor
- Fine-tune a small classifier head on labeled tennis data
- Needs very few labeled samples (~100-200)
```

## Interference Handling (Outdoor Amateur Courts)

### Adjacent courts / Pickleball
- **Amplitude gating**: neighboring hits arrive attenuated; energy percentile
  filter (top 5%) naturally rejects them
- **Temporal cross-validation**: audio hit must coincide (±100ms) with ball
  trajectory change or stroke classifier activation

### Wind noise
- Bandpass 150-8000 Hz handles most wind
- Strong wind: raise HPF to 250-300 Hz (sacrifices some bounce detection)

### Phone mic AGC
- Auto gain control makes fixed thresholds unreliable
- Use **relative** thresholds (percentile-based, per-rally normalization)

### People talking
- Speech is 300-3000 Hz, overlaps with hit band
- But speech has gradual onset; onset_strength favors sharp transients
- Post-filter with ball tracker timestamps removes most false positives

## Labeling Strategy (Bootstrap from Video)

1. Extract audio from video via ffmpeg
2. Use existing stroke classifier timestamps as weak labels (±100ms)
3. Run energy-based detector on full audio
4. Match: audio event within 100ms of stroke → **hit**;
   near bounce detection → **bounce**; far from both → **noise**
5. Extract 50ms clips per labeled event
6. Manual review ~100 clips per class (~15-20 min)
7. After 3-5 videos → ~1000+ labeled samples → train CNN

## Integration with Rally Detection

Audio hits feed into the fusion layer alongside ball trajectory and pose:

```
Cascaded fallback (Phase 1, no training):
  if ball_trajectory.confidence > 0.7:  event = ball_trajectory
  elif audio_hit.confidence > 0.6:      event = audio_hit
  else:                                 event = pose_stroke

Weighted late fusion (Phase 2):
  score = 0.5 * ball + 0.3 * audio + 0.2 * pose

Gated fusion (Phase 3, needs labeled data):
  GMU with modality dropout training
```

Two rally hits within crossing_silence_seconds → rally is active.
Audio is especially valuable when:
- Far-side ball tracking is lost
- Far-side player detection fails
- Ball is occluded by net or player

## References

- [IBM US Open 2019 (Baughman et al.)](https://shiqiang.wang/papers/AB_MMSports2019.pdf) — CNN + MFCC, F1=92.39%
- [tt_sounds (Tubingen)](https://github.com/cogsys-tuebingen/tt_sounds) — table tennis bounce detection, full code + dataset
- [Yamamoto et al.](https://www.scitepress.org/Papers/2020/101076/101076.pdf) — tennis spin from sound, 250-1100Hz band
- [librosa onset_detect](https://librosa.org/doc/main/generated/librosa.onset.onset_detect.html)
- [Audiolabs onset detection tutorial](https://www.audiolabs-erlangen.de/resources/MIR/FMP/C6/C6S1_OnsetDetection.html)

## Implementation Plan

1. **Prototype** (`scripts/test_audio_hits.py`):
   - Extract audio from sample_short.mp4
   - Run no-ML spectral-flux detector (Option A)
   - Output: list of (timestamp, type, confidence) + annotated waveform plot
   - Tune delta/threshold on this one video

2. **Bootstrap labels**:
   - Match audio detections with stroke classifier timestamps from Kaggle run
   - Generate labeled 50ms clips

3. **Train CNN** (if Option A accuracy insufficient):
   - Use tt_sounds repo as code template
   - 64-band Mel spectrogram, 2-3 conv layers
   - Train on bootstrapped labels

4. **Integrate into pipeline**:
   - Add `tennisvision/audio/` module
   - Wire into rally state machine as additional signal
   - Cascaded fallback first, learned fusion later
