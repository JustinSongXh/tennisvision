# CLAUDE.md — TennisVision

## Project overview

Tennis video analysis pipeline: court calibration, ball detection & tracking, player pose extraction, serve detection, rally segmentation, and optional annotated video rendering. Single fixed-camera input, outputs structured JSON + rally cut videos.

## Repository layout

```
scripts/          CLI entry points
  run_full_pipeline.py   5-step orchestrator (primary workflow)
  calibrate.py           Court calibration
  extract_keypoints.py   Player detection + pose (yolo11m + yolo26s-pose)
  extract_ball_positions.py  Ball detection + Kalman tracker (WASB HRNet)
  detect_serves.py       GRU v4 serve classification
  detect_rallies.py      Rally fusion + video cutting
  analyze.py             Integrated two-pass pipeline (with rendering)
tennisvision/     Main package
  ball/           Ball detection (WASB HRNet, classical HSV) + Kalman tracker + InpaintNet
  bounce/         Bounce detection (CatBoost, peak)
  court/          Court detection, homography, calibration
  action/         Player pose (YOLO-pose) + slot mapper + GRU classifier
  pipeline/       Two-pass orchestration + rally state machines (rally.py, rally_fusion.py)
  render/         Trail, minimap, HUD, player overlay
configs/          YAML config overrides (local_run, remote_inpaint, etc.)
weights/          Model checkpoints (gitignored — download separately)
tests/            Unit tests
docs/             Design docs and surveys
```

## Key conventions

### Branching
- Work branches follow the pattern `issue-XX` where XX is the GitHub issue number (e.g. `issue-06`).
- Before starting new work, check if an existing issue already covers the requirement — reuse it if so, otherwise create a new issue first.
- Always work on an `issue-XX` branch, not directly on `master`.

### Commits
- Never add `Co-Authored-By: Claude ...` or any assistant trailer to commit messages.
- One commit per concern — split feat/fix/tuning into separate commits even when small.
- Use conventional prefix with capitalized first word after colon: `feat: Add ...`, `fix: Correct ...`, `refactor: Extract ...`, `chore: Update ...`, `docs: Add ...`.
- First line MUST end with ` #XX` where XX is the issue number (e.g. `feat: Add CLAUDE.md #11`).

### Code style
- Temporal thresholds MUST be in **seconds** at the config/API surface. Frame counts are derived at runtime via `int(round(seconds * fps))`. Never hardcode frame counts.
- Pipeline outputs go to `results/`, never `samples/`. Clear `results/` before each run.
- Weights are NOT tracked in git. Keep them in local `./weights/` (gitignored).

### Running the pipeline

Standalone 5-step pipeline (primary):
```bash
python scripts/run_full_pipeline.py \
    --video samples/sample2.mp4 \
    --weights-dir weights/
```

Integrated two-pass pipeline (with rendering):
```bash
cd /Users/justinsong/WorkSpace/solo/tennisvision && \
/Users/justinsong/WorkSpace/solo/venv/bin/python -u scripts/analyze.py \
    --video <path>.mp4 --calib <path>.json \
    --config configs/local_run.yaml \
    --out /Users/justinsong/WorkSpace/solo/results/<name>.mp4
```

Remote (10.13.9.247, GPU):
```bash
ssh root@10.13.9.247
cd /root/solo/tennisvision
python -u scripts/run_full_pipeline.py \
    --video /root/solo/samples/<video>.mp4 \
    --weights-dir /root/solo/weights/
```

For long runs use `nohup ... > log 2>&1 &` and `caffeinate -i -s -w <PID> &` locally.

### Test videos
- `sample2.mp4` — doubles (4 players), use `results/sample2/calib.json`
- `sample3.mp4` — TBD

### Remote machine notes
- Weights live at `/root/solo/` (outside repo tree), scp from local `./weights/`.
- `pip install --break-system-packages` is pre-authorized on the remote.
- Proxy for downloads: `export https_proxy=http://10.13.11.1:1080`.
- After installing ultralytics: `pip install --force-reinstall opencv-python-headless`.

## Config system

Defaults in `tennisvision/config.py` DEFAULTS dict. Override via YAML files in `configs/`. Key switches:
- `action.enabled: true` — enables Pass 1b (pose + stroke classification) in analyze.py
- `rally.online: false` — default offline two-pass mode
- `inpainter.enabled: false` — trajectory gap-filling (optional)
- `ball.two_stage: true` — two-stage ball detection (main + far-court crop)

## Architecture

### Standalone pipeline (run_full_pipeline.py)

```
Step 1: calibrate.py        → calib.json (court homography)
Step 2: extract_keypoints.py → keypoints.json (player pose per frame)
Step 3: extract_ball_positions.py → ball_positions.json (ball trajectory)
Step 4: detect_serves.py    → serve_events.json (GRU v4 + filtering)
Step 5: detect_rallies.py   → rally_events.json + rally_cuts.mp4
```

### Integrated pipeline (analyze.py)

```
Pass 1a: ball detection -> Kalman tracker -> online rally detector
            (optional: trajectory inpainting + bounce detection)
Pass 1b: per-rally pose estimation -> stroke classification (only if action.enabled)
Pass 2:  re-read video -> render overlays -> write annotated output + rally cuts + JSON
```

## Model weights

| File | Module | Notes |
|------|--------|-------|
| `court_resnet.pth` | Court calibration | ResNet50, 14-point regression |
| `wasb_tennis_best.pth.tar` | Ball detection | WASB HRNet, auto-exports .onnx |
| `stroke_gru_v4_best.pt` | Serve detection | GRU v4, 4-class (standalone pipeline) |
| `bounce_catboost.cbm` | Bounce detection | CatBoost classifier |
| `yolo26s-pose.pt` | Keypoint extraction | YOLO26s-pose (standalone pipeline) |
| `yolo26n-pose.pt` | Player pose | YOLO26n-pose (analyze.py pipeline) |
| `tennis_rnn.h5` | Stroke classification | Keras GRU (analyze.py pipeline) |
| `InpaintNet_best.pt` | Trajectory inpainting | TrackNetV3, optional |
