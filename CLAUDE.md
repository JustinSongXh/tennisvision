# CLAUDE.md — TennisVision

## Project overview

Tennis video analysis pipeline: ball detection & tracking, bounce detection, rally segmentation, player pose estimation & stroke classification, and annotated video rendering. Single fixed-camera input, outputs annotated video + rally cuts + JSON stats.

## Repository layout

```
scripts/          CLI entry points (analyze.py, calibrate.py, diagnose.py)
tennisvision/     Main package
  ball/           Ball detection (WASB HRNet, classical HSV) + Kalman tracker + InpaintNet
  bounce/         Bounce detection (CatBoost, peak)
  court/          Court detection, homography, calibration
  action/         Player pose (YOLO-pose) + stroke classifier (GRU)
  pipeline/       Two-pass orchestration + rally state machine
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

Local (Mac, venv):
```bash
cd /Users/justinsong/WorkSpace/solo/tennisvision && \
/Users/justinsong/WorkSpace/solo/venv/bin/python -u scripts/analyze.py \
    --video <path>.mp4 --calib <path>.json \
    --config configs/local_run.yaml \
    --out /Users/justinsong/WorkSpace/solo/results/<name>.mp4
```

Remote (192.168.196.196, GPU):
```bash
ssh 192.168.196.196
cd /root/personal/tennisvision
python -u scripts/analyze.py \
    --video /root/personal/<video>.mp4 \
    --calib /root/personal/<calib>.json \
    --config configs/remote_inpaint.yaml \
    --out /root/personal/results/<name>.mp4
```

For long runs use `nohup ... > log 2>&1 &` and `caffeinate -i -s -w <PID> &` locally.

### Test videos
- `tennis_raw.mp4` — singles, use `calib.json`
- `tennis_raw2.mp4` — doubles (4 players), use `calib2.json`

### Remote machine notes
- Weights live at `/root/personal/` (outside repo tree), scp from local `./weights/`.
- `pip install --break-system-packages` is pre-authorized on the remote.
- Proxy for downloads: `export https_proxy=http://10.13.11.1:1080`.
- After installing ultralytics: `pip install --force-reinstall opencv-python-headless`.

## Config system

Defaults in `tennisvision/config.py` DEFAULTS dict. Override via YAML files in `configs/`. Key switches:
- `action.enabled: true` — enables Pass 1b (pose + stroke classification)
- `rally.online: false` — default offline two-pass mode
- `inpainter.enabled: false` — trajectory gap-filling (optional)

## Architecture: two-pass pipeline

```
Pass 1a: ball detection -> Kalman tracker -> online rally detector
            (optional: trajectory inpainting + bounce detection)
Pass 1b: per-rally pose estimation -> stroke classification (only if action.enabled)
Pass 2:  re-read video -> render overlays -> write annotated output + rally cuts + JSON
```
