#!/usr/bin/env bash
# TennisVision — Full rally detection pipeline.
#
# Orchestrates five steps, each calling a standalone Python script:
#   1. Court calibration      -> calib.json + court_overlay.jpg
#   2. Keypoints extraction   -> keypoints.json
#   3. Ball trajectory        -> ball_positions.json
#   4. Serve detection        -> serve_events.json
#   5. Rally detection        -> rally_events.json + rally_cuts.mp4
#
# Usage:
#   # Full pipeline
#   bash scripts/run_full_pipeline.sh --video samples/sample2.mp4
#
#   # Single step
#   bash scripts/run_full_pipeline.sh --video samples/sample2.mp4 --step 3
#
#   # From step 3 onwards
#   bash scripts/run_full_pipeline.sh --video samples/sample2.mp4 --from 3
#
#   # Custom weights / limit frames
#   bash scripts/run_full_pipeline.sh --video samples/sample2.mp4 --weights-dir /path/to/weights --max-frames 1800

set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python3}"

# ---- defaults ----
VIDEO=""
WEIGHTS_DIR="weights"
CONFIG=""
MAX_FRAMES=0
NO_HALF=""
STEP=0      # 0 = all
FROM=1      # start step

usage() {
    sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
    exit 1
}

# ---- parse args ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --video)       VIDEO="$2";       shift 2 ;;
        --weights-dir) WEIGHTS_DIR="$2"; shift 2 ;;
        --config)      CONFIG="$2";      shift 2 ;;
        --max-frames)  MAX_FRAMES="$2";  shift 2 ;;
        --no-half)     NO_HALF="--no-half"; shift ;;
        --step)        STEP="$2";        shift 2 ;;
        --from)        FROM="$2";        shift 2 ;;
        -h|--help)     usage ;;
        *)             echo "Unknown arg: $1"; usage ;;
    esac
done

[[ -z "$VIDEO" ]] && { echo "Error: --video is required"; usage; }

# ---- derived paths ----
BASENAME="$(basename "${VIDEO%.*}")"
OUT_DIR="results/${BASENAME}"
mkdir -p "$OUT_DIR"

# ---- CUDA detection ----
HAS_CUDA=$("$PYTHON" -c "
try:
    import torch; print('1' if torch.cuda.is_available() else '0')
except ImportError:
    print('0')
" 2>/dev/null)
DEVICE="cpu"
[[ "$HAS_CUDA" == "1" ]] && DEVICE="cuda"

# ---- helpers ----
banner() {
    echo ""
    echo "============================================================"
    echo "  Step $1: $2"
    echo "============================================================"
}

should_run() {
    local step=$1
    if [[ $STEP -ne 0 ]]; then
        [[ $step -eq $STEP ]]
    else
        [[ $step -ge $FROM ]]
    fi
}

# ---- Step 1: Court Calibration ----
if should_run 1; then
    banner 1 "Court Calibration"
    CALIB="$OUT_DIR/calib.json"
    if [[ -f "$CALIB" ]]; then
        echo "  Already exists: $CALIB"
    else
        "$PYTHON" -u "$SCRIPTS_DIR/calibrate.py" blue-resnet \
            --video "$VIDEO" \
            --out "$CALIB" \
            --vis "$OUT_DIR/court_overlay.jpg" \
            --weights "$WEIGHTS_DIR/court_resnet.pth"
    fi
fi

# ---- Step 2: Keypoints Extraction ----
if should_run 2; then
    banner 2 "Keypoints Extraction"
    KP="$OUT_DIR/keypoints.json"
    if [[ -f "$KP" ]]; then
        echo "  Already exists: $KP"
    else
        ARGS=(--video "$VIDEO" --out "$KP")
        [[ $MAX_FRAMES -gt 0 ]] && ARGS+=(--max-frames "$MAX_FRAMES")
        [[ -n "$NO_HALF" ]]     && ARGS+=("$NO_HALF")
        "$PYTHON" -u "$SCRIPTS_DIR/extract_keypoints.py" "${ARGS[@]}"
    fi
fi

# ---- Step 3: Ball Trajectory ----
if should_run 3; then
    banner 3 "Ball Trajectory"
    BALL="$OUT_DIR/ball_positions.json"
    if [[ -f "$BALL" ]]; then
        echo "  Already exists: $BALL"
    else
        ARGS=(--video "$VIDEO" --out "$BALL"
              --weights "$WEIGHTS_DIR/wasb_tennis_best.pth.tar"
              --device "$DEVICE")
        [[ -n "$CONFIG" ]]      && ARGS+=(--config "$CONFIG")
        [[ $MAX_FRAMES -gt 0 ]] && ARGS+=(--max-frames "$MAX_FRAMES")
        "$PYTHON" -u "$SCRIPTS_DIR/extract_ball_positions.py" "${ARGS[@]}"
    fi
fi

# ---- Step 4: Serve Detection ----
if should_run 4; then
    banner 4 "Serve Detection"
    SERVES="$OUT_DIR/serve_events.json"
    if [[ -f "$SERVES" ]]; then
        echo "  Already exists: $SERVES"
    else
        ARGS=(--keypoints "$OUT_DIR/keypoints.json"
              --calib "$OUT_DIR/calib.json"
              --gru "$WEIGHTS_DIR/stroke_gru_v4_best.pt"
              --out "$SERVES")
        [[ -f "$OUT_DIR/ball_positions.json" ]] && ARGS+=(--ball "$OUT_DIR/ball_positions.json")
        "$PYTHON" -u "$SCRIPTS_DIR/detect_serves.py" "${ARGS[@]}"
    fi
fi

# ---- Step 5: Rally Detection ----
if should_run 5; then
    banner 5 "Rally Detection"
    RALLIES="$OUT_DIR/rally_events.json"
    if [[ -f "$RALLIES" ]]; then
        echo "  Already exists: $RALLIES"
    else
        "$PYTHON" -u "$SCRIPTS_DIR/detect_rallies.py" \
            --video "$VIDEO" \
            --results-dir "$OUT_DIR"
    fi
fi

# ---- Summary ----
if [[ $STEP -eq 0 ]]; then
    echo ""
    echo "============================================================"
    echo "  Pipeline Complete!"
    echo "============================================================"
    echo "  Results in: $OUT_DIR/"
    ls -lhS "$OUT_DIR/" 2>/dev/null | tail -n +2 | awk '{printf "    %-30s %6s\n", $NF, $5}'
    echo "============================================================"
fi
