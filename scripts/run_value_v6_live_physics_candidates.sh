#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:-/home/jia/twingraph-v6-20260917}
CASE_DIR=${2:-results/v6/formal/systems/case_71400}
OUTPUT_DIR=${3:-results/v6/formal/systems/case_71400/videos/live_physics_v4}
JOBS=${JOBS:-4}
PYTHON=${PYTHON:-/home/jia/miniconda3/envs/swdp/bin/python}

cd "$ROOT"
mkdir -p "$OUTPUT_DIR/logs"

run_one() {
  local index=$1
  local gpu=$(( (index - 1) % 2 ))
  local label
  printf -v label '%02d' "$index"
  env \
    PYTHONPATH="$ROOT" \
    MUJOCO_GL=egl \
    MUJOCO_EGL_DEVICE_ID="$gpu" \
    "$PYTHON" scripts/render_value_v6_live_physics_candidates.py \
      --case-dir "$CASE_DIR" \
      --output-dir "$OUTPUT_DIR" \
      --candidate-index "$index" \
      > "$OUTPUT_DIR/logs/candidate_${label}.log" 2>&1
}

export ROOT CASE_DIR OUTPUT_DIR PYTHON
export -f run_one
seq 1 12 | xargs -n 1 -P "$JOBS" bash -c 'run_one "$1"' _
