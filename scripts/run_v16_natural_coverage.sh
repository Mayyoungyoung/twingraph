#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/jia/twingraph-v16-natural-coverage-r6}
VENV=${VENV:-/home/jia/twingraph-v8-mj237}
CHECKPOINT=${CHECKPOINT:-/home/jia/twingraph-v13-mechanism-r1/artifacts/v15_full_system/value_training/pretrain_r2/value_v15_selected.pt}
FREEZE="$ROOT/artifacts/v16_natural_coverage/freeze_r6/manifest.json"
MATRIX="$ROOT/artifacts/v16_natural_coverage/stage_a_matrix_r6"

source "$VENV/bin/activate"
cd "$ROOT"
export PYTHONPATH=.
export MUJOCO_GL=${MUJOCO_GL:-egl}
export EGL_DEVICE_ID=${EGL_DEVICE_ID:-1}
export SIMBENCH_FROZEN_RUNTIME_MANIFEST="$FREEZE"

python scripts/collect_v12_parallel.py --out "$MATRIX" --seeds 1950 1951 1952 \
  --n 16 --level L1 --domain train --workers 4
python scripts/analyze_v16_natural_coverage.py --matrix "$MATRIX" \
  --checkpoint "$CHECKPOINT" --k 4 \
  --out artifacts/v16_natural_coverage/RESULTS.json
