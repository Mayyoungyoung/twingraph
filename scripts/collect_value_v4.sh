#!/usr/bin/env bash
set -euo pipefail
TG_PYTHON="${TG_PYTHON:-/home/jia/twingraph-v3-venv/bin/python}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl
"$TG_PYTHON" -m simbench.value.schema_collect --out results/v4/data --seed 51100 --groups 12 --n 8 --repeats 2 --workers 8 --checkpoints 2 3 --alternate-checkpoints --split train > collection_train.log 2>&1
"$TG_PYTHON" -m simbench.value.schema_collect --out results/v4/data --seed 51200 --groups 4 --n 8 --repeats 2 --workers 4 --checkpoints 2 3 --alternate-checkpoints --split val > collection_val.log 2>&1
# Test outcomes are not collected/opened until models have been frozen separately.
