#!/usr/bin/env bash
# Reproducible first-stage experiment. A compatible Torch/torchvision pair and
# MuJoCo EGL runtime must already be installed; see docs/value-screening.md.
set -euo pipefail
cd "$(dirname "$0")/.."
VALUE_PYTHON="${VALUE_PYTHON:-python}"
VALUE_OUT="${1:-results/value}"
VALUE_GROUPS="${VALUE_GROUPS:-100}"
VALUE_WORKERS="${VALUE_WORKERS:-12}"
VALUE_EPOCHS="${VALUE_EPOCHS:-60}"
export MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

"$VALUE_PYTHON" -m simbench.value.collect --out "$VALUE_OUT/data" \
    --groups "$VALUE_GROUPS" --start-seed 1000 --repeats 4 --workers "$VALUE_WORKERS"
"$VALUE_PYTHON" -m simbench.value.vision --data "$VALUE_OUT/data" --device cuda
"$VALUE_PYTHON" -m simbench.value.train --data "$VALUE_OUT/data" \
    --out "$VALUE_OUT/dual" --device cuda --epochs "$VALUE_EPOCHS" --k 2
"$VALUE_PYTHON" -m simbench.value.train --data "$VALUE_OUT/data" \
    --out "$VALUE_OUT/direct" --device cuda --epochs "$VALUE_EPOCHS" --k 2 --objective direct
"$VALUE_PYTHON" -m simbench.value.rank --checkpoint "$VALUE_OUT/dual/best.pt" \
    --input "$VALUE_OUT/data/group_001000/inputs.json" --k 2 --device cuda \
    --out "$VALUE_OUT/top_k.json"
"$VALUE_PYTHON" -m simbench.value.decision --checkpoint "$VALUE_OUT/dual/best.pt" \
    --out "$VALUE_OUT/decision" --seed 9000 --k 2 --budget 2 --device cuda --execute
