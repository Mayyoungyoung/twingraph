#!/usr/bin/env bash
set -euo pipefail
TG_PYTHON="${TG_PYTHON:-/home/jia/twingraph-v3-venv/bin/python}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl
# Two processes use separate GPUs. Models see train/validation labels only.
train_lane() {
  local gpu="$1"; shift
  for kind in "$@"; do
    for seed in 17 29 43; do
      CUDA_VISIBLE_DEVICES="$gpu" "$TG_PYTHON" -m simbench.value.schema_learning \
        --data results/v4/data --out "results/v4/models/${kind}_${seed}" \
        --kind "$kind" --seed "$seed" --epochs 60 > "training_${kind}_${seed}.log" 2>&1
    done
  done
}
train_lane 0 compact graph &
p0=$!
train_lane 1 port_mlp sequence &
p1=$!
wait "$p0"
wait "$p1"
