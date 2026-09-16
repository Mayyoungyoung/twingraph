#!/usr/bin/env bash
# Reproduce the final training matrix after unpacking the documented datasets.
set -euo pipefail
TG_PYTHON="${TG_PYTHON:-python}"
TG_DATA_OLD="${TG_DATA_OLD:-runs/v3/old}"
TG_DATA_NEW="${TG_DATA_NEW:-runs/v3/fresh}"
TG_MODELS="${TG_MODELS:-runs/v3/models_final}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl
for kind in graph sequence; do
  for seed in 17 29 43; do
    "$TG_PYTHON" -m simbench.value.graph_learning --data "$TG_DATA_OLD" "$TG_DATA_NEW" \
      --out "$TG_MODELS/${kind}_$seed" --kind "$kind" --seed "$seed" \
      --pooling attention --normalize --epochs 180
  done
done
for seed in 17 29 43; do
  "$TG_PYTHON" -m simbench.value.graph_learning --data "$TG_DATA_OLD" "$TG_DATA_NEW" \
    --out "$TG_MODELS/port_mlp_$seed" --kind port_mlp --seed "$seed" --epochs 60
done
for kind in mlp field; do
  "$TG_PYTHON" -m simbench.value.graph_learning --data "$TG_DATA_OLD" "$TG_DATA_NEW" \
    --out "$TG_MODELS/${kind}_17" --kind "$kind" --seed 17 --epochs 60
done
