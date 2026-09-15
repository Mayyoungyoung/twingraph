#!/usr/bin/env bash
set -euo pipefail
PYTHON=${VALUE_PYTHON:-/root/rivermind-data/twingraph-venv/bin/python}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl PYTHONPATH=.
DATA=${VALUE_DATA:-results/value_v2/data}
OUT=${VALUE_OUT:-results/value_v2/models}
mkdir -p "$OUT"
"$PYTHON" -m simbench.value.vision --data "$DATA" --device cuda
for seed in 17 29 43; do
  for kind in mlp residual; do
    "$PYTHON" -m simbench.value.research_learning --data "$DATA" --out "$OUT/${kind}_${seed}" --kind "$kind" --seed "$seed" --epochs 120 > "$OUT/${kind}_${seed}.log" 2>&1
  done
done
"$PYTHON" -m simbench.value.research_learning --data "$DATA" --out "$OUT/pin_mlp_17" --kind mlp --family sliding_stage_pin --epochs 120 > "$OUT/pin_mlp_17.log" 2>&1
"$PYTHON" -m simbench.value.research_learning --data "$DATA" --out "$OUT/fewshot_mlp_17" --kind mlp --few-shot 2 --initialize "$OUT/pin_mlp_17/best.pt" --epochs 120 > "$OUT/fewshot_mlp_17.log" 2>&1
for kind in no_vision vision; do
  "$PYTHON" -m simbench.value.research_learning --data "$DATA" --out "$OUT/${kind}_17" --kind "$kind" --epochs 40 --batch-size 4 > "$OUT/${kind}_17.log" 2>&1
done
"$PYTHON" - <<'PY'
import json
from pathlib import Path
root=Path('results/value_v2/models')
mapping={p.name:str(p/'best.pt') for p in root.iterdir() if p.is_dir() and (p/'best.pt').exists()}
mapping['v1_fixed']='results/value/direct/best.pt'
(root/'all_models.json').write_text(json.dumps(mapping,indent=2))
print(mapping)
PY
