#!/usr/bin/env bash
# Reproduce the three paired layout experiments on the documented 901 host.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
runtime="${TWINGRAPH_PYTHON:-/home/jia/twingraph-v8-mj237/bin/python}"
output="${1:-results/value_v11_system}"
checkpoint="${2:-docs/evidence/value_v10_targeted_screening/frozen_transfer/v6_bce.pt}"
mkdir -p "$output"
export PYTHONPATH=. MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
sha256sum "$checkpoint" simbench/value/{planner_v11,system_v11,full_task_v7}.py \
  simbench/assembly/{ports,contracts}.py simbench/configs/planner_v11.json > "$output/source.sha256"
pids=()
for item in "1520:0" "1521:2" "1522:4"; do
  seed="${item%:*}"
  core="${item#*:}"
  taskset -c "$core" "$runtime" scripts/run_v11_system.py --out "$output" \
    --checkpoint "$checkpoint" --seeds "$seed" --methods all_twin random_top_k value_top_k \
    > "$output/seed_$seed.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
if [[ "$failed" != 0 ]]; then
  echo "At least one system worker failed; see per-layout logs." >&2
  exit 1
fi
"$runtime" scripts/sweep_v11_budget.py --root "$output" --checkpoint "$checkpoint" --out "$output/analysis/budget_sweep.json"
"$runtime" scripts/audit_v11_value_transfer.py --root "$output" --checkpoint "$checkpoint" --out "$output/analysis/value_transfer_audit.json"
"$runtime" scripts/analyze_v11_system.py --root "$output" --out "$output/analysis"
