# Reproduce the graph-input value experiment

Working host used: `jia@192.168.110.244`, two RTX 2080 Ti cards. Existing key-based
SSH was sufficient. Work directory: `/home/jia/twingraph-skill-graph`.
Isolated Python: `/home/jia/twingraph-v3-venv/bin/python`, inheriting the installed
CUDA Torch environment; MuJoCo 2.3.2 and NumPy 1.26.4 installed in this isolated
venv. Other projects/environments were not changed. No credentials are stored.

Runtime/collection changes are in research branch `research/skill-graph-value`.
No v1/v2 data or checkpoints were replaced. `docs/value-v3-protocol.md` records
the first data plan, development amendments and test-freeze rules.

## Data and training

Unpack the old `datasets/value/two_family_v2.tar.gz` under `runs/v3/old` and new
`datasets/value/skill_graph_v3.tar.gz` under `runs/v3/fresh`. Only old pin train/val
groups are loaded for training; old connector and old pin test groups are ignored.

```bash
export TG_PYTHON=/home/jia/twingraph-v3-venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl

# To recollect, choose a new output directory; do not overwrite frozen data.
$TG_PYTHON -m simbench.value.research_collect --out runs/v3/recollect \
  --family sliding_stage_pin --seed 41000 --groups 16 --split train \
  --n 16 --repeats 2 --workers 4 --emit-graph
$TG_PYTHON -m simbench.value.research_collect --out runs/v3/recollect \
  --family sliding_stage_pin --seed 41100 --groups 4 --split val \
  --n 16 --repeats 2 --workers 4 --emit-graph
$TG_PYTHON -m simbench.value.research_collect --out runs/v3/recollect \
  --family sliding_stage_pin --seed 41200 --groups 12 --split test --domain reference \
  --n 16 --repeats 2 --workers 4 --emit-graph

# Final complete matrix; set CUDA_VISIBLE_DEVICES to choose an existing GPU.
bash scripts/train_value_graph_v3.sh
```

All new labels are real physical MuJoCo rollouts. Input graphs are saved before
the trials. Each trial records the hash of the graph actually executed.
Source hashes can differ on recollection after instrumentation changes; keep
that version in the new manifest. Do not mix incomplete groups into training.

## Test, latency and independent execution

```bash
$TG_PYTHON scripts/evaluate_skill_graph_value.py freeze \
  --models runs/v3/models_final --freeze runs/v3/frozen.json
$TG_PYTHON scripts/evaluate_skill_graph_value.py evaluate \
  --freeze runs/v3/frozen.json --data runs/v3/fresh --out runs/v3/evidence
$TG_PYTHON scripts/evaluate_skill_graph_value.py benchmark \
  --freeze runs/v3/frozen.json --out runs/v3/evidence
$TG_PYTHON scripts/evaluate_skill_graph_value.py deployment \
  --freeze runs/v3/frozen.json --out runs/v3/evidence
```

The freeze file contains absolute paths on the running host. To evaluate packaged
models on a different host, create a **new** freeze file pointing at their unpacked
model directory; retain the published freeze/hash file as evidence of the original
run. Do not alter the frozen weights. Published test has now been inspected and
should be treated as regression for subsequent research iterations.

The latency command runs alone with no parallel collection/training jobs. It
reports shared graph construction/integrity, optional geometry, encoding, resident
GPU inference and sorting/export. It does not call a cached success table or count
N/K as speedup. A second view subtracts the explicitly shared graph stages.

## Score a saved candidate pool

```bash
$TG_PYTHON -m simbench.value.graph_rank \
  --inputs runs/v3/fresh/group_sliding_stage_pin_41200_0/inputs.json \
  --checkpoint models/value/v3/best_graph_input.pt \
  --out top_k_graph.json --k 4 --device cuda
```

Only `inputs.json` is needed for scoring; no outcome file is read. The returned
Top-K includes full graphs and PlanIR, ready for `PhysicalRunner.run(graph, trial)`.
The chosen checkpoint is selected using validation ranking quality. It is not a
formal certificate that a candidate will succeed.

```bash
$TG_PYTHON -m pytest simbench/tests -q
```
