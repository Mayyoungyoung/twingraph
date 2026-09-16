# v3 evidence audit before the next value experiment

Date: 2026-09-16. This audit reads the published archives and predictions; it
does not recollect labels, retune a model, or reopen this test for model selection.
The old test is regression evidence for subsequent development.

## What the evidence establishes

The released compiler binds executable skill interfaces, PlanIR, observations,
typed ports and dependencies to the value input. The selected port MLP obtains
Hit@1/2/4 = 12/12 and Top-4 mean empirical quality 0.9583 on the 12 test
configurations. The MLP87 baseline has Top-4 quality 1.0 and lower empirical
Brier error (0.0464 versus 0.2464). Correct graph relations have no demonstrated
benefit over the matched sequence model. These are the scope limits already
acknowledged in [the original report](../VALUE_GRAPH_REPORT.md).

The fresh archive contains 32 configurations, each with 16 candidates and two
paired repetitions: 1,024 physical MuJoCo executions, 506 full successes and
1,024 successful grasp/lift prefixes. Every candidate has 19 calls and every
pool has one structure branch. Full slide assembly already exists in
[`assembly/task.py`](../simbench/assembly/task.py), but these value labels are
the single-pin task, not executions of that complete assembly procedure.

## Diversity and label diagnostics

Archive-wide empirical labels for 512 candidates: 258 are zero, 252 are one,
and only two are 0.5. Every one of the 192 locked-test candidates has label zero
or one. The friction/gain perturbations therefore changed no test candidate's
binary result between its two repetitions; these repetitions provide little
evidence about robustness under realistic pose, alignment or contact uncertainty.

Failures are concentrated at the terminal release controller:

| Last failed skill | Trials | Fraction of all 518 failures |
|---|---:|---:|
| `open_gripper` | 455 | 87.84% |
| `place_object` (support check) | 60 | 11.58% |
| `press_seat` | 3 | 0.58% |

[`Arm.open`](../simbench/assembly/control.py) requires a jaw span above 75 mm.
[`place_object`](../simbench/assembly/library.py) invokes this controller before
retreat. This is a legitimate test of this specific controller, but it largely
measures complete jaw opening in a constrained channel. It does not establish
that every rejected configuration prevents releasing the much smaller pin head
and retreating. A development-only physical check of object separation, support
and retreat is needed before replacing the criterion; do not relabel the old
test by silently weakening a controller threshold.

The generator in [`research_scenarios.py`](../simbench/value/research_scenarios.py)
shuffles a finite grid of grasp yaw/height, clearance, force and speed, removes
semantic duplicates and performs label-blind necessary checks. These are actual
control alternatives, but not recorded LLM proposals or diverse task structures.
Archived program calls contain no materialized `joint_path` or `path` arguments:
the two `plan_path` targets are deferred and paths are solved during execution.
Thus this release does not yet validate a model consuming whole planned
trajectories. Nominal planning before execution and feedback-dependent future
parameters must remain distinguishable in the next representation.

## Hit saturation and unstable-score controls

For each test pool, uniform random sampling without replacement has exact
Hit@K = `1 - C(N-M,K)/C(N,K)`, where M is the number of empirical near-best
candidates. Averaging the 12 configurations gives:

| K | Exact random expected near-best Hit |
|---|---:|
| 1 | 57.2917% |
| 2 | 80.0000% |
| 4 | 94.8397% |

Expected random Top-K mean empirical quality is 57.2917% for every K.
Geometry and the selected port MLP both achieve 100% Hit@4, only 5.1603
percentage points above this random expectation. Top-1 and selected-set quality
are more informative here. The original sampled random baseline is one draw,
not this analytical expectation.

Median within-pool score ranges expose optimization failures in some controls:
`field_17`: 2.09e-7; `sequence_29`: 2.00e-6; `sequence_43`: 2.32e-6;
`graph_29`: 5.10e-5; `graph_43`: 2.29e-6. Their nominal rankings depend on very
small score differences. Report this diagnostic rather than treating their
Hit values alone as evidence of learned discrimination. Tolerance analysis must
not silently rerank scores or become a test-tuned selection rule.

## Evaluator gaps to address before varied programs

At the audited revision, `PhysicalRunner.run` marks success when all submitted
calls return successfully; it does not independently evaluate the requested
goal. A read-only reproduction removed the last five calls from an archived
19-call candidate. The resulting 14-call program ends at `guarded_descent`, has
no release/retreat/acceptance, yet passes `PlanIR.from_dict`, `audit_program` and
`compile_graph` with `no_known_contract_conflict`. Therefore a truncated candidate
can receive a positive label if those remaining calls succeed. This does not
invalidate the fixed-generator labels, whose full plans include acceptance,
but it must be fixed before LLM or variable-length candidates are labeled.

The v3 `graph_learning.metrics` and `evaluate_skill_graph_value.intervals` average
and bootstrap decision rows independently. This is appropriate for the current
one-row-per-configuration test, but would overcount correlated checkpoints in
the next experiment. Configuration siblings must share splits and bootstrap
units. The v3 loader also hardcodes the pin family and the evaluator hardcodes
the 41200–41211 test seeds; these are experiment-specific restrictions.

## Bounded next experiment

Use physically reached checkpoints of the existing complete sliding assembly,
with candidate alternatives tied to actual grasp/route/control branches and
valid part-order choices. Include the carriage regrasp and stroke stage so that
remaining plans consume genuinely different object/holding/scene histories.
Keep all unresolved and all-failure pools and record attempted/unique candidates,
source template, topology, path bindings and failed goals. Use new disjoint
configuration seeds, grouping every checkpoint sibling in one split.

Before scaling data collection, audit a small development batch for independent
goal completion, failure-mode coverage, trajectory binding and whether candidate
differences alter execution. Do not force a target success balance or add dummy
calls to increase diversity. Compare port MLP, a numerically healthy matched
sequence model, graph model and exact random expectation; freeze using validation
only, then measure Top-1, selected-set quality, Hit@K, uncertainty and independent
online execution under identical physical budgets.

## Reproduction and lineage

The new read-only analyzer accepts v3 results or v4 predictions whose methods
contain rows with `group_id`, `config_id`, `seed`, `scores`, and `reference`:

```bash
python scripts/analyze_value_v4.py \
  --predictions docs/evidence/value_v3/test_metrics.json \
  --out runs/v3/audit
```

It exports JSON and CSV, preserves all-failure groups, reports exact uniform
random expectations, and bootstraps configurations. `config_id` must group all
trajectory/checkpoint siblings. Without it, the v3 fallback is family plus seed.
Score arrays and reference arrays must use the same input candidate order.
Reference fractions remain noisy finite-repetition observations, not true
success probabilities; a degenerate 100% interval is not a reliability guarantee.

- Archive: `datasets/value/skill_graph_v3.tar.gz`, SHA256
  `06275fc1edd7a46dfe2a27164b6858f56f5adc16da851c569f7c5e60bd44ca83`.
- Predictions: `docs/evidence/value_v3/test_metrics.json`, SHA256
  `0d2503080538472b32204d329c979919aee92bd21b5199a515ee1063466a7318`.
- Model/source/split lineage remains in the original
  [`manifest_skill_graph_v3.json`](../datasets/value/manifest_skill_graph_v3.json)
  and [`frozen.json`](evidence/value_v3/frozen.json). This audit changes neither.
