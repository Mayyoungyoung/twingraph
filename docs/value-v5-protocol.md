# Value module v5: prospective evaluation draft

Status: PROSPECTIVE FREEZE, 2026-09-16. The machine-readable
`experiments/value_v5/protocol.json` binds exact source bytes and settings before
formal collection. Development seeds61000/61001 establish the executable
five-part recipe and remain excluded from all formal splits. Model weights and
thresholds are frozen separately after validation and before test generation.
Preserve all v4 artifacts and the development evidence.

## Question and interface

Can an outcome-trained value module correctly classify executable assembly
candidates and retain useful plans in Top-K, while reducing measured time against
validating every candidate in the digital twin? The primary aim is a functional
coarse filter. Graph-versus-sequence novelty is not a required positive result.

The external input remains observation plus complete candidate programs. Goals
belong to the observation; candidates contain skill calls, physical parameters,
materialized trajectories where available and explicit deferred quantities.
The same executable content must reach the scorer and physical executor. Scorer
inference cannot read outcomes or candidate generation success labels. No manual
weighted geometric reward or retrospective positive/negative construction.
Structured scene geometry and observed poses provide state in this experiment;
omitting images is a scope choice, not an image-ablation finding.

## Physical scope and data

Start each candidate from the original unassembled slide scene, not a successful
robot-created suffix checkpoint or teleportation of installed parts. A development
pilot must establish which full assembly operations and acceptance conditions can
be executed. If the implemented task lacks slide motion or another operation,
name that limitation before freeze; do not call it complete functional assembly.

Candidates are distinct executable programs produced by legal grasp, route,
parameter and precedence choices. Audit exact executable duplicates, actual
trajectory variation, solver failures, all-success and all-failure pools, first
failure skills, and marginal outcome distributions. IDs, repeated disturbances
and duplicated programs do not constitute candidate diversity. Marginal yaw,
force and route correlations cannot establish causes or model reliance.

Frozen configuration split:

- Train: 48 configurations, seeds61100--61147.
- Validation: 12 configurations, seeds61200--61211.
- Locked test: 12 configurations, seeds61300--61311.
- Each reached group: N=12 unique executable candidates. Primary K=4; also K1/2.
- Primary labels: one nominal physical execution per candidate (R=1).
- No extra perturbed repeats in the classification dataset. Every candidate is
  a distinct executable program, not a noise repetition counted as a new plan.
- Separate system cases: seeds61400--61403, N12/K4, independently created twin
  and target scenes; two twin repeats and one target execution. Both roles use
  independent draws of friction scale1±3% and actuator gain1±0.5%, with fixed
  namespaces. These are simulated uncertainty, not measured hardware noise.

Fit MLPs with seeds17/29/43 and a linear model with seed17, using automatic
training-input schema and nominal binary cross entropy. All four models are
eligible for validation-Brier selection; this rule is fixed before training.
Use120epochs, batch48, AdamW learning rate0.001 and weight decay0.01. The MLP
has64/32hidden units and0.1dropout. No test-selected fallback model is allowed.

Keep every candidate, trajectory sibling, checkpoint and repetition from a
physical configuration in the same split. Preserve missing/failed setup and
unresolved generation records without resampling. Programming errors are errors,
not physical negative labels; censored executions are explicitly incomplete.

## Model, threshold and freeze

Fit encoders, normalization and any vocabulary on training inputs only. Learn
physical-success targets from executions. Select model/checkpoint and any scalar
probability calibration from validation only, then hash and publish them before
test. Record parameter count, input size and runtime as separate quantities.

The evaluator chooses the probability threshold on nominal validation trials:
maximize balanced accuracy, then recall, then proximity to0.5, then the lower
threshold. If validation contains only one class, use0.5 and explicitly record
the fallback. This threshold is a validation decision, not an engineered reward.
Apply it unchanged to test. Top-K rank uses saved logits if supplied, with stable
input-order ties. Score tolerance is diagnostic only and does not rerank.

`scripts/evaluate_value_v5.py freeze` reads validation prediction JSON only and
records model hashes, validation configurations, K values and thresholds.
The test analyzer rejects hash drift, method/pool mismatch and configuration
overlap. A deployment-model freeze and exact test manifest must accompany this
threshold freeze; threshold selection alone does not freeze model selection.

## Correctness and screening

Primary candidate correctness uses the declared nominal binary trial: report
TP/FP/TN/FN, accuracy, balanced accuracy, precision, recall, specificity, F1,
ROC-AUC, average precision, Brier and log loss. AUC/AP and balanced accuracy are
unavailable without both classes. Undefined precision/recall is null, not perfect.
Include trivial constant/majority prediction controls chosen from validation.
Accuracy is different from the probability of retaining one successful plan.

For each K report feasible precision (successful selected/K), feasible recall
(successful selected/all successful in the pool), feasible Hit (at least one),
mean selected quality, best selected quality and regret. Exact uniform-random
expectations are computed over subsets without replacement, including Hit,
precision, recall and best quality; do not rely on one lucky random ordering.
Retain all-failure pools in precision, feasible Hit and quality. Feasible recall
and near-best Hit are undefined there and use explicit smaller denominators.

When repeats exist, separately report binary trial correctness and Brier across
all recorded trials, squared error to candidate empirical success fractions,
and empirical-fraction screening. Observed feasibility then means at least one
success in the finite repeat budget; it is not known true feasibility. Never
majority-binarize ambiguous fractions for the nominal accuracy result. Main R1
results characterize nominal outcomes, not robustness probabilities.

Bootstrap entire physical configurations, preserving their candidates, repeats
and siblings. Classification reports trial micro metrics with configuration
bootstrap; screening weights configurations equally. Show individual groups,
pool types and any predeclared task strata. A degenerate100% interval is not a
reliability guarantee. Classification conditions on candidate creation; report
setup attrition separately and requested-configuration feasible Hit with failed
setup contributing zero.

## Actual all-candidate digital-twin comparison

Run the full strategy and Top-K strategy separately on matched configurations,
the same candidate content, paired validation disturbances and the same acceptance
and final selection rule. The full strategy actually validates all N candidates;
the screened strategy pays the complete value-module cost and validates K.
Keep candidates rejected by necessary checks/solver resolution visible in the
generation audit; do not silently charge them as executed twin validations.

After validation, independently create the target scene and execute the chosen
program with the prescribed physical mismatch. Do not transfer twin state to the
target. Record no-accepted-plan cases as unresolved failures. Verify the target
outcome rather than reusing a twin/reference label. The target is another simulator
domain, not real hardware. Include a source-order Top-K control when practical.

The system runner records actual scene setup, candidate generation, graph building,
integrity checking, encoding, inference, validation, selection, target setup and
rebinding, execution, rendering and total wall time. Define decision time separately
from the full scene-creation-to-target-render clock. Count actual validation and
target executions, simulation steps, and early termination. Compare measured
paired ratios and differences jointly with independent success, including equal
hardware/worker policy and any cold-start/model-loading costs. Never report N/K
as measured acceleration; successful and early-failed runs can have different
costs. Per-policy elapsed and parallel batch wall time are separate measurements.

## Machine-readable evaluator input

Prediction JSON has schema `twingraph.value.predictions.v5`, split `validation`
or `test`, and `methods[name] = {model_sha256, rows}`. Each row contains
`group_id`, `config_id`, ordered unique `candidate_ids`, probability `scores`,
optional logit `ranking_scores`, and binary `outcomes[candidate][trial]`.
Set `label_regime` to `nominal` for R1, otherwise `repeated` and explicitly set
`nominal_index`. Optional checkpoint metadata enables fixed stratum reporting.
Nonprobability baselines declare `probability_metrics:false` and still supply
a hash binding their deterministic rule. They receive ranking metrics only.

Test metadata can declare `requested_config_ids` and explicit `setup_failures`;
these must partition reached and failed configurations exactly. Archive inputs,
outcomes, candidate/graph hashes, model/source hashes and the inference-before-
outcome-read ordering separately. The evaluator checks supplied bindings but
does not substitute for that execution provenance.
