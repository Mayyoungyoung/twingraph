# Executable-plan value module v4: prospective protocol

Date: 2026-09-16. Starting commit: 4c7a2c8. Previous v3 test results are
development/regression evidence, never a new held-out test in this round.

## Questions

1. Can materialized joint trajectories be represented and executed from the
   same immutable graph, without a parallel hand-selected feature vector?
2. Can training-input-only removal of constant/redundant port columns give a
   smaller scorer, and validation-only calibration improve probability quality?
3. Does the representation work for multiple remaining parts in the original
   sliding-stage scene, starting from an actual robot-executed checkpoint?

Labels are complete-program physical success/failure from existing skill
acceptance conditions. No weighted geometric score, manually annotated reward,
or manufactured negative plan is a training target. Controller thresholds and
task acceptance tolerances remain engineering specifications, not rewards.

## Data and development rules

Use a new isolated directory on the 2080 host. Pilot scene/controller changes
use seeds 51000--51009 only and are explicitly development. Never force successful
prefixes by teleporting installed parts. Preserve failed checkpoint creation,
unresolved necessary checks, all-success and all-failure candidate pools.
Candidate identity is executable-content identity, not an arbitrary ID or label.
Exact duplicates are removed before rollout. Vary physically legal grasp choices,
routes and independent part choices; preserve assembly precedence constraints.

After the pilot fixes the feasible scope, record the exact family, seed ranges,
candidate count and repetitions in this file before opening any final outcomes.
Split by entire physical configuration, keeping all checkpoints/candidates/repeats
of a configuration together. Online verification and final simulated execution
use independent paired disturbance namespaces. Simulation is not real-robot data.

## Models and selection

Compare the previous typed-port MLP, compact typed-port MLP, and matched graph /
sequence controls as resources permit. All consume complete pre-execution graph
inputs; graph edges are an independently tested architectural choice. Compact
columns and normalizers are fitted only on training inputs, without labels.
Main MLP seeds: 17, 29, 43; 60 epochs; best validation Brier checkpoint. A scalar
affine logit calibrator is fitted on validation labels after training, retaining
uncalibrated scores as a control. No test fitting. This uses the validation set
for both model selection and calibration and is explicitly small-sample evidence.
Select deployment by validation Hit@4, Top4 quality, Hit@1, then lower Brier;
freeze checkpoints, hashes, input schema and selection before final test analysis.

## Measurements and claims

Report near-best Hit@1/2/4, feasible hit, mean selected success rate, regret,
Brier, log loss and configuration-level uncertainty. With only a few repeats,
these refer to noisy empirical candidate quality, not exact success probability.
Report all training seeds, all pool types, and exact executable duplicate counts.
Measure graph construction/checking, encoding, inference and export separately.
End-to-end claims require actual Top-K simulation followed by independent
execution, and wall-clock comparison to the same verification rule.

PIGINet already fuses plans, images, goals, poses and joint angles; it predicts
whether a task skeleton can be refined. Our proposed distinction is a shared
executable representation, including available trajectory/control data and
explicit deferred values. Neither Transformers nor industrial context alone
establish novelty. Source: https://roboticsproceedings.org/rss19/p061.pdf

## Frozen data specification after development pilots

Original sliding-stage scene, after actual robot installation of carriage/end
stop (checkpoint 2, 58 calls) or also the left pin (checkpoint 3, 40 calls).
Checkpoint 3 uses the repository's existing production pick/transfer/release
recipe, after an arbitrary fixed grasp choice failed on development seed 51002.
That failed attempt is preserved, not relabeled or included as a continuation.
Development seeds 51000/51001: 16 trials, 3 successes, no timeouts; seed 51003
checkpoint 3: 4 trials, 1 success, no timeouts. These are feasibility pilots.

- Training configurations: seeds 51100--51111 (12).
- Validation configurations: seeds 51200--51203 (4).
- Locked test configurations: seeds 51300--51307 (8).
- Every even seed uses checkpoint 2; every odd seed uses checkpoint 3. Exactly
  one checkpoint per configuration in this round; no unseen-length claim.
- Each group: N=8 unique candidates, R=2 paired physical repetitions.
- Training/validation namespace `train`; test namespace `reference`.
- Final grid includes the existing nominal 3 N grip command alongside 2.5/3.5 N,
  with uniform label-blind shuffle. No preferred successful anchor is inserted.
- Expected maximum: 24 groups, 192 plans, 384 continuation executions, plus
  separately recorded physical checkpoint preparation. Failed preparations or
  censored groups stay visible and are not silently resampled.
- Models: compact port MLP and unpruned port MLP, seeds17/29/43, 60 epochs;
  same-input sequence and graph, seeds17/29/43, 60 epochs. All final models use
  the new dataset only. Earlier v3 fitting remains development evidence.
- Deployment test configurations: seeds51300--51303 with their assigned
  checkpoints. K4, two `online` trials per candidate, select by online success
  fraction then original rank; two independent `deployment` trials. Include
  source-order and shortest-initial-joint-path controls under the same budget.
  An additional exhaustive control validates all N8 with the same two online
  repetitions, then independently executes the selected plan twice. Its larger
  measured budget is reported explicitly, not presented as a matched-budget
  baseline or an assumed N/K speedup. Four configurations may run in parallel;
  policy wall times and overall parallel elapsed are separate quantities.

Before test collection, define setup attrition as a recorded pre-input robot
SkillFailure, IK-unreachable result or exhausted materialized-candidate solver.
Do not treat an arbitrary programming ValueError as attrition. Expected test
configurations must have either a complete reached decision group or this
explicit setup failure; missing/incomplete groups are errors. Ranking metrics
condition on reaching the decision point; full workflow outcomes also count
failed setup configurations with zero success. Training seeds51107/51109 have
already demonstrated why this distinction is needed. They are not resampled.

This prospective small dataset can test implementation and an initial ranking
effect. It cannot establish broad RAL-level generalization, real-robot validity,
or optimality of a representation. A negative graph-vs-sequence result will be
reported rather than repaired by inspecting the locked test.
