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
