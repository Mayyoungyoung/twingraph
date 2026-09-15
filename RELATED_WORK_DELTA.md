# Related work and the v2 experimental claim

## Primary sources consulted

- Yang et al., **Sequence-Based Plan Feasibility Prediction for Efficient Task
  and Motion Planning (PIGINet)**, RSS 2023.
  [Paper and proceedings](https://roboticsproceedings.org/rss19/p061.html),
  [author project](https://piginet.github.io/).
  It predicts plan feasibility from the plan, initial scene and goal, prioritizes
  expensive refinement, and evaluates planning runtime. Therefore a Transformer
  ranking plans with scene images is prior art, including its runtime rationale.
- Agia et al., **STAP: Sequencing Task-Agnostic Policies**, ICRA 2023.
  [Author project](https://sites.google.com/stanford.edu/stap/home),
  [official implementation](https://github.com/agiachris/STAP).
  STAP coordinates skill dependencies by optimizing estimates of joint skill
  success; its mechanism uses learned skill values and state prediction.
  TwinGraph retains feedback controllers and actual physics verification.
- Ait Bouhsain et al., **Learning to Predict Action Feasibility for Task and
  Motion Planning in 3D Environments**, ICRA 2023.
  [Author project](https://smail8.github.io/action-feasibility-prediction/).
  Predicting geometric feasibility to avoid planner calls is adjacent work;
  replacing a geometry call with a classifier is not alone a novelty claim.

## What this prototype tests

The prototype investigates **learning full-program assembly quality from
geometry and execution parameters under an explicit verification budget**, evaluated
on executable remaining programs, with a separate deployment trial. A single
serialized program binds perception/planning/execution and deferred grasp frames.
The two families contain release/access and previously assembled-part effects.
Budget is counted in repeated physical rollouts, with both first-verified and
best-within-budget protocols. Necessary collision checks and optional geometric
scoring are separately metered.

The additive geometric-logit/residual hypothesis was tested against a direct MLP
with the same features. Both reach configuration-weighted near-optimal Hit@1=1.0
on the locked reference set in three training seeds, versus 0.806 for the nominal
geometry rule. The residual does not improve retention over the simpler MLP, so
the validation-selected MLP is the recommended implementation. Geometry Hit@2
already saturates; the useful empirical distinction is which plan is examined
first and what that costs under actual repeated validation.

The actual six-configuration budget study supports that distinction: at B=2
(one candidate, two online repeats), MLP solves and independently executes all
six configurations, while geometry does so on five. At B=4 and B=8 both solve
six. In the separate main B=8 comparison, mean measured decision time is
90.51 seconds for MLP and 96.20 seconds for geometry, with about 5% fewer
verification physics steps for MLP. Its selected-plan secondary reference
execution-time regret is 0.255 simulation seconds versus 1.917 for geometry.
The fitted geometric prior remains a competitive simple control; the result
does not establish a need for residual scoring or a large visual model.
These are exploratory paired simulation results, not reliability guarantees.

A controlled development intervention holds each insert's grasp fixed and
reverses their legal order. At a 40 mm port spacing, one order completes and the
other blocks the second release against the previously installed insert. Wider
rule-friendly cases remain in the data. This establishes an executable stage
dependency, but the model's observed gain has not yet been causally isolated to
that dependency: grasp height, orientation and release behavior also vary.

The positive research direction is a small, identity-free full-suffix predictor
combined with explicit physical validation and independently measured deployment.
It remains necessary to establish broader external validity and a causal feature
ablation before claiming a general new TAMP method. The residual formula, an
industrial-looking asset, the Transformer, and PlanIR are not novelty claims.

## Scope of evidence

Use NEXT_ROUND_REPORT.md and EXPERIMENT_TABLE.csv for actual outcomes. Simulation
deployment is not a real-robot experiment. Extrapolation beyond measured
configurations, checkpoint lengths and part combinations remains future work.
Online protocol and budget results are reported separately from ranking results;
PIGINet already establishes learned plan prioritization and runtime evaluation.
This study does not claim to originate either idea.
