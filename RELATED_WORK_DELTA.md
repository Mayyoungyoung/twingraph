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
  [Author-hosted paper](https://www.jiajunwu.com/papers/stap_icra.pdf).
  STAP coordinates skill dependencies by optimizing estimates of joint skill
  success; its mechanism uses learned skill values and state prediction.
  TwinGraph retains feedback controllers and actual physics verification.
- Ait Bouhsain et al., **Learning to Predict Action Feasibility for Task and
  Motion Planning in 3D Environments**, IROS 2023.
  [Author project](https://smail8.github.io/action-feasibility-prediction/).
  Predicting geometric feasibility to avoid planner calls is adjacent work;
  replacing a geometry call with a classifier is not alone a novelty claim.

## What this prototype tests

The prospective contribution is **a learned correction to assembly-dependent
geometric screening under an explicit physical verification budget**, evaluated
on executable remaining programs, with a separate deployment trial. A single
serialized program binds perception/planning/execution and deferred grasp frames.
The two families contain release/access and previously assembled-part effects.
Budget is counted in repeated physical rollouts, with both first-verified and
best-within-budget protocols. Necessary collision checks and optional geometric
scoring are separately metered.

The additive geometric-logit/residual formula is a deliberately modest testable
hypothesis. Neither that formula, a new industrial-looking asset, the Transformer,
nor PlanIR by itself establishes a publishable contribution. A positive result
must identify which execution dependencies the model generalizes, or show a
measured quality/cost improvement against geometric and small-model controls.

## Scope of evidence

Use NEXT_ROUND_REPORT.md and EXPERIMENT_TABLE.csv for actual outcomes. Simulation
deployment is not a real-robot experiment. Extrapolation beyond measured
configurations, checkpoint lengths and part combinations remains future work.
The research delta must be revised to fit results rather than assuming a win.
