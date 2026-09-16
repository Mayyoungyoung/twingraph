# Skill graph value screening v3 — preregistered first experiment

Date: 2026-09-16. Base: c458c9c5e1f6b11b3a65b7983542c32dc90bbd2c.

## Question and scope

Can the same executable skill interfaces supply a value model's typed inputs
and state dependencies without a product-specific feature vector? Does relation
bias improve a matched-input compact model? The first experiment uses the
sliding-stage pin assembly suffix, including physical grasp, transfer, seating,
release and retreat (19 calls). This is not full sliding-stage assembly or a
real-robot result. Existing full-system v2 results are not module latency.

## Frozen first data plan

- Old v2 sliding_stage_pin train/val groups: development reuse only; old test is
  regression only. All outcomes and all-success/failure groups survive.
- Fresh train: seeds 41000–41015; val: 41100–41103; locked test: 41200–41211.
- N=16 nested, label-blind candidates; 2 paired repetitions per candidate.
- Train/val disturbance namespace `train`; test namespace `reference`.
- No hyperparameter selection using fresh test. Test is opened once after models
  and selection rules are saved. Report test as small-sample empirical rates.
- Main seeds 17,29,43 for graph and same-input sequence. 60 epochs, AdamW,
  BCE on empirical full-suffix success. Best validation Brier; no unique-winner
  ranking labels with only two repetitions.
- Models: optional geometric rule; numeric MLP87; old field-token Transformer;
  typed port pooling + sequence Transformer; identical graph relation model.
  Inference-only relation removal and type-preserving edge permutation are
  diagnostics, not separately trained fair alternatives. A matched training
  no-relation model is the main relation control.
- Small architecture: width64, 2 layers, 4 heads, single full-plan logit. Frozen
  ResNet18 vision is optional and tested separately; no new image backbone.
- Report Hit@1/2/4 (within epsilon=.1 of nonzero pool maximum), feasible hit,
  regret, average quality of selected K, Brier, and independent-group bootstrap
  intervals. All-failure groups excluded only from near-best Hit denominator.

## Input and execution restrictions

One graph contains pre-execution observation, goal predicates, typed ports,
deferred producers, versioned object/holding/scene/robot dependencies and the
exact PlanIR. No candidate IDs, object names, success labels, failure logs,
future measured states or geometric ranking scores enter the graph network.
Necessary geometry stays in candidate generation. Optional geometric scoring is
separate, with its cost included for the rule/MLP baseline.

Graph schema and edge integrity are checked before execution. State effects
denote possible successful effects, not measured future state. Edges do not
establish collision-free feasibility or authorize skipping intermediate calls.

## Cost and deployment

Measure graph construction, typed encoding, resident synchronized GPU inference,
sorting/export separately. Geometry baseline includes optional proxy computation;
MLP includes its proxy and feature costs. Report median/P95 across configurations,
batch N=16, same machine/threads/cache rights. Label collection and training costs
are separate. No N/K speedup claim. Execute selected PlanIR on paired independent
`deployment` disturbances on at least four fixed test configurations after freeze.

## Related work boundary

PIGINet already predicts feasibility from Plans, Images, Goals and Initial states
and measures refinement runtime ([paper](https://roboticsproceedings.org/rss19/p061.html),
[project](https://piginet.github.io/)). Our compact sequence control borrows typed
token fusion; it is not an exact PIGINet reproduction. The proposed distinction
is shared executable interfaces with explicit state provenance, evaluated by a
matched information/architecture relation ablation. A representation or residual
formula alone is not evidence of a new effective method.

## Development amendment before test opening

Training/validation-only trials found port-mean and attention pooling with only
fixed multiscale tanh values stayed near constant BCE after 60 epochs; extending
that model to 300 epochs did not fix it. Per-typed-port numerical normalization
(statistics fitted on training inputs only) lowered validation Brier. The final
matched graph/sequence comparison therefore uses attention pooling, this
normalization and 180 epochs, keeping minimum validation Brier selection. Seeds
17/29/43 are retained; all pilot histories are archived. This changes the initial
60-epoch graph budget, before inspecting any fresh test outcomes.

A generic port-set MLP is added to separate interface-derived inputs from the
need for a Transformer. Its vocabulary is formed from training graph input
keys/types, with generic numeric channels and occurrence counts; there are no
product-specific columns or geometric ranking proxies. It sacrifices exact
execution order. MLP87, field Transformer and port-set MLP retain 60 epochs.
The field control is a small width64/layers2 variant, not the unchanged v1
ResNet18/128-wide model or an exact PIGINet reproduction. This round's primary
graph models use available poses/geometry without images.

Before opening test, choose the deployed graph-derived method/seed by validation
Hit@4, then mean Top4 quality, then Hit@1, then lower Brier. Each run's epoch
selection still uses minimum validation Brier. The freeze also records the
calibration-selected method, so ranking and calibration choices are explicit.
Report all seeds, not just the selected method.
