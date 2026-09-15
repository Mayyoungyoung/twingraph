# TwinGraph value v2 protocol (pre-test specification)

Research branch: `research/value-v2`. v1 data/checkpoints/splits are immutable.

## Fixed experiment questions

1. Under identical rollout allocation, does a learned or geometric ranker retain
   feasible/near-best candidates while reducing **measured** decision latency?
2. Does a training-only geometric logit plus a learned numeric correction improve
   over a small MLP with exactly the same geometry and plan features?
3. Does fixed-weight v1 inference transfer to actual connector assembly programs?

These are hypotheses, not expected winning conclusions. Runtime screening and
ranking-model contributions are reported separately.

## Physical families

The original sliding-stage pin remains unchanged. The new rigid connector has a
passively fixtured housing, two rectangular keyed sockets/inserts, a rear guard
and a locating pin. Both legal insert orders are supported. The housing is
already located; no claim that the robot assembles the passive fixture itself.
All four loose components are manipulated by the Panda with collisions enabled.
Matching stem/socket dimensions are coupled. No flexible or electrical behavior.

Before collection, development seeds 20000/20001 demonstrated full 76-call plans.
The initial origin-at-shoulder asset defect was fixed by moving the local body
origin 1 mm above the supporting shoulder. The support contract was preserved.
Pilot data is excluded from the research dataset.

Development revision before locked-test collection: wide ports (65--95 mm)
largely isolated the two inserts. A 40 mm paired probe preserved each object's
grasp and final goal but reversed insert order: the second release contacted the
previous insert's head and failed, while the other order completed. Main port
spacing now spans 38--70 mm (22 mm heads, linked rectangular keyed sockets).
Completed wide-port training groups are retained as augmentation, including
rule-friendly cases. The old source/data are archived; an explicit source
compatibility manifest admits this sampling-only change, never arbitrary mixed
execution implementations. No test labels were opened to make this revision.

## Selection semantics

- `K`: initial candidate count. `B`: physical rollout budget. `r`: repetitions
  per candidate. Only complete equal-sized repeat blocks are started.
- `first_verified`: stop on first candidate satisfying the acceptance rate.
- `best_within_budget`: evaluate the eligible candidates within B; choose by
  empirical success fraction descending, successful-rollout mean simulation time
  ascending, then rank. The same acceptance rule applies.
- Default acceptance rate: 0.5 for r=2 in research comparisons; deterministic
  regression retains r=1 and 1.0. Both parameters are stored per experiment.
- `allow_expand=False` confines both protocols to Top-K. Explicit expansion
  admits score-ordered batches of K; repeats never create a second candidate visit.
- Exhaustion/no accepted candidate means unresolved, never task infeasibility.

## Data / seeds / reference

Families use disjoint fixed seed ranges: connector 20010--20021 train,
20030--20033 validation, 20040--20045 locked test; pin 21010--21021 train,
21030--21033 validation, 21040--21045 locked test. Pilot 20000--20001 excluded.
All checkpoints of one geometry share a split. The collection and training code
must never choose candidate pools using outcomes. N=16/32/64 use a deterministic
nested permutation of actual grasp/route/force/descent-speed/order combinations.

Train initially uses 16 candidates, 2 paired perturbations/configuration.
Test labels use the independent `reference` domain. Online verification uses
`online`; independent selected-plan execution uses `deployment`. Their PRNG
SeedSequence domain identifiers are distinct. Unknown friction/gain variations
are sampled from the same bounded distribution; the model predicts robustness,
not the realization of an unobserved perturbation.

Actual checkpoint states come from physical execution of 18-call component
stages, never from placing bodies at targets. cp0=76 calls, cp1=58, cp2=40,
cp3=22 for connector, including final inspection of all four components.
cp3 is reserved as unseen remaining length. Pin programs have 19 calls.

## Model comparisons / stopping

Geometric ranking uses nominal future assembly jaw-sweep checks, including
previously assembled bodies, in scratch data. Necessary initial route checks are
shared; optional full rule scoring is only paid by consumers of those features.
Numeric MLP and geometric-prior residual consume identical numeric features.
The direct Transformer is trained independently with and without frozen
ImageNet ResNet18 features; direct inference skips the untrained prefix head.
All model selection uses train/validation only. Fixed v1 weights are never
updated in zero-shot evaluation. Few-shot uses a fixed count of train configs.
Its two target seeds are chosen by numerical seed order, with all their
checkpoints/geometry variants kept together. Adaptation starts from the pin-only
MLP, retains its input normalization, and trains with source data plus those two
target configurations. Target validation labels are additional cost, reported
separately; "two-shot" never means only two target configurations were labeled.

Primary online comparisons use seed 17 models. MLP/residual seeds 29 and 43
measure training variability offline. Additional budget/scale runs select one
primary model using validation Hit@4, Regret@4, parameter count, then Brier;
they do not choose a winner using locked-test results. Main and curve run sets
are separate statistical cells, even when configuration/budget overlap.

## Reporting

All-success, mixed and all-failure groups are retained. Near-optimal Hit@K is
undefined for all-failure groups. Feasible hit and independent deployment
success still include those groups. Finite reference rates are not guarantees.
Statistical unit is configuration; checkpoint siblings are clustered together.

Measure generation, necessary checks, optional geometry, render/vision,
inference, restore, physical verification, deferred solves and deployment.
Deferred solving is included in physical wall time and shown as a subset, not
double-counted. GPU timing synchronizes. Initial model loading is separate from
resident inference. Workers/cache policy must match across online methods.
One worker per online method; no speculative unused rollouts. Reference and
training collection can use parallel workers and report their separate cost.

Compare random/geometric/learned first-verified and geometric/learned/hybrid
best-within-budget; exhaustive candidate verification is a pool reference,
not global optimality. Report success, medians/P95, budget exhaustion,
rollouts/physics steps, quality-budget and success-latency curves with clustered
paired intervals or explicit small-sample limitations. Never infer speedup N/K.
