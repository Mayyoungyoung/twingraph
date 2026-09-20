# V8 audit and fixes (development evidence)

Base: `220e6c4`; working branch: `research/value-v8-credible-screening`.

| Issue | Finding and minimal reproduction | Change and check |
|---|---|---|
| Pin aperture | Confirmed in `pin_geometry.py`: the old predicate allowed a parallel pin with 4 mm center offset in a 5.5 mm ring despite its 3.3 mm radius. The original `scene.holed_plate` square opening is only 4 mm half width. | The evaluator now uses the smaller inscribed radius, includes the shaft radius and tilt projection, and requires uninterrupted coverage from entry to required depth. `test_pin_radius_and_original_plate_limit_both_apply` covers the counterexample and inconsistent apertures. This circular bound is conservative for the square corners. |
| Candidate state | Confirmed in `Session.snapshot/restore`: `stage_targets` and `execution_relocalizations` were omitted although `full_task_v7` mutates them. | Both are explicitly copied or removed on restore; `test_session_checkpoint_restores_execution_targets` checks it. Full A/B order invariance remains unverified. |
| Task requirement | Confirmed in `stage_v7.build_pool`: 80 or 90 mm `stroke_minimum` was sampled per candidate. | All candidates now use 80 mm; generation and execution reject an override. Pool and rejection tests pass. |
| Trial perturbations | Confirmed in `collect_v7.trial_spec`: mass, damping and perception draws were emitted but the collector used `PhysicalRunner`, which applies only friction and actuator gain. | V7 trials now declare only applied fields. `PhysicalRunner` rejects unsupported fields and records requested/applied values. Perception, mass and damping perturbation experiments remain to be implemented. |

These fixes alter the protocol. V7 success labels must not be pooled with V8 labels. The old successful video is retained as historical evidence, but its geometry label requires re-evaluation under V8.
