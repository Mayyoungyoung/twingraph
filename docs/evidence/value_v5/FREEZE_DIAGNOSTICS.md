# Frozen validation diagnostics: clarification

The original `thresholds.json` remains unchanged. Its file SHA256 is
`90ff39b9483f48999f731badea8f5302ffa9869bcbf162b462c19578d8ee5d15`.
This clarification was prepared after model/threshold freezing and before reading
any final test outcomes.

The fields
`methods.*.validation_primary_screening.{feasible_hit,quality}.bootstrap95`
are **invalid confidence intervals and must not be reported as uncertainty**.
The freeze helper inadvertently computed them from one bootstrap resample. Both
endpoints therefore equal that one resample's mean, which can differ from the
observed mean. This affects the diagnostic intervals for all four models.

The point estimates and denominators remain valid. For selected model `mlp_29`,
validation feasible Hit@4 is `11/12 = 0.9166666667`; mean selected nominal quality
is `35/48 = 0.7291666667`. The saved quality interval
`[0.7708333333, 0.7708333333]` has no confidence interpretation.

These fields were not used to select a model or threshold. The selected model
remains `mlp_29`, selected by nominal validation Brier, with unchanged probability
threshold `0.5695365870155553`. Its checkpoint SHA256 is
`7149a504030ce60de1a06a86f132664cc19adbf4d8dc438fdf6d1ddae91dd479`.
The original model-selection file SHA256 is
`291ef301deb64d68d552e355f409a4b89d4e561b52ba706e44eee66fcd31ee86`.

Locked-test classification and screening use the frozen model, threshold and
ranking settings; the evaluator does not consume these validation intervals.
Test uncertainty is independently computed with the protocol's 2,000
configuration bootstrap resamples. This remains a finite-sample diagnostic,
not a reliability guarantee. No collection, fitting, model selection or threshold
selection was repeated to address this documentation issue.

The helper is corrected for future freezes to emit validation point summaries
without confidence intervals. Regression tests verify that legacy interval
metadata cannot change locked-test decisions or metrics. The existing frozen
artifact and its hash are preserved for provenance.
