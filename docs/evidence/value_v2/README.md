# v2 evidence index

Start with the Chinese [research report](../../../NEXT_ROUND_REPORT.md) and
[run guide](../../value-v2-running.md).

| Evidence | What it establishes |
|---|---|
| [EXPERIMENT_TABLE.csv](../../../EXPERIMENT_TABLE.csv) | One row per actual online decision, physical work and exclusive/subset timing fields |
| [online_summary.json](online_summary.json) | Method × protocol × budget × family summaries, median/P95 and independent deployment |
| [paired_intervals.json](paired_intervals.json) | Configuration-cluster paired bootstrap intervals against geometry |
| [OFFLINE_RANKING_TABLE.csv](OFFLINE_RANKING_TABLE.csv) | Locked family/length/pool-type ranking strata; no cached-outcome runtime claim |
| [locked_metrics.json](locked_metrics.json) | Candidate and problem-level source for offline metrics |
| [validation_recommendation.json](validation_recommendation.json) | Checkpoint choice made from validation metrics |
| [data_training_summary.json](data_training_summary.json) | Label counts, splits, actual training and collection cost |
| [physical_checks.json](physical_checks.json) | Actual parameter changes, trajectories and nested pools |
| [order_intervention.json](order_intervention.json) | Paired development intervention on legal insert order |
| [selected_error_cases.json](selected_error_cases.json) | Post-test analysis of geometry/MLP first-choice differences |
| [vision_audit.json](vision_audit.json) | Frozen pretrained ResNet18 weight verification |
| [v1_regression_metrics.json](v1_regression_metrics.json) | Original v1 ranking regression |
| [tests.txt](tests.txt) | Final full regression test output |
| [table_audit.json](table_audit.json) | Read-only table-to-raw, common-input and paired-perturbation checks |
| [cold_start.json](cold_start.json) | Fresh-process startup with OS caches warm, separate from resident planning |
| [plot_manifest.json](plot_manifest.json) | Locally rendered scientific figures bound to exact metric content and image hashes |
| [environment.json](environment.json) | Actual interpreter, Torch, MuJoCo and GPU versions |
| [release_checksums.json](release_checksums.json) | Release-directory SHA-256 manifest, mapped to repository paths by the dataset README |
| [release_integrity.json](release_integrity.json) | Local verification of all archives, 975 data files, 143 source files, models, plots and executable export |
| [new_configuration/top_k_with_provenance.json](new_configuration/top_k_with_provenance.json) | Complete fresh-scene Top-K PlanIRs with model/code/input hashes |
| [new_configuration/decision.json](new_configuration/decision.json) | Actual validation and independent deployment of the selected fresh-scene plan |

Raw archives and split/source manifests are described in
[datasets/value/README_v2.md](../../../datasets/value/README_v2.md). Full trained
weights and their histories are in [models/value/v2](../../../models/value/v2/README.md).

All results are simulation experiments. Online validation, final deployment and
offline reference use separate paired perturbation domains. A failed budget is
not a proof of task infeasibility. The selected test configurations are now
public regression cases, not a fresh test set for future method development.
