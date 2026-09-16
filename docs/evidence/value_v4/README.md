# v4 frozen experiment evidence

See the [Chinese experiment report](../../../VALUE_V4_REPORT.md) and [reproduction instructions](../../value-v4-running.md).

| Evidence | Contents |
|---|---|
| [frozen.json](frozen.json) | Twelve immutable model hashes and validation-selected `sequence_17`; published before test results |
| [predictions.json](predictions.json) | Raw candidate outcomes, per-model logits/probabilities, source hashes and setup attrition |
| [ranking_audit.json](ranking_audit.json), [ranking_summary.csv](ranking_summary.csv) | All seeds, raw/calibrated errors, exact random expectations, checkpoint strata and configuration bootstrap |
| [module_timing.json](module_timing.json) | Warm model, graph construction/checking/encoding/inference/export measurements |
| [deployment_summary.json](deployment_summary.json) | Actual online validation and independent execution, requested denominators, parallel and per-policy timings |
| [selected_top_k.json](selected_top_k.json) | Full Top-K graphs copied from the frozen selected policy's actual decisions |
| [device_diagnostic.json](device_diagnostic.json) | Validation-input-only CPU/CUDA and candidate-order stability checks |
| [candidate_audit.json](candidate_audit.json) | Real candidate variation and descriptive correlations with observed failures |
| [deployment_integrity_audit.json](deployment_integrity_audit.json) | Independent checks against raw trials, hashes, selection and render sidecars |
| [figures](figures/) | Static PNG/SVG figures and every plotted numerical value |
| [manifest.json](manifest.json), [manifest.sha256](manifest.sha256) | Byte-level release and archived-member checksums |

The original package uses `evidence/` paths; in this repository those map to `docs/evidence/value_v4/`. `models/` and `datasets/` paths remain repository-relative. The manifest itself is unmodified. Convenient copies of CSV, diagnostic JSON and figures supplement the original manifest-listed paths under `original_paths/` and `run_provenance/`.

The formal data archive contains 24 requests, 20 reached groups, 320 continuation executions and 4 failed preparations. Online evaluation separately contains 80 validation and 14 independent execution trials. All failures remain. Development data is archived separately; any missing historical development source is explicitly listed, while exact formal source bytes are retained.

Five eligible test pools yielded selected-model near-best Hit@4=5/5; uniform random expected Hit@4 was 80.86%. This is not 100% assembly success. Counting two setup failures, reference feasible coverage is 5/8. Independent selected-policy execution succeeded 3/8 requested trials, the same as source order. Only two independent deployment configurations reached the decision point. Graph relations showed no additional benefit; calibration worsened selected-model test error. None of these simulation measurements establishes real-robot reliability or broad generalization.
