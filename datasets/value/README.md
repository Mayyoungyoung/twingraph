# Value datasets

## Sliding-stage executable-plan dataset v4

`stage_graph_v4.tar.gz` contains all 24 requested configurations, including 4 failed preparations. The 20 reached groups contain 160 unique candidate programs and 320 paired physical continuation executions: 160 train, 64 validation, 96 test. Plans start from robot-executed sliding-stage checkpoints and finish multiple remaining parts. Test was opened after validation model selection was frozen.

`stage_graph_v4_execution.tar.gz` preserves the separate online validation and independent execution decisions, traces, failed preparations, and actual terminal-state PNGs. `stage_graph_v4_development.tar.gz` separately retains development pilots and models. `stage_graph_v4_source.tar.gz` contains source, assets, and exact formal source overlays; explicitly listed historical development source gaps are not claimed reproducible.

The [release manifest](../../docs/evidence/value_v4/manifest.json) records every archived member and its hash. The [runbook](../../docs/value-v4-running.md) explains extraction, source reconstruction, and the release-to-repository path mapping. The [report](../../VALUE_V4_REPORT.md) gives actual denominators and limitations. These are simulation outcomes, not real-robot measurements; there is no LLM-generated or camera-image candidate distribution in this release.

## Pin suffix dataset v1

- 100 parameterized configurations (seeds 1000–1099).
- 8 bound grasp/approach plans per configuration, 4 paired perturbations each.
- 3,200 physical trials; all prefixes successful; 1,330 complete suffix successes.
- Separate immutable pre-rollout inputs and post-rollout labels, initial images and frozen ResNet-18 features, scene XML and numeric initial physics arrays.
- Fixed 13-call template in one sliding-stage pin subtask family. No cross-family or real-robot generalization claim.

Extract `pin_suffix_v1.tar.gz` from the repository root to restore `results/value/data/`. `inputs.json` is the ranker interface; `outcomes.json` is used only for training/evaluation. Every group includes input hashes, source hashes, repeat identifiers and a completion marker. Pairing and conditional labels are validated by the loader.

`pin_suffix_v1_source.tar.gz` contains the precise collection source and assets. Extract this into an isolated directory only when reproducing collection; it predates the final inference-only input validation improvements. The data and collection source hashes are recorded in [manifest.json](manifest.json).

Train/validation/test are separated by configuration hash, never by candidate row. The trained checkpoints include the full dataset fingerprint and chosen validation epoch. See [training evidence](../../docs/evidence/value/README.md).
