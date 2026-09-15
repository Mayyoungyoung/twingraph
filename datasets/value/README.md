# Pin suffix dataset v1

- 100 parameterized configurations (seeds 1000–1099).
- 8 bound grasp/approach plans per configuration, 4 paired perturbations each.
- 3,200 physical trials; all prefixes successful; 1,330 complete suffix successes.
- Separate immutable pre-rollout inputs and post-rollout labels, initial images and frozen ResNet-18 features, scene XML and numeric initial physics arrays.
- Fixed 13-call template in one sliding-stage pin subtask family. No cross-family or real-robot generalization claim.

Extract `pin_suffix_v1.tar.gz` from the repository root to restore `results/value/data/`. `inputs.json` is the ranker interface; `outcomes.json` is used only for training/evaluation. Every group includes input hashes, source hashes, repeat identifiers and a completion marker. Pairing and conditional labels are validated by the loader.

`pin_suffix_v1_source.tar.gz` contains the precise collection source and assets. Extract this into an isolated directory only when reproducing collection; it predates the final inference-only input validation improvements. The data and collection source hashes are recorded in [manifest.json](manifest.json).

Train/validation/test are separated by configuration hash, never by candidate row. The trained checkpoints include the full dataset fingerprint and chosen validation epoch. See [training evidence](../../docs/evidence/value/README.md).
