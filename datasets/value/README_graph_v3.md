# Graph-derived value module release

See [actual results](../../VALUE_GRAPH_REPORT.md) and
[reproduction commands](../../docs/value-v3-running.md).

- `skill_graph_v3.tar.gz`: 32 fresh sliding-stage pin configurations, train16/
  val4/test12; N16 x 2 paired physical repetitions = 1,024 labels. Contains
  pre-trial graphs, exact input/graph bindings, outcomes, images and snapshots.
- `two_family_v2.tar.gz`: unchanged earlier dataset. Only its old pin train12/
  val4 groups contribute here; its tests and connector groups are not trained on.
- `skill_graph_v3_execution.tar.gz`: 12 actual online decisions, 96 online
  validation rollouts plus 24 independent deployment rollouts.
- `skill_graph_v3_development.tar.gz`: training/validation pilot histories and
  checkpoints, including unsuccessful mean/attention/longer-training variants.
- `skill_graph_v3_source.tar.gz`: full source/assets snapshot used for the release.
- `skill_graph_v3_training_source.tar.gz` and `skill_graph_v3_collection_source.tar.gz`:
  exact historical Python source bytes matching the hashes recorded by every
  final training run and every fresh collection. These were reconstructed by
  reversing only the subsequent CLI timing-key rename and explicit-selection
  binding rejection; independent SHA256 checks establish byte identity. Use the
  full source archive for assets. This provenance is recorded in the manifest.
- `manifest_skill_graph_v3.json`: every group/file/archive hash, environment,
  work counts and offline costs. Worker/process time sums are not parallel elapsed
  time. Model weights and their hashes are under `models/value/v3`.

Verify without Torch or a GPU:

```bash
python scripts/audit_skill_graph_release.py
```

The published test has been inspected. Retain it for regression and predeclare
new test configurations for subsequent method development. No success-filtered
pool construction and no post-rollout state are used as inputs.
