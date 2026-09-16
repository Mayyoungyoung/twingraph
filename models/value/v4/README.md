# Value v4 checkpoints

Four complete executable-graph input scorers × seeds 17, 29, 43:

| Kind | Parameters | Distinction |
|---|---:|---|
| compact | 43,201 | Training-input-only constant/exact-duplicate column removal, 56,275 → 641 |
| port_mlp | 3,603,777 | Same typed-port pooling without column removal |
| sequence | 639,282 | Call sequence with typed-port attention pooling |
| graph | 639,282 | Matched sequence architecture plus declared relation attention bias |

All use the new sliding-stage continuation dataset: 10 reached training configurations, 160 physical outcomes, and 4 validation configurations. Each directory retains `best.pt`, `history.json`, `summary.json`, `source.json`, and `split.json`. Training uses whole-plan physical success BCE, not a weighted geometric reward.

`best_graph_input.pt` is a byte-identical alias of **sequence_17/best.pt**, selected on validation before test collection. “graph_input” describes the shared input format; this selected architecture is a sequence model. All twelve models remain published. Do not select a replacement using the released test set.

The checkpoint binds the skill interface and encoding source signatures, training-only input transforms, split, and validation-only calibration. Its probability is not a demonstrated calibrated physical reliability estimate: selected-model test Brier worsened after calibration. Graph relations did not improve the matched control in this experiment.

See the [full results](../../../VALUE_V4_REPORT.md), [runbook](../../../docs/value-v4-running.md), and [frozen manifest](../../../docs/evidence/value_v4/frozen.json). Source and artifact SHA256 records are preserved in the [release manifest](../../../docs/evidence/value_v4/manifest.json).
