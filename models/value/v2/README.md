# v2 checkpoints

These are actual RTX 3090 training outputs, selected using validation data only.
The recommended primary checkpoint is `../best_value_v2.pt`, byte-identical to
`mlp_17/best.pt`. SHA-256:

`2d1b73af8d31e70aad10de1dc7c6d493aca154509addb13409d433b8216f8459`

Use `simbench.value.research_rank` or `simbench.value.research_decision` for v2
models. The original v1 model files and interfaces remain available.

| Directory | Training / purpose |
|---|---|
| mlp_17, mlp_29, mlp_43 | Shared two-family numeric MLP, three training seeds |
| residual_17, residual_29, residual_43 | Same numeric features; frozen fitted geometric logit plus correction |
| prior_17 | Fitted geometric logit alone; only its six-feature linear layer is active |
| no_vision_17 | Independently trained direct Transformer without visual features |
| vision_17 | Direct Transformer with frozen ImageNet ResNet18 features |
| pin_mlp_17 | Source-family-only training, fixed-weight transfer control |
| fewshot_mlp_17 | Adapt pin_mlp_17 with two target seed clusters; all target label cost recorded |

Each directory contains `best.pt`, the training/validation split manifest,
epoch history and measured training summary. `MANIFEST.json` hashes the weights.
Numeric models store 7,752 parameters in the shared checkpoint class; the prior
ablation does not execute the stored MLP block. The selected MLP executes its
single scalar full-task head and does not execute the stored prior layer.

Scores rank candidates; they are not calibrated reliability certificates.
The reference evaluation uses only two perturbations per candidate. See
[locked reference results](../../../docs/evidence/value_v2/locked_metrics.json)
and [protocol](../../../docs/value-v2-protocol.md). All deployment experiments
are in simulation.
