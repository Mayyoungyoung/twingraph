# Plan value checkpoints

| File | Objective | Recommended use |
|---|---|---|
| `plan_value_direct_v1.pt` | Entire remaining-plan success | Default for the released pin-subtask dataset; selected using validation Hit@2/Regret |
| `plan_value_v1.pt` | Prefix success × conditional suffix success | Research variant; needs additional prefix-failure data |

Both use a shared 128-dimensional, three-layer Transformer with frozen ResNet-18 image features. Model/config/objective, protocol, dataset fingerprint, epoch and validation metrics are stored in each checkpoint. Training seed 17; 100 configurations split 67/17/16; 3,200 physical MuJoCo trials.

Use `python -m simbench.value.rank --checkpoint <file> --input <inputs.json> --k 2`. The released data archive includes cached vision features; scoring those cases requires no download of visual encoder weights. New observations use torchvision's official ImageNet ResNet-18 weights.

Scores are not calibrated guarantees. Both models are limited to the current simulator pin-assembly family. The geometric baseline matches the recommended model on this small test set. See the [full evidence and limitations](../../docs/evidence/value/README.md).


## v2 两家族研究模型

[完整模型与训练记录](v2/README.md)。推荐 [数值特征 MLP](best_value_v2.pt)，使用 `simbench.value.research_rank` / `research_decision`。v1 文件保持原样。
