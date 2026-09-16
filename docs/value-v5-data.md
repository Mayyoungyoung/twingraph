# v5 原始数据与复现归档

[归档清单](../datasets/value/v5/manifest.json)绑定24个压缩包及冻结模型记录。
[独立审计](evidence/value_v5/package_audit.json)核对了1496个包内成员、1153个原始文件、源码快照及场景几何；原始文件没有在打包期间改变。

正式采集为72个配置、864条候选程序。正式系统为128次孪生验证和8次独立目标执行。
开发数据另外保留24条候选以及1次孪生、1次目标执行，不能计入正式结果。
每条正式候选均保存完整输入、执行参数与结果；系统试验另含控制步状态轨迹。

`archives/source_000.tar.gz` 的成员路径从仓库根开始；其他包的成员路径从运行目录开始。
例如在 Linux 下，将包路径记为 `PACKAGE`，在一个新的空目录复原：

```bash
PACKAGE=/absolute/path/to/datasets/value/v5
REPRO=/absolute/path/to/new/reproduction
cd "$PACKAGE"
sha256sum -c manifest.sha256
mkdir -p "$REPRO/results/v5"
tar -xzf "$PACKAGE/archives/source_000.tar.gz" -C "$REPRO"
for category in data development evidence models system; do
  for archive in "$PACKAGE"/archives/"${category}"_*.tar.gz; do
    tar -xzf "$archive" -C "$REPRO/results/v5"
  done
done
```

上面的第一步仅验证清单本身。全部压缩包的大小和SHA256列在清单的 `artifacts` 中，
使用前还应逐个验证；公开仓库副本的复核结果随本次交付提供。
不要把数据包直接解压到仓库根，也不要再人为添加一个 `data/` 或 `systems/` 前缀。

基础源码已匹配正式运行时及几何。`source_index.json` 记录精确字节快照：
正式运行时快照为 `snapshots/666ccc5b6c66cbff830a488559cc7ddcdef315d3abee39a1dcf869360b5d1170/`。
需要重放历史实现时，依据对应记录选择快照，将其中 `simbench/` 和可选 `scripts/`
覆盖到新建复现目录，再逐项核对对应 `source.json`。不能对现有实验结果目录做覆盖式重跑。

解包后四个权重位于 `results/v5/models/NAME/best.pt`。复用冻结选择时，
系统命令使用 `--checkpoint-root results/v5/models`，测试预测导出器可用
`--checkpoints NAME=path`；原选择文件中的历史绝对路径和哈希保持不变。
具体实验命令见[运行指南](value-v5-running.md)。

归档含失败结果和早期派发脚本；后续路径迁移工具没有重新训练、调阈值或重跑物理实验。
完整数字孪生与目标场景都由MuJoCo实现，数据不代表真实硬件实验。
