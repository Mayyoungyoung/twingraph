# v7 复现命令

```powershell
cd F:\RAL\twingraph-dev
$env:PYTHONPATH='.'
pytest -q tests/test_system_v5.py
python -m simbench.value.collect_v7 --out results/value_v7/dev_collect --seed 71000 --groups 4 --n 12 --repeats 1 --split development --level L0 --workers 1 --timeout 360
```

正式数据在远程 Linux 上使用同一脚本，将 `--split train/val/test`、`--level L0/L1/L2` 和场景 seed 清单固定写入 manifest；不得复用旧 v6 标签。训练可复用 v6 的四视角网络定义，但必须用 v7 `inputs.json`、v7 source hash 和新标签重新训练；checkpoint 选择只看验证 Brier。
