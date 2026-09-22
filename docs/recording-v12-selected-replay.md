# 录制已选候选的额外闭环回放

`scripts/record_v12_selected_replay.py` 接入原 V12 `ClosedLoop` 和 `rollout(record=True)`，不改 `simbench`。录制是新的执行回放，不是在线计时执行的原始录像；其全部结果明确标为 `additional_recorded_replay=true`、`excluded_from_timing=true`。

只在在线五波结束、root 指定成功的 seed/method 后运行。准备脚本时没有读取任何在线测试结果，也没有访问服务器。

在与源记录相同的冻结 runtime 中，先预检：

```bash
PYTHONPATH=. python scripts/record_v12_selected_replay.py \
  --summary PATH_TO_METHOD/summary.json \
  --checkpoint PATH_TO_FROZEN_VALUE_CHECKPOINT.pt \
  --out results/recorded_replays/CHOSEN_REPLAY \
  --original-k 4
```

预检不创建输出目录、不加载 tensor 模型、不启动物理或渲染。确认输出后，在同一命令末尾加 `--run` 才开始录制；CPU/GPU/EGL 分配由调用者设置，脚本不抢占资源。

输入目录应为原在线方法目录，包含 `summary.json`、`request.json`、`twins/<selected>/result.json`、`deployment/result.json`。脚本只接受原摘要中已选且完整执行成功的候选，没有 `--candidate` 覆盖或重新排序入口；孪生和部署结果都必须 valid、全部六项功能判据通过、运行 SHA/seed/计划一致。value 方法的 checkpoint SHA 必须匹配 request 保存的 SHA。

闭环使用原选中孪生的 `boundaries`，不能用成功部署的观测冒充预测。seed、method、N、level 从原记录恢复，max_replans=1 与原 `run_method` 相同。top-k 方法的 K 从 request 恢复；value_early_stop 从 `progressive.initial_k` 恢复。all_twin/random_early_stop 的 request.k 只保存总预算 N，因此必须用 `--original-k` 提供原运行参数；当前协议原参数为 4。其他方法传入该参数时也会检查是否一致。

输出目录必须不存在，而且必须位于原在线方法目录之外。成功后主要文件为：

- `protocol.json`：来源 runtime、checkpoint、summary/request/两份结果的 SHA、原闭环配置和录制限制。
- `selected_proposal.json`、`selected_twin_expected_boundaries.json`：原选中计划和最初预测。
- `deployment/execution.mp4`、`execution.png`、`result.json`、完整执行日志。
- `closed_loop/`：此次新回放的真实监控事件及必要时的后缀验证。
- `recording_summary.json`：回放真实成功/失败、输入不可变校验、结果/视频 SHA；不写回在线摘要。异常则保存 `recording_failure.json`。

录制可能改变墙钟负载并暴露新的失败，不能保证复现原来的成功；失败也会保留，脚本不会自动挑另一候选。沿用现有 recorder 的 900 秒活动执行预算，未录制原运行是 600 秒；因此录制时间不能用于方法加速对照。后缀孪生仍沿用精确仿真检查点同步。显示相机与腕部传感输入分开，此回放不能证明真实机器人或真实相机标定已经完成。

本地仅运行了合成文件接口检查：7 passed，覆盖成功来源、选中孪生预测、K 恢复、runtime/checkpoint 一致性以及防覆盖；没有运行物理或读取验证/测试布局。
