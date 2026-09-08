# 接口与候选重构验证

- `tests.txt`：31 项回归通过，包含原 16 项。
- `assembly.json`：种子 0 完整装配，108 次组件调用通过。
- `perturbations.json`：本轮实现的种子 1/19 回归，观测噪声标准差 0.25 mm、供料 XY 扰动 ±0.5 mm 及小偏航扰动；不是大规模泛化评估。
- `standalone.json`：30 个原组件在独立环境执行成功，没有重录整套精选视频。
- `recording-smoke.json`：抬升组件录像 7.72 秒，检查真实抓持、150 mm 抬升和可视化；原始视频保存在 results/contract_refactor/recording_smoke。
- `candidate-inputs.json`：仿真前生成的 6 个方块抓取候选（2 抓法 × 3 路线）。
- `candidate-outcomes.json`：从同一快照执行两种不同抓法，2/2 成功，只覆盖抓取前缀。
- `assembly-candidate-summary.json`：装配中 6 次抓取的候选及所选前缀的搬运扩展索引。
- `catalog-graph.json`：契约派生的图谱、类型及兼容接口。
- `provenance.json`：基线提交、源码指纹、控制器未改变的校验和及限制。

候选输入与回放结果分别存储。上述验证没有训练或评估终态价值网络，也没有完成全部候选的整链孪生比较。其他 evidence 文件保留自此前整理和实验，不应混作本轮新结果。
