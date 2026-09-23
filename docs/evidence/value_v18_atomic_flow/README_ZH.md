# V18：通用原子计划、价值 Top-K、端挡局部物理精验

## 范围与结论

首版目标是“从供料位置抓取端挡，搬运到底座声明区域，松爪撤离后保持稳定支撑”。`full_task_success` 始终为 `null`：本轮没有执行双销、手柄或完整滑台功能验收。

系统现在有两条可审计路径：

1. Codex CLI 从 TaskSpec、初始观测和注册原子技能及端口生成完整原子调用；`PlanIR`、技能端口和图谱前置条件拒绝非法计划。模型可直接给完整调用，也可在求解器事先给出的合法端口备选值上组合计划。后者适用于抓取位姿、路径解等连续分支。
2. 同一可执行图送入 V15 通用图价值模型排序，Top-K 候选按顺序从相同合法前序物理检查点进入 MuJoCo 精验。输入图、模型输出、分数、检查点和物理结果保存哈希；无效或资源截断不能算物理失败。

原子技能仍是 `detect / estimate_pose / estimate_grasp / plan_path / move / grasp / place / insert / press / wipe`。端挡、滑台和插销是对象及 CAD/任务数据，不是新增的技能类别。`select_grasp`、`inspect` 等属于原有技能执行和验收辅助接口。

## 自然候选覆盖

使用当前 RGB-D/CAD 几何目录及 `planner_v12.propose`，在看到本轮物理结果前按构造顺序提交每布局前 12 条候选。每个布局先通过正常物理执行取得一份合法 carriage 前序检查点；端挡在每条计分候选中都从供料位抓取、搬运、放置。成功和失败由相同的稳定支撑判据产生，没有按标签改目标、几何、验收或挑选正负参数。

| 随机供料布局 | 有效候选 | 真实局部成功 | 池类型 | V15 Value Top-4 保留成功 | 首次成功价值名次 |
|---|---:|---:|---|---|---:|
| 1950 | 12 | 5 | 混合 | 是 | 1 |
| 1951 | 12 | 3 | 混合 | 是 | 4 |
| 1952 | 12 | 4 | 混合 | 是 | 4 |
| 1953 | 12 | 8 | 混合 | 是 | 2 |
| 1954 | 12 | 4 | 混合 | 是 | 1 |
| **合计** | **60** | **24** | **5/5 混合** | **5/5** | **均值 2.4** |

价值模型 checkpoint SHA-256 为 `9720138360c20ffa6b07ea2af0c3d2062d5b54bc482f22675e135dbf1a89020b`。它原本按完整任务训练，未用这 60 条局部标签重训。5 个池的精确均匀随机 Top-4 预期命中数为 4.39，首次成功调用均值为 2.412；V15 对应是 5 和 2.4。因此这里只能证明旧模型能读取新图并将成功候选送入 Top-K，**不能证明排序优于随机**。原构造顺序 Top-4 在 4/5 个池中保留成功。

逐候选分数、标签和失败首步见 [analysis_5_layouts.json](analysis_5_layouts.json)。完整请求、输入图和物理结果见 [natural_pool_raw.tar.gz](natural_pool_raw.tar.gz)，SHA-256 `f17288e724b118e49880a72f62bc585836f8710125834863f0915431a76be638`。原始运行仍保存在 901 的 `/home/jia/twingraph-v17a-end-stop-place/artifacts/v18_end_stop_pool_{r2,more}`。

## 直接执行与模型组合

`run_atomic_end_stop_once.py` 在相同合法 carriage 检查点上直接执行 20 调用的端挡原子 `PlanIR`，不经过完整任务封装重写；`grounded_001` 的局部放置成功，`grounded_000` 失败。两者都实际执行了检测、位姿/抓取估计、路径规划、移动、夹取、放置及撤离后的独立稳定支撑检查。

模型在端挡任务、观测、原子技能及预执行端口备选值上输出了 4 个不同计划：原基线、居中抓取、更高抓取、较低路径净空。原始模型响应见 [llm_raw_output.json](llm_raw_output.json)，验证后端口编辑见 [llm_compositions.json](llm_compositions.json)。它没有看到物理成败标签。端口编辑只接受事先生成的值，编译后的每个计划均有自己的完整调用序列和价值图。

这 4 条计划的 V15 分数集中在 0.54953–0.54989，Top-2 顺序为“较低路径净空、原基线”。这组分数不能作概率校准声明。第一条在撤离后失去稳定支撑，第二条真实放置成功；两次精验从相同的物理/控制检查点开始，并逐条核对评分图哈希与实际执行图哈希。结果见 [llm_top2_summary.json](llm_top2_summary.json)、[首条失败](llm_rank1_result.json)与[第二条成功](llm_rank2_result.json)。模型请求、4 条完整计划及价值图、两次可恢复的完整初始孪生检查点保存在 [llm_compiled_raw.tar.gz](llm_compiled_raw.tar.gz)，SHA-256 `8db99a055fde289ac12dacff9cdaafeeac8e51ffd589c09b00bcfc58d97447ca3`。

## 复现与限制

```bash
# 901；与 V17 的局部物理运行时一致
PYTHONPATH=. MUJOCO_GL=egl EGL_DEVICE_ID=1 \
  ~/twingraph-v8-mj237/bin/python scripts/collect_end_stop_natural_pool.py \
  --out artifacts/v18_end_stop_pool --seeds 1950 1951 1952 1953 1954 \
  --pool-n 48 --submit-n 12

PYTHONPATH=. ~/twingraph-v8-mj237/bin/python scripts/analyze_end_stop_pool.py \
  --matrices artifacts/v18_end_stop_pool --checkpoint /path/to/value_v15_selected.pt \
  --k 4 --out artifacts/v18_end_stop_pool/analysis.json
```

模型生成在已登录 Codex CLI 的机器上运行，`scripts/generate_atomic_candidates.py` 接收任意 TaskSpec 和观测；`scripts/compose_atomic_from_graphs.py` 接收同一原子调用骨架的预执行图及求解器备选值。仓库不包含 API 密钥。

当前随机化主要是供料位置和场景扰动，5 个布局还不足以证明跨任务泛化。端挡判据是局部“放稳”，不保证插销孔对齐或后续整机装配可行。完整滑台任务仍需要处理 V16/V17 报告中的接合和可达性问题。
