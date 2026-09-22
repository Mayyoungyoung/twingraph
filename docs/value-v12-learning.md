# V12 价值模型：统一接口、真实标签和训练边界

当前模型是白色腕部 RGB-D **原生 r6 的 linear，epoch 7**，按预声明的最低验证 Brier 选中。
训练已完成；[原生 r6 报告与权重](evidence/value_v12_system/value_training/native_r6/README_ZH.md)
保存完整门槛审计、三模型训练历史、选择证明和来源摘要。
截至本次更新，未读取预留测试 1660–1663；五波在线实验已启动，但本文件不报告其结果，
也不把参与模型选择的验证成绩当作未见测试结论。

[r4 历史迁移报告](evidence/value_v12_system/value_training/r4/REPORT_ZH.md) 中的 sequence，
以及此前的 [r3 历史报告](evidence/value_v12_system/value_training/REPORT_ZH.md)，均作为旧开发材料保留。
这里的“历史训练 r4”与物理运行时版本分别记载，不能混为当前原生模型。

用户确认实际零件全白，并进一步明确真机使用腕部双目深度相机、尚未手眼标定。
最终 r6 使用腕部 RGB-D 仿真，已完成一次从头全任务开发验证；历史彩色记录仅作为
独立开发材料，不混入原生 pilot 拟合。仿真成功不代表真实相机或机器人硬件已验证。

## 模型与统一接口

`ValueRankerV12(checkpoint, device="cpu").score(graphs)` 返回与输入同序的分数，
`rank` 是同一接口的别名，不返回排序索引。调用方用降序排序得到 Top-K。
checkpoint 的 `sha256` 属性供实验请求绑定具体权重。

输入是 `planner_v12.normalized_graph` 原样输出。编码前重编译检查
`assembly` 与其可执行 PlanIR、技能接口一致；检查全任务封装中的
`order/choices/cleaning` 与图一致。读取的值来自执行端口与前置观测，
不另建与执行不一致的 candidate 数值向量。

当前实现针对这套装配任务，将原子调用汇聚为擦拭、五个装配零件、往返推动、
保持检查共八个阶段。保留有向执行/数据/物体状态等关系、阶段顺序、已完成状态、
物体 RGB-D 位置/方向/质量、机器人关节与末端状态，以及端口的已知/延后/缺失掩码。
新 `joint_checked_v10` 抓取路径、学习插销策略、接触力上限、下压力、分段提起均进入编码。
`capabilities` 可以保留在对象图内，不会导致编码异常；当前模型不将其额外学习为能力嵌入。

角度使用 sin/cos。数值先按明确物理单位缩放，再使用**仅训练集**统计量归一化，
标准差下限为 0.1，最终限幅 ±8；不会再把未见过的 90° 抓法放大到十万量级。
这使输入有限，不代表解决了分布变化或完成概率校准。

当前 schema 为 `twingraph.graph_stage_ports.v12.r2_path_yaw`。每个阶段按执行顺序分别保留
源抓取路径与目标装配路径的 yaw sin/cos、known/deferred/present 掩码，不能把两者混成一个抓取角。
当前最多支持每阶段四个 yaw-bearing 路径，超出时显式拒绝；旧 schema checkpoint 也会明确拒绝，
不会静默补零或用错误维度执行。原生 r6 每阶段有172维规范特征，当前选定权重为
`evidence/value_v12_system/value_training/native_r6/fit/linear.pt`，SHA256：
`a45649de1a7a97fde94f69bc4817929d08080257803550d36243b96e54aae66b`。

模型包含完整计划可行性头、局部阶段成功头，以及同一布局候选间的成对排序损失。
未执行阶段用 mask 排除，不能因整任务失败就把所有后续阶段标为失败。
阶段头依据初始状态和计划预测后续阶段，尚不是从每个真实中间状态采样训练的通用技能 Q 函数。

设计参考 [PIGINet](https://piginet.github.io/) 的整计划可行性排序思路，以及
[SayCan](https://say-can.github.io/) 用状态条件化技能价值约束高层动作选择的思路。
这里没有复现 PIGINet 的 CLIP 图像/文本编码器，也未训练 SayCan 原论文的强化学习价值函数。
不能把借鉴思路写成完整复现。

本轮实际选中的是规范化技能图特征上的线性价值头。图谱承担状态、技能、执行端口及
可执行计划的一致输入；这与“图消息传递网络是否优于其他预测头”是两个不同问题。
原生验证 Brier 未支持 graph 头优于 linear，不能为了强调图谱而改选模型或隐藏这一结果。

## 历史数据开发与新几何训练分开

历史 V9/V10 实际物理记录提供 744 次试验、312 个候选聚合记录。
旧 V9 未保存 pre-execution graph hash，因此适配器从归档的候选参数与 RGB-D
重建接口；它不读取最终物体位置为模型特征。这个来源弱于 V12 原生图哈希绑定，
不能声称旧记录执行的就是当前新技能实现。

历史公开材料已经被看过，因此保留原布局划分只是回溯开发控制，不构成新的盲测。
模型和 epoch 仅按验证 Brier 选择，K=4 固定，历史留出结果不用于事后换模型。
历史 r4 的 `sequence.pt` 是当时按该规则选出的迁移开发权重；历史 `graph.pt` 的某项
留出指标更好也不能取代它而伪装成同一次无偏选择。二者均不是当前原生 r6 的已选权重，
旧 V9/V10 标签也没有混入本次原生拟合。

## 已完成的原生 r6 Pilot 训练

原生采集、训练与验证均绑定物理运行时
`c6b79761f3555a682f402f4f408b0f94d35b4c50b4fad65f5c2251ac1f8824da`，
协议 SHA256 为 `5c40af05a41d2933d5ab6e6e7285184841fea66c86b86c26ba7ce3048bf1d22d`。
每布局 N=48、K=4，整个采集没有更换布局、剔除全负布局或修改验收标签。

| 用途 | 布局 | 全任务成功 | 失败 | 有效候选 |
|---|---:|---:|---:|---:|
| 训练 | 1610 | 7 | 41 | 48 |
| 训练 | 1611 | 0 | 48 | 48 |
| 训练 | 1612 | 3 | 45 | 48 |
| 训练 | 1613 | 0 | 48 | 48 |
| 验证 | 1630 | 0 | 48 | 48 |
| 验证 | 1631 | 6 | 42 | 48 |

共288条真实 MuJoCo 全任务结果：训练10正/182负，验证6正/90负。六矩阵完整，
逐候选 `valid=true`、`resource_censored=false`、`timeout=false`，请求、结果、归档输入图、
初始观测与运行时摘要一致。训练布局1610和1612提供正负排序监督。
审计记录见 [training_gate_audit.json](evidence/value_v12_system/value_training/native_r6/training_gate_audit.json)。

采集及CPU18/19辅助任务结束后才运行一次训练：CPU9、CUDA设备1、数学库线程1，
实际子进程 exit 0，含数据加载的训练进程墙钟29.091秒。保持seed1212、每模型120 epochs、
width48、原有优化器、学习率、损失权重和仅训练集归一化；每个模型选最早最低验证Brier的epoch，
再按验证Brier选择唯一模型，没有依据验证命中率或测试表现改选择准则。

| 固定模型 | 选中epoch | 验证Brier |
|---|---:|---:|
| **linear（当前已选）** | **7** | **0.05777312** |
| sequence | 5 | 0.05878179 |
| graph | 3 | 0.05842031 |

[selection_proof.json](evidence/value_v12_system/value_training/native_r6/fit/selection_proof.json)
核对了三份120-epoch历史、最早最小值、跨模型选择和权重摘要；实际checkpoint的r6绑定、
训练/验证布局、K4和schema也已核查。权重在访问预留测试之前冻结。
训练曲线的 [PNG](evidence/value_v12_system/value_training/native_r6/training_history.png) 与
[PDF](evidence/value_v12_system/value_training/native_r6/training_history.pdf) 已归档。

### 验证矩阵诊断，不是未见测试

下面只回放已参与模型选择的两个验证布局。1630全池无成功；1631有6个成功，
已选linear排序第2次找到一个成功。随机值是均匀无放回排列的精确期望，不是挑选某次随机结果。

| 方法 | 可行计划命中率 | 平均验证次数 | 缓存物理验证成本参考，秒/布局 |
|---|---:|---:|---:|
| 全量孪生 | 50.00% | 48.00 | 8087.26 |
| 精确随机Top-4 | 21.24% | 3.65 | 596.72 |
| **已选价值Top-4** | **50.00%** | **3.00** | **517.28** |
| 精确随机早停，预算48 | 50.00% | 27.50 | 3519.00 |
| 价值渐进早停，批量4、预算48 | 50.00% | 25.00 | 2991.93 |

成本是18-worker采集阶段各候选记录耗时的回放参考，**不是在线决策墙钟**。
验证数据用于模型选择，且只有两个布局，不能据此声称未见任务上优于随机、统计非劣效或独立部署等效。
价值渐进和随机早停在无可行候选时仍验证全部48个；同一有限有效标签池的存在性覆盖不保证
新的扰动部署一定成功。最终结论须等待独立在线记录及冻结测试回放，由另一个报告审计。

## 原生矩阵入口与主协议

V12 原生矩阵入口 `scripts/train_value_v12_matrix.py` 接收：

```
<root>/seed_<id>/collect/request.json
<root>/seed_<id>/collect/candidates/<name>/input_graph.json
<root>/seed_<id>/collect/candidates/<name>/result.json
```

加载时必须有 request 中每个候选的结果，且没有额外候选；验证候选/图/执行哈希绑定、
同一初始 RGB-D、几何版本和接口。request与每个result的runtime摘要必须一致，
所有参与拟合的布局也必须来自同一个runtime；仅几何版本字符串相同不能混合版本。
部分矩阵或无效结果会直接报错，不能将未执行候选默认为失败。r6真实执行超时标为无效资源截尾，
不能进入训练；同步重规划时间只从父执行截止预算扣除，完整实测墙钟仍保留这些成本。
训练至少需要一个同布局有成功也有失败的候选池；全失败池不能单独用来证明排序价值。

预设划分文件为 `simbench/configs/value_v12_data_split.json`，与 V12 协议一致。
训练 1610–1625、验证 1630–1635 保持原设置，新的预留测试为 1660–1671。
开发 1600–1607 不能作未见测试；1640 已用于白色零件孤立技能开发，因此原先整段
1640–1651 测试预留已撤销。加载器会拒绝把这些开发/撤销布局重新声明为未见测试。
训练程序只打开 train/validation 结果，测试由独立 `--evaluate-test` 路径加载冻结权重，
拒绝训练/验证布局出现在测试集合中。上述原生pilot只完成主协议的固定子集；
主协议34个布局尚未全部完成。下列命令仅说明完整主协议入口，**不能覆盖已完成的原生pilot输出**：

```bash
CUDA_VISIBLE_DEVICES=1 taskset -c 9 /home/jia/twingraph-v8-mj237/bin/python -m scripts.train_value_v12_matrix \
  --roots results/printed_matrix --splits simbench/configs/value_v12_data_split.json \
  --out results/printed_value_fit --device cuda

TG_SELECTED=$(/home/jia/twingraph-v8-mj237/bin/python -c \
  'import json; print(json.load(open("results/printed_value_fit/selection.json"))["selected"])')
CUDA_VISIBLE_DEVICES=1 taskset -c 9 /home/jia/twingraph-v8-mj237/bin/python -m scripts.train_value_v12_matrix \
  --roots results/printed_matrix --splits simbench/configs/value_v12_data_split.json \
  --out results/printed_value_test --evaluate-test --checkpoint "results/printed_value_fit/$TG_SELECTED.pt"
```

最后一个 checkpoint 应使用实际 `selection.json` 指定的名字。
完整矩阵评估报告 Top-K 首次成功、所需验证次数、实测候选耗时之和、精确均匀随机期望、
全量孪生、随机成功早停及价值渐进早停。它是完整物理标签的离线回放，不能替代重新运行的在线耗时或独立执行结果。
本次pilot在线使用已预声明的五个方法wave、四seed分别绑定CPU0–3、共享EGL设备1，轮间屏障；
报告并发服务负载下实际墙钟，不能称为独占单机器人延迟。完整资源说明见
[已采纳资源方案](evidence/value_v12_system/protocol/ADOPTED_RESOURCE_PLAN_ZH.md)。

## 测试与限制

相关回归覆盖角度周期性、候选名字不能泄漏、缺失值掩码、图篡改拒绝、策略差异、
完成阶段别名、图关系确实影响模型、局部标签删失、随机期望与穷举一致、矩阵完整性、
图标签绑定、划分隔离和 capabilities 兼容。
新增路径装配方向确实进入执行端口对应特征、deferred 区分和旧 checkpoint schema 拒绝检查。
历史阶段曾将两个新几何开发的真实 `input_graph.json` 输入旧选定模型，验证接口可运行；
本轮进一步用完整288条原生绑定图实际训练，不能再把历史接口冒烟测试称为当前模型的完整验证。

原生训练已有两个mixed布局，但样本仍少；冻结测试结果截至本文件更新尚未读取，
不能提前承诺价值模块在未见布局胜过随机或保持全量孪生的部署成功率。
今后若根据pilot测试修改模型，已看过的pilot布局不能再称未见测试。
更大规模独立布局和真实硬件仍需单独验证；增加模型层数或候选数本身不能替代这些证据。
