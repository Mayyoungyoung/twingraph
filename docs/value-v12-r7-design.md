# 新版价值模块：实现设计与实验边界

状态：r6 在线实验中止，旧权重及结果保留为历史；本文记录新的实现，**尚未使用新版物理标签训练，也没有新版成功率或加速结果**。r6 的 4 个训练布局只有 10 个全任务成功候选，其验证改善不足以证明价值模块必要性。

## PIGINet 提供的依据

PIGINet 将完整动作序列、初始状态关系及目标联合编码，预测计划能否完成运动细化。后续动作的需求可以约束前序选择，例如先打开的空间是否足够容纳后续物体；预测用于改变细化顺序，仍由运动规划验证。论文实验采用每类 150–600 个不同问题，不能与当前仅 4 个训练布局直接比较。[RSS 官方论文](https://roboticsproceedings.org/rss19/p061.pdf)、[作者项目页](https://piginet.github.io/)

官方项目的 Code 链接指向 kitchen-worlds；其公开 PIGINet 采集入口调用 `PDDLStreamAgent`。公开子模块的 `PVT` 接口把 facts/goals/visuals 与每个计划动作参数组合，批量评分并排序。不过神经模型及 dataset 类从外部 `fastamp` 路径导入，当前公开主仓不含该训练实现。本项目不声称复现官方训练，也不通过安装 CLIP 包冒充复现。[官方采集入口](https://github.com/Learning-and-Intelligent-Systems/kitchen-worlds/blob/main/examples/test_data_generation_pigi.py)、[公开评分接口](https://github.com/zt-yang/pybullet_planning/blob/master/pigi_tools/feasibility_checkers.py)

对本任务的设计推论是：先让价值输入看到“这个抓法和前序装配，会如何影响后面的动作”，再研究模型容量。夹爪碰撞、双层孔桥接及候选覆盖必须先在新物理合同下成立；学习器不能补救候选池中不存在的可行计划。

## 输入与模型

新输入 schema 为 `twingraph.graph_stage_ports.v12.r5_object_relative_grasp`。旧 schema（包括早期 `r4_geometry_catalog`）权重直接拒绝加载，不能把旧模型应用到新维度后声称迁移验证通过。

- 输入仍由受验证的 executable PlanIR/技能端口、RGB-D 观测和声明 CAD 组成，不另维护独立候选参数向量。名称、候选编号、seed、候选池大小、标签和 rollout 真值不进入特征。
- 保留 pickup 与 placement 各路径 yaw，新增源物体姿态到目标姿态的相对方向、CAD 尺寸、配合间隙与视觉拟合残差的关系、命令抓力相对接触预算、抓取宽度，以及深插/补压端口。接触预算不是实际所需力，拟合残差也不是已校准的概率方差。
- 非圆柱零件的抓取方向可使用显式 `grasp_yaw_frame=object`：编码器保存 frame 分类和命令角，并以初观测姿态计算世界抓取角，用于源到目标的方向关系。实际技能在后续 RGB-D 观测时重新绑定，初观测估计不伪装成未来已实现方向。缺失姿态保留 unknown，源路径 deferred yaw 保留遮蔽；圆柱 pin 的 world yaw 与 receiving placement yaw 独立。
- 对前序装配，使用声明目标构造“若前序成功，物体将占据这里”的条件几何。使用外接 XY 圆计算粗略分离量，作为学习特征；它不是完整夹爪扫掠或碰撞判定，配合件的负间距也不直接判失败。完整手掌/手指尺寸尚未声明时保留 unknown mask。
- 接收 `fixtures.guide_base`、`assembly_targets`、`receiver_geometry`、`fixture_relations`。尚未装配的松散端挡初始不对齐是正常状态：未完成 end_stop 时，观测到的 route=false 不作为已装配失败输入。深插终态与浅释放 waypoint 区分。
- 关系包括原执行/数据/物体状态等类型、阶段依赖、计划前后顺序、条件占位干涉和功能目标依赖。图模型使用一次双向关系消息传递，使后续要求能进入前序阶段表示。
- `geometry_conditions` 是完整归档图的一部分，使用 `twingraph.geometry_conditions.v13` 合同。它绑定当前 PlanIR、choices、order、RGB-D observation 和 CAD 摘要，并逐项核对选中目录行的 yaw/height/placement_yaw/插入深度、抓取宽度及方向 frame 与可执行 choices。相对方向目录还必须与初观测四元数和 `evaluated_world_yaw_rad` 一致。学习器和 cheap geometric baseline 读取同一目录。已计算的微小正净距与“通过不确定度余量要求”区分：unknown 可以有已知数值，缺失数值则另有 known=0；不能将未知当作零间隙或成功证书。

固定比较 `linear / shared / graph` 三组，width 48。linear 保留为图派生特征的线性基线，辅助损失权重明确为零。shared 使用共享阶段嵌入和阶段头；graph 在同一结构上增加关系消息传递。后二者的阶段监督能直接更新整计划使用的共享表示，整计划头也读取阶段概率和 active mask。不会把阶段概率当独立事件直接相乘。

`shared` 与 `graph` 可比较关系传播的增量作用；它们与 linear 的差异同时包含非线性和共享阶段监督，不能单独据此宣称辅助监督的因果收益。若要证明辅助项必要性，需另行预声明消融。

## 固定训练与选择

训练按布局等权：每个更新使用同一布局的完整候选池。仅训练集拟合归一化，尺度下限 0.1，输入裁剪范围 ±8。固定 seed 1212、120 epochs、AdamW、weight decay 0.03；linear 学习率 0.003，其余 0.001，梯度最大范数 2。

```text
L = whole-plan BCE + 1.0 × within-layout pair ranking + 0.25 × masked stage BCE
```

整计划 BCE 保留实际成功率分布；正负排序对按同池均值归一，各正例和负例在其类内均等参与，且每个布局等权。排序均值形式原版本已具备，新版不将其包装成新算法。阶段辅助先按已观察的每阶段正负类别分别求均值，再对已观察阶段等权；未执行阶段始终遮蔽。linear 不使用阶段项。全负布局保留，排序项为零，不伪造正例或重复成独立样本。

修正阶段标签语义：`retained_after_stroke` 失败属于保持阶段，不回写为“此前从未插入成功”；未发生的阶段不依据初始化的 false 标志标负。

选择顺序提前固定为：**最大验证 Hit@K → 最小归一化首成功调用数 → 最小 Brier → 最早 epoch / 固定模型顺序**。归一化调用为每布局首成功前所需次数除以该池 N；无成功池记 N/N，仍保留于统计。固定 Top-K 命中优先，完整顺序的调用效率第二，概率校准第三。缓存采集时间不参与选择，避免并发噪声选模型。若验证池全部无可行计划，训练器拒绝执行无信息的 utility 模型选择。

新版 split JSON 必须提前包含精确 `selection_criterion`；checkpoint、history、manifest 保存选择与损失配置。旧 r6 的 Brier 选择结果不改写。

## 新数据协议草案与采集门槛

由主任务确定的新分组为 train 2000–2031 共 32 布局、validation 2100–2107 共 8 布局、test 2200–2207 共 8 布局；开发 1800–1831。训练每布局 N=24，验证/测试 N=48，K=4。分组与最终物理源须在正式采集前冻结；旧 1600–1714 均不再当新盲测。具体状态以主任务新协议和配置为准。

loader 支持各布局完整池大小不同，在 manifest 保留池大小，按 `candidate_counts` 检查预声明 N。运行时/几何/初始观测/图/提案摘要必须一致，资源删失不能标作物理失败，也不能把旧 r6 或局部 Top-K 池拼成新版完整矩阵。

先在不计入训练/验证/测试的开发布局检查候选覆盖、完整夹爪碰撞及真正的两层孔装配；失败就修正并重新冻结。正式采集全负布局仍需保留，不能收集后只挑 mixed 布局。更多布局旨在增加独立场景变化，不承诺一定胜随机。

正式评估仍分开报告完整标签上的同池回放和独立在线执行。主比较固定全量/随机 Top-K/价值 Top-K；渐进价值必须同时对比随机早停，区分排序收益与早停收益。训练/验证之外的测试只能在模型和参数冻结后打开；本次实现阶段没有运行新拟合或物理任务。

## 候选公平性与便宜几何基线

新候选生成器应先排除已知的完整夹爪硬碰撞，保留必要几何检查通过、但接触/执行仍有不确定性的候选。不能为了放大价值模型的收益，人为保留几何上已明确不可行的抓取方向。

除全量、随机和学习价值方法外，增加由同一个公开几何 catalog 的余量排序得到的 **cheap geometric baseline**，其规则由主任务在测试前声明并实现。它与模型只能使用相同的执行前感知、CAD 和公开必要几何余量，不能读取真实标签。学习器相对随机的收益与相对便宜几何排序的增量收益分别报告；几何计算和模型打分均计入对应在线决策成本。此比较不增加模型家族、超参搜索或标签筛选。

开发 coverage gate 应先验证候选池能包含完整任务的可行计划，然后才冻结正式采集。该门槛不授权用正式测试标签修补答案：未见布局中若仍无解，按失败报告完整池覆盖；保留低成功率和全负布局，不能删掉或事后追加已知成功候选。

## 软件验证记录

在独立 `/home/jia/twingraph-v12-value-r7-tests` 目录，使用 CPU 9、空 CUDA 可见设备运行 `tests/test_value_v12.py`、`tests/test_value_v12_interactions.py`、`tests/test_value_v12_matrix.py`，结果 **42 passed in 3.42s**。覆盖缺失/变化 CAD 与接收几何、源/目标相对姿态、前序条件占位、松散端挡状态、阶段标签遮蔽、辅助梯度共享、排序选择、完整矩阵与新版分组/预算声明。fixture 是软件合同测试，不是物理成功率证据。未改 frozen r6，也未运行新的训练或物理 rollout。

接入统一几何目录后，增加 `tests/test_value_geometry_conditions_v13.py` 和 `tests/test_placement_catalog_v13.py`，五个文件合计 **63 passed in 4.82s**，资源和隔离目录相同。新增覆盖目录绑定过期拒绝、源/目标坐标与夹持宽度、原始 pad 侧面重叠、前序条件占位、确定穿透与小正余量的不同状态，以及真实完整夹爪几何查询的静态碰撞反例。没有运行物理控制、动力学步进或训练。
