你现在负责把一个“长程机器人任务中的 Terminal-State / Future-Value Propagation”研究实验真正落地到 MuJoCo + Franka Panda。

这是一个研究实验，不是普通工程任务。

============================================================
一、研究问题
============================================================

我们研究：

“当前 subtask 已经成功，并不意味着得到的 terminal state 对后续任务同样有利。”

形式化：

对于第 i 个 subtask：

G_i(x_i^+) = 1

并不意味着：

V_i(x_i^+) = P(G_{i+1:T}=1 | x_i^+)

在所有成功 terminal state 上相同。

我们希望证明：

1. 当前 subtask 可以同时存在多个合法 terminal states；
2. 所有这些 terminal states 都满足当前 subtask 的 local success predicate；
3. 不同 terminal states 对后续任务的成功概率不同；
4. terminal-state 差异可以来自：
   - Planner objective
   - Execution uncertainty
   - Planner × Execution interaction
5. 这种影响可以沿多个 stage 传播，而不仅仅影响紧邻的下一步。

最终论文核心叙事：

Local Success ≠ Downstream Usefulness

即：

G_i(x_i^+) = 1
≠
V_i(x_i^+) 相同
 一个 subtask 即使满足当前阶段的成功条件，也可能落入一个对未来任务不友好的 terminal state；不同的规划、抓取和执行过程会产生不同的合法 terminal states，而这些 terminal states 对后续任务的可完成性具有不同的 Future Value
============================================================
二、总体实验原则
============================================================

必须严格遵守以下原则：

[原则 1]
不能为了制造 spread 而直接人工修改最终目标位置。

例如禁止：

target_y = 0.12 / 0.13 / 0.14

然后声称这是 planner-induced variation。

必须：

同一个 initial state
+
同一个 symbolic subgoal
+
不同 planner objective
→ 不同 trajectory
→ 自然产生不同 terminal state

[原则 2]
Local Success 必须具有合理的工业语义。

不能故意把明显错误的状态定义成 success。

允许“工程上仍可接受但质量不同”的状态。

例如：

插入深度 8.2~9.5 mm 都算 local success，

但 8.2 mm 和 9.5 mm 的 downstream value 可以不同。

[原则 3]
必须记录真实 terminal state。

不要只记录：

success/failure

必须记录：

position
orientation
insertion depth
clearance
contact configuration
joint configuration
path length
joint motion
execution time
smoothness
position error
等。

[原则 4]
不要使用 qpos teleport 制造候选。

所有候选 terminal states 必须由真实闭环 robot execution 产生。

允许：

- planner
- controller
- perception noise
- grasp noise
- execution noise

但不能直接修改物体 qpos 生成候选。

[原则 5]
每个 candidate 必须：

initial state
→ execute current stage
→ local success gate
→ save terminal state
→ restore terminal state
→ execute remaining task

[原则 6]
必须检查 restore determinism。

由于 robosuite OSC controller 可能存在内部 goal state，
restore 后必须同步：

controller.goal_pos
controller.goal_ori

以及必要的：

ctrl
qfrc_applied
xfrc_applied
act
mocap

保证：

同一个 terminal state
→ 重复执行后续任务
→ 结果一致。

[原则 7]
不能因为某个任务设计漂亮就强行实验。

如果 pilot 发现：

- terminal state 实际分布非常窄
- 所有 planner 收敛到同一轨迹
- downstream success 在可达区域几乎恒定
- contact dynamics 不稳定
- task local success 定义不合理

必须停止扩大实验，并报告：

UNSUITABLE

然后修改任务设计。

============================================================
三、Task C：Fixture Loading
============================================================

优先实现 Task C。

工业原型：

矩形加工工件
+
双定位销夹具
+
压板
+
工具对准
+
压入/打标操作
+
卸料

使用 Franka Panda 单机械臂。

------------------------------------------------------------
C.1 Stage
------------------------------------------------------------

S1 Grasp Workpiece

S2 Place Workpiece into Fixture

S3 Engage Dual Locators

S4 Clamp Workpiece

S5 Move Tool to Nominal Operation Point

S6 Execute Operation

S7 Release Clamp

S8 Unload Workpiece

总共 8 stages。

不要人为增加无意义 stage。

------------------------------------------------------------
C.2 Geometry
------------------------------------------------------------

Workpiece：

- rectangular block
- 带两个 locator holes
- 可以用简单 box + cylinder holes/markers 表示

Fixture：

- fixed base
- shallow cavity
- two locator pins
- clamp

Locator：

- 两个可移动/可插入圆柱销
- 使用 slide joint 或合理简化机械结构

Clamp：

- hinge 或 slide joint
- 允许真实接触
- 不允许 teleport

Tool：

- Panda end-effector 上安装简单 cylindrical tool
- 只模拟接触/压入/打标
- 不模拟真实切削

------------------------------------------------------------
C.3 Local Success
------------------------------------------------------------

S1：

grasp stable

S2：

workpiece center 在 fixture cavity 内。

允许合理误差，例如：

x/y ±2 mm
yaw ±3~5 deg

但必须通过 pilot 确认这些阈值与实际几何一致。

S3：

不要简单定义“至少一个销进入”就认为完整定位。

使用：

primary locator engaged

作为进入下一阶段的最低 local success。

同时记录：

pin_A_depth
pin_B_depth

并区分：

FULL
PARTIAL
FAILED

例如：

A:
pin A full
pin B full

B:
pin A full
pin B partial

A/B 都可以满足最低 S3 local success，

但 B 必须产生更大的 residual mobility。

S4：

clamped = true。

注意：

Clamp 不能把所有 positional/orientation error 完全消除。

应存在合理的 residual pose：

x_residual
y_residual
yaw_residual

例如：

placement yaw = 3 deg
→ clamp 后可能变成约 2 deg

但具体数值必须通过 pilot 调整。

不能人为设定一个公式直接生成 residual pose。

必须让 MuJoCo 接触/约束自然产生。

S5：

tool 到 nominal operation point 上方。

S6：

tool 下压到指定深度并产生接触。

这里必须区分：

Local execution success

和

Final operation quality。

推荐：

S6 local success：

tool successfully reaches workpiece and executes nominal downward motion.

同时计算：

实际工具接触点
vs
workpiece true feature

的相对误差。

例如：

operation_error = distance(tool_contact_projection,
                             true_feature_position)

最终质量：

quality = f(operation_error, contact_force)

然后定义：

operation_success = quality > threshold

必须避免变成：

“因为没有做坐标变换所以工具打偏”。

真正原因必须来自：

Place
→ Locator contact
→ Clamp residual state
→ tool/workpiece relative configuration

------------------------------------------------------------
C.4 重点研究链
------------------------------------------------------------

核心链：

S2 Place
↓
S3 Locator engagement
↓
S4 Clamp
↓
Residual workpiece pose
↓
S5/S6 Tool alignment
↓
Operation quality

必须验证：

两个 candidate：

Candidate A：
placement 更接近 nominal

Candidate B：
placement 仍满足 S2 local success

但：

x_A^+ != x_B^+

并且：

G_2(A)=G_2(B)=1

但是：

P(G_6=1 | x_A^+)
>
P(G_6=1 | x_B^+)

------------------------------------------------------------
C.5 Planner effect
------------------------------------------------------------

至少实现：

P1 Shortest Cartesian Path
P2 Minimum Joint Motion
P3 Clearance-aware
P4 Smoothness-aware

注意：

当前使用 robosuite OSC_POSE，
不能假装自己是 joint-space controller。

如果不能直接实现 joint-space planner：

使用 waypoint-level joint-distance objective 生成 waypoint，
然后仍通过 OSC_POSE 执行。

论文中必须诚实说明。

关键不是 planner 名字，
而是：

同一个 subgoal
+
同一个 initial state
+
不同 planner
→ 不同 trajectory
→ 不同 terminal state。

必须通过实验验证 terminal-state diversity。

------------------------------------------------------------
四、Task B：Connector Assembly
============================================================

在 Task C 稳定后实现 Task B。

工业原型：

控制柜/设备面板
+
圆形连接器 housing
+
connector
+
locking clip
+
cable
+
cover
+
cover lock
+
continuity test

------------------------------------------------------------
B.1 Stage
------------------------------------------------------------

S1 Install / Seat Housing

S2 Insert Connector

S3 Lock Connector

S4 Route Cable

S5 Mate Cable

S6 Close Cover

S7 Lock Cover

S8 Continuity / Electrical Test

8 stages 足够。

不要为了 9 stages 人为增加步骤。

------------------------------------------------------------
B.2 Dependency Graph
------------------------------------------------------------

必须实现成：

S1 → S2 → S3
          ↓
         S5
S4 ─────→ S6 → S7 → S8
S5 ───────────────→ S8

即：

Connector alignment
会影响：

S3 locking
S5 cable mating
S8 test

Cable routing
会影响：

S6 cover closing
S7 cover lock

S5 cable mating
直接影响：

S8 continuity quality

------------------------------------------------------------
B.3 Local Success
------------------------------------------------------------

S1：

housing seated

radial error < reasonable threshold
z within seating tolerance
yaw within ±4 deg

但必须通过 geometry pilot 验证。

记录：

r
z
yaw

S2：

connector insertion：

8 mm <= depth <= 9.5 mm

radial <= 2 mm
yaw <= 5 deg

所有参数必须根据实际几何尺寸调整。

记录：

depth
radial error
yaw
pitch

S3：

locking：

clip_depth >= local threshold

区分：

partial lock
full lock

两者都可以满足 local success，

但 full lock 的 mobility 必须明显更低。

不能通过人工变量定义 mobility。

必须从 MuJoCo contact/constraint/response 中测量。

S4：

cable centerline 进入 cable channel。

记录：

lateral offset
clearance to cover
curvature/configuration

S5：

cable inserted/mated。

记录：

depth
radial
angular error

S6：

cover closed。

S7：

cover lock engaged。

S8：

continuity test。

continuity/contact quality 应当尽量是连续变量：

contact_quality ∈ [0,1]

最终：

test_pass = contact_quality > threshold

------------------------------------------------------------
B.4 最重要的两个 propagation channel
------------------------------------------------------------

CHANNEL A：

S1 Housing yaw
↓
S2 Connector insertion pose
↓
S3 Lock quality
↓
S5 Cable mating
↓
S8 Continuity

CHANNEL B：

S4 Cable routing
↓
Cover clearance
↓
S6 Cover closing
↓
S7 Cover lock
↓
S8 Test

------------------------------------------------------------
五、Candidate Types
============================================================

对关键 stage 生成三种 candidate。

TYPE I — Planner-induced

同一：

initial state
subgoal
object geometry
noise = 0

只改变 planner objective：

P1 shortest
P2 min-joint
P3 clearance
P4 smoothness

然后比较：

terminal state
path length
joint motion
time
clearance
future value

必须证明：

planner objective
→ trajectory difference
→ terminal-state difference

不能直接修改 endpoint。

TYPE II — Execution-induced

固定 planner。

增加：

perception error
grasp position error
pose estimation error
controller noise
small execution perturbation

但是所有候选仍必须通过合理 local success。

研究：

execution error
→ terminal state distribution
→ downstream success

TYPE III — Combined

Planner × execution noise。

例如：

P1 × N0
P1 × N1
P1 × N2

P3 × N0
P3 × N1
P3 × N2

用于研究：

planning × execution interaction。

============================================================
六、必须建立的核心数学量
============================================================

对于每一个 stage i：

Local success：

G_i(x_i^+) ∈ {0,1}

定义：

V_i(x_i^+)
=
P(
G_{i+1}=1,
G_{i+2}=1,
...,
G_T=1
|
x_i^+
)

也就是说：

Future Value

不是只看下一步。

例如 C：

V_2(x_2^+)
=
P(S3,S4,S5,S6,S7,S8 successful
|
x_2^+)

同时可以定义：

One-step value：

V_i^(1)(x_i^+)
=
P(G_{i+1}=1|x_i^+)

Long-horizon value：

V_i^(T)(x_i^+)
=
P(G_{i+1:T}=1|x_i^+)

论文中重点比较二者。

============================================================
七、实验矩阵
============================================================

Phase 0：

Physics feasibility

每个 stage 单独验证。

Phase 1：

Coupling pilot

只做：

10~30 candidates

目标：

验证 terminal-state diversity。

Phase 2：

Planner experiment

至少：

4 planners × 10 trials

同 initial state 配对。

Phase 3：

Execution experiment

至少：

3 noise levels × 10 trials

Phase 4：

Combined experiment

planner × noise

Phase 5：

Full long-horizon

完整：

S1 → ... → ST

统计：

current success
next-stage success
remaining-task success
full-task success

------------------------------------------------------------
八、关键统计
------------------------------------------------------------

必须报告：

1. Local success rate

2. Terminal-state distribution

3. Planner-induced terminal variance

4. Execution-induced terminal variance

5. Future success probability

6. Long-horizon success probability

7. Correlation：

terminal state dimension
vs
future value

例如：

Spearman correlation

rho(x_i^+, V_i)

8. Candidate pair comparison：

same initial state
same subgoal
different planner

9. Planner × noise interaction

------------------------------------------------------------
九、最重要的实验判据
============================================================

一个任务只有满足以下条件，才能进入正式大规模实验：

Criterion A：

存在至少两个自然产生的 terminal-state clusters。

Criterion B：

这些 terminal states 都满足当前 local success。

即：

G_i(x_A)=G_i(x_B)=1

Criterion C：

downstream value 显著不同：

V_i(x_A) != V_i(x_B)

Criterion D：

差异不能来自人为 endpoint 修改。

Criterion E：

MuJoCo execution repeatable。

Criterion F：

完整任务至少有 5 stages。

如果任意条件不满足：

不要强行扩大实验。

报告：

UNSUITABLE

并说明具体原因。

============================================================
十、实现顺序
============================================================

严格按照：

1. 审查当前 repository
2. 创建 Task C geometry
3. 验证 Panda manipulation feasibility
4. 实现 C stage executor
5. 实现 local predicates
6. 实现 terminal-state recorder
7. 实现 save/restore
8. 做 C coupling pilot
9. 检查 terminal-state diversity
10. 检查 downstream value diversity
11. 只有通过 pilot 后才实现 planner matrix
12. 再做 execution noise
13. 再做 combined
14. 最后完整 long-horizon rollout
15. Task C 稳定后再实现 Task B

============================================================
十一、非常重要：不要重复 T3/T9 的问题
============================================================

此前实验已经发现：

1. T3：
   planner trajectory 被狭窄几何走廊压缩，
   不同 planner 最终状态差异太小。

2. T9：
   虽然理论存在 coupling，
   但实际可达 terminal state 被几何边界压缩，
   downstream success ≈ 1。

因此这次必须在正式实验之前计算：

Reachable Terminal State Set

即：

R_i =
{x_i^+ | executable trajectory satisfies G_i=1}

然后检查：

R_i 是否足够宽。

必须优先寻找：

“局部成功区域很宽，但未来可行区域只占其中一部分”

这种任务。

理想结构：

          Local Success Region
     ┌───────────────────────────┐
     │                            │
     │   Future-good              │
     │   ┌───────────────┐        │
     │   │               │        │
     │   └───────────────┘        │
     │                            │
     │   Future-bad               │
     │                            │
     └───────────────────────────┘

也就是说：

G_i = 1 的区域明显大于 downstream-feasible region。

这是最重要的任务设计原则。

============================================================
十二、最终输出
============================================================

不要只给我代码。

每完成一个 phase，必须输出：

1. 做了什么
2. 为什么这样设计
3. 运行了多少 trials
4. 成功率
5. terminal state 分布
6. planner 是否真的产生不同轨迹
7. downstream success 是否真的分化
8. 是否通过当前 phase
9. 下一步做什么
10. 如果失败，明确标记：
    UNSUITABLE / NEEDS REDESIGN

特别是 Pilot 阶段：

不要因为实验已经运行很久就默认任务成立。

如果：

terminal states 被压缩
或
future value ≈ constant

必须停止。

============================================================
十三、最终研究目标
============================================================

最终希望得到这样的实验证据：

同一个 initial state：

             Planner A
                ↓
             x_A^+
                ↓
       G_i(x_A^+) = 1
                ↓
      Future success = 85%

             Planner B
                ↓
             x_B^+
                ↓
       G_i(x_B^+) = 1
                ↓
      Future success = 45%

以及：

同一个 planner：

             Noise low
                ↓
             x_C^+
                ↓
       G_i(x_C^+) = 1
                ↓
      Future success = 90%

             Noise high
                ↓
             x_D^+
                ↓
       G_i(x_D^+) = 1
                ↓
      Future success = 50%

最终证明：

Local Success
does not uniquely determine
Downstream Value.

并且这种差异能够：

Planner → Terminal State → Future Affordance → Long-Horizon Success

以及：

Execution Error → Terminal State → Future Affordance → Long-Horizon Success

进行传播。

现在不要直接开始大规模实验。

第一步先审查现有代码和环境，
然后完成 Task C 的 Phase 0 feasibility analysis，
并给出：

- geometry
- object dimensions
- joint design
- stage definitions
- predicates
- candidate generation mechanism
- planner implementation feasibility
- reachable terminal-state analysis

只有我确认 Phase 0 设计合理后，再进入正式实现。