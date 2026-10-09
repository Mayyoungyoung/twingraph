# V34 整理与 Panda 接口验证记录

2026-10-09，SSH 901 的 TwinGraph 项目统一至 `/home/jia/twingraph/`。48 个旧目录在实验记录逐个归档、逐文件核验后删除；完整归档 SHA256 复核通过。原 V33 的 50 个测试布局、450 个配对流程、2102 个原始结果文件以及模型保留。`organization_complete.json` 和 `final_integrity_verification.json` 保存整理与核查结果。

## 本次新鲜孪生精验

场景是预先声明的 seed 4000 名义 CAD 摆放方案，Panda 原生基座为参考原点，桌面 z=0、固定导轨使用 6 mm 垫高。不是实际测得的机器人现场。候选使用 grounded 几何候选池生成；本次没有使用 LLM 生成器，也没有训练新模型。

| 环节 | 实际结果 |
|---|---|
| 完整候选数 | 8 |
| 冻结价值 Top3 | grounded_018、grounded_038、grounded_036 |
| 首个精验候选 | grounded_018 |
| 独立新建孪生精验 | valid=true，success=true，约 390.21 秒 |
| 最终条件 | 装配、底座孔接合、夹具捕获、最终就位、释放/退回均通过 |
| 记录的控制轨迹 | 2247 原语、11861 控制采样 |
| 真机编译 | 87 条完整有界指令，44 段自由路线、9 段接触路线 |
| 单元测试 | 41 项通过；全部使用仿真/假机器人状态 |
| Python 3.8 硬件接口编译与预演 | 通过；状态 blocked_reference |
| 真机连接/运动/成功率 | 未连接、未执行、没有真机成功率 |

这是单一名义布局的接口验证，不是新的统计对照结果。V33 的价值 Top3 87.3%、随机 Top3 47.3%、Full8 86.0% 仍见 [原实验报告](../value_v33_robust_top3/README.md)。

## 文件与完整性

`release_summary.json` 给出精验、代码 SHA256、所有现场试验文件的哈希及远端绝对路径。请求、布局、价值顺序、选中 PlanIR、原子程序和 `validation/grounded_018/result.json` 保留原始字节。`source_snapshot/prepare_real_experiment_v34.py` 保存本次进程实际加载的桥接代码，来源信息和哈希与输出参考方案绑定。

完整 `reference_schedule.json`、逐控制步 TCP 记录、控制台日志、场景文件及三次开发精验留在 901 的 `experiments/final/`。Git 中保存本次最终试验的摘要和硬件指令提案，完整轨迹按 `release_summary.json` 的哈希索引查阅；没有丢弃实验记录。

本次发现 Git 自动换行转换会把 V33 冻结源文件从实际 CRLF 改为 LF，导致 Git 检出的运行时哈希与原实验不同。`git_source_eol_audit.json` 确认 180 个指纹文件的差异全部仅为换行；`.gitattributes` 现将 `simbench` 按原字节保存，Git 中的 215 个指纹文件逐项哈希已与 901 一致：

```text
42a1eda6fe8996d4312daa40c0855795c6e9142d7911427195d14bd306fd65be
```

冻结物理算法、CAD 和权重未改变，没有重训或改写 V33 实验结果。Git 的换行修复在忽略行尾空白后没有物理代码差异。

## 进入真机实验

`hardware_schedule_review.json` 具有完整有限动作，当前保留 146 项测量/复核阻塞原因；其数量不代表 146 个未实现功能。可按整阶段复核路线，接触流已合并。标定模板提案为每条指令 120 秒、完整程序 3600 秒，最长实际参考段约 118.66 秒、总超时预算约 2775 秒，参数仍需现场审核。该上限是有限等待预算，不是实测真机耗时。

需要实际初始关节/夹爪、真实 EE/TCP 与负载、抓持宽度/力、接触参数及整段避碰复核，并使用该实测布局重新做孪生精验。阶段和最终结果由现场视觉或操作者真实测量提供。当前有限 Cartesian 控制程序尚未接入在线视觉重定位；测量偏离则停止并重规划。真机执行不会由名义成功自动触发。

使用命令见 [完整流程](../../real_robot_workflow_v34.md)、[真机控制接口](../../../real_robot/README.md) 和 [摆放坐标](../../real_robot_coordinates_v34.md)。
