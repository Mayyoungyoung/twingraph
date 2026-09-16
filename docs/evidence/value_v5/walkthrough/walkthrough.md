# 滑台装配完整流程：指定记录案例

公开副本说明：仅将媒体链接迁移为本目录终态图与仓库演示视频的相对路径；其余实验内容保持原记录。权威原始 Markdown 保留在完整原始归档中。

案例：`/home/jia/twingraph-v5-formal-20260916/results/v5/systems/case_61401`。这是一次已有记录的讲解；没有重新采样候选、重新排序或重新仿真。

结果完成后，为演示完整成功流程，选择预注册 TopK 成功案例中最小的 seed 61401。该展示选择不参与性能估计；总体报告保留全部 4 个预注册案例和失败。

本文件保留该案例 TopK 与全量验证两种策略的全部候选结果和目标试验。总体性能请以所有预注册案例的汇总报告为准。

## 1. 任务与规划来源

从供料位置抓取并装配滑台的滑块、端挡、左右定位销和手柄。全部零件应装到目标位姿，释放后稳定，机器人退出。

来源：llm_proxy；提供方：OpenAI；模型：Codex assistant in this task。

保存的 `proposed_orders` 被 `stage_v5.build_pool` 实际消费：

1. 滑块 → 端挡 → 左销 → 右销 → 手柄
2. 滑块 → 端挡 → 右销 → 左销 → 手柄

`proposed_skill_programs` 保留在来源记录中；当前编译器消费的是上述部件顺序。`stage_calls` 展开各部件的原子技能与检查调用，几何/运动求解器绑定抓取、控制参数与初始关节路径。实际调用数见下表（正式五部件计划通常为 101 条）。LLM 代理没有计算关节轨迹。

声明的最终任务验收谓词与阈值（物理条件，不是加权奖励）：

|部件|谓词|目标世界位置 (m)|位置容差 (m)|倾斜容差 (deg)|最低末端净空 (m)|
|---|---|---|---:|---:|---:|
|滑块|seated_released_retracted|[0.105,0.085,0.824]|0.0015|3|0.02|
|端挡|seated_released_retracted|[-0.017,0.085,0.836]|0.0015|3|0.02|
|左销|seated_released_retracted|[-0.017,0.053000000000000005,0.855]|0.0015|3|0.02|
|右销|seated_released_retracted|[-0.017,0.117,0.855]|0.0015|3|0.02|
|手柄|seated_released_retracted|[0.105,0.085,0.872]|0.0015|3|0.02|

`seated_released_retracted` 同时要求位置/倾斜通过、释放抓持、没有手指接触且末端已退让。

## 2. 实际候选与冻结价值输出

下表保持候选原始输入顺序，名次和概率直接读取冻结模型记录。概率是模型估计，并不等于该候选已通过物理验证。TopK 未验证项不记为失败。

|输入序号|候选 ID|部件顺序|调用数|冻结概率|原始 logit|记录名次|TopK|TopK 验证|全量验证|
|---:|---|---|---:|---:|---:|---:|---|---|---|
|1|0bfa0ac76255dd74db2e|滑块 → 端挡 → 左销 → 右销 → 手柄|101|0.49848|-0.00608064|3|是|2/2，超时 0；被选中|2/2，超时 0；被选中|
|2|b9d06b1db75db688fccb|滑块 → 端挡 → 右销 → 左销 → 手柄|101|0.0997931|-2.19953|11|否|未验证|0/2，超时 0|
|3|16a2af746a99c9ec71e7|滑块 → 端挡 → 左销 → 右销 → 手柄|101|0.0602373|-2.74734|12|否|未验证|0/2，超时 0|
|4|e8c79160c927b84af7f1|滑块 → 端挡 → 左销 → 右销 → 手柄|101|0.157026|-1.68053|8|否|未验证|0/2，超时 0|
|5|6fe16d5ba8494de21a8e|滑块 → 端挡 → 右销 → 左销 → 手柄|101|0.477857|-0.0886309|5|否|未验证|1/2，超时 0|
|6|70979e438f048976f731|滑块 → 端挡 → 左销 → 右销 → 手柄|101|0.61658|0.475059|2|是|2/2，超时 0|2/2，超时 0|
|7|7556d5033cd144cc4929|滑块 → 端挡 → 左销 → 右销 → 手柄|101|0.125233|-1.94378|9|否|未验证|0/2，超时 0|
|8|902986fb91933d9389ed|滑块 → 端挡 → 左销 → 右销 → 手柄|101|0.495341|-0.0186358|4|是|2/2，超时 0|2/2，超时 0|
|9|2ca70bd6f80999c6bf94|滑块 → 端挡 → 右销 → 左销 → 手柄|101|0.661277|0.66899|1|是|2/2，超时 0|2/2，超时 0|
|10|d19463dd8bcef0848ad2|滑块 → 端挡 → 左销 → 右销 → 手柄|101|0.221326|-1.25796|7|否|未验证|2/2，超时 0|
|11|f28a041b35ffd2a07545|滑块 → 端挡 → 左销 → 右销 → 手柄|101|0.32809|-0.716838|6|否|未验证|2/2，超时 0|
|12|77cce49a6c674bd1848c|滑块 → 端挡 → 右销 → 左销 → 手柄|101|0.121205|-1.98107|10|否|未验证|0/2，超时 0|

冻结检查点 SHA256：`7149a504030ce60de1a06a86f132664cc19adbf4d8dc438fdf6d1ddae91dd479`。

物理参数是已执行接口的取值摘要，不是人为加权评分。`force` 是夹持力；接触插入/压靠的其他固定参数仍完整保存在原始 calls 中。初始轨迹摘要不替代输入中的全部路点。

### 候选 1：`0bfa0ac76255dd74db2e`

初始路线索引：2；已知/延后参数计数：{"known":206,"deferred":15}。

|部件|yaw (rad)|height (m)|clearance (m)|force (N)|speed (m/s)|
|---|---:|---:|---:|---:|---:|
|滑块|0|0|0.98|3.2|未记录|
|端挡|1.5708|-0.001|0.98|3.2|0.006|
|左销|1.5708|0.001|0.98|3.2|0.006|
|右销|1.5708|0|0.98|3.2|0.006|
|手柄|0|0|0.98|3.2|0.006|

初始轨迹 `transfer`：3 个路点 × 7 关节；世界坐标目标 (m)：[-0.25572499998497666,-0.2036632999872876,0.9329829565938731]；几何 SHA256：`bca0d557403db4a24e2d4529c75653484e12a844760feea8605e24ec23920781`。

|策略|重复编号|成功|超时|已执行调用|错误|
|---|---:|---|---|---:|---|
|top_k|0|是|否|106|无|
|top_k|1|是|否|106|无|
|full|0|是|否|106|无|
|full|1|是|否|106|无|

### 候选 2：`b9d06b1db75db688fccb`

初始路线索引：0；已知/延后参数计数：{"known":206,"deferred":15}。

|部件|yaw (rad)|height (m)|clearance (m)|force (N)|speed (m/s)|
|---|---:|---:|---:|---:|---:|
|滑块|1.5708|0.001|0.98|3.2|未记录|
|端挡|0|-0.001|0.98|3.2|0.007|
|右销|1.5708|0.001|0.98|3.2|0.007|
|左销|1.5708|0|0.98|3.2|0.007|
|手柄|1.5708|0|0.98|3.2|0.007|

初始轨迹 `transfer`：3 个路点 × 7 关节；世界坐标目标 (m)：[-0.25572499998497666,-0.2036632999872876,0.9339829565938731]；几何 SHA256：`fe8dfd6760670285832ae01ffe55e2f4e4f9c314ed4d00dceca51a62051a7378`。

|策略|重复编号|成功|超时|已执行调用|错误|
|---|---:|---|---|---:|---|
|full|0|否|否|32|lift: lift or retention failed {'object_lift_m': 1.650893040050505e-05, 'held': False}|
|full|1|否|否|32|lift: lift or retention failed {'object_lift_m': 1.6339407475585155e-05, 'held': False}|

### 候选 3：`16a2af746a99c9ec71e7`

初始路线索引：2；已知/延后参数计数：{"known":206,"deferred":15}。

|部件|yaw (rad)|height (m)|clearance (m)|force (N)|speed (m/s)|
|---|---:|---:|---:|---:|---:|
|滑块|1.5708|-0.001|0.98|3|未记录|
|端挡|0|-0.001|0.98|3|0.006|
|左销|1.5708|0.001|0.98|3|0.006|
|右销|1.5708|0|0.98|3|0.006|
|手柄|0|-0.001|0.98|3|0.006|

初始轨迹 `transfer`：3 个路点 × 7 关节；世界坐标目标 (m)：[-0.25572499998497666,-0.2036632999872876,0.9319829565938731]；几何 SHA256：`0d585b463985bdc6648f60870c186a6db8fc2538cb2cda2201821caa8c37cea9`。

|策略|重复编号|成功|超时|已执行调用|错误|
|---|---:|---|---|---:|---|
|full|0|否|否|32|lift: lift or retention failed {'object_lift_m': 1.642111084454445e-05, 'held': False}|
|full|1|否|否|32|lift: lift or retention failed {'object_lift_m': 1.622806375578545e-05, 'held': False}|

### 候选 4：`e8c79160c927b84af7f1`

初始路线索引：2；已知/延后参数计数：{"known":206,"deferred":15}。

|部件|yaw (rad)|height (m)|clearance (m)|force (N)|speed (m/s)|
|---|---:|---:|---:|---:|---:|
|滑块|1.5708|0.001|0.98|2.8|未记录|
|端挡|1.5708|-0.001|0.98|2.8|0.007|
|左销|1.5708|-0.001|0.98|2.8|0.007|
|右销|1.5708|0.001|0.98|2.8|0.007|
|手柄|1.5708|0.001|0.98|2.8|0.007|

初始轨迹 `transfer`：3 个路点 × 7 关节；世界坐标目标 (m)：[-0.25572499998497666,-0.2036632999872876,0.9339829565938731]；几何 SHA256：`fe8dfd6760670285832ae01ffe55e2f4e4f9c314ed4d00dceca51a62051a7378`。

|策略|重复编号|成功|超时|已执行调用|错误|
|---|---:|---|---|---:|---|
|full|0|否|否|62|inspect_seat: assembly pose outside tolerance {'position_error_m': 0.08764862066717596, 'tilt_deg': 89.97123160504157}|
|full|1|否|否|36|align_axis: verified held(end_stop) required; contact lost|

### 候选 5：`6fe16d5ba8494de21a8e`

初始路线索引：2；已知/延后参数计数：{"known":206,"deferred":15}。

|部件|yaw (rad)|height (m)|clearance (m)|force (N)|speed (m/s)|
|---|---:|---:|---:|---:|---:|
|滑块|0|0|0.98|3|未记录|
|端挡|1.5708|-0.001|0.98|3|0.007|
|右销|1.5708|0.001|0.98|3|0.007|
|左销|1.5708|0|0.98|3|0.007|
|手柄|1.5708|0.001|0.98|3|0.007|

初始轨迹 `transfer`：3 个路点 × 7 关节；世界坐标目标 (m)：[-0.25572499998497666,-0.2036632999872876,0.9329829565938731]；几何 SHA256：`bca0d557403db4a24e2d4529c75653484e12a844760feea8605e24ec23920781`。

|策略|重复编号|成功|超时|已执行调用|错误|
|---|---:|---|---|---:|---|
|full|0|是|否|106|无|
|full|1|否|否|36|align_axis: verified held(end_stop) required; contact lost|

### 候选 6：`70979e438f048976f731`

初始路线索引：2；已知/延后参数计数：{"known":206,"deferred":15}。

|部件|yaw (rad)|height (m)|clearance (m)|force (N)|speed (m/s)|
|---|---:|---:|---:|---:|---:|
|滑块|0|-0.001|0.98|3|未记录|
|端挡|1.5708|-0.001|0.98|3|0.006|
|左销|1.5708|-0.001|0.98|3|0.006|
|右销|1.5708|0.001|0.98|3|0.006|
|手柄|1.5708|0.001|0.98|3|0.006|

初始轨迹 `transfer`：3 个路点 × 7 关节；世界坐标目标 (m)：[-0.25572499998497666,-0.2036632999872876,0.9319829565938731]；几何 SHA256：`67fa61475887597dee983ec6769eaee29fb9b575f43792d36702b854d33fbdcc`。

|策略|重复编号|成功|超时|已执行调用|错误|
|---|---:|---|---|---:|---|
|top_k|0|是|否|106|无|
|top_k|1|是|否|106|无|
|full|0|是|否|106|无|
|full|1|是|否|106|无|

### 候选 7：`7556d5033cd144cc4929`

初始路线索引：2；已知/延后参数计数：{"known":206,"deferred":15}。

|部件|yaw (rad)|height (m)|clearance (m)|force (N)|speed (m/s)|
|---|---:|---:|---:|---:|---:|
|滑块|1.5708|0|0.98|3|未记录|
|端挡|0|0.001|0.98|3|0.006|
|左销|1.5708|-0.001|0.98|3|0.006|
|右销|1.5708|0.001|0.98|3|0.006|
|手柄|0|0|0.98|3|0.006|

初始轨迹 `transfer`：3 个路点 × 7 关节；世界坐标目标 (m)：[-0.25572499998497666,-0.2036632999872876,0.9329829565938731]；几何 SHA256：`6f0ee9d1a4f5c4de9a333f7ac7c55011cd06078e0637c67aef7ac29a82624fa7`。

|策略|重复编号|成功|超时|已执行调用|错误|
|---|---:|---|---|---:|---|
|full|0|否|否|58|press_seat: shoulder contact/height not verified {'height_error_m': 0.04543036909993625, 'contact_force_n': 1.306700720407705}|
|full|1|否|否|58|press_seat: shoulder contact/height not verified {'height_error_m': 0.04542290893717027, 'contact_force_n': 1.2710229823284596}|

### 候选 8：`902986fb91933d9389ed`

初始路线索引：0；已知/延后参数计数：{"known":206,"deferred":15}。

|部件|yaw (rad)|height (m)|clearance (m)|force (N)|speed (m/s)|
|---|---:|---:|---:|---:|---:|
|滑块|0|-0.001|0.98|3|未记录|
|端挡|1.5708|-0.001|0.98|3|0.007|
|左销|1.5708|0|0.98|3|0.007|
|右销|1.5708|0.001|0.98|3|0.007|
|手柄|0|0.001|0.98|3|0.007|

初始轨迹 `transfer`：3 个路点 × 7 关节；世界坐标目标 (m)：[-0.25572499998497666,-0.2036632999872876,0.9319829565938731]；几何 SHA256：`67fa61475887597dee983ec6769eaee29fb9b575f43792d36702b854d33fbdcc`。

|策略|重复编号|成功|超时|已执行调用|错误|
|---|---:|---|---|---:|---|
|top_k|0|是|否|106|无|
|top_k|1|是|否|106|无|
|full|0|是|否|106|无|
|full|1|是|否|106|无|

### 候选 9：`2ca70bd6f80999c6bf94`

初始路线索引：2；已知/延后参数计数：{"known":206,"deferred":15}。

|部件|yaw (rad)|height (m)|clearance (m)|force (N)|speed (m/s)|
|---|---:|---:|---:|---:|---:|
|滑块|1.5708|0|0.98|2.8|未记录|
|端挡|1.5708|-0.001|0.98|2.8|0.006|
|右销|1.5708|0|0.98|2.8|0.006|
|左销|1.5708|0.001|0.98|2.8|0.006|
|手柄|1.5708|-0.001|0.98|2.8|0.006|

初始轨迹 `transfer`：3 个路点 × 7 关节；世界坐标目标 (m)：[-0.25572499998497666,-0.2036632999872876,0.9329829565938731]；几何 SHA256：`6f0ee9d1a4f5c4de9a333f7ac7c55011cd06078e0637c67aef7ac29a82624fa7`。

|策略|重复编号|成功|超时|已执行调用|错误|
|---|---:|---|---|---:|---|
|top_k|0|是|否|106|无|
|top_k|1|是|否|106|无|
|full|0|是|否|106|无|
|full|1|是|否|106|无|

### 候选 10：`d19463dd8bcef0848ad2`

初始路线索引：1；已知/延后参数计数：{"known":206,"deferred":15}。

|部件|yaw (rad)|height (m)|clearance (m)|force (N)|speed (m/s)|
|---|---:|---:|---:|---:|---:|
|滑块|1.5708|-0.001|0.98|3.2|未记录|
|端挡|1.5708|-0.001|0.98|3.2|0.007|
|左销|1.5708|0|0.98|3.2|0.007|
|右销|1.5708|0|0.98|3.2|0.007|
|手柄|1.5708|0|0.98|3.2|0.007|

初始轨迹 `transfer`：3 个路点 × 7 关节；世界坐标目标 (m)：[-0.25572499998497666,-0.2036632999872876,0.9319829565938731]；几何 SHA256：`0d585b463985bdc6648f60870c186a6db8fc2538cb2cda2201821caa8c37cea9`。

|策略|重复编号|成功|超时|已执行调用|错误|
|---|---:|---|---|---:|---|
|full|0|是|否|106|无|
|full|1|是|否|106|无|

### 候选 11：`f28a041b35ffd2a07545`

初始路线索引：1；已知/延后参数计数：{"known":206,"deferred":15}。

|部件|yaw (rad)|height (m)|clearance (m)|force (N)|speed (m/s)|
|---|---:|---:|---:|---:|---:|
|滑块|1.5708|0|0.98|3.2|未记录|
|端挡|1.5708|-0.001|0.98|3.2|0.007|
|左销|1.5708|0.001|0.98|3.2|0.007|
|右销|1.5708|0.001|0.98|3.2|0.007|
|手柄|1.5708|-0.001|0.98|3.2|0.007|

初始轨迹 `transfer`：3 个路点 × 7 关节；世界坐标目标 (m)：[-0.25572499998497666,-0.2036632999872876,0.9329829565938731]；几何 SHA256：`6f0ee9d1a4f5c4de9a333f7ac7c55011cd06078e0637c67aef7ac29a82624fa7`。

|策略|重复编号|成功|超时|已执行调用|错误|
|---|---:|---|---|---:|---|
|full|0|是|否|106|无|
|full|1|是|否|106|无|

### 候选 12：`77cce49a6c674bd1848c`

初始路线索引：2；已知/延后参数计数：{"known":206,"deferred":15}。

|部件|yaw (rad)|height (m)|clearance (m)|force (N)|speed (m/s)|
|---|---:|---:|---:|---:|---:|
|滑块|1.5708|0.001|0.98|2.8|未记录|
|端挡|0|0|0.98|2.8|0.007|
|右销|1.5708|0|0.98|2.8|0.007|
|左销|1.5708|-0.001|0.98|2.8|0.007|
|手柄|1.5708|0|0.98|2.8|0.007|

初始轨迹 `transfer`：3 个路点 × 7 关节；世界坐标目标 (m)：[-0.25572499998497666,-0.2036632999872876,0.9339829565938731]；几何 SHA256：`fe8dfd6760670285832ae01ffe55e2f4e4f9c314ed4d00dceca51a62051a7378`。

|策略|重复编号|成功|超时|已执行调用|错误|
|---|---:|---|---|---:|---|
|full|0|否|否|77|press_seat: shoulder contact/height not verified {'height_error_m': 0.0466794391714096, 'contact_force_n': 1.26061185334067}|
|full|1|否|否|77|press_seat: shoulder contact/height not verified {'height_error_m': 0.046653929951261386, 'contact_force_n': 1.256695560318903}|

## 3. 孪生选择与独立目标执行

先验证策略提交的候选，再按观测成功比例从高到低选择；并列时按原始候选输入顺序。至少一次有效、未超时成功，且比例达到记录中的 accept_rate 才能接受。超时保留在分母中。这里仅展示系统已经作出的选择。

### 策略 `top_k`

状态：`executed_independent_target`。

请求 N=12，实际 N=12，K=4；真实孪生调用 8 次；接受比例阈值 0.5。

选中候选：`0bfa0ac76255dd74db2e`；独立目标成功 1/1（请求目标试验为分母，包含设置/重绑定失败与未执行）。

#### 目标重复 0：executed

实际成功：是；错误：无。

独立实例检查：{"distinct_session":true,"distinct_context":true,"distinct_model":true,"distinct_data":true,"disjoint_buffers":["data.qpos","data.qvel","data.ctrl","model.geom_friction","model.actuator_gainprm","model.actuator_biasprm"],"snapshot_transfer":false,"target_initialization":"separate_scene_factory_call"}。

实际参数扰动：{"domain":"v5_target","namespace":7901,"repeat":0,"friction_scale":1.0079668374044344,"actuator_gain_scale":1.0007319775179955}。目标扰动值在选择后采样，没有输入价值模型。

所选 ID：`0bfa0ac76255dd74db2e`；目标重绑定 ID：`0bfa0ac76255dd74db2e`；语义 SHA256：`4c037d408defc2ae248a24746f471e34007162946f88f130e38b732eb58d2dcd`。

执行 106 条调用，物理步 104090，仿真时间 208.18 s；超时：否。

完整任务独立验收：是。

|部件|通过|位置误差 (m)|倾斜 (deg)|已释放|触碰手指|末端净空 (m)|
|---|---|---:|---:|---|---|---:|
|滑块|是|4.13611e-05|1.6109e-05|是|否|0.450378|
|端挡|是|0.000232676|0.000103813|是|否|0.438356|
|左销|是|0.000146165|0.00310006|是|否|0.419361|
|右销|是|0.00011426|0.00376975|是|否|0.419361|
|手柄|是|5.81848e-05|3.32205e-05|是|否|0.402379|

<details>
<summary>逐条实际执行调用（参数与完整指标保留在 walkthrough.json）</summary>

|执行序号|实际技能|成功|仿真秒|指标|
|---:|---|---|---:|---|
|1|observe_parts|是|0|{"detected":["carriage","end_stop","handle","pin_left","pin_right"],"backend":"simulator pose observations","noise_std_m":0.0}|
|2|estimate_pose|是|0|{"position_m":[-0.25572499998497666,-0.2036632999872876,0.8059829565938731],"quaternion_wxyz":[1.0,1.853787810586852e-17,-7.802413561649229e-19,-5.861269240787358e-12]}|
|3|propose_grasps|是|0|{"feasible_candidates":1,"unknown_candidates":0}|
|4|select_grasp|是|0|{"chosen_index":0,"yaw_rad":0.0,"jaw_width_m":0.028}|
|5|execute_joint_path|是|6.46|{"target_error_m":6.148725198088871e-05}|
|6|approach|是|4.72|{"position_error_m":3.6262186979362074e-05}|
|7|close_gripper|是|4.04|{"held":true,"left_n":3.4853184547037257,"right_n":3.485153833919606}|
|8|verify_grasp|是|0|{"held":true,"left_n":3.4853184547037257,"right_n":3.485153833919606}|
|9|lift|是|3.66|{"object_lift_m":0.09979612291549167,"held":true}|
|10|plan_transfer|是|0|{"waypoints":3,"clearance_m":0.98,"collision_checks":[{"route_id":"carriage:yaw:0.000000:pose:2ecec72bad12:route:0","valid":true,"samples":31,"resolution_rad":0.035},{"route_id":"carriage:yaw:0.000000:pose:2ecec72bad12:route:1","valid":true,"samples":42,"resolution_rad":0.035},{"route_id":"carriage:yaw:0.000000:pose:2ecec72bad12:route:2","valid":true,"samples":52,"resolution_rad":0.035}],"candidate_count":3,"selected_id":"carriage:yaw:0.000000:pose:2ecec72bad12:route:0"}|
|11|execute_joint_path|是|5.2|{"target_error_m":0.00017913980601561008}|
|12|lower|是|2.12|{"distance_m":0.03}|
|13|align_axis|是|1|{"object_error_m":0.00015481133856011484}|
|14|plan_insertion|是|0|{"target_m":[0.105,0.085,0.8254999999999999],"axis":[1.0,0.0,0.0],"speed_m_s":0.025,"force_limit_n":18.0}|
|15|slide_insert|是|14.4|{"error_m":0.0008273345613889823,"travel_m":0.18488452022774646,"peak_force_n":0.0}|
|16|press_seat|是|0.8|{"height_error_m":8.693014448446501e-05,"contact_force_n":2.6386873885630626}|
|17|open_gripper|是|1|{"jaw_span_m":0.08476185527957439}|
|18|place_object|是|1.34|{"position_error_m":4.601296120799459e-05,"tilt_deg":0.0,"support_force_n":2.6386873885630626,"jaw_span_m":0.08535007386000498}|
|19|retreat|是|2.38|{"height_m":0.9515262981842846}|
|20|inspect_seat|是|0|{"position_error_m":4.601296032653013e-05,"tilt_deg":0.0}|
|21|measure_value|是|0|{"quantity":"clearance","value":0.0039948726998839355,"unit":"m"}|
|22|inspect_measurement|是|0|{"value":0.0039948726998839355,"unit":"m","minimum":1e-12,"maximum":null}|
|23|observe_parts|是|0|{"detected":["carriage","end_stop","handle","pin_left","pin_right"],"backend":"simulator pose observations","noise_std_m":0.0}|
|24|estimate_pose|是|0|{"position_m":[-0.1158292,-0.2511363,0.8179991724371571],"quaternion_wxyz":[1.0,9.080992117995346e-17,-5.644285468869596e-19,-1.5467289191592357e-28]}|
|25|propose_grasps|是|0|{"feasible_candidates":1,"unknown_candidates":0}|
|26|select_grasp|是|0|{"chosen_index":0,"yaw_rad":1.5707963267948966,"jaw_width_m":0.022}|
|27|plan_transfer|是|0|{"waypoints":3,"clearance_m":0.98,"collision_checks":[{"route_id":"end_stop:yaw:1.570796:pose:582a81d4adb7:route:0","valid":true,"samples":66,"resolution_rad":0.035},{"route_id":"end_stop:yaw:1.570796:pose:582a81d4adb7:route:1","valid":true,"samples":71,"resolution_rad":0.035},{"route_id":"end_stop:yaw:1.570796:pose:582a81d4adb7:route:2","valid":true,"samples":75,"resolution_rad":0.035}],"candidate_count":3,"selected_id":"end_stop:yaw:1.570796:pose:582a81d4adb7:route:0"}|
|28|execute_joint_path|是|8.2|{"target_error_m":4.821834738361297e-06}|
|29|approach|是|4.72|{"position_error_m":3.534636716993767e-05}|
|30|close_gripper|是|4.3|{"held":true,"left_n":3.470319739738081,"right_n":3.633832945421097}|
|31|verify_grasp|是|0|{"held":true,"left_n":3.470319739738081,"right_n":3.633832945421097}|
|32|lift|是|3.66|{"object_lift_m":0.09967653927982445,"held":true}|
|33|plan_transfer|是|0|{"waypoints":3,"clearance_m":0.98,"collision_checks":[{"route_id":"end_stop:yaw:1.570796:pose:582a81d4adb7:route:0","valid":true,"samples":30,"resolution_rad":0.035},{"route_id":"end_stop:yaw:1.570796:pose:582a81d4adb7:route:1","valid":true,"samples":39,"resolution_rad":0.035},{"route_id":"end_stop:yaw:1.570796:pose:582a81d4adb7:route:2","valid":true,"samples":47,"resolution_rad":0.035}],"candidate_count":3,"selected_id":"end_stop:yaw:1.570796:pose:582a81d4adb7:route:0"}|
|34|execute_joint_path|是|5.62|{"target_error_m":9.096699723346191e-05}|
|35|plan_linear|是|0|{"length_m":0.02632307882262229,"ik_samples":12}|
|36|execute_cartesian_path|是|1.5|{"target_error_m":6.384815052682294e-05}|
|37|align_axis|是|0|{"object_error_m":0.00016785008758968354}|
|38|guarded_descent|是|1.8|{"z_m":0.8361783846203926,"peak_contact_n":0.0,"travel_m":0.009740059198354523,"stopped_by":"height"}|
|39|press_seat|是|0.2|{"height_error_m":6.612827171770252e-05,"contact_force_n":2.5596913318972536}|
|40|open_gripper|是|1|{"jaw_span_m":0.0847446503327485}|
|41|place_object|是|1.34|{"position_error_m":0.0002368960072410355,"tilt_deg":0.00014098127185602726,"support_force_n":2.2314425010919225,"jaw_span_m":0.08534142624747959}|
|42|retreat|是|2.38|{"height_m":0.9668296743197187}|
|43|inspect_seat|是|0|{"position_error_m":0.00023689601096756518,"tilt_deg":0.00014098902724134303}|
|44|observe_parts|是|0|{"detected":["carriage","end_stop","handle","pin_left","pin_right"],"backend":"simulator pose observations","noise_std_m":0.0}|
|45|estimate_pose|是|0|{"position_m":[-0.35945838511931044,-0.22037007215372847,0.8479659616768326],"quaternion_wxyz":[0.9999997986328425,0.00030126940950069784,-0.00040726105636993293,-0.0003822426580707792]}|
|46|propose_grasps|是|0|{"feasible_candidates":1,"unknown_candidates":0}|
|47|select_grasp|是|0|{"chosen_index":0,"yaw_rad":1.5707963267948966,"jaw_width_m":0.018}|
|48|plan_transfer|是|0|{"waypoints":3,"clearance_m":0.98,"collision_checks":[{"route_id":"pin_left:yaw:1.570796:pose:7db79ea05412:route:0","valid":true,"samples":33,"resolution_rad":0.035},{"route_id":"pin_left:yaw:1.570796:pose:7db79ea05412:route:1","valid":true,"samples":43,"resolution_rad":0.035},{"route_id":"pin_left:yaw:1.570796:pose:7db79ea05412:route:2","valid":true,"samples":53,"resolution_rad":0.035}],"candidate_count":3,"selected_id":"pin_left:yaw:1.570796:pose:7db79ea05412:route:0"}|
|49|execute_joint_path|是|6.1|{"target_error_m":4.64710557864046e-06}|
|50|approach|是|4.72|{"position_error_m":5.5259248492407024e-05}|
|51|close_gripper|是|4.84|{"held":true,"left_n":4.055433601909784,"right_n":4.202828605806714}|
|52|verify_grasp|是|0|{"held":true,"left_n":4.055433601909784,"right_n":4.202828605806714}|
|53|lift|是|3.66|{"object_lift_m":0.0996705832108703,"held":true}|
|54|plan_transfer|是|0|{"waypoints":3,"clearance_m":0.98,"collision_checks":[{"route_id":"pin_left:yaw:1.570796:pose:7db79ea05412:route:0","valid":true,"samples":33,"resolution_rad":0.035},{"route_id":"pin_left:yaw:1.570796:pose:7db79ea05412:route:1","valid":true,"samples":44,"resolution_rad":0.035},{"route_id":"pin_left:yaw:1.570796:pose:7db79ea05412:route:2","valid":true,"samples":54,"resolution_rad":0.035}],"candidate_count":3,"selected_id":"pin_left:yaw:1.570796:pose:7db79ea05412:route:0"}|
|55|execute_joint_path|是|5.94|{"target_error_m":2.5340922903438738e-05}|
|56|align_axis|是|1|{"object_error_m":8.854111312600698e-05}|
|57|guarded_descent|是|11.52|{"z_m":0.8551849520034253,"peak_contact_n":0.0,"travel_m":0.06873136280723813,"stopped_by":"height"}|
|58|press_seat|是|2.8|{"height_error_m":9.089450773680507e-05,"contact_force_n":1.5653941261679125}|
|59|open_gripper|是|1|{"jaw_span_m":0.0847092121020399}|
|60|place_object|是|1.34|{"position_error_m":0.00014593746744628682,"tilt_deg":0.003018546640597884,"support_force_n":1.5485309458116399,"jaw_span_m":0.08532518983487498}|
|61|retreat|是|2.38|{"height_m":0.9599281944537097}|
|62|inspect_seat|是|0|{"position_error_m":0.00014591042654424553,"tilt_deg":0.002985853956973213}|
|63|observe_parts|是|0|{"detected":["carriage","end_stop","handle","pin_left","pin_right"],"backend":"simulator pose observations","noise_std_m":0.0}|
|64|estimate_pose|是|0|{"position_m":[-0.36509916087049804,-0.33869476926859315,0.8479683066344419],"quaternion_wxyz":[0.9999997626191228,-0.0001886043394026373,0.0006601593806395788,5.8135130042322076e-05]}|
|65|propose_grasps|是|0|{"feasible_candidates":1,"unknown_candidates":0}|
|66|select_grasp|是|0|{"chosen_index":0,"yaw_rad":1.5707963267948966,"jaw_width_m":0.018}|
|67|plan_transfer|是|0|{"waypoints":3,"clearance_m":0.98,"collision_checks":[{"route_id":"pin_right:yaw:1.570796:pose:2d4d49473528:route:0","valid":true,"samples":37,"resolution_rad":0.035},{"route_id":"pin_right:yaw:1.570796:pose:2d4d49473528:route:1","valid":true,"samples":46,"resolution_rad":0.035},{"route_id":"pin_right:yaw:1.570796:pose:2d4d49473528:route:2","valid":true,"samples":54,"resolution_rad":0.035}],"candidate_count":3,"selected_id":"pin_right:yaw:1.570796:pose:2d4d49473528:route:0"}|
|68|execute_joint_path|是|6.58|{"target_error_m":4.888223548686608e-06}|
|69|approach|是|4.72|{"position_error_m":3.583461618818946e-05}|
|70|close_gripper|是|4.82|{"held":true,"left_n":4.017779656645222,"right_n":4.0176835770851405}|
|71|verify_grasp|是|0|{"held":true,"left_n":4.017779656645222,"right_n":4.0176835770851405}|
|72|lift|是|3.66|{"object_lift_m":0.09967250865003041,"held":true}|
|73|plan_transfer|是|0|{"waypoints":3,"clearance_m":0.98,"collision_checks":[{"route_id":"pin_right:yaw:1.570796:pose:2d4d49473528:route:0","valid":true,"samples":43,"resolution_rad":0.035},{"route_id":"pin_right:yaw:1.570796:pose:2d4d49473528:route:1","valid":true,"samples":52,"resolution_rad":0.035},{"route_id":"pin_right:yaw:1.570796:pose:2d4d49473528:route:2","valid":true,"samples":60,"resolution_rad":0.035}],"candidate_count":3,"selected_id":"pin_right:yaw:1.570796:pose:2d4d49473528:route:0"}|
|74|execute_joint_path|是|6.94|{"target_error_m":2.8544563098466923e-05}|
|75|align_axis|是|1|{"object_error_m":8.629722329901278e-05}|
|76|guarded_descent|是|11.52|{"z_m":0.8551714804452912,"peak_contact_n":0.0,"travel_m":0.06874387922184688,"stopped_by":"height"}|
|77|press_seat|是|2.8|{"height_error_m":8.963164520059408e-05,"contact_force_n":1.5336115308028597}|
|78|open_gripper|是|1|{"jaw_span_m":0.084710517159676}|
|79|place_object|是|1.34|{"position_error_m":0.00011430612822852717,"tilt_deg":0.0037331809907561616,"support_force_n":1.0235398026022755,"jaw_span_m":0.08532575360749861}|
|80|retreat|是|2.38|{"height_m":0.9592503447035415}|
|81|inspect_seat|是|0|{"position_error_m":0.0001143059481385892,"tilt_deg":0.003686844520229548}|
|82|observe_parts|是|0|{"detected":["carriage","end_stop","handle","pin_left","pin_right"],"backend":"simulator pose observations","noise_std_m":0.0}|
|83|estimate_pose|是|0|{"position_m":[0.010171799965279392,-0.25599289989216795,0.8079991478511446],"quaternion_wxyz":[1.0,-6.716286880157963e-09,-2.1822531328460097e-09,4.085706350860226e-15]}|
|84|propose_grasps|是|0|{"feasible_candidates":1,"unknown_candidates":0}|
|85|select_grasp|是|0|{"chosen_index":0,"yaw_rad":0.0,"jaw_width_m":0.042}|
|86|plan_transfer|是|0|{"waypoints":3,"clearance_m":0.98,"collision_checks":[{"route_id":"handle:yaw:0.000000:pose:e5cea90dbbbf:route:0","valid":true,"samples":67,"resolution_rad":0.035},{"route_id":"handle:yaw:0.000000:pose:e5cea90dbbbf:route:1","valid":true,"samples":72,"resolution_rad":0.035},{"route_id":"handle:yaw:0.000000:pose:e5cea90dbbbf:route:2","valid":true,"samples":74,"resolution_rad":0.035}],"candidate_count":3,"selected_id":"handle:yaw:0.000000:pose:e5cea90dbbbf:route:0"}|
|87|execute_joint_path|是|8.28|{"target_error_m":2.493788917088902e-06}|
|88|approach|是|4.72|{"position_error_m":3.484415998824313e-05}|
|89|close_gripper|是|3.18|{"held":true,"left_n":3.5686547280679877,"right_n":3.625718517426804}|
|90|verify_grasp|是|0|{"held":true,"left_n":3.5686547280679877,"right_n":3.625718517426804}|
|91|lift|是|3.66|{"object_lift_m":0.09982058871229893,"held":true}|
|92|plan_transfer|是|0|{"waypoints":3,"clearance_m":0.98,"collision_checks":[{"route_id":"handle:yaw:0.000000:pose:e5cea90dbbbf:route:0","valid":true,"samples":26,"resolution_rad":0.035},{"route_id":"handle:yaw:0.000000:pose:e5cea90dbbbf:route:1","valid":true,"samples":35,"resolution_rad":0.035},{"route_id":"handle:yaw:0.000000:pose:e5cea90dbbbf:route:2","valid":true,"samples":41,"resolution_rad":0.035}],"candidate_count":3,"selected_id":"handle:yaw:0.000000:pose:e5cea90dbbbf:route:0"}|
|93|execute_joint_path|是|5.3|{"target_error_m":5.292983950033166e-05}|
|94|align_axis|是|1.68|{"object_error_m":6.438251348696391e-05}|
|95|guarded_descent|是|3.98|{"z_m":0.8721865308610344,"peak_contact_n":0.0,"travel_m":0.022751082179581328,"stopped_by":"height"}|
|96|press_seat|是|0.42|{"height_error_m":9.509161840792757e-05,"contact_force_n":2.066395092284337}|
|97|open_gripper|是|1|{"jaw_span_m":0.08481859688298436}|
|98|place_object|是|1.34|{"position_error_m":5.818483071991298e-05,"tilt_deg":3.322046313188779e-05,"support_force_n":2.0663931754682214,"jaw_span_m":0.08538335624934315}|
|99|retreat|是|2.38|{"height_m":0.9719380527773821}|
|100|inspect_seat|是|0|{"position_error_m":5.8184832888730075e-05,"tilt_deg":3.3242398071968026e-05}|
|101|inspect_seat|是|0|{"position_error_m":4.136111561973082e-05,"tilt_deg":1.6108968939569464e-05}|
|102|inspect_seat|是|0|{"position_error_m":0.00023267598546206794,"tilt_deg":0.00010381690943943502}|
|103|inspect_seat|是|0|{"position_error_m":0.00014616736337714112,"tilt_deg":0.0031348895346205163}|
|104|inspect_seat|是|0|{"position_error_m":0.00011426263693778932,"tilt_deg":0.0038725737384261425}|
|105|inspect_seat|是|0|{"position_error_m":5.8184832888730075e-05,"tilt_deg":3.3242398071968026e-05}|
|106|home|是|3.34|{"joint_error_rad":1.3708518320498797e-08}|

</details>

记录决策时间：400.326 s；场景创建至独立执行和终态渲染：452.979 s（不含离线 LLM 代理生成与进程启动，模型加载另计）。

[实际目标终态图像](<top_k_target_terminal_0.png>)：来自当次执行的终态数据，无重放；图像 SHA256 `5293c11ff0c104e2277ac1421153a782cc5a72052042ab8f14fecc5a26981900`。

### 策略 `full`

状态：`executed_independent_target`。

请求 N=12，实际 N=12，K=12；真实孪生调用 24 次；接受比例阈值 0.5。

选中候选：`0bfa0ac76255dd74db2e`；独立目标成功 1/1（请求目标试验为分母，包含设置/重绑定失败与未执行）。

#### 目标重复 0：executed

实际成功：是；错误：无。

独立实例检查：{"distinct_session":true,"distinct_context":true,"distinct_model":true,"distinct_data":true,"disjoint_buffers":["data.qpos","data.qvel","data.ctrl","model.geom_friction","model.actuator_gainprm","model.actuator_biasprm"],"snapshot_transfer":false,"target_initialization":"separate_scene_factory_call"}。

实际参数扰动：{"domain":"v5_target","namespace":7901,"repeat":0,"friction_scale":1.0079668374044344,"actuator_gain_scale":1.0007319775179955}。目标扰动值在选择后采样，没有输入价值模型。

所选 ID：`0bfa0ac76255dd74db2e`；目标重绑定 ID：`0bfa0ac76255dd74db2e`；语义 SHA256：`4c037d408defc2ae248a24746f471e34007162946f88f130e38b732eb58d2dcd`。

执行 106 条调用，物理步 104090，仿真时间 208.18 s；超时：否。

完整任务独立验收：是。

|部件|通过|位置误差 (m)|倾斜 (deg)|已释放|触碰手指|末端净空 (m)|
|---|---|---:|---:|---|---|---:|
|滑块|是|4.13611e-05|1.6109e-05|是|否|0.450378|
|端挡|是|0.000232676|0.000103813|是|否|0.438356|
|左销|是|0.000146165|0.00310006|是|否|0.419361|
|右销|是|0.00011426|0.00376975|是|否|0.419361|
|手柄|是|5.81848e-05|3.32205e-05|是|否|0.402379|

<details>
<summary>逐条实际执行调用（参数与完整指标保留在 walkthrough.json）</summary>

|执行序号|实际技能|成功|仿真秒|指标|
|---:|---|---|---:|---|
|1|observe_parts|是|0|{"detected":["carriage","end_stop","handle","pin_left","pin_right"],"backend":"simulator pose observations","noise_std_m":0.0}|
|2|estimate_pose|是|0|{"position_m":[-0.25572499998497666,-0.2036632999872876,0.8059829565938731],"quaternion_wxyz":[1.0,1.853787810586852e-17,-7.802413561649229e-19,-5.861269240787358e-12]}|
|3|propose_grasps|是|0|{"feasible_candidates":1,"unknown_candidates":0}|
|4|select_grasp|是|0|{"chosen_index":0,"yaw_rad":0.0,"jaw_width_m":0.028}|
|5|execute_joint_path|是|6.46|{"target_error_m":6.148725198088871e-05}|
|6|approach|是|4.72|{"position_error_m":3.6262186979362074e-05}|
|7|close_gripper|是|4.04|{"held":true,"left_n":3.4853184547037257,"right_n":3.485153833919606}|
|8|verify_grasp|是|0|{"held":true,"left_n":3.4853184547037257,"right_n":3.485153833919606}|
|9|lift|是|3.66|{"object_lift_m":0.09979612291549167,"held":true}|
|10|plan_transfer|是|0|{"waypoints":3,"clearance_m":0.98,"collision_checks":[{"route_id":"carriage:yaw:0.000000:pose:2ecec72bad12:route:0","valid":true,"samples":31,"resolution_rad":0.035},{"route_id":"carriage:yaw:0.000000:pose:2ecec72bad12:route:1","valid":true,"samples":42,"resolution_rad":0.035},{"route_id":"carriage:yaw:0.000000:pose:2ecec72bad12:route:2","valid":true,"samples":52,"resolution_rad":0.035}],"candidate_count":3,"selected_id":"carriage:yaw:0.000000:pose:2ecec72bad12:route:0"}|
|11|execute_joint_path|是|5.2|{"target_error_m":0.00017913980601561008}|
|12|lower|是|2.12|{"distance_m":0.03}|
|13|align_axis|是|1|{"object_error_m":0.00015481133856011484}|
|14|plan_insertion|是|0|{"target_m":[0.105,0.085,0.8254999999999999],"axis":[1.0,0.0,0.0],"speed_m_s":0.025,"force_limit_n":18.0}|
|15|slide_insert|是|14.4|{"error_m":0.0008273345613889823,"travel_m":0.18488452022774646,"peak_force_n":0.0}|
|16|press_seat|是|0.8|{"height_error_m":8.693014448446501e-05,"contact_force_n":2.6386873885630626}|
|17|open_gripper|是|1|{"jaw_span_m":0.08476185527957439}|
|18|place_object|是|1.34|{"position_error_m":4.601296120799459e-05,"tilt_deg":0.0,"support_force_n":2.6386873885630626,"jaw_span_m":0.08535007386000498}|
|19|retreat|是|2.38|{"height_m":0.9515262981842846}|
|20|inspect_seat|是|0|{"position_error_m":4.601296032653013e-05,"tilt_deg":0.0}|
|21|measure_value|是|0|{"quantity":"clearance","value":0.0039948726998839355,"unit":"m"}|
|22|inspect_measurement|是|0|{"value":0.0039948726998839355,"unit":"m","minimum":1e-12,"maximum":null}|
|23|observe_parts|是|0|{"detected":["carriage","end_stop","handle","pin_left","pin_right"],"backend":"simulator pose observations","noise_std_m":0.0}|
|24|estimate_pose|是|0|{"position_m":[-0.1158292,-0.2511363,0.8179991724371571],"quaternion_wxyz":[1.0,9.080992117995346e-17,-5.644285468869596e-19,-1.5467289191592357e-28]}|
|25|propose_grasps|是|0|{"feasible_candidates":1,"unknown_candidates":0}|
|26|select_grasp|是|0|{"chosen_index":0,"yaw_rad":1.5707963267948966,"jaw_width_m":0.022}|
|27|plan_transfer|是|0|{"waypoints":3,"clearance_m":0.98,"collision_checks":[{"route_id":"end_stop:yaw:1.570796:pose:582a81d4adb7:route:0","valid":true,"samples":66,"resolution_rad":0.035},{"route_id":"end_stop:yaw:1.570796:pose:582a81d4adb7:route:1","valid":true,"samples":71,"resolution_rad":0.035},{"route_id":"end_stop:yaw:1.570796:pose:582a81d4adb7:route:2","valid":true,"samples":75,"resolution_rad":0.035}],"candidate_count":3,"selected_id":"end_stop:yaw:1.570796:pose:582a81d4adb7:route:0"}|
|28|execute_joint_path|是|8.2|{"target_error_m":4.821834738361297e-06}|
|29|approach|是|4.72|{"position_error_m":3.534636716993767e-05}|
|30|close_gripper|是|4.3|{"held":true,"left_n":3.470319739738081,"right_n":3.633832945421097}|
|31|verify_grasp|是|0|{"held":true,"left_n":3.470319739738081,"right_n":3.633832945421097}|
|32|lift|是|3.66|{"object_lift_m":0.09967653927982445,"held":true}|
|33|plan_transfer|是|0|{"waypoints":3,"clearance_m":0.98,"collision_checks":[{"route_id":"end_stop:yaw:1.570796:pose:582a81d4adb7:route:0","valid":true,"samples":30,"resolution_rad":0.035},{"route_id":"end_stop:yaw:1.570796:pose:582a81d4adb7:route:1","valid":true,"samples":39,"resolution_rad":0.035},{"route_id":"end_stop:yaw:1.570796:pose:582a81d4adb7:route:2","valid":true,"samples":47,"resolution_rad":0.035}],"candidate_count":3,"selected_id":"end_stop:yaw:1.570796:pose:582a81d4adb7:route:0"}|
|34|execute_joint_path|是|5.62|{"target_error_m":9.096699723346191e-05}|
|35|plan_linear|是|0|{"length_m":0.02632307882262229,"ik_samples":12}|
|36|execute_cartesian_path|是|1.5|{"target_error_m":6.384815052682294e-05}|
|37|align_axis|是|0|{"object_error_m":0.00016785008758968354}|
|38|guarded_descent|是|1.8|{"z_m":0.8361783846203926,"peak_contact_n":0.0,"travel_m":0.009740059198354523,"stopped_by":"height"}|
|39|press_seat|是|0.2|{"height_error_m":6.612827171770252e-05,"contact_force_n":2.5596913318972536}|
|40|open_gripper|是|1|{"jaw_span_m":0.0847446503327485}|
|41|place_object|是|1.34|{"position_error_m":0.0002368960072410355,"tilt_deg":0.00014098127185602726,"support_force_n":2.2314425010919225,"jaw_span_m":0.08534142624747959}|
|42|retreat|是|2.38|{"height_m":0.9668296743197187}|
|43|inspect_seat|是|0|{"position_error_m":0.00023689601096756518,"tilt_deg":0.00014098902724134303}|
|44|observe_parts|是|0|{"detected":["carriage","end_stop","handle","pin_left","pin_right"],"backend":"simulator pose observations","noise_std_m":0.0}|
|45|estimate_pose|是|0|{"position_m":[-0.35945838511931044,-0.22037007215372847,0.8479659616768326],"quaternion_wxyz":[0.9999997986328425,0.00030126940950069784,-0.00040726105636993293,-0.0003822426580707792]}|
|46|propose_grasps|是|0|{"feasible_candidates":1,"unknown_candidates":0}|
|47|select_grasp|是|0|{"chosen_index":0,"yaw_rad":1.5707963267948966,"jaw_width_m":0.018}|
|48|plan_transfer|是|0|{"waypoints":3,"clearance_m":0.98,"collision_checks":[{"route_id":"pin_left:yaw:1.570796:pose:7db79ea05412:route:0","valid":true,"samples":33,"resolution_rad":0.035},{"route_id":"pin_left:yaw:1.570796:pose:7db79ea05412:route:1","valid":true,"samples":43,"resolution_rad":0.035},{"route_id":"pin_left:yaw:1.570796:pose:7db79ea05412:route:2","valid":true,"samples":53,"resolution_rad":0.035}],"candidate_count":3,"selected_id":"pin_left:yaw:1.570796:pose:7db79ea05412:route:0"}|
|49|execute_joint_path|是|6.1|{"target_error_m":4.64710557864046e-06}|
|50|approach|是|4.72|{"position_error_m":5.5259248492407024e-05}|
|51|close_gripper|是|4.84|{"held":true,"left_n":4.055433601909784,"right_n":4.202828605806714}|
|52|verify_grasp|是|0|{"held":true,"left_n":4.055433601909784,"right_n":4.202828605806714}|
|53|lift|是|3.66|{"object_lift_m":0.0996705832108703,"held":true}|
|54|plan_transfer|是|0|{"waypoints":3,"clearance_m":0.98,"collision_checks":[{"route_id":"pin_left:yaw:1.570796:pose:7db79ea05412:route:0","valid":true,"samples":33,"resolution_rad":0.035},{"route_id":"pin_left:yaw:1.570796:pose:7db79ea05412:route:1","valid":true,"samples":44,"resolution_rad":0.035},{"route_id":"pin_left:yaw:1.570796:pose:7db79ea05412:route:2","valid":true,"samples":54,"resolution_rad":0.035}],"candidate_count":3,"selected_id":"pin_left:yaw:1.570796:pose:7db79ea05412:route:0"}|
|55|execute_joint_path|是|5.94|{"target_error_m":2.5340922903438738e-05}|
|56|align_axis|是|1|{"object_error_m":8.854111312600698e-05}|
|57|guarded_descent|是|11.52|{"z_m":0.8551849520034253,"peak_contact_n":0.0,"travel_m":0.06873136280723813,"stopped_by":"height"}|
|58|press_seat|是|2.8|{"height_error_m":9.089450773680507e-05,"contact_force_n":1.5653941261679125}|
|59|open_gripper|是|1|{"jaw_span_m":0.0847092121020399}|
|60|place_object|是|1.34|{"position_error_m":0.00014593746744628682,"tilt_deg":0.003018546640597884,"support_force_n":1.5485309458116399,"jaw_span_m":0.08532518983487498}|
|61|retreat|是|2.38|{"height_m":0.9599281944537097}|
|62|inspect_seat|是|0|{"position_error_m":0.00014591042654424553,"tilt_deg":0.002985853956973213}|
|63|observe_parts|是|0|{"detected":["carriage","end_stop","handle","pin_left","pin_right"],"backend":"simulator pose observations","noise_std_m":0.0}|
|64|estimate_pose|是|0|{"position_m":[-0.36509916087049804,-0.33869476926859315,0.8479683066344419],"quaternion_wxyz":[0.9999997626191228,-0.0001886043394026373,0.0006601593806395788,5.8135130042322076e-05]}|
|65|propose_grasps|是|0|{"feasible_candidates":1,"unknown_candidates":0}|
|66|select_grasp|是|0|{"chosen_index":0,"yaw_rad":1.5707963267948966,"jaw_width_m":0.018}|
|67|plan_transfer|是|0|{"waypoints":3,"clearance_m":0.98,"collision_checks":[{"route_id":"pin_right:yaw:1.570796:pose:2d4d49473528:route:0","valid":true,"samples":37,"resolution_rad":0.035},{"route_id":"pin_right:yaw:1.570796:pose:2d4d49473528:route:1","valid":true,"samples":46,"resolution_rad":0.035},{"route_id":"pin_right:yaw:1.570796:pose:2d4d49473528:route:2","valid":true,"samples":54,"resolution_rad":0.035}],"candidate_count":3,"selected_id":"pin_right:yaw:1.570796:pose:2d4d49473528:route:0"}|
|68|execute_joint_path|是|6.58|{"target_error_m":4.888223548686608e-06}|
|69|approach|是|4.72|{"position_error_m":3.583461618818946e-05}|
|70|close_gripper|是|4.82|{"held":true,"left_n":4.017779656645222,"right_n":4.0176835770851405}|
|71|verify_grasp|是|0|{"held":true,"left_n":4.017779656645222,"right_n":4.0176835770851405}|
|72|lift|是|3.66|{"object_lift_m":0.09967250865003041,"held":true}|
|73|plan_transfer|是|0|{"waypoints":3,"clearance_m":0.98,"collision_checks":[{"route_id":"pin_right:yaw:1.570796:pose:2d4d49473528:route:0","valid":true,"samples":43,"resolution_rad":0.035},{"route_id":"pin_right:yaw:1.570796:pose:2d4d49473528:route:1","valid":true,"samples":52,"resolution_rad":0.035},{"route_id":"pin_right:yaw:1.570796:pose:2d4d49473528:route:2","valid":true,"samples":60,"resolution_rad":0.035}],"candidate_count":3,"selected_id":"pin_right:yaw:1.570796:pose:2d4d49473528:route:0"}|
|74|execute_joint_path|是|6.94|{"target_error_m":2.8544563098466923e-05}|
|75|align_axis|是|1|{"object_error_m":8.629722329901278e-05}|
|76|guarded_descent|是|11.52|{"z_m":0.8551714804452912,"peak_contact_n":0.0,"travel_m":0.06874387922184688,"stopped_by":"height"}|
|77|press_seat|是|2.8|{"height_error_m":8.963164520059408e-05,"contact_force_n":1.5336115308028597}|
|78|open_gripper|是|1|{"jaw_span_m":0.084710517159676}|
|79|place_object|是|1.34|{"position_error_m":0.00011430612822852717,"tilt_deg":0.0037331809907561616,"support_force_n":1.0235398026022755,"jaw_span_m":0.08532575360749861}|
|80|retreat|是|2.38|{"height_m":0.9592503447035415}|
|81|inspect_seat|是|0|{"position_error_m":0.0001143059481385892,"tilt_deg":0.003686844520229548}|
|82|observe_parts|是|0|{"detected":["carriage","end_stop","handle","pin_left","pin_right"],"backend":"simulator pose observations","noise_std_m":0.0}|
|83|estimate_pose|是|0|{"position_m":[0.010171799965279392,-0.25599289989216795,0.8079991478511446],"quaternion_wxyz":[1.0,-6.716286880157963e-09,-2.1822531328460097e-09,4.085706350860226e-15]}|
|84|propose_grasps|是|0|{"feasible_candidates":1,"unknown_candidates":0}|
|85|select_grasp|是|0|{"chosen_index":0,"yaw_rad":0.0,"jaw_width_m":0.042}|
|86|plan_transfer|是|0|{"waypoints":3,"clearance_m":0.98,"collision_checks":[{"route_id":"handle:yaw:0.000000:pose:e5cea90dbbbf:route:0","valid":true,"samples":67,"resolution_rad":0.035},{"route_id":"handle:yaw:0.000000:pose:e5cea90dbbbf:route:1","valid":true,"samples":72,"resolution_rad":0.035},{"route_id":"handle:yaw:0.000000:pose:e5cea90dbbbf:route:2","valid":true,"samples":74,"resolution_rad":0.035}],"candidate_count":3,"selected_id":"handle:yaw:0.000000:pose:e5cea90dbbbf:route:0"}|
|87|execute_joint_path|是|8.28|{"target_error_m":2.493788917088902e-06}|
|88|approach|是|4.72|{"position_error_m":3.484415998824313e-05}|
|89|close_gripper|是|3.18|{"held":true,"left_n":3.5686547280679877,"right_n":3.625718517426804}|
|90|verify_grasp|是|0|{"held":true,"left_n":3.5686547280679877,"right_n":3.625718517426804}|
|91|lift|是|3.66|{"object_lift_m":0.09982058871229893,"held":true}|
|92|plan_transfer|是|0|{"waypoints":3,"clearance_m":0.98,"collision_checks":[{"route_id":"handle:yaw:0.000000:pose:e5cea90dbbbf:route:0","valid":true,"samples":26,"resolution_rad":0.035},{"route_id":"handle:yaw:0.000000:pose:e5cea90dbbbf:route:1","valid":true,"samples":35,"resolution_rad":0.035},{"route_id":"handle:yaw:0.000000:pose:e5cea90dbbbf:route:2","valid":true,"samples":41,"resolution_rad":0.035}],"candidate_count":3,"selected_id":"handle:yaw:0.000000:pose:e5cea90dbbbf:route:0"}|
|93|execute_joint_path|是|5.3|{"target_error_m":5.292983950033166e-05}|
|94|align_axis|是|1.68|{"object_error_m":6.438251348696391e-05}|
|95|guarded_descent|是|3.98|{"z_m":0.8721865308610344,"peak_contact_n":0.0,"travel_m":0.022751082179581328,"stopped_by":"height"}|
|96|press_seat|是|0.42|{"height_error_m":9.509161840792757e-05,"contact_force_n":2.066395092284337}|
|97|open_gripper|是|1|{"jaw_span_m":0.08481859688298436}|
|98|place_object|是|1.34|{"position_error_m":5.818483071991298e-05,"tilt_deg":3.322046313188779e-05,"support_force_n":2.0663931754682214,"jaw_span_m":0.08538335624934315}|
|99|retreat|是|2.38|{"height_m":0.9719380527773821}|
|100|inspect_seat|是|0|{"position_error_m":5.8184832888730075e-05,"tilt_deg":3.3242398071968026e-05}|
|101|inspect_seat|是|0|{"position_error_m":4.136111561973082e-05,"tilt_deg":1.6108968939569464e-05}|
|102|inspect_seat|是|0|{"position_error_m":0.00023267598546206794,"tilt_deg":0.00010381690943943502}|
|103|inspect_seat|是|0|{"position_error_m":0.00014616736337714112,"tilt_deg":0.0031348895346205163}|
|104|inspect_seat|是|0|{"position_error_m":0.00011426263693778932,"tilt_deg":0.0038725737384261425}|
|105|inspect_seat|是|0|{"position_error_m":5.8184832888730075e-05,"tilt_deg":3.3242398071968026e-05}|
|106|home|是|3.34|{"joint_error_rad":1.3708518320498797e-08}|

</details>

记录决策时间：836.367 s；场景创建至独立执行和终态渲染：890.618 s（不含离线 LLM 代理生成与进程启动，模型加载另计）。

[实际目标终态图像](<full_target_terminal_0.png>)：来自当次执行的终态数据，无重放；图像 SHA256 `28ec98b1c97e1083d7913ef21c5d1e8b5358eeac12a31f377c07b39448e4da14`。

## 4. 视频与证据边界

[所选 TopK 目标的记录状态重放视频](<../../../demos/value_v5_full_pipeline.mp4>)。它与目标执行文件、扰动试验、状态轨迹和视频 SHA256 绑定，没有新物理积分。

场景阅读说明：供料区两个灰色圆筒是被动定位销支架，装配后仍留在原位，不是漏装的两枚销钉。左右销是否装配通过，应查看独立目标验收中的 pin_left / pin_right 结果。

- 案例由调用者显式指定；所有预注册案例及失败必须保留在总体报告中，不能用本案例估计性能。
- LLM 代理只提供任务/符号顺序；stage_calls 和几何/运动求解器实例化物理程序，不是 LLM 计算关节轨迹。
- 保存的 proposed_orders 被编译器消费；proposed_skill_programs 是保留的规划来源说明，未逐条作为可执行程序解析。
- 初始自由运动轨迹已物化；依赖实际抓取的后续轨迹仍在执行时求解。
- 价值输入使用结构化仿真观测和接口参数，没有相机图像；尚未证明输入表示最优或图表示优于序列。
- 独立目标是另一个 MuJoCo 场景，未验证真实机器人；五部件装配终验不包含滑台功能行程试验。
- 若提供视频，它仅重放所选目标的已记录状态，不增加成功样本，也不是新的物理执行。

完整记录与核验文件 SHA256 见同目录 `walkthrough.json`。已核验：

- top_k: planner hash binds recorded task and proposed_orders
- top_k: exact input hash, original candidate order and consumed proposed_orders
- top_k: saved ranks/probabilities read directly; no model invocation or reranking
- top_k: all executed target identities, semantics, trial namespaces, isolation and trace lengths
- full: planner hash binds recorded task and proposed_orders
- full: exact input hash, original candidate order and consumed proposed_orders
- full: all executed target identities, semantics, trial namespaces, isolation and trace lengths
- both policies: identical saved candidate pool and observation
- video: selected TopK target execution/trial/trace and video hash; no new rollout
