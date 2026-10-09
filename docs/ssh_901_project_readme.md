# TwinGraph on SSH 901

统一项目目录：`/home/jia/twingraph/`。

```text
code/                     最后使用的代码、冻结技能/价值模块、新真机接口
  results -> ../experiments/final
experiments/final/        原始数据、训练补采、完整测试、新实验及运行元数据
experiments/videos/       2026-10-02/03 演示视频
experiments/historical/   旧实验归档、少量旧 Git 本地状态、逐项删除核验凭据
runtime/twin/             MuJoCo 2.3.7 / torch 2.1.0 的运行环境
maintenance/              整理计划、48 个旧目录删除记录、完整哈希核验结果
```

已删除 48 个旧项目/运行环境目录，原实验记录归档后校验保存。外部 `miniconda3/envs/gbi-audit` 是该孪生 Python 环境的解释器依赖，仍需保留。机器人原有脚本位于 `/home/jia/robot_panda/src/robot/scripts/`，新增入口为 `twingraph_task.py`。

V33 完整实验：50 个新布局、450 个配对流程；价值 Top3 131/150（87.3%）、随机 Top3 71/150（47.3%）、Full8 129/150（86.0%）。两种同池方案成功率只差 2 次，不能据此宣称价值排序统计显著优于 Full8；价值主要减少完整孪生调用和时间。新 V34 名义场景精验保存在 `experiments/final/real_v34_release_20261009/`。真机尚未执行。

摆放坐标采用 Panda 原生基座坐标，桌面 z=0。原导轨夹具实体底面距桌面 6 mm，复现需垫高；直接放桌面需要重新规划与精验。请查阅：

- [全流程与命令](code/docs/real_robot_workflow_v34.md)
- [所有零件坐标、偏航及装配目标](code/docs/real_robot_coordinates_v34.md)
- [可打印摆放图](code/docs/panda_layout_4000.pdf)
- [名义场景配置](code/real_robot/config/nominal_layout_4000.json)
- [待实测 Panda 标定配置](code/real_robot/config/panda_calibration_template.json)

真机入口默认预演。名义场景、未审核路线、未标定 TCP/接触参数和缺少阶段实测验收时保持执行关闭。当前执行程序使用审核后的有限 Cartesian 路线，尚未接入在线视觉识别/接收件自动重定位。

```bash
cd /home/jia/twingraph/code
/home/jia/twingraph/runtime/twin/bin/python -m scripts.verify_901_project --full-archives
python3 /home/jia/robot_panda/src/robot/scripts/twingraph_task.py --help
```
