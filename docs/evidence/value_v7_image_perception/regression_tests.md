# 回归与实验边界

## 软件/几何单元测试

`pytest -q tests/test_v7_image_perception.py tests/test_system_v5.py` 当前结果为
14 passed。测试包括纯图像检测器签名和移动响应、空白图像 invalid、倾斜但孔内的
几何插入、孔边/低位失败、松爪掉出分阶段结果及端口兼容性。

## 实际仿真轨迹

`dev_exec_recorded3`：RGB-D 观测、擦拭和滑块/挡块真实动作完成，但销在插入孔口时
夹持丢失，失败原因为 `slide_insert: grasp lost`；不能算完整任务成功。

`dev_full_pinforce8`：真实双视角记录，清洁和滑块已通过，随后录像带来的墙钟超时在
挡块搬运后截断；它是有效的超时失败样本。

`dev_full_pinforce8_record2`：使用 pin 8 N、1800 s 墙钟预算的完整轨迹，仍以
`result.json` 为准，未把录像超时改写为任务成功。

`dev_full_success_candidate20`：无录像的真实 MuJoCo 轨迹，seed=1200、L1、同一候选
完成清洁、五零件装配、两销松爪后插入、安装后把手 RGB-D 重定位、双向滑动和最终
释放/回位（`full_success=true`）。`dev_full_success_video_record3` 是同一冻结输入的
带录像复跑；录像会显著降低墙钟速度，若最终超时，仍只作为截断视频失败证据，不能
覆盖上述无录像的完整成功记录。

这些是仿真物理执行，不是单元测试；旧 v6/v7 标签不重命名为新完整任务标签。
