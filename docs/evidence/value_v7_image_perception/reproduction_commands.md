# 复现命令

```powershell
cd F:\RAL\twingraph-dev
$env:PYTHONPATH='.'
pytest -q tests/test_v7_image_perception.py tests/test_system_v5.py
python scripts/evaluate_v7_rgbd_perception.py --out results/value_v7_image_perception/perception_audit --seeds-per-level 4
python scripts/run_v7_full_task_smoke.py --seed 1200 --out results/value_v7_image_perception/dev_full_pinforce8_record2 --level L1 --pin-force 8 --timeout 1800
# 完整成功案例录像：每3个控制步记录一帧，降低渲染开销但保持等效时间轴
python scripts/run_v7_full_task_smoke.py --seed 1200 --out results/value_v7_image_perception/dev_full_success_video_record4 --level L1 --pin-force 8 --pin-speed 0.006 --timeout 1500 --video-width 480 --video-height 360 --video-stride 3
```

`run_v7_full_task_smoke.py` 会生成 `full_task.mp4`、`result.json`、真实 steps 和
候选/计划摘要。`video_stride` 只降低录像采样率和等效播放帧率，不改变控制步、观测或
任务标签。录像超时或阶段失败仍保留原始文件，不得删除后重跑挑选结果。
