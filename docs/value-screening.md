# 计划价值粗筛：实现与运行

模块位于 `simbench/value/`，读取当前观测和带参数的剩余计划，输出完整可执行的 Top-K PlanIR。筛选后仍须进行预算内的孪生验证；输出不表示全局最优或已经通过物理验证。

**首轮数据和训练已完成**，见 [实际结果与推荐权重](evidence/value/README.md)。当前推荐单头整计划模型，依据验证集选择；下述双头设计同时保留为研究接口。本批所有前缀均成功，尚不能验证前缀失败预测，也未证明双头优于单头或几何规则。

## 设计

| 模块 | 职责 |
|---|---|
| `plan.py` | 统一计划、类型/坐标系/已知与待求解参数、执行边界、执行与评分载荷一致性 |
| `scenarios.py` | 滑台销装配子任务的参数化场景、候选、初态图像 |
| `collect.py` | 完整前后缀执行、配对扰动、输入/标签隔离、断点续采 |
| `encode.py` / `vision.py` | 变长类型 token、冻结 ResNet-18 全景与对象特征 |
| `network.py` / `losses.py` | 共享 Transformer、条件概率监督与可选 Top-K 保留损失 |
| `dataset.py` / `train.py` / `evaluate.py` | 分组切分、GPU 训练、保留集评估与基线 |
| `rank.py` / `decision.py` | Top-K 推理、预算验证及新问题仿真执行 |

设 A 为抓取并抬升前缀成功，B 为随后的搬运、插入、压靠、释放、退出和最终位姿验收成功。模型预测 `p_A` 和 `p_B=P(B|A)`，按 `q=p_A*p_B` 排序。乘法来自条件概率分解，不假设两阶段独立。

两个视图共享编码器权重，前缀视图屏蔽后缀及目标，后续视图读取完整已知计划。分别编码可防止多层注意力通过上下文间接泄漏后续信息。默认 128 维、3 层、4 头，可用 `--width 256` 扩展；变长支持不等于已验证任意长度泛化。

参数名、单位、坐标系、已知/待求解状态及相对生产者位置均参与编码。对象引用解析到当前观测属性；场景、候选、路径 ID、几何总代价和 rollout 结果不作为学习输入。待求解搬运目标在实际抓取后依据测量的手—物关系生成。评分视图与执行载荷在加载时校验一致性。

视觉采用冻结 ImageNet ResNet-18，同一组共享全景与对象裁剪特征。当前位姿、对象裁剪使用**仿真特权感知**。结构化消融需要用 `--no-vision` 另训模型，不能在视觉 checkpoint 上静默丢失输入。

## 数据范围与标签

当前任务是**滑台销装配子任务的参数化家族**。变化供料位置、装配位置、孔间隙、夹具通道方向/宽度/高度与退出距离。最多 8 个候选改变抓取高度、朝向与接近路线。同组使用配对的摩擦/执行器增益扰动，每次恢复完整快照，额外恢复本轮扰动的执行器参数。零件靠真实接触运动，没有执行中的传送或自动焊接。

当前没有把连接器和轴系包装成已经搭好或验证过的场景，也不宣称跨家族、真实机器人泛化。先验证同样抓取成功而后续不同的筛选链路。

| 实际执行 | 前缀标签 | 条件后续标签 |
|---|---:|---:|
| 前缀失败 | 0 | null |
| 前缀成功、后续失败 | 1 | 0 |
| 全部成功 | 1 | 1 |

先保存 `inputs.json`，再执行试验并写 `outcomes.json`；保存输入哈希、源码哈希、初态数值快照、场景 XML、图像与完成标记。程序错误不转成负标签。几何张爪空间代理仅用于基线，不能当作真实结果。

基础损失监督所有前缀试验，仅对前缀成功试验监督条件后续。预热后可加入近优保留损失：`max(0, margin + 第 k 高非近优分数 - 最高近优分数)`。全失败组不参加保留监督。有限重复仅产生经验参考率，不能提供精确概率保证。

## 运行

推荐 Linux、Python 3.10、匹配版本的 PyTorch/CUDA/torchvision，以及 `libegl1`、`libgl1` 和 NVIDIA EGL 驱动。已有兼容 PyTorch 的服务器可保留其配对 torchvision。

已有环境可通过 `bash scripts/train_plan_value.sh` 运行整套流程。用 `VALUE_PYTHON=/path/to/python` 指定解释器，`VALUE_WORKERS` 和 `VALUE_EPOCHS` 调整并行数与训练轮次；默认包含完整采集、视觉缓存、双头/单头训练、Top-K 导出及新问题执行验证。

```bash
python -m pip install -r requirements-value.txt
export MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m simbench.value.collect --out results/value/data --groups 100 --start-seed 1000 --repeats 4 --workers 12
python -m simbench.value.vision --data results/value/data --device cuda
python -m simbench.value.train --data results/value/data --out results/value/dual --device cuda --epochs 60 --k 2
# 独立训练、相同输入的整计划单头对照。
python -m simbench.value.train --data results/value/data --out results/value/direct --device cuda --epochs 60 --k 2 --objective direct

python -m simbench.value.rank --checkpoint results/value/dual/best.pt --input results/value/data/group_001000/inputs.json --k 2 --out results/value/top_k.json
# 新问题，最多两次孪生验证，通过后在当前仿真执行。
python -m simbench.value.decision --checkpoint results/value/dual/best.pt --seed 9000 --k 2 --budget 2 --execute
python -m pytest simbench/tests/test_plan_value.py -q
```

`top_k.json` 的每项包含 ID、分数、两个条件概率、状态和完整计划。只过滤已知 conflict，unknown 保留供后续求解。全部失败时可以显式增加总预算继续扩展；耗尽预算不代表证明任务不可行。

## 评估口径

配置哈希固定约 70%/15%/15% 训练/验证/测试划分；同一初态的兄弟候选不拆分，相同几何目标换种子也不能改变分组。验证集的 Hit@K、Regret@K、经验率 Brier 决定 checkpoint，测试集只在训练结束后评估。

对照包含随机 Top-K（小集合精确枚举期望）、原几何代价、固定前序候选、前缀头消融及独立整计划分类器。全失败组不计入有效近优 Hit。

`selected_reference` 是 Top-K 内经验最优候选的参考率，**不是实测闭环成功率**。`estimated_candidate_rollout_reduction` 是候选验证次数的减少比例，不含训练/采集/生成/推理，不是端到端加速比。新问题 `decision.json` 另保存实际验证次数、耗时和执行结果。

已完成的训练成绩和限制见 `docs/evidence/value/`；早期 pilot 仅证明数据链路。

## 参考

编码结构借鉴 [PIGINet](https://arxiv.org/abs/2211.01576)，监督改为带连续参数的实际物理后续结果。变长批次采用 [PyTorch TransformerEncoder padding mask](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html)。这些设计选择不等于已证明优于对应研究方法。
