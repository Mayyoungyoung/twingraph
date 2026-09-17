"""Create the concise v6 accuracy, view-ablation and system comparison report."""
import argparse
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def pct(value):
    return f"{100*value:.2f}%"


def bootstrap_config(rows, fn, samples=5000, seed=621):
    rng = np.random.default_rng(seed)
    observed = float(fn(rows))
    values = []
    for _ in range(samples):
        selected = [rows[i] for i in rng.integers(0, len(rows), len(rows))]
        values.append(float(fn(selected)))
    return dict(value=observed, low=float(np.quantile(values, .025)), high=float(np.quantile(values, .975)),
                unit="physical configuration", samples=samples)


def accuracy_row(test):
    threshold = test["threshold"]
    def acc(rows):
        return np.mean([np.mean((np.asarray(r["scores"]) >= threshold)
                               == (np.asarray(r["success_rates"]) >= .5)) for r in rows])
    def hit(rows):
        return np.mean([r["top4_hit"] for r in rows])
    return dict(accuracy=bootstrap_config(test["rows"], acc),
                top4_hit=bootstrap_config(test["rows"], hit))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation", required=True); p.add_argument("--system", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    evaluation = read(a.evaluation); system = read(a.system)
    selected = evaluation["selected_test"]
    intervals = accuracy_row(selected)
    top = system["summaries"]["top_k"]; full = system["summaries"]["full"]
    reduction = 1 - top["mean_decision_seconds"] / full["mean_decision_seconds"]
    for summary in (top, full):
        summary["target_executed_configurations"] = int(
            summary["target_execution_calls"] // max(1, summary["requested_target_trials"] // summary["requested_configurations"]))
        summary["abstained_configurations"] = int(
            summary["requested_configurations"] - summary["target_executed_configurations"])
    result = dict(schema="twingraph.value_v6.report", selected_model=evaluation["selected"],
        selected_view_mode=evaluation["selected_view_mode"],
        value_metrics={k: selected[k] for k in ("test_configurations", "test_candidates", "threshold",
            "accuracy", "balanced_accuracy", "roc_auc", "brier_to_empirical_rate",
            "top4_hit_rate", "top4_mean_success_rate", "pool_mean_success_rate",
            "random_top4_hit_expectation")},
        configuration_bootstrap_95=intervals,
        view_ablation={mode: dict(validation_brier=evaluation["best_by_view"][mode]["validation_brier"],
            validation_accuracy=evaluation["best_by_view"][mode]["validation_accuracy"],
            test_accuracy=test["accuracy"], test_brier=test["brier_to_empirical_rate"],
            test_top4_hit=test["top4_hit_rate"]) for mode, test in evaluation["held_out_test"].items()},
        system=dict(top_k=top, full=full, decision_time_reduction=reduction,
                    decision_speedup=full["mean_decision_seconds"] / top["mean_decision_seconds"]),
        limitations=["independent target is MuJoCo, not hardware",
                     "classification uses empirical majority over three perturbation draws",
                     "LLM plans are supplied by the recorded assistant proxy",
                     "two configurations had no candidate meeting the twin acceptance rule; their target trials remain failures in the fixed denominator"])
    output = Path(a.out); output.mkdir(parents=True, exist_ok=True)
    write(output / "summary.json", result)
    view_labels = {"none": "无图像", "task": "全局斜俯视", "top": "顶视", "both": "双视角"}
    lines = ["# 鲁棒价值模块 v6 实验报告", "",
        f"按验证集 Brier 分数冻结的模型为 **{evaluation['selected']}**，输入视角为 **{view_labels[evaluation['selected_view_mode']]}**。",
        "", "## 价值模块独立测试", "",
        "| 指标 | 结果 |", "|---|---:|",
        f"| 测试场景 / 候选 | {selected['test_configurations']} / {selected['test_candidates']} |",
        f"| 正确率 | {pct(selected['accuracy'])} |",
        f"| 平衡正确率 | {pct(selected['balanced_accuracy'])} |",
        f"| ROC-AUC | {selected['roc_auc']:.4f} |",
        f"| 对经验成功率 Brier | {selected['brier_to_empirical_rate']:.4f} |",
        f"| Top-4 至少保留一条鲁棒成功候选 | {pct(selected['top4_hit_rate'])} |",
        f"| Top-4 平均经验成功率 | {pct(selected['top4_mean_success_rate'])} |",
        f"| 全候选池平均经验成功率 | {pct(selected['pool_mean_success_rate'])} |",
        "", "正确率配置级 bootstrap 95% 区间："
        f"{pct(intervals['accuracy']['low'])}–{pct(intervals['accuracy']['high'])}。",
        "", "## 视角消融", "", "| 输入 | 验证 Brier | 测试正确率 | 测试 Brier | Top-4 命中 |",
        "|---|---:|---:|---:|---:|"]
    for mode in ("none", "task", "top", "both"):
        row = result["view_ablation"][mode]
        lines.append(f"| {view_labels[mode]} | {row['validation_brier']:.4f} | {pct(row['test_accuracy'])} | {row['test_brier']:.4f} | {pct(row['test_top4_hit'])} |")
    lines += ["", "## 完整系统：有价值模块与全量数字孪生筛选", "",
        "| 策略 | 平均决策耗时 (s) | 平均端到端耗时 (s) | 目标成功率 | 孪生执行次数 | 目标执行调用 | 无候选配置 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| 价值模块 Top-4 | {top['mean_decision_seconds']:.2f} | {top['mean_total_seconds']:.2f} | {pct(top['target_success_rate'])} | {top['twin_validation_calls']} | {top['target_execution_calls']} | {top['abstained_configurations']} |",
        f"| 无价值模块，全量 12 条 | {full['mean_decision_seconds']:.2f} | {full['mean_total_seconds']:.2f} | {pct(full['target_success_rate'])} | {full['twin_validation_calls']} | {full['target_execution_calls']} | {full['abstained_configurations']} |",
        "", f"价值模块使平均决策耗时减少 **{pct(reduction)}**（{result['system']['decision_speedup']:.2f}× 加速）。",
        "", "系统成功率分母为每种策略 12 个配置 × 3 次独立目标执行；失败和缺失不会从分母删除。",
        "", "## 范围限制", "",
        "目标端仍是独立 MuJoCo 仿真而非真实机械臂；鲁棒标签来自三次声明扰动；任务候选顺序由保存的助手代理计划提供。", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")

    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        modes = ["none", "task", "top", "both"]
        axes[0].bar([view_labels[m] for m in modes], [result["view_ablation"][m]["test_accuracy"] for m in modes])
        axes[0].set_ylim(0, 1); axes[0].set_ylabel("Held-out accuracy"); axes[0].tick_params(axis="x", rotation=18)
        axes[1].bar(["Top-4", "Full-12"], [top["mean_decision_seconds"], full["mean_decision_seconds"]])
        axes[1].set_ylabel("Decision time (s)"); axes[1].set_title("Digital-twin screening")
        fig.tight_layout(); fig.savefig(output / "overview.png", dpi=180); plt.close(fig)
    except ImportError:
        pass
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
