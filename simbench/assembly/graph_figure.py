"""Render the actual shared-contract graph; no hand-written connectivity claims."""

import argparse
import json
from pathlib import Path
import subprocess
from .graph import catalog_graph


def render(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    graph = catalog_graph()
    (out / "skill_graph.json").write_text(
        json.dumps(graph, ensure_ascii=False, indent=2)
    )
    lines = [
        "digraph skills {",
        'graph [rankdir=LR, bgcolor="#f5f7fa", pad="0.45", nodesep="0.48", ranksep="0.85", fontname="Noto Sans CJK SC", fontsize=20, labelloc=t, label="TwinGraph · 10 个原子技能\n条件依赖图：连线表示提供部分输入或状态，不等于整段动作已可行"];',
        'node [shape=box, style="rounded,filled", color="#c1cad5", penwidth=1.5, fontname="Noto Sans CJK SC", fontsize=18, margin="0.20,0.15"];',
        'edge [fontname="Noto Sans CJK SC", fontsize=12, penwidth=1.6, arrowsize=.75, color="#54708c"];',
    ]
    for n in graph["nodes"]:
        fill = "#dceafb" if n["kind"] == "information" else "#e1f0e4"
        lines.append(
            f'{n["name"]} [label="{n["label"]}\n{n["name"]}", fillcolor="{fill}"];'
        )
    labels = {
        "seen": "对象观测",
        "artifact:pose": "物体位姿",
        "artifact:grasp": "抓取参数",
        "artifact:joint_path": "关节路径",
        "artifact:cartesian_path": "笛卡尔路径",
        "artifact:recovery": "恢复参数",
        "artifact:insertion": "插入参数",
        "artifact:wipe_path": "表面路径",
        "held": "同一对象已抓持",
        "empty": "夹爪已释放",
    }
    for edge in graph["edges"]:
        label = " / ".join(labels.get(x, x) for x in edge["supplies"])
        lines.append(f'{edge["source"]} -> {edge["target"]} [label="{label}"];')
    lines.append("}")
    dot = out / "skill_graph.dot"
    dot.write_text("\n".join(lines), encoding="utf-8")
    for fmt in ["svg", "png"]:
        subprocess.run(
            [
                "dot",
                f"-T{fmt}",
                "-Gdpi=150",
                str(dot),
                "-o",
                str(out / f"skill_graph.{fmt}"),
            ],
            check=True,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/skill_graph")
    a = p.parse_args()
    render(a.out)
