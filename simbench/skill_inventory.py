#!/usr/bin/env python3
"""Skill-inventory deliverable: emit the structured atomic-skill list.

    python -m simbench.skill_inventory                 # write md + json
    python -m simbench.skill_inventory --print         # also print to stdout
    python -m simbench.skill_inventory --md PATH --json PATH

Reads the contract of every skill registered in
:class:`simbench.skills.base.SkillRegistry` (populated by importing
``simbench.skills``) and renders the deliverable table:

  技能名称 / 类别 / 功能描述 / 输入 / 输出 / 实现方式 / 依赖项 /
  是否已封装 / 验证方式或成功标准

plus a per-skill detail section (pre/post conditions, failure policy).
"""
import argparse
import json
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import simbench.skills as S            # noqa: E402  (populates REGISTRY)

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MD = os.path.join(_HERE, "docs", "skill_inventory.md")
DEFAULT_JSON = os.path.join(_HERE, "docs", "skill_inventory.json")

CAT_ORDER = ["exec", "plan", "trans", "ext"]


def _clean(s):
    """Flatten a description for a Markdown table cell."""
    return str(s).replace("\n", " ").replace("|", "\\|").strip()


def _keys(d):
    if isinstance(d, dict):
        return ", ".join(f"`{k}`" for k in d)
    return _clean(d)


def _cond(lst):
    """Render a pre/post-condition list (str | callable) as text."""
    out = []
    for c in lst or []:
        if callable(c):
            out.append(f"`{getattr(c, '__name__', 'check')}()`")
        else:
            out.append(_clean(c))
    return "; ".join(out) if out else "—"


def build_rows():
    return S.REGISTRY.inventory()


def to_markdown(rows):
    by_cat = {c: [r for r in rows if r["category"] == c] for c in CAT_ORDER}
    n = len(rows)
    lines = []
    lines.append("# simbench 原子技能清单（Skill Inventory）")
    lines.append("")
    lines.append("> 由 `python -m simbench.skill_inventory` 从 "
                 "`SkillRegistry` 契约自动生成；请勿手工编辑。")
    lines.append("")
    lines.append("分层可组合技能库：**L0** 原语（`MjContext` / "
                 "`CartesianController` / `Gripper`）→ **L1** 原子技能"
                 "（执行/规划/过渡/扩展四类）→ **L2** 契约与门控"
                 "（`SkillSpec` + `SkillRegistry.run`）→ **L3** 组合"
                 "（`run_chain` 与 planner/executor 的 plan→exec 工件握手）。")
    lines.append("")
    lines.append(f"**技能总数：{n}**　"
                 + "　".join(f"{S.CAT_CN[c]}：{len(by_cat[c])}"
                             for c in CAT_ORDER))
    lines.append("")
    # ---- flat summary table
    lines.append("## 汇总表")
    lines.append("")
    lines.append("| 技能名称 | 类别 | 功能描述 | 输入 | 输出 | 实现方式 "
                 "| 依赖项 | 已封装 | 验证方式 / 成功标准 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for c in CAT_ORDER:
        for r in by_cat[c]:
            lines.append(
                f"| `{r['name']}` | {r['category_cn']} "
                f"| {_clean(r['description'])} "
                f"| {_keys(r['inputs'])} | {_keys(r['outputs'])} "
                f"| {r['impl']} | {_clean(', '.join(r['deps']))} "
                f"| {'是' if r['wrapped'] else '否'} "
                f"| {_cond(r['postconditions'])} |")
    lines.append("")
    # ---- per-skill detail
    lines.append("## 技能明细（契约）")
    for c in CAT_ORDER:
        lines.append("")
        lines.append(f"### {S.CAT_CN[c]}（{c}）")
        for r in by_cat[c]:
            lines.append("")
            lines.append(f"#### `{r['name']}`")
            lines.append(f"- **功能描述**：{_clean(r['description'])}")
            lines.append(f"- **实现方式**：{r['impl']}")
            lines.append(f"- **失败策略**：{r['failure_policy']}")
            lines.append(f"- **是否已封装**：{'是' if r['wrapped'] else '否'}")
            lines.append(f"- **依赖项**：{_clean(', '.join(r['deps']))}")
            lines.append("- **输入**：")
            for k, v in (r["inputs"] or {}).items():
                lines.append(f"    - `{k}`：{_clean(v)}")
            lines.append("- **输出**：")
            for k, v in (r["outputs"] or {}).items():
                lines.append(f"    - `{k}`：{_clean(v)}")
            lines.append(f"- **前置条件**：{_cond(r['preconditions'])}")
            lines.append(f"- **后置条件 / 成功标准**："
                         f"{_cond(r['postconditions'])}")
    lines.append("")
    lines.append("## 学习类技能说明（扩展类插销）")
    lines.append("")
    lines.append("`peg_insert` 提供三种 `mode`：`press`（到点压入）、"
                 "`thread`（底部导向螺旋下降 + 抗卡摆动，规则闭环，A/B/C "
                 "生产链使用）、`policy`（学习策略逐步驱动 EEF）。"
                 "`policy` 模式由 `simbench/skills/learned/` 实现：")
    lines.append("")
    lines.append("- **观测**（`insert_env.build_obs`，8 维，场景无关）："
                 "销底相对孔心的横向偏差(2)、销底距座面高度(1)、销轴姿态"
                 "偏差(2)、销与环境接触力/卡滞信号(1)、EEF 相对孔心横向"
                 "偏差(2)。")
    lines.append("- **动作**（`insert_env.action_to_delta`，3 维）：EEF "
                 "xyz 增量（每步 ≤1mm）。")
    lines.append("- **奖励**：基于势函数的下降/对齐整形 − 接触力(卡滞)惩罚 "
                 "− 姿态偏差惩罚 − 步长惩罚 + 落座奖励；显式覆盖"
                 "“接触、卡滞、姿态偏差”反馈。")
    lines.append("- **训练**（`train_insert.py`）：`--algo ppo`（强化学习，"
                 "torch 自包含 PPO+GAE）与 `--algo bc`（模仿学习，克隆脚本"
                 "专家）。产出 `.pt` 检查点，`peg_insert(mode='policy')` "
                 "自动加载 `checkpoints/peg_insert.pt`。")
    lines.append("")
    return "\n".join(lines)


def to_json(rows):
    return json.dumps(rows, ensure_ascii=False, indent=2, default=str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default=DEFAULT_MD)
    ap.add_argument("--json", default=DEFAULT_JSON)
    ap.add_argument("--print", dest="do_print", action="store_true")
    ap.add_argument("--category", default=None,
                    choices=CAT_ORDER + [None])
    args = ap.parse_args()

    rows = build_rows()
    if args.category:
        rows = [r for r in rows if r["category"] == args.category]
    md = to_markdown(rows)
    js = to_json(rows)

    os.makedirs(os.path.dirname(os.path.abspath(args.md)), exist_ok=True)
    with open(args.md, "w") as f:
        f.write(md)
    with open(args.json, "w") as f:
        f.write(js)
    print(f"wrote {args.md} ({len(rows)} skills)")
    print(f"wrote {args.json}")
    counts = {c: len([r for r in rows if r['category'] == c])
              for c in CAT_ORDER}
    print(f"counts by category: {counts}  total={len(rows)}")
    if args.do_print:
        print("\n" + md)


if __name__ == "__main__":
    main()
