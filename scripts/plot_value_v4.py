"""Static scientific figures from completed value-v4 audit evidence.

This script reads saved metrics only. It does not rank candidates, select a
model, resample outcomes, estimate confidence intervals, or run any physics.
Matplotlib is imported only when plotting; the command-line help needs no GUI.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

KINDS = ("compact", "port_mlp", "sequence", "graph")
KIND_LABELS = {"compact": "Compact ports", "port_mlp": "All ports MLP",
               "sequence": "Skill sequence", "graph": "Skill graph"}
COLORS = {"random": "#9AA4B2", "source_order": "#64748B", "shortest_initial": "#C58B3A",
          "compact": "#268B8D", "port_mlp": "#72AB91", "sequence": "#7975B5", "graph": "#3578AE",
          "selected": "#24384E", "exhaustive": "#9B657D"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"expected a finite saved numeric metric, got {value!r}")
    return float(value)


def metric(method, key):
    value = method.get("summary", {}).get(key)
    if not isinstance(value, dict):
        return None
    return number(value.get("mean"))


def model_label(name, frozen, *, selected_marker=True):
    record = next((r for r in frozen.get("models", []) if r["name"] == name), None)
    if record:
        label = f"{KIND_LABELS.get(record['kind'], record['kind'])}, seed {record['seed']}"
    else:
        label = {"source_order": "Source order", "shortest_initial": "Shortest initial path",
                 "exhaustive": "Exhaustive"}.get(name, name)
    return label + (" [selected]" if selected_marker and name == frozen.get("selected") else "")


def reference_signature(method):
    return sorted((str(r["group_id"]), str(r["config_id"]), r.get("n"),
                   r.get("random_quality1"), r.get("random_quality4")) for r in method.get("rows", []))


def ranking_data(audit, frozen):
    methods = audit.get("methods", {})
    if not methods:
        return dict(status="no_reached_ranking_groups", rows=[], counts={},
                    attrition=audit.get("attrition", audit.get("source_metadata", {}).get("attrition")))
    required = ["source_order", "shortest_initial", *[r["name"] for r in frozen["models"]]]
    missing = [name for name in required if name not in methods]
    if missing:
        raise ValueError(f"ranking audit lacks frozen methods: {missing}")
    base = methods["source_order"]
    signature = reference_signature(base)
    if any(reference_signature(methods[name]) != signature for name in required):
        raise ValueError("ranking methods do not share identical configuration/reference pools")
    rows = []
    random = dict(name="exact_uniform_random", label="Exact uniform random", color=COLORS["random"],
                  values={str(k): metric(base, f"random_quality{k}") for k in (1, 4)}, model_seeds=0)
    rows.append(random)
    for name in ("source_order", "shortest_initial"):
        rows.append(dict(name=name, label=model_label(name, frozen), color=COLORS[name],
                         values={str(k): metric(methods[name], f"quality{k}") for k in (1, 4)}, model_seeds=0))
    for kind in KINDS:
        records = [r for r in frozen["models"] if r["kind"] == kind]
        if len(records) != 3 or len({r["seed"] for r in records}) != 3:
            raise ValueError(f"{kind}: expected three distinct frozen model seeds")
        members = {str(k): [metric(methods[r["name"]], f"quality{k}") for r in records] for k in (1, 4)}
        values = {k: statistics.mean(v) if all(x is not None for x in v) else None for k, v in members.items()}
        rows.append(dict(name=kind, label=KIND_LABELS[kind] + "\nmean of 3 model seeds", color=COLORS[kind],
                         values=values, model_seeds=3, seed_names=[r["name"] for r in records], seed_values=members))
    selected = frozen["selected"]
    rows.append(dict(name="selected", label="Frozen selected policy\n" + model_label(selected, frozen, selected_marker=False),
                     color=COLORS["selected"], values={str(k): metric(methods[selected], f"quality{k}") for k in (1, 4)},
                     model_seeds=1, source_method=selected))
    for row in rows:
        if any(value is not None and not 0 <= value <= 1 for value in row["values"].values()):
            raise ValueError("quality must be a saved empirical fraction in [0,1]")
    meta = audit.get("source_metadata", {})
    counts = dict(configurations=base["configurations"], groups=base["groups"],
                  all_failure_groups=base.get("all_failure_groups"),
                  pool_sizes=sorted({r["n"] for r in base["rows"]}),
                  physical_repeats=sorted({r["repeats"] for r in meta.get("sources", []) if r.get("repeats") is not None}),
                  model_seed_count_is_not_an_additional_configuration_count=True)
    return dict(status="available", rows=rows, counts=counts,
                attrition=meta.get("attrition", audit.get("attrition")),
                aggregation="Use saved equal-configuration quality means; average three model-seed means within each model kind.")


def timing_data(timing, frozen):
    records = timing.get("rows", [])
    if not records:
        return dict(status=timing.get("status", "no_timing_input_records"), rows=[])
    desired = list(dict.fromkeys([frozen["selected"], "compact_29", "port_mlp_29", "sequence_29", "graph_29",
                                 "source_order", "shortest_initial"]))
    rows, unavailable = [], []
    for name in desired:
        samples = [r for r in records if r.get("method") == name]
        summary = timing.get("summary", {}).get(name)
        if not samples or not summary:
            unavailable.append(name)
            continue
        repetitions = defaultdict(set)
        for sample in samples:
            repetitions[str(sample["group_id"])].add(sample["repeat"])
        model = next((r for r in frozen["models"] if r["name"] == name), {})
        rows.append(dict(name=name, label=model_label(name, frozen),
                         color=COLORS.get(model.get("kind", name), COLORS["selected"]),
                         module_total_ms=(None if summary.get("module_total", {}).get("median_seconds") is None
                                          else 1000 * number(summary["module_total"]["median_seconds"])),
                         inference_ms=(None if summary.get("inference", {}).get("median_seconds") is None
                                       else 1000 * number(summary["inference"]["median_seconds"])),
                         configurations=len({str(r["config_id"]) for r in samples}),
                         groups=len(repetitions), timing_records=len(samples),
                         repeats_per_group=sorted({len(v) for v in repetitions.values()})))
    return dict(status="available", rows=rows, unavailable_methods=unavailable,
                aggregation="Saved median across configurations; each configuration averages checkpoint medians of repeated timings.",
                requested_groups=timing.get("expected_groups"), reached_groups=timing.get("reached_groups"),
                failed_setup_groups=timing.get("failed_setup_groups"))


def deployment_data(summary, frozen):
    decisions = summary.get("decisions", [])
    names = list(dict.fromkeys([frozen["selected"], "source_order", "shortest_initial", "exhaustive"]))
    rows = []
    for name in names:
        selected = [r for r in decisions if r.get("method") == name]
        if not selected:
            rows.append(dict(name=name, label=model_label(name, frozen), status="not_recorded"))
            continue
        identities = [(r["seed"], r.get("checkpoint")) for r in selected]
        if len(identities) != len(set(identities)):
            raise ValueError(f"duplicate deployment configuration/checkpoint for {name}")
        for row in selected:
            for key in ("deployment_successes", "deployment_attempts", "requested_deployment_attempts", "validation_calls", "online_budget"):
                if not isinstance(row.get(key), int) or row[key] < 0:
                    raise ValueError(f"missing or invalid actual/requested deployment count: {name}/{key}")
            if not row["deployment_successes"] <= row["deployment_attempts"] <= row["requested_deployment_attempts"]:
                raise ValueError(f"inconsistent deployment numerator/denominators: {name}")
        reached = [r for r in selected if r.get("status") in {"deployed", "unresolved_no_accepted_candidate"}]
        wall_by_config = defaultdict(list)
        for row in reached:
            value = number(row.get("timing", {}).get("policy_wall_seconds"))
            if value is not None:
                wall_by_config[str(row["seed"])].append(value)
        wall = [statistics.mean(v) for _, v in sorted(wall_by_config.items())]
        requested = sum(r["requested_deployment_attempts"] for r in selected)
        succeeded = sum(r["deployment_successes"] for r in selected)
        model = next((r for r in frozen["models"] if r["name"] == name), {})
        budgets = sorted({r["online_budget"] for r in selected})
        rows.append(dict(name=name, label=model_label(name, frozen), status="available",
            color=COLORS.get(model.get("kind", name), COLORS["selected"]),
            requested_attempts=requested, actual_attempts=sum(r["deployment_attempts"] for r in selected),
            successes=succeeded, success_fraction_requested=succeeded / requested if requested else None,
            configurations=len({r["seed"] for r in selected}), groups=len(selected),
            reached_configurations=len({r["seed"] for r in reached}),
            setup_failed_groups=sum(r.get("status") == "checkpoint_setup_failed" for r in selected),
            requested_validation_budgets_per_group=budgets,
            actual_validation_rollouts=sum(r["validation_calls"] for r in selected),
            reached_policy_wall_seconds_by_configuration=wall,
            reached_policy_wall_median_seconds=statistics.median(wall) if wall else None,
            reached_policy_wall_configurations=len(wall),
            missing_reached_policy_wall_groups=sum(number(r.get("timing", {}).get("policy_wall_seconds")) is None for r in reached)))
    return dict(rows=rows, status="available" if decisions else "no_deployment_decisions",
                requested_design=frozen.get("deployment"), parallel_workers=summary.get("parallel_workers"),
                recorded_overall_wall_seconds=summary.get("overall_wall_seconds"),
                wall_aggregation="Average checkpoint siblings within a reached configuration, then show the median across configurations.")


def styling():
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
                         "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.spines.left": False, "svg.fonttype": "path", "savefig.facecolor": "white"})
    return plt


def save(figure, out, stem, plt):
    paths = []
    for extension in ("png", "svg"):
        path = out / f"{stem}.{extension}"
        figure.savefig(path, dpi=240, bbox_inches="tight")
        paths.append(path)
    plt.close(figure)
    return paths


def empty_axis(axis, text):
    axis.text(.5, .5, text, transform=axis.transAxes, ha="center", va="center", color="#475569")
    axis.set_axis_off()


def horizontal(axis, rows, key, *, fraction=False, seed_key=None, annotations=None):
    from matplotlib.ticker import PercentFormatter
    for i, row in enumerate(rows):
        value = row.get(key)
        if value is None:
            axis.text(.02, i, "Not recorded", color="#64748B", va="center", transform=axis.get_yaxis_transform())
            continue
        axis.barh(i, value, height=.60, color=row.get("color", COLORS["selected"]), alpha=.86,
                  edgecolor="white", linewidth=.7, zorder=2)
        if seed_key:
            seeds = [x for x in row.get(seed_key, []) if x is not None]
            offsets = [0.] if len(seeds) == 1 else [(.16 * j / (len(seeds) - 1) - .08) for j in range(len(seeds))]
            axis.scatter(seeds, [i + y for y in offsets], marker="o", s=22, facecolor="white",
                         edgecolor="#25374C", linewidth=.8, zorder=4)
        label = annotations[i] if annotations is not None else f"{100 * value:.1f}%" if fraction else f"{value:.2f}"
        axis.annotate(label, xy=(value, i), xytext=(5, 0), textcoords="offset points", va="center", fontsize=9)
    axis.set_yticks(range(len(rows)), [r["label"] for r in rows])
    # A fixed limit is idempotent when two panels share their y-axis.
    axis.set_ylim(len(rows) - .6, -.6)
    axis.tick_params(axis="y", length=0, pad=8)
    axis.grid(axis="x", color="#E2E8F0", linewidth=.7, zorder=0)
    axis.set_axisbelow(True)
    if fraction:
        axis.set_xlim(0, 1.19)
        axis.set_xticks([0, .25, .5, .75, 1.])
        axis.xaxis.set_major_formatter(PercentFormatter(1.))
    else:
        maxima = [r[key] for r in rows if r.get(key) is not None]
        axis.set_xlim(0, max(maxima, default=1.) * 1.30 or 1.)


def plot_ranking(data, out, plt):
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 7.4), sharey=True)
    rows = data["rows"]
    if not rows:
        for axis in axes:
            empty_axis(axis, "No physically reached checkpoint\nNo candidate-ranking quality is available")
    else:
        for axis, k in zip(axes, (1, 4)):
            panel = [{**row, "quality": row["values"][str(k)], "seeds": row.get("seed_values", {}).get(str(k), [])} for row in rows]
            horizontal(axis, panel, "quality", fraction=True, seed_key="seeds")
            axis.set_title(f"Top-{k} selected-set quality", pad=13)
            axis.set_xlabel("Mean empirical success fraction in the selected set")
        axes[1].tick_params(axis="y", labelleft=False)
    counts = data.get("counts", {})
    title = f"Candidate-ranking quality | {counts.get('configurations', 0)} configurations, {counts.get('groups', 0)} reached groups"
    figure.suptitle(title, fontsize=14, y=.98)
    attr = data.get("attrition") or {}
    attr_text = (f"Requested groups: {attr['expected_groups']}; reached: {attr.get('reached_groups', 'not recorded')}; "
                 f"setup failures: {attr.get('failed_setup_groups', 'not recorded')}. " if "expected_groups" in attr else "")
    pool = ", ".join(map(str, counts.get("pool_sizes", []))) or "not recorded"
    repeats = ", ".join(map(str, counts.get("physical_repeats", []))) or "not recorded"
    figure.text(.02, .035,
        f"{attr_text}Pool N: {pool}; physical repeats per candidate: {repeats}.\n"
        "Bars use equal configuration weights; model-kind bars average 3 model-seed estimates from the same configurations.\n"
        "White dots show individual model seeds, not additional configurations or confidence intervals. All-failure pools remain in quality; setup failures have no candidate ranking.",
        fontsize=8.4, color="#475569", linespacing=1.45)
    figure.subplots_adjust(left=.24, right=.98, top=.90, bottom=.20, wspace=.18)
    return save(figure, out, "ranking_quality", plt)


def plot_timing(data, out, plt):
    rows = data["rows"]
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 6.3), sharey=True)
    if not rows:
        for axis in axes:
            empty_axis(axis, "No reached input records for timing")
    else:
        display = [{**r, "label": r["label"] + f"\n{r['configurations']} configs; {r['groups']} groups"} for r in rows]
        for axis, key, title in zip(axes, ("module_total_ms", "inference_ms"),
                                    ("Complete value-module time", "Network inference time")):
            horizontal(axis, display, key)
            axis.set_title(title, pad=13)
            axis.set_xlabel("Saved configuration median (ms / candidate pool)")
        axes[1].tick_params(axis="y", labelleft=False)
    counts = sorted({tuple(r["repeats_per_group"]) for r in rows})
    repeat_text = "; ".join(", ".join(map(str, value)) for value in counts) or "not recorded"
    figure.suptitle("Measured value-module timing", fontsize=14, y=.98)
    figure.text(.02, .045,
        "Complete module = graph construction + integrity checks + encoding + inference + export.\n"
        f"Timing repetitions/group: {repeat_text}. Saved medians aggregate repeated timing within checkpoints, then configurations.\n"
        "Timing repetitions and model seeds do not increase the configuration count. No confidence intervals or rollout-time estimates are shown.",
        fontsize=8.5, color="#475569", linespacing=1.45)
    figure.subplots_adjust(left=.25, right=.97, top=.90, bottom=.20, wspace=.30)
    return save(figure, out, "module_timing", plt)


def plot_deployment(data, frozen, out, plt):
    rows = data["rows"]
    display = []
    for row in rows:
        budgets = row.get("requested_validation_budgets_per_group", [])
        budget_label = "/".join(map(str, budgets)) or "not recorded"
        display.append({**row, "label": row["label"] + f"\nvalidation budget = {budget_label} rollouts/group"})
    figure, axes = plt.subplots(1, 2, figsize=(14.5, 6.0), sharey=True)
    if not any(r.get("status") == "available" for r in rows):
        for axis in axes:
            empty_axis(axis, "No deployment decisions recorded")
    else:
        annotations = [f"{r.get('successes', 0)}/{r.get('requested_attempts', 0)} requested\n"
                       f"{r.get('actual_attempts', 0)} actually executed" if r.get("status") == "available" else "Not recorded" for r in rows]
        horizontal(axes[0], display, "success_fraction_requested", fraction=True, annotations=annotations)
        axes[0].set_xlim(0, 1.57)
        axes[0].set_title("Independent deployment outcomes", pad=13)
        axes[0].set_xlabel("Successful executions / requested attempts")
        annotations = [f"{r.get('reached_policy_wall_median_seconds', 0):.1f} s\n"
                       f"n={r.get('reached_policy_wall_configurations', 0)} reached configs"
                       if r.get("reached_policy_wall_median_seconds") is not None else "Not recorded" for r in rows]
        horizontal(axes[1], display, "reached_policy_wall_median_seconds", seed_key="reached_policy_wall_seconds_by_configuration", annotations=annotations)
        wall_values = [x for r in rows for x in r.get("reached_policy_wall_seconds_by_configuration", [])]
        if wall_values:
            axes[1].set_xlim(0, max(wall_values) * 1.40 or 1.)
        axes[1].set_title("Policy wall time at reached checkpoints", pad=13)
        axes[1].set_xlabel("Median across reached configurations (s)")
        axes[1].tick_params(axis="y", labelleft=False)
    design = data.get("requested_design") or {}
    requested = len(design.get("tasks", design.get("seeds", [])))
    setups = {r["name"]: r.get("setup_failed_groups") for r in rows if r.get("status") == "available"}
    setup_values = sorted({x for x in setups.values() if x is not None})
    figure.suptitle(f"Frozen-policy deployment | {requested} requested configuration/checkpoint groups", fontsize=14, y=.98)
    figure.text(.02, .035,
        f"Recorded setup-failure groups per policy: {', '.join(map(str, setup_values)) or 'not recorded'}; parallel configuration workers: {data.get('parallel_workers', 'not recorded')}.\n"
        "Setup failures contribute zero successes to the requested denominator and are excluded from wall-time samples. White dots are reached configurations.\n"
        "Policy wall time includes ranking, validation and independent deployment; shared checkpoint/candidate/graph preparation is excluded.\n"
        "Exhaustive validates K=N candidates under its explicitly larger budget. No equal-budget speedup or confidence interval is inferred.",
        fontsize=8.3, color="#475569", linespacing=1.4)
    figure.subplots_adjust(left=.265, right=.98, top=.88, bottom=.25, wspace=.26)
    return save(figure, out, "deployment_outcomes", plt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, help="Directory containing the completed ranking/timing/deployment JSON files")
    parser.add_argument("--out", required=True)
    parser.add_argument("--freeze", help="Frozen manifest; defaults to --evidence/frozen.json")
    parser.add_argument("--no-timing", action="store_true", help="Do not emit the optional timing figure")
    args = parser.parse_args()
    evidence, out = Path(args.evidence), Path(args.out)
    paths = {"audit": evidence / "ranking_audit.json", "timing": evidence / "module_timing.json",
             "deployment": evidence / "deployment_summary.json",
             "freeze": Path(args.freeze) if args.freeze else evidence / "frozen.json"}
    payload = {name: read(path) for name, path in paths.items()}
    frozen = payload["freeze"]
    if frozen.get("schema") != "twingraph.schema_experiment.freeze.v1":
        raise ValueError("unsupported frozen experiment schema")
    for name in ("timing", "deployment"):
        if payload[name].get("freeze_sha256") != digest(frozen):
            raise ValueError(f"{name} evidence does not belong to the supplied freeze")
    audit_freeze = payload["audit"].get("source_metadata", {}).get("freeze_sha256")
    if audit_freeze is not None and audit_freeze != digest(frozen):
        raise ValueError("ranking audit does not belong to the supplied freeze")
    data = dict(schema="twingraph.value.figures.v4", sources={name: dict(path=str(path), sha256=file_sha(path)) for name, path in paths.items()},
                ranking=ranking_data(payload["audit"], frozen), timing=timing_data(payload["timing"], frozen),
                deployment=deployment_data(payload["deployment"], frozen),
                intervals="No new confidence intervals are estimated or displayed.")
    out.mkdir(parents=True, exist_ok=True)
    plt = styling()
    artifacts = plot_ranking(data["ranking"], out, plt)
    if not args.no_timing:
        artifacts.extend(plot_timing(data["timing"], out, plt))
    artifacts.extend(plot_deployment(data["deployment"], frozen, out, plt))
    data["artifacts"] = [dict(path=p.name, sha256=file_sha(p)) for p in artifacts]
    (out / "plot_data.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(output=str(out), artifacts=[p.name for p in artifacts]), indent=2))


if __name__ == "__main__":
    main()
