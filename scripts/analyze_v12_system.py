"""Read-only audit/report of actual online runs; partial runs stay partial.

No simulation, training, cached rollout replay, or time imputation is performed.
Legacy V11 requires an explicit flag and is always identified as historical.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


METHODS = ("all_twin", "random_top_k", "value_top_k", "random_early_stop", "value_early_stop",
           "geometry_top_k", "geometry_early_stop")
VALUE_METHODS = ("value_top_k", "value_early_stop")
SCORED_METHODS = (*VALUE_METHODS, "geometry_top_k", "geometry_early_stop")
PRIMARY = METHODS[:3]
LABELS = {"all_twin": "全量孪生", "random_top_k": "随机 Top-K", "value_top_k": "价值 Top-K",
          "random_early_stop": "随机成功早停", "value_early_stop": "价值渐进成功早停",
          "geometry_top_k": "几何余量 Top-K", "geometry_early_stop": "几何余量渐进成功早停"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(data, *, compact=False):
    kwargs = dict(sort_keys=True, allow_nan=False)
    if compact:
        kwargs["separators"] = (",", ":")
    return hashlib.sha256(json.dumps(data, **kwargs).encode()).hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def valid_hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def measured(data, key):
    value = data.get(key)
    require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0,
            f"missing/invalid measured timer {key}; no inferred or cached replacement is allowed")
    return float(value)


def audit_batches(request, completed, order, attempted):
    """Require the archived opening of each progressive batch, not inferred cost."""
    meta=request.get("progressive")
    require(isinstance(meta,dict), "missing progressive verification policy")
    size=meta.get("batch_size")
    require(isinstance(size,int) and not isinstance(size,bool) and 1<=size<=len(order)
            and meta.get("initial_k")==size and meta.get("max_candidates")==len(order)
            and meta.get("policy")=="ordered_batches_first_verified_success", "invalid progressive batch policy")
    expected=[dict(batch=start//size+1,rank_start=start,rank_stop=min(start+size,len(order)),
                   attempted_indices=order[start:min(start+size,attempted)])
              for start in range(0,attempted,size)]
    require(completed.get("verification_batches")==expected, "progressive batch log disagrees with actual ordered trials")
    return len(expected)


def stop_record(result, seed, method, phase, name):
    if result["success"]:
        return None
    steps = result.get("executed_parameters") or []
    failed = next((row for row in steps if row.get("ok") is False), None)
    part = (failed or {}).get("params", {}).get("part")
    passes = result.get("stage_passes", {})
    boundaries = result.get("boundaries") or []
    last_stage = boundaries[-1].get("stage") if boundaries else None
    if passes.get("functional_test_pass"):
        stage, basis = "final_release_or_retention", "recorded_stage_passes"
    elif passes.get("assembly_pass"):
        stage, basis = "functional_motion", "recorded_stage_passes"
    elif part:
        stage, basis = ("cleaning" if part == "wipe_tool" else str(part)), "first_failed_call_part"
    else:
        stage, basis = (f"after_{last_stage}" if last_stage else "before_first_completed_stage"), "last_recorded_boundary_only"
    error = str(result.get("error", ""))
    return dict(seed=seed, method=method, phase=phase, candidate=name, stopped_stage=stage, stage_basis=basis,
                last_completed_boundary=last_stage, first_failed_skill=(failed or {}).get("skill"),
                terminal_failure=error.split(":", 1)[0] if error else "unrecorded", error=error)


def audit_method(directory, seed, method, *, legacy=False, expected_checkpoint=None):
    summary = read(directory/"summary.json")
    request = read(directory/"request.json")
    require(summary.get("seed") == request.get("seed") == seed, "seed disagrees with directory/request")
    require(summary.get("method") == request.get("method") == method, "method disagrees with directory/request")
    missing_provenance = []
    if not legacy:
        require(summary.get("valid") is True, "summary valid must be explicitly true")
        require(request.get("domain") == "online", "request is not an online experiment")
    pool = request["pool"]
    names = [p["name"] for p in pool]
    require(len(names) > 0 and len(names) == len(set(names)), "empty/duplicated candidate pool")
    require(summary.get("pool_size") == len(pool), "summary pool size mismatch")
    order = request["order"]
    require(sorted(order) == list(range(len(pool))), "request order is not a candidate permutation")
    budget = summary["k"]
    require(isinstance(budget, int) and 1 <= budget <= len(pool) and request["k"] == budget, "invalid candidate budget")
    if method.endswith("early_stop"):
        require(budget == len(pool), "unlimited early-stop method does not cover the full pool")
    runtime = request.get("runtime_sha256")
    geometry = request.get("geometry_version")
    observation = request.get("initial_observation", {}).get("sha256") or request.get("source", {}).get("observation_sha256")
    if not legacy:
        require(valid_hash(runtime) and summary.get("runtime_sha256") == runtime, "missing/inconsistent runtime fingerprint")
        provenance = read(directory/"runtime_sources.json")
        require(provenance.get("sha256") == runtime and digest(provenance["files"], compact=True) == runtime,
                "runtime source-file manifest is not bound to request fingerprint")
        require(isinstance(geometry, str) and geometry, "missing geometry version")
        require(valid_hash(observation), "missing initial RGB-D fingerprint")
        require("model_sha256" in request, "checkpoint identity was not recorded, including explicit null for no model")
    else:
        for field in ("valid", "runtime_sha256", "geometry_version"):
            if field not in summary and field not in request:
                missing_provenance.append(field)
    checkpoint = request.get("model_sha256")
    if method in VALUE_METHODS:
        require(valid_hash(checkpoint), "value ranking checkpoint hash is missing")
        if expected_checkpoint:
            require(checkpoint == expected_checkpoint, "ranking checkpoint differs from expected frozen file")
    if method in SCORED_METHODS:
        scores = request.get("scores")
        require(isinstance(scores, list) and len(scores) == len(pool) and all(isinstance(s, (int, float)) and math.isfinite(s) for s in scores),
                "missing/nonfinite value scores")
        require(order == sorted(range(len(pool)), key=lambda i: -scores[i]), "recorded order does not follow stable value scores")
    trials = summary["trials"]
    require(isinstance(trials, list) and 1 <= len(trials) <= budget, "invalid/incomplete trial list")
    batches_opened=None
    if method in ("value_early_stop","geometry_early_stop"):
        require(summary.get("progressive")==request.get("progressive"), "progressive summary/request policy mismatch")
        batches_opened=audit_batches(request,summary,order,len(trials))
    if method == "all_twin":
        require(budget == len(pool) and len(trials) == len(pool), "full twin method lacks complete actual rollouts")
    stops = []
    physical_seconds = 0.
    first_success = None
    graph_geometry = set()
    for j, trial in enumerate(trials):
        i = order[j]
        require(trial.get("index") == i and trial.get("name") == names[i], "executed trial order/name differs from request")
        path = directory/"twins"/names[i]
        detail = read(path/"result.json")
        require(detail.get("valid") is True, f"invalid physical rollout {names[i]}")
        require(detail.get("domain") == "online", "twin detail was not an independent online rollout")
        require(detail.get("proposal") == pool[i], "physical proposal differs from requested candidate")
        require(isinstance(detail.get("success"), bool) and detail["success"] == trial.get("success"), "success label mismatch")
        require(detail.get("seed", seed) == seed, "physical seed mismatch")
        graph = read(path/"input_graph.json")
        if legacy and detail.get("input_graph_sha256") is None:
            if "input_graph_sha256" not in missing_provenance:
                missing_provenance.append("input_graph_sha256")
        else:
            require(detail.get("input_graph_sha256") == digest(graph), "physical result is not bound to archived input graph")
        require(graph.get("proposal") == pool[i], "graph proposal mismatch")
        if not legacy:
            require(detail.get("runtime_sha256") == runtime, "physical runtime differs from online request")
            require(detail.get("geometry_version") == geometry == graph.get("task_geometry_version"), "physical/graph geometry mismatch")
            require(detail.get("initial_observation", {}).get("sha256") == observation, "twin initial RGB-D differs from decision input")
            require(graph["assembly"]["observation"]["perception"]["observation_sha256"] == observation,
                    "value graph does not carry the decision RGB-D fingerprint")
            graph_geometry.add(digest(graph.get("planning_cad", {})))
        duration = measured(trial, "wall_seconds")
        require(math.isclose(duration, measured(detail, "total_wall_seconds"), rel_tol=1.e-8, abs_tol=1.e-6), "trial timer not measured from its physical run")
        physical_seconds += duration
        if detail["success"] and first_success is None:
            first_success = names[i]
        failure = stop_record(detail, seed, method, "twin", names[i])
        if failure:
            stops.append(failure)
    require(len(graph_geometry) <= 1, "CAD changed between candidates")
    if not legacy and request.get("source", {}).get("cad_sha256"):
        require(graph_geometry == {request["source"]["cad_sha256"]}, "graph CAD differs from candidate grounding CAD")
    if method != "all_twin":
        successes = [j for j, t in enumerate(trials) if t["success"]]
        require((successes == [len(trials)-1]) if successes else len(trials) == budget,
                "early-stop method is truncated or continues after success")
    require(summary.get("selected") == first_success and summary.get("twin_success") is (first_success is not None), "selected/twin success contradicts actual rollouts")
    execution_seconds = measured(summary, "execution_seconds")
    require(isinstance(summary.get("execution_success"), bool), "missing execution outcome")
    if first_success is not None:
        deployment = read(directory/"deployment"/"result.json")
        require(deployment.get("valid") is True and deployment.get("domain") == "deployment", "missing/invalid independent deployment")
        require(isinstance(deployment.get("success"), bool) and deployment["success"] == summary["execution_success"], "deployment label mismatch")
        require(deployment.get("proposal") == pool[names.index(first_success)], "deployment selected proposal mismatch")
        require(math.isclose(execution_seconds, measured(deployment, "total_wall_seconds"), rel_tol=1.e-8, abs_tol=1.e-6), "deployment timer mismatch")
        if not legacy:
            require(deployment.get("runtime_sha256") == runtime and deployment.get("geometry_version") == geometry, "deployment runtime/geometry drift")
            graph = read(directory/"deployment"/"input_graph.json")
            require(deployment.get("input_graph_sha256") == digest(graph), "deployment is not bound to archived input graph")
            require(graph.get("proposal") == deployment["proposal"] and graph.get("task_geometry_version") == geometry,
                    "deployment graph proposal/geometry mismatch")
        failure = stop_record(deployment, seed, method, "deployment", first_success)
        if failure:
            stops.append(failure)
    else:
        require(summary["execution_success"] is False and execution_seconds == 0., "unverified plan has a fabricated deployment outcome/time")
    # Suffix validations are real additional twin calls, but their wall time
    # is already inside deployment_seconds. Do not add that time twice.
    replans = list(directory.glob("closed_loop/replan_*/*/result.json"))
    replan_requests = {}
    if not legacy:
        for request_path in sorted(directory.glob("closed_loop/replan_*/request.json")):
            recovery = read(request_path)
            require(recovery.get("method") == method, "suffix ranking method changed from the online method")
            suffix_pool = recovery["pool"]
            suffix_names = [p["name"] for p in suffix_pool]
            require(suffix_names and len(set(suffix_names)) == len(suffix_names), "empty/duplicated suffix pool")
            suffix_order = recovery["order"]
            require(sorted(suffix_order) == list(range(len(suffix_pool))), "invalid suffix candidate permutation")
            suffix_k = recovery["k"]
            require(isinstance(suffix_k, int) and not isinstance(suffix_k, bool) and suffix_k > 0, "invalid suffix budget")
            suffix_budget = min(suffix_k, len(suffix_pool))
            if method.endswith("early_stop"):
                require(suffix_budget == len(suffix_pool), "unlimited suffix budget does not cover the pool")
            scores = recovery.get("scores")
            if method in SCORED_METHODS:
                require(isinstance(scores, list) and len(scores) == len(suffix_pool)
                    and all(isinstance(s, (int, float)) and math.isfinite(s) for s in scores), "invalid suffix value scores")
                require(suffix_order == sorted(range(len(scores)), key=lambda i:-scores[i]), "suffix order does not follow value scores")
            else:
                require(scores is None, "non-value suffix method unexpectedly uses value scores")
            suffix_paths = {p.parent.name:p for p in request_path.parent.glob("*/result.json")}
            require(0 < len(suffix_paths) <= suffix_budget, "missing/incomplete suffix trial set")
            require(set(suffix_paths) == {suffix_names[i] for i in suffix_order[:len(suffix_paths)]},
                    "suffix trials do not match the ordered candidate prefix")
            if method in ("value_early_stop","geometry_early_stop"):
                audit_batches(recovery,read(request_path.with_name("progress.json")),suffix_order,len(suffix_paths))
            outcomes = []
            for i in suffix_order[:len(suffix_paths)]:
                detail = read(suffix_paths[suffix_names[i]])
                require(detail.get("proposal") == suffix_pool[i], "suffix physical proposal differs from recovery request")
                outcomes.append(detail.get("success"))
            if method == "all_twin":
                require(suffix_budget == len(suffix_pool) and len(outcomes) == len(suffix_pool),
                        "full twin replanning lacks complete actual suffix rollouts")
            else:
                succeeded = [j for j, success in enumerate(outcomes) if success is True]
                require((succeeded == [len(outcomes)-1]) if succeeded else len(outcomes) == suffix_budget,
                        "suffix early-stop record is truncated or continues after success")
            replan_requests[request_path.parent] = recovery
    for path in replans:
        detail = read(path)
        require(detail.get("valid") is True and detail.get("domain") == "online", "invalid suffix twin verification")
        require(isinstance(detail.get("success"), bool), "suffix twin success label missing")
        measured(detail, "total_wall_seconds")
        graph = read(path.with_name("input_graph.json"))
        if not legacy:
            require(path.parent.parent in replan_requests, "suffix trial has no audited recovery request")
            require(detail.get("runtime_sha256") == runtime and detail.get("geometry_version") == geometry,
                    "suffix twin runtime/geometry drift")
            require(detail.get("input_graph_sha256") == digest(graph), "suffix twin is not bound to archived graph")
            require(graph.get("proposal") == detail.get("proposal"), "suffix graph proposal mismatch")
        failure = stop_record(detail, seed, method, "replan_twin", path.parent.name)
        if failure:
            stops.append(failure)
    verified_replan = any(str(event.get("action", "")).startswith(("verified_remaining_suffix", "replan_verified"))
                         for event in summary.get("events", []))
    require(not verified_replan or bool(replans), "verified replanning event has no archived suffix trials")
    decision_seconds = measured(summary, "decision_seconds")
    require(decision_seconds+1.e-5 >= physical_seconds, "decision wall time is shorter than its measured serial twin rollouts")
    total_seconds = measured(summary, "total_wall_seconds")
    require(total_seconds+1.e-5 >= decision_seconds+execution_seconds, "total wall time is shorter than decision plus execution")
    return dict(seed=seed, method=method, status="valid_complete_method", legacy=legacy, k=budget,
        pool_size=len(pool), pool_sha256=digest(pool), initial_observation_sha256=observation,
        geometry_version=geometry, cad_sha256=next(iter(graph_geometry), None), runtime_sha256=runtime,
        model_sha256=checkpoint, twin_calls=len(trials)+len(replans), initial_twin_calls=len(trials),
        replan_twin_calls=len(replans), twin_success=summary["twin_success"],
        progressive_batches_opened=batches_opened,
        full_pool_fallback_enabled=method=="all_twin" or method.endswith("early_stop"),
        deployment_attempted=first_success is not None, execution_success=summary["execution_success"],
        decision_seconds=decision_seconds, execution_seconds=execution_seconds, total_wall_seconds=total_seconds,
        generation_seconds=measured(summary, "generation_seconds"), ranking_seconds=measured(summary, "ranking_seconds"),
        missing_legacy_provenance=missing_provenance, stops=stops,
        source_summary=str(directory/"summary.json"), summary_sha256=hashlib.sha256((directory/"summary.json").read_bytes()).hexdigest())


def aggregate(rows, methods):
    output = []
    for method in methods:
        group = [row for row in rows if row["method"] == method]
        if not group:
            output.append(dict(method=method, completed_layouts=0, metrics=None))
            continue
        mean = lambda key: sum(r[key] for r in group)/len(group)
        deployed = [r for r in group if r["deployment_attempted"]]
        output.append(dict(method=method, completed_layouts=len(group), layouts=[r["seed"] for r in group], metrics=dict(
            twin_success_rate=mean("twin_success"), execution_success_rate=mean("execution_success"),
            twin_successes=sum(r["twin_success"] for r in group), execution_successes=sum(r["execution_success"] for r in group),
            mean_pool_size=mean("pool_size"), mean_twin_calls=mean("twin_calls"), budgets=sorted({r["k"] for r in group}),
            mean_initial_twin_calls=mean("initial_twin_calls"), mean_replan_twin_calls=mean("replan_twin_calls"),
            mean_decision_seconds=mean("decision_seconds"), mean_deployment_seconds_all_layouts=mean("execution_seconds"),
            mean_deployment_seconds_when_attempted=sum(r["execution_seconds"] for r in deployed)/len(deployed) if deployed else None,
            deployments_attempted=len(deployed), mean_total_seconds=mean("total_wall_seconds"))))
    return output


def analyze(root, *, seeds=None, methods=PRIMARY, legacy=False, expected_checkpoint=None):
    root = Path(root)
    discovered = sorted(int(p.name.split("_")[-1]) for p in root.glob("seed_*") if p.is_dir() and p.name.split("_")[-1].isdigit())
    expected = sorted(set(seeds if seeds is not None else discovered))
    rows, incomplete, invalid, pairs = [], [], [], []
    for seed in expected:
        entries = []
        for method in methods:
            directory = root/f"seed_{seed}"/method
            if not (directory/"summary.json").exists():
                progress, progress_error = {}, None
                try:
                    if (directory/"progress.json").exists():
                        progress = read(directory/"progress.json")
                        require(isinstance(progress, dict) and isinstance(progress.get("trials", []), list), "invalid progress structure")
                except (ValueError, TypeError, OSError) as exc:
                    # A running writer may not have completed its latest JSON.
                    # Report the partial state without fabricating a trial count.
                    progress, progress_error = {}, str(exc)
                incomplete.append(dict(seed=seed, method=method, status="partial" if directory.exists() else "missing",
                    recorded_progress_trials=None if progress_error else len(progress.get("trials", [])),
                    progress_read_error=progress_error, summary_present=False))
                continue
            try:
                row = audit_method(directory, seed, method, legacy=legacy, expected_checkpoint=expected_checkpoint)
            except (ValueError, KeyError, TypeError, OSError) as exc:
                invalid.append(dict(seed=seed, method=method, reason=str(exc)))
                continue
            rows.append(row); entries.append(row)
        fields = ("pool_sha256", "initial_observation_sha256") if legacy else (
            "pool_sha256", "initial_observation_sha256", "geometry_version", "cad_sha256", "runtime_sha256")
        differences = [field for field in fields if len({r[field] for r in entries}) > 1]
        top_k = [r["k"] for r in entries if r["method"] in ("random_top_k", "value_top_k")]
        if len(set(top_k)) > 1:
            differences.append("top_k_budget")
        pairs.append(dict(seed=seed, complete_methods=[r["method"] for r in entries],
            status="paired_complete" if len(entries) == len(methods) and not differences else "mismatch" if differences else "partial",
            mismatched_fields=differences))
    model_hashes = sorted({r["model_sha256"] for r in rows if r["method"] in VALUE_METHODS})
    runtime_hashes = sorted({r["runtime_sha256"] for r in rows if r["runtime_sha256"]})
    geometry_versions = sorted({r["geometry_version"] for r in rows if r["geometry_version"]})
    cross_layout_consistent = len(model_hashes) <= 1 and len(runtime_hashes) <= 1 and len(geometry_versions) <= 1
    paired_seeds = [p["seed"] for p in pairs if p["status"] == "paired_complete"] if cross_layout_consistent else []
    paired_rows = [r for r in rows if r["seed"] in paired_seeds]
    stops = [stop for row in rows for stop in row["stops"]]
    failure_counts = {phase: dict(Counter(s["stopped_stage"] for s in stops if s["phase"] == phase)) for phase in ("twin", "replan_twin", "deployment")}
    complete = bool(expected) and len(paired_seeds) == len(expected) and not incomplete and not invalid
    return dict(schema="twingraph.actual_online_audit.v12", data_origin="historical_V11_actual_online" if legacy else "V12_actual_online_simulation",
        status="complete" if complete else "partial_or_invalid", expected_layouts=expected, expected_methods=list(methods),
        expected_layouts_explicit=seeds is not None, complete_paired_layouts=paired_seeds,
        paired_aggregate=aggregate(paired_rows, methods), available_aggregate=aggregate(rows, methods),
        paired_checks=pairs, incomplete=incomplete, invalid=invalid, rows=rows,
        cross_layout_provenance_consistent=cross_layout_consistent, checkpoint_hashes=model_hashes,
        runtime_hashes=runtime_hashes, geometry_versions=geometry_versions,
        stopping_stage_distribution=failure_counts, stopping_details=stops,
        limitations=["Only source summary timers from actual online execution are reported; no cached cost estimates or missing-as-zero values.",
            "No hardware claim. Independent deployment is a fresh perturbed simulation.",
            "Value/random unlimited early stop cover existence in the same finite valid labelled pool; independent deployment outcomes need not be equal.",
            "Missing entire layout directories can only be detected when --seeds is explicitly supplied.",
            "Legacy V11 mode cannot certify V12 runtime/geometry fingerprints and never relabels historical runs as V12."])


def markdown(report):
    title = "历史 V11 实际在线记录兼容审计" if report["data_origin"].startswith("historical") else "V12 实际在线系统对照审计"
    lines = [f"# {title}", "", f"状态：**{report['status']}**。预期 {len(report['expected_layouts'])} 个布局，"
             f"满足同池、同观测与来源校验的完整配对布局：{len(report['complete_paired_layouts'])}。", "",
             "下面仅比较完整配对且来源一致的布局。缺失方法、缺失布局和未完成计时显示为未完成，不填零，不用离线缓存耗时代替。", "",
             "| 方法 | 完整布局 | 孪生找到可行计划 | 独立仿真执行成功 | 平均候选池 / 验证数 | 决策墙钟秒 | 部署墙钟秒* |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in report["paired_aggregate"]:
        m, n = row["metrics"], row["completed_layouts"]
        if m is None:
            lines.append(f"| {LABELS[row['method']]} | 未完成 | — | — | — | — | — |")
        else:
            lines.append(f"| {LABELS[row['method']]} | {n} | {m['twin_successes']}/{n} | {m['execution_successes']}/{n} | "
                         f"{m['mean_pool_size']:.1f} / {m['mean_twin_calls']:.2f} | {m['mean_decision_seconds']:.2f} | {m['mean_deployment_seconds_all_layouts']:.2f} |")
    lines += ["", "*部署均值按这些完整布局计算：记录明确说明没有可行计划、未启动部署时才保留实际记录的 0 秒；"
              "缺失部署记录不会填 0。条件于实际启动部署的均值另存 JSON。验证数包含实际归档的后缀重规划验证；"
              "后缀时间已包含在部署时间内，不重复相加。价值渐进方法按预声明 K 扩展，首次验证成功停止、无成功则验证全部候选；"
              "同一有限有效标签池的可行解存在性覆盖与全量一致，但不保证独立部署成功率相同。随机早停单独比较。", "", "## 完整性与来源", ""]
    if report["data_origin"].startswith("historical"):
        missing = sorted({field for row in report["rows"] for field in row["missing_legacy_provenance"]})
        lines.append("这是明确指定 --legacy-v11 的历史兼容读取，不能作为 V12 实验。缺少的旧版来源字段："+", ".join(missing)+"；这些字段未被补写或伪造。")
    if not report["expected_layouts_explicit"]:
        lines.append("未显式给出预期 seed；无法判断尚未创建目录的布局是否缺失。")
    for row in report["incomplete"]:
        progress = "进度文件暂不可读" if row["recorded_progress_trials"] is None else f"仅记录到 {row['recorded_progress_trials']} 次进度试验"
        lines.append(f"- seed {row['seed']} / {row['method']}：{row['status']}，{progress}，无完整 summary。")
    for row in report["invalid"]:
        lines.append(f"- seed {row['seed']} / {row['method']}：审计排除，{row['reason']}。")
    for row in report["paired_checks"]:
        if row["mismatched_fields"]:
            lines.append(f"- seed {row['seed']} 方法间不一致：{', '.join(row['mismatched_fields'])}。")
    if not report["cross_layout_provenance_consistent"]:
        lines.append("- 跨布局发现不同运行时、几何版本或价值权重，未合并为同一配对对照。")
    if not report["incomplete"] and not report["invalid"]:
        lines.append("已发现的方法记录通过逐试验来源、结果及实际计时一致性校验。")
    lines += ["", "## 实际可用方法记录", "", "此表仅展示已有完整方法的样本数，不能将不同布局分母直接当配对比较。", ""]
    lines += [f"- {LABELS[r['method']]}：{r['completed_layouts']} 个有效完整方法记录。" for r in report["available_aggregate"]]
    lines += ["", "## 停止阶段", "", "来自所有通过审计的完整方法；同一候选在不同方法中重新运行，分别计数。"
              "JSON 同时保留首次失败技能、终止错误、最后完成边界，避免把局部首次失败混成最终停止原因。", ""]
    for phase, counts in report["stopping_stage_distribution"].items():
        lines.append(f"- {phase}：" + ("；".join(f"{key} {count}" for key, count in sorted(counts.items())) if counts else "没有可汇总的已记录失败。"))
    lines += ["", "![实际记录对照](system_comparison.png)", "", "权重 SHA256：" + ("、".join(report["checkpoint_hashes"]) or "未发现价值方法权重记录。"), "",
              "本报告不训练模型、不调用仿真、不重新估计候选耗时；不表示真实机器人验证。"]
    return "\n".join(lines)+"\n"


def plot(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    available = [r for r in report["paired_aggregate"] if r["metrics"] is not None]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), layout="constrained")
    origin = "Historical V11" if report["data_origin"].startswith("historical") else "V12"
    if not available:
        for ax in axes:
            ax.axis("off")
        axes[1].text(.5, .5, "No complete audited paired layouts\nMissing / partial / invalid data are not zero", ha="center", va="center", wrap=True)
    else:
        labels = [r["method"].replace("_", " ")+f"\n(n={r['completed_layouts']})" for r in available]
        x = list(range(len(available)))
        decision = [r["metrics"]["mean_decision_seconds"] for r in available]
        deployment = [r["metrics"]["mean_deployment_seconds_all_layouts"] for r in available]
        axes[0].bar(x, decision, color="#376d91", label="Measured decision")
        axes[0].bar(x, deployment, bottom=decision, color="#8fbf9d", label="Measured deployment")
        axes[0].set_ylabel("Wall-clock seconds / layout"); axes[0].legend(fontsize=8)
        axes[1].bar([i-.18 for i in x], [r["metrics"]["twin_success_rate"]*100 for r in available], width=.35, label="Twin feasible", color="#9bc0d5")
        axes[1].bar([i+.18 for i in x], [r["metrics"]["execution_success_rate"]*100 for r in available], width=.35, label="Independent execution", color="#2d6f54")
        axes[1].set_ylim(0, 105); axes[1].set_ylabel("Success rate (%)"); axes[1].legend(fontsize=8)
        axes[2].bar(x, [r["metrics"]["mean_twin_calls"] for r in available], color="#b88141")
        axes[2].set_ylabel("Actual twin rollouts / layout")
        for ax in axes:
            ax.set_xticks(x, labels, rotation=23, ha="right", fontsize=8)
            ax.grid(axis="y", alpha=.2)
    fig.suptitle(f"{origin}: actual online records — {report['status']}")
    fig.savefig(path, dpi=180); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(PRIMARY))
    p.add_argument("--legacy-v11", action="store_true")
    p.add_argument("--checkpoint", type=Path, help="Optional expected frozen weight file; only hashes it")
    args = p.parse_args()
    expected = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if args.checkpoint else None
    report = analyze(args.root, seeds=args.seeds, methods=args.methods, legacy=args.legacy_v11, expected_checkpoint=expected)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/"summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.out/"REPORT_ZH.md").write_text(markdown(report), encoding="utf-8")
    plot(report, args.out/"system_comparison.png")
    print(json.dumps(dict(status=report["status"], origin=report["data_origin"], paired=report["complete_paired_layouts"],
        missing=len(report["incomplete"]), invalid=len(report["invalid"])), ensure_ascii=False))


if __name__ == "__main__":
    main()
