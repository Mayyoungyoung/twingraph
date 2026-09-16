"""Summarize preserved v5 evidence, with explicit pending data and denominators.

No fitting, simulation or model selection occurs here. The majority control is
fixed by validation labels only; the order control uses saved input order.
Plots use standard Matplotlib and contain only supplied/audited measurements.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import evaluate_value_v5 as evaluator
from simbench.value.plan import digest, JOINT_PATH_FIELDS


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def first_method(payload, split):
    methods = evaluator.validate_predictions(payload, split)
    return methods, next(iter(methods.values()))["rows"]


def controls(validation, test, k=4):
    """Validation-only majority decision and deterministic input-order screening."""
    _, va = first_method(validation, "validation")
    _, te = first_method(test, "test")
    if {r["config_id"] for r in va} & {r["config_id"] for r in te}:
        raise ValueError("validation and test configurations overlap")
    validation_y = np.concatenate([r["outcomes"][:,r["nominal_index"]] for r in va])
    # Fixed decision rule; a validation tie selects failure (0).
    label = int(np.sum(validation_y) > len(validation_y)/2)
    test_y = np.concatenate([r["outcomes"][:,r["nominal_index"]] for r in te])
    majority = evaluator.binary_metrics(test_y, np.full(len(test_y), label), .5)
    pools = []
    for row in te:
        source = dict(row, ranking_scores=-np.arange(len(row["candidate_ids"]),dtype=float))
        pools.append(evaluator.screening_row(source,k,nominal=True))
    failed = [str(r["config_id"]) for r in test.get("setup_failures", [])]
    reached = {r["config_id"] for r in te}
    requested = {str(v) for v in test.get("requested_config_ids", sorted(reached | set(failed)))}
    if requested != reached | set(failed) or reached & set(failed):
        raise ValueError("requested/reached/setup-failed control denominator mismatch")
    summaries = evaluator.cluster_means(pools,
        ("feasible_hit","random_feasible_hit","feasible_precision","feasible_recall","quality","regret"),
        bootstrap_samples=2000)
    with_failures = pools + [dict(config_id=c,feasible_hit=0.,random_feasible_hit=0.) for c in failed]
    requested_summaries = evaluator.cluster_means(with_failures,("feasible_hit","random_feasible_hit"),bootstrap_samples=2000)
    return dict(majority=dict(chosen_label=label,selection="strict validation nominal majority; tie -> 0",
        validation_trials=len(validation_y),validation_positive_trials=int(validation_y.sum()),test=majority),
        source_order=dict(k=k,rows=pools,summary=summaries,requested_summary=requested_summaries,
            requested_configurations=len(requested),reached_configurations=len(reached),setup_failed_configurations=len(failed)),
        random=dict(method="exact uniform K-subsets without replacement",summary=summaries["random_feasible_hit"],
                    requested_summary=requested_summaries["random_feasible_hit"]))


def program_identity(plan):
    """Identity ignores labels/trace IDs, retaining exact physical program data."""
    indices={call["id"]:i for i,call in enumerate(plan["calls"])}
    calls=[]
    for call in plan["calls"]:
        args=copy.deepcopy(call["arguments"])
        for item in args.values():
            if item.get("source_call") is not None:
                item["source_call"]=indices[item["source_call"]]
        calls.append(dict(skill=call["skill"],arguments=args,roles=call["roles"],kind=call.get("kind","executable")))
    paths={name:{key:item[key] for key in JOINT_PATH_FIELDS}
           for name,item in plan["prefix"].get("initial_artifacts",{}).items()}
    return digest(dict(protocol=plan["protocol"],boundary=plan["boundary"],calls=calls,paths=paths)),digest(paths) if paths else None


def collection_audit(roots, protocol=None):
    directories=sorted({p.parent.resolve() for root in roots if Path(root).exists()
        for name in ("request.json","inputs.json","failure.json") for p in Path(root).rglob(name)
        if p.parent.name.startswith("group_")})
    rows=[]
    bindings={}
    for directory in directories:
        inp=read(directory/"inputs.json") if (directory/"inputs.json").exists() else None
        if inp and inp.get("schema")!="twingraph.group.v5":
            continue
        request=read(directory/"request.json") if (directory/"request.json").exists() else {}
        failure=read(directory/"failure.json") if (directory/"failure.json").exists() else None
        if not request and failure:
            request=failure.get("request",{})
        complete=read(directory/"complete.json") if (directory/"complete.json").exists() else None
        outcomes=read(directory/"outcomes.json") if (directory/"outcomes.json").exists() else {}
        trials=outcomes.get("trials",[])
        split=(inp or {}).get("declared_split",request.get("split","unknown"))
        seed=request.get("seed",(inp or {}).get("task",{}).get("seed"))
        if inp and outcomes and outcomes.get("input_sha256")!=digest(inp):
            raise ValueError(f"raw collection input/outcome binding mismatch: {directory}")
        plans=(inp or {}).get("candidates",[])
        ids=[p["id"] for p in plans]
        if len(ids)!=len(set(ids)) or any(row["candidate_id"] not in ids for row in trials):
            raise ValueError("raw collection candidate identity mismatch")
        keys=[program_identity(p) for p in plans]
        nominal=[t for t in trials if t["trial"]["repeat"]==(inp or {}).get("nominal_index",0)]
        failure_skills=Counter()
        for trial in trials:
            if trial.get("success"):
                continue
            failed=next((step for step in trial.get("executed_parameters") or [] if step.get("ok") is False),None)
            failure_skills[failed["skill"] if failed else "timeout" if trial.get("timeout") else "goal_or_untraced_failure"]+=1
        row=dict(directory=str(directory),split=split,seed=seed,
            status="completed" if complete else "failed" if failure else "incomplete",
            requested_candidates=request.get("n"),candidates=len(plans),trials=len(trials),
            nominal_trials=len(nominal),nominal_successes=sum(t.get("success") is True for t in nominal),
            timeouts=sum(t.get("timeout") is True for t in trials),
            setup_failed=bool(failure and inp is None),failed_after_inputs=bool(failure and inp is not None),
            exact_unique_programs=len({key for key,_ in keys}),
            exact_program_duplicates=len(plans)-len({key for key,_ in keys}),
            initial_trajectory_branches=len({key for _,key in keys if key is not None}),
            order_branches=len({tuple(p["prefix"].get("order",[])) for p in plans}),
            grasp_choice_branches=len({digest({name:{key:value for key,value in choice.items() if key in {"yaw","height"}}
                for name,choice in p["prefix"].get("choices",{}).items()}) for p in plans}),
            plan_lengths=sorted({len(p["calls"]) for p in plans}),first_failure_skills=dict(failure_skills),
            all_nominal_failure=bool(plans) and len(nominal)==len(plans) and all(t.get("valid") and not t.get("timeout") for t in nominal) and not any(t.get("success") for t in nominal),
            all_nominal_success=bool(plans) and len(nominal)==len(plans) and all(t.get("success") and t.get("valid") and not t.get("timeout") for t in nominal),
            known_geometry_conflicts=(inp or {}).get("pool_counts",{}).get("known_conflict"),
            solver_unknown=(inp or {}).get("pool_counts",{}).get("materialization_unknown"))
        rows.append(row)
        if inp and split in {"train","val","test"}:
            gid=str(inp["group_id"])
            if gid in bindings:
                raise ValueError("duplicate raw physical group across supplied data roots")
            bindings[gid]=dict(input_sha256=digest(inp),candidate_ids=ids)
    splits={}
    settings=(protocol or {}).get("collection",{})
    for split in sorted({r["split"] for r in rows}|{"train","val","test"}):
        group=[r for r in rows if r["split"]==split]
        expected=settings.get(("validation" if split=="val" else split)+"_seeds")
        actual={r["seed"] for r in group}
        splits[split]=dict(requested_observed=len(group),requested_prospective=len(expected) if expected is not None else None,
            missing_request_seeds=sorted(set(expected)-actual) if expected is not None else None,
            unexpected_request_seeds=sorted(actual-set(expected)) if expected is not None else None,
            completed=sum(r["status"]=="completed" for r in group),setup_failed=sum(r["setup_failed"] for r in group),
            failed_after_inputs=sum(r["failed_after_inputs"] for r in group),incomplete=sum(r["status"]=="incomplete" for r in group),
            candidates=sum(r["candidates"] for r in group),nominal_trials=sum(r["nominal_trials"] for r in group),
            nominal_successes=sum(r["nominal_successes"] for r in group),timeouts=sum(r["timeouts"] for r in group),
            exact_program_duplicates=sum(r["exact_program_duplicates"] for r in group),
            initial_trajectory_branch_range=[min(r["initial_trajectory_branches"] for r in group if r["candidates"]),max(r["initial_trajectory_branches"] for r in group if r["candidates"])] if any(r["candidates"] for r in group) else None,
            order_branch_counts=dict(Counter(r["order_branches"] for r in group if r["candidates"])),
            grasp_choice_branch_range=[min(r["grasp_choice_branches"] for r in group if r["candidates"]),max(r["grasp_choice_branches"] for r in group if r["candidates"])] if any(r["candidates"] for r in group) else None,
            plan_lengths=sorted({length for r in group for length in r["plan_lengths"]}),
            all_failure_pools=sum(r["all_nominal_failure"] for r in group),all_success_pools=sum(r["all_nominal_success"] for r in group),
            first_failure_skills=dict(sum((Counter(r["first_failure_skills"]) for r in group),Counter())))
    return dict(groups=rows,splits=splits,bindings=bindings,
        diversity_scope="Exact executable identity within each initial configuration; repeated disturbances and IDs do not create new plans. Branch counts do not demonstrate causal effects.")


def diagnostics(rows):
    records=[item for row in rows for item in row.get("input_diagnostics") or []]
    result=dict(candidates_with_diagnostics=len(records),scored_candidates=sum(len(row["candidate_ids"]) for row in rows))
    for field in ("unseen_fields","changed_training_constants","broken_duplicate_relations"):
        values=[int(item[field]) for item in records if field in item]
        result[field]=dict(candidates_observed=len(values),affected_candidates=sum(v>0 for v in values),
                           total=sum(values),maximum=max(values) if values else None)
    result["examples"]=list(dict.fromkeys(example for item in records for example in item.get("examples",[])))[:20]
    return result


def value_call_latency(rows):
    """One observation is one complete, resident-model candidate-pool call."""
    def stats(values):
        values=[float(value) for value in values if value is not None]
        if any(not np.isfinite(value) or value<0 for value in values):
            raise ValueError("value-call latency requires finite nonnegative measured times")
        return dict(observed_calls=len(values),mean_seconds=float(np.mean(values)) if values else None,
            median_seconds=float(np.median(values)) if values else None,
            p95_seconds=float(np.quantile(values,.95)) if values else None,
            min_seconds=min(values) if values else None,max_seconds=max(values) if values else None)
    return dict(candidate_counts=sorted({len(row["candidate_ids"]) for row in rows}),
        available_pool_rows=len(rows),physical_configurations=len({row["config_id"] for row in rows}),
        whole_call=stats([row.get("measured_rank_wall_seconds") for row in rows]),
        phases={phase:stats([(row.get("seconds") or {}).get(phase) for row in rows])
                for phase in ("graph_construction","encoding","inference","export","total")},
        scope="Resident model, one entire candidate pool per call. Whole-call wall includes rank invocation overhead; phase total overlaps other phases and must not be summed with them. Excludes model loading, process startup and offline LLM proxy generation; system acceleration comes only from paired actual pipeline clocks.")


def model_details(name,metadata,roots=()):
    paths=[]
    if metadata.get("path"):
        paths.append(Path(metadata["path"]).parent/"summary.json")
    paths.extend(Path(root)/name/"summary.json" for root in roots)
    present=list(dict.fromkeys(path.resolve() for path in paths if path.is_file()))
    if not present:
        return dict(status="pending_summary_detail",parameters=None,raw_input_dim=None,retained_input_dim=None,
                    searched=[str(path) for path in paths])
    records=[read(path) for path in present]
    if any(row.get("checkpoint_sha256")!=metadata.get("sha256") for row in records):
        raise ValueError(f"neighbor model summary checkpoint binding mismatch: {name}")
    dimensions=[(row.get("parameters"),row.get("raw_dim"),row.get("input_dim")) for row in records]
    if len(set(dimensions))!=1:
        raise ValueError(f"conflicting relocated model summary dimensions: {name}")
    parameters,raw,retained=dimensions[0]
    return dict(status="verified_summary_detail",parameters=parameters,raw_input_dim=raw,retained_input_dim=retained,
        sources=[dict(path=str(path),sha256=file_sha(path)) for path in present],
        input_scope="All typed program/state fields are flattened automatically; constant and duplicate columns are removed using training inputs only. Dimensions are data-derived, not an 87-feature manual list; optimality is not established.")


def system_audit(roots, protocol=None):
    paths=sorted({p.resolve() for root in roots if Path(root).exists() for p in Path(root).rglob("result.json")})
    rows=[read(path) for path in paths if read(path).get("schema")=="twingraph.system_run.v5"]
    if not rows:
        return dict(status="pending",rows=[],paired=[],methods={})
    groups={}
    for row in rows:
        target_trials=row.get("target_trials",[])
        executed=[r for r in target_trials if r.get("status")=="executed"]
        if row["simulation_calls"]!=dict(twin_validation=len(row.get("validation",[])),target_execution=len(executed)):
            raise ValueError("system call counters disagree with raw trial rows")
        if row["target_successes"]!=sum(r.get("success") is True for r in target_trials):
            raise ValueError("system target success counter disagrees with raw trial rows")
        for trial in executed:
            isolation=trial.get("isolation",{})
            if isolation.get("snapshot_transfer") is not False or not all(isolation.get(k) is True for k in
                ("distinct_session","distinct_context","distinct_model","distinct_data")):
                raise ValueError("independent target isolation evidence missing")
        key=(row["config_id"],row["group_id"],evaluator.digest(row["paired_namespace"]))
        groups.setdefault(key,[]).append(row)
    paired=[row for group in groups.values() if {r["policy"] for r in group}=={"full","top_k"} for row in group]
    analysis=evaluator.analyze_timing(dict(schema="twingraph.system_run.v5",rows=paired),bootstrap_samples=2000) if paired else None
    methods={}
    prospective=(protocol or {}).get("system",{})
    for name in sorted({r["method"] for r in rows}):
        subset=[r for r in rows if r["method"]==name]
        requested=sum(r["requested_target_trials"] for r in subset)
        successes=sum(r["target_successes"] for r in subset)
        expected=prospective.get("seeds")
        observed={r["config"]["seed"] for r in subset}
        methods[name]=dict(recorded_configurations=len(subset),requested_target_trials=requested,
            target_successes=successes,target_success_rate=successes/requested,
            prospective_configurations=len(expected) if expected is not None else None,
            missing_case_seeds=sorted(set(expected)-observed) if expected is not None else None,
            twin_validation_calls=sum(r["simulation_calls"]["twin_validation"] for r in subset),
            target_execution_calls=sum(r["simulation_calls"]["target_execution"] for r in subset),
            statuses=dict(Counter(r["status"] for r in subset)))
    cold_paths=sorted({p.resolve() for root in roots if Path(root).exists() for p in Path(root).rglob("timing_rows_cold_start.json")})
    cold_rows=[row for path in cold_paths for row in read(path).get("rows",[])]
    originals={(row["config_id"],row["method"]):row for row in rows}
    for cold in cold_rows:
        original=originals.get((cold["config_id"],cold["method"]))
        if original is None or cold["inputs_sha256"]!=original["inputs_sha256"] or digest(cold.get("target_trials",[]))!=digest(original.get("target_trials",[])):
            raise ValueError("cold-start sidecar differs from actual resident policy results")
        load=cold.get("model_initialization_wall_seconds")
        if not isinstance(load,(int,float)) or not np.isfinite(load) or load<0:
            raise ValueError("cold-start sidecar lacks measured model initialization")
        if any(not np.isclose(cold["seconds"][boundary],original["seconds"][boundary]+load,rtol=0,atol=1e-8) for boundary in ("total","decision")):
            raise ValueError("cold-start clock does not equal resident clock plus measured initialization")
    cold_groups={}
    for row in cold_rows:
        cold_groups.setdefault((row["config_id"],row["group_id"],evaluator.digest(row["paired_namespace"])),[]).append(row)
    cold_paired=[r for group in cold_groups.values() if {r["policy"] for r in group}=={"full","top_k"} for r in group]
    cold_analysis=evaluator.analyze_timing(dict(schema="twingraph.system_run.v5",rows=cold_paired),bootstrap_samples=2000) if cold_paired else None
    load_rows=[dict(config_id=row["config_id"],method=row["method"],seconds=row["model_initialization_wall_seconds"]) for row in cold_rows]
    batches=[]
    for path in sorted({p.resolve() for root in roots if Path(root).exists() for p in Path(root).rglob("system_batch.json")}):
        value=read(path)
        if value.get("schema")=="twingraph.system_batch.v5":
            batches.append(dict(path=str(path),sha256=file_sha(path),status=value["status"],
                actual_parallel_wall_seconds=value["wall_seconds"],denominator_summary=value["denominator_summary"],
                initialization_records=[{key:item[key] for key in ("seed","torch_import_wall_seconds","value_scorer_initialization_wall_seconds") if key in item}
                    for item in value.get("records",[])]))
    return dict(status="observed",rows=[dict(path=str(path),sha256=file_sha(path)) for path in paths],
                paired=analysis,methods=methods,recorded_policy_rows=len(rows),paired_policy_rows=len(paired),
                cold_start=dict(status="observed" if cold_paths else "not_supplied_optional",paired=cold_analysis,model_initialization=load_rows,
                    source_files=[dict(path=str(path),sha256=file_sha(path)) for path in cold_paths],
                    scope="Resident clocks plus separately measured model initialization; Python/Torch imports remain separate."),
                batch_records=batches,
                timing_scope="Actual separately executed paired clocks; no inferred N/K acceleration or claim about parallel batch wall time.")


def build_summary(*,protocol=None,validation=None,test=None,metrics=None,selection=None,thresholds=None,data_roots=(),system_roots=(),model_roots=()):
    pending=[]
    for name,value in (("protocol",protocol),("validation_predictions",validation),("test_predictions",test),
                       ("audited_test_metrics",metrics),("model_selection",selection),("threshold_freeze",thresholds)):
        if value is None:
            pending.append(name)
    chosen={p.get("selected") for p in (validation,test,selection,thresholds) if p and p.get("selected") is not None}
    if len(chosen)>1:
        raise ValueError("selected model differs across retained evidence")
    if metrics and test and metrics.get("predictions_sha256")!=evaluator.digest(test):
        raise ValueError("evaluator metrics do not bind supplied test predictions")
    if metrics and thresholds and metrics.get("freeze_sha256")!=evaluator.digest(thresholds):
        raise ValueError("evaluator metrics do not bind supplied threshold freeze")
    if thresholds and validation and thresholds.get("validation_sha256")!=evaluator.digest(validation):
        raise ValueError("threshold freeze does not bind supplied validation predictions")
    for payload in (validation,test):
        if payload and selection and payload.get("selection_metadata",{}).get("model_selection_sha256") not in {None,digest(selection)}:
            raise ValueError("prediction model-selection hash mismatch")
        if payload and protocol and payload.get("source_sha256") and payload["source_sha256"]!=protocol.get("source_sha256"):
            raise ValueError("prediction source differs from prospective protocol")
    raw=collection_audit(data_roots,protocol)
    if not raw["groups"]:pending.append("raw_collection_records")
    models={}
    checked={}
    binding_coverage={}
    for split,payload in (("validation",validation),("test",test)):
        if payload:
            checked[split]=evaluator.validate_predictions(payload,split)
            inspected=set();matched=set()
            for method in checked[split].values():
                for row in method["rows"]:
                    inspected.add(row["group_id"])
                    bound=raw["bindings"].get(row["group_id"])
                    if bound and (bound["candidate_ids"]!=row["candidate_ids"] or bound["input_sha256"]!=row.get("input_sha256")):
                        raise ValueError("prediction order/input hash differs from preserved physical input")
                    if bound:matched.add(row["group_id"])
            binding_coverage[split]=dict(prediction_groups=len(inspected),matched_raw_groups=len(matched),missing_raw_groups=sorted(inspected-matched))
            if inspected-matched:pending.append(split+"_raw_input_binding_checks")
    names=set((selection or {}).get("models",{}))|set((metrics or {}).get("methods",{}))
    names|={name for methods in checked.values() for name in methods}
    for name in sorted(names):
        model=(selection or {}).get("models",{}).get(name,{})
        report=(metrics or {}).get("methods",{}).get(name,{})
        section=report.get("strata",{}).get("overall",{})
        screening=section.get("screening",{}).get("nominal",{}).get("4",{})
        known=[x for x in (model.get("sha256"),report.get("model_sha256"),
            (thresholds or {}).get("methods",{}).get(name,{}).get("model_sha256")) if x]
        known.extend(methods[name]["model_sha256"] for methods in checked.values() if name in methods)
        if len(set(known))>1:
            raise ValueError(f"model hash mismatch: {name}")
        models[name]=dict(selected=name in chosen,model_sha256=known[0] if known else None,
            kind=model.get("kind"),seed=model.get("seed"),epoch=model.get("epoch"),
            validation_export_model_load_seconds=model.get("model_load_seconds"),
            model_details=model_details(name,{**model,"sha256":known[0] if known else model.get("sha256")},model_roots),
            validation_brier=model.get("validation_nominal_brier"),
            threshold=(thresholds or {}).get("methods",{}).get(name,{}).get("threshold"),
            classification_nominal=section.get("classification_nominal"),
            top4={k:v for k,v in screening.items() if k!="rows"},score_diagnostics=report.get("score_diagnostics"),
            input_diagnostics={split:diagnostics(methods[name]["rows"]) for split,methods in checked.items() if name in methods},
            value_call_latency={split:value_call_latency(methods[name]["rows"]) for split,methods in checked.items() if name in methods})
    baseline=controls(validation,test,4) if validation and test else None
    system=system_audit(system_roots,protocol)
    if system["status"]=="pending":pending.append("independent_system_results")
    elif system["paired_policy_rows"]!=system["recorded_policy_rows"]:
        pending.append("complete_full_topk_system_pairs")
    if any(v["missing_request_seeds"] or v["incomplete"] for v in raw["splits"].values()):
        pending.append("all_prospective_collection_requests_closed")
    if any(m.get("missing_case_seeds") for m in system["methods"].values()):
        pending.append("all_prospective_system_cases_closed")
    compact_protocol={k:v for k,v in (protocol or {}).items() if k not in {"sources","geometry_sources"}}
    return dict(schema="twingraph.value.report.v5",status="evidence_pending" if pending else "supplied_evidence_complete",
        pending=pending,protocol=compact_protocol,selected=next(iter(chosen)) if chosen else None,
        models=models,controls=baseline,disposition=(metrics or {}).get("disposition"),
        prediction_raw_binding_coverage=binding_coverage,
        collection={k:v for k,v in raw.items() if k!="bindings"},system=system,
        limitations=["LLM代理提出符号顺序/技能框架；任务编译器及抓取、运动求解器实例化物理程序与轨迹。",
            "输入为仿真器结构化位姿和几何，未评估视觉感知或图像输入。",
            "名义单次成功标签与扰动下真实成功概率不同；分类结论不等同于鲁棒性保证。",
            "首段自由运动已物化完整关节路径；后续依赖实测抓取的路径保留为延迟求解参数，不能宣称预先提供了全部未来真实轨迹。",
            "目标执行使用独立MuJoCo模型/数据和未装配初态；仍为仿真，未完成真机验证。",
            "终验覆盖五部件位姿、释放和末端退出，不包含滑动行程或长期装配质量验证。",
            "未证明所用表示最优、图结构优于其他表示，也未据此判定满足RAL录用要求。"])


def number(value,percent=False):
    return "待测/不适用" if value is None else f"{value*100:.1f}%" if percent else f"{value:.3f}"


def markdown(summary):
    lines=["# 滑台装配价值模块 v5 实验报告","",f"证据状态：{summary['status']}；验证集选定模型：{summary['selected'] or '尚未冻结'}。",""]
    if summary["pending"]:lines += ["尚缺："+"、".join(summary["pending"])+"。当前文件不构成已完成实验的结论。",""]
    lines += ["## 候选预测与筛选","","分类指标以成功生成候选的名义物理执行为分母；Hit4 表示至少保留一条成功候选。分类阈值固定自验证集，随机对照为无放回子集的精确期望。", "",
        "| 模型 | 验证Brier | 阈值 | 测试准确率 | 平衡准确率 | Hit4（达到候选池） | 随机Hit4 | Hit4（全部请求） |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name,row in summary["models"].items():
        c=(row["classification_nominal"] or {}).get("point",{})
        s=row["top4"].get("summary",{})
        lines.append("| "+" | ".join([name+("（选定）" if row["selected"] else ""),number(row["validation_brier"]),number(row["threshold"]),
            number(c.get("accuracy"),True),number(c.get("balanced_accuracy"),True),number(s.get("feasible_hit",{}).get("mean"),True),
            number(s.get("random_feasible_hit",{}).get("mean"),True),number(row["top4"].get("requested_feasible_hit",{}).get("mean"),True)])+" |")
    lines += ["","外部接口为结构化观测与完整类型化技能程序；字段按接口自动展开，仅用训练输入删除常量及重复列。下表为实际数据产生的维度，不是手工选择的87个特征；表示是否最优尚未验证。", "",
        "| 模型 | 参数量 | 自动展开原始维度 | 保留输入维度 | 元数据状态 |","|---|---:|---:|---:|---|"]
    for name,row in summary["models"].items():
        details=row["model_details"]
        lines.append(f"| {name} | {details['parameters'] if details['parameters'] is not None else '待补'} | {details['raw_input_dim'] if details['raw_input_dim'] is not None else '待补'} | {details['retained_input_dim'] if details['retained_input_dim'] is not None else '待补'} | {details['status']} |")
    lines += ["","价值调用延迟按模型已加载后的整池调用统计，正式协议每池 N=12；实际候选数及观测次数列于下表。下列数值单位均为毫秒，阶段括号内为该阶段的实际观测次数。内部 total 与各阶段重叠，不可相加；整体加速仍以独立运行的成对系统计时为准。", "",
        "| 模型/划分 | 实测N | 整池调用次数/可用池 | 整池均值 | 整池P95 | 建图均值(n) | 编码均值(n) | 推理均值(n) | 导出均值(n) | 内部total均值(n) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name,row in summary["models"].items():
        for split,latency in row["value_call_latency"].items():
            whole=latency["whole_call"]
            ms=lambda value:number(value*1000 if value is not None else None)
            phases=[f"{ms(latency['phases'][phase]['mean_seconds'])} ({latency['phases'][phase]['observed_calls']})"
                    for phase in ("graph_construction","encoding","inference","export","total")]
            lines.append("| "+" | ".join([f"{name}/{split}",str(latency["candidate_counts"]),
                f"{whole['observed_calls']}/{latency['available_pool_rows']}",ms(whole["mean_seconds"]),ms(whole["p95_seconds"]),*phases])+" |")
    control=summary["controls"]
    if control:
        m=control["majority"];s=control["source_order"]
        lines += ["",f"验证集多数类固定预测为 {m['chosen_label']}（验证正例 {m['validation_positive_trials']}/{m['validation_trials']}）；测试准确率 {number(m['test']['accuracy'],True)}，平衡准确率 {number(m['test']['balanced_accuracy'],True)}。",
            f"原始候选顺序 Top4 的 Hit4 为 {number(s['summary']['feasible_hit']['mean'],True)}；计入候选生成失败后的请求分母结果为 {number(s['requested_summary']['feasible_hit']['mean'],True)}。候选池达到 {s['reached_configurations']}/{s['requested_configurations']}，失败 {s['setup_failed_configurations']}。"]
    lines += ["","## 原始数据与候选多样性","","| 划分 | 已记录请求/预定 | 完成 | 输入前失败 | 未完成 | 候选 | 名义成功/执行 | 超时 | 完全重复程序 | 全失败池 | 全成功池 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for split,row in summary["collection"]["splits"].items():
        lines.append(f"| {split} | {row['requested_observed']}/{row['requested_prospective'] if row['requested_prospective'] is not None else '?'} | {row['completed']} | {row['setup_failed']} | {row['incomplete']} | {row['candidates']} | {row['nominal_successes']}/{row['nominal_trials']} | {row['timeouts']} | {row['exact_program_duplicates']} | {row['all_failure_pools']} | {row['all_success_pools']} |")
    for split,row in summary["collection"]["splits"].items():
        if row["candidates"]:
            lines += ["",f"{split} 每池初始轨迹分支范围 {row['initial_trajectory_branch_range']}，完整抓取选择分支范围 {row['grasp_choice_branch_range']}，程序长度 {row['plan_lengths']}；首次失败技能计数 {json.dumps(row['first_failure_skills'],ensure_ascii=False)}。"]
    lines += ["","多样性按同一初态下的实际调用、物理参数和完整初始轨迹检查，剔除候选ID与来源标签；轨迹、顺序、抓取分支及首次失败技能逐池记录于 summary.json。中途失败和开发试验不删除，也不混作测试数据。", "","## 独立目标执行与实际时间",""]
    if not summary["system"]["methods"]:
        lines.append("尚无独立目标执行结果；不填入预计成功率或按 N/K 推算速度。")
    for name,row in summary["system"]["methods"].items():
        lines.append(f"{name}：目标成功 {row['target_successes']}/{row['requested_target_trials']}，实际目标调用 {row['target_execution_calls']}，孪生验证调用 {row['twin_validation_calls']}；已记录 {row['recorded_configurations']} 个配置。")
    paired=summary["system"].get("paired")
    if paired:
        for name,row in paired["methods"].items():
            s=row["summary"]
            lines += ["",f"{name} 成对决策时间：全量 {number(s.get('decision_full_seconds',{}).get('mean'))} 秒，筛选 {number(s.get('decision_screened_seconds',{}).get('mean'))} 秒；实际成对时间比均值 {number(s.get('decision_speedup',{}).get('mean'))}。",
                f"场景创建至独立执行和终态渲染（不含离线LLM代理生成与进程启动）：全量 {number(s.get('total_full_seconds',{}).get('mean'))} 秒，筛选 {number(s.get('total_screened_seconds',{}).get('mean'))} 秒。均值仅来自完成相同候选预算的成对配置；详细分母和置信区间见 summary.json。"]
        lines += ["","上述系统时间遵循原始 timing_boundaries；模型加载在调用前已完成，提前准备的LLM代理规划记录及进程启动均不在计时内，也不代表多进程批次的总墙钟时间。少量配置的退化置信区间不构成可靠性保证。"]
    cold=summary["system"].get("cold_start",{})
    if cold.get("paired"):
        loads=[row["seconds"] for row in cold["model_initialization"] if row["method"]=="top_k"]
        lines += ["",f"另行记录的模型初始化平均耗时为 {number(float(np.mean(loads)) if loads else None)} 秒。"]
        for name,row in cold["paired"]["methods"].items():
            s=row["summary"]
            lines.append(f"计入该实测初始化后，{name} 冷启动决策时间均值 {number(s.get('decision_screened_seconds',{}).get('mean'))} 秒，全量对照 {number(s.get('decision_full_seconds',{}).get('mean'))} 秒；Python/Torch 导入仍单独记录，不混入该值。")
    for batch in summary["system"].get("batch_records",[]):
        lines += ["",f"批次记录的实际并行墙钟时间为 {number(batch['actual_parallel_wall_seconds'])} 秒，状态 {batch['status']}。"]
        for name,row in batch["denominator_summary"].items():
            lines.append(f"{name} 按预定全部请求计为 {row['independent_target_successes']}/{row['requested_target_trials']}；缺失策略结果 {row['missing_policy_results']}。缺失/程序错误属于未解决请求，不冒充物理负例标签。")
    lines += ["","## 输入范围与结论边界",""]
    for name,row in summary["models"].items():
        d=row["input_diagnostics"].get("test")
        if d:
            lines.append(f"{name} 测试输入：诊断 {d['candidates_with_diagnostics']}/{d['scored_candidates']} 条；未见字段影响 {d['unseen_fields']['affected_candidates']} 条，训练常量变化影响 {d['changed_training_constants']['affected_candidates']} 条，重复字段关系变化影响 {d['broken_duplicate_relations']['affected_candidates']} 条。")
    lines += [""]+["- "+text for text in summary["limitations"]]
    lines += ["","![预测、筛选与独立执行](overview.png)",""]
    return "\n".join(lines)


def plot(summary,path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(16,4.8),constrained_layout=True)
    names=[];accuracy=[];balanced=[];hit=[];random=[];accuracy_ci=[];balanced_ci=[];hit_ci=[];random_ci=[]
    for name,row in summary["models"].items():
        point=(row["classification_nominal"] or {}).get("point",{})
        if "accuracy" not in point:continue
        names.append(name+(" *" if row["selected"] else ""));accuracy.append(point["accuracy"])
        balanced.append(point.get("balanced_accuracy"));stats=row["top4"].get("summary",{})
        hit.append(stats.get("feasible_hit",{}).get("mean"));random.append(stats.get("random_feasible_hit",{}).get("mean"))
        ci=(row["classification_nominal"] or {}).get("bootstrap95",{})
        accuracy_ci.append(ci.get("accuracy"));balanced_ci.append(ci.get("balanced_accuracy"))
        hit_ci.append(stats.get("feasible_hit",{}).get("bootstrap95"));random_ci.append(stats.get("random_feasible_hit",{}).get("bootstrap95"))
    def intervals(ax,positions,values,bounds):
        for x,value,bound in zip(positions,values,bounds):
            if value is not None and bound is not None:
                # Percentile bootstrap bounds need not straddle the point estimate.
                ax.vlines(x,bound[0],bound[1],color="#202020",linewidth=1)
                ax.hlines(bound,x-.035,x+.035,color="#202020",linewidth=1)
    if names:
        x=np.arange(len(names));axes[0].bar(x-.18,accuracy,.36,label="Accuracy")
        axes[0].bar(x+.18,[np.nan if v is None else v for v in balanced],.36,label="Balanced accuracy")
        intervals(axes[0],x-.18,accuracy,accuracy_ci);intervals(axes[0],x+.18,balanced,balanced_ci)
        if summary["controls"]:axes[0].axhline(summary["controls"]["majority"]["test"]["accuracy"],color="black",ls="--",label="Validation-majority accuracy")
        axes[0].set_xticks(x,names,rotation=20);axes[0].legend(fontsize=8)
        axes[1].bar(x-.18,[np.nan if v is None else v for v in hit],.36,label="Model Hit4")
        axes[1].bar(x+.18,[np.nan if v is None else v for v in random],.36,label="Exact random Hit4")
        intervals(axes[1],x-.18,hit,hit_ci);intervals(axes[1],x+.18,random,random_ci)
        if summary["controls"]:axes[1].axhline(summary["controls"]["source_order"]["summary"]["feasible_hit"]["mean"],color="black",ls="--",label="Source-order Hit4")
        axes[1].set_xticks(x,names,rotation=20);axes[1].legend(fontsize=8)
    else:
        for ax in axes[:2]:ax.text(.5,.5,"Audited test results pending",ha="center",va="center",transform=ax.transAxes)
    axes[0].set(title="Nominal candidate classification",ylim=(0,1.08),ylabel="Rate")
    axes[1].set(title="Top4 screening: reached pools",ylim=(0,1.08),ylabel="Hit probability")
    paired=summary["system"].get("paired")
    entries=[]
    if paired:
        for name,row in paired["methods"].items():
            stats=row["summary"]
            if stats.get("decision_full_seconds",{}).get("mean") is not None:
                entries.append((name,stats))
    if entries:
        labels=[];values=[]
        for name,stats in entries:
            labels.extend(["Full",name]);values.extend([stats["decision_full_seconds"]["mean"],stats["decision_screened_seconds"]["mean"]])
        bars=axes[2].bar(np.arange(len(values)),values,color=["#64748b","#0f766e"]*len(entries))
        for bar,value in zip(bars,values):axes[2].text(bar.get_x()+bar.get_width()/2,value,f"{value:.1f}s",ha="center",va="bottom",fontsize=8)
        axes[2].set_xticks(np.arange(len(labels)),labels,rotation=20)
        rates="; ".join(f"{name}: {r['target_successes']}/{r['requested_target_trials']} target successes" for name,r in summary["system"]["methods"].items())
        axes[2].text(.5,-.25,rates,ha="center",va="top",transform=axes[2].transAxes,fontsize=8,wrap=True)
    else:
        axes[2].text(.5,.5,"Paired physical timing pending",ha="center",va="center",transform=axes[2].transAxes)
        axes[2].set_xticks([]);axes[2].set_yticks([])
    axes[2].set(title="Decision time (model already loaded)",ylabel="Seconds")
    fig.suptitle("Five-part assembly v5 | simulation | * validation-selected | whiskers: configuration bootstrap 95%",fontsize=12)
    fig.savefig(path,dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",default="results/v5")
    for name in ("protocol","validation","test","metrics","selection","thresholds"):
        parser.add_argument("--"+name)
    parser.add_argument("--data",nargs="+")
    parser.add_argument("--systems",nargs="+")
    parser.add_argument("--models",nargs="+",help="Optional relocated model roots containing NAME/summary.json")
    parser.add_argument("--out")
    args=parser.parse_args();run=Path(args.run)
    defaults=dict(protocol=Path("experiments/value_v5/protocol.json"),validation=run/"evidence/validation_predictions.json",
        test=run/"evidence/test_predictions.json",metrics=run/"evidence/test_metrics/metrics.json",
        selection=run/"evidence/model_selection.json",thresholds=run/"evidence/thresholds.json")
    values={};files={}
    for name,default in defaults.items():
        path=Path(getattr(args,name) or default)
        values[name]=read(path) if path.exists() else None
        files[name]=dict(path=str(path.resolve()),present=path.exists(),sha256=file_sha(path) if path.exists() else None)
    summary=build_summary(**values,data_roots=args.data or [run/"data"],system_roots=args.systems or [run/"systems"],model_roots=args.models or [run/"models"])
    summary["input_files"]=files;summary["report_script_sha256"]=file_sha(__file__)
    output=Path(args.out or run/"report");output.mkdir(parents=True,exist_ok=True)
    (output/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    (output/"report.md").write_text(markdown(summary),encoding="utf-8")
    plot(summary,output/"overview.png")
    print(json.dumps(dict(status=summary["status"],out=str(output.resolve()),pending=summary["pending"]),ensure_ascii=False))


if __name__=="__main__":main()
