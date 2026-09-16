"""Read one explicitly selected recorded v5 case; never score or simulate.

The JSON preserves both policies' complete recorded outcomes. The Chinese
walkthrough explains the TopK policy in detail without using it as a population
performance estimate. Only Python's standard library is required.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


PARTS = dict(carriage="滑块", end_stop="端挡", pin_left="左销", pin_right="右销", handle="手柄")


def digest(value):
    # Same canonical JSON convention as simbench.value.plan.digest, without
    # importing any value/scoring/physical execution module.
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def passed(row):
    return bool(row.get("success") and row.get("valid") and not row.get("timeout"))


def physical_summary(plan):
    prefix = plan.get("prefix", {})
    paths = {}
    for name, item in prefix.get("initial_artifacts", {}).items():
        paths[name] = dict(artifact_sha256=digest(item), type=item.get("type"), part=item.get("part"),
            waypoints=len(item.get("joints", [])), joint_dof=len(item.get("start_q", [])),
            start_q_rad=item.get("start_q"), target_world_m=item.get("target"),
            rotation_world=item.get("rotation"), binding=item.get("binding"),
            geometry_sha256=digest({k: item.get(k) for k in ("type", "part", "start_q", "joints", "target", "rotation")}))
    calls = plan.get("calls", [])
    arguments = [value for call in calls for value in call.get("arguments", {}).values()]
    return dict(order=prefix.get("order", []), choices=prefix.get("choices", {}),
        choice_units=dict(yaw="rad", height="m", clearance="m", force="N", speed="m/s"),
        initial_route_index=prefix.get("initial_route_index"), semantic_program_id=prefix.get("semantic_program_id"),
        task_scope=prefix.get("task_scope"), targets_world_m=prefix.get("targets"),
        protocol=plan.get("protocol"), call_count=len(calls),
        skill_counts=dict(Counter(call.get("skill") for call in calls)),
        argument_status_counts=dict(Counter(value.get("status", "known") for value in arguments)),
        initial_paths=paths, plan_sha256=digest(plan))


def trial_summary(rows):
    return dict(observed_trials=len(rows), successful_trials=sum(passed(row) for row in rows),
        timeouts=sum(bool(row.get("timeout")) for row in rows),
        trials=[dict(repeat=row.get("trial", {}).get("repeat"), trial=row.get("trial"),
            trial_sha256=row.get("trial_sha256"), success=passed(row), recorded_success=row.get("success"),
            valid=row.get("valid"), timeout=row.get("timeout"), error=row.get("error"),
            goal_check=row.get("goal_check"), executed_steps=row.get("executed_steps"),
            wall_seconds=row.get("wall_seconds"), sim_seconds=row.get("sim_seconds"),
            first_failed_step=next((dict(index=i + 1, **step) for i, step in
                enumerate(row.get("executed_parameters") or []) if not step.get("ok")), None)) for row in rows])


def build_walkthrough(case, replay=None):
    case = Path(case).resolve()
    require(case.is_dir(), "explicit --case directory does not exist")
    files, checks, warnings = [], [], []

    def recorded(path):
        value = read(path)
        files.append(dict(path=str(Path(path).resolve()), file_sha256=file_sha(path), canonical_sha256=digest(value)))
        return value

    policies = {}
    inputs_by_policy = {}
    planners = {}
    for method in ("top_k", "full"):
        directory = case / method
        result_path = directory / "result.json"
        if not result_path.exists():
            warnings.append(f"{method}: result.json 缺失，不能推断该策略完成或成功")
            policies[method] = dict(status="missing_record", recorded_result=None)
            continue
        result = recorded(result_path)
        require(result.get("schema") == "twingraph.system_run.v5", f"{method}: unsupported result schema")
        require(result.get("method") == method, f"{method}: result method mismatch")
        planner = recorded(directory / "planner.json")
        require(digest(planner) == result.get("planner_sha256"), f"{method}: planner hash mismatch")
        planners[method] = planner
        checks.append(f"{method}: planner hash binds recorded task and proposed_orders")
        source_path = directory / "source.json"
        if source_path.exists():
            require(digest(recorded(source_path)) == result.get("implementation_sha256"), f"{method}: source hash mismatch")
        else:
            warnings.append(f"{method}: source.json 缺失，未核验源代码清单绑定")
        inputs_path = directory / "inputs.json"
        ids = []
        if inputs_path.exists():
            inp = recorded(inputs_path)
            require(inp.get("schema") == "twingraph.system_inputs.v5", f"{method}: unsupported input schema")
            require(digest(inp) == result.get("inputs_sha256"), f"{method}: input hash mismatch")
            require(inp.get("planner_sha256") == result.get("planner_sha256"), f"{method}: input/planner binding mismatch")
            ids = [plan["id"] for plan in inp["candidates"]]
            require(len(ids) == len(set(ids)) == result.get("actual_n"), f"{method}: candidate count or identity mismatch")
            require(all(plan.get("prefix", {}).get("order") in planner["proposed_orders"] for plan in inp["candidates"]),
                    f"{method}: executable order is absent from saved proposed_orders")
            inputs_by_policy[method] = inp
            checks.append(f"{method}: exact input hash, original candidate order and consumed proposed_orders")
        else:
            require(not result.get("inputs_sha256") and not result.get("actual_n"), f"{method}: materialized input artifact missing")
        ranking = result.get("ranking")
        if ranking is not None:
            require(method == "top_k" and ids, f"{method}: ranking without candidate input")
            require(len(ranking.get("scores", [])) == len(ids) == len(ranking.get("logits", [])), "ranking vector size mismatch")
            require(all(isinstance(v, (int, float)) and math.isfinite(v) for v in ranking["logits"]), "nonfinite recorded logits")
            require(all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1 for v in ranking["scores"]), "invalid recorded probability")
            require(len(ranking.get("order", [])) == len(ids) and set(ranking["order"]) == set(ids), "recorded ranking IDs mismatch")
            require(ranking.get("top_k") == ranking["order"][:min(result["k"], len(ids))], "recorded TopK/order mismatch")
            require(result.get("validated_candidate_ids") == [cid for cid in ids if cid in ranking["top_k"]],
                    "recorded TopK validation membership/source order mismatch")
            checks.append("top_k: saved ranks/probabilities read directly; no model invocation or reranking")
        elif method == "top_k" and ids:
            warnings.append("top_k: 没有保存价值排序，概率与名次保持缺失")
        validation = result.get("validation", [])
        keys = [(row.get("candidate_id"), row.get("trial_sha256")) for row in validation]
        require(len(set(keys)) == len(keys), f"{method}: duplicate recorded validation trial")
        require(result.get("simulation_calls", {}).get("twin_validation") == len(validation), f"{method}: validation call count mismatch")
        require(set(row.get("candidate_id") for row in validation) == set(result.get("validated_candidate_ids", [])),
                f"{method}: declared validated IDs differ from recorded trials")
        for row in validation:
            require(row.get("candidate_id") in ids, f"{method}: validation references unknown candidate")
            require(digest(row.get("trial")) == row.get("trial_sha256"), f"{method}: trial hash mismatch")
            require(row.get("trial", {}).get("namespace") == result.get("paired_namespace", {}).get("twin"),
                    f"{method}: validation namespace mismatch")
        selected = result.get("selected_candidate_id")
        require(selected is None or selected in result.get("validated_candidate_ids", []), f"{method}: selected candidate was not validated")
        for summary in result.get("candidate_validation", []):
            rows = [row for row in validation if row["candidate_id"] == summary["candidate_id"]]
            require(summary.get("trials") == len(rows) and summary.get("successful_trials") == sum(passed(row) for row in rows),
                    f"{method}: candidate validation summary mismatch")
        targets = result.get("target_trials", [])
        require(len({row.get("repeat") for row in targets}) == len(targets), f"{method}: duplicate target repeat")
        require(len(targets) <= result.get("requested_target_trials", 0), f"{method}: too many target trials")
        require(result.get("target_successes") == sum(bool(row.get("success")) for row in targets), f"{method}: target success count mismatch")
        require(result.get("simulation_calls", {}).get("target_execution") == sum(row.get("status") == "executed" for row in targets),
                f"{method}: target execution count mismatch")
        for target in targets:
            if target.get("status") != "executed":
                continue
            execution = target["execution"]
            require(target.get("selected_candidate_id") == selected, f"{method}: target selected candidate mismatch")
            require(target.get("rebound_candidate_id") == execution.get("candidate_id"), f"{method}: rebound identity mismatch")
            require(target.get("success") == passed(execution), f"{method}: target success/timeout mismatch")
            require(target.get("semantic_sha256") == result.get("selected_semantic_sha256"), f"{method}: target semantic binding mismatch")
            require(digest(execution.get("trial")) == execution.get("trial_sha256"), f"{method}: target trial hash mismatch")
            require(execution.get("trial", {}).get("namespace") == result.get("paired_namespace", {}).get("target")
                    != result.get("paired_namespace", {}).get("twin"), f"{method}: target/twin namespace isolation mismatch")
            isolation = target.get("isolation", {})
            require(all(isolation.get(k) is True for k in ("distinct_session", "distinct_context", "distinct_model", "distinct_data"))
                    and isolation.get("snapshot_transfer") is False, f"{method}: independent scene evidence missing")
            require(execution.get("executed_steps") == len(execution.get("executed_parameters") or []), f"{method}: target step trace count mismatch")
            sidecar = directory / f"target_execution_{target['repeat']}.json"
            if sidecar.exists():
                require(recorded(sidecar) == target, f"{method}: target execution sidecar mismatch")
            else:
                warnings.append(f"{method}: 目标重复 {target['repeat']} 的独立 sidecar 缺失")
            target_input_path = directory / f"target_input_{target['repeat']}.json"
            if target_input_path.exists():
                bound = recorded(target_input_path)
                require(digest(bound.get("graph")) == execution.get("input_graph_sha256"), f"{method}: executed target graph hash mismatch")
                require(bound.get("selected_candidate_id") == selected and bound.get("semantic_sha256") == target.get("semantic_sha256"),
                        f"{method}: target input binding mismatch")
            else:
                warnings.append(f"{method}: 目标重复 {target['repeat']} 的输入文件缺失，未核验图文件")
        if any(target.get("status") == "executed" for target in targets):
            checks.append(f"{method}: all executed target identities, semantics, trial namespaces, isolation and trace lengths")
        terminal_evidence = []
        for item in result.get("images", []):
            if not item.get("path"):
                terminal_evidence.append(item)
                continue
            matches = [t for t in targets if t.get("status") == "executed"
                       and t["execution"].get("candidate_id") == item.get("candidate_id")
                       and t["execution"].get("trial_sha256") == item.get("trial_sha256")
                       and t["execution"].get("input_graph_sha256") == item.get("graph_sha256")]
            require(len(matches) == 1 and item.get("source") == "actual_target_terminal_state"
                    and item.get("no_replay") is True, f"{method}: terminal image execution binding mismatch")
            # Relative original paths and copied artifacts can both resolve here.
            local_image = directory / f"target_terminal_{matches[0]['repeat']}.png"
            if local_image.exists():
                require(file_sha(local_image) == item.get("sha256"), f"{method}: terminal image hash mismatch")
                files.append(dict(path=str(local_image), file_sha256=file_sha(local_image)))
                terminal_evidence.append(dict(item, resolved_path=str(local_image)))
            else:
                terminal_evidence.append(dict(item, resolved_path=None))
                warnings.append(f"{method}: 终态图像 {item['path']} 未找到，只保留元数据")
        policies[method] = dict(status=result.get("status"), recorded_result=result, terminal_evidence=terminal_evidence)
    require(planners, "no completed policy result exists in the explicitly selected case")
    if len(planners) == 2:
        require(digest(planners["top_k"]) == digest(planners["full"]), "paired planner records differ")
    if len(inputs_by_policy) == 2:
        require(digest(inputs_by_policy["top_k"]) == digest(inputs_by_policy["full"]), "paired inputs differ")
        checks.append("both policies: identical saved candidate pool and observation")
    pair = recorded(case / "pair.json") if (case / "pair.json").exists() else None
    if pair and inputs_by_policy:
        require(pair.get("input_sha256") == digest(next(iter(inputs_by_policy.values()))), "pair input hash mismatch")
    primary_inputs = inputs_by_policy.get("top_k", inputs_by_policy.get("full", {}))
    topk = policies["top_k"].get("recorded_result") or {}
    ranking = topk.get("ranking") or {}
    candidates = []
    for index, plan in enumerate(primary_inputs.get("candidates", [])):
        cid = plan["id"]
        item = dict(source_index=index + 1, candidate_id=cid, physical=physical_summary(plan),
            frozen_probability=ranking.get("scores", [None] * len(primary_inputs["candidates"]))[index],
            frozen_logit=ranking.get("logits", [None] * len(primary_inputs["candidates"]))[index],
            frozen_rank=ranking["order"].index(cid) + 1 if cid in ranking.get("order", []) else None,
            top_k_member=cid in ranking["top_k"] if ranking else None, policies={})
        for method, record in policies.items():
            result = record.get("recorded_result") or {}
            rows = [row for row in result.get("validation", []) if row["candidate_id"] == cid]
            item["policies"][method] = dict(trial_summary(rows),
                selected=result.get("selected_candidate_id") == cid,
                recorded_summary=next((r for r in result.get("candidate_validation", []) if r["candidate_id"] == cid), None))
        candidates.append(item)
    replay_record = None
    if replay is not None:
        replay_path = Path(replay).resolve()
        metadata = recorded(replay_path)
        require(metadata.get("schema") == "twingraph.recorded_target_replay.v5", "unsupported replay schema")
        require(metadata.get("simulation_integration_steps") == 0, "replay declares new physical integration")
        matches = [row for row in topk.get("target_trials", []) if row.get("status") == "executed"
                   and row.get("selected_candidate_id") == metadata.get("selected_candidate_id")
                   and row.get("rebound_candidate_id") == metadata.get("rebound_candidate_id")
                   and row["execution"].get("trial") == metadata.get("recorded_trial")]
        require(len(matches) == 1, "replay is not the selected TopK target trial of this explicit case")
        target = matches[0]
        require(file_sha(case / "top_k" / f"target_execution_{target['repeat']}.json") == metadata.get("execution_sha256"),
                "replay target execution file hash mismatch")
        require(metadata.get("trace_sha256") == target["execution"].get("state_trace", {}).get("sha256"), "replay trace hash mismatch")
        require(metadata.get("recorded_success") == target.get("success"), "replay success disagrees with recorded execution")
        video = replay_path.with_suffix(".mp4")
        require(video.exists() and file_sha(video) == metadata.get("video_sha256"), "replay video missing or hash mismatch")
        files.append(dict(path=str(video), file_sha256=file_sha(video)))
        replay_record = dict(metadata_path=str(replay_path), video_path=str(video), metadata=metadata)
        checks.append("video: selected TopK target execution/trial/trace and video hash; no new rollout")
    return dict(schema="twingraph.case_walkthrough.v5", explicitly_selected_case=str(case),
        interpretation="posthoc explanation of one user-selected recorded case; not a population performance estimate or a new experiment",
        generation=dict(simulation_calls=0,model_calls=0,training_calls=0,reranked=False,
                        generator_sha256=file_sha(__file__)),
        planner=planners.get("top_k", next(iter(planners.values()))), observation=primary_inputs.get("observation"),
        candidates=candidates, policies=policies, pair=pair, replay=replay_record,
        integrity=dict(checks=checks,warnings=warnings,checked_files=files),
        limitations=["案例由调用者显式指定；所有预注册案例及失败必须保留在总体报告中，不能用本案例估计性能。",
            "LLM 代理只提供任务/符号顺序；stage_calls 和几何/运动求解器实例化物理程序，不是 LLM 计算关节轨迹。",
            "保存的 proposed_orders 被编译器消费；proposed_skill_programs 是保留的规划来源说明，未逐条作为可执行程序解析。",
            "初始自由运动轨迹已物化；依赖实际抓取的后续轨迹仍在执行时求解。",
            "价值输入使用结构化仿真观测和接口参数，没有相机图像；尚未证明输入表示最优或图表示优于序列。",
            "独立目标是另一个 MuJoCo 场景，未验证真实机器人；五部件装配终验不包含滑台功能行程试验。",
            "若提供视频，它仅重放所选目标的已记录状态，不增加成功样本，也不是新的物理执行。"])


def cell(value):
    if value is None:
        return "未记录"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value).replace("|", "\\|").replace("\n", " ")


def order_text(order):
    return " → ".join(PARTS.get(part, part) for part in order)


def markdown(summary):
    planner = summary["planner"]
    lines = ["# 滑台装配完整流程：指定记录案例", "",
        f"案例：`{summary['explicitly_selected_case']}`。这是一次已有记录的讲解；没有重新采样候选、重新排序或重新仿真。", "",
        "本文件保留该案例 TopK 与全量验证两种策略的全部候选结果和目标试验。总体性能请以所有预注册案例的汇总报告为准。", "",
        "## 1. 任务与规划来源", "", planner.get("task_text", "未记录"), "",
        f"来源：{cell(planner.get('source'))}；提供方：{cell(planner.get('provider'))}；模型：{cell(planner.get('model'))}。", "",
        "保存的 `proposed_orders` 被 `stage_v5.build_pool` 实际消费：", ""]
    lines += [f"{i + 1}. {order_text(order)}" for i, order in enumerate(planner.get("proposed_orders", []))]
    lines += ["", "`proposed_skill_programs` 保留在来源记录中；当前编译器消费的是上述部件顺序。`stage_calls` 展开各部件的原子技能与检查调用，几何/运动求解器绑定抓取、控制参数与初始关节路径。实际调用数见下表（正式五部件计划通常为 101 条）。LLM 代理没有计算关节轨迹。", ""]
    goals = (summary.get("observation") or {}).get("goals", [])
    if goals:
        lines += ["声明的最终任务验收谓词与阈值（物理条件，不是加权奖励）：", "",
            "|部件|谓词|目标世界位置 (m)|位置容差 (m)|倾斜容差 (deg)|最低末端净空 (m)|", "|---|---|---|---:|---:|---:|"]
        for goal in goals:
            lines.append("|" + "|".join(cell(v) for v in [PARTS.get(goal.get("manipulated"), goal.get("manipulated")),
                goal.get("predicate"), goal.get("position"), goal.get("position_tolerance", .0015),
                goal.get("tilt_tolerance_deg", 3.), goal.get("minimum_eef_clearance_m", 0.)]) + "|")
        lines += ["", "`seated_released_retracted` 同时要求位置/倾斜通过、释放抓持、没有手指接触且末端已退让。", ""]
    lines += [
        "## 2. 实际候选与冻结价值输出", "",
        "下表保持候选原始输入顺序，名次和概率直接读取冻结模型记录。概率是模型估计，并不等于该候选已通过物理验证。TopK 未验证项不记为失败。", "",
        "|输入序号|候选 ID|部件顺序|调用数|冻结概率|原始 logit|记录名次|TopK|TopK 验证|全量验证|",
        "|---:|---|---|---:|---:|---:|---:|---|---|---|"]
    for candidate in summary["candidates"]:
        def outcome(method):
            row = candidate["policies"][method]
            text = f"{row['successful_trials']}/{row['observed_trials']}，超时 {row['timeouts']}" if row["observed_trials"] else "未验证"
            return text + ("；被选中" if row["selected"] else "")
        values = [candidate["source_index"], candidate["candidate_id"], order_text(candidate["physical"]["order"]),
            candidate["physical"]["call_count"], candidate["frozen_probability"], candidate["frozen_logit"],
            candidate["frozen_rank"], candidate["top_k_member"], outcome("top_k"), outcome("full")]
        lines.append("|" + "|".join(cell(v) for v in values) + "|")
    topk = summary["policies"]["top_k"].get("recorded_result") or {}
    ranking = topk.get("ranking") or {}
    lines += ["", f"冻结检查点 SHA256：`{ranking.get('checkpoint_sha256', '未记录')}`。", "",
        "物理参数是已执行接口的取值摘要，不是人为加权评分。`force` 是夹持力；接触插入/压靠的其他固定参数仍完整保存在原始 calls 中。初始轨迹摘要不替代输入中的全部路点。", ""]
    for candidate in summary["candidates"]:
        physical = candidate["physical"]
        lines += [f"### 候选 {candidate['source_index']}：`{candidate['candidate_id']}`", "",
            f"初始路线索引：{cell(physical['initial_route_index'])}；已知/延后参数计数：{cell(physical['argument_status_counts'])}。", "",
            "|部件|yaw (rad)|height (m)|clearance (m)|force (N)|speed (m/s)|", "|---|---:|---:|---:|---:|---:|"]
        for part in physical["order"]:
            choice = physical["choices"].get(part, {})
            lines.append("|" + "|".join(cell(v) for v in [PARTS.get(part, part), *[choice.get(k) for k in ("yaw", "height", "clearance", "force", "speed")]]) + "|")
        for name, path in physical["initial_paths"].items():
            lines += ["", f"初始轨迹 `{name}`：{path['waypoints']} 个路点 × {path['joint_dof']} 关节；世界坐标目标 (m)：{cell(path['target_world_m'])}；几何 SHA256：`{path['geometry_sha256']}`。"]
        lines += ["", "|策略|重复编号|成功|超时|已执行调用|错误|", "|---|---:|---|---|---:|---|"]
        for method, record in candidate["policies"].items():
            for row in record["trials"]:
                lines.append("|" + "|".join(cell(v) for v in [method, row["repeat"], row["success"], row["timeout"], row["executed_steps"], row["error"] or "无"]) + "|")
        lines.append("")
    lines += ["## 3. 孪生选择与独立目标执行", "",
        "先验证策略提交的候选，再按观测成功比例从高到低选择；并列时按原始候选输入顺序。至少一次有效、未超时成功，且比例达到记录中的 accept_rate 才能接受。超时保留在分母中。这里仅展示系统已经作出的选择。", ""]
    for method, record in summary["policies"].items():
        result = record.get("recorded_result")
        lines += [f"### 策略 `{method}`", "", f"状态：`{record['status']}`。", ""]
        if result is None:
            continue
        lines += [f"请求 N={result.get('n')}，实际 N={result.get('actual_n', 0)}，K={result.get('k')}；真实孪生调用 {result.get('simulation_calls', {}).get('twin_validation', 0)} 次；接受比例阈值 {cell(result.get('config', {}).get('accept_rate'))}。", "",
            f"选中候选：`{result.get('selected_candidate_id') or '未选出'}`；独立目标成功 {result.get('target_successes', 0)}/{result.get('requested_target_trials')}（请求目标试验为分母，包含设置/重绑定失败与未执行）。", ""]
        for key in ("setup_error", "candidate_generation_error"):
            if result.get(key):
                lines += [f"{key}：{cell(result[key])}。", ""]
        if not result.get("target_trials"):
            lines += ["没有独立目标执行记录；不能从孪生结果推断目标成功。", ""]
        for target in result.get("target_trials", []):
            lines += [f"#### 目标重复 {target.get('repeat')}：{target.get('status')}", "",
                f"实际成功：{cell(target.get('success'))}；错误：{cell(target.get('error', target.get('execution', {}).get('error')) or '无')}。", ""]
            execution = target.get("execution")
            if execution is None:
                continue
            lines += [f"独立实例检查：{cell(target.get('isolation'))}。", "",
                f"实际参数扰动：{cell(execution.get('trial'))}。目标扰动值在选择后采样，没有输入价值模型。", "",
                f"所选 ID：`{target.get('selected_candidate_id')}`；目标重绑定 ID：`{target.get('rebound_candidate_id')}`；语义 SHA256：`{target.get('semantic_sha256')}`。", "",
                f"执行 {execution.get('executed_steps')} 条调用，物理步 {execution.get('physics_steps')}，仿真时间 {cell(execution.get('sim_seconds'))} s；超时：{cell(execution.get('timeout'))}。", ""]
            goals = execution.get("goal_check")
            if goals is None:
                lines += ["独立最终目标检查未到达；不能将早停时的零件位置解释为通过最终验收。", ""]
            else:
                lines += [f"完整任务独立验收：{cell(goals.get('success'))}。", "",
                    "|部件|通过|位置误差 (m)|倾斜 (deg)|已释放|触碰手指|末端净空 (m)|", "|---|---|---:|---:|---|---|---:|"]
                for goal in goals.get("goals", []):
                    lines.append("|" + "|".join(cell(v) for v in [PARTS.get(goal.get("part"), goal.get("part")),
                        *[goal.get(key) for key in ("success", "position_error_m", "tilt_deg", "released", "touching_finger", "eef_clearance_m")]]) + "|")
                lines.append("")
            lines += ["<details>", "<summary>逐条实际执行调用（参数与完整指标保留在 walkthrough.json）</summary>", "",
                "|执行序号|实际技能|成功|仿真秒|指标|", "|---:|---|---|---:|---|"]
            for index, step in enumerate(execution.get("executed_parameters") or []):
                lines.append("|" + "|".join(cell(v) for v in [index + 1, step.get("skill"), step.get("ok"), step.get("sim_seconds"), step.get("metrics")]) + "|")
            lines += ["", "</details>", ""]
        seconds = result.get("seconds", {})
        lines += [f"记录决策时间：{cell(seconds.get('decision'))} s；场景创建至独立执行和终态渲染：{cell(seconds.get('total'))} s（不含离线 LLM 代理生成与进程启动，模型加载另计）。", ""]
        for item in record.get("terminal_evidence", []):
            if item.get("resolved_path"):
                lines += [f"[实际目标终态图像](<{item['resolved_path']}>)：来自当次执行的终态数据，无重放；图像 SHA256 `{item['sha256']}`。", ""]
            elif item.get("error"):
                lines += [f"终态渲染未成功：{cell(item['error'])}。", ""]
    lines += ["## 4. 视频与证据边界", ""]
    if summary["replay"]:
        replay = summary["replay"]
        lines += [f"[所选 TopK 目标的记录状态重放视频](<{replay['video_path']}>)。它与目标执行文件、扰动试验、状态轨迹和视频 SHA256 绑定，没有新物理积分。", ""]
    else:
        lines += ["未附加视频。若后续提供视频，只能绑定本案例所选目标的已记录状态重放；本工具不生成或重跑视频。", ""]
    lines += ["- " + value for value in summary["limitations"]]
    lines += ["", "完整记录与核验文件 SHA256 见同目录 `walkthrough.json`。已核验：", ""]
    lines += ["- " + check for check in summary["integrity"]["checks"]]
    if summary["integrity"]["warnings"]:
        lines += ["", "缺失证据：", ""] + ["- " + warning for warning in summary["integrity"]["warnings"]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, help="Explicit case directory containing top_k/ and full/; never auto-selected")
    parser.add_argument("--out", required=True, help="Output directory for walkthrough.json and walkthrough.md")
    parser.add_argument("--replay", help="Optional existing selected TopK target replay JSON sidecar; matching .mp4 must exist")
    args = parser.parse_args()
    summary = build_walkthrough(args.case, args.replay)
    output = Path(args.out).resolve()
    require(output != Path(args.case).resolve() and Path(args.case).resolve() not in output.parents,
            "output must be outside the selected immutable case directory")
    output.mkdir(parents=True, exist_ok=True)
    (output / "walkthrough.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    (output / "walkthrough.md").write_text(markdown(summary), encoding="utf-8")
    print(json.dumps(dict(output=str(output), candidates=len(summary["candidates"]),
        warnings=summary["integrity"]["warnings"], simulation_calls=0, model_calls=0), ensure_ascii=False))


if __name__ == "__main__":
    main()
