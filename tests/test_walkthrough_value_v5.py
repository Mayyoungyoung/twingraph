"""Synthetic artifact fixtures only: no simulation, trained model or real result."""
import copy
import json

import pytest

from scripts import walkthrough_value_v5 as walk


ORDER = ["carriage", "end_stop", "pin_left", "pin_right", "handle"]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def plan(cid):
    return dict(id=cid, protocol="synthetic-fixture", calls=[dict(id="call0", skill="move",
        arguments={"path": dict(value="transfer", status="known")}, roles={})],
        prefix=dict(order=ORDER, choices={part: dict(yaw=.2, height=.001, clearance=.98, force=10.) for part in ORDER},
            initial_route_index=0, semantic_program_id="fixture_semantic_" + cid,
            initial_artifacts={"transfer": dict(type="joint_path", part=None, start_q=[0., .2],
                joints=[[.1, .2], [.3, .4]], target=[.1, .2, .3],
                rotation=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], binding={"prefix_id": cid})}))


def trial(cid, success, role="twin", repeat=0):
    draw = dict(domain="v5_" + role, namespace=5107 if role == "twin" else 7901,
                repeat=repeat, friction_scale=1., actuator_gain_scale=1.)
    goals = dict(success=success, goals=[dict(part=p, success=success, position_error_m=.0001,
        tilt_deg=0., released=True, touching_finger=False, eef_clearance_m=.05) for p in ORDER])
    return dict(candidate_id=cid, trial=draw, trial_sha256=walk.digest(draw), valid=True,
        success=success, timeout=False, error="" if success else "fixture physical failure", goal_check=goals,
        executed_steps=1, executed_parameters=[dict(skill="move", params={"path": "transfer"},
            ok=success, metrics={"position_error": .0001}, sim_seconds=.3)],
        physics_steps=100, wall_seconds=.5, sim_seconds=.3, final_positions={})


def case_fixture(path, abstain=False, target_failure=False):
    planner = dict(task_text="synthetic fixture, not research evidence", source="llm_proxy", provider="fixture",
        model="fixture", proposed_orders=[ORDER], raw_response="fixture response")
    inp = dict(schema="twingraph.system_inputs.v5", observation={"goals": [dict(manipulated=p,
        predicate="seated_released_retracted", position=[0., 0., 0.], position_tolerance=.0015,
        tilt_tolerance_deg=3., minimum_eef_clearance_m=.02) for p in ORDER]},
        candidates=[plan(cid) for cid in ("p0", "p1", "p2")], planner_sha256=walk.digest(planner))
    for method in ("top_k", "full"):
        directory = path / method
        subset = ["p1", "p2"] if method == "top_k" else ["p0", "p1", "p2"]
        validation = [trial(cid, not abstain and cid != "p1") for cid in subset]
        selected = None if abstain else "p2" if method == "top_k" else "p0"
        targets = []
        if selected:
            graph = {"fixture_graph": selected}
            execution = trial(selected + "_rebound", not target_failure, "target")
            execution.update(input_graph_sha256=walk.digest(graph), state_trace={"sha256": "a" * 64})
            if target_failure:
                execution["goal_check"] = None
            target = dict(repeat=0, status="executed", success=not target_failure,
                selected_candidate_id=selected, rebound_candidate_id=selected + "_rebound",
                semantic_sha256="s" * 64, execution=execution,
                isolation=dict(distinct_session=True, distinct_context=True, distinct_model=True,
                    distinct_data=True, snapshot_transfer=False))
            targets.append(target)
            write(directory / "target_execution_0.json", target)
            write(directory / "target_input_0.json", dict(graph=graph, selected_candidate_id=selected,
                                                        semantic_sha256="s" * 64))
        result = dict(schema="twingraph.system_run.v5", method=method, config={"accept_rate": .5}, n=3,
            actual_n=3, k=len(subset), planner_sha256=walk.digest(planner), inputs_sha256=walk.digest(inp),
            implementation_sha256=walk.digest({}), paired_namespace=dict(twin=5107, target=7901),
            selected_candidate_id=selected, selected_semantic_sha256="s" * 64,
            validated_candidate_ids=subset, validation=validation, candidate_validation=[dict(candidate_id=cid,
                trials=1, successful_trials=int(not abstain and cid != "p1")) for cid in subset],
            target_trials=targets, requested_target_trials=1, target_successes=sum(r["success"] for r in targets),
            simulation_calls=dict(twin_validation=len(validation), target_execution=len(targets)),
            status="abstained_no_validated_plan" if abstain else "executed_independent_target", seconds={})
        if method == "top_k":
            result["ranking"] = dict(scores=[.4, .7, .7], logits=[-.4, .9, .9], order=["p1", "p2", "p0"],
                top_k=["p1", "p2"], checkpoint_sha256="c" * 64)
        for name, value in (("result.json", result), ("planner.json", planner), ("inputs.json", inp), ("source.json", {})):
            write(directory / name, value)
    write(path / "pair.json", dict(schema="twingraph.system_pair.v5", input_sha256=walk.digest(inp)))
    return path


def test_source_order_ranking_twin_outcomes_and_both_targets_are_preserved(tmp_path):
    case = case_fixture(tmp_path / "explicit_case")
    before = {str(p): walk.file_sha(p) for p in case.rglob("*.json")}
    result = walk.build_walkthrough(case)
    assert [row["candidate_id"] for row in result["candidates"]] == ["p0", "p1", "p2"]
    assert [row["frozen_rank"] for row in result["candidates"]] == [3, 1, 2]
    assert [row["top_k_member"] for row in result["candidates"]] == [False, True, True]
    assert result["candidates"][0]["policies"]["top_k"]["observed_trials"] == 0
    assert result["candidates"][0]["policies"]["full"]["successful_trials"] == 1
    assert result["candidates"][1]["policies"]["top_k"]["trials"][0]["first_failed_step"]["skill"] == "move"
    assert result["candidates"][0]["physical"]["initial_paths"]["transfer"]["waypoints"] == 2
    assert result["policies"]["top_k"]["recorded_result"]["selected_candidate_id"] == "p2"
    assert result["policies"]["full"]["recorded_result"]["selected_candidate_id"] == "p0"
    assert result["generation"]["model_calls"] == result["generation"]["simulation_calls"] == 0
    text = walk.markdown(result)
    assert "未验证" in text and "stage_calls" in text and "101" in text
    assert "未逐条作为可执行程序解析" in text and "位置误差 (m)" in text
    assert "位置容差 (m)" in text and "最低末端净空 (m)" in text
    assert {str(p): walk.file_sha(p) for p in case.rglob("*.json")} == before


def test_abstained_and_failed_target_do_not_acquire_success_or_goal_checks(tmp_path):
    abstained = walk.build_walkthrough(case_fixture(tmp_path / "abstained", abstain=True))
    assert abstained["policies"]["top_k"]["recorded_result"]["target_successes"] == 0
    assert "没有独立目标执行记录" in walk.markdown(abstained)
    failed = walk.build_walkthrough(case_fixture(tmp_path / "failed", target_failure=True))
    assert failed["policies"]["top_k"]["recorded_result"]["target_trials"][0]["success"] is False
    assert "独立最终目标检查未到达" in walk.markdown(failed)
    assert "fixture physical failure" in walk.markdown(failed)


def test_input_tampering_is_rejected_before_display(tmp_path):
    case = case_fixture(tmp_path / "case")
    path = case / "top_k" / "inputs.json"
    changed = walk.read(path)
    changed["candidates"][0]["prefix"]["initial_artifacts"]["transfer"]["joints"][1][0] += .01
    write(path, changed)
    with pytest.raises(ValueError, match="input hash mismatch"):
        walk.build_walkthrough(case)


def test_wrong_topk_membership_or_shared_target_scene_is_rejected(tmp_path):
    case = case_fixture(tmp_path / "case")
    path = case / "top_k" / "result.json"
    original = walk.read(path)
    changed = copy.deepcopy(original)
    changed["ranking"]["top_k"] = ["p0", "p1"]
    write(path, changed)
    with pytest.raises(ValueError, match="TopK/order mismatch"):
        walk.build_walkthrough(case)
    changed = copy.deepcopy(original)
    changed["target_trials"][0]["isolation"]["snapshot_transfer"] = True
    write(path, changed)
    with pytest.raises(ValueError, match="independent scene evidence"):
        walk.build_walkthrough(case)


def test_selected_target_replay_requires_exact_execution_trace_and_video_hash(tmp_path):
    case = case_fixture(tmp_path / "case")
    target = walk.read(case / "top_k" / "target_execution_0.json")
    video = tmp_path / "synthetic_fixture.mp4"
    video.write_bytes(b"synthetic fixture bytes; not an actual video")
    metadata = dict(schema="twingraph.recorded_target_replay.v5", simulation_integration_steps=0,
        selected_candidate_id=target["selected_candidate_id"], rebound_candidate_id=target["rebound_candidate_id"],
        recorded_trial=target["execution"]["trial"], recorded_success=True,
        execution_sha256=walk.file_sha(case / "top_k" / "target_execution_0.json"),
        trace_sha256="a" * 64, video_sha256=walk.file_sha(video))
    write(video.with_suffix(".json"), metadata)
    result = walk.build_walkthrough(case, video.with_suffix(".json"))
    assert result["replay"]["video_path"] == str(video.resolve())
    metadata["selected_candidate_id"] = "another_case"
    write(video.with_suffix(".json"), metadata)
    with pytest.raises(ValueError, match="not the selected TopK target trial"):
        walk.build_walkthrough(case, video.with_suffix(".json"))


def test_setup_failure_keeps_requested_target_denominator_without_invented_inputs(tmp_path):
    case = case_fixture(tmp_path / "case")
    for method in ("top_k", "full"):
        directory = case / method
        result = walk.read(directory / "result.json")
        result.update(status="twin_setup_failed", actual_n=0, validation=[], candidate_validation=[],
            selected_candidate_id=None, validated_candidate_ids=[], target_trials=[], target_successes=0,
            simulation_calls=dict(twin_validation=0, target_execution=0), setup_error="fixture unreachable IK")
        result.pop("ranking", None)
        result.pop("inputs_sha256")
        (directory / "inputs.json").unlink()
        write(directory / "result.json", result)
    (case / "pair.json").unlink()
    result = walk.build_walkthrough(case)
    assert result["candidates"] == []
    assert "fixture unreachable IK" in walk.markdown(result)
    assert "独立目标成功 0/1" in walk.markdown(result)


def test_actual_terminal_frame_binding_and_bytes_are_checked(tmp_path):
    case = case_fixture(tmp_path / "case")
    directory = case / "top_k"
    image = directory / "target_terminal_0.png"
    image.write_bytes(b"synthetic terminal image bytes")
    result = walk.read(directory / "result.json")
    execution = result["target_trials"][0]["execution"]
    result["images"] = [dict(path="old/original/target_terminal_0.png", sha256=walk.file_sha(image),
        candidate_id=execution["candidate_id"], trial_sha256=execution["trial_sha256"],
        graph_sha256=execution["input_graph_sha256"], source="actual_target_terminal_state", no_replay=True)]
    write(directory / "result.json", result)
    summary = walk.build_walkthrough(case)
    assert summary["policies"]["top_k"]["terminal_evidence"][0]["resolved_path"] == str(image)
    assert "实际目标终态图像" in walk.markdown(summary)
    image.write_bytes(b"changed image")
    with pytest.raises(ValueError, match="terminal image hash mismatch"):
        walk.build_walkthrough(case)
