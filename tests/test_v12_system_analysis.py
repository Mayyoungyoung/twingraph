"""Synthetic audit fixtures only; these are not physical experiment labels."""
import json
import pytest

from scripts.analyze_v12_system import PRIMARY, analyze, digest, markdown


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def write_method(root, method, seed=1660, variation=0):
    directory = root/f"seed_{seed}"/method
    pool = [dict(name=f"candidate_{i}", parameter=i+variation) for i in range(2)]
    files = {"simbench/example.py": "b"*64}
    runtime = digest(files, compact=True)
    observation = "a"*64
    request = dict(seed=seed, method=method, pool=pool, order=[0, 1], k=2, domain="online",
        source=dict(cad_sha256=digest({})), runtime_sha256=runtime, geometry_version="printed_test_fixture",
        initial_observation=dict(sha256=observation), model_sha256="c"*64 if method.startswith("value_") else None,
        scores=[.8, .2] if method.startswith("value_") else None)
    if method=="value_early_stop":
        request["progressive"]=dict(initial_k=1,batch_size=1,max_candidates=2,policy="ordered_batches_first_verified_success")
    save(directory/"request.json", request)
    save(directory/"runtime_sources.json", dict(files=files, sha256=runtime))
    trials = []
    for i, proposal in enumerate(pool):
        graph = dict(proposal=proposal, task_geometry_version="printed_test_fixture", planning_cad={},
            assembly=dict(observation=dict(perception=dict(observation_sha256=observation))))
        detail = dict(seed=seed, valid=True, domain="online", success=False, proposal=proposal,
            input_graph_sha256=digest(graph), runtime_sha256=runtime, geometry_version="printed_test_fixture",
            initial_observation=dict(sha256=observation), total_wall_seconds=1., error="approach: blocked",
            executed_parameters=[dict(skill="approach", params=dict(part="carriage"), ok=False)], boundaries=[])
        save(directory/"twins"/proposal["name"]/"result.json", detail)
        save(directory/"twins"/proposal["name"]/"input_graph.json", graph)
        trials.append(dict(index=i, name=proposal["name"], success=False, wall_seconds=1.))
    summary = dict(seed=seed, method=method, valid=True, runtime_sha256=runtime, trials=trials, k=2,
        pool_size=2, selected=None, twin_success=False, execution_success=False, execution_seconds=0.,
        decision_seconds=3., total_wall_seconds=3.1, generation_seconds=.3, ranking_seconds=.1)
    if method=="value_early_stop":
        summary.update(progressive=request["progressive"],verification_batches=[
            dict(batch=i+1,rank_start=i,rank_stop=i+1,attempted_indices=[i]) for i in range(2)])
    save(directory/"summary.json", summary)
    return directory


def test_complete_actual_record_schema_and_failure_stage_counts(tmp_path):
    for method in PRIMARY:
        write_method(tmp_path, method)
    report = analyze(tmp_path, seeds=[1660])
    assert report["status"] == "complete" and report["complete_paired_layouts"] == [1660]
    assert report["stopping_stage_distribution"]["twin"] == {"carriage": 6}
    assert report["paired_aggregate"][0]["metrics"]["mean_decision_seconds"] == 3.


def test_missing_methods_and_layouts_are_partial_not_zero(tmp_path):
    write_method(tmp_path, "all_twin")
    report = analyze(tmp_path, seeds=[1660, 1661])
    assert report["status"] == "partial_or_invalid" and len(report["incomplete"]) == 5
    assert all(row["metrics"] is None for row in report["paired_aggregate"])
    assert report["available_aggregate"][0]["completed_layouts"] == 1
    assert "未完成" in markdown(report)


def test_candidate_pool_mismatch_excludes_paired_comparison(tmp_path):
    for method in PRIMARY:
        write_method(tmp_path, method, variation=int(method == "value_top_k"))
    report = analyze(tmp_path, seeds=[1660])
    assert report["paired_checks"][0]["mismatched_fields"] == ["pool_sha256"]
    assert not report["complete_paired_layouts"]


def test_invalid_physical_result_and_fabricated_timer_are_rejected(tmp_path):
    directory = write_method(tmp_path, "all_twin")
    path = directory/"twins/candidate_0/result.json"
    detail = json.loads(path.read_text()); detail["valid"] = False; save(path, detail)
    report = analyze(tmp_path, seeds=[1660], methods=["all_twin"])
    assert "invalid physical" in report["invalid"][0]["reason"]
    directory = write_method(tmp_path, "all_twin")
    path = directory/"summary.json"
    summary = json.loads(path.read_text()); summary["decision_seconds"] = .5; save(path, summary)
    report = analyze(tmp_path, seeds=[1660], methods=["all_twin"])
    assert "shorter" in report["invalid"][0]["reason"]


def test_value_checkpoint_record_is_required(tmp_path):
    directory = write_method(tmp_path, "value_top_k")
    path = directory/"request.json"
    request = json.loads(path.read_text()); request["model_sha256"] = None; save(path, request)
    report = analyze(tmp_path, seeds=[1660], methods=["value_top_k"])
    assert "checkpoint hash is missing" in report["invalid"][0]["reason"]


def test_runtime_source_manifest_hash_must_match(tmp_path):
    directory = write_method(tmp_path, "all_twin")
    path = directory/"runtime_sources.json"
    manifest = json.loads(path.read_text()); manifest["files"]["simbench/example.py"] = "d"*64; save(path, manifest)
    report = analyze(tmp_path, seeds=[1660], methods=["all_twin"])
    assert "runtime source-file manifest" in report["invalid"][0]["reason"]


def test_unfinished_progress_json_stays_partial_without_invented_count(tmp_path):
    directory = tmp_path/"seed_1660"/"all_twin"
    directory.mkdir(parents=True)
    (directory/"progress.json").write_text('{"trials": [')
    report = analyze(tmp_path, seeds=[1660], methods=["all_twin"])
    assert report["status"] == "partial_or_invalid" and report["paired_aggregate"][0]["metrics"] is None
    assert report["incomplete"][0]["recorded_progress_trials"] is None
    assert "进度文件暂不可读" in markdown(report)


def test_successful_deployment_requires_archived_graph_binding(tmp_path):
    directory = write_method(tmp_path, "all_twin")
    path = directory/"twins/candidate_0/result.json"
    detail = json.loads(path.read_text()); detail["success"] = True; save(path, detail)
    path = directory/"summary.json"
    summary = json.loads(path.read_text())
    summary["trials"][0]["success"] = True
    summary.update(selected="candidate_0", twin_success=True, execution_success=True,
                   execution_seconds=1., total_wall_seconds=4.1)
    save(path, summary)
    graph = json.loads((directory/"twins/candidate_0/input_graph.json").read_text())
    detail["domain"] = "deployment"
    save(directory/"deployment/result.json", detail)
    save(directory/"deployment/input_graph.json", graph)
    assert analyze(tmp_path, seeds=[1660], methods=["all_twin"])["status"] == "complete"
    graph["proposal"]["parameter"] = -1
    save(directory/"deployment/input_graph.json", graph)
    report = analyze(tmp_path, seeds=[1660], methods=["all_twin"])
    assert "deployment is not bound" in report["invalid"][0]["reason"]


def add_successful_deployment_and_replan(directory, method):
    request = json.loads((directory/"request.json").read_text())
    path = directory/"twins/candidate_1/result.json"
    detail = json.loads(path.read_text()); detail["success"] = True; save(path, detail)
    graph = json.loads(path.with_name("input_graph.json").read_text())
    summary_path = directory/"summary.json"
    summary = json.loads(summary_path.read_text())
    summary["trials"][1]["success"] = True
    summary.update(selected="candidate_1", twin_success=True, execution_success=True,
                   execution_seconds=3., total_wall_seconds=6.1,
                   events=[dict(action="verified_remaining_suffix", selected="candidate_1")])
    save(summary_path, summary)
    deployment = {**detail, "domain":"deployment", "total_wall_seconds":3.}
    save(directory/"deployment/result.json", deployment)
    save(directory/"deployment/input_graph.json", graph)
    replan = directory/"closed_loop/replan_1"
    save(replan/"request.json", dict(pool=request["pool"], method=method, k=2,
        order=[0,1], scores=[.8,.2] if method.startswith("value_") else None,
        progressive=request.get("progressive")))
    if method=="value_early_stop":
        save(replan/"progress.json",dict(verification_batches=summary["verification_batches"]))
    for index in (0,1):
        source = directory/f"twins/candidate_{index}"
        save(replan/f"candidate_{index}/result.json", json.loads((source/"result.json").read_text()))
        save(replan/f"candidate_{index}/input_graph.json", json.loads((source/"input_graph.json").read_text()))


@pytest.mark.parametrize("method", ["all_twin", "random_top_k", "value_top_k", "random_early_stop", "value_early_stop"])
def test_method_specific_replanning_allows_null_nonvalue_scores_and_counts_real_calls(tmp_path, method):
    directory = write_method(tmp_path, method)
    add_successful_deployment_and_replan(directory, method)
    report = analyze(tmp_path, seeds=[1660], methods=[method])
    assert report["status"] == "complete", report["invalid"]
    row = report["rows"][0]
    assert row["twin_calls"] == 4 and row["replan_twin_calls"] == 2
    # Replanning is already inside measured deployment wall time.
    assert row["execution_seconds"] == 3. and row["total_wall_seconds"] == 6.1


def test_full_twin_replanning_cannot_silently_skip_candidates(tmp_path):
    directory = write_method(tmp_path, "all_twin")
    add_successful_deployment_and_replan(directory, "all_twin")
    (directory/"closed_loop/replan_1/candidate_1/result.json").unlink()
    report = analyze(tmp_path, seeds=[1660], methods=["all_twin"])
    assert "full twin replanning lacks" in report["invalid"][0]["reason"]


@pytest.mark.parametrize("corruption", ["budget", "batches", "score", "truncate"])
def test_progressive_audit_rejects_missing_coverage_or_incorrect_order(tmp_path,corruption):
    directory=write_method(tmp_path,"value_early_stop")
    request=json.loads((directory/"request.json").read_text())
    summary=json.loads((directory/"summary.json").read_text())
    if corruption=="budget": request["k"]=summary["k"]=1
    if corruption=="batches": summary["verification_batches"].pop()
    if corruption=="score": request["scores"]=[.1,.9]
    if corruption=="truncate":
        summary["trials"].pop()
        summary["verification_batches"].pop()
    save(directory/"request.json",request);save(directory/"summary.json",summary)
    report=analyze(tmp_path,seeds=[1660],methods=["value_early_stop"])
    assert report["status"]=="partial_or_invalid" and len(report["invalid"])==1
