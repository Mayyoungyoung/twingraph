"""Synthetic archive-format tests, never new physical experiment labels."""
import json
from pathlib import Path
import pytest

from scripts.export_v12_all_twin_matrix import export_matrices, sha
from scripts.analyze_v12_system import digest
from tests.test_v12_system_analysis import write_method


def test_export_preserves_result_graph_request_bytes_and_recorded_timing(tmp_path):
    source, target = tmp_path/"online", tmp_path/"export"
    directory = write_method(source, "all_twin")
    path = directory/"twins/candidate_0/result.json"
    result = json.loads(path.read_text()); result["timing_mode"] = "actual_online_serial"
    path.write_text(json.dumps(result))
    originals = {str(p):sha(p) for p in directory.rglob("*.json")}
    report = export_matrices(source, target, [1660], expected_pool_size=2)
    collect = target/"seed_1660/collect"
    assert (collect/"request.json").read_bytes() == (directory/"request.json").read_bytes()
    assert (collect/"candidates/candidate_0/result.json").read_bytes() == path.read_bytes()
    assert report["original_outcomes_and_times_modified"] is False
    modes = report["layouts"][0]["source_timings"]
    assert modes[0]["source_recorded_timing_mode"] == "actual_online_serial"
    assert modes[1]["source_recorded_timing_mode"] is None
    assert {str(p):sha(p) for p in directory.rglob("*.json")} == originals


def test_partial_pool_and_wrong_required_count_are_rejected(tmp_path):
    source = tmp_path/"online"
    directory = write_method(source, "all_twin")
    with pytest.raises(ValueError, match="required complete candidate count"):
        export_matrices(source, tmp_path/"out_a", [1660], expected_pool_size=48)
    (directory/"twins/candidate_1/result.json").unlink()
    with pytest.raises(ValueError, match="incomplete or invalid"):
        export_matrices(source, tmp_path/"out_b", [1660], expected_pool_size=2)
    assert not (tmp_path/"out_b").exists()


def test_top_k_archive_cannot_be_passed_off_as_full_matrix(tmp_path):
    source = tmp_path/"online"
    directory = write_method(source, "random_top_k")
    directory.rename(directory.with_name("all_twin"))
    with pytest.raises(ValueError, match="method disagrees"):
        export_matrices(source, tmp_path/"export", [1660], expected_pool_size=2)


def test_tampered_graph_or_invalid_label_is_rejected_before_writing(tmp_path):
    source = tmp_path/"online"
    directory = write_method(source, "all_twin")
    graph_path = directory/"twins/candidate_0/input_graph.json"
    graph = json.loads(graph_path.read_text()); graph["proposal"]["parameter"] = 999
    graph_path.write_text(json.dumps(graph))
    with pytest.raises(ValueError, match="not bound"):
        export_matrices(source, tmp_path/"export", [1660], expected_pool_size=2)
    assert not (tmp_path/"export").exists()


def test_existing_output_and_nested_source_output_are_refused(tmp_path):
    source = tmp_path/"online"
    write_method(source, "all_twin")
    target = tmp_path/"export"; target.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        export_matrices(source, target, [1660], expected_pool_size=2)
    with pytest.raises(ValueError, match="separate"):
        export_matrices(source, source/"export", [1660], expected_pool_size=2)


def test_export_is_consumable_by_native_matrix_loader(tmp_path):
    from tests.test_value_v12_matrix import write_matrix
    from scripts.train_value_v12_matrix import load_matrices
    from simbench.value.plan import PlanIR
    from simbench.value.skill_graph import compile_graph
    source, target = tmp_path/"online", tmp_path/"export"
    old, candidate = write_matrix(source)
    graph = json.loads((candidate/"input_graph.json").read_text())
    graph["assembly"]["observation"]["perception"]["observation_sha256"] = "a"*64
    graph["assembly"] = compile_graph(graph["assembly"]["observation"], PlanIR.from_dict(graph["assembly"]["plan"]))
    files = {"simbench/example.py":"b"*64}; runtime = digest(files, compact=True)
    result = json.loads((candidate/"result.json").read_text())
    result.update(domain="online", runtime_sha256=runtime, initial_observation=dict(sha256="a"*64), input_graph_sha256=digest(graph))
    (candidate/"input_graph.json").write_text(json.dumps(graph))
    (candidate/"result.json").write_text(json.dumps(result))
    request = json.loads((old/"request.json").read_text())
    request.update(method="all_twin", domain="online", order=[0], k=1, model_sha256=None,
        runtime_sha256=runtime, initial_observation=dict(sha256="a"*64), source=dict(cad_sha256=digest({})))
    (old/"request.json").write_text(json.dumps(request))
    (old/"runtime_sources.json").write_text(json.dumps(dict(files=files,sha256=runtime)))
    summary = dict(seed=1610, method="all_twin", valid=True, runtime_sha256=runtime, pool_size=1, k=1,
        trials=[dict(index=0,name=result["proposal"]["name"],success=False,wall_seconds=2.5)],
        selected=None,twin_success=False,execution_success=False,execution_seconds=0.,
        decision_seconds=3., total_wall_seconds=3.1, generation_seconds=.3,ranking_seconds=.1)
    (old/"summary.json").write_text(json.dumps(summary))
    (old/"candidates").rename(old/"twins")
    old.rename(old.with_name("all_twin"))
    export_matrices(source, target, [1610], expected_pool_size=1)
    rows, manifest = load_matrices([target], [1610])
    assert len(rows) == 1 and manifest[0]["runtime_sha256"] == runtime
