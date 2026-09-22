"""Synthetic archive-format tests only; no physical scene or real labels opened."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts import analyze_v12_pool_budget as analyzer


ROOT = Path(__file__).resolve().parents[1]
SEED = 99001  # Artificial format fixture, never a physical experiment seed.


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, allow_nan=False), encoding="utf-8")


def read(path):
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def generator():
    # Imports and hashes only. In particular no make_scene, render, or mj_step.
    return analyzer._load_generator(ROOT)


@pytest.fixture
def matrix(tmp_path, generator, monkeypatch):
    proposer, _, frozen = generator
    from simbench.value.skill_graph import interface_hash
    # This suite checks archive integrity and exact budget replay, not CAD.
    # Keep the real nested generator but declare its geometry as a software stub.
    from simbench.value import planner_v12
    def scripted_catalogs(observation, cad, completed):
        rows = {part: [dict(part=part, yaw=0., height=.004, placement_yaw=0.,
            source_axis_offset_rad=0., status="unknown", min_clearance_m=None,
            required_clearance_m=.003, software_fixture=True)]
            for part in planner_v12.PARTS if part not in completed}
        return rows, deepcopy(rows)
    monkeypatch.setattr(planner_v12, "_geometry_catalogs", scripted_catalogs)
    # The real diagnostic remains pinned to r6. These synthetic archives bind
    # their own runtime so newer software can exercise the same integrity rules.
    monkeypatch.setattr(analyzer, "FROZEN_RUNTIME_SHA256", frozen["sha256"])
    monkeypatch.setattr(analyzer, "_load_generator", lambda runtime: (proposer, lambda: frozen, frozen))
    root = tmp_path/"synthetic_matrix"
    directory = root/f"seed_{SEED}"/"collect"
    observation = dict(backend="rgbd_geometry", objects={
        name: dict(valid=True, position_m=[-.32+.07*i, -.21, .84], quat_wxyz=[1., 0., 0., 0.], fit_residual_m=.001)
        for i, name in enumerate(("carriage", "end_stop", "pin_left", "pin_right", "handle", "wipe_tool"))})
    observation["sha256"] = analyzer.digest(observation)
    cad = dict(functional_stroke_minimum_m=.02, wipe_halfspan_m=[.055, .005])
    pool, source = map(analyzer.canonical, proposer(observation, cad=cad, n=48, seed=SEED))
    request = dict(seed=SEED, domain="train", pool=pool, source=source,
        runtime_sha256=frozen["sha256"], geometry_version="synthetic_archive_fixture",
        initial_observation=observation)
    dump(directory/"request.json", request)
    dump(directory/"runtime_sources.json", frozen)
    for proposal in pool:
        graph = dict(proposal=proposal, planning_cad=cad, task_geometry_version=request["geometry_version"],
            assembly=dict(interface_sha256=interface_hash(), observation=dict(perception=dict(observation_sha256=observation["sha256"]))),
            synthetic_format_fixture=True)
        result = dict(valid=True, success=proposal["name"] in {"grounded_000", "grounded_013", "grounded_029"},
            domain="train", seed=SEED, runtime_sha256=frozen["sha256"],
            proposal=proposal, initial_observation=observation, geometry_version=request["geometry_version"],
            input_graph_sha256=analyzer.digest(graph), synthetic_format_fixture=True)
        target = directory/"candidates"/proposal["name"]
        dump(target/"input_graph.json", graph)
        dump(target/"result.json", result)
    return root, directory


def test_same_generator_budget_membership_and_exact_random_coverage(matrix):
    root, _ = matrix
    report = analyzer.analyze(root, [SEED], ROOT)
    assert report["complete"] and report["no_new_physical_experiments"]
    assert [r["feasible_candidates"] for r in report["rows"]] == [1, 2, 3]
    assert [r["random_top4"]["hit_fraction"] for r in report["rows"]][0] == "1/3"
    assert report["rows"][0]["candidate_names"] == [f"grounded_{i:03d}" for i in range(12)]
    regeneration = report["layout_bindings"][0]["regeneration"]
    assert regeneration[0]["shuffled_generation_order"] != regeneration[2]["shuffled_generation_order"][:12]
    assert all(r["exact_same_name_proposals"] for r in regeneration)
    assert len(report["layout_bindings"][0]["label_files"]) == 48


@pytest.mark.parametrize("n,m,expected", [(12, 0, "0/1"), (24, 24, "1/1"), (48, 1, "1/12"), (12, 10, "1/1")])
def test_random_top4_is_exact_without_replacement(n, m, expected):
    assert analyzer.random_top_k(n, m)["hit_fraction"] == expected


def test_non_nested_generator_is_rejected_even_when_names_match(matrix, generator, monkeypatch):
    root, _ = matrix
    proposer, _, frozen = generator
    def changed(observation, **kwargs):
        pool, source = proposer(observation, **kwargs)
        if kwargs["n"] == 12:
            pool[0]["wipe_force"] += .1
        return pool, source
    monkeypatch.setattr(analyzer, "_load_generator", lambda runtime: (changed, lambda: frozen, frozen))
    with pytest.raises(ValueError, match="non-nested"):
        analyzer.analyze(root, [SEED], ROOT)


def test_n48_original_shuffled_order_must_reproduce(matrix):
    root, directory = matrix
    request = read(directory/"request.json")
    request["pool"].reverse()
    dump(directory/"request.json", request)
    with pytest.raises(ValueError, match="generation order"):
        analyzer.analyze(root, [SEED], ROOT)


def test_missing_label_never_becomes_failure(matrix):
    root, directory = matrix
    (directory/"candidates/grounded_000/result.json").unlink()
    with pytest.raises(FileNotFoundError):
        analyzer.analyze(root, [SEED], ROOT)


@pytest.mark.parametrize("update", [dict(success=None), dict(valid=False), dict(domain="online")])
def test_invalid_or_unpaired_labels_are_rejected(matrix, update):
    root, directory = matrix
    path = directory/"candidates/grounded_000/result.json"
    dump(path, {**read(path), **update})
    with pytest.raises(ValueError, match="label|domain"):
        analyzer.analyze(root, [SEED], ROOT)


def test_graph_hash_tamper_cannot_relabel_a_candidate(matrix):
    root, directory = matrix
    path = directory/"candidates/grounded_000/input_graph.json"
    graph = read(path)
    graph["assembly"]["interface_sha256"] = "e"*64
    dump(path, graph)
    with pytest.raises(ValueError, match="bound to its archived input graph"):
        analyzer.analyze(root, [SEED], ROOT)


def test_observation_contents_must_match_hash(matrix):
    root, directory = matrix
    path = directory/"request.json"
    request = read(path)
    request["initial_observation"]["objects"]["carriage"]["position_m"][0] += .1
    dump(path, request)
    with pytest.raises(ValueError, match="observation hash"):
        analyzer.analyze(root, [SEED], ROOT)


def test_cad_cannot_change_even_with_updated_graph_hash(matrix):
    root, directory = matrix
    target = directory/"candidates/grounded_000"
    graph = read(target/"input_graph.json")
    graph["planning_cad"]["wipe_halfspan_m"][0] += .01
    dump(target/"input_graph.json", graph)
    result = read(target/"result.json")
    result["input_graph_sha256"] = analyzer.digest(graph)
    dump(target/"result.json", result)
    with pytest.raises(ValueError, match="CAD missing or not bound"):
        analyzer.analyze(root, [SEED], ROOT)


def test_runtime_or_incomplete_topk_cannot_be_substituted(matrix):
    root, directory = matrix
    path = directory/"request.json"
    request = read(path)
    dump(path, {**request, "runtime_sha256":"a"*64})
    with pytest.raises(ValueError, match="runtime binding"):
        analyzer.analyze(root, [SEED], ROOT)
    dump(path, {**request, "method":"random_top_k"})
    with pytest.raises(ValueError, match="cannot substitute"):
        analyzer.analyze(root, [SEED], ROOT)


def test_unrequested_layout_is_never_opened(matrix):
    root, _ = matrix
    ignored = root/"seed_99002/collect/request.json"
    ignored.parent.mkdir(parents=True)
    ignored.write_text("deliberately invalid JSON: must never be read")
    report = analyzer.analyze(root, [SEED], ROOT)
    assert all("seed_99002" not in p for p in report["input_files"])
    with pytest.raises(FileNotFoundError):
        analyzer.analyze(root, [SEED, 99003], ROOT)


def make_export(root, directory):
    request = read(directory/"request.json")
    request.update(method="all_twin", domain="online")
    dump(directory/"request.json", request)
    trials = []
    for i, proposal in enumerate(request["pool"]):
        path = directory/"candidates"/proposal["name"]/"result.json"
        result = read(path)
        result["domain"] = "online"
        dump(path, result)
        trials.append(dict(index=i, name=proposal["name"], success=result["success"]))
    summary = dict(valid=True, seed=SEED, method="all_twin", pool_size=48,
                   runtime_sha256=request["runtime_sha256"], trials=trials)
    dump(directory/"source_online_summary.json", summary)
    files = [directory/name for name in ("request.json", "runtime_sources.json", "source_online_summary.json")]
    files += list((directory/"candidates").glob("*/*.json"))
    entry = dict(seed=SEED, complete=True, pool_size=48, source_runtime_sha256=request["runtime_sha256"],
        source_method="all_twin", source_domain="online",
        source_request_sha256=analyzer.sha(directory/"request.json"),
        source_summary_sha256=analyzer.sha(directory/"source_online_summary.json"),
        files=[dict(exported=p.relative_to(root).as_posix(), sha256=analyzer.sha(p),
                    source="synthetic original path: deliberately not followed") for p in files])
    dump(directory/"export_manifest.json", entry)
    dump(root/"export_manifest.json", dict(schema="twingraph.actual_online_all_twin_matrix_export.v12",
        complete=True, expected_pool_size=48, expected_seeds=[SEED], source_runtime_sha256=request["runtime_sha256"],
        training_or_simulation_executed=False, original_outcomes_and_times_modified=False, layouts=[entry]))


def test_all_twin_export_is_byte_bound_and_complete(matrix):
    root, directory = matrix
    make_export(root, directory)
    report = analyzer.analyze(root, [SEED], ROOT)
    assert report["layout_bindings"][0]["label_origin"] == "actual online all_twin byte-preserved export"
    path = directory/"candidates/grounded_000/result.json"
    result = read(path)
    result["success"] = not result["success"]
    dump(path, result)
    with pytest.raises(ValueError, match="bytes differ"):
        analyzer.analyze(root, [SEED], ROOT)


def test_source_change_during_analysis_is_rejected(matrix, generator, monkeypatch):
    root, _ = matrix
    proposer, _, frozen = generator
    changed = deepcopy(frozen)
    changed["sha256"] = "0"*64
    monkeypatch.setattr(analyzer, "_load_generator", lambda runtime: (proposer, lambda: changed, frozen))
    with pytest.raises(ValueError, match="changed during analysis"):
        analyzer.analyze(root, [SEED], ROOT)
