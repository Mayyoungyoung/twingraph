import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.train_value_v12_matrix import DEFAULT_SPLITS, load_matrices, validate_splits, validate_checkpoint_runtime, validate_training_declaration
from scripts.train_value_v12 import historical_graph
from simbench.value.graph_value_v12 import encode_graph
from simbench.value.plan import digest, PlanIR
from simbench.value.skill_graph import compile_graph
from tests.test_value_v12 import sample


def write_matrix(root, seed=1610, runtime="a"*64):
    directory = root/f"seed_{seed}"/"collect"
    proposal = sample()["candidate"]
    graph = historical_graph(sample())
    observation = graph["assembly"]["observation"]
    observation["perception"] = dict(observation_sha256="shared_rgbd", backend="rgbd_geometry")
    observation["objects"]["pin_left"]["capabilities"] = ["insert", "release", "inspect"]
    graph["assembly"] = compile_graph(observation, PlanIR.from_dict(graph["assembly"]["plan"]))
    graph.update(proposal=proposal, task_geometry_version="test_printed_geometry")
    result = dict(success=False, valid=True, seed=seed, proposal=proposal, input_graph_sha256=digest(graph),
                  geometry_version="test_printed_geometry", initial_observation=dict(sha256="shared_rgbd"),
                  domain="train", total_wall_seconds=2.5, stage_passes={}, executed_parameters=[], runtime_sha256=runtime)
    request = dict(seed=seed, pool=[proposal], geometry_version="test_printed_geometry", initial_observation=dict(sha256="shared_rgbd"),
                   runtime_sha256=runtime)
    target = directory/"candidates"/proposal["name"]
    target.mkdir(parents=True)
    (directory/"request.json").write_text(json.dumps(request))
    (target/"input_graph.json").write_text(json.dumps(graph))
    (target/"result.json").write_text(json.dumps(result))
    return directory, target


def test_complete_hash_bound_matrix_accepts_capabilities(tmp_path):
    directory, target = write_matrix(tmp_path)
    rows, manifest = load_matrices([tmp_path], [1610])
    assert len(rows) == len(manifest) == 1 and rows[0]["y"] == 0
    assert np.isfinite(rows[0]["encoded"]["x"]).all()


def test_incomplete_matrix_cannot_be_used_as_complete(tmp_path):
    directory, target = write_matrix(tmp_path)
    request = json.loads((directory/"request.json").read_text())
    extra = copy.deepcopy(request["pool"][0]); extra["name"] = "unexecuted"
    request["pool"].append(extra)
    (directory/"request.json").write_text(json.dumps(request))
    with pytest.raises(ValueError, match="incomplete matrix"):
        load_matrices([tmp_path], [1610])


def test_mismatched_graph_label_binding_is_rejected(tmp_path):
    directory, target = write_matrix(tmp_path)
    result = json.loads((target/"result.json").read_text())
    result["input_graph_sha256"] = "wrong"
    (target/"result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="not bound"):
        load_matrices([tmp_path], [1610])


def test_train_loading_does_not_open_unallowed_test_results(tmp_path):
    directory, target = write_matrix(tmp_path)
    hidden = tmp_path/"seed_1660"/"collect"
    hidden.mkdir(parents=True)
    (hidden/"request.json").write_text(json.dumps(dict(seed=1660)))
    rows, _ = load_matrices([tmp_path], [1610])
    assert len(rows) == 1


def test_splits_cannot_leak_layouts():
    with pytest.raises(ValueError, match="leakage"):
        validate_splits(dict(train=[1, 2], validation=[2], test=[3]))


@pytest.mark.parametrize("seed", [1600, 1640, 1651, 1660, 1663, 1714, 1800, 1831])
def test_development_and_retired_layouts_cannot_be_redeclared_blind_test(seed):
    with pytest.raises(ValueError, match="cannot be an unseen test"):
        validate_splits(dict(train=[1610], validation=[1630], test=[seed]))


def test_reserved_splits_match_config_and_preserve_train_validation():
    path = Path(__file__).resolve().parents[1]/"simbench/configs/value_v12_data_split.json"
    configured = json.loads(path.read_text())
    assert {k: configured[k] for k in DEFAULT_SPLITS} == DEFAULT_SPLITS
    assert DEFAULT_SPLITS["train"] == list(range(2000, 2032))
    assert DEFAULT_SPLITS["validation"] == list(range(2100, 2108))
    assert DEFAULT_SPLITS["test"] == list(range(2200, 2208))
    retired = {n for lo, hi in configured["retired_layout_ranges"] for n in range(lo, hi+1)}
    assert set(range(1600, 1715)).issubset(retired)
    validate_splits(configured)
    validate_training_declaration(configured, 120)
    with pytest.raises(ValueError, match="training budget"):
        validate_training_declaration(configured, 121)
    with pytest.raises(ValueError, match="model_families"):
        validate_training_declaration({**configured, "model_families": ["graph"]}, 120)


def test_same_geometry_string_cannot_hide_request_result_runtime_drift(tmp_path):
    directory, target = write_matrix(tmp_path)
    result = json.loads((target/"result.json").read_text())
    result["runtime_sha256"] = "b"*64
    (target/"result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="runtime differs"):
        load_matrices([tmp_path], [1610])


@pytest.mark.parametrize("runtime", [None, "not-a-hash"])
def test_missing_or_malformed_runtime_is_rejected(tmp_path, runtime):
    write_matrix(tmp_path, runtime=runtime)
    with pytest.raises(ValueError, match="valid physical runtime SHA256"):
        load_matrices([tmp_path], [1610])


def test_allowed_layouts_from_two_roots_must_share_one_runtime(tmp_path):
    root_a, root_b = tmp_path/"r1", tmp_path/"r2"
    write_matrix(root_a, seed=1610, runtime="a"*64)
    write_matrix(root_b, seed=1630, runtime="b"*64)
    with pytest.raises(ValueError, match="mixed physical runtimes"):
        load_matrices([root_a, root_b], [1610, 1630])


def test_unallowed_layout_runtime_does_not_leak_into_training_audit(tmp_path):
    write_matrix(tmp_path, seed=1610, runtime="a"*64)
    write_matrix(tmp_path, seed=1660, runtime="b"*64)
    rows, manifest = load_matrices([tmp_path], [1610])
    assert len(rows) == 1 and {r["runtime_sha256"] for r in manifest} == {"a"*64}


def test_native_checkpoint_cannot_evaluate_a_new_physics_revision_as_same_experiment():
    manifest = [dict(runtime_sha256="a"*64)]
    saved = dict(label_domain="printed_kit_v12", physical_runtime_sha256="a"*64)
    validate_checkpoint_runtime(saved, manifest)
    for wrong in ("b"*64, None):
        with pytest.raises(ValueError, match="different or unbound"):
            validate_checkpoint_runtime({**saved, "physical_runtime_sha256":wrong}, manifest)
