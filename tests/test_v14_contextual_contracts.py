"""V14 ranking contracts; synthetic checks are not experimental evidence."""
import copy

import numpy as np
import pytest

from scripts.train_value_v14_contextual import reversal_targets
from simbench.value.graph_value_v12 import encode_graph
from tests.test_value_v12_interactions import interaction_fixture


def test_candidate_identity_metadata_does_not_change_encoding():
    graph, _, _ = interaction_fixture()
    original = encode_graph(graph)
    changed = copy.deepcopy(graph)
    changed["proposal"] = dict(name="renamed_only", source="another_source", rationale="different prose")
    encoded = encode_graph(changed)
    for key in ("x", "relations", "active"):
        np.testing.assert_array_equal(original[key], encoded[key])


def test_executable_strategy_and_speed_change_encoding():
    graph, _, _ = interaction_fixture()
    original = encode_graph(graph, check=False)["x"]
    changed = copy.deepcopy(graph)
    ports = [port for node in changed["assembly"]["nodes"] for port in node["ports"]]
    strategy = next(port for port in ports if port["name"] == "strategy")
    strategy.update(status="known", value="joint_checked_v10")
    speed = next(port for port in ports if port["name"] == "speed")
    speed.update(status="known", value=float(speed.get("value") or .004) * .5)
    assert not np.array_equal(original, encode_graph(changed, check=False)["x"])


def test_reversal_targets_require_both_training_directions():
    rows = [dict(seed=seed, name=name, y=float(label))
            for seed, labels in ((1, {"a": 1, "b": 0, "c": 0}),
                                 (2, {"a": 0, "b": 1, "c": 0}))
            for name, label in labels.items()]
    targets, pairs = reversal_targets(rows)
    assert {tuple((p["candidate_a"], p["candidate_b"])) for p in pairs} == {("a", "b")}
    assert targets[1] == [("a", "b")]
    assert targets[2] == [("b", "a")]


def test_graph_scores_are_batch_order_invariant_and_roundtrip(tmp_path):
    torch = pytest.importorskip("torch")
    from simbench.value.value_v12 import StageValueNet, collate
    graph, _, _ = interaction_fixture()
    changed = copy.deepcopy(graph)
    port = next(port for node in changed["assembly"]["nodes"] for port in node["ports"]
                if port["name"] == "speed")
    port.update(status="known", value=float(port.get("value") or .004) * .5)
    encoded = [encode_graph(graph, check=False), encode_graph(changed, check=False)]
    torch.manual_seed(14)
    model = StageValueNet(encoded[0]["x"].shape[-1], kind="graph").eval()
    with torch.inference_mode():
        direct = model(collate(encoded))["plan_logit"]
        reverse = model(collate(encoded[::-1]))["plan_logit"].flip(0)
    torch.testing.assert_close(direct, reverse)
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)
    restored = StageValueNet(encoded[0]["x"].shape[-1], kind="graph").eval()
    restored.load_state_dict(torch.load(path, weights_only=True))
    with torch.inference_mode():
        roundtrip = restored(collate(encoded))["plan_logit"]
    torch.testing.assert_close(direct, roundtrip)
