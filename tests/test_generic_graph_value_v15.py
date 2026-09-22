"""V15 input-contract tests.  Synthetic graphs are never training evidence."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from simbench.value.generic_graph_value_v15 import (
    AtomicValueNetV15,
    FEATURES,
    MAX_NODES,
    RELATIONS,
    collate,
    encode_graph,
)
from simbench.value.plan import Call, PlanIR, argument, digest
from simbench.value.skill_graph import compile_graph
from simbench.value.stage_v5 import PRECEDENCE, nominal_targets, stage_calls
from simbench.value.v9_candidates import proposals


def executable_graph():
    proposal = copy.deepcopy(proposals()[0])
    targets = nominal_targets()
    calls = [call for i, part in enumerate(proposal["order"])
             for call in stage_calls(part, targets[part], proposal["choices"][part], i,
                                     v7=True, functional_clearance=True)]
    calls.append(Call("final_home", "move", dict(target=argument("home"))))
    plan_id = digest(proposal)[:20]
    prefix = dict(id=plan_id, part=proposal["order"][0], execution="program",
                  start_state="synthetic_contract_fixture", order=proposal["order"],
                  choices=proposal["choices"], targets=targets,
                  steps=[dict(skill=call.skill,
                              params={key: value.value for key, value in call.arguments.items()})
                         for call in calls[:10]])
    plan = PlanIR(plan_id, calls, 10, prefix, "unknown",
                  protocol="assembly.program.feedback.v2")
    parts = (*proposal["order"], "wipe_tool")
    observation = dict(robot=dict(joints=[0.] * 7, eef=[0., 0., .9]),
        objects={part: dict(position=[i * .02, -.2, .85], quaternion=[1., 0., 0., 0.],
                                 valid=True, quality=.8, fit_residual_m=.001)
                 for i, part in enumerate(parts)},
        goals=[dict(predicate="functional_task")])
    graph = dict(assembly=compile_graph(observation, plan),
                 phase_edges=[["cleaning", "carriage"], *map(list, PRECEDENCE)])
    return graph


def test_dynamic_atomic_graph_has_no_fixed_stage_axis():
    graph = executable_graph()
    encoded = encode_graph(graph)
    assert encoded["x"].shape == (len(graph["assembly"]["nodes"]), len(FEATURES))
    assert encoded["relations"].shape == (len(RELATIONS), len(encoded["x"]), len(encoded["x"]))
    # A different graph length remains legal: batching pads to the observed
    # maximum instead of assuming eight assembly stages.
    shorter = copy.deepcopy(graph)
    shorter["assembly"]["nodes"] = shorter["assembly"]["nodes"][:-3]
    shorter["assembly"]["edges"] = [edge for edge in shorter["assembly"]["edges"]
                                        if edge["source"] < len(shorter["assembly"]["nodes"])
                                        and edge["target"] < len(shorter["assembly"]["nodes"])]
    second = encode_graph(shorter, check=False)
    assert len(second["x"]) == len(encoded["x"]) - 3


def test_result_candidate_name_and_seed_cannot_leak_into_input():
    graph = executable_graph()
    baseline = encode_graph(graph)["x"]
    graph.update(candidate_id="success_999", seed=1970, success=True,
                 result=dict(success=True), cached_rollout_seconds=0.)
    np.testing.assert_array_equal(baseline, encode_graph(graph)["x"])


def test_registered_port_and_observation_changes_are_visible():
    graph = executable_graph()
    baseline = encode_graph(graph)["x"]
    changed = copy.deepcopy(graph)
    port = next(port for node in changed["assembly"]["nodes"] for port in node.get("ports", ())
                if port.get("status") == "known" and isinstance(port.get("value"), (int, float))
                and not isinstance(port.get("value"), bool))
    port["value"] += .123
    assert not np.array_equal(baseline, encode_graph(changed, check=False)["x"])
    moved = copy.deepcopy(graph)
    manipulated = next(node["roles"]["manipulated"] for node in moved["assembly"]["nodes"]
                       if node.get("roles", {}).get("manipulated") in moved["assembly"]["observation"]["objects"])
    moved["assembly"]["observation"]["objects"][manipulated]["position"][0] += .05
    assert not np.array_equal(baseline, encode_graph(moved, check=False)["x"])


def test_vector_port_direction_is_not_reduced_to_its_mean():
    graph = executable_graph()
    port = next(port for node in graph["assembly"]["nodes"] for port in node.get("ports", ())
                if isinstance(port.get("value"), list) and len(port["value"]) >= 2
                and all(isinstance(value, (int, float)) for value in port["value"])
                and port["value"][0] != port["value"][1])
    changed = copy.deepcopy(graph)
    target = next(candidate for node in changed["assembly"]["nodes"] for candidate in node.get("ports", ())
                  if candidate["name"] == port["name"] and candidate.get("value") == port["value"])
    target["value"][0], target["value"][1] = target["value"][1], target["value"][0]
    assert sum(target["value"]) == pytest.approx(sum(port["value"]))
    assert not np.array_equal(encode_graph(graph)["x"], encode_graph(changed, check=False)["x"])


def test_invalid_size_and_nonfinite_input_are_rejected():
    graph = executable_graph()
    graph["assembly"]["nodes"] *= MAX_NODES + 1
    with pytest.raises(ValueError, match="node count"):
        encode_graph(graph, check=False)
    graph = executable_graph()
    graph["assembly"]["observation"]["robot"]["joints"][0] = float("nan")
    with pytest.raises(ValueError, match="non-finite robot joints"):
        encode_graph(graph, check=False)


def test_variable_size_batch_and_model_are_order_equivariant():
    torch = pytest.importorskip("torch")
    full = encode_graph(executable_graph())
    small = copy.deepcopy(full)
    small["x"] = small["x"][:-4]
    small["active"] = small["active"][:-4]
    small["relations"] = small["relations"][:, :-4, :-4]
    batch = collate([full, small])
    assert batch["x"].shape[1] == len(full["x"])
    torch.manual_seed(15)
    model = AtomicValueNetV15.build().eval()
    with torch.inference_mode():
        first = model(batch)
        reverse = model(collate([small, full]))
    torch.testing.assert_close(first, reverse.flip(0))
