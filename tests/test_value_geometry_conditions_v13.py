import copy

import numpy as np
import pytest

from scripts.train_value_v12 import historical_graph
from simbench.value.graph_value_v12 import encode_graph, geometry_condition_features
from simbench.value.plan import PlanIR, digest
from simbench.value.skill_graph import compile_graph, validate_geometry_conditions
from tests.test_value_v12 import sample


def bound_graph():
    fixture = sample()
    fixture["candidate"]["choices"]["carriage"]["width"] = .028
    graph = historical_graph(fixture)
    observation = graph["assembly"]["observation"]
    observation["perception"] = dict(observation_sha256="pre_execution_rgbd")
    plan = PlanIR.from_dict(graph["assembly"]["plan"])
    graph["assembly"] = compile_graph(observation, plan)
    graph["planning_cad"] = dict(source="declared fixture CAD")
    choices = plan.prefix["choices"]
    graph["geometry_conditions"] = dict(schema="twingraph.geometry_conditions.v13",
        plan_sha256=graph["assembly"]["plan_sha256"], choices_sha256=digest(choices),
        order_sha256=digest(plan.prefix["order"]), observation_sha256="pre_execution_rgbd",
        cad_sha256=digest(graph["planning_cad"]), parts={"carriage": dict(
            part="carriage", yaw=choices["carriage"]["yaw"], height=choices["carriage"]["height"],
            status="unknown", min_clearance_m=.001, required_clearance_m=.003,
            source_grasp_ik_checked=False, grasp_width_m=.028)})
    return graph


def test_shared_catalogue_changes_features_without_label_or_candidate_name():
    graph = bound_graph()
    original = encode_graph(graph)["x"]
    graph["geometry_conditions"]["parts"]["carriage"]["min_clearance_m"] = .002
    changed = encode_graph(graph)["x"]
    assert not np.array_equal(original, changed)
    graph.update(result=dict(success=True), learned_value=1., seed=2200)
    np.testing.assert_array_equal(changed, encode_graph(graph)["x"])


@pytest.mark.parametrize("binding", ["plan_sha256", "choices_sha256", "order_sha256", "observation_sha256", "cad_sha256"])
def test_stale_catalogue_bindings_are_rejected(binding):
    graph = bound_graph(); graph["geometry_conditions"][binding] = "stale"
    with pytest.raises(ValueError, match=f"geometry_conditions {binding}"):
        encode_graph(graph)


def test_current_observation_cad_and_selected_command_must_match_readout():
    graph = bound_graph(); graph["planning_cad"]["source"] = "different"
    with pytest.raises(ValueError, match="cad_sha256"):
        encode_graph(graph)
    graph = bound_graph(); graph["geometry_conditions"]["parts"]["carriage"]["yaw"] += .5
    with pytest.raises(ValueError, match="differs from executable choice"):
        encode_graph(graph)
    graph = bound_graph(); graph["assembly"]["observation"]["perception"]["observation_sha256"] = "new"
    with pytest.raises(ValueError, match="observation_sha256"):
        encode_graph(graph)
    graph = bound_graph(); graph["geometry_conditions"]["parts"]["carriage"]["grasp_width_m"] += .01
    with pytest.raises(ValueError, match="grasp_width_m differs"):
        encode_graph(graph)


def test_relative_catalogue_frame_and_evaluated_world_yaw_are_bound():
    import math
    fixture = sample()
    fixture["candidate"]["choices"]["carriage"].update(yaw=.5, grasp_yaw_frame="object")
    fixture["pre_execution_observation"]["objects"]["carriage"]["quat_wxyz"] = [math.cos(.15), 0, 0, math.sin(.15)]
    graph = historical_graph(fixture)
    observation = graph["assembly"]["observation"]
    observation["perception"] = dict(observation_sha256="relative-frame-fixture")
    plan = PlanIR.from_dict(graph["assembly"]["plan"])
    graph["assembly"] = compile_graph(observation, plan)
    graph["geometry_conditions"] = dict(schema="twingraph.geometry_conditions.v13",
        plan_sha256=graph["assembly"]["plan_sha256"], choices_sha256=digest(plan.prefix["choices"]),
        order_sha256=digest(plan.prefix["order"]), observation_sha256="relative-frame-fixture",
        cad_sha256=digest({}), parts={"carriage": dict(yaw=.5, grasp_yaw_frame="object",
            evaluated_world_yaw_rad=.8, status="unknown")})
    encode_graph(graph)
    row = graph["geometry_conditions"]["parts"]["carriage"]
    row["grasp_yaw_frame"] = "world"
    with pytest.raises(ValueError, match="grasp_yaw_frame"):
        encode_graph(graph)
    row["grasp_yaw_frame"] = "object"
    row["evaluated_world_yaw_rad"] = .5
    with pytest.raises(ValueError, match="evaluated world yaw"):
        encode_graph(graph)


def test_below_reserve_is_unknown_not_a_fake_clearance_certificate():
    graph = bound_graph()
    row = graph["geometry_conditions"]["parts"]["carriage"]
    validate_geometry_conditions(graph)
    known_small = geometry_condition_features(row)
    missing = geometry_condition_features({**row, "min_clearance_m": None})
    assert known_small != missing
    row["status"] = "necessary_pass"
    with pytest.raises(ValueError, match="clearance reserve"):
        encode_graph(graph)


@pytest.mark.parametrize("schema", ["twingraph.graph_stage_ports.v12.r3_state_plan_interaction",
    "twingraph.graph_stage_ports.v12.r4_geometry_catalog"])
def test_previous_interaction_schema_is_rejected_before_weight_loading(tmp_path, schema):
    torch = pytest.importorskip("torch")
    from simbench.value.graph_value_v12 import ValueRankerV12
    path = tmp_path/"old.pt"
    torch.save(dict(schema=schema), path)
    with pytest.raises(ValueError, match="incompatible value input schema"):
        ValueRankerV12(path)


def test_missing_catalogue_keeps_fixed_dimension_with_explicit_masks():
    graph = bound_graph()
    with_catalogue = encode_graph(graph)["x"]
    del graph["geometry_conditions"]
    without = encode_graph(graph)["x"]
    assert with_catalogue.shape == without.shape
    assert not np.array_equal(with_catalogue, without)
