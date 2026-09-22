"""Contract/gradient checks only; synthetic fixtures are not training evidence."""
import copy
import math

import numpy as np
import pytest

from scripts.train_value_v12 import evaluate, historical_graph, local_labels, selection_key
from simbench.value.graph_value_v12 import STAGES, VALUE_RELATIONS, encode_graph, state_plan_interactions
from tests.test_value_v12 import sample


def interaction_fixture():
    graph = historical_graph(sample())
    graph["planning_cad"] = dict(parts={p: dict(dimensions_m=[.04, .04, .03]) for p in STAGES[1:6]},
        pin_hole_width_m=.008, pin_shaft_radius_m=.0033,
        handle_bore_radius_m=.0061, handle_post_radius_m=.0055)
    stages = {s: [] for s in STAGES}
    for node in graph["assembly"]["nodes"]:
        part = node["roles"].get("manipulated")
        if part in stages:
            stages[part].append(node)
    order = ["cleaning", *graph["assembly"]["plan"]["prefix"]["order"], "bidirectional_stroke", "pin_retention"]
    return graph, stages, order


def test_cad_clearance_and_uncertainty_have_distinct_missing_masks():
    graph, stages, order = interaction_fixture()
    original, _ = state_plan_interactions(graph, stages, order, set())
    narrower = copy.deepcopy(graph)
    narrower["planning_cad"]["pin_hole_width_m"] = .0068
    changed, _ = state_plan_interactions(narrower, stages, order, set())
    assert not np.array_equal(original["pin_left"], changed["pin_left"])
    del narrower["planning_cad"]["pin_hole_width_m"]
    unknown, _ = state_plan_interactions(narrower, stages, order, set())
    assert not np.array_equal(unknown["pin_left"], changed["pin_left"])
    assert np.isfinite(np.asarray(list(unknown.values()))).all()


def test_future_placement_changes_later_input_without_using_rollout_metadata():
    graph, stages, order = interaction_fixture()
    original, _ = state_plan_interactions(graph, stages, order, set())
    graph["assembly"]["observation"]["assembly_targets"] = {
        "carriage": dict(position_m=[.5, .4, .83], quat_wxyz=[1, 0, 0, 0])}
    changed, _ = state_plan_interactions(graph, stages, order, set())
    assert not np.array_equal(original["handle"], changed["handle"])
    graph.update(success=True, candidate_id=9999, seed=2200, result=dict(success=True), final_positions={})
    metadata, _ = state_plan_interactions(graph, stages, order, set())
    np.testing.assert_array_equal(changed["handle"], metadata["handle"])


def test_initial_loose_stop_route_does_not_become_observed_installed_failure():
    graph, stages, order = interaction_fixture()
    relation = dict(observable=True, geometric_route_exists=False, holes={"pin_left": dict(geometric_margin_m=-.01)})
    graph["assembly"]["observation"]["fixture_relations"] = dict(end_stop_to_base=relation)
    pending, _ = state_plan_interactions(graph, stages, order, set())
    relation["geometric_route_exists"] = True
    relation["holes"]["pin_left"]["geometric_margin_m"] = .001
    still_pending, _ = state_plan_interactions(graph, stages, order, set())
    np.testing.assert_array_equal(pending["pin_left"], still_pending["pin_left"])
    installed, _ = state_plan_interactions(graph, stages, order, {"end_stop"})
    assert not np.array_equal(still_pending["pin_left"], installed["pin_left"])


def test_placement_orientation_is_relative_to_pickup_and_object():
    graph, stages, order = interaction_fixture()
    obs = graph["assembly"]["observation"]
    obs["assembly_targets"] = {"carriage": dict(position_m=[.1, .08, .83], quat_wxyz=[1, 0, 0, 0])}
    original, _ = state_plan_interactions(graph, stages, order, set())
    obs["objects"]["carriage"]["quaternion"] = [math.cos(.3), 0., 0., math.sin(.3)]
    changed, _ = state_plan_interactions(graph, stages, order, set())
    assert not np.array_equal(original["carriage"], changed["carriage"])


def test_object_relative_grasp_preserves_goal_orientation_when_supply_rotates():
    from simbench.value.graph_value_v12 import pickup_yaw_state
    graph, stages, order = interaction_fixture()
    obs = graph["assembly"]["observation"]
    obs["assembly_targets"] = {"carriage": dict(position_m=[.1, .08, .83], quat_wxyz=[1, 0, 0, 0])}
    grasp = next(node for node in stages["carriage"] if any(p["name"] == "yaws" for p in node["ports"]))
    next(p for p in grasp["ports"] if p["name"] == "yaws")["frame"] = "object"
    frames = [p for p in grasp["ports"] if p["name"] == "yaw_frame"]
    if frames:
        frames[0].update(status="known", value="object")
    else:
        grasp["ports"].append(dict(name="yaw_frame", status="known", value="object"))
    original, _ = state_plan_interactions(graph, stages, order, set())
    obj = obs["objects"]["carriage"]
    command, frame, world = pickup_yaw_state(stages["carriage"], obj)
    obj["quaternion"] = [math.cos(.3), 0., 0., math.sin(.3)]
    changed, _ = state_plan_interactions(graph, stages, order, set())
    # Same relative closing axis and fixed receiving orientation retain the
    # intended final body orientation, despite different supply yaw.
    np.testing.assert_allclose(original["carriage"], changed["carriage"], atol=1.e-12)
    assert frame == "object"
    assert pickup_yaw_state(stages["carriage"], obj)[2] == pytest.approx(world+.6)
    obj["quaternion"] = None
    assert pickup_yaw_state(stages["carriage"], obj) == (command, "object", None)


def test_base_bridge_requirement_and_new_depth_ports_are_not_dropped():
    graph, stages, order = interaction_fixture()
    original, _ = state_plan_interactions(graph, stages, order, set())
    graph["assembly"]["observation"]["goals"][0]["pin_base_bridge_required"] = True
    changed, _ = state_plan_interactions(graph, stages, order, set())
    assert not np.array_equal(original["pin_left"], changed["pin_left"])
    from simbench.value.graph_value_v12 import PORTS
    assert {"pin_command_depth_m", "pin_press_extra_m", "width"}.issubset(dict(PORTS))


def test_receiver_fields_survive_compilation_and_change_input_hash():
    from simbench.value.plan import PlanIR, digest
    from simbench.value.skill_graph import compile_graph
    graph = historical_graph(sample())
    obs = graph["assembly"]["observation"]
    obs["assembly_targets"] = {"handle": dict(position_m=[.12, .08, .88], quat_wxyz=[1, 0, 0, 0])}
    graph["assembly"] = compile_graph(obs, PlanIR.from_dict(graph["assembly"]["plan"]))
    assert "assembly_targets" in graph["assembly"]["observation"]
    encoded = encode_graph(graph)
    assert np.isfinite(encoded["x"]).all()
    # The observation is a declared pre-execution input (not secret truth);
    # target changes are legitimate new inputs and must change the readout.
    changed = copy.deepcopy(graph)
    changed["assembly"]["observation"]["assembly_targets"]["handle"]["position_m"][0] += .05
    assert digest(changed) != digest(graph)
    assert not np.array_equal(encoded["x"], encode_graph(changed)["x"])


def test_utility_selection_prioritizes_hit_then_calls_and_ignores_cached_time():
    a = dict(success=.75, normalized_first_success_calls=.5, brier=.3, verification_seconds=1e9)
    b = dict(success=.5, normalized_first_success_calls=.1, brier=.01, verification_seconds=0.)
    assert selection_key(a) < selection_key(b)
    c = {**a, "normalized_first_success_calls": .4, "brier": .5}
    assert selection_key(c) < selection_key(a)
    assert selection_key({**a, "verification_seconds": 0.}) == selection_key(a)


def test_evaluation_normalizes_each_pool_and_retains_impossible_layouts():
    rows = [dict(seed=seed, source="s", name=f"candidate_{i}", y=float(i == positive),
                 outcomes=[dict(condition="c", success=i == positive, seconds=1.)])
            for seed, size, positive in ((1, 2, 1), (2, 4, -1)) for i in range(size)]
    report = evaluate(rows, [0., 1., 4., 3., 2., 1.], k=1)
    assert report["success"] == .5
    assert report["normalized_first_success_calls"] == .75
    assert report["hit_at_k_given_feasible"] == 1.


def test_retention_failure_is_not_insertion_failure_and_unknown_stages_stay_masked():
    y, mask = local_labels(dict(success=False, stage_passes={}, executed_parameters=[
        dict(skill="inspect_pin_inserted", params=dict(part="pin_left"), ok=True),
        dict(skill="inspect", params=dict(part="pin_left", phase="retained_after_stroke"), ok=False)]))
    assert (y[3], mask[3]) == (1., 1.)
    assert (y[7], mask[7]) == (0., 1.)
    assert mask[0] == mask[6] == 0.


@pytest.mark.parametrize("kind", ["shared", "graph"])
def test_auxiliary_gradients_reach_shared_plan_representation(kind):
    torch = pytest.importorskip("torch")
    from simbench.value.value_v12 import StageValueNet, collate
    torch.manual_seed(1)
    encoded = encode_graph(historical_graph(sample()))
    model = StageValueNet(encoded["x"].shape[-1], kind=kind)
    output = model(collate([encoded, encoded]))
    loss = torch.nn.functional.binary_cross_entropy_with_logits(output["local_logits"], torch.ones(2, 8))
    loss.backward()
    assert model.embed[0].weight.grad.abs().sum() > 0
    assert model.config["relation_types"] == len(VALUE_RELATIONS)


def test_censored_stage_values_do_not_affect_loss_or_gradient():
    torch = pytest.importorskip("torch")
    from simbench.value.value_v12 import supervised_loss
    prediction = dict(plan_logit=torch.tensor([0., 1.], requires_grad=True),
                      local_logits=torch.zeros(2, 8, requires_grad=True))
    y = torch.zeros(2, 8)
    mask = torch.zeros(2, 8); mask[:, 0] = 1.
    first = supervised_loss(prediction, torch.tensor([0., 1.]), y, mask)
    y[:, 1:] = 1.
    second = supervised_loss(prediction, torch.tensor([0., 1.]), y, mask)
    assert torch.equal(first, second)
    second.backward()
    assert not prediction["local_logits"].grad[:, 1:].any()
