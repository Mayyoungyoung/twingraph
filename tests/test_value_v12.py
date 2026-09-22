import copy
import itertools
import math

import numpy as np
import pytest

from scripts.train_value_v12 import historical_graph, local_labels, random_metrics
from simbench.value.graph_value_v12 import encode_graph, path_yaw_features
from simbench.value.v9_candidates import proposals


def sample(yaw=0.):
    proposal = copy.deepcopy(proposals()[0])
    proposal["choices"]["pin_left"]["yaw"] = yaw
    objects = {p: dict(position_m=[i*.02, -.2, .85], quat_wxyz=[1., 0., 0., 0.],
                      valid=True, quality=.8, fit_residual_m=.001)
               for i, p in enumerate(("carriage", "end_stop", "pin_left", "pin_right", "handle", "wipe_tool"))}
    return dict(candidate=proposal, pre_execution_observation=dict(objects=objects))


def test_periodic_angles_and_candidate_metadata_are_inert():
    a, b = sample(0.), sample(2*math.pi)
    b["candidate"]["name"] = "cannot leak label or candidate index"
    x = encode_graph(historical_graph(a))["x"]
    y = encode_graph(historical_graph(b))["x"]
    np.testing.assert_allclose(x, y, atol=1.e-6)


def test_missing_state_and_new_parameter_are_finite_and_distinct():
    a, b = sample(), sample()
    b["pre_execution_observation"]["objects"]["pin_left"]["fit_residual_m"] = None
    b["candidate"]["choices"]["pin_left"]["force"] = 9.
    x, y = encode_graph(historical_graph(a))["x"], encode_graph(historical_graph(b))["x"]
    assert np.isfinite(y).all() and np.max(abs(y)) <= 10
    assert not np.array_equal(x, y)


def test_graph_tamper_is_rejected():
    g = historical_graph(sample())
    g["assembly"]["nodes"][0]["implementation"] = "different_execution"
    with pytest.raises(ValueError, match="mismatch"):
        encode_graph(g)


def test_path_yaws_preserve_placement_direction_and_deferred_mask():
    nodes = [{"ports": [{"name": "yaw", "status": "known", "value": 0.}]},
             {"ports": [{"name": "yaw", "status": "known", "value": math.pi/2}]}]
    a = np.asarray(path_yaw_features(nodes))
    np.testing.assert_allclose(a[5:10], [1., 0., 1., 0., 1.], atol=1.e-7)
    nodes[1]["ports"][0]["value"] += 2*math.pi
    np.testing.assert_allclose(a, path_yaw_features(nodes), atol=1.e-7)
    nodes[1]["ports"][0].update(value=None, status="deferred")
    np.testing.assert_array_equal(path_yaw_features(nodes)[5:10], [0., 0., 0., 1., 1.])
    with pytest.raises(ValueError, match="too many path yaw"):
        path_yaw_features(nodes*3)


def test_previous_feature_schema_checkpoint_is_rejected(tmp_path):
    torch = pytest.importorskip("torch")
    from simbench.value.graph_value_v12 import ValueRankerV12
    checkpoint = tmp_path/"legacy.pt"
    torch.save(dict(schema="twingraph.graph_stage_ports.v12"), checkpoint)
    with pytest.raises(ValueError, match="incompatible value input schema"):
        ValueRankerV12(checkpoint)


def test_compiled_stage_placement_yaw_reaches_value_slots():
    from simbench.value.stage_v5 import stage_calls
    choice = dict(yaw=.1, placement_yaw=.7, height=.001, clearance=1., force=4.,
                  speed=.012, force_limit=12., press_force=2.)
    calls = stage_calls("carriage", [.1, .08, .82], choice, 0, v7=True, v12=True)
    nodes = [{"ports": [dict(name=k, value=a.value, status=a.status) for k, a in call.arguments.items()]}
             for call in calls]
    values = path_yaw_features(nodes)
    np.testing.assert_allclose(values[:5], [math.sin(.1), math.cos(.1), 1., 0., 1.])
    np.testing.assert_allclose(values[5:10], [math.sin(.7), math.cos(.7), 1., 0., 1.])


def test_approach_strategy_and_completed_phase_are_encoded():
    a, b = sample(), sample()
    a["candidate"]["choices"]["carriage"]["approach_strategy"] = "cartesian"
    b["candidate"]["choices"]["carriage"]["approach_strategy"] = "joint_checked_v10"
    ga, gb = historical_graph(a), historical_graph(b)
    assert not np.array_equal(encode_graph(ga)["x"], encode_graph(gb)["x"])
    ga["completion"]["completed"] = ["stroke", "retention"]
    np.testing.assert_array_equal(encode_graph(ga)["active"][-2:], [0., 0.])


def test_local_supervision_censors_unexecuted_stages():
    y, mask = local_labels(dict(success=False, stage_passes=dict(cleaning_pass=True),
        executed_parameters=[dict(skill="inspect_placement", params=dict(part="carriage"), ok=True),
                             dict(skill="approach", params=dict(part="end_stop"), ok=False)]))
    np.testing.assert_array_equal(y[:3], [1., 1., 0.])
    np.testing.assert_array_equal(mask[:3], [1., 1., 1.])
    assert not mask[3:].any()


def test_random_expectation_matches_exhaustive_permutations():
    success, seconds, k = [False, True, False, True], [1., 4., 2., 6.], 3
    actual = []
    for order in itertools.permutations(range(4)):
        attempted = []
        for i in order[:k]:
            attempted.append(i)
            if success[i]:
                break
        actual.append((success[attempted[-1]], len(attempted), sum(seconds[i] for i in attempted)))
    expected = random_metrics(success, seconds, k)
    np.testing.assert_allclose([expected[key] for key in ("success", "calls", "seconds")], np.mean(actual, 0))


def test_message_passing_uses_graph_relations():
    torch = pytest.importorskip("torch")
    from simbench.value.value_v12 import StageValueNet, collate
    torch.manual_seed(1212)
    row = encode_graph(historical_graph(sample()))
    changed = copy.deepcopy(row)
    changed["relations"][:] = 0.
    model = StageValueNet(row["x"].shape[-1], kind="graph").eval()
    with torch.inference_mode():
        scores = model(collate([row, changed]))["plan_logit"]
    assert not torch.isclose(scores[0], scores[1], atol=1.e-6)
