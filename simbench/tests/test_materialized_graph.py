"""Scoring and execution share every waypoint of a preplanned free path."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from simbench.value.plan import PlanIR, initial_artifacts, materialize_initial_path
from simbench.value.research_scenarios import program
from simbench.value.skill_graph import compile_graph, validate_graph
from simbench.value.graph_encode import encode_graph


def example():
    data = SimpleNamespace(qpos=np.zeros(7), qvel=np.zeros(7), ctrl=np.zeros(7))
    model = SimpleNamespace(geom_size=np.zeros((1, 3)), geom_pos=np.zeros((1, 3)),
                            geom_friction=np.ones((1, 3)))
    session = SimpleNamespace(parts=["part"], held=None, grasp_epoch=0, artifacts={},
                              ctx=SimpleNamespace(data=data, model=model, arm_qpos=np.zeros(7)))
    choices = {"part": dict(yaw=0., height=.003, clearance=1., force=3., speed=.006)}
    plan = program(session, {"part": [.04, 0., .85]}, ["part"], choices)
    observation = dict(robot=dict(joints=[0.] * 7, eef=[0., 0., 1.]),
                       objects={"part": dict(position=[-.2, 0., .85], quaternion=[1., 0., 0., 0.], geoms=[])},
                       goals=[dict(predicate="seated", manipulated="part", position=[.04, 0., .85])])
    path = dict(type="joint_path", part=None, id="free:route:0", start_q=[0.] * 7,
                joints=(np.arange(35)[:, None] * np.ones((1, 7)) * .001).tolist(),
                target=[-.2, 0., .98], rotation=np.eye(3).tolist(),
                binding=dict(held=None, grasp_epoch=0, grasp_artifact=None, grasp_id=None, prefix_id=None))
    return session, observation, plan, path


def test_materialized_path_is_a_known_execution_port_and_later_path_is_deferred():
    session, observation, original, path = example()
    plan = materialize_initial_path(session, original, path)
    assert len(plan.calls) == len(original.calls) - 1
    assert plan.boundary == original.boundary - 1
    assert plan.id != original.id
    assert "initial_artifacts" not in original.prefix
    assert PlanIR.from_dict(plan.to_dict()).to_dict() == plan.to_dict()
    graph = compile_graph(observation, plan)
    assert validate_graph(graph).to_dict() == plan.to_dict()
    first = graph["nodes"][4]["ports"][0]
    assert first["name"] == "artifact" and first["status"] == "known"
    assert first["materialized"]["joints"] == path["joints"]
    later = graph["nodes"][10]["ports"][0]
    assert "materialized" not in later and later["status"] == "deferred"
    assert later["source"]["call"] == 9


def test_all_waypoints_reach_model_and_metadata_identifiers_do_not():
    from simbench.value.encode import bucket
    session, observation, plan, path = example()
    plan = materialize_initial_path(session, plan, path)
    before = encode_graph(compile_graph(observation, plan))
    for i, point in enumerate(path["joints"]):
        key = bucket(f"materialized/artifact/joints/{i}|joint_path|rad|robot_joint")
        matches = np.where(before["keys"] == key)[0]
        assert len(matches) > 0, f"missing waypoint {i}"
        assert any(np.allclose(before["numbers"][j, :7], np.tanh(point)) for j in matches)
    # Deliberately beyond the old 24-waypoint sampling limit, away from endpoints.
    for i in range(len(path["joints"])):
        changed = copy.deepcopy(plan)
        changed.prefix["initial_artifacts"]["transfer"]["joints"][i][3] += .002
        after = encode_graph(compile_graph(observation, changed))
        assert not np.array_equal(before["numbers"], after["numbers"]), i
    renamed = copy.deepcopy(plan)
    renamed.id = renamed.prefix["id"] = "renamed_candidate"
    artifact = renamed.prefix["initial_artifacts"]["transfer"]
    artifact["id"] = "renamed_route"
    artifact["binding"]["prefix_id"] = renamed.id
    after = encode_graph(compile_graph(observation, renamed))
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])


@pytest.mark.parametrize("mutation", ["waypoint", "binding", "nonfinite", "label", "shape", "grasp", "string"])
def test_artifact_integrity_and_schema_fail_closed(mutation):
    session, observation, plan, path = example()
    plan = materialize_initial_path(session, plan, path)
    graph = compile_graph(observation, plan)
    item = graph["plan"]["prefix"]["initial_artifacts"]["transfer"]
    if mutation == "waypoint": item["joints"][0][0] += .01
    elif mutation == "binding": item["binding"]["prefix_id"] = "another_program"
    elif mutation == "nonfinite": item["joints"][0][0] = float("nan")
    elif mutation == "label": item["success"] = True
    elif mutation == "shape": item["joints"][0].pop()
    elif mutation == "grasp": item["binding"]["grasp_id"] = "nominal_future_grasp"
    else: item["joints"][0][0] = "0.1"
    with pytest.raises(ValueError): validate_graph(graph)


def test_no_artifact_serialization_change_for_old_programs():
    _, observation, plan, _ = example()
    old = plan.to_dict()
    graph = compile_graph(observation, plan)
    assert "initial_artifacts" not in old["prefix"]
    plan.prefix["initial_artifacts"] = {}
    assert plan.to_dict() == old
    assert compile_graph(observation, plan) == graph


def test_runtime_receives_independent_exact_artifact_copy(monkeypatch):
    import simbench.value.plan as implementation
    session, _, plan, path = example()
    plan = materialize_initial_path(session, plan, path)
    captured = []
    def execute(s, p, calls):
        captured.append(copy.deepcopy(s.artifacts["transfer"]))
        s.artifacts["transfer"]["joints"][0, 0] = 999.
    monkeypatch.setattr(implementation, "execute_calls", execute)
    implementation.execute_prefix(session, plan)
    np.testing.assert_array_equal(captured[0]["joints"], path["joints"])
    assert initial_artifacts(plan)["transfer"]["joints"] == path["joints"]
    assert session.active_candidate_id == plan.id


def test_materialization_rejects_stale_start_and_future_grasp():
    session, _, plan, path = example()
    stale = copy.deepcopy(path); stale["start_q"][0] += .1
    with pytest.raises(ValueError, match="stale joint-space start"):
        materialize_initial_path(session, plan, stale)
    bound = copy.deepcopy(path); bound["binding"]["grasp_id"] = "future"
    with pytest.raises(ValueError, match="another grasp acquisition"):
        materialize_initial_path(session, plan, bound)


def test_free_path_after_actual_grasp_checkpoint_preserves_epoch(monkeypatch):
    import simbench.value.plan as implementation
    session, observation, plan, path = example()
    session.grasp_epoch = 4
    session.artifacts["grasp"] = dict(type="grasp", id="old_grasp", part="other_part")
    path["binding"]["grasp_epoch"] = 4
    plan = materialize_initial_path(session, plan, path)
    assert plan.prefix["initial_grasp_epoch"] == 4
    validate_graph(compile_graph(observation, plan))
    monkeypatch.setattr(implementation, "execute_calls", lambda *args: None)
    implementation.execute_prefix(session, plan)
    session.grasp_epoch += 2
    with pytest.raises(ValueError, match="grasp acquisition changed"):
        implementation.execute_prefix(session, plan)


def test_actual_robot_reuses_supplied_joint_waypoints(monkeypatch):
    import simbench.value.plan as implementation
    from simbench.assembly.control import make_context
    from simbench.assembly.library import Session
    from simbench.assembly.candidates import transfer_routes

    session = Session(make_context())
    target = session.ctx.obj_pos("carriage").tolist()
    choices = {"carriage": dict(yaw=0., height=.003, clearance=1.035, force=3., speed=.006)}
    plan = program(session, {"carriage": target}, ["carriage"], choices)
    routes, _ = transfer_routes(session, session.ctx.eef_pos() + [0., 0., .015], clearance=1.035)
    path = next(r["path"] for r in routes if r["status"] == "necessary_pass")
    plan = materialize_initial_path(session, plan, path)
    execute = implementation.execute_calls
    monkeypatch.setattr(implementation, "execute_calls", lambda s, p, calls: execute(s, p, calls[:5]))
    supplied = []
    servo = session.arm.execute_joint
    def track(q):
        supplied.append(np.asarray(q).copy())
        return servo(q)
    monkeypatch.setattr(session.arm, "execute_joint", track)
    # A replan would falsify that the scorer and executor share one trajectory.
    monkeypatch.setattr(session, "plan_transfer", lambda **kwargs: pytest.fail("unexpected replanning"))
    implementation.execute_prefix(session, plan)
    np.testing.assert_array_equal(supplied, path["joints"])
    assert session.results[-1]["skill"] == "execute_joint_path"
    assert session.results[-1]["ok"]
