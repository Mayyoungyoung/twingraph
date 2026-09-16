"""Full assembly semantics and read-only, label-blind proposal geometry."""
import copy
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from simbench.assembly.candidates import fingerprint
from simbench.assembly.library import GRASP, SkillFailure
from simbench.assembly.scene import SCENE
from simbench.value import stage_v5 as stage
from simbench.value.skill_graph import compile_graph, validate_graph
from simbench.value.system_v5 import assert_independent, semantic_payload


def fake_session():
    data = SimpleNamespace(qpos=np.zeros(7), qvel=np.zeros(7), ctrl=np.zeros(7))
    model = SimpleNamespace(geom_size=np.zeros((1, 3)), geom_pos=np.zeros((1, 3)),
                            geom_friction=np.ones((1, 3)))
    return SimpleNamespace(parts=stage.PARTS, grasp_specs=GRASP, held=None, grasp_epoch=0,
        stage_completed=(), ctx=SimpleNamespace(data=data, model=model, arm_qpos=data.qpos,
        obj_pos=lambda part: np.array([-.2, -.2, .85])))


def catalogue(*args):
    return {p: [dict(yaw=0., height=0.), dict(yaw=np.pi / 2, height=.001)] for p in stage.PARTS}, []


def routes(session, target, clearance, yaw, grasp=None):
    rows = []
    for index in range(3):
        path = dict(type="joint_path", part=None, start_q=session.ctx.arm_qpos.tolist(),
            joints=[(session.ctx.arm_qpos + .01 * (index + 1)).tolist()],
            target=np.asarray(target).tolist(), rotation=np.eye(3).tolist(),
            binding=dict(held=None, grasp_artifact=None, grasp_id=None,
                         grasp_epoch=session.grasp_epoch, prefix_id=None))
        rows.append(dict(status="necessary_pass", path=path))
    return rows, []


def test_complete_original_phases_compile_without_stroke(monkeypatch):
    monkeypatch.setattr(stage, "grasp_catalogue", catalogue)
    monkeypatch.setattr(stage, "transfer_routes", routes)
    session = fake_session()
    plans, counts = stage.build_pool(session, stage.nominal_targets(), 61000)
    assert len(plans) == 12 and counts["label_blind"]
    assert counts["structure_branches"] == 2
    assert counts["initial_trajectory_branches"] > 1
    observation = dict(robot={}, objects={p: {} for p in stage.PARTS}, goals=[])
    for plan in plans:
        assert len(plan.calls) == 101 and plan.boundary == 9
        assert plan.prefix["order"][:2] == ["carriage", "end_stop"]
        assert plan.prefix["order"][-1] == "handle"
        assert not any(c.arguments.get("mode", SimpleNamespace(value=None)).value == "constrained" for c in plan.calls)
        assert len([c for c in plan.calls if c.skill == "grasp"]) == 5
        assert len([c for c in plan.calls if c.skill == "insert"]) == 1
        assert len([c for c in plan.calls if c.skill == "place"]) == 5
        assert {c.arguments["part"].value for c in plan.calls[-6:-1]} == set(stage.PARTS)
        assert plan.calls[-1].arguments["target"].value == "home"
        assert validate_graph(compile_graph(observation, plan)).to_dict() == plan.to_dict()


def test_nested_pool_and_executable_identity_ignore_cosmetic_ids(monkeypatch):
    monkeypatch.setattr(stage, "grasp_catalogue", catalogue)
    monkeypatch.setattr(stage, "transfer_routes", routes)
    session = fake_session()
    small, _ = stage.build_pool(session, stage.nominal_targets(), 3, n=12)
    large, _ = stage.build_pool(session, stage.nominal_targets(), 3, n=24)
    assert [p.to_dict() for p in small] == [p.to_dict() for p in large[:12]]
    assert len({stage.semantic_key(p) for p in large}) == 24
    altered = copy.deepcopy(small[0])
    altered.id = "cosmetic"
    altered.prefix["semantic_program_id"] = "cosmetic"
    altered.prefix["initial_route_index"] = 99
    assert stage.semantic_key(altered) == stage.semantic_key(small[0])
    altered.calls[5].arguments["part"].value = "end_stop"
    assert stage.semantic_key(altered) != stage.semantic_key(small[0])


def test_planner_orders_are_complete_dependency_valid():
    assert len(stage.legal_orders()) == 2
    right_first = ["carriage", "end_stop", "pin_right", "pin_left", "handle"]
    assert stage.legal_orders([right_first, right_first]) == [tuple(right_first)]
    for bad in ([], [["carriage"]], [list(reversed(right_first))]):
        with pytest.raises(ValueError):
            stage.legal_orders(bad)


def test_target_rebind_preserves_semantics_and_uses_current_joints(monkeypatch):
    monkeypatch.setattr(stage, "grasp_catalogue", catalogue)
    monkeypatch.setattr(stage, "transfer_routes", routes)
    source = fake_session()
    target = fake_session()
    target.ctx.data.qpos[:] = .002
    targets = stage.nominal_targets()
    plan = stage.build_pool(source, targets, 61000, n=1)[0][0]
    rebound, details = stage.rebind_plan(target, targets, plan)
    assert semantic_payload(plan) == semantic_payload(rebound)
    assert plan.id != rebound.id
    assert plan.prefix["start_state"] != rebound.prefix["start_state"]
    assert details["initial_route_index"] == plan.prefix["initial_route_index"]
    np.testing.assert_array_equal(rebound.prefix["initial_artifacts"]["transfer"]["start_q"], target.ctx.arm_qpos)
    invalid_goals = copy.deepcopy(targets)
    invalid_goals["handle"][0] += .01
    with pytest.raises(ValueError, match="identical goals"):
        stage.rebind_plan(target, invalid_goals, plan)


def test_geometry_exhaustion_and_target_route_failure_are_distinct(monkeypatch):
    monkeypatch.setattr(stage, "grasp_catalogue", catalogue)
    monkeypatch.setattr(stage, "transfer_routes", routes)
    session, targets = fake_session(), stage.nominal_targets()
    plan = stage.build_pool(session, targets, 2, n=1)[0][0]
    monkeypatch.setattr(stage, "transfer_routes", lambda *args, **kwargs: ([dict(status="conflict")] * 3, []))
    with pytest.raises(SkillFailure, match="target initial approach"):
        stage.rebind_plan(session, targets, plan)
    # Tiny catalogue keeps the exhaustion test bounded without weakening checks.
    monkeypatch.setattr(stage, "grasp_catalogue", lambda *args: ({p: [dict(yaw=0., height=0.)] for p in stage.PARTS}, []))
    with pytest.raises(stage.CandidateGenerationError, match="pool exhausted"):
        stage.build_pool(session, targets, 2, n=1)


def test_only_initial_supplies_and_holders_move_in_scene_xml(tmp_path):
    spec = stage.StageV5Spec.sample(61000)
    original = ET.parse(SCENE).getroot()
    written = ET.parse(stage.write_scene(spec, tmp_path)).getroot()
    for name in (*stage.PARTS, "guide_base", "pin_left_holder", "pin_right_holder"):
        before = original.find(f"worldbody/body[@name='{name}']")
        after = written.find(f"worldbody/body[@name='{name}']")
        delta = [*spec.supply_shifts.get(name.removesuffix("_holder"), [0., 0.]), 0.]
        np.testing.assert_allclose(np.fromstring(after.get("pos", "0 0 0"), sep=" ") -
            np.fromstring(before.get("pos", "0 0 0"), sep=" "), delta, atol=1e-7)
        before.set("pos", after.get("pos", "0 0 0"))
        assert ET.tostring(before) == ET.tostring(after)


def test_real_initial_scenes_and_catalogue_do_not_mutate_live_physics(tmp_path):
    spec, twin, _, targets = stage.make_scene(61000, tmp_path / "twin", "twin")
    other, target, _, _ = stage.make_scene(61000, tmp_path / "target", "target")
    assert spec.config_id == other.config_id
    assert_independent(twin, target)
    assert twin.stage_completed == () and twin.results == []
    for part in stage.PARTS:
        assert np.linalg.norm(twin.ctx.obj_pos(part) - targets[part]) > .02
    initial, initial_time = fingerprint(twin), twin.ctx.data.time
    artifacts = copy.deepcopy(twin.artifacts)
    grasps, evidence = stage.grasp_catalogue(twin, targets)
    assert fingerprint(twin) == initial and twin.ctx.data.time == initial_time
    assert twin.artifacts == artifacts and twin.results == []
    assert set(grasps) == set(stage.PARTS) and all(grasps.values())
    assert any(row["status"] == "nominal_release_collision" for row in evidence)
    assert all("success" not in row for row in evidence)
    observation = stage.observed(twin, targets)
    assert {g["manipulated"] for g in observation["goals"]} == set(stage.PARTS)
    assert all(g["minimum_eef_clearance_m"] == .02 for g in observation["goals"])
