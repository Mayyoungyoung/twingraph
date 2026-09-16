"""Protocol, process constraints, and source-layout integrity for the real stage."""
import copy
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from simbench.assembly.library import PARTS, GRASP, SkillFailure
from simbench.assembly.scene import SCENE
from simbench.value import stage_assembly as stage
from simbench.value.plan import PlanIR
from simbench.value.research_scenarios import semantic_key
from simbench.value.skill_graph import compile_graph, validate_graph


def session_and_targets():
    data = SimpleNamespace(qpos=np.zeros(7), qvel=np.zeros(7), ctrl=np.zeros(7))
    model = SimpleNamespace(geom_size=np.zeros((1, 3)), geom_pos=np.zeros((1, 3)),
                            geom_friction=np.ones((1, 3)))
    session = SimpleNamespace(parts=PARTS, grasp_specs=GRASP, held=None,
                              stage_completed=stage.CHECKPOINT_PARTS,
                              inspect_seat=lambda *a, **kw: SimpleNamespace(ok=True),
                              ctx=SimpleNamespace(data=data, model=model))
    targets = {p: [0., 0., .85] for p in PARTS}
    return session, targets


def test_only_physically_permitted_orders_and_complete_programs():
    assert set(stage.legal_orders(stage.CONTINUATION_PARTS)) == {
        ("pin_left", "pin_right", "handle"), ("pin_right", "pin_left", "handle")}
    assert stage.legal_orders(("handle",), completed=("carriage", "end_stop")) == []
    session, targets = session_and_targets()
    choices = {p: dict(yaw=0., height=0., clearance=1.035, force=3.5, speed=.008)
               for p in stage.CONTINUATION_PARTS}
    with pytest.raises(ValueError, match="precedence"):
        stage.program(session, targets, ("handle", "pin_left", "pin_right"), choices)
    with pytest.raises(ValueError, match="every remaining"):
        stage.program(session, targets, ("pin_left",), {"pin_left": choices["pin_left"]})


def test_pool_is_nested_unique_and_compiles_same_executable_graph():
    session, targets = session_and_targets()
    small, counts = stage.build_pool(session, targets, seed=31, n=16, precheck=False)
    large, _ = stage.build_pool(session, targets, seed=31, n=32, precheck=False)
    assert [p.to_dict() for p in small] == [p.to_dict() for p in large[:16]]
    assert len({semantic_key(p) for p in large}) == 32
    assert counts["structure_branches"] == 2
    assert counts["label_blind"]
    assert {p.prefix["choices"]["pin_left"]["force"] for p in large} == {2.5, 3., 3.5}
    obs = dict(robot={}, objects={p: {} for p in PARTS}, goals=[])
    for plan in small:
        assert len(plan.calls) == 59  # Three real 18-call stages + all five final checks.
        graph = compile_graph(obs, plan)
        assert validate_graph(graph).to_dict() == PlanIR.from_dict(plan.to_dict()).to_dict()
        assert plan.prefix["order"][-1] == "handle"
        assert set(plan.prefix["order"][:2]) == {"pin_left", "pin_right"}
        assert {c.arguments["part"].value for c in plan.calls[-5:]} == set(PARTS)


def test_checkpoint_preconditions_cannot_be_replaced_by_completed_metadata():
    session, targets = session_and_targets()
    with pytest.raises(ValueError, match="robot-installed"):
        stage.build_pool(session, targets, 1, completed=("carriage",), precheck=False)
    session.held = "pin_left"
    with pytest.raises(ValueError, match="released"):
        stage.build_pool(session, targets, 1, precheck=False)
    session.held = None
    session.inspect_seat = lambda *a, **kw: SimpleNamespace(ok=False, metrics={"position_error_m": .1})
    with pytest.raises(SkillFailure, match="not assembled"):
        stage.build_pool(session, targets, 1, precheck=False)


def test_checkpoint_after_one_pin_has_only_the_valid_remaining_order():
    session, targets = session_and_targets()
    plans, counts = stage.build_pool(session, targets, 5, n=8,
                                     completed=("carriage", "end_stop", "pin_right"), precheck=False)
    assert counts["structure_branches"] == 1
    assert all(p.prefix["order"] == ["pin_left", "handle"] for p in plans)
    assert all(len(p.calls) == 41 for p in plans)


def test_materialized_route_branches_remain_distinct_executable_inputs(monkeypatch):
    session, targets = session_and_targets()
    session.grasp_epoch = 4
    session.ctx.arm_qpos = session.ctx.data.qpos
    session.ctx.obj_pos = lambda part: np.array([-.2, -.2, .85])
    def routes(s, target, clearance, yaw):
        result = []
        for index in range(3):
            path = dict(type="joint_path", part=None, id=f"route_{index}",
                        start_q=s.ctx.arm_qpos.tolist(), joints=[(s.ctx.arm_qpos + .01 * (index + 1)).tolist()],
                        target=np.asarray(target).tolist(), rotation=np.eye(3).tolist(),
                        binding=dict(held=None, grasp_artifact=None, grasp_id=None, grasp_epoch=4, prefix_id=None))
            result.append(dict(status="necessary_pass", path=path))
        return result, []
    monkeypatch.setattr(stage, "transfer_routes", routes)
    plans, counts = stage.build_pool(session, targets, 8, n=32, precheck=True)
    assert counts["initial_trajectory_branches"] > 1
    assert len({stage.semantic_key(p) for p in plans}) == len(plans)
    obs = dict(robot={}, objects={p: {} for p in PARTS}, goals=[])
    for plan in plans:
        assert len(plan.calls) == 58
        assert plan.boundary == 9
        assert plan.prefix["initial_grasp_epoch"] == 4
        assert plan.calls[4].skill == "move"
        assert plan.prefix["initial_artifacts"]["transfer"]["binding"]["prefix_id"] == plan.id
        validate_graph(compile_graph(obs, plan))


def test_scene_changes_only_initial_supplies_and_keeps_passive_holder_alignment(tmp_path):
    spec = stage.StageAssemblySpec.sample(26)
    assert spec.config_id == stage.StageAssemblySpec.sample(26).config_id
    assert spec.config_id != stage.StageAssemblySpec.sample(27).config_id
    original = ET.parse(SCENE).getroot()
    copied = ET.parse(stage.write_scene(spec, tmp_path)).getroot()
    for name in (*PARTS, "guide_base", "pin_left_holder", "pin_right_holder"):
        before = original.find(f"worldbody/body[@name='{name}']")
        after = copied.find(f"worldbody/body[@name='{name}']")
        bpos = np.fromstring(before.get("pos", "0 0 0"), sep=" ")
        apos = np.fromstring(after.get("pos", "0 0 0"), sep=" ")
        part = name.removesuffix("_holder")
        delta = [*spec.supply_shifts.get(part, [0., 0.]), 0.]
        np.testing.assert_allclose(apos - bpos, delta, atol=1e-7)
        # All shape, contact, joint and material descriptions are untouched.
        before = copy.deepcopy(before)
        before.set("pos", after.get("pos", "0 0 0"))
        assert ET.tostring(before) == ET.tostring(after)
    assert len(copied.findall(".//actuator/*")) == len(original.findall(".//actuator/*"))


def test_invalid_supply_layout_is_rejected(tmp_path):
    spec = stage.StageAssemblySpec.sample(1)
    spec.supply_shifts["pin_left"] = [.5, 0.]
    with pytest.raises(ValueError, match="layout range"):
        stage.write_scene(spec, tmp_path)
