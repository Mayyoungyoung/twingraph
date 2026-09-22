"""Real printed-scene preflight: graph contracts before expensive rollouts."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from simbench.assembly.skills_v12 import configure_v12_skills
from simbench.value import planner_v12, stage_v12
from simbench.value.graph_value_v12 import STAGES, VALUE_RELATIONS, encode_graph
from simbench.value.skill_graph import validate_graph


@pytest.fixture(scope="module")
def preflight(tmp_path_factory):
    session = stage_v12.make_scene(1603, tmp_path_factory.mktemp("v12_system"))[1]
    configure_v12_skills(session)
    pool, metadata = planner_v12.propose(session.decision_observation,
        cad=session.planning_cad, n=48, seed=1603)
    graphs = [planner_v12.normalized_graph(session, proposal) for proposal in pool]
    return session, pool, metadata, graphs


def test_actual_printed_scene_generates_48_valid_distinct_executable_graphs(preflight):
    session, pool, metadata, graphs = preflight
    assert session.task_version == "printed_functional_assembly_v12"
    assert metadata["online_llm_call"] is False
    assert metadata["requested"] == metadata["unique_executable_parameters"] == 48
    assert session.planning_cad["pin_hole_width_m"] == .008
    fingerprints = set()
    for graph in graphs:
        plan = validate_graph(graph["assembly"])
        assert plan.prefix["skill_version"] == "feedback.v12"
        learned_pins = [c for c in plan.calls if c.skill == "insert"
                        and c.arguments.get("strategy") and c.arguments["strategy"].value == "learned"]
        assert {c.roles["manipulated"] for c in learned_pins} == {"pin_left", "pin_right"}
        encoded = encode_graph(graph)
        assert encoded["x"].shape[0] == len(STAGES)
        assert encoded["relations"].shape == (len(VALUE_RELATIONS), len(STAGES), len(STAGES))
        assert np.isfinite(encoded["x"]).all()
        assert encoded["active"].sum() == len(STAGES)
        assert encoded["relations"].sum() > 0
        fingerprints.add(encoded["x"].tobytes() + encoded["relations"].tobytes())
    assert len(fingerprints) == 48


@pytest.mark.parametrize("port_name", ["force", "force_stop", "strategy"])
def test_value_encoder_rejects_tampered_force_press_and_approach_ports(preflight, port_name):
    graphs = preflight[3]
    original = next(g for g in graphs if any(
        p["name"] == port_name and (port_name != "strategy" or p["value"] == "joint_checked_v10")
        for n in g["assembly"]["nodes"] for p in n["ports"]))
    changed = deepcopy(original)
    port = next(p for n in changed["assembly"]["nodes"] for p in n["ports"]
                if p["name"] == port_name and (port_name != "strategy" or p["value"] == "joint_checked_v10"))
    port["value"] = "feedback" if port_name == "strategy" else port["value"] + .5
    with pytest.raises(ValueError, match="mismatch"):
        encode_graph(changed)


@pytest.mark.parametrize("kind", ["force", "press", "approach"])
def test_planir_rejects_changed_executable_arguments_with_stale_scoring_choices(preflight, kind):
    graphs = preflight[3]
    def selected(call):
        args = call["arguments"]
        return ((kind == "force" and call["skill"] == "grasp" and "force" in args)
                or (kind == "press" and call["skill"] == "press" and "force_stop" in args)
                or (kind == "approach" and call["skill"] == "move" and "grasp" in args and "strategy" in args))
    original = next(g for g in graphs if any(selected(c) for c in g["assembly"]["plan"]["calls"]))
    changed = deepcopy(original)
    call = next(c for c in changed["assembly"]["plan"]["calls"] if selected(c))
    key = {"force": "force", "press": "force_stop", "approach": "strategy"}[kind]
    call["arguments"][key]["value"] = "feedback" if kind == "approach" else call["arguments"][key]["value"] + .5
    with pytest.raises(ValueError, match="disagree|mismatch"):
        validate_graph(changed["assembly"])


def test_missing_pin_capability_rejects_learned_insertion_preflight(preflight):
    session, pool, _, _ = preflight
    previous = deepcopy(session.capabilities)
    try:
        session.capabilities["pin_left"] = ()
        with pytest.raises(ValueError, match="capability|pin"):
            planner_v12.normalized_graph(session, pool[0])
    finally:
        session.capabilities = previous


@pytest.mark.parametrize("detached_part", [None, "handle", "carriage"])
def test_final_home_rechecks_seats_instead_of_reusing_pre_stroke_success(monkeypatch, detached_part):
    """The real geometry predicates must see state after the last motion."""
    from simbench.assembly import skills_v12
    from simbench.assembly.control import HOME
    from simbench.assembly.library import Result, SkillFailure
    from simbench.value import system_v12

    targets = {"carriage": np.array([.105, .085, .824]),
               "handle": np.array([.105, .085, .872])}
    positions = {part: xyz.copy() for part, xyz in targets.items()}
    ctx = SimpleNamespace(arm_qpos=HOME+1., obj_pos=lambda part: positions[part].copy())
    session = SimpleNamespace(ctx=ctx, held=None, artifacts={}, stage_targets=targets,
        planning_cad={"functional_stroke_minimum_m": .02},
        pin_insertion_config=SimpleNamespace(required_depth_m=.006),
        stage_passes={"cleaning_pass": True, "assembly_pass": True,
            "functional_test_pass": False, "fixture_capture_pass": False,
            "final_release_and_retraction_pass": False},
        call=lambda *args, **kwargs: Result(True))
    for part in targets:
        assert skills_v12.evaluate_functional_seat(session, part, targets[part])[0]

    def finish_stroke_and_home(current, minimum):
        assert current is session and minimum == .02
        for xyz in positions.values(): xyz[0] += .025
        current.stage_passes["functional_test_pass"] = True
        current.ctx.arm_qpos = HOME.copy()
        # The prior stroke passed, but its final retreat can disturb a part.
        if detached_part == "handle":
            positions["handle"][2] += .040
        elif detached_part == "carriage":
            # Keep the ring on its post; only the shoe is now outside the rail.
            positions["carriage"][1] += .030
            positions["handle"][1] += .030

    monkeypatch.setattr(system_v12, "_stroke", finish_stroke_and_home)
    monkeypatch.setattr(system_v12, "assembly_program", lambda *args: SimpleNamespace(calls=[]))
    monkeypatch.setattr(system_v12, "execute_calls", lambda *args: None)
    monkeypatch.setattr(system_v12, "observe_boundary",
        lambda current, stage, completed: {"stage": stage, "completed": list(completed)})
    monkeypatch.setattr(skills_v12, "evaluate_end_stop_fixture", lambda current: (True, {"success": True}))
    order = ["carriage", "end_stop", "pin_left", "pin_right", "handle"]
    kwargs = dict(order=order, choices={}, completed=["cleaning", *order])
    if detached_part is None:
        assert system_v12.run_staged(session, **kwargs).ok
    else:
        with pytest.raises(SkillFailure, match="complete functional task predicate failed"):
            system_v12.run_staged(session, **kwargs)
    checked = session.artifacts["final_seat_acceptance"]
    assert session.stage_passes["final_seat_pass"] == (detached_part is None)
    assert set(checked) == {"handle", "carriage"}
    for part, row in checked.items():
        assert row["success"] == (part != detached_part)
        assert row["measurement_source"] == "independent_simulator_acceptance_evaluator"
