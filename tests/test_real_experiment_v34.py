"""The physical boundary must retain actual layout/plan/trace provenance."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import prepare_real_experiment_v34 as prep


@pytest.fixture
def nominal():
    path = Path(__file__).resolve().parents[1] / "real_robot/config/nominal_layout_4000.json"
    return prep.read(path)


def test_native_base_coordinates_and_explicit_pose_inputs(nominal):
    with pytest.raises(ValueError, match="not measured hardware"):
        prep.normalize_layout(nominal)
    world = prep.normalize_layout(nominal, allow_nominal=True)
    np.testing.assert_allclose(world["poses"]["guide_base"]["position_m"], [.075, .085, .812])
    np.testing.assert_allclose(world["table"]["center_xy_m"], [-.21, 0.])
    changed = deepcopy(nominal)
    changed["seed"] = 98765
    other = prep.normalize_layout(changed, allow_nominal=True)
    assert world["poses"] == other["poses"]  # seed cannot secretly choose placement
    changed["poses"]["carriage"]["position_m"][0] += .05
    supplied = prep.normalize_layout(changed, allow_nominal=True)
    assert supplied["poses"]["carriage"]["position_m"][0] == pytest.approx(
        world["poses"]["carriage"]["position_m"][0] + .05)


def test_incomplete_false_calibration_and_invalid_frames_fail_closed(nominal):
    measured = deepcopy(nominal)
    measured["measured"] = True
    with pytest.raises(ValueError, match="placement_verified"):
        prep.normalize_layout(measured)
    measured.update(placement_verified=True, cad_match_verified=True,
                    measurement_source="operator/caliper", measured_at_utc="2026-10-09T01:00:00Z")
    measured["robot"].update(start_state_verified=True, base_frame_verified=True, state_measured=True)
    with pytest.raises(ValueError, match="calibrated mass/friction"):
        prep.normalize_layout(measured)
    measured["physics"]["calibrated"] = True
    with pytest.raises(ValueError, match="all free-body masses"):
        prep.normalize_layout(measured)
    malformed = deepcopy(nominal)
    malformed["robot"]["T_world_base"][0][0] = 2.
    with pytest.raises(ValueError, match="rigid right-handed"):
        prep.normalize_layout(malformed, allow_nominal=True)
    malformed = deepcopy(nominal)
    malformed["poses"]["pin_left"]["quat_wxyz"] = [1, 0, 0, 1]
    with pytest.raises(ValueError, match="unit quaternion"):
        prep.normalize_layout(malformed, allow_nominal=True)


def test_normalized_measured_layout_preserves_hardware_measurement_binding(nominal):
    measured = deepcopy(nominal)
    measured.update(measured=True, placement_verified=True, cad_match_verified=True,
                    measurement_source="test-fixture/operator", measured_at_utc="2026-10-09T01:00:00Z")
    measured["robot"].update(start_state_verified=True, base_frame_verified=True,
                             state_source="test-fixture/Panda get_state", state_measured_at_utc="2026-10-09T01:00:00Z")
    measured["physics"] = dict(calibrated=True,
        mass_kg={p:.1 for p in prep.FREE_BODIES},
        friction={p:[.5, .01, .001] for p in prep.POSE_NAMES})
    with pytest.raises(ValueError, match="robot.state_measured"):
        prep.normalize_layout(measured)
    measured["robot"]["state_measured"] = True
    world = prep.normalize_layout(measured)
    assert world["robot"]["state_measured"] is True
    assert world["robot"]["start_state_verified"] is True
    assert world["robot"]["base_frame_verified"] is True
    assert world["robot"]["state_source"] == measured["robot"]["state_source"]
    assert world["robot"]["state_measured_at_utc"] == measured["robot"]["state_measured_at_utc"]
    for key in ("placement_verified", "cad_match_verified", "measurement_source", "measured_at_utc"):
        assert world[key] == measured[key]
    assert world["layout_input_sha256"] == prep.digest(measured)
    # A nominal trial remains an explicit unmeasured demonstration and keeps
    # the original normalized robot shape/digest used by frozen demo requests.
    demo = prep.normalize_layout(nominal, allow_nominal=True)
    assert set(demo["robot"]) == {"T_world_base", "joints_rad", "finger_width_m"}


def test_digest_is_identical_to_frozen_plan_hash():
    from simbench.value.plan import digest
    payload = {"order": ["carriage", "end_stop"], "choices": {"force": 3., "width": .02}}
    assert prep.digest(payload) == digest(payload)


def test_reference_trace_retains_skill_stage_and_restores_handlers():
    class Context:
        data = SimpleNamespace(time=1.)
        pos = np.array([.1, .2, .3])
        def eef_mat(self):
            return np.eye(3)
        def eef_pos(self):
            return self.pos.copy()
    ctx = Context()
    class Arm:
        rotation = np.eye(3)
        def move(self, xyz, rotation=None, speed=.01, tol=.001):
            ctx.pos = np.asarray(xyz)
            return True
        def servo(self, xyz, rotation=None):
            ctx.pos = np.asarray(xyz)
        def execute_joint(self, qgoal, duration=None):
            return True
        def execute_joint_waypoints(self, joints):
            return True
        def open(self):
            return True
        def close(self, part, force=3.):
            return True
    arm = Arm()
    session = SimpleNamespace(ctx=ctx, arm=arm)
    def call(name, **params):
        arm.move([.2, .3, .4])
        return SimpleNamespace(ok=True, metrics={"contact": True}, reason="")
    session.call = call
    with prep.TwinTrace(session) as trace:
        session.call("insert", part="pin_left")
        assert trace.events[0]["stage"] == "pin_left"
        assert trace.events[0]["primitive"] == "move"
        assert trace.events[0]["parameters"]["speed"] == .01
        assert trace.skill_events[0]["metrics"]["contact"] is True
        np.testing.assert_allclose(np.asarray(trace.events[0]["destination_world_eef"])[:3, 3], [.2, .3, .4])
    assert session.call is call
    assert arm.move.__func__ is Arm.move


def stub_request():
    entries = []
    for i in range(8):
        entries.append(dict(name=f"candidate_{i}", plan_sha256=f"plan_{i}",
            proposal=dict(order=list(prep.PARTS), choices={p:{} for p in prep.PARTS}),
            complete_candidate_plan_ir={"id": f"plan_{i}"}))
    return dict(candidates=entries, runtime_sha256="source", seed=4000,
        decision_observation={"assembly_targets": {p:{"position_m": [0, 0, .01]} for p in prep.PARTS}})


def test_top3_receipt_links_selected_plan_same_layout_and_twin_result(tmp_path, monkeypatch, nominal):
    layout_path = tmp_path / "layout_input.json"
    prep.write(layout_path, nominal)
    def freeze(layout, out, **kwargs):
        r = stub_request()
        r["explicit_layout_sha256"] = prep.digest(layout)
        prep.write(out / "request.json", r)
        return r
    monkeypatch.setattr(prep, "freeze_candidates", freeze)
    monkeypatch.setattr(prep, "rank", lambda *args:dict(order=[7, 2, 4, 0, 1, 3, 5, 6], budget=3,
        model_freeze_sha256="frozen"))
    calls = []
    def rollout(request, entry, layout, folder, **kwargs):
        calls.append((entry["name"], prep.digest(layout)))
        result = dict(valid=True, success=entry["name"] == "candidate_2", fresh_total_wall_seconds=1.)
        prep.write(folder / "result.json", result)
        prep.write(folder / "tcp_reference_trace.json", dict(events=[], skill_events=[], hardware_executable=False))
        return result
    monkeypatch.setattr(prep, "fresh_rollout", rollout)
    output = tmp_path / "run"
    report = prep.prepare(layout_path, tmp_path / "training", output,
                          allow_nominal=True, planner="grounded")
    assert [r[0] for r in calls] == ["candidate_7", "candidate_2"]
    assert len({r[1] for r in calls}) == 1
    assert report["selected"] == "candidate_2" and report["phase"] == "twin_validated"
    reference = prep.read(output / "reference_schedule.json")
    assert reference["hardware_executable"] is False
    assert reference["measured_layout"] is False
    for key, file in reference["artifact_files"].items():
        assert reference["evidence"][f"{key}_sha256"] == prep.file_sha(output / file)
    source = reference["bridge_source"]
    assert source == report["bridge_source"]
    assert source["script_sha256"] == prep.file_sha(output / source["script_file"])
    assert source["provenance_sha256"] == prep.file_sha(output / source["provenance_file"])
    assert len(prep.read(output / "selected_feedback_program.json")["stages"]) == 5
    with pytest.raises(ValueError, match="new empty folder"):
        prep.prepare(layout_path, tmp_path / "training", output, allow_nominal=True)


def test_unsolvable_or_invalid_twin_never_creates_executable_artifact(tmp_path, monkeypatch, nominal):
    layout_path = tmp_path / "layout_input.json"
    prep.write(layout_path, nominal)
    def freeze(layout, out, **kwargs):
        r = stub_request()
        prep.write(out / "request.json", r)
        return r
    monkeypatch.setattr(prep, "freeze_candidates", freeze)
    monkeypatch.setattr(prep, "rank", lambda *args:dict(order=list(range(8)), budget=3, model_freeze_sha256="frozen"))
    calls = []
    def rollout(request, entry, layout, folder, **kwargs):
        calls.append(entry["name"])
        result = dict(valid=True, success=False, fresh_total_wall_seconds=1.)
        prep.write(folder / "result.json", result)
        return result
    monkeypatch.setattr(prep, "fresh_rollout", rollout)
    output = tmp_path / "fail_run"
    report = prep.prepare(layout_path, tmp_path / "training", output, allow_nominal=True)
    assert len(calls) == 3 and report["phase"] == "no_top3_passed"
    assert not (output / "reference_schedule.json").exists()
    assert not (output / "selected_candidate.json").exists()
