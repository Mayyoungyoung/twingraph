"""Physical-adapter tests use fake feedback exclusively; no Panda connection."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import time

import numpy as np
import pytest

from real_robot import panda_adapter as adapter


def pose(x=.3, y=0., z=.3):
    result = np.eye(4)
    result[:3, 3] = [x, y, z]
    return result.tolist()


def calibration():
    return dict(schema_version=1, T_base_world=np.eye(4).tolist(),
                T_ee_tcp=np.eye(4).tolist(), T_sim_eef_tcp=np.eye(4).tolist(),
                workspace_base=dict(min=[.1, -.4, .05], max=[.8, .4, .7]),
                verified=True, tcp_verified=True, sim_eef_tcp_verified=True,
                payload_verified=True, workspace_verified=True, layout_verified=True,
                measured_at_utc=datetime.now(timezone.utc).isoformat(),
                expected_F_T_EE=np.eye(4).tolist(), expected_total_payload_mass_kg=.5,
                allowed_observation_sources=["test_measured_fixture"])


def acceptance():
    return dict(id="final", kind="verify_observation", observation_file="observations/final.json",
                expected_flags=dict(assembly_accepted=True), timeout_s=1., task_acceptance=True)


def schedule(commands=None):
    return dict(schema_version=1, task_id="test_task", candidate_id="candidate_1",
                hardware_executable=True, frame="twin_world", start_pose_world_tcp=pose(),
                commands=(commands or []) + [acceptance()])


def add_evidence(root, program, config, wrong_twin_plan=False, measured=True):
    layout = dict(measured=measured, robot=dict(state_measured=measured, joints_rad=[0]*7,
                                              finger_width_m=.08, T_world_base=np.eye(4).tolist()))
    ir = dict(schema="test", actions=[dict(skill="test")])
    candidate = dict(name="candidate_1", complete_candidate_plan_ir=ir,
                     plan_sha256=adapter._frozen_digest(ir))
    request = dict(explicit_layout_sha256=adapter._frozen_digest(layout), runtime_sha256="runtime-test",
                   candidates=[candidate])
    result = dict(valid=True, success=True, timeout=False, candidate_name="candidate_1",
                  request_sha256=adapter._frozen_digest(request), explicit_layout_sha256=adapter._frozen_digest(layout),
                  plan_sha256="wrong" if wrong_twin_plan else candidate["plan_sha256"],
                  runtime_sha256=request["runtime_sha256"])
    program["evidence"], program["artifact_files"] = {}, {}
    for name, value in (("layout", layout), ("candidate", candidate), ("request", request), ("twin_result", result)):
        raw = json.dumps(value).encode()
        (root / (name + ".json")).write_bytes(raw)
        program["evidence"][name + "_sha256"] = hashlib.sha256(raw).hexdigest()
        program["artifact_files"][name] = name + ".json"
    config["collision_review_schedule_sha256"] = adapter.canonical_sha256(program)


class FakeContext:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class FakePanda:
    def __init__(self, root=None, grasped=True, wrench=None):
        self.pose = np.asarray(pose())
        self.reference = self.pose.copy()
        self.stopped, self.gripper_stopped, self.began = False, False, False
        self.robot_time = 0.
        self.root, self.grasped = root, grasped
        self.wrench = np.zeros(6) if wrench is None else np.asarray(wrench)

    def feedback(self):
        self.robot_time += .01
        return dict(pose_base_ee=self.pose.copy(), wrench_base=self.wrench.copy(), state_time_s=self.robot_time,
                    errors=False, collision=False, F_T_EE=np.eye(4), total_payload_mass_kg=.5, q=np.zeros(7), dq=np.zeros(7))

    def begin(self, *args):
        self.began = True
        return FakeContext()

    def tick(self):
        time.sleep(.001)
        self.pose = self.reference.copy()
        if self.root is not None:
            path = self.root / "observations/final.json"
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps(dict(task_id="test_task", command_id="final", source="test_measured_fixture",
                measured_at_utc=datetime.now(timezone.utc).isoformat(), flags=dict(assembly_accepted=True))))
        return True

    def set_target(self, goal):
        self.reference = np.asarray(goal).copy()

    def gripper_action(self, command):
        return dict(ok=True, width_m=command["width_m"], is_grasped=self.grasped)

    def gripper_feedback(self):
        return dict(width_m=.08, is_grasped=False)

    def stop_gripper(self):
        self.gripper_stopped = True

    def stop(self):
        self.stopped = True


def test_dry_run_does_not_connect_or_claim_task_success(monkeypatch):
    monkeypatch.setattr(adapter, "PandaPyBackend", lambda *args: pytest.fail("hardware connected during dry-run"))
    result = adapter.execute_schedule(schedule(), calibration())
    assert result["status"] == "dry_run" and result["task_accepted"] is False
    assert result["arm_token"].startswith("EXECUTE_PANDA:")


def test_reference_and_invalid_arm_never_connect(monkeypatch):
    monkeypatch.setattr(adapter, "PandaPyBackend", lambda *args: pytest.fail("hardware connected"))
    program = schedule()
    program["hardware_executable"] = False
    assert adapter.execute_schedule(program, calibration())["status"] == "blocked_reference"
    with pytest.raises(adapter.SafetyError, match="not hardware executable"):
        adapter.execute_schedule(program, calibration(), execute=True)
    program["hardware_executable"] = True
    with pytest.raises(adapter.SafetyError, match="arm token"):
        adapter.execute_schedule(program, calibration(), execute=True, arm_token="not armed")


@pytest.mark.parametrize("change,expected", [
    (lambda s: s["commands"].insert(0, dict(id="j", kind="joint_position", q=[0]*7)), "Unsupported"),
    (lambda s: s["commands"].insert(0, dict(id="m", kind="move_tcp", pose_world_tcp=pose(2.0))), "workspace"),
    (lambda s: s["commands"].insert(0, dict(id="m", kind="move_tcp", pose_world_tcp=pose(.31), speed_m_s=.5)), "range"),
    (lambda s: s["commands"].insert(0, dict(id="g", kind="gripper_grasp", width_m=.02, force_n=100)), "range"),
    (lambda s: s["commands"].clear(), "nonempty"),
])
def test_commands_reject_unsafe_or_unimplemented_inputs(change, expected):
    program = schedule()
    change(program)
    with pytest.raises(adapter.SafetyError, match=expected):
        adapter.validate_schedule(program, calibration())


def test_tool_transform_affects_ee_goal_and_workspace():
    config = calibration()
    config["T_ee_tcp"][2][3] = .10
    program = schedule([dict(id="m", kind="move_tcp", pose_world_tcp=pose(.31))])
    command = adapter.validate_schedule(program, config)[0]
    np.testing.assert_allclose(command["pose_base_ee"][:3, 3], [.31, 0, .20])
    program["start_pose_world_tcp"] = pose(z=.06)
    with pytest.raises(adapter.SafetyError, match="workspace"):
        adapter.validate_schedule(program, config)


def test_rigid_transform_rejects_nan_reflection_and_nonorthogonal_rotation():
    for matrix in (np.full((4, 4), np.nan), np.diag([-1, 1, 1, 1]), np.diag([2, 1, 1, 1])):
        with pytest.raises(adapter.SafetyError):
            adapter.transform(matrix)


def test_quaternion_interpolation_preserves_orientation_and_speed_bound():
    start, goal = np.eye(4), np.eye(4)
    goal[:3, :3] = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    half = adapter.interpolate_pose(start, goal, .5)
    assert adapter._rotation_angle(start[:3, :3], half[:3, :3]) == pytest.approx(np.pi/2)
    np.testing.assert_allclose(half[:3, :3].T @ half[:3, :3], np.eye(3), atol=1e-8)


def test_bound_evidence_detects_wrong_successful_twin_and_file_mutation(tmp_path):
    program, config = schedule(), calibration()
    add_evidence(tmp_path, program, config, wrong_twin_plan=True)
    with pytest.raises(adapter.SafetyError, match="different candidate plan"):
        adapter._verify_evidence(program, tmp_path)
    add_evidence(tmp_path, program, config)
    adapter._verify_evidence(program, tmp_path)
    (tmp_path / "candidate.json").write_text("{}")
    with pytest.raises(adapter.SafetyError, match="contents changed"):
        adapter._verify_evidence(program, tmp_path)


def test_nominal_scene_and_path_escape_cannot_arm(tmp_path):
    program, config = schedule(), calibration()
    add_evidence(tmp_path, program, config, measured=False)
    with pytest.raises(adapter.SafetyError, match="nominal layout"):
        adapter._verify_evidence(program, tmp_path)
    program["artifact_files"]["request"] = "../request.json"
    with pytest.raises(adapter.SafetyError, match="remain inside"):
        adapter._verify_evidence(program, tmp_path)


def test_task_acceptance_requires_fresh_bound_measurement(tmp_path):
    program, config, command = schedule(), calibration(), acceptance()
    path = tmp_path / command["observation_file"]
    path.parent.mkdir()
    observation = dict(task_id="test_task", command_id="final", source="test_measured_fixture",
                       measured_at_utc=datetime.now(timezone.utc).isoformat(), flags=dict(assembly_accepted=True))
    path.write_text(json.dumps(observation))
    adapter.verify_observation(command, program, config, tmp_path, time.time() - .1)
    with pytest.raises(adapter.SafetyError, match="after this checkpoint"):
        adapter.verify_observation(command, program, config, tmp_path, time.time() + 1)
    observation["command_id"] = "previous_stage"
    path.write_text(json.dumps(observation))
    with pytest.raises(adapter.SafetyError, match="different task/checkpoint"):
        adapter.verify_observation(command, program, config, tmp_path, 0)


def test_force_guard_includes_raw_force_even_with_bias():
    config = calibration()
    config["wrench_bias_base"] = [5, 0, 0, 0, 0, 0]
    state = FakePanda(wrench=[16, 0, 0, 0, 0, 0]).feedback()
    with pytest.raises(adapter.SafetyError, match="force/torque"):
        adapter._FeedbackGuard(config).check(state)


def test_feedback_freshness_and_tracking_are_checked(monkeypatch):
    state = FakePanda().feedback()
    guard = adapter._FeedbackGuard(calibration())
    guard.check(state)
    monkeypatch.setattr(adapter.time, "monotonic", lambda: guard.last_change + 1)
    with pytest.raises(adapter.SafetyError, match="stopped updating"):
        guard.check(state)
    state["state_time_s"] += 1
    with pytest.raises(adapter.SafetyError, match="tracking error"):
        guard.check(state, np.asarray(pose(.4)))


def test_measured_motion_and_final_observation_complete_fake_program(tmp_path):
    program = schedule([dict(id="move", kind="move_tcp", pose_world_tcp=pose(.3001), timeout_s=1.)])
    config = calibration()
    add_evidence(tmp_path, program, config)
    backend = FakePanda(tmp_path)
    result = adapter.execute_schedule(program, config, execute=True,
        arm_token=adapter.required_arm_token(program, config), backend=backend, evidence_root=tmp_path)
    assert result["status"] == "completed" and result["task_accepted"] is True
    assert backend.stopped and len(result["commands"]) == 2


def test_failed_grasp_stops_program_without_false_success(tmp_path):
    program = schedule([dict(id="grasp", kind="gripper_grasp", width_m=.02, force_n=5, timeout_s=1)])
    config = calibration()
    add_evidence(tmp_path, program, config)
    backend = FakePanda(tmp_path, grasped=False)
    result = adapter.execute_schedule(program, config, execute=True,
        arm_token=adapter.required_arm_token(program, config), backend=backend, evidence_root=tmp_path)
    assert result["status"] == "failed" and result["task_accepted"] is False
    assert backend.gripper_stopped and backend.stopped and not result["commands"]


def test_guarded_contact_before_minimum_depth_fails(tmp_path):
    program = schedule([dict(id="contact", kind="guarded_move_tcp", pose_world_tcp=pose(.301),
        contact_axis_world=[1, 0, 0], contact_force_n=2, min_progress_m=.0005, timeout_s=1)])
    config = calibration()
    add_evidence(tmp_path, program, config)
    backend = FakePanda(tmp_path, wrench=[-3, 0, 0, 0, 0, 0])
    result = adapter.execute_schedule(program, config, execute=True,
        arm_token=adapter.required_arm_token(program, config), backend=backend, evidence_root=tmp_path)
    assert result["status"] == "failed" and "before the required insertion depth" in result["error"]
    assert backend.stopped


def test_compiler_preserves_joint_gap_and_does_not_duplicate_nested_wrapper():
    reference = dict(task_id="test_task", candidate_id="candidate_1", twin_validated=True,
                     start_pose_world_eef=pose(), events=[
        dict(id="outer", stage="carriage", primitive="move", pose_before_world_eef=pose(),
             destination_world_eef=pose(.4), pose_after_world_eef=pose(.4)),
        dict(id="inner", parent_primitive_id="outer", stage="carriage", primitive="execute_joint",
             pose_before_world_eef=pose(), pose_after_world_eef=pose(.4))])
    compiled = adapter.compile_reference_schedule(reference, calibration())
    assert compiled["hardware_executable"] is False
    assert sum(command["kind"] == "unbound_joint_route" for command in compiled["commands"]) == 1
    assert not any(command.get("source_event_id") == "outer" for command in compiled["commands"])


def test_arm_token_changes_with_calibration_or_schedule():
    program, config = schedule(), calibration()
    token = adapter.required_arm_token(program, config)
    config["T_ee_tcp"][2][3] = .01
    assert adapter.required_arm_token(program, config) != token
    token = adapter.required_arm_token(program, config)
    program["candidate_id"] = "different"
    assert adapter.required_arm_token(program, config) != token


def test_contact_insert_requires_actual_goal_then_independent_acceptance(tmp_path):
    program = schedule([dict(id="insert", kind="contact_move_tcp", pose_world_tcp=pose(.3005),
        contact_axis_world=[1, 0, 0], min_progress_m=.0004, timeout_s=1)])
    config = calibration()
    add_evidence(tmp_path, program, config)
    backend = FakePanda(tmp_path)
    result = adapter.execute_schedule(program, config, execute=True,
        arm_token=adapter.required_arm_token(program, config), backend=backend, evidence_root=tmp_path)
    assert result["status"] == "completed" and result["task_accepted"] is True
    assert result["commands"][0]["detail"]["contact_insertion"] is True


def test_press_hold_requires_measured_force_band(tmp_path):
    program = schedule([dict(id="press", kind="press_tcp", pose_world_tcp=pose(.301),
        contact_axis_world=[1, 0, 0], contact_force_n=2, contact_force_band_n=[2, 4],
        hold_s=.01, timeout_s=1)])
    config = calibration()
    add_evidence(tmp_path, program, config)
    backend = FakePanda(tmp_path, wrench=[-3, 0, 0, 0, 0, 0])
    result = adapter.execute_schedule(program, config, execute=True,
        arm_token=adapter.required_arm_token(program, config), backend=backend, evidence_root=tmp_path)
    assert result["status"] == "completed"
    assert result["commands"][0]["detail"]["hold_s"] == .01


def test_cartesian_trace_proposal_retains_curve_instead_of_only_endpoints():
    event = dict(id="joint", stage="carriage", primitive="execute_joint",
                 pose_before_world_eef=pose(), pose_after_world_eef=pose(.4))
    reference = dict(task_id="test_task", candidate_id="candidate_1", twin_validated=True,
        start_pose_world_eef=pose(), events=[event], control_samples=[
            dict(primitive_id="joint", pose_world_eef=pose(.3)),
            dict(primitive_id="joint", pose_world_eef=pose(.35, .05)),
            dict(primitive_id="joint", pose_world_eef=pose(.4))])
    compiled = adapter.compile_reference_schedule(reference, calibration())
    movements = [command for command in compiled["commands"] if command["kind"] == "move_path_tcp"]
    assert movements and compiled["hardware_executable"] is False
    assert any(point[1][3] > .049 for command in movements for point in command["waypoints_world_tcp"])
    assert all(command["requires_route_review"] for command in movements)


@pytest.mark.parametrize("changed", ["q", "dq", "gripper"])
def test_different_elbow_motion_or_gripper_state_prevents_controller_start(tmp_path, changed):
    program, config = schedule(), calibration()
    add_evidence(tmp_path, program, config)
    backend = FakePanda(tmp_path)
    if changed == "gripper":
        backend.gripper_feedback = lambda: dict(width_m=.04, is_grasped=False)
    else:
        original = backend.feedback
        def feedback():
            state = original()
            state[changed] = np.ones(7) * .1
            return state
        backend.feedback = feedback
    with pytest.raises(adapter.SafetyError):
        adapter.execute_schedule(program, config, execute=True,
            arm_token=adapter.required_arm_token(program, config), backend=backend, evidence_root=tmp_path)
    assert backend.began is False


def test_long_servo_stream_compiles_to_one_finite_curve_not_thousands_of_commands():
    events, samples = [], []
    for index in range(2000):
        identifier = "servo_{}".format(index)
        events.append(dict(id=identifier, stage="pin_left", skill="insert", call_stack=[], primitive="servo",
            pose_before_world_eef=pose(.3 + index*1e-6), pose_after_world_eef=pose(.3+(index+1)*1e-6),
            destination_world_eef=pose(.3+(index+1)*1e-6)))
        samples.append(dict(primitive_id=identifier, pose_world_eef=pose(.3+(index+1)*1e-6)))
    reference = dict(task_id="test_task", candidate_id="candidate_1", twin_validated=True,
                     start_pose_world_eef=pose(), events=events, control_samples=samples)
    compiled = adapter.compile_reference_schedule(reference, calibration())
    paths = [command for command in compiled["commands"] if command["kind"] == "contact_path_tcp"]
    assert len(paths) == 1 and len(compiled["commands"]) == 3
    assert len(paths[0]["source_event_ids"]) == 2000
    assert len(paths[0]["waypoints_world_tcp"]) == 1
    assert paths[0]["timeout_s"] < 10 and compiled["hardware_executable"] is False


def test_finite_contact_curve_executes_fake_waypoints_and_checks_actual_depth(tmp_path):
    program = schedule([dict(id="curve", kind="contact_path_tcp", waypoints_world_tcp=[pose(.3001), pose(.3002)],
        speed_m_s=.003, progress_axis_world=[1, 0, 0], min_progress_m=.00015, timeout_s=1)])
    config = calibration()
    add_evidence(tmp_path, program, config)
    backend = FakePanda(tmp_path)
    result = adapter.execute_schedule(program, config, execute=True,
        arm_token=adapter.required_arm_token(program, config), backend=backend, evidence_root=tmp_path)
    assert result["status"] == "completed"
    assert result["commands"][0]["detail"]["waypoints"] == 2
    assert result["commands"][0]["detail"]["measured_progress_m"] >= .00015


def test_trace_simplification_keeps_collinear_advance_retract_reversal():
    event = dict(id="probe", primitive="servo_stream", source_event_ids=["probe"],
                 pose_before_world_eef=pose(.30), pose_after_world_eef=pose(.40))
    reference = dict(control_samples=[dict(primitive_id="probe", pose_world_eef=pose(x)) for x in (.37, .34, .40)])
    route = adapter._trace_cartesian_route(reference, event, np.eye(4))
    xs = [command["pose_world_tcp"][0][3] for command in route]
    assert xs == pytest.approx([.37, .34, .40])


def test_waypoints_keep_commanded_reference_continuous_when_feedback_lags(tmp_path):
    class LaggingPanda(FakePanda):
        def __init__(self, root):
            super().__init__(root)
            self.targets = []

        def tick(self):
            super().tick()
            self.pose[0, 3] -= .0004
            return True

        def set_target(self, goal):
            self.targets.append(float(goal[0, 3]))
            super().set_target(goal)

    program = schedule([dict(id="curve", kind="move_path_tcp", waypoints_world_tcp=[pose(.301), pose(.302)], timeout_s=2)])
    config = calibration()
    add_evidence(tmp_path, program, config)
    backend = LaggingPanda(tmp_path)
    result = adapter.execute_schedule(program, config, execute=True,
        arm_token=adapter.required_arm_token(program, config), backend=backend, evidence_root=tmp_path)
    assert result["status"] == "completed"
    assert min(backend.targets) >= .30
    assert np.min(np.diff(backend.targets)) >= -1e-10


def test_stationary_complete_joint_trace_becomes_reviewed_pose_checkpoint():
    event = dict(id="nullspace", stage="acceptance", primitive="execute_joint",
                 pose_before_world_eef=pose(), pose_after_world_eef=pose())
    reference = dict(task_id="test_task", candidate_id="candidate_1", twin_validated=True,
        start_pose_world_eef=pose(), events=[event], control_samples=[dict(primitive_id="nullspace", pose_world_eef=pose())])
    compiled = adapter.compile_reference_schedule(reference, calibration())
    assert compiled["hardware_executable"] is False
    assert compiled["commands"][0]["kind"] == "verify_pose"
    assert compiled["commands"][0]["stationary_sim_joint_posture_change_discarded"] is True


def test_only_free_paths_are_chunked_with_conservative_time_budget():
    event = dict(id="free", pose_before_world_eef=pose(.30))
    route = [dict(pose_world_tcp=pose(x)) for x in (.45, .6, .75, .6)]
    config = calibration()
    proposal = adapter._path_proposal(route, event, "move_path_tcp", config, np.eye(4))
    chunks = adapter._split_free_path(proposal, event, config, np.eye(4))
    assert len(chunks) > 1
    assert all(command["timeout_s"] <= 90 for command in chunks)
    assert [point for command in chunks for point in command["waypoints_world_tcp"]] == [command["pose_world_tcp"] for command in route]


def test_depth_feedback_measures_real_tcp_not_ee_origin_when_tool_rotates():
    config = calibration()
    config["T_ee_tcp"][2][3] = .1
    origin, actual = np.asarray(pose()), np.asarray(pose(.305))
    angle = .04
    actual[:3, :3] = [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
    depth = adapter._tcp_progress(actual, origin, np.array([1., 0., 0.]), config)
    assert depth == pytest.approx(.005 + .1*np.sin(angle))


def test_cli_failed_preflight_replaces_a_stale_success_report(tmp_path):
    program_path, config_path, report = (tmp_path / name for name in ("program.json", "config.json", "report.json"))
    program_path.write_text("{}", encoding="utf-8")
    config_path.write_text("{}", encoding="utf-8")
    report.write_text(json.dumps({"status": "completed", "task_accepted": True}), encoding="utf-8")
    assert adapter.main(["run", "--schedule", str(program_path), "--calibration", str(config_path),
                         "--report", str(report), "--execute"]) == 2
    failure = json.loads(report.read_text(encoding="utf-8"))
    assert failure["status"] == "blocked" and failure["task_accepted"] is False
