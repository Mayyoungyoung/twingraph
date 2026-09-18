"""Execution of one complete v7 candidate.

The value plan is a single explicit full-task atom so a candidate cannot be
mistaken for the old assembly-only prefix.  Internally it still calls the
existing feedback-controlled atomic skills; no body is teleported or welded.
"""

import copy
import numpy as np

from simbench.assembly.control import HOME
from simbench.assembly.scene import CENTER
from simbench.assembly.task import pick, transfer_part, release
from simbench.assembly.library import Result, SkillFailure
from . import stage_v5 as assembly
from .plan import execute_calls


TOOL_HOME = np.array([-0.31, 0.20, 0.804])


def _update_execution_targets(plan, session):
    """Use post-placement feedback for fixture-relative pin/handle targets.

    This is an execution-controller observation, not a label or candidate
    feature.  It prevents a noisy end-stop placement from making the pins aim
    at stale world coordinates.
    """
    end = np.asarray(session.ctx.obj_pos("end_stop"), dtype=float)
    bid = session.ctx.body_id("end_stop")
    R = session.ctx.data.xmat[bid].reshape(3, 3)
    pin_targets = {
        "pin_left": (end + R @ np.array([0.0, -0.032, 0.0])).tolist(),
        "pin_right": (end + R @ np.array([0.0, 0.032, 0.0])).tolist(),
    }
    pin_targets["pin_left"][2] = float(session.stage_targets["pin_left"][2])
    pin_targets["pin_right"][2] = float(session.stage_targets["pin_right"][2])
    carriage = np.asarray(session.ctx.obj_pos("carriage"), dtype=float)
    pin_targets["handle"] = (carriage + np.array([0.0, 0.0, 0.048])).tolist()
    for part, target in pin_targets.items():
        session.stage_targets[part] = list(target)
    for call in plan.calls:
        part = call.roles.get("manipulated")
        if part not in pin_targets:
            continue
        target = np.asarray(pin_targets[part], dtype=float)
        for name, arg in call.arguments.items():
            value = arg.value
            # Do not rewrite the deferred source/hover target of a pin pick.
            # Only concrete fixture-relative placement/approach targets are
            # rebound after the measured end-stop placement.
            if (isinstance(value, (list, tuple)) and len(value) == 3
                    and name == "target" and getattr(arg, "source_call", None) is None):
                arg.value = [float(target[0]), float(target[1]), float(value[2])]


def _clean(session, variant, force, duration):
    tool = "wipe_tool"
    pick(session, tool, lift=False,
         terminal_targets=[dict(id="wipe", part=tool, xyz=session.ctx.obj_pos(tool).copy())])
    halfspan = np.array([.055, .006], dtype=float)
    session.call("plan_path", method="surface", part=tool, center=CENTER.tolist(),
                 halfspan=halfspan.tolist(), surface="guide_base", height=.8215,
                 duration=float(duration))
    points = np.asarray(session.artifacts["wipe"]["points"], dtype=float)
    if int(variant) % 2:
        points[:, 0] = 2 * CENTER[0] - points[:, 0]
    if int(variant) // 2:
        points = points[::-1].copy()
    session.artifacts["wipe"]["points"] = points.tolist()
    start = points[0]
    transfer_part(session, tool, start + [0, 0, .075])
    session.call("move", reference="object", part=tool, target=start + [0, 0, .000])
    session.call("wipe", part=tool, target_force=float(force), force_limit=12.0,
                 minimum_coverage=.72)
    session.call("move", part=tool, delta=[0, 0, .10])
    transfer_part(session, tool, TOOL_HOME + [0, 0, .06])
    session.call("move", mode="guarded", part=tool, target_z=float(TOOL_HOME[2]), force_stop=3.0)
    session.call("press", part=tool, target_z=float(TOOL_HOME[2]), force_stop=2.0)
    session.call("place", part=tool, target=TOOL_HOME, tol=.002, settle=.25)
    session.call("move", delta=[0, 0, .12])
    session.call("inspect", what="clean", threshold=.05)
    session.stage_passes["cleaning_pass"] = True


def _stroke(session, minimum):
    carriage = "carriage"
    handle = "handle"
    current = session.ctx.obj_pos(handle).copy()
    # The installed handle is the real operating interface.  Holding it
    # couples the end-effector to the carriage through the assembled post;
    # the carriage body remains the independently measured sliding body.
    pick(session, handle, lift=False, execution_feedback=True,
         terminal_targets=[dict(id="stroke", part=handle, xyz=current.copy())])
    session.call("move", mode="constrained", part=carriage,
                 target_x=float(CENTER[0] - .04))
    session.call("move", mode="constrained", part=carriage,
                 target_x=float(CENTER[0] + .060))
    session.call("inspect", what="stroke", minimum=float(minimum))
    release(session)
    session.call("move", target="home")
    session.stage_passes["functional_test_pass"] = True


def execute_full_task(session, order, choices, wipe_variant=0, wipe_force=1.5,
                      wipe_duration=14.0, stroke_minimum=.08):
    required = set(assembly.PARTS)
    if tuple(order) not in assembly.legal_orders([tuple(order)]):
        raise SkillFailure("full-task candidate violates assembly precedence")
    if set(choices) != required:
        raise SkillFailure("full-task candidate choices do not cover five parts")
    session.stage_passes = {"cleaning_pass": False, "assembly_pass": False,
                            "functional_test_pass": False,
                            "final_release_and_retraction_pass": False}
    _clean(session, wipe_variant, wipe_force, wipe_duration)
    # Reuse the audited assembly program, but execute it in this live session
    # after the clean stage.  All deferred poses are solved from the measured
    # grasp and the candidate's force/speed/yaw choices.
    assembly_choices = copy.deepcopy(choices)
    # The carriage uses the fixed constrained insertion speed in the audited
    # assembly atom; carrying an unused speed field into stage_v5's executable
    # payload would falsely make the semantic choice disagree with its calls.
    # Keep the pin insertion speed as an executable candidate parameter.  The
    # earlier v7 smoke path discarded it together with the carriage speed,
    # making every pin use the same relatively aggressive 6 mm/s descent.  A
    # slower candidate must be able to change the real contact trajectory;
    # only the carriage's fixed horizontal insertion ignores this field.
    assembly_choices.get("carriage", {}).pop("speed", None)
    plan = assembly.program(session, session.stage_targets, tuple(order),
                            assembly_choices, initial_route_index=0, v7=True)
    first_pin = next((i for i, call in enumerate(plan.calls)
                      if call.roles.get("manipulated") in {"pin_left", "pin_right"}), len(plan.calls))
    execute_calls(session, plan, plan.calls[:first_pin])
    _update_execution_targets(plan, session)
    execute_calls(session, plan, plan.calls[first_pin:])
    session.stage_passes["assembly_pass"] = True
    _stroke(session, stroke_minimum)
    all_released = session.held is None and np.max(np.abs(session.ctx.arm_qpos - HOME)) < .02
    session.stage_passes["final_release_and_retraction_pass"] = bool(all_released)
    if not all(session.stage_passes.values()):
        raise SkillFailure("complete-task stage predicate failed")
    return Result(True, {
        "stage_passes": copy.deepcopy(session.stage_passes),
        "cleaning": copy.deepcopy(session.artifacts.get("wipe_result", {})),
        "stroke": {"runs": copy.deepcopy(session.stroke_runs),
                    "peak_force_n": max(session.stroke_peak_forces, default=0.0)},
        "task_scope": "clean_assemble_and_post_handle_bidirectional_stroke",
    })
