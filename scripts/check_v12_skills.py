"""Independent physical skill smoke tests on the original printed geometry.

Pin tests start with a preinstalled (unwelded) end-stop. This is an isolated
skill fixture, not a successful complete-task trial. All later motion is
ordinary robot physics; no object teleport, weld or oracle localization.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
from scipy.spatial.transform import Rotation

from simbench.assembly.skills_v12 import configure_v12_skills
from simbench.value.stage_v12 import make_scene
from simbench.value.stage_v7 import refresh_visual_observation
from simbench.value.stage_v5 import stage_calls
from simbench.value.plan import execute_calls, argument
from simbench.value.full_task_v7 import _clean


def run(seed, skill, output, fixture_mode="free_visual"):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    spec, session, path, targets = make_scene(seed, output, preinstalled_end_stop=skill == "pin")
    if skill == "pin" and fixture_mode == "fixed_calibrated":
        # Independent fixture experiment: the exact same CAD part is bolted
        # at a known installation calibration before a fresh simulation starts.
        # No running object's state is rewritten, and no pose truth is read.
        import xml.etree.ElementTree as ET
        import mujoco
        from simbench.assembly.control import HOME
        from simbench.assembly.library import Session
        from simbench.core.sim_context import MjContext
        root = ET.parse(path).getroot()
        body = root.find(".//body[@name='end_stop']")
        for joint in list(body.findall("freejoint")):
            body.remove(joint)
        body.set("pos", " ".join(map(str, targets["end_stop"])))
        body.set("quat", "1 0 0 0")
        path = output / "fixed_calibrated_fixture.xml"
        ET.ElementTree(root).write(path, encoding="unicode")
        old = session
        ctx = MjContext(path, control_freq=50); ctx.reset(); ctx.data.qpos[ctx.arm_qadr] = HOME
        mujoco.mj_forward(ctx.model, ctx.data); ctx.hold_arm(); ctx.set_finger_ctrl(.04)
        for _ in range(80): ctx.step()
        session = Session(ctx, seed=seed, out=output, parts=old.parts,
                          grasp_specs=old.grasp_specs, capabilities=old.capabilities)
        for key in ("stage_targets", "task_version", "pin_insertion_config", "visual_templates_fn", "planning_cad"):
            setattr(session, key, getattr(old, key))
        for key in ("perception_estimator_fn", "capture_detector_fn", "detector_size"):
            if hasattr(old, key):
                setattr(session, key, getattr(old, key))
    configure_v12_skills(session)
    observation = refresh_visual_observation(session)
    (output / "observation.json").write_text(json.dumps(observation, indent=2), encoding="utf-8")
    success = False; error = None
    try:
        if skill == "wipe":
            _clean(session, variant=seed % 4, force=1.5, duration=14.)
        else:
            if fixture_mode == "fixed_calibrated":
                target = np.asarray(targets["end_stop"], float) + [0., -.032,
                    .018 - session.pin_insertion_config.shaft_tip_offset_m - .008]
            else:
                row = observation["objects"]["end_stop"]
                if not row.get("valid"):
                    raise ValueError("no valid installed fixture RGB-D observation")
                q = np.asarray(row["quat_wxyz"])
                R = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
                target = np.asarray(row["position_m"]) + R @ [0, -.032,
                    .018 - session.pin_insertion_config.shaft_tip_offset_m - .008]
            choices = dict(yaw=0., height=.001, clearance=.99, force=6.,
                           speed=.004, force_limit=12., press_force=2.)
            calls = stage_calls("pin_left", target, choices, 0, v7=True, v12=True)
            calls[0].arguments["required_parts"] = argument(["pin_left"])
            plan = SimpleNamespace(prefix={"part": "pin_left", "targets": {"pin_left": target.tolist()}})
            execute_calls(session, plan, calls)
            session.hold(1.)
            session.call("inspect", what="pin", part="pin_left", hole_part="end_stop",
                         hole_offset_m=[0., -.032, 0.], minimum_insertion_depth_m=.006)
        success = True
    except Exception as exc:
        error = str(exc)
    report = dict(seed=seed, skill=skill, success=success, error=error,
        setup=fixture_mode if skill == "pin" else "original_printed_wipe_tool",
        complete_task_trial=False, scene=str(path), physical_steps=int(session.ctx.data.time / .002),
        wall_seconds=time.perf_counter()-start,
        learned_policy_called=any(x["skill"] == ("learned_insert" if skill == "pin" else "wipe_surface") for x in session.results),
        steps=session.results)
    (output / "summary.json").write_text(json.dumps(report, indent=2, default=lambda x: np.asarray(x).tolist()), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "steps"}), flush=True)
    return report


if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=1620)
    ap.add_argument("--skill", choices=("pin", "wipe"), required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--fixture-mode", choices=("free_visual", "fixed_calibrated"), default="free_visual")
    a=ap.parse_args(); run(a.seed,a.skill,a.out,a.fixture_mode)
