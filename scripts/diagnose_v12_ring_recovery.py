"""Development-only real prefix replay and checked simulator checkpoint probe.

This never reconstructs object poses from final_positions. A checkpoint is
captured only after the proposal prefix has physically executed to press_seat.
Checkpoint probes are local continuation experiments, not independent trials.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle
import time
import numpy as np

from simbench.assembly.library import Session, Result
from simbench.assembly.ring_insertion_v12 import execute
from simbench.value.system_v11 import save, twin_checkpoint, restore_twin_checkpoint
from simbench.value.system_v12 import make_scene, rollout
from simbench.value.provenance_v12 import fingerprint


def capture(session, params):
    return dict(state=twin_checkpoint(session), params=params,
        held_visual_transforms=deepcopy(session.held_visual_transforms_v12),
        model_dynamic={name:getattr(session.ctx.model,name).copy() for name in
            ("geom_friction", "actuator_gainprm", "actuator_biasprm")},
        prefix_steps=deepcopy(session.results), runtime=fingerprint())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved-result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--continue-suffix", action="store_true")
    args = parser.parse_args()
    saved = json.loads(args.saved_result.read_text())
    seed = int(saved["seed"])
    if seed not in range(1600, 1608):
        raise ValueError("this development probe only accepts already-used development seeds 1600-1607")
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    provenance = dict(seed=seed, saved_source=str(args.saved_result),
        saved_source_sha256=hashlib.sha256(args.saved_result.read_bytes()).hexdigest(),
        proposal=saved["proposal"], domain=saved["domain"], runtime=fingerprint(),
        checkpoint_is_local_continuation_only=bool(args.checkpoint),
        control_hook="development-only replacement of handle _press_functional_v12")
    if args.checkpoint:
        # Only consume a trusted local checkpoint created by this script.
        state = pickle.loads(args.checkpoint.read_bytes())
        _, session, _, _ = make_scene(seed, args.out/"scene", domain=saved["domain"])
        restore_twin_checkpoint(session, state["state"])
        session.held_visual_transforms_v12 = deepcopy(state["held_visual_transforms"])
        for name, value in state["model_dynamic"].items():
            getattr(session.ctx.model, name)[:] = value
        metrics = execute(session, **state["params"])
        save(args.out/"ring_metrics.json", metrics)
        provenance.update(checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            prefix_runtime=state["runtime"], prefix_executed_steps=len(state["prefix_steps"]),
            ring_success=metrics["success"], full_system_success=None)
        if args.continue_suffix and metrics["success"]:
            from simbench.value.planner_v12 import assembly_program
            from simbench.value.plan import execute_calls
            from simbench.value.full_task_v7 import _stroke
            plan = assembly_program(session, saved["proposal"])
            handle_calls = [c for c in plan.calls if c.roles.get("manipulated") == "handle"
                            and not c.id.startswith("accept_")]
            press_index = next(i for i,c in enumerate(handle_calls) if c.skill == "press")
            suffix = dict(success=False, attempted_calls=[c.id for c in handle_calls[press_index+1:]],
                          source="actual continuation after ring, all existing release/stroke predicates unchanged")
            session.results.clear()
            try:
                execute_calls(session, plan, handle_calls[press_index+1:])
                session.call("move", target="home")
                suffix["released_and_retreated"] = True
                _stroke(session, float(session.planning_cad["functional_stroke_minimum_m"]))
                suffix["success"] = True
                # Independent final diagnostics after all controller actions;
                # these do not feed back a correction or invent a full run.
                from simbench.assembly.skills_v12 import evaluate_end_stop_fixture
                from simbench.assembly.control import HOME
                for pin, y in (("pin_left", -.032), ("pin_right", .032)):
                    session.call("inspect", what="pin", part=pin, hole_part="end_stop",
                        hole_offset_m=[0., y, 0.], minimum_insertion_depth_m=session.pin_insertion_config.required_depth_m,
                        phase="retained_after_stroke")
                fixture_ok, fixture = evaluate_end_stop_fixture(session)
                suffix["post_stroke_fixture"] = fixture
                suffix["final_functional_checks_passed"] = bool(fixture_ok and session.held is None
                    and np.max(np.abs(session.ctx.arm_qpos-HOME)) < .02)
            except Exception as exc:
                suffix["error"] = str(exc)
            suffix["executed_steps"] = session.results
            save(args.out/"suffix_result.json", suffix)
            provenance["continued_suffix_success"] = suffix["success"]
    else:
        original = Session._press_functional_v12
        def hooked(session, part, target_z, force_stop):
            if part != "handle":
                return original(session, part, target_z, force_stop)
            params = dict(part=part, target_z=target_z, force_stop=force_stop)
            state = capture(session, params)
            (args.out/"before_ring.pkl").write_bytes(pickle.dumps(state))
            save(args.out/"prefix_execution.json", dict(steps=state["prefix_steps"], params=params,
                source="actual full-task execution before the ring skill; no pose reset from labels"))
            metrics = execute(session, **params)
            save(args.out/"ring_metrics.json", metrics)
            return Result(metrics["success"], metrics, "ring functional engagement or force guard failed")
        Session._press_functional_v12 = hooked
        try:
            result = rollout(seed, saved["proposal"], args.out/"rollout", domain=saved["domain"])
        finally:
            Session._press_functional_v12 = original
        provenance.update(full_system_success=result["success"], rollout_valid=result["valid"],
            ring_reached=(args.out/"ring_metrics.json").exists())
    provenance["actual_wall_seconds"] = time.perf_counter()-start
    save(args.out/"development_provenance.json", provenance)
    print(json.dumps({k:v for k,v in provenance.items() if k in
        ("seed", "ring_reached", "ring_success", "full_system_success", "actual_wall_seconds")}))


if __name__ == "__main__":
    main()
