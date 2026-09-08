"""Conditional skill graph and pre-execution checking of JSON skeletons."""

import argparse
from dataclasses import asdict
import inspect
import json
from pathlib import Path
from .library import CATALOG, Session, PARTS, SkillFailure


def validate_skeleton(steps, initial_held=None, initial_artifacts=None):
    if not steps:
        return dict(valid=False, errors=["empty skeleton"])
    held = initial_held
    seen = set()
    artifacts = dict(initial_artifacts or {})
    errors = []
    bindings = []
    for i, row in enumerate(steps):
        name = row.get("skill")
        params = row.get("params", {})
        if name not in CATALOG:
            errors.append(dict(step=i, reason="unknown skill"))
            break
        try:
            bound = inspect.signature(getattr(Session, name)).bind(None, **params)
            bound.apply_defaults()
        except TypeError as exc:
            errors.append(dict(step=i, reason=str(exc)))
            break
        p = dict(bound.arguments)
        p.pop("self")
        part = p.get("part")
        spec = CATALOG[name]
        if part is not None and part not in PARTS:
            errors.append(dict(step=i, reason="unknown part"))
            break
        for requirement in spec.requires:
            reason = None
            if requirement == "held" and held != part:
                reason = f"held({part}) required; currently {held}"
            if requirement == "empty" and held is not None:
                reason = "empty gripper required"
            if requirement == "seen" and part not in seen:
                reason = "observation missing"
            if requirement == "pin" and part not in ("pin_left", "pin_right"):
                reason = "pin geometry required"
            if requirement.startswith("artifact:"):
                artifact = artifacts.get(p.get("artifact"))
                kind = requirement.split(":", 1)[1]
                if artifact is None or artifact["type"] != kind:
                    reason = f"{kind} artifact missing"
                elif part is not None and artifact.get("part") not in (None, part):
                    reason = "artifact bound to a different part"
                else:
                    bindings.append(
                        dict(
                            producer=artifact.get("step", -1),
                            consumer=i,
                            type=kind,
                            part=part,
                        )
                    )
            if reason:
                errors.append(dict(step=i, reason=reason))
        if errors:
            break
        if name == "observe_parts":
            seen = set(PARTS)
        if name == "close_gripper":
            held = part
        if name == "open_gripper":
            held = None
        if spec.produces and "as_" in p:
            artifacts[p["as_"]] = dict(type=spec.produces, part=part, step=i)
    return dict(
        valid=not errors,
        errors=errors,
        artifact_edges=bindings,
        note="Structural necessary conditions; live contact gates and continuous planning remain required.",
    )


def execute_skeleton(session, steps):
    verdict = validate_skeleton(steps, session.held, session.artifacts)
    if not verdict["valid"]:
        raise SkillFailure(str(verdict["errors"]))
    for row in steps:
        session.call(row["skill"], **row.get("params", {}))


def catalog_graph():
    edges = []
    for a, source in CATALOG.items():
        if source.produces:
            for b, target in CATALOG.items():
                if "artifact:" + source.produces in target.requires:
                    edges.append(
                        dict(
                            source=a,
                            target=b,
                            condition=source.produces + "(same object)",
                        )
                    )
    handoffs = {
        "observe_parts": ["estimate_pose"],
        "select_grasp": ["plan_transfer"],
        "execute_joint_path": ["approach", "align_axis", "lower"],
        "approach": ["close_gripper"],
        "close_gripper": ["verify_grasp", "lift", "move_constrained"],
        "verify_grasp": ["lift", "plan_insertion"],
        "lift": ["plan_transfer", "orient_wrist"],
        "orient_wrist": ["plan_transfer"],
        "lower": ["align_axis", "guarded_descent"],
        "align_axis": ["plan_insertion", "guarded_descent"],
        "slide_insert": ["open_gripper", "inspect_seat"],
        "guarded_descent": ["press_seat", "plan_recovery"],
        "retract_contact": ["plan_insertion"],
        "spiral_search": ["learned_insert", "guarded_descent"],
        "learned_insert": ["press_seat"],
        "press_seat": ["open_gripper"],
        "open_gripper": ["retreat"],
        "retreat": ["inspect_seat", "observe_parts", "home"],
        "inspect_seat": ["measure_clearance", "observe_parts"],
        "move_constrained": ["verify_stroke", "open_gripper"],
        "verify_stroke": ["open_gripper"],
    }
    for a, targets in handoffs.items():
        for b in targets:
            edges.append(
                dict(
                    source=a,
                    target=b,
                    condition="all destination contracts + continuous feasibility",
                )
            )
    return dict(
        nodes=[asdict(x) for x in CATALOG.values()],
        edges=edges,
        note="Conditional composability, not unconditional pairwise reachability. Data/resource bindings are checked on complete skeletons.",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps")
    ap.add_argument("--out", default="results/tabletop/skill_graph.json")
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    graph = catalog_graph()
    if a.steps:
        graph["skeleton_validation"] = validate_skeleton(
            json.loads(Path(a.steps).read_text())
        )
    out.write_text(json.dumps(graph, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            dict(
                nodes=len(graph["nodes"]),
                edges=len(graph["edges"]),
                validation=graph.get("skeleton_validation"),
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
