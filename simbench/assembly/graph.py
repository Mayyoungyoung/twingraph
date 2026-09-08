"""Contract-derived dependency graph and conservative whole-skeleton validation."""

import argparse
import copy
from dataclasses import asdict
import inspect
import json
from pathlib import Path
from .library import HANDLERS, Session, PARTS, DEFAULT_CAPABILITIES, SkillFailure
from .contracts import State, check, apply_effects
from .interfaces import resolve, INTERFACES, PUBLIC_SKILLS


def validate_skeleton(
    steps,
    initial_held=None,
    initial_artifacts=None,
    parts=PARTS,
    initial_seen=(),
    capabilities=None,
    grasp_epoch=None,
):
    state = State(
        tuple(parts),
        initial_held,
        set(initial_seen),
        copy.deepcopy(initial_artifacts or {}),
        (
            capabilities
            if capabilities is not None
            else {
                p: DEFAULT_CAPABILITIES[p] for p in parts if p in DEFAULT_CAPABILITIES
            }
        ),
    )
    state.grasp_epoch = grasp_epoch
    errors, unknown, bindings, checked = [], [], [], []
    if not steps:
        errors.append(dict(step=-1, reason="empty skeleton"))
    for i, row in enumerate(steps):
        try:
            name, params = resolve(row.get("skill"), row.get("params", {}))
            if name not in HANDLERS:
                raise ValueError("unknown skill")
            bound = inspect.signature(getattr(Session, name)).bind(None, **params)
            bound.apply_defaults()
            params = dict(bound.arguments)
            params.pop("self")
        except (ValueError, TypeError) as exc:
            errors.append(dict(step=i, reason=str(exc)))
            break
        spec = HANDLERS[name]
        verdict = check(spec, params, state)
        errors.extend(dict(step=i, reason=x) for x in verdict["conflicts"])
        unknown.extend(dict(step=i, reason=x) for x in verdict["unknown"])
        if verdict["conflicts"]:
            break
        checked.append(dict(step=i, component=name, checks=list(spec.requires)))
        for requirement in spec.requires:
            if requirement.startswith("artifact:"):
                item = state.artifacts[params["artifact"]]
                bindings.append(
                    dict(
                        producer=item.get("step", -1),
                        consumer=i,
                        type=item["type"],
                        part=item.get("part"),
                    )
                )
        apply_effects(spec, params, state, step=i)
    return dict(
        valid=not errors,
        status="conflict" if errors else ("unknown" if unknown else "necessary_pass"),
        necessary_checks_passed=not errors,
        errors=errors,
        unknown=unknown,
        checked=checked,
        artifact_edges=bindings,
        final_state=dict(held=state.held, seen=sorted(state.seen)),
        note="Success-conditional structural propagation; unknown obligations require solver/physics verification.",
    )


def execute_skeleton(session, steps):
    verdict = validate_skeleton(
        steps,
        session.held,
        session.artifacts,
        session.parts,
        session.observations,
        session.capabilities,
        session.grasp_epoch,
    )
    if not verdict["valid"]:
        raise SkillFailure(str(verdict["errors"]))
    for row in steps:
        session.call(row["skill"], **row.get("params", {}))
    return verdict


def catalog_graph(include_components=False):
    from .candidates import TEMPLATES

    edges = []
    for a, source in HANDLERS.items():
        supplied = set(source.effects)
        if source.produces:
            supplied.add("artifact:" + source.produces)
        for b, target in HANDLERS.items():
            needs = set(target.requires)
            supported = supplied & needs
            for effect, predicate in (
                ("held:part", "held"),
                ("held:empty", "empty"),
                ("seen:all", "seen"),
            ):
                if effect in supplied and predicate in needs:
                    supported.add(predicate)
            if supported:
                edges.append(
                    dict(
                        source=a,
                        target=b,
                        supplies=sorted(supported),
                        condition="same object binding; all remaining contracts still required",
                    )
                )
    if include_components:
        return dict(
            nodes=[asdict(x) for x in HANDLERS.values()],
            edges=edges,
            note="Implementation details only; these are not separate task-facing atoms.",
        )
    grouped = {}
    for edge in edges:
        source, target = (
            HANDLERS[edge["source"]].family,
            HANDLERS[edge["target"]].family,
        )
        if (
            source not in PUBLIC_SKILLS
            or target not in PUBLIC_SKILLS
            or source == target
        ):
            continue
        key = (source, target)
        entry = grouped.setdefault(
            key, dict(source=source, target=target, supplies=[], variants=[])
        )
        entry["supplies"] = sorted(set(entry["supplies"]) | set(edge["supplies"]))
        entry["variants"].append(edge)
    nodes = []
    for name, atom in PUBLIC_SKILLS.items():
        nodes.append(
            dict(
                name=name,
                **atom,
                contracts=[
                    asdict(spec) for spec in HANDLERS.values() if spec.family == name
                ]
            )
        )
    return dict(
        nodes=nodes,
        edges=list(grouped.values()),
        atom_count=len(nodes),
        templates=[dict(name=n, kind="macro") for n in TEMPLATES],
        note="One node per atom. Edge variants are conditional alternatives; never combine their preconditions as one contract. Validate complete bound calls.",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps")
    ap.add_argument(
        "--components",
        action="store_true",
        help="export internal handlers instead of task-facing atoms",
    )
    ap.add_argument("--out", default="results/tabletop/skill_graph.json")
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    graph = catalog_graph(include_components=a.components)
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
