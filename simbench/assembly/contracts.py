"""One contract interpreter for symbolic propagation and live dispatch.

Symbolic effects are conditional on successful execution. Unknown continuous
constraints are obligations, never evidence that an alternative is impossible.
"""

from dataclasses import dataclass, field
import copy


@dataclass
class State:
    parts: tuple
    held: str | None = None
    seen: set = field(default_factory=set)
    artifacts: dict = field(default_factory=dict)
    # Object capabilities belong to the scene, not to graph node names.
    capabilities: dict = field(default_factory=dict)
    grasp_epoch: int | None = None


def check(spec, params, state, contact=None):
    conflicts, unknown = [], []
    part = params.get("part")
    if part is not None and part not in state.parts:
        conflicts.append(f"unknown part {part}")
    requested = params.get("required_parts")
    if requested is not None:
        for item in requested:
            if item not in state.parts:
                conflicts.append(f"unknown required observation part {item}")
    for requirement in spec.requires:
        if requirement == "ownership":
            # Optional part is an explicit promise: None means empty motion.
            if state.held != part:
                conflicts.append(
                    f"move ownership mismatch: held={state.held}, part={part}"
                )
            elif part is not None:
                if contact is None:
                    unknown.append(f"live bilateral grasp contact({part})")
                elif not contact(part)["held"]:
                    conflicts.append(f"verified held({part}) required; contact lost")
        if requirement == "empty" and state.held is not None:
            conflicts.append(f"empty gripper required; occupied by {state.held}")
        elif requirement == "held":
            if state.held != part:
                conflicts.append(
                    f"verified held({part}) required; currently {state.held}"
                )
            elif contact is None:
                unknown.append(f"live bilateral grasp contact({part})")
            elif not contact(part)["held"]:
                conflicts.append(f"verified held({part}) required; contact lost")
        elif requirement == "seen" and part not in state.seen:
            conflicts.append(f"observation missing: observe {part} first")
        elif requirement == "pin":
            if "pin" not in state.capabilities.get(part, ()):
                conflicts.append(f"pin geometry capability required for {part}")
        elif requirement.startswith("capability:"):
            capability = requirement.split(":", 1)[1]
            if capability not in state.capabilities.get(part, ()):
                conflicts.append(f"{capability} capability required for {part}")
        elif requirement.startswith("artifact:"):
            key = params.get("artifact", "default")
            item = state.artifacts.get(key)
            kind = requirement.split(":", 1)[1]
            if item is None or item.get("type") != kind:
                conflicts.append(f"requires {kind} artifact {key}")
            elif part is not None and item.get("part") not in (None, part):
                conflicts.append("artifact bound to wrong part")
            elif item.get("binding") is not None:
                binding = item["binding"]
                if binding.get("held") != state.held:
                    conflicts.append("artifact held-object binding changed")
                if "grasp_epoch" in binding:
                    if state.grasp_epoch is None:
                        unknown.append("resolve grasp acquisition epoch")
                    elif binding["grasp_epoch"] != state.grasp_epoch:
                        conflicts.append("artifact grasp acquisition epoch changed")
                grasp_key = binding.get("grasp_artifact")
                if grasp_key:
                    current = state.artifacts.get(grasp_key, {})
                    if current.get("deferred") and "id" not in current:
                        unknown.append(
                            "resolve generated grasp identity before path reuse"
                        )
                    elif current.get("id") != binding.get("grasp_id"):
                        conflicts.append("artifact grasp binding changed")
            if item is not None and item.get("deferred"):
                message = f"materialize {key} after its bound prefix succeeds"
                (unknown if contact is None else conflicts).append(message)
    if contact is None:
        unknown.extend(spec.obligations)
    return dict(conflicts=conflicts, unknown=unknown)


def apply_effects(spec, params, state, step=-1, symbolic=True):
    """Apply only success effects; runtime measurements stay in controller methods."""
    if not symbolic and spec.produces and "as_" in params:
        output = state.artifacts.get(params["as_"], {})
        if output.get("type") != spec.produces:
            raise ValueError("generator output violates declared artifact type")
        for field_name, parameter in spec.output_bindings:
            if (
                params.get(parameter) is not None
                and output.get(field_name) != params[parameter]
            ):
                raise ValueError(f"generator output violates {field_name} binding")
    for effect in spec.effects:
        if effect == "seen:all":
            requested=params.get("required_parts")
            state.seen = set(state.parts if requested is None else requested)
        elif effect == "held:part":
            state.held = params["part"]
            if state.grasp_epoch is not None:
                state.grasp_epoch += 1
        elif effect == "held:empty":
            state.held = None
            if state.grasp_epoch is not None:
                state.grasp_epoch += 1
    if symbolic and spec.produces and "as_" in params:
        state.artifacts[params["as_"]] = dict(
            type=spec.produces, part=params.get("part"), step=step, deferred=True
        )
        for field_name, parameter in spec.output_bindings:
            if params.get(parameter) is not None:
                state.artifacts[params["as_"]][field_name] = params[parameter]
    return state


def session_state(session):
    return State(
        session.parts,
        session.held,
        set(session.observations),
        copy.deepcopy(session.artifacts),
        session.capabilities,
        session.grasp_epoch,
    )
