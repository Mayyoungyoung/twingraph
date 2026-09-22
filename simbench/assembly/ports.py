"""Execution interface metadata, shared by skill dispatch and graph compilation.

Names/defaults come from the actual Python signature. Units and state effects
are controller interface declarations, never product-specific value features.
"""
from dataclasses import dataclass
import inspect
import math


@dataclass(frozen=True)
class Port:
    name: str
    kind: str
    unit: str = ""
    frame: str = ""
    required: bool = False


METRIC = {
    "target": ("position", "m", "world"),
    "delta": ("position", "m", "world_delta"),
    "axis": ("vector", "1", "world"),
    "yaw": ("scalar", "rad", "world"),
    "yaws": ("vector", "rad", "world"),
    **{k: ("scalar", "m", "world" if k in {"target_z", "target_x", "clearance"} else "")
       for k in ("height", "height_offset", "width", "radius", "tol", "tolerance", "target_z", "target_x", "clearance")},
    **{k: ("scalar", "N", "") for k in ("force", "force_stop", "force_limit", "max_force", "minimum_support", "target_force")},
    "speed": ("scalar", "m/s", ""), "settle": ("scalar", "s", ""),
    "duration": ("scalar", "s", ""), "center": ("position", "m", "world"),
    "halfspan": ("vector", "m", "surface"),
    "hole_offset_m": ("vector", "m", "fixture"),
    "minimum_insertion_depth_m": ("scalar", "m", "fixture"),
    "pin_command_depth_m": ("scalar", "m", "receiver_axis"),
    "pin_press_extra_m": ("scalar", "m", "receiver_axis"),
    "hole_entry_m": ("position", "m", "world"),
    "acceptance": ("category", "", ""), "hole_part": ("object_ref", "", ""),
    "phase": ("category", "", ""), "strategy": ("category", "", ""),
    "yaw_frame": ("category", "", ""),
    "required_parts": ("object_refs", "", ""),
}


def declare_ports(fn):
    ports = []
    for name, p in inspect.signature(fn).parameters.items():
        if name == "self":
            continue
        kind, unit, frame = METRIC.get(name, ("scalar", "", ""))
        if name in {"part", "surface"}: kind = "object_ref"
        elif name in {"artifact", "grasp_artifact", "as_"}: kind = "artifact_ref"
        elif name in {"candidate_id", "index"}: kind = "binding"
        elif name in {"order", "choices"}: kind = "record"
        elif name in {"quantity", "policy"}: kind = "category"
        ports.append(Port(name, kind, unit, frame, p.default is inspect.Parameter.empty))
    return tuple(ports)


def state_interface(name, category):
    """Conservative possible reads/writes, not a proof of physical feasibility.

Holding is the persistent grasp relation, separate from the changing pose.
Scene writes preserve cross-object dependencies after release and motion.
"""
    reads = ["scene", "robot", "holding", "object:{part}"]
    writes = []
    if category not in {"planning", "perception", "verification"}:
        writes = ["robot", "scene", "object:{part}"]
    if name == "observe_parts":
        writes = ["observations"]
    if name in {"estimate_pose", "inspect_receiver_relation"}:
        reads.append("observations")
    if name in {"close_gripper", "open_gripper", "place_object"}:
        writes.append("holding")
    return tuple(reads), tuple(writes)


def validate_ports(spec, params):
    """Cheap shared interface check; geometric obligations remain in contracts."""
    declared = {p.name: p for p in spec.ports}
    if set(params) - set(declared):
        raise ValueError("parameters outside declared execution interface")
    for name, value in params.items():
        p = declared[name]
        if p.kind == "object_refs" and value is not None:
            if (not isinstance(value, (tuple, list)) or not value
                    or any(not isinstance(item, str) for item in value)):
                raise ValueError(f"expected nonempty object list for {name}")
        if p.kind == "scalar" and value is not None:
            # NumPy numeric scalars are accepted; strings cannot become controls.
            if isinstance(value, str) or not math.isfinite(float(value)):
                raise ValueError(f"nonfinite/non-numeric scalar port {name}")
        if name == "max_steps" and value is not None:
            if isinstance(value, bool) or int(value) != value or not 1 <= int(value) <= 1000:
                raise ValueError("max_steps must be an integer in [1,1000]")
