"""Seeded CAD-only supply layouts, independent of eventual task outcomes.

Different printed parts share a supply region, so identities are not encoded
by permanent slots. Identical pin tracks keep a declared observed-Y naming
convention, and each pin moves together with its passive supply holder.
"""
from copy import deepcopy
import math
import numpy as np

from simbench.assembly.printed_kit import planning_metadata
from simbench.assembly.scene import fmt

SCHEMA = "twingraph.spatial_supply_layout.r7.v1"
PARTS = ("carriage", "end_stop", "handle", "pin_left", "pin_right", "wipe_tool")
REGIONS = {
    "mixed": [[-.285, -.345], [.105, -.135]],
    "pin_left": [[-.425, -.265], [-.320, -.160]],
    "pin_right": [[-.425, -.405], [-.310, -.305]],
    "wipe_tool": [[-.285, -.105], [-.175, -.035]],
}


def footprint(bounds, xy, yaw):
    """Conservative world XY AABB of a rotated source-CAD body."""
    bounds = np.asarray(bounds, float)
    corners = np.array([[x, y] for x in bounds[:, 0] for y in bounds[:, 1]])
    c, s = math.cos(yaw), math.sin(yaw)
    points = corners @ np.array([[c, s], [-s, c]]) + np.asarray(xy)
    return np.array([points.min(axis=0), points.max(axis=0)])


def separated(first, second, margin):
    return bool(np.any(first[1] + margin <= second[0]) or
                np.any(second[1] + margin <= first[0]))


def sample(seed, *, cad=None, level="L1", minimum_gap_m=.010):
    if level not in ("L0", "L1", "L2"):
        raise ValueError("unknown layout level")
    if not 0 <= minimum_gap_m <= .03:
        raise ValueError("invalid source footprint gap")
    cad = cad or planning_metadata()
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 13071]))
    # This is a broad planar supply distribution, not pose jitter around a
    # successful answer. No perception, rollout or success label is queried.
    yaw_limit = math.radians({"L0": 20., "L1": 60., "L2": 90.}[level])
    result, boxes, attempts = {}, {}, {}
    order = ["pin_left", "pin_right", "wipe_tool", *rng.permutation(PARTS[:3])]
    for part in order:
        region = np.asarray(REGIONS.get(part, REGIONS["mixed"]), float)
        bound_part = part + "_holder" if part.startswith("pin_") else part
        bounds = cad["parts"][bound_part]["body_bounds_m"]
        for attempt in range(1, 2001):
            xy = rng.uniform(region[0], region[1])
            yaw = float(rng.uniform(-yaw_limit, yaw_limit))
            box = footprint(bounds, xy, yaw)
            # Table boundary and initial source separation are cheap scene
            # construction constraints, not a claim of robot reachability.
            if np.any(box[0] < [-.55, -.50]) or np.any(box[1] > [.20, .015]):
                continue
            if not all(separated(box, old, minimum_gap_m) for old in boxes.values()):
                continue
            result[part] = dict(xy_m=xy.tolist(), yaw_rad=yaw,
                footprint_m=box.tolist(), footprint_cad_part=bound_part)
            boxes[part] = box; attempts[part] = attempt
            break
        else:
            raise ValueError(f"CAD supply packing exhausted for {part}; no fallback or success-based rejection")
    if result["pin_left"]["xy_m"][1] <= result["pin_right"]["xy_m"][1]:
        raise AssertionError("anonymous pin track Y convention violated")
    return dict(schema=SCHEMA, seed=int(seed), level=level, parts=result,
        source_regions_m=deepcopy(REGIONS), yaw_range_rad=[-yaw_limit, yaw_limit],
        minimum_source_gap_m=minimum_gap_m, packing_attempts=attempts,
        outcome_filtering=False, perception_filtering=False,
        distinction="CAD-separated initial sources; visibility, reachability and execution remain to be checked")


def apply(root, source_spec):
    """stage_v12 scene_layout_hook, after original printed CAD installation."""
    layout = sample(source_spec.seed, level=source_spec.level)
    world = root.find("worldbody")
    for part, placement in layout["parts"].items():
        names = (part, part + "_holder") if part.startswith("pin_") else (part,)
        for name in names:
            body = world.find(f"body[@name='{name}']")
            if body is None:
                raise ValueError(f"source scene lacks {name}")
            pos = np.fromstring(body.get("pos", "0 0 0"), sep=" ")
            pos[:2] = placement["xy_m"]
            yaw = placement["yaw_rad"]
            body.set("pos", fmt(pos))
            body.set("quat", fmt([math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)]))
    return layout
