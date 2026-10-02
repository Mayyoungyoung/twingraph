"""Outcome-independent supply neighborhoods for V23 development collection."""
from copy import deepcopy
import math
import numpy as np

from simbench.assembly.printed_kit import planning_metadata
from simbench.assembly.scene import fmt
from . import supply_layout_v13 as broad

SCHEMA = "twingraph.spatial_supply_neighborhood.v23.r1"
PARENT_SEED = 2050
POSITION_RADIUS_M = .012
YAW_RADIUS_RAD = math.radians(8.)


def sample(seed, *, parent_seed=PARENT_SEED, position_radius_m=POSITION_RADIUS_M,
           yaw_radius_rad=YAW_RADIUS_RAD, minimum_gap_m=.010, cad=None):
    """Perturb one declared source layout, checking only source CAD packing."""
    cad = cad or planning_metadata()
    parent = broad.sample(parent_seed, cad=cad, level="L0", minimum_gap_m=minimum_gap_m)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(parent_seed), 23051]))
    result, boxes, attempts = {}, {}, {}
    for part in parent["parts"]:
        original = parent["parts"][part]
        region = np.asarray(broad.REGIONS.get(part, broad.REGIONS["mixed"]), float)
        body = part + "_holder" if part.startswith("pin_") else part
        bounds = cad["parts"][body]["body_bounds_m"]
        for attempt in range(1, 2001):
            xy = np.asarray(original["xy_m"]) + rng.uniform(-position_radius_m, position_radius_m, 2)
            yaw = float(original["yaw_rad"] + rng.uniform(-yaw_radius_rad, yaw_radius_rad))
            box = broad.footprint(bounds, xy, yaw)
            if np.any(xy < region[0]) or np.any(xy > region[1]):
                continue
            if np.any(box[0] < [-.55, -.50]) or np.any(box[1] > [.20, .015]):
                continue
            if not all(broad.separated(box, old, minimum_gap_m) for old in boxes.values()):
                continue
            result[part] = dict(xy_m=xy.tolist(), yaw_rad=yaw,
                                footprint_m=box.tolist(), footprint_cad_part=body)
            boxes[part] = box
            attempts[part] = attempt
            break
        else:
            raise ValueError(f"neighborhood CAD packing exhausted for {part}")
    if result["pin_left"]["xy_m"][1] <= result["pin_right"]["xy_m"][1]:
        raise ValueError("neighborhood pin naming order changed")
    return dict(schema=SCHEMA, seed=int(seed), level="L0", parent_seed=int(parent_seed),
                split_group=f"neighborhood_parent_{parent_seed}",
                position_radius_m=float(position_radius_m), yaw_radius_rad=float(yaw_radius_rad),
                minimum_source_gap_m=float(minimum_gap_m), parts=result,
                source_regions_m=deepcopy(broad.REGIONS), packing_attempts=attempts,
                outcome_filtering=False, perception_filtering=False,
                distinction="deterministic CAD-separated perturbation of one declared parent supply layout")


def apply(root, source_spec, *, parent_seed=PARENT_SEED):
    layout = sample(source_spec.seed, parent_seed=parent_seed)
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
