#!/usr/bin/env python3
"""M2 acceptance: skill-library unit check (headless, Task A scene).

Checks:
  1. detect() with noise -- reported error stays within ~4 sigma
  2. estimate_grasp_pose() -- sensible target from scene metadata
  3. grasp -> place the 30 mm test cube (nominal, zero noise)
  4. push the cube 5 cm in +x
  5. grasp -> place the cube back to its start
Prints control steps/s (target >> 100).
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))  # repo root

import numpy as np  # noqa: E402

from simbench.core.controller import CartesianController, Gripper  # noqa: E402
from simbench.core.sim_context import MjContext  # noqa: E402
from simbench.skills import manipulation, perception  # noqa: E402

SCENE = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scenes", "taskA_gearbox.xml")

CUBE_START = np.array([-0.02, 0.24])
CUBE_PLACE = np.array([-0.12, 0.24])
CUBE_Z = 0.815


def main():
    np.random.seed(0)
    ctx = MjContext(SCENE)
    ctx.reset()
    arm = CartesianController(ctx)
    gripper = Gripper(ctx)

    results = {}

    # -- 1. detect with noise ----------------------------------------
    true_pos = ctx.obj_pos("gear").copy()
    errs = []
    for _ in range(5):
        p, yaw = perception.detect(ctx, "gear", noise_std=0.003)
        errs.append(np.linalg.norm(p - true_pos))
    ok = max(errs) < 0.012
    results["detect_noise"] = ok
    print(f"detect(gear, noise 3mm): max err {max(errs)*1000:.1f} mm "
          f"-> {'ok' if ok else 'FAIL'}")

    # -- 2. estimate_grasp_pose --------------------------------------
    gp = perception.estimate_grasp_pose(ctx, "gear")
    # the grasp target sits grasp_dz (0.002) ABOVE the part centre
    ok = (abs(gp["pos"][2] - (true_pos[2] + 0.002)) < 0.004
          and abs(gp["outer_d"] - 0.020) < 1e-6)
    results["estimate_grasp_pose"] = ok
    print(f"estimate_grasp_pose(gear): pos {np.round(gp['pos'], 4)} "
          f"type {gp['type']} outer_d {gp['outer_d']*1000:.0f} mm "
          f"-> {'ok' if ok else 'FAIL'}")

    # -- 3. grasp -> place cube --------------------------------------
    n0 = ctx.n_control_steps
    t0 = time.time()
    ok = (manipulation.grasp(ctx, arm, gripper, "test_cube", verbose=True)
          and manipulation.place(ctx, arm, gripper, "test_cube",
                                 CUBE_PLACE, target_z=CUBE_Z,
                                 verbose=True))
    results["grasp_place"] = ok
    print(f"grasp->place cube: {'ok' if ok else 'FAIL'} "
          f"-> {np.round(ctx.obj_pos('test_cube'), 4)}")

    # -- 4. push cube 5 cm -------------------------------------------
    ok = manipulation.push(ctx, arm, gripper, "test_cube", (0.05, 0.0),
                            verbose=True)
    results["push"] = ok

    # -- 5. grasp -> place cube back ---------------------------------
    ok = (manipulation.grasp(ctx, arm, gripper, "test_cube", verbose=True)
          and manipulation.place(ctx, arm, gripper, "test_cube",
                                 CUBE_START, target_z=CUBE_Z,
                                 verbose=True))
    results["grasp_place_back"] = ok

    # -- summary ------------------------------------------------------
    wall = time.time() - t0
    steps = ctx.n_control_steps - n0
    print(f"\nsteps={steps} wall={wall:.1f}s "
          f"({steps / max(wall, 1e-9):.0f} control steps/s)")
    all_ok = all(results.values())
    for k, v in results.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    print(f"\nSKILLS CHECK: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
