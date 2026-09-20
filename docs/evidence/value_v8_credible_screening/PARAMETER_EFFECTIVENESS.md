# Parameter path, current scope

`stage_v7.build_pool` is a scripted random sampler, not an LLM planner. It samples legal assembly order, per-part yaw, grasp height, clearance, force and speed, plus wipe variant, contact force and duration. `full_task_v7.execute_full_task` expands these into cleaning and assembly atoms; carriage speed is deliberately removed because its atom uses a fixed constrained speed. Pin speed remains in the assembly calls. The current graph contains one `run_full_task_v7` call, so graph encoding does not expose internal atomic dependencies before execution.

`stroke_minimum` is an 80 mm task requirement and is fixed across the pool. The `wipe_force` field is an action target; `friction_scale` and `actuator_gain_scale` are environment trial variables, not candidate choices. Actual per-atom arguments are available in `steps.json` and optionally `executed_parameters`; requested/applied trial changes are in each outcome row.

This pass has not established that each sampled field changes the physical trajectory or that the sample is adapted to the observed layout. Those claims require paired trajectory comparisons, including a separate test for defaults that erase candidate differences.
