# Physical and observation boundary

The revised pin predicate checks shaft volume against a conservative inscribed circle of the original square opening and the annular guide. It checks both entry and required-depth endpoints of a straight pin; convexity bounds the intervening radial offset. This is a geometric acceptance check, not a proof that MuJoCo contact generated the insertion or retention. Release and post-stroke checks remain distinct.

`Session.call` uses simulated contact identity for grasp preconditions. In `library.py`, `stream_part`, `align_axis`, `press_seat`, `inspect_seat`, and related atoms read `ctx.obj_pos`, `ctx.obj_axis`, or `ctx.grasp_contacts` during execution. Those are ideal simulator feedback (category C) unless replaced with an explicit estimator. Rendering, reset, checkpoint restoration and independent final evaluation may use truth (category A). Robot joint state and actuator signals are category B. Existing RGB-D detection does not eliminate the ideal feedback in these atoms.

No independent physical causal tests (open-gripper carry, hidden weld, support removal, timestep sensitivity) or real-robot validation have been completed under V8. Results must be described as MuJoCo diagnostic executions with ideal feedback.
