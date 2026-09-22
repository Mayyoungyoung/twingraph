"""Observation-conditioned proposals with an auditable parameter derivation.

The language-level alternatives are authored for this task. No online LLM API,
material identification from RGB, or feasibility oracle is implied. CAD and
actuator limits are declared priors; measured position/uncertainty conditions
the executable parameters. All outcomes remain unknown until rollout.
"""
from copy import deepcopy
import math
import numpy as np

from . import planner_v11
from .plan import digest

PARTS = ("carriage", "end_stop", "pin_left", "pin_right", "handle")
SOURCE = "codex_symbolic_plan_with_observed_geometry_catalog_v13"

# Calibrated controller envelopes are not estimates of an unknown material.
# Keep these explicit instead of hiding constants in candidate templates.
LIMITS = dict(clearance_m=(.94, 1.08), grasp_force_n=(2.5, 9.),
              insertion_speed_m_s=(.002, .008), slide_speed_m_s=(.008, .025),
              contact_force_n=(4., 18.), press_force_n=(1., 4.), wipe_duration_s=(10., 24.))
DEFAULT_PRIORS = dict(density_kg_m3=1240., effective_solid_fraction=.6,
                     friction_lower_bound=.35, acceleration_m_s2=1.,
                     gripper_pad_friction_lower_bound=.8, contact_load_safety_factor=1.2,
                     effective_contact_stiffness_n_m=1500.,
                     holding_safety_factor=2.5,
                     source="declared PLA/contact prior; not inferred from RGB-D")


def _halton(index, base):
    result, scale = 0., 1.
    while index:
        scale /= base
        index, remainder = divmod(index, base)
        result += scale * remainder
    return result


def _bounded_quantile(bounds, nominal, quantile):
    """Cover a declared uncertainty envelope around a state-derived median.

    This is a proposal distribution, not a measured material posterior. Half
    its mass is below the nominal value and half above; no outcome is read.
    """
    low, high = map(float, bounds)
    middle = float(np.clip(nominal, low, high))
    q = float(quantile)
    return low + 2*q*(middle-low) if q <= .5 else middle + (2*q-1)*(high-middle)


def _part_geometry(cad, part):
    defaults = dict(carriage=[.054, .046, .068], end_stop=[.024, .086, .049],
                    pin_left=[.018, .018, .060], pin_right=[.018, .018, .060],
                    handle=[.042, .042, .016])
    row = cad.get("parts", {}).get(part, {})
    dimensions = np.asarray(row.get("dimensions_m", row.get("size_m", defaults[part])), float)
    if dimensions.shape != (3,) or not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
        raise ValueError(f"invalid declared CAD dimensions: {part}")
    # Missing solid volume is a conservative bounding-box estimate, identified
    # in the ledger; it never reads a simulator body's mass.
    volume = float(row.get("solid_volume_m3", np.prod(dimensions)))
    return dimensions, volume, "stl_solid_volume" if "solid_volume_m3" in row else "cad_bounding_box_upper_bound"


def _geometry_catalogs(observation, cad, completed):
    """Generate executable geometry alternatives before reading any outcome.

    The construction set is a task prior. Its accepted members, grasp world
    frames and future receiver frames depend on this particular observation.
    A necessary CAD condition is never labelled as physical task success.
    """
    from simbench.assembly.gripper_clearance_v12 import pin_clearance_catalog
    from simbench.assembly.placement_catalog_v13 import placement_clearance_catalog
    required = ("parts", "collision_primitives", "pin_shaft_offsets_m")
    if any(key not in cad for key in required):
        raise ValueError("declared printed-kit CAD required for candidate geometry")
    if not observation.get("fixtures", {}).get("guide_base", {}).get("valid"):
        raise ValueError("reobserve required: guide_base")
    if set(observation.get("assembly_targets", {})) != set(PARTS):
        raise ValueError("observed receiver/CAD assembly targets required")
    all_rows, usable = {}, {}
    for part in PARTS:
        if part in completed:
            # The monitor restores the executed prefix verbatim. Do not
            # pretend that a now occluded completed part was re-planned.
            continue
        if part.startswith("pin_"):
            rows = pin_clearance_catalog(observation, cad, part,
                goal_mode="observed_installed" if "end_stop" in completed else "predicted_mated")
        else:
            rows = placement_clearance_catalog(observation, cad, part, completed=completed)
        all_rows[part] = rows
        feasible = [r for r in rows if r["status"] != "rejected"
                    and all(r.get(k) is not None for k in ("yaw", "height", "placement_yaw"))]
        if not feasible and not part.startswith("pin_"):
            # Native/conservative gripper hulls can report intended source-part
            # contacts as penetrations.  Such rows are never relabelled as a
            # pass: retain them as explicit DT-only unknowns only when every
            # checked environment/receiver path is nonpenetrating and the
            # declared grasp face is valid.  The expanded centre constructor
            # can therefore be tested physically without hiding its risk.
            deferred=[]
            for original in rows:
                env=original.get("environment_min_clearance_m")
                source=original.get("source_body_min_clearance_m")
                if (all(original.get(k) is not None for k in ("yaw","height","placement_yaw"))
                        and env is not None and env >= 0 and source is not None and source < 0
                        and not original.get("source_body_invalid_grasp_face_contact",False)):
                    row=deepcopy(original)
                    row.update(status="unknown",original_status="rejected",
                        deferred_source_target_contact=True,
                        reason=("conservative source target/gripper contact deferred to physical twin; "
                                "all sampled environment/receiver geometry is nonpenetrating"),
                        necessary_geometry_pass=False)
                    deferred.append(row)
            feasible=deferred
        if not feasible:
            from .stage_v5 import CandidateGenerationError
            raise CandidateGenerationError(f"no executable necessary-geometry choices for {part}; "
                f"reobserve/reposition or expand the grasp constructor; {len(rows)} rows examined")
        def conservative_key(row):
            margin = row.get("min_clearance_m")
            reserve = max(float(row.get("required_clearance_m") or .003), 1e-6)
            # Extra clearance beyond the uncertainty reserve cannot
            # compensate for poor pad overlap and resulting slippage.
            reserve_fraction = min(1., float(margin)/reserve) if margin is not None else -1.
            overlap = row.get("pad_face_axial_overlap_m", row.get("head_pad_axial_overlap_m", 0.)) or 0.
            rotation = abs(math.atan2(math.sin(row["placement_yaw"]-row["yaw"]),
                                      math.cos(row["placement_yaw"]-row["yaw"])))
            return (row["status"] != "necessary_pass", -reserve_fraction, -float(overlap),
                    abs(row["height"]), rotation, abs(row["yaw"]), row["placement_yaw"])
        usable[part] = sorted(feasible, key=conservative_key)
    return usable, all_rows


def propose(observation, *, cad=None, n=48, seed=0, completed=(), priors=None):
    if not isinstance(n, int) or not 1 <= n <= 512:
        raise ValueError("candidate budget must be an integer in [1, 512]")
    cad = deepcopy(cad or {})
    physical = {**DEFAULT_PRIORS, **(priors or {})}
    if not (0 < physical["friction_lower_bound"] <= 2 and 0 < physical["effective_solid_fraction"] <= 1):
        raise ValueError("invalid explicit material prior")
    if not (0 < physical["gripper_pad_friction_lower_bound"] <= 2
            and physical["contact_load_safety_factor"] >= 1):
        raise ValueError("invalid explicit grasp contact-load prior")
    objects = observation["objects"]
    pending = [p for p in (*PARTS, "wipe_tool") if p not in completed and not (p == "wipe_tool" and "cleaning" in completed)]
    for part in pending:
        row = objects.get(part, {})
        if not row.get("valid") or row.get("position_m") is None:
            raise ValueError(f"reobserve required: {part}")
        if not np.isfinite(row["position_m"]).all():
            raise ValueError(f"nonfinite observation: {part}")
    catalogs, catalog_audit = _geometry_catalogs(observation, cad, set(completed))
    valid = {p: r for p, r in objects.items() if r.get("valid") and r.get("position_m") is not None}
    highest = max(float(r["position_m"][2]) for r in valid.values())
    uncertainty = {p: max(.0005, float(objects.get(p, {}).get("fit_residual_m") or .003)) for p in PARTS}
    origin = np.asarray(valid.get("end_stop", next(iter(valid.values())))["position_m"])
    pins = sorted(("pin_left", "pin_right"), key=lambda p: float(np.linalg.norm(np.asarray(valid[p]["position_m"])[:2]-origin[:2])) if p in valid else float("inf"))
    base_order = ["carriage", "end_stop", *pins, "handle"]
    baselines, derivation = {}, {}
    for part in PARTS:
        dims, volume, volume_source = _part_geometry(cad, part)
        mass = volume * physical["density_kg_m3"] * physical["effective_solid_fraction"]
        force_balance = physical["holding_safety_factor"] * mass * (9.81 + physical["acceleration_m_s2"]) / (2 * physical["friction_lower_bound"])
        grip = float(np.clip(force_balance, *LIMITS["grasp_force_n"]))
        # Supply holders produce extra extraction resistance. This allowance
        # is an explicit controller prior, not an invented friction measurement.
        extraction = float(cad.get("pin_extraction_allowance_n", 4.)) if part.startswith("pin_") else 0.
        grip = float(np.clip(grip + extraction, *LIMITS["grasp_force_n"]))
        clearance = float(np.clip(highest + .10 + .5 * dims[2] + 3 * uncertainty[part], *LIMITS["clearance_m"]))
        speed_limit = LIMITS["slide_speed_m_s"] if part == "carriage" else LIMITS["insertion_speed_m_s"]
        nominal_speed = speed_limit[1] / (1 + uncertainty[part] / .004)
        speed = float(np.clip(nominal_speed, *speed_limit))
        # Gravity alone underestimates contact transients at a constrained
        # entrance. Keep uncertainty-induced loading as an explicit stiffness
        # prior, and explore nearby limits inside the calibrated safe envelope.
        contact = float(np.clip(4 + 2 * mass * 9.81 / physical["friction_lower_bound"]
            + 3 * uncertainty[part] * physical["effective_contact_stiffness_n_m"], *LIMITS["contact_force_n"]))
        baselines[part] = dict(yaw=0., height=0., clearance=clearance, force=grip,
                               speed=speed, force_limit=contact,
                               press_force=float(np.clip(mass * 9.81 + 1., 1., 4.)))
        if part not in ("carriage", "pin_left", "pin_right"):
            baselines[part].pop("force_limit")
        derivation[part] = dict(dimensions_m=dims.tolist(), volume_m3=volume,
            volume_source=volume_source, assumed_mass_kg=mass, fit_residual_m=uncertainty[part],
            force_balance_n=force_balance, extraction_allowance_n=extraction,
            clearance_rule="observed maximum height + 0.10 + half part height + 3 sigma",
            speed_rule="skill max speed / (1 + sigma / 0.004), clipped to skill envelope",
            baseline=deepcopy(baselines[part]))
    wipe_extent = np.asarray(cad.get("wipe_halfspan_m", [.055, .005]), float)
    wipe_length = 6 * wipe_extent[0] + 4 * wipe_extent[1]
    wipe_sigma = float(objects.get("wipe_tool", {}).get("fit_residual_m") or .003)
    wipe_speed = .03 / (1 + wipe_sigma / .006)
    wipe_duration = float(np.clip(wipe_length / wipe_speed, *LIMITS["wipe_duration_s"]))
    pool, unique = [], set()
    for i in range(n):
        # Nested low-discrepancy continuous proposals and balanced semantic
        # branches. Unlike twelve one-variable variants, combinations change
        # grasp/approach at the early carriage stage as well as later pins.
        h = [_halton(i + 1, b) for b in (2, 3, 5, 7, 11, 13, 17)]
        choices = deepcopy(baselines)
        selected_geometry = {}
        for j, part in enumerate(PARTS):
            c = choices[part]
            if part in catalogs:
                rows = catalogs[part]
                # Include the conservative constructor then spread over the
                # scene-dependent catalogue, keeping N=24 nested in N=48.
                idx = 0 if i == 0 else min(len(rows)-1, int(_halton(i, (2,3,5,7,11)[j])*len(rows)))
                selected = deepcopy(rows[idx])
                if not part.startswith("pin_"):
                    selected["evaluated_world_yaw_rad"] = selected["yaw"]
                    selected["yaw"] = float(selected["source_axis_offset_rad"])
                    selected["grasp_yaw_frame"] = "object"
                selected_geometry[part] = selected
                for key in ("yaw", "height", "placement_yaw", "pin_command_depth_m", "pin_press_extra_m"):
                    if key in selected:
                        c[key] = float(selected[key])
                if selected.get("grasp_width_m") is not None:
                    c["width"] = float(selected["grasp_width_m"])
                if selected.get("grasp_center_offset_body_m") is not None:
                    c["center_offset"] = [float(v) for v in selected["grasp_center_offset_body_m"]]
                if "grasp_yaw_frame" in selected:
                    c["grasp_yaw_frame"] = selected["grasp_yaw_frame"]
            else:
                selected_geometry[part] = dict(part=part, status="unknown", known=False,
                    reason="completed action retained by closed-loop monitor", min_clearance_m=None,
                    source_grasp_ik_checked=False)
            c["clearance"] = float(np.clip(c["clearance"] + (h[1] - .5) * .04, *LIMITS["clearance_m"]))
            # The mass/friction balance is only a lower-bound prior: the
            # actuator closes compliant pads, it is not an ideal force source.
            # Cover that uncertainty within the skill envelope instead of
            # clipping nearly every light printed part to the same 2.5 N.
            c["force"] = float(np.clip(c["force"] * (.8 + 2.2 * h[(j+2) % len(h)]), *LIMITS["grasp_force_n"]))
            envelope = LIMITS["slide_speed_m_s"] if part == "carriage" else LIMITS["insertion_speed_m_s"]
            # A narrow +/- perturbation misses the low-speed contact regime.
            # Explore the already declared skill envelope, with the measured
            # uncertainty still conditioning its median.
            c["speed"] = _bounded_quantile(envelope, baselines[part]["speed"], h[3])
            if "force_limit" in c:
                c["force_limit"] = _bounded_quantile(LIMITS["contact_force_n"],
                    baselines[part]["force_limit"], h[4])
                # A larger contact budget also needs enough frictional grip.
                # Saturation is disclosed, not treated as guaranteed holding.
                load_grip = physical["contact_load_safety_factor"] * c["force_limit"] / (
                    2 * physical["gripper_pad_friction_lower_bound"])
                c["force"] = float(np.clip(max(c["force"], load_grip), *LIMITS["grasp_force_n"]))
            c["press_force"] = _bounded_quantile(LIMITS["press_force_n"],
                baselines[part]["press_force"], h[(j+5) % len(h)])
            if (i // 4 + j) % 2:
                c["approach_strategy"] = "joint_checked_v10"
            if part.startswith("pin_"):
                # Withdraw clear of the supply cylinder before changing the
                # EEF orientation for the receiver. This is an executed port.
                c["lift_first_m"] = .08
        if i == 0:
            # A strategy family should include the CAD's nominal grasp, not
            # perturb every part in every proposal. This explicit conservative
            # family is generated before twin outcomes like all other plans.
            for part, c in choices.items():
                c["force"] = float(np.clip(2.4 * baselines[part]["force"], *LIMITS["grasp_force_n"]))
                if "approach_strategy" in c: c.pop("approach_strategy")
                if "force_limit" in c: c["force_limit"] = baselines[part]["force_limit"]
                c["press_force"] = baselines[part]["press_force"]
                envelope = LIMITS["slide_speed_m_s"] if part == "carriage" else LIMITS["insertion_speed_m_s"]
                c["speed"] = float(np.clip(.8 * baselines[part]["speed"], *envelope))
        order = list(base_order)
        if (i // 2) % 2: order[2:4] = reversed(order[2:4])
        proposal = dict(name=f"grounded_{i:03d}", source=SOURCE,
            rationale=("observed-geometry conservative grasp with uncertainty clearance and pad overlap" if i == 0
                       else "observed-scene geometry alternatives and uncertainty-conditioned contact parameters"),
            order=order, choices=choices, wipe_variant=i % 4,
            wipe_force=float(np.clip(1.5 * (.85 + .3 * h[5]), .8, 2.2)),
            wipe_duration=float(np.clip(wipe_duration * (.9 + .2 * h[6]), *LIMITS["wipe_duration_s"])),
            stroke_minimum=float(cad.get("functional_stroke_minimum_m", .02)),
            necessary_geometry=selected_geometry)
        signature = digest({k:v for k,v in proposal.items() if k not in ("name", "source", "rationale")})
        if signature in unique: raise ValueError("duplicate executable proposal")
        unique.add(signature); pool.append(proposal)
    # Pool membership for a smaller budget is nested; display/order cannot
    # encode a preferred answer or use any outcome label.
    order = np.random.default_rng(np.random.SeedSequence([int(seed), 1211])).permutation(n)
    pool = [pool[i] for i in order]
    return pool, dict(source=SOURCE, online_llm_call=False, requested=n,
        unique_executable_parameters=len(unique), observation_sha256=observation.get("sha256"),
        cad_sha256=digest(cad), priors=physical, controller_envelopes=LIMITS,
        geometry_catalogs=catalog_audit,
        geometry_counts={p:dict(generated=len(catalog_audit[p]), admitted=len(rows),
            necessary_pass=sum(r["status"]=="necessary_pass" for r in rows),
            unknown=sum(r["status"]=="unknown" for r in rows)) for p,rows in catalogs.items()},
        derivation=derivation, wipe_path_length_m=float(wipe_length),
        placement_yaw_rule="observed source principal axes and observed receiver/CAD goal axes; pins have independent grasp and insertion yaw",
        grasp_force_search_rule="0.8 to 3.0 times mass/friction/extraction prior, clipped to skill force envelope; compliant-pad uncertainty",
        contact_search_rule="piecewise quantiles across declared force/speed/press envelopes, with state-derived medians; force limits are stopping budgets, not predicted required force",
        contact_grip_rule="at least contact-load safety factor * stopping budget / (2 * assumed pad friction), clipped to actuator envelope; saturation is not a holding guarantee",
        pin_yaw_search_rule="round CAD grasp quadrants and receiver-relative insertion quadrants; full-shell insertion/opening/retreat collision rows rejected before selection",
        wipe_duration_rule="normalized raster length / uncertainty-adjusted wipe speed",
        distinction="proposal diversity is not a claim of physically feasible coverage")


def normalized_graph(session, proposal, *, completed=()):
    graph = planner_v11.normalized_graph(session, proposal, completed=completed)
    graph["schema"] = "twingraph.full_task_graph.v12"
    graph["planning_cad"] = deepcopy(getattr(session, "planning_cad", {}))
    graph["task_geometry_version"] = getattr(session, "task_version", "unknown")
    graph["acceptance_mode"] = "functional_v12" if getattr(session, "functional_acceptance_v12", False) else "legacy"
    graph["assembly"]["observation"]["goals"][0]["stroke_minimum_m"] = float(proposal["stroke_minimum"])
    graph["assembly"]["observation"]["goals"][0].update(
        fixture_capture_required=True, pin_base_bridge_required=True)
    if getattr(session, "pin_insertion_config", None):
        graph["assembly"]["observation"]["goals"][0]["pin_minimum_depth_m"] = session.pin_insertion_config.required_depth_m
    if "necessary_geometry" in proposal:
        graph["geometry_conditions"] = dict(schema="twingraph.geometry_conditions.v13",
            plan_sha256=graph["assembly"]["plan_sha256"],
            choices_sha256=digest(proposal["choices"]), order_sha256=digest(proposal["order"]),
            observation_sha256=graph["assembly"]["observation"]["perception"]["observation_sha256"],
            cad_sha256=digest(graph["planning_cad"]), parts=deepcopy(proposal["necessary_geometry"]))
        from .skill_graph import validate_geometry_conditions
        validate_geometry_conditions(graph)
    return graph


assembly_program = planner_v11.assembly_program
