"""Behavior-cloned insertion using visual grasp registration and robot sensing.

This policy is trained from a procedural Cartesian expert. Synthetic sensor
demonstrations and held-out action errors are reported honestly; these are not
MuJoCo success trials. Physical deployment uses the actual arm/contact model.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

SCHEMA = "twingraph.sensor_insertion_bc.v12"
CENTERS = np.linspace(-1., 1., 25)


def features(observation):
    x = np.atleast_2d(np.asarray(observation, float))
    # Per-axis smooth bases retain near-contact resolution while bounding
    # large approach errors; force interaction learns compliant retreat.
    err = np.tanh(x[:, :3])
    rbf = np.exp(-.5 * ((err[:, :, None] - CENTERS) / .10) ** 2).reshape(len(x), -1)
    force = x[:, 6:7]
    return np.c_[np.ones(len(x)), err, rbf, force, force * rbf[:, -25:], x[:, 3:6]]


def expert_actions(x):
    x = np.asarray(x)
    action = np.clip(x[:, :3] * 4.0 - .08 * x[:, 3:6], -1., 1.)
    # Force is normalized by the candidate's physically declared limit.
    action[:, 2] = np.where(x[:, 6] > .55, np.maximum(action[:, 2], .15), action[:, 2])
    return action


def sensor_demonstrations(seed, episodes, extrapolate=False):
    rng = np.random.default_rng(seed)
    states, actions, groups, task_rows = [], [], [], []
    for episode in range(episodes):
        speed = rng.uniform(.002, .009 if extrapolate else .007)
        radius = rng.uniform(.008, .020) if extrapolate else rng.uniform(.002, .015)
        # These are sensor-state trajectories, never hidden simulator poses.
        error = np.r_[rng.uniform(-radius, radius, 2), -rng.uniform(.020, .085)]
        velocity = np.zeros(3)
        force = 0.
        for k in range(450):
            obs = np.r_[error / .003, velocity / .01, force]
            action = expert_actions(obs[None])[0]
            states.append(obs.copy()); actions.append(action.copy()); groups.append(episode)
            # Kinematic demonstrator, declared separately from physical tests.
            delta = action * speed * .02
            error -= delta + rng.normal(0, .000006, 3)
            velocity = delta / .02
            force = float(rng.uniform(.6, .95)) if (k % 47 == 0) else 0.
        task_rows.append(dict(episode=episode, speed_m_s=float(speed), lateral_extent_m=float(radius)))
    return np.asarray(states), np.asarray(actions), np.asarray(groups), task_rows


def train(path, episodes=72):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    x, y, group, tasks = sensor_demonstrations(1221, episodes)
    train_rows = group % 6 != 0
    phi = features(x)
    # Fit on whole training episodes; held-out episodes never enter the solve.
    weights = np.linalg.solve(phi[train_rows].T @ phi[train_rows] + .002 * np.eye(phi.shape[1]),
                              phi[train_rows].T @ y[train_rows])
    np.savez_compressed(path, weights=weights, schema=SCHEMA, centers=CENTERS)
    np.savez_compressed(path.with_name("insert_sensor_demonstrations_v12.npz"),
                        observations=x, actions=y, episode_ids=group)
    test_x, test_y, _, test_tasks = sensor_demonstrations(91221, 18, extrapolate=True)
    report = dict(schema=SCHEMA, algorithm="behavior_cloning_regularized_rbf_regression",
        teacher="procedural bounded Cartesian force-aware expert",
        demonstration_environment="synthetic sensor-state kinematics, not MuJoCo contact rollouts",
        input="RGB-D grasp registration + encoder FK error/velocity + force/declared limit",
        observation_dim=7, training_episodes=int(sum(i % 6 != 0 for i in range(episodes))),
        validation_episodes=int(sum(i % 6 == 0 for i in range(episodes))), samples=int(len(x)),
        validation_action_mse=float(np.mean((np.clip(phi[~train_rows] @ weights, -1, 1) - y[~train_rows]) ** 2)),
        heldout_action_mse=float(np.mean((np.clip(features(test_x) @ weights, -1, 1) - test_y) ** 2)),
        heldout_tasks=test_tasks, train_tasks=tasks,
        physical_generalization="Not established by this report; requires independent MuJoCo trials",
        sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    path.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load_actor(path):
    with np.load(path, allow_pickle=False) as blob:
        if str(blob["schema"]) != SCHEMA:
            raise ValueError("V12 sensor policy schema mismatch")
        weights = blob["weights"].copy()
    return lambda observation: np.clip((features(observation) @ weights)[0], -1., 1.)


def _execute_legacy(session, part, artifact, policy, max_steps):
    from .library import Result
    from .skills_v12 import control_position
    actor = load_actor(policy)
    plan = session.artifacts[artifact]
    target = np.asarray(plan["target"], float).copy()
    semantic_target = target.copy()
    config = session.pin_insertion_config
    speed, force_limit = float(plan["speed"]), float(plan["force_limit"])
    if not (.001 <= speed <= .02 and 0 < force_limit <= 30):
        return Result(False, reason="sensor insertion speed/force outside calibrated policy envelope")
    dt = session.ctx.control_dt
    previous = control_position(session, part)
    command = session.ctx.eef_pos().copy()
    peak, steps, reached = 0., 0, False
    trace = []
    row = (session.decision_observation or {}).get("objects", {}).get(part, {})
    visual_sigma = max(.0005, float(row.get("fit_residual_m") or 0.))
    # Pixel/depth quantization and grasp registration contribute separately;
    # cap the scan within the real receiver's CAD neighborhood.
    radius = min(.75 * config.plate_hole_half_width_m, 3 * (visual_sigma + .0005))
    search = None
    recovered = False
    contact_threshold = min(.35, .04 * force_limit)
    search_enabled = not getattr(session, "disable_insert_search_v12", False)
    search_metrics = dict(enabled=search_enabled, maximum_radius_m=radius,
                          visual_sigma_floor_m=visual_sigma, grasp_registration_sigma_m=.0005,
                          trigger_force_n=contact_threshold, iterations=0, entry_detected=False,
                          entry_minimum_encoder_descent_m=.004, entry_low_force_n=.10,
                          entry_continuous_low_force_steps=15)
    for steps in range(int(max_steps)):
        position = control_position(session, part)
        force = float(session.external_force(part))
        peak = max(peak, force)
        if force > force_limit:
            return Result(False, {"peak_force_n": peak, "policy": str(policy)}, "learned insertion force limit")
        if not session.ctx.grasp_contacts(part)["held"]:
            return Result(False, {"policy": str(policy), "peak_force_n": peak, "steps": steps,
                                  "search": search_metrics, "trace": trace}, "learned insertion grasp lost")
        error = target - position
        if search_enabled and search is None and not recovered and force >= contact_threshold:
            search = dict(start_z=float(position[2]), steps=0,
                          center_xy=target[:2].copy(), command_z=float(command[2]), low_force_steps=0)
            search_metrics["triggered"] = True
        if search is None and np.linalg.norm(error) <= max(.0005, speed * dt * 2):
            reached = True
            break
        obs = np.r_[error / .003, (position - previous) / dt / .01, force / force_limit]
        action = actor(obs) * speed * dt
        previous = position.copy()
        mode = "behavior_cloned_insertion"
        if search is not None:
            mode = "bounded_force_guided_spiral"
            search["steps"] += 1
            n = search["steps"]
            search_metrics["iterations"] = n
            search["low_force_steps"] = search["low_force_steps"] + 1 if force < .10 else 0
            # Entry is inferred from encoder descent and contact unloading,
            # never from the simulated pin/hole evaluator.
            if search["start_z"] - position[2] >= .004 and search["low_force_steps"] >= 15:
                target[:2] = position[:2]
                feed = float(plan["recovery_feed_depth_m"])
                if not 0 < feed <= float(plan["recovery_feed_upper_bound_m"]):
                    return Result(False, reason="recovery feed is outside its declared CAD bound")
                target[2] = float(position[2] - feed)
                search_metrics.update(entry_detected=True,
                    inferred_correction_m=(target[:2]-semantic_target[:2]).tolist(),
                    additional_feed_depth_m=feed, feed_reference_encoder_position_m=position.tolist(),
                    recovery_control_target_m=target.tolist(),
                    feed_source=plan["recovery_feed_source"],
                    semantic_graph_target_remains_m=semantic_target.tolist())
                search = None
                recovered = True
                command = session.ctx.eef_pos().copy()
            else:
                if n > 650:
                    break
                r = radius * min(n / 600., 1.)
                xy = search["center_xy"] + r * np.array([np.cos(.03*n), np.sin(.03*n)])
                # Convert the visually registered held-part center to an
                # end-effector command with the same FK adapter as the actor.
                command[:2] = session.ctx.eef_pos()[:2] + xy - position[:2]
                search["command_z"] += (-min(speed*dt, .00006) if force < .5 else .000015)
                command[2] = search["command_z"]
        else:
            # Integrate velocity commands: sub-IK-tolerance increments must
            # accumulate rather than vanish at each measured-state reset.
            command += action
        command = np.clip(command, session.ctx.eef_pos() - .003, session.ctx.eef_pos() + .003)
        session.arm.servo(command)
        if steps % 25 == 0:
            trace.append(dict(step=steps, observation=obs.tolist(), behavior_cloned_action_m=action.tolist(),
                              commanded_eef_position_m=command.tolist(), mode=mode))
    metrics = dict(policy=str(policy), policy_sha256=hashlib.sha256(Path(policy).read_bytes()).hexdigest(),
                   steps=steps + 1, peak_force_n=peak, visual_kinematic_target_reached=reached,
                   input_source="RGB-D grasp registration, encoder FK, simulated force/tactile sensor",
                   functional_acceptance="subsequent finite-bore insertion and release checks", trace=trace,
                   functional_insertion_control_target_m=target.tolist(),
                   semantic_graph_target_m=semantic_target.tolist(),
                   required_shaft_entry_depth_m=config.required_depth_m,
                   controller="hybrid behavioral cloning and explicit bounded force-guided search",
                   search=search_metrics)
    from simbench.value.pin_geometry import evaluate_pin_context
    geometry = evaluate_pin_context(session.ctx, part, fixture_part="end_stop",
        hole_offset_m=(0., -.032 if part == "pin_left" else .032, 0.),
        phase="inserted_while_held", released=False, touching_finger=True,
        config=session.pin_insertion_config)
    metrics["independent_insertion_evaluation"] = geometry
    metrics["evaluation_source"] = "simulator evaluator only; not an actor input or online correction"
    session.artifacts["learned_insertion_result"] = metrics
    return Result(bool(geometry["success"]), metrics,
                  "learned policy did not achieve physical shaft-in-bore engagement")


def pin_depth_contract(session, part, plan):
    """Explicit receiver-relative command envelope; no free-body truth."""
    if "command_depth_m" not in plan: return None
    axis=np.asarray(plan["axis"],float);entry=np.asarray(plan["hole_entry_m"],float)
    if axis.shape!=(3,) or entry.shape!=(3,) or not np.isfinite(np.r_[axis,entry]).all() or np.linalg.norm(axis)<1e-8:
        raise ValueError("invalid observed insertion axis/entry")
    axis=axis/np.linalg.norm(axis)
    depth,extra=float(plan["command_depth_m"]),float(plan["press_extra_m"])
    total,absolute=float(plan["maximum_total_depth_m"]),float(plan["absolute_depth_limit_m"])
    cad_limit=float(session.planning_cad["pin_head_seated_total_depth_m"])
    if (not np.isfinite([depth,extra,total,absolute,cad_limit]).all()
            or not (depth>0 and extra>=0 and abs(total-depth-extra)<1e-9 and total<=absolute+1e-9 and absolute<=cad_limit+1e-9)):
        raise ValueError("pin command/press depth exceeds declared CAD envelope")
    reference=np.array([1.,0.,0.]) if abs(axis[0])<.9 else np.array([0.,1.,0.])
    u=np.cross(axis,reference);u/=np.linalg.norm(u);v=np.cross(axis,u)
    tip_offset=float(session.pin_insertion_config.shaft_tip_offset_m)
    if not np.isfinite(tip_offset) or tip_offset>=0:
        raise ValueError("invalid CAD pin shaft tip offset")
    return dict(entry=entry,axis=axis,tangents=np.array([u,v]),command_depth=depth,
        press_extra=extra,total_depth=total,absolute_depth=absolute,
        tip_offset=tip_offset)


def sensor_pin_axis(session,part):
    from .skills_v12 import validated_held_registration
    registration=validated_held_registration(session,part)
    return session.ctx.eef_mat() @ np.asarray(registration["local_rotation"])[:,2]


def sensor_depth(session,part,position,contract):
    position=np.asarray(position,float)
    if position.shape!=(3,) or not np.isfinite(position).all():
        raise ValueError("invalid registered pin position")
    tip=position+sensor_pin_axis(session,part)*contract["tip_offset"]
    depth=float((tip-contract["entry"])@contract["axis"])
    if not np.isfinite(depth):raise ValueError("nonfinite registered pin depth")
    return depth


def _check_measured_depth_envelope(depth,contract):
    # A registered estimate already beyond the CAD head-seated bound is an
    # inconsistent control state, not an instruction to servo back by an
    # arbitrarily large correction. This is not a physical acceptance label.
    if depth>contract["absolute_depth"]+1.e-9:
        raise ValueError("registered pin depth already beyond absolute CAD command envelope")


def limit_pin_eef_command(session,part,command,contract,limit):
    from .skills_v12 import control_position
    command=np.asarray(command,float)
    if command.shape!=(3,) or not np.isfinite(command).all() or not np.isfinite(limit):
        raise ValueError("invalid pin actuator command or depth bound")
    predicted=control_position(session,part)+command-session.ctx.eef_pos()
    depth=sensor_depth(session,part,predicted,contract)
    excess=max(0.,depth-float(limit))
    return np.asarray(command)-contract["axis"]*excess,excess


def advance_pin_search_command(command,eef,position,goal,axis,feed_step):
    """Preserve sub-IK-tolerance axial increments while replacing lateral aim."""
    axis=np.asarray(axis,float);eef=np.asarray(eef,float)
    lateral=np.asarray(goal)-np.asarray(position)
    lateral-=axis*float(lateral@axis)
    accumulated=float((np.asarray(command)-eef)@axis)+float(feed_step)
    return eef+lateral+axis*accumulated


def gripper_fixture_contacts(session,fixtures=("end_stop","guide_base")):
    """Explicit simulation tactile adapter, never a visual-pose correction."""
    import mujoco
    m,d=session.ctx.model,session.ctx.data
    fixture_ids={session.ctx.body_id(name) for name in fixtures}
    hand_names={"right_gripper","right_hand","leftfinger","rightfinger","finger_joint1_tip","finger_joint2_tip"}
    rows=[]
    for index,c in enumerate(d.contact):
        b1,b2=map(int,m.geom_bodyid[[c.geom1,c.geom2]])
        other=b2 if b1 in fixture_ids else b1 if b2 in fixture_ids else None
        if other is None or m.body(other).name not in hand_names:continue
        force=np.zeros(6);mujoco.mj_contactForce(m,d,index,force)
        if not np.isfinite(force).all():
            raise ValueError("nonfinite gripper tactile contact force")
        rows.append(dict(geoms=[m.geom(c.geom1).name,m.geom(c.geom2).name],
            distance_m=float(c.dist),normal_force_n=max(0.,float(force[0]))))
    return dict(normal_force_n=sum(row["normal_force_n"] for row in rows),contacts=rows,
        source="simulation gripper/fixture tactile-contact adapter; separate from carried-pin force")


def _pin_sensor_contract():
    return dict(position_source="latest pre-close RGB-D registration transformed by current encoder FK",
        rigid_attachment_assumed=True, axial_slip_observed=False,
        bilateral_contact_is_not_no_slip_evidence=True,
        depth_cap_applies_to="registered sensor estimate; actual engagement requires independent acceptance",
        unknown_registration_action="fail before further servo; reobserve or regrasp required")


def _checked_pin_feedback(session,part):
    """Force and bilateral touch only; no simulator object position/velocity."""
    force=float(session.external_force(part))
    contact=gripper_fixture_contacts(session)
    grasp=session.ctx.grasp_contacts(part)
    values=[force,float(contact["normal_force_n"]),float(grasp["left_n"]),float(grasp["right_n"])]
    if not np.isfinite(values).all() or min(values)<0:
        raise ValueError("nonfinite or negative pin force / tactile sensor reading")
    return force,contact,grasp


def bounded_pin_press(session,part,target_z,force_stop):
    from .library import Result
    try:
        return _bounded_pin_press(session,part,target_z,force_stop)
    except (ValueError,TypeError,KeyError) as exc:
        return Result(False,dict(sensor_boundary_status="unknown",sensor_contract=_pin_sensor_contract()),
            "pin press sensor/command boundary: "+str(exc))


def _bounded_pin_press(session,part,target_z,force_stop):
    """Consume the single declared total-depth budget after learned insertion."""
    from .library import Result
    from .skills_v12 import control_position
    plan=session.artifacts.get("insert",{})
    contract=pin_depth_contract(session,part,plan)
    if contract is None: raise ValueError("bounded pin press requires explicit insertion-depth artifact")
    if not np.isfinite([target_z,force_stop]).all() or force_stop<=0:
        raise ValueError("invalid bounded pin press target/force threshold")
    peak=0.;steps=0;reason="press_step_budget_exhausted";trace=[];stopped=False
    stop=float(plan.get("gripper_contact_stop_n",.15))
    if not np.isfinite(stop) or stop<=0:raise ValueError("invalid pin gripper contact threshold")
    # Substep command cap remains tied to observed entry, not current trigger.
    for steps in range(241):
        position=control_position(session,part);depth=sensor_depth(session,part,position,contract)
        _check_measured_depth_envelope(depth,contract)
        force,contact,grasp=_checked_pin_feedback(session,part);peak=max(peak,force)
        if contact["normal_force_n"]>stop:
            return Result(False,dict(gripper_fixture_contact=contact,depth_m=depth),"gripper contacted receiver during pin press")
        if not grasp["held"]:
            return Result(False,reason="pin grasp lost during bounded press")
        if force>=force_stop:reason="pin_force_stop";stopped=True;break
        if depth>=contract["total_depth"]-.00005:reason="declared_total_depth_reached";stopped=True;break
        command=session.ctx.eef_pos()+contract["axis"]*.000025
        command,clipped=limit_pin_eef_command(session,part,command,contract,contract["total_depth"])
        session.arm.servo(command)
        if steps%20==0:trace.append(dict(depth_m=depth,force_n=force,clipped_m=clipped))
    return Result(stopped,dict(criterion="bounded contact action; released two-layer acceptance still required",
        maximum_total_depth_m=contract["total_depth"],peak_force_n=peak,steps=steps+1,stop_reason=reason,
        target_z_argument_diagnostic=float(target_z),trace=trace,sensor_contract=_pin_sensor_contract()),
        "" if stopped else "bounded pin press exhausted without observing depth or force stop")


def execute(session,part,artifact,policy,max_steps):
    from .library import Result
    try:
        return _execute_checked(session,part,artifact,policy,max_steps)
    except (ValueError,TypeError,KeyError) as exc:
        metrics=dict(sensor_boundary_status="unknown",sensor_contract=_pin_sensor_contract())
        session.artifacts["learned_insertion_result"]=metrics
        return Result(False,metrics,"pin insertion sensor/command boundary: "+str(exc))


def _execute_checked(session,part,artifact,policy,max_steps):
    """BC with axis-aware search and one explicit absolute CAD depth budget."""
    from .library import Result
    from .skills_v12 import control_position
    plan=session.artifacts[artifact];contract=pin_depth_contract(session,part,plan)
    if contract is None:return _execute_legacy(session,part,artifact,policy,max_steps)
    if (not np.isfinite(max_steps) or int(max_steps)!=max_steps or not 1<=int(max_steps)<=1000):
        raise ValueError("explicit-depth pin policy requires a finite 1..1000 step budget")
    actor=load_actor(policy);speed=float(plan["speed"]);force_limit=float(plan["force_limit"])
    if not (.001<=speed<=.02 and 0<force_limit<=30):return Result(False,reason="invalid pin policy envelope")
    dt=float(session.ctx.control_dt);target=np.asarray(plan["target"],float).copy();semantic=target.copy()
    if (not np.isfinite(dt) or dt<=0 or target.shape!=(3,) or not np.isfinite(target).all()):
        raise ValueError("invalid pin control timestep or target")
    rotation=session.arm.rotation.copy();position=control_position(session,part);previous=position.copy()
    if (np.asarray(rotation).shape!=(3,3) or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T@rotation,np.eye(3),atol=1.e-6,rtol=0.)
            or not np.isclose(np.linalg.det(rotation),1.,atol=1.e-6,rtol=0.)):
        raise ValueError("invalid pin commanded rotation")
    command=session.ctx.eef_pos().copy();peak=0.;trace=[];search=None;recovered=False;reached=False
    residual=float(session.decision_observation["objects"][part].get("fit_residual_m") or 0.)
    if not np.isfinite(residual) or residual<0:raise ValueError("invalid pin visual fit residual")
    sigma=max(.0005,residual)
    radius=min(.75*session.pin_insertion_config.plate_hole_half_width_m,3*(sigma+.0005))
    nominal_clearance=(min(session.pin_insertion_config.guide_inner_radius_m,
                           session.pin_insertion_config.plate_hole_half_width_m)
                       -session.pin_insertion_config.shaft_radius_m
                       -session.pin_insertion_config.radial_clearance_m)
    if nominal_clearance <= 0:
        raise ValueError("declared pin aperture has no radial insertion clearance")
    spiral_pitch=min(.0005,.5*nominal_clearance)
    trigger=min(.35,.04*force_limit);touch_stop=float(plan.get("gripper_contact_stop_n",.15))
    if not np.isfinite([sigma,touch_stop]).all() or touch_stop<=0:
        raise ValueError("invalid pin visual uncertainty or gripper force threshold")
    search_metrics=dict(maximum_radius_m=radius,spiral_pitch_m=spiral_pitch,
        iterations=0,entry_detected=False,
        reference="observed hole entry and encoder/FK registered shaft tip",depth_never_reset_at_trigger=True)
    reason="step_budget_exhausted";last_contact=None
    for steps in range(int(max_steps)):
        position=control_position(session,part);depth=sensor_depth(session,part,position,contract)
        _check_measured_depth_envelope(depth,contract)
        force,last_contact,grasp=_checked_pin_feedback(session,part);peak=max(peak,force)
        if last_contact["normal_force_n"]>touch_stop:
            return Result(False,dict(gripper_fixture_contact=last_contact,depth_m=depth,trace=trace),"gripper contacted receiver during insertion")
        if force>force_limit:return Result(False,dict(peak_force_n=peak,trace=trace),"learned insertion force limit")
        if not grasp["held"]:return Result(False,dict(trace=trace),"learned insertion grasp lost")
        error=target-position
        if search is None and (depth>=contract["command_depth"]-.0001 or np.linalg.norm(error)<=max(.0005,speed*dt*2)):
            reached=True;reason="declared_encoder_depth_or_target_reached";break
        if search is None and not recovered and force>=trigger:
            search=dict(start_depth=depth,steps=0,low_steps=0,center=target.copy(),probe=None,probe_high_steps=0)
        obs=np.r_[error/.003,(position-previous)/dt/.01,force/force_limit]
        action=np.asarray(actor(obs))*speed*dt;previous=position.copy();mode="behavior_cloned_insertion"
        if action.shape!=(3,) or not np.isfinite(action).all():
            raise ValueError("nonfinite or malformed pin policy action")
        if search is not None:
            mode="bounded_receiver_plane_spiral";search["steps"]+=1;n=search["steps"]
            search_metrics["iterations"]=n;search["low_steps"]=search["low_steps"]+1 if force<.10 else 0
            # As in the ring controller, pause XY at an early sensor-space
            # entry hint so a continuously expanding spiral cannot sweep a
            # newly entering shaft back out. This is not an acceptance label.
            if search["probe"] is None and depth-search["start_depth"]>=.0002 and search["low_steps"]>=5:
                search["probe"]=position.copy();search["probe_high_steps"]=0
                search_metrics["provisional_entry_probes"]=search_metrics.get("provisional_entry_probes",0)+1
            if search["probe"] is not None:
                search["probe_high_steps"]=search["probe_high_steps"]+1 if force>max(.2,2*trigger) else 0
                if search["probe_high_steps"]>=10:search["probe"]=None
            if depth-search["start_depth"]>=.004 and search["low_steps"]>=15:
                # Preserve the measured transverse correction, but go only
                # to the declared absolute depth, never trigger + 8 mm.
                target=position+contract["axis"]*(contract["command_depth"]-depth)
                search_metrics.update(entry_detected=True,recovery_control_target_m=target.tolist(),
                    depth_at_entry_detection_m=depth,remaining_declared_feed_m=max(0.,contract["command_depth"]-depth))
                search=None;recovered=True;command=session.ctx.eef_pos().copy()
            elif n>900:reason="bounded_search_exhausted";break
            else:
                sweep_radius=radius*min(n/900.,1.)
                sweep_angle=2*np.pi*sweep_radius/spiral_pitch
                circle=sweep_radius*(np.cos(sweep_angle)*contract["tangents"][0]
                    +np.sin(sweep_angle)*contract["tangents"][1])
                goal=search["center"]+circle if search["probe"] is None else search["probe"]
                if search["probe"] is not None:mode="provisional_entry_hold_transverse"
                command=advance_pin_search_command(command,session.ctx.eef_pos(),position,goal,contract["axis"],
                    min(speed*dt,.00006) if force<.5 else -.000015)
        else:command+=action
        command=np.clip(command,session.ctx.eef_pos()-.003,session.ctx.eef_pos()+.003)
        command,clipped=limit_pin_eef_command(session,part,command,contract,contract["command_depth"])
        session.arm.servo(command,rotation=rotation)
        if steps%25==0:trace.append(dict(step=steps,mode=mode,depth_m=depth,force_n=force,
            commanded_eef_position_m=command.tolist(),actor_observation=obs.tolist(),actor_action_m=action.tolist(),depth_clipped_m=clipped))
    from simbench.value.pin_geometry import evaluate_pin_context
    geometry=evaluate_pin_context(session.ctx,part,hole_offset_m=(0.,-.032 if part=="pin_left" else .032,0.),
        phase="inserted_while_held",released=False,touching_finger=True,config=session.pin_insertion_config)
    metrics=dict(policy=str(policy),policy_sha256=hashlib.sha256(Path(policy).read_bytes()).hexdigest(),
        controller="BC plus receiver-axis search with absolute CAD feed cap",steps=steps+1,peak_force_n=peak,
        visual_kinematic_target_reached=reached,stop_reason=reason,search=search_metrics,trace=trace,
        semantic_graph_target_m=semantic.tolist(),functional_insertion_control_target_m=target.tolist(),
        command_depth_m=contract["command_depth"],maximum_total_depth_m=contract["total_depth"],
        absolute_depth_limit_m=contract["absolute_depth"],gripper_contact_stop_n=touch_stop,
        gripper_fixture_contact=last_contact,independent_insertion_evaluation=geometry,
        input_source="RGB-D receiver/grasp registration, encoder FK, explicit simulated force/tactile adapters",
        sensor_contract=_pin_sensor_contract(),
        final_base_bridge_acceptance="separate mandatory released two-layer evaluator; no policy correction")
    session.artifacts["learned_insertion_result"]=metrics
    return Result(bool(geometry["success"]),metrics,"physical stop-bore engagement not reached")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--out", required=True)
    parser.add_argument("--episodes", type=int, default=72); args = parser.parse_args()
    result = train(args.out, args.episodes)
    print(json.dumps({k: v for k, v in result.items() if k not in ("train_tasks", "heldout_tasks")}, indent=2))
