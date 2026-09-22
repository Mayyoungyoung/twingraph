"""Bounded end-stop seating with observed geometry and force feedback only.

No simulator object pose or functional evaluator is used for correction or
termination. The subsequent independent released/corridor checks are required.
"""
import itertools
import numpy as np
from scipy.spatial.transform import Rotation


def search_offsets(radius):
    """Origin plus three unbiased eight-direction rings inside the envelope."""
    yield np.zeros(2)
    directions=((1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1))
    for fraction in (1/3,2/3,1):
        for direction in directions:
            unit=np.asarray(direction,float)/np.linalg.norm(direction)
            yield unit*float(radius)*fraction


def seat_contract(session,target_z,config):
    obs=session.decision_observation or {};cad=session.planning_cad
    if obs.get("backend")!="rgbd_geometry":raise ValueError("end-stop seating needs RGB-D geometry")
    base=obs.get("fixtures",{}).get("guide_base",{});part=obs.get("objects",{}).get("end_stop",{})
    if not base.get("valid") or not part.get("valid"):raise ValueError("receiver/source visual pose unknown")
    bp=np.asarray(base.get("position_m"),float);bq=np.asarray(base.get("quat_wxyz"),float)
    if (bp.shape!=(3,) or bq.shape!=(4,) or not np.isfinite(np.r_[bp,bq]).all() or np.linalg.norm(bq)<1.e-8):
        raise ValueError("invalid receiver RGB-D pose")
    bR=Rotation.from_quat(bq[[1,2,3,0]]).as_matrix()
    mate=np.asarray(cad["end_stop_mating_pose_in_base"]["position_m"],float)
    bounds=np.asarray(cad["parts"]["end_stop"]["body_bounds_m"],float)
    mass=float(cad["parts"]["end_stop"]["assumed_mass_kg"])
    if (mate.shape!=(3,) or bounds.shape!=(2,3) or not np.isfinite(np.r_[mate,bounds.ravel(),mass]).all()
            or not np.all(bounds[1]>bounds[0]) or mass<=0):raise ValueError("invalid seating CAD")
    target=bp+bR@mate
    if not np.isfinite(target_z) or abs(float(target_z)-target[2])>1.e-6:
        raise ValueError("press target does not match observed base/CAD mating target")
    posts=[r for r in cad["collision_primitives"]["guide_base"] if r["name"].startswith("stop_locator_")]
    if len(posts)!=4 or any(r["type"]!="cylinder" for r in posts):
        raise ValueError("four declared printed locator cylinders required")
    post_top=max(float(r["pos"][2])+float(r["size"][1]) for r in posts)
    surface=float(mate[2]+bounds[0,2]);rise=post_top-surface
    if not np.isfinite(rise) or rise<=0:raise ValueError("invalid locator height above support surface")
    indicators=[float(r.get("fit_residual_m") or 0.) for r in (base,part)]
    if not np.isfinite(indicators).all() or min(indicators)<0:raise ValueError("invalid visual uncertainty indicator")
    uncertainty=config.uncertainty_multiplier*sum(max(config.minimum_pose_indicator_m,x) for x in indicators)
    radius=min(config.maximum_radius_m,uncertainty)
    # Seating keeps the measured gripper orientation fixed.  Clearance over
    # the locator posts therefore depends on the lowest declared body point,
    # not on a bounding sphere that lifts this light part roughly 50 mm and
    # can destroy the grasp before every retry.
    corners=np.array(list(itertools.product(*zip(bounds[0],bounds[1]))))
    body_radius=float(np.max(np.linalg.norm(corners,axis=1)))
    bottom_body_z=float(bounds[0,2])
    safe_center_height=post_top-bottom_body_z+uncertainty
    return dict(base_position=bp,base_rotation=bR,axis=bR[:,2],target=target,
        target_height=float(mate[2]),safe_center_height=safe_center_height,
        locator_top_height=post_top,support_surface_height=surface,search_radius=radius,
        seating_band=(1-config.minimum_locator_inset_fraction)*rise,
        minimum_support_force=config.supporting_force_weight_fraction*mass*9.81,
        uncertainty_indicator=uncertainty,body_bounding_radius=body_radius,
        bottom_body_z_m=bottom_body_z,
        clearance_rule="fixed-orientation lowest CAD point above locator top plus visual uncertainty")


def execute(session,part,target_z,force_stop):
    from .library import Result
    from .skills_v12 import EndStopSeatConfig,control_position
    from .sensor_learning_v12 import gripper_fixture_contacts
    config=getattr(session,"end_stop_seat_config_v12",EndStopSeatConfig())
    trace=[];attempts=[];steps=0;peak=0.;contract=None
    def finish(ok,reason):
        metrics=dict(controller=config.schema,config=config.manifest(),steps=steps,attempts=attempts,trace=trace,
            peak_force_n=peak,force_stop_n=force_stop,stop_reason=reason,
            actual_capture_or_bridge_verified=False,
            criterion="sensor-estimated locator inset with support force; independent release/corridor/bridge acceptance follows",
            input_source="pre-close RGB-D registration, encoder FK, explicit simulated force/tactile adapters",
            axial_slip_observed=False)
        if contract is not None:
            metrics["geometry"]={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in contract.items()}
        session.artifacts["end_stop_seating_result"]=metrics
        return Result(bool(ok),metrics,"" if ok else reason)
    try:
        if part!="end_stop":raise ValueError("bounded end-stop seating only supports end_stop")
        if not np.isfinite(force_stop) or force_stop<=0:raise ValueError("invalid seating force stop")
        if session.planning_cad.get("end_stop_seat_search")!=config.manifest():
            raise ValueError("graph CAD and end-stop controller envelope disagree")
        contract=seat_contract(session,target_z,config)
        dt=float(session.ctx.control_dt)
        if not np.isfinite(dt) or dt<=0:raise ValueError("invalid control timestep")
        axis=contract["axis"];rotation=np.asarray(session.arm.rotation,float).copy()
        if (rotation.shape!=(3,3) or not np.isfinite(rotation).all()
                or not np.allclose(rotation.T@rotation,np.eye(3),atol=1.e-6,rtol=0.)
                or not np.isclose(np.linalg.det(rotation),1.,atol=1.e-6,rtol=0.)):
            raise ValueError("invalid seating command rotation")
        def sample(phase,attempt):
            nonlocal peak
            position=control_position(session,part)
            force=float(session.external_force(part));contact=session.ctx.grasp_contacts(part)
            shell=gripper_fixture_contacts(session,fixtures=("guide_base",))
            values=[force,float(contact["left_n"]),float(contact["right_n"]),float(shell["normal_force_n"])]
            if not np.isfinite(values).all() or min(values)<0:raise ValueError("invalid seating force/tactile input")
            if not contact["held"]:raise ValueError("end-stop grasp lost")
            if shell["normal_force_n"]>config.gripper_base_contact_stop_n:
                raise ValueError("gripper shell contacted base during seating")
            peak=max(peak,force);height=float((position-contract["base_position"])@axis)
            if steps%10==0:
                trace.append(dict(step=steps,phase=phase,attempt=attempt,registered_height_m=height,
                    registered_position_m=position.tolist(),force_n=force,gripper_base_force_n=shell["normal_force_n"]))
            return position,height,force
        def servo(command):
            nonlocal steps
            if steps>=config.maximum_control_steps:raise ValueError("bounded seating total step budget exhausted")
            if not np.isfinite(command).all():raise ValueError("invalid seating command")
            session.arm.servo(command,rotation=rotation);steps+=1
        def move_registered(goal,phase,attempt,force_ceiling):
            for _ in range(config.maximum_control_steps):
                position,height,force=sample(phase,attempt)
                if force>force_ceiling:raise ValueError("seating retract/transfer force increased instead of unloading")
                delta=goal-position;distance=float(np.linalg.norm(delta))
                if distance<=config.control_tracking_tolerance_m:return
                command=session.ctx.eef_pos()+delta*min(1.,config.lift_speed_m_s*dt/distance)
                servo(command)
            raise ValueError("seating transfer budget exhausted")
        position,height,force=sample("entry",0)
        if (height<=contract["target_height"]+contract["seating_band"]
                and force>=contract["minimum_support_force"]):
            return finish(True,"already within sensor seating band with contact")
        for index,offset in enumerate(search_offsets(contract["search_radius"])):
            if index>=config.maximum_attempts:break
            position,height,force=sample("before_lift",index)
            unload_ceiling=force+float(force_stop)
            lift=position+axis*max(0.,contract["safe_center_height"]-height)
            move_registered(lift,"lift_above_locators",index,unload_ceiling)
            goal=contract["target"]+contract["base_rotation"][:,:2]@offset
            high_goal=goal+axis*(contract["safe_center_height"]-contract["target_height"])
            # Lateral transfer is still an unloading motion.  Bound any
            # increase relative to the force observed before retract rather
            # than pretending a pre-existing contact load is zero.
            move_registered(high_goal,"translate_above_locators",index,unload_ceiling)
            command=session.ctx.eef_pos().copy();status="budget"
            while steps<config.maximum_control_steps:
                position,height,force=sample("guarded_seating",index)
                if (height<=contract["target_height"]+contract["seating_band"]
                        and force>=contract["minimum_support_force"]):
                    attempts.append(dict(attempt=index,offset_base_xy_m=offset.tolist(),result="sensor_seat_contact"))
                    return finish(True,"sensor seating band and support contact reached")
                if force>=force_stop:status="contact_above_seating_band";break
                if height<=contract["target_height"]+config.control_tracking_tolerance_m:
                    status="target_height_without_support_contact";break
                # Accumulate sub-IK increments, while clamping absolute
                # registered depth to this observed CAD mating plane.
                command-=axis*config.descent_speed_m_s*dt
                predicted=position+command-session.ctx.eef_pos()
                predicted_height=float((predicted-contract["base_position"])@axis)
                command+=axis*max(0.,contract["target_height"]-predicted_height)
                servo(command)
            attempts.append(dict(attempt=index,offset_base_xy_m=offset.tolist(),result=status,
                registered_height_m=height,force_n=force))
        return finish(False,"bounded seating attempts exhausted; no sensor-confirmed seat")
    except (ValueError,TypeError,KeyError) as exc:
        return finish(False,"end-stop seating boundary: "+str(exc))
