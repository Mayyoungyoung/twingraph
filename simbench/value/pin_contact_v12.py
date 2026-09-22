"""Independent V12 contact-aware acceptance; never a policy observation.

The 5% shaft-radius / 25% clearance cap is a declared numerical-robustness
assumption, audited with an independent CAD fixture. It is not measured PLA
deformation or a manufacturing tolerance. Physical hole geometry is unchanged.
"""
from dataclasses import replace
import numpy as np
from .pin_geometry import evaluate_pin_state


def numerical_contact_guard(config):
    clearance=float(config.plate_hole_half_width_m-config.shaft_radius_m)
    if config.aperture_shape!='square' or clearance<=0 or config.radial_clearance_m!=0:
        raise ValueError('V12 contact guard requires the original square hole and zero extra CAD clearance')
    return min(.05*config.shaft_radius_m,.25*clearance)


def evaluate_contact_state(origin,axis,entry,rotation,*,released,touching_finger,phase,config,contacts):
    guard=numerical_contact_guard(config)
    common=dict(pin_origin=origin,pin_axis=axis,hole_entry=entry,hole_axis=rotation[:,2],
        released=released,touching_finger=touching_finger,phase=phase,hole_axes=rotation[:,:2].T)
    strict=evaluate_pin_state(**common,config=replace(config,contact_robustness=False))
    robust=evaluate_pin_state(**common,config=replace(config,contact_robustness=False,radial_clearance_m=-guard))
    maximum=max([max(0.,-float(c['distance_m'])) for c in contacts] or [0.])
    hole_axis=np.asarray(rotation[:,2],float)
    annotated=[]
    for contact in contacts:
        row=dict(contact)
        normal=np.asarray(row.get('normal_world',[]),float)
        valid=normal.shape==(3,) and np.isfinite(normal).all() and np.linalg.norm(normal)>1e-12
        axial=abs(float((normal/np.linalg.norm(normal))@hole_axis)) if valid else None
        radial=float(np.sqrt(max(0.,1.-axial**2))) if valid else None
        row.update(hole_axis_world=hole_axis.tolist(),normal_axial_component=axial,
            normal_radial_component=radial,radially_dominant=bool(valid and radial>axial))
        annotated.append(row)
    actual_wall=any(c.get('shaft_receiver') and float(c.get('normal_force_n',0))>0
        and c['radially_dominant'] for c in annotated)
    explanation=bool(strict['inserted'] or actual_wall)
    guard_pass=bool(maximum<=guard and explanation)
    geometry_ok=bool(robust['inserted'] and guard_pass)
    release_ok=bool(released and not touching_finger)
    success=geometry_ok and (phase=='inserted_while_held' or release_ok)
    return dict(robust,success=bool(success),inserted=geometry_ok,inserted_while_held=geometry_ok,
        inserted_after_release=bool(geometry_ok and release_ok),
        retained_after_stroke=bool(geometry_ok and release_ok and phase=='retained_after_stroke'),
        acceptance_schema='twingraph.v12.pin_contact_function.v1',
        strict_zero_interpenetration=strict,numerical_contact_guard_m=guard,
        maximum_receiver_contact_penetration_m=maximum,receiver_contacts=annotated,
        wall_contact_explains_strict_failure=explanation,contact_guard_pass=guard_pass,
        numerical_guard_source='Predeclared min(5% shaft radius,25% nominal radial clearance); independent CAD load/dt audit',
        evaluation_source='independent simulator geometry/contact diagnostics; never detector or policy input')


def evaluate_context_contact(ctx,part,fixture,origin,axis,entry,rotation,*,released,touching_finger,phase,config):
    import mujoco
    model,data=ctx.model,ctx.data;bid=ctx.body_id(part);fid=ctx.body_id(fixture)
    contacts=[];finger=bool(touching_finger)
    for index,contact in enumerate(data.contact):
        bodies=list(map(int,model.geom_bodyid[[contact.geom1,contact.geom2]]))
        if bid not in bodies: continue
        other=bodies[1] if bodies[0]==bid else bodies[0]
        if 'finger' in model.body(other).name and contact.dist<=0: finger=True
        if other!=fid: continue
        force=np.zeros(6);mujoco.mj_contactForce(model,data,index,force)
        geoms=[model.geom(contact.geom1).name,model.geom(contact.geom2).name]
        contacts.append(dict(geoms=geoms,distance_m=float(contact.dist),normal_force_n=float(force[0]),
            normal_world=contact.frame[:3].tolist(),
            shaft_receiver=bool(f'{part}_shaft' in geoms)))
    result=evaluate_contact_state(origin,axis,entry,rotation,released=released,touching_finger=finger,
        phase=phase,config=config,contacts=contacts)
    joint=int(model.body_jntadr[bid]);adr=int(model.jnt_dofadr[joint]);velocity=data.qvel[adr:adr+6]
    result['motion_diagnostics']=dict(linear_speed_m_s=float(np.linalg.norm(velocity[:3])),
        angular_speed_rad_s=float(np.linalg.norm(velocity[3:])),
        criterion='diagnostic only: yaw, tilt velocity and small motion are not task failure criteria')
    result['sample_time_s']=float(data.time)
    return result


def functional_retention_window(sample,wait,initial,*,duration,samples):
    """Observe unchanged controls; fail if a sample no longer occupies the bore.

    Existing strict diagnostics remain per sample. No velocity threshold or
    desired orientation is introduced. Initial failure is never waited away.
    """
    if duration<=0 or samples<2: raise ValueError('invalid functional retention window')
    rows=[initial]
    if initial['success']:
        for _ in range(samples-1):
            wait(duration/(samples-1));row=sample();rows.append(row)
            if not row['success']: break
    result=dict(rows[-1])
    result['success']=bool(len(rows)==samples and all(r['success'] for r in rows))
    actual_duration=(float(rows[-1]['sample_time_s']-rows[0]['sample_time_s'])
        if all('sample_time_s' in r for r in rows) else None)
    if actual_duration is not None and actual_duration+1e-9<duration: result['success']=False
    result['functional_retention_window']=dict(duration_required_s=float(duration),samples_required=int(samples),
        duration_observed_s=actual_duration,samples_observed=len(rows),pass_all=result['success'],samples=rows,
        method='sampled retention under fixed controls; >=6mm contiguous shaft occupancy, contact guard and no fingers at every sample; no continuous-time guarantee')
    return result
