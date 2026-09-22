"""Independent CAD/soft-contact audit, not a task-label replacement.

The dimensionless numerical guard is declared before these isolated runs.
No system seed, candidate result, or perception observation is read.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from simbench.value.pin_geometry import PinInsertionConfig, insertion_geometry


CONFIG=PinInsertionConfig(guide_inner_radius_m=.004,plate_hole_half_width_m=.004,
    guide_length_m=.036,required_depth_m=.006,radial_clearance_m=0.,
    aperture_shape='square',acceptance_mode='functional_contiguous',
    source='Independent 8 mm square bore / 6.6 mm shaft CAD contact audit')
MASS=.0036109135857152154


def numerical_guard(config=CONFIG):
    clearance=config.plate_hole_half_width_m-config.shaft_radius_m
    return min(.05*config.shaft_radius_m,.25*clearance)


def evaluate(origin,axis,*,released=True,touching_finger=False,contacts=(),stable=True,retained=True,config=CONFIG):
    """Audit-only proposed predicate; never used by the robot or system labels."""
    guard=numerical_guard(config)
    kwargs=dict(hole_axes=np.eye(3)[:2])
    strict=insertion_geometry(origin,axis,[0,0,0],[0,0,1],config,**kwargs)
    robust=insertion_geometry(origin,axis,[0,0,0],[0,0,1],
        replace(config,radial_clearance_m=-guard),**kwargs)
    penetration=max([max(0.,-float(c['distance_m'])) for c in contacts] or [0.])
    actual_wall_contact=any(c.get('shaft_receiver',False) and c.get('normal_force_n',0.)>0 for c in contacts)
    contact_consistent=bool(strict['inserted'] or actual_wall_contact)
    success=bool(robust['inserted'] and released and not touching_finger and retained
        and penetration<=guard and contact_consistent)
    return dict(success=success,strict_inserted=strict['inserted'],
        strict_contiguous_depth_m=strict['longest_contiguous_depth_span_m'],
        robust_geometry_inserted=robust['inserted'],robust_contiguous_depth_m=robust['longest_contiguous_depth_span_m'],
        numerical_guard_m=guard,maximum_receiver_contact_penetration_m=penetration,
        wall_contact_explains_strict_failure=contact_consistent,low_speed_diagnostic=stable,
        released=released,touching_finger=touching_finger)


def model_xml(dt,friction):
    # Exactly the task's pin collision primitives and STL-derived inertial
    # parameters. This fixture is a separate square receiver, not a full robot.
    root=ET.Element('mujoco',model='independent_square_receiver_contact_audit')
    # ``implicitfast`` is unavailable in the MuJoCo version used by the
    # reproducible 901 environment.  The contact audit does not rely on the
    # fast factorization variant, so use the portable implicit integrator.
    ET.SubElement(root,'option',timestep=str(dt),gravity='0 0 -9.81',integrator='implicit',
        cone='elliptic',iterations='100',tolerance='1e-10')
    default=ET.SubElement(root,'default')
    ET.SubElement(default,'geom',solref='.008 1',solimp='.95 .99 .001',margin='0',gap='0',
        friction=f'{friction} .01 .0001',condim='4')
    world=ET.SubElement(root,'worldbody')
    fixture=ET.SubElement(world,'body',name='receiver')
    walls=[('xp',(.009,0,-.018),(.005,.014,.018)),('xm',(-.009,0,-.018),(.005,.014,.018)),
           ('yp',(0,.009,-.018),(.004,.005,.018)),('ym',(0,-.009,-.018),(.004,.005,.018))]
    for name,pos,size in walls:
        ET.SubElement(fixture,'geom',name='wall_'+name,type='box',pos=' '.join(map(str,pos)),size=' '.join(map(str,size)))
    ET.SubElement(world,'geom',name='support_floor',type='box',pos='0 0 -.041',size='.014 .014 .005')
    body=ET.SubElement(world,'body',name='pin',pos='0 0 .012')
    ET.SubElement(body,'freejoint',name='pin_free')
    ET.SubElement(body,'inertial',mass=str(MASS),pos='0 0 -.003388584',
        diaginertia='1.007371e-06 1.007371e-06 1.070236e-07')
    for name,pos,size in [('shaft','0 0 -.0205','.0033 .0265'),('tip','0 0 -.047','.0027 .001'),('head','0 0 .006','.009 .007')]:
        ET.SubElement(body,'geom',name='pin_'+name,type='cylinder',pos=pos,size=size,mass='0')
    return ET.tostring(root,encoding='unicode')


def run_case(dt,friction,load_multiple,direction,out):
    import mujoco
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    xml=model_xml(dt,friction);(out/'fixture.xml').write_text(xml)
    model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model)
    bid=model.body('pin').id;receiver=model.body('receiver').id
    mujoco.mj_forward(model,data)
    load=np.array([np.cos(direction),np.sin(direction),0.])*MASS*9.81*load_multiple
    rows=[]
    clearance=CONFIG.plate_hole_half_width_m-CONFIG.shaft_radius_m
    window=.25;vmax=.1*clearance/window;wmax=vmax/(CONFIG.shaft_head_offset_m-CONFIG.shaft_tip_offset_m)
    # 0.5 s gravity settling, 1 s bounded lateral fixture load, 2 s unloading.
    duration=3.5;sample_every=max(1,int(round(.02/dt)))
    for index in range(int(round(duration/dt))):
        data.xfrc_applied[bid,:3]=load if .5<=data.time<1.5 else 0.
        mujoco.mj_step(model,data)
        if index%sample_every: continue
        contacts=[];floor=[]
        for ci,c in enumerate(data.contact):
            bodies=model.geom_bodyid[[c.geom1,c.geom2]]
            if bid not in bodies: continue
            force=np.zeros(6);mujoco.mj_contactForce(model,data,ci,force)
            geoms=[model.geom(c.geom1).name,model.geom(c.geom2).name]
            item=dict(geoms=geoms,distance_m=float(c.dist),normal_force_n=float(force[0]),
                shaft_receiver=bool(receiver in bodies and 'pin_shaft' in geoms))
            (contacts if receiver in bodies else floor).append(item)
        axis=data.xmat[bid].reshape(3,3)[:,2].copy()
        qvel=data.qvel[:6]
        # Free-joint angular velocity is local. Axial spin does not change
        # occupancy of a circular pin and must not become a pose-precision goal.
        tilt_speed=float(np.linalg.norm(qvel[3:5]))
        instantaneous_low_speed=bool(np.linalg.norm(qvel[:3])<=vmax and tilt_speed<=wmax)
        raw=evaluate(data.xpos[bid].copy(),axis,contacts=contacts,stable=instantaneous_low_speed)
        rows.append(dict(time_s=float(data.time),origin_m=data.xpos[bid].tolist(),axis=axis.tolist(),
            linear_speed_m_s=float(np.linalg.norm(qvel[:3])),angular_speed_rad_s=float(np.linalg.norm(qvel[3:])),
            axis_angular_speed_rad_s=tilt_speed,external_lateral_load_present=bool(.5<=data.time<1.5),
            contacts=contacts,floor_contacts=floor,**raw))
    tail=[r for r in rows if r['time_s']>=duration-window]
    loaded_tail=[r for r in rows if 1.5-window<=r['time_s']<1.5]
    stable=bool(max(r['linear_speed_m_s'] for r in tail)<=vmax
        and max(r['axis_angular_speed_rad_s'] for r in tail)<=wmax)
    last=rows[-1]
    final=evaluate(last['origin_m'],last['axis'],contacts=last['contacts'],stable=stable,
        retained=all(r['success'] for r in tail))
    summary=dict(dt_s=dt,friction=friction,load_multiple_of_weight=load_multiple,direction_rad=direction,
        load_n=load.tolist(),stable_window_s=window,linear_speed_limit_m_s=vmax,angular_speed_limit_rad_s=wmax,
        max_loaded_penetration_m=max(r['maximum_receiver_contact_penetration_m'] for r in rows if .5<=r['time_s']<1.5),
        max_loaded_final_window_penetration_m=max(r['maximum_receiver_contact_penetration_m'] for r in loaded_tail),
        loaded_final_window_low_speed=all(r['low_speed_diagnostic'] for r in loaded_tail),
        loaded_final_window_guard_pass=all(r['maximum_receiver_contact_penetration_m']<=numerical_guard() for r in loaded_tail),
        max_released_penetration_m=max(r['maximum_receiver_contact_penetration_m'] for r in tail),
        max_released_linear_speed_m_s=max(r['linear_speed_m_s'] for r in tail),
        max_released_angular_speed_rad_s=max(r['angular_speed_rad_s'] for r in tail),
        max_released_axis_angular_speed_rad_s=max(r['axis_angular_speed_rad_s'] for r in tail),
        final=final,strict_pass_through_final_window=all(r['strict_inserted'] for r in tail),
        robust_geometry_pass_through_final_window=all(r['robust_geometry_inserted'] for r in tail))
    (out/'trace.json').write_text(json.dumps(rows,indent=2));(out/'summary.json').write_text(json.dumps(summary,indent=2))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists(): raise FileExistsError('refusing to overwrite audit evidence')
    args.out.mkdir(parents=True)
    protocol=dict(schema='twingraph.v12.independent_contact_guard_audit.v3',
        guard_formula='min(0.05 * shaft_radius, 0.25 * nominal_radial_clearance)',
        guard_m=numerical_guard(),config=CONFIG.manifest(),mass_kg=MASS,
        timesteps_s=[.001,.002,.004],frictions=[.2,.45,.8],load_multiples_of_weight=[0.,.5,1.,2.],directions_rad=[0.,np.pi/4],
        physical_setup='Isolated free CAD pin in a fixed square bore with lower support; lateral force then 2 seconds unload',
        retention_rule='Every sampled instant in final 0.25 s retains continuous >=6mm shaft occupancy, no fingers and bounded penetration; velocity is diagnostic only',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=['Guard is a numerical-robustness assumption, not measured printer or real contact tolerance',
            'Analytic square fixture contact calibration is not a full assembly trial',
            'Fixture forces and initial poses are explicit privileged experiment interventions',
            'No candidate or development task state is used by this audit'])
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[]
    for dt in protocol['timesteps_s']:
        for friction in protocol['frictions']:
            for load in protocol['load_multiples_of_weight']:
                for direction in protocol['directions_rad']:
                    name=f'case_{len(rows):03d}'
                    row=run_case(dt,friction,load,direction,args.out/name);row['case']=name;rows.append(row)
                    print(json.dumps(dict(case=name,strict=row['final']['strict_inserted'],functional=row['final']['success'],
                        penetration_m=row['max_released_penetration_m'])),flush=True)
    report=dict(protocol=protocol,count=len(rows),strict_count=sum(r['final']['strict_inserted'] for r in rows),
        proposed_functional_count=sum(r['final']['success'] for r in rows),
        max_released_penetration_m=max(r['max_released_penetration_m'] for r in rows),cases=rows)
    (args.out/'summary.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__': main()
