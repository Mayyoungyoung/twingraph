"""Typed, label-free frozen geometry and control features for V33 value learning."""
from copy import deepcopy
import numpy as np
from simbench.assembly.library import DEFAULT_CAPABILITIES
from simbench.value.plan import PlanIR,digest
from simbench.value.skill_graph import compile_graph
from simbench.value.generic_graph_value_v15 import encode_graph,FEATURES as BASE_FEATURES

GEOMETRY=('min_clearance_m','required_clearance_m','environment_min_clearance_m',
    'source_body_min_clearance_m','pad_face_axial_overlap_m','head_pad_axial_overlap_m',
    'grasp_width_m','minimum_final_depth_m','goal_orientation_residual_rad')
PARAMETERS=('height','clearance','force','speed','force_limit','press_force','width',
    'pin_command_depth_m','pin_press_extra_m','lift_first_m')
EXTRA=tuple([f'geometry_{n}_{s}' for n in GEOMETRY for s in ('value','known')]+
    [f'command_{n}_{s}' for n in PARAMETERS for s in ('value','known')]+
    ['geometry_available','necessary_pass','necessary_unknown','necessary_rejected','conditioned_source_geometry_known',
     'source_ik_checked','body_geometry_checked','inside_grasp_face','released_push','held_insert',
     'grasp_yaw_sin','grasp_yaw_cos','placement_yaw_sin','placement_yaw_cos','angles_known',
     'target_delta_x','target_delta_y','target_delta_z','target_distance','target_known',
     'eef_transfer_x','eef_transfer_y','eef_transfer_z','eef_transfer_distance','eef_transfer_known',
     'nearest_conditioned_object_distance','nearest_conditioned_object_known','planned_predecessor_fraction',
     'node_generator','node_executable'])
FEATURES=tuple(BASE_FEATURES)+EXTRA
SCHEMA='twingraph.value.v33.typed_geometry.r1'


def optional(value):
    if value is None:return (0.,0.)
    if not isinstance(value,(int,float)) or not np.isfinite(value):raise ValueError('invalid typed numeric feature')
    return float(value),1.


def vector(value):
    if value is None:return None
    v=np.asarray(value,dtype=float)
    if v.shape!=(3,) or not np.isfinite(v).all():raise ValueError('invalid pre-execution position')
    return v


def encode_candidate(request,entry,enhanced=True):
    if digest(entry['assembly_plan_ir'])!=entry['assembly_plan_sha256'] or digest(entry['complete_candidate_plan_ir'])!=entry['plan_sha256']:
        raise ValueError('frozen plan hash changed')
    plan=PlanIR.from_dict(entry['assembly_plan_ir'])
    proposal=entry['proposal']
    if proposal['choices']!=plan.prefix['choices'] or proposal['order']!=plan.prefix['order']:
        raise ValueError('proposal parameters do not match the executable graph')
    observation=deepcopy(request['decision_observation']);observation.setdefault('robot',{})
    observation['goals']=[dict(predicate='assembly_through_handle',both_pins_through_base=True)]
    for part,caps in DEFAULT_CAPABILITIES.items():
        if part in observation['objects']:observation['objects'][part]['capabilities']=list(caps)
    graph=compile_graph(observation,plan);encoded=encode_graph({'assembly':graph})
    if not enhanced:return encoded
    objects=observation['objects'];targets=observation.get('assembly_targets',{})
    parts=proposal['order'];extra=[]
    for node in graph['nodes']:
        part=node.get('roles',{}).get('manipulated');choice=proposal['choices'].get(part,{})
        geom=proposal.get('necessary_geometry',{}).get(part,{})
        for key in ('yaw','height','placement_yaw','pin_command_depth_m','pin_press_extra_m'):
            if key in geom and geom[key]!=choice.get(key):raise ValueError(f'geometry/plan mismatch: {part}.{key}')
        row=[]
        for name in GEOMETRY:row.extend(optional(geom.get(name)))
        for name in PARAMETERS:row.extend(optional(choice.get(name)))
        row.extend([bool(geom),geom.get('status')=='necessary_pass',geom.get('status')=='unknown',geom.get('status')=='rejected',
            'source_receiver_pose_modes' in geom,bool(geom.get('source_grasp_ik_checked')),bool(geom.get('source_body_checked')),
            geom.get('grasp_face_status')=='inside',choice.get('transport_mode')=='released_push',choice.get('transport_mode')=='held_insert'])
        yaw=choice.get('yaw');placement=choice.get('placement_yaw')
        row.extend([np.sin(yaw or 0.),np.cos(yaw or 0.) if yaw is not None else 0.,
            np.sin(placement or 0.),np.cos(placement or 0.) if placement is not None else 0.,yaw is not None and placement is not None])
        pos=vector(objects.get(part,{}).get('position_m'));target=vector(targets.get(part,{}).get('position_m'))
        def delta(a,b):
            if a is None or b is None:return [0.]*5
            d=b-a;return [*d,float(np.linalg.norm(d)),1.]
        row.extend(delta(pos,target));row.extend(delta(vector(geom.get('source_eef_m')),vector(geom.get('target_eef_m'))))
        predecessors=parts[:parts.index(part)] if part in parts else []
        others=[]
        for other,obj in objects.items():
            if other==part:continue
            # Distance to a conditional predecessor target, not its stale supply pose.
            other_pos=targets.get(other,{}).get('position_m') if other in predecessors else obj.get('position_m')
            other_pos=vector(other_pos)
            if pos is not None and other_pos is not None:others.append(float(np.linalg.norm(pos-other_pos)))
        row.extend([min(others) if others else 0.,bool(others),len(predecessors)/max(1,len(parts)),
                    node.get('kind')=='generator',node.get('kind')=='executable'])
        if len(row)!=len(EXTRA):raise AssertionError('typed feature schema mismatch')
        extra.append(row)
    encoded['x']=np.concatenate([encoded['x'],np.asarray(extra,np.float32)],axis=1)
    if not np.isfinite(encoded['x']).all():raise ValueError('nonfinite V33 features')
    return encoded


def encode_candidates(request,enhanced=True):
    return [encode_candidate(request,entry,enhanced) for entry in request['candidates']]
