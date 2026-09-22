"""Static full-gripper / visual-CAD clearance queries, without a rollout.

The isolated model contains the unscaled Panda collision meshes (including
the shell meshes disabled in historical r6) and declared receiver primitives.
Only supplied RGB-D poses or explicit fixture calibration position receivers.
No live MjData object, free-body state, success label, or policy is accepted.
Mesh collision uses MuJoCo's convex hull, conservatively including concavities.
"""
from copy import deepcopy
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

PANDA = Path(__file__).resolve().parents[1] / "assets" / "panda" / "panda.xml"
SHELLS = ("hand_collision", "finger1_collision", "finger2_collision")
PADS = ("finger1_pad_collision", "finger2_pad_collision")


def _fmt(x):
    return " ".join(str(float(v)) for v in x)


def pose_matrix(row):
    if not row.get("valid") or row.get("position_m") is None or row.get("quat_wxyz") is None:
        raise ValueError("clearance requires an explicitly valid RGB-D/CAD pose")
    p, q = np.asarray(row["position_m"], float), np.asarray(row["quat_wxyz"], float)
    if p.shape != (3,) or q.shape != (4,) or not np.isfinite(np.r_[p,q]).all() or np.linalg.norm(q)<1e-8:
        raise ValueError("invalid clearance pose")
    return p, Rotation.from_quat(q[[1,2,3,0]]).as_matrix()


def common_corridor_entry(route):
    """Intersect the shared shaft line with the observed stop entry plane."""
    if "stop_entry_on_common_axis_m" in route:
        entry=np.asarray(route["stop_entry_on_common_axis_m"],float)
        if entry.shape!=(3,) or not np.isfinite(entry).all():raise ValueError("invalid shared-axis entry")
        return entry
    point=np.asarray(route["common_axis_point_m"],float)
    outward=np.asarray(route["axis"],float);outward/=np.linalg.norm(outward)
    normal=np.asarray(route.get("stop_axis",outward),float)
    denominator=float(normal@outward)
    if abs(denominator)<1e-6:raise ValueError("shaft axis does not cross the observed stop entry plane")
    return point+outward*float(normal@(np.asarray(route["stop_entry_m"])-point))/denominator


class GripperClearance:
    """Collision-only CAD scene. Queries advance no physics or robot control."""
    def __init__(self, primitives, *, panda_path=PANDA, distance_bound=.02):
        import mujoco
        self.mj=mujoco; self.distance_bound=float(distance_bound)
        if not 0 < self.distance_bound <= .05: raise ValueError("invalid distance bound")
        panda_path=Path(panda_path); source=ET.parse(panda_path).getroot()
        root=ET.Element("mujoco", model="declared_full_gripper_clearance")
        ET.SubElement(root,"compiler", angle="radian")
        ET.SubElement(root,"option", gravity="0 0 0")
        assets=ET.SubElement(root,"asset")
        for name in ("hand","finger"):
            mesh=deepcopy(source.find(f"asset/mesh[@name='{name}']"))
            mesh.set("file",str((panda_path.parent/mesh.get("file")).resolve()))
            assets.append(mesh)
        world=ET.SubElement(root,"worldbody")
        gripper=deepcopy(source.find(".//body[@name='right_gripper']"))
        gripper.set("pos","0 0 0"); gripper.set("quat","1 0 0 0")
        # A separate free root avoids MuJoCo's fixed-body/weld contact
        # exclusion between the palm and static receivers. It is placed only
        # in this scratch CAD query; no dynamics or real object is advanced.
        ET.SubElement(gripper,"freejoint",name="query_gripper_pose")
        self.original_masks={}
        for body in gripper.iter("body"):
            for geom in list(body.findall("geom")):
                if geom.get("name") not in SHELLS+PADS:
                    body.remove(geom); continue
                name=geom.get("name")
                self.original_masks[name]=dict(contype=int(geom.get("contype",1)),
                    conaffinity=int(geom.get("conaffinity",1)))
                # Only gripper/receiver pairs; this query is not the robot
                # self-collision checker. No assembly pair is filtered.
                geom.set("contype","1"); geom.set("conaffinity","2")
                geom.set("margin",str(self.distance_bound)); geom.attrib.pop("material",None)
            for site in list(body.findall("site")):
                if site.get("name")!="grip_site": body.remove(site)
        world.append(gripper)
        self.receivers=tuple(primitives)
        if not self.receivers: raise ValueError("receiver CAD primitives required")
        for part, rows in primitives.items():
            body=ET.SubElement(world,"body",name=part,pos="0 0 0")
            for row in rows:
                if row["type"] not in ("box","cylinder","sphere","capsule"):
                    raise ValueError("receiver query requires declared primitive geometry")
                ET.SubElement(body,"geom",name=row["name"],type=row["type"],
                    pos=_fmt(row["pos"]),quat=_fmt(row.get("quat_wxyz",[1,0,0,0])),
                    size=_fmt(row["size"]),contype="2",conaffinity="1",
                    margin=str(self.distance_bound))
        self.model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding="unicode"))
        self.data=mujoco.MjData(self.model)
        self.root_id=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,"right_gripper")
        self.eef_id=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_SITE,"grip_site")
        self.finger_qadr=[int(self.model.jnt_qposadr[mujoco.mj_name2id(self.model,
            mujoco.mjtObj.mjOBJ_JOINT,n)]) for n in ("finger_joint1","finger_joint2")]
        self.root_qadr=int(self.model.jnt_qposadr[mujoco.mj_name2id(self.model,
            mujoco.mjtObj.mjOBJ_JOINT,"query_gripper_pose")])
        mujoco.mj_forward(self.model,self.data)
        self.eef_local=self.data.site_xpos[self.eef_id].copy()
        self.provenance=dict(panda_xml_sha256=hashlib.sha256(panda_path.read_bytes()).hexdigest(),
            mesh_sha256={name:hashlib.sha256((panda_path.parent/"gripper_meshes"/(name+".stl")).read_bytes()).hexdigest()
                        for name in ("hand","finger")}, original_masks=self.original_masks,
            shell_policy="unscaled original collision-mesh convex hulls plus original pads",
            no_simulation_steps=True, pose_source="caller-provided RGB-D or explicitly declared calibration")

    def query(self, eef_position, eef_rotation, finger_gap, receiver_poses):
        if not 0 <= finger_gap <= .04: raise ValueError("finger gap outside actual actuator range")
        p=np.asarray(eef_position,float); R=np.asarray(eef_rotation,float)
        if p.shape!=(3,) or R.shape!=(3,3) or not np.isfinite(np.r_[p,R.ravel()]).all():
            raise ValueError("invalid declared gripper pose")
        self.data.qpos[self.root_qadr:self.root_qadr+3]=p-R@self.eef_local
        q=Rotation.from_matrix(R).as_quat(); self.data.qpos[self.root_qadr+3:self.root_qadr+7]=q[[3,0,1,2]]
        self.data.qpos[self.finger_qadr]=[finger_gap,-finger_gap]
        for part in self.receivers:
            xyz,rotation=pose_matrix(receiver_poses.get(part,{}))
            body_id=self.mj.mj_name2id(self.model,self.mj.mjtObj.mjOBJ_BODY,part)
            self.model.body_pos[body_id]=xyz
            quat=Rotation.from_matrix(rotation).as_quat(); self.model.body_quat[body_id]=quat[[3,0,1,2]]
        self.mj.mj_forward(self.model,self.data)
        pairs=[]
        for contact in self.data.contact:
            names=[self.model.geom(contact.geom1).name,self.model.geom(contact.geom2).name]
            if not any(n in SHELLS+PADS for n in names): continue
            pairs.append(dict(geoms=names,distance_m=float(contact.dist),
                shell_involved=any(n in SHELLS for n in names),
                contact_world_m=contact.pos.tolist()))
        minimum=min([c["distance_m"] for c in pairs] or [self.distance_bound])
        return dict(min_clearance_m=float(minimum), distance_is_lower_bound=not pairs,
            collision=bool(minimum<0), pairs=sorted(pairs,key=lambda c:c["distance_m"]))


def receiver_primitives_from_model(model, parts=("end_stop","guide_base")):
    """Copy immutable body-local CAD only; no MjData or world body pose read."""
    import mujoco
    names={int(mujoco.mjtGeom.mjGEOM_BOX):("box",3),int(mujoco.mjtGeom.mjGEOM_CYLINDER):("cylinder",2),
           int(mujoco.mjtGeom.mjGEOM_SPHERE):("sphere",1),int(mujoco.mjtGeom.mjGEOM_CAPSULE):("capsule",2)}
    rows={}
    for part in parts:
        bid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,part)
        if bid<0: raise ValueError(f"missing receiver CAD: {part}")
        rows[part]=[]
        for gid in np.flatnonzero(model.geom_bodyid==bid):
            if not (model.geom_contype[gid] or model.geom_conaffinity[gid]): continue
            kind=int(model.geom_type[gid])
            if kind not in names: raise ValueError("nonprimitive receiver CAD requires explicit decomposition")
            name,count=names[kind]
            rows[part].append(dict(name=model.geom(gid).name,type=name,pos=model.geom_pos[gid].tolist(),
                quat_wxyz=model.geom_quat[gid].tolist(),size=model.geom_size[gid,:count].tolist()))
    return rows


def pin_clearance_catalog(observation,cad,part,*,source_yaws=(0.,np.pi/2,np.pi,-np.pi/2),
        height_offsets=(0.,.004,.007),command_depths=None,press_extras=(0.,),placement_yaws=None,
        goal_mode="predicted_mated",required_clearance_m=None):
    """Generate pin grasp/goal choices conditioned on visual receiver geometry.

    Source grasp/withdrawal still requires plan_grasp's continuous robot IK
    and full-shell collision checks. This query certifies only the sampled
    future gripper/fixture sweep, not contact insertion or grasp retention.
    It never selects an unobserved fixture pose as a perception result.
    """
    from .control import down
    if part not in ("pin_left","pin_right"):raise ValueError("pin catalog only")
    if goal_mode not in ("predicted_mated","observed_installed"):raise ValueError("unknown fixture prediction mode")
    source_row=observation.get("objects",{}).get(part,{})
    base_row=observation.get("fixtures",{}).get("guide_base",{})
    try:
        source_position,source_R=pose_matrix(source_row)
        base_position,base_R=pose_matrix(base_row)
    except ValueError as exc:
        return [dict(part=part,status="unknown",reason=str(exc),min_clearance_m=None)]
    if goal_mode=="observed_installed":
        stop_row=observation.get("objects",{}).get("end_stop",{})
        relation=observation.get("fixture_relations",{}).get("end_stop_to_base",{})
        route=relation.get("holes",{}).get(part,{})
        try:stop_position,stop_R=pose_matrix(stop_row)
        except ValueError as exc:return [dict(part=part,status="unknown",reason=str(exc),min_clearance_m=None)]
        if not route.get("geometric_route_exists"):
            return [dict(part=part,status="rejected",reason="observed two-layer shaft corridor unavailable",min_clearance_m=None)]
        entry=common_corridor_entry(route)
        outward=np.asarray(route.get("axis",route["base_axis"]),float)
        minimum=float(route["minimum_total_depth_m"])
        goal_source="observed end_stop/base CAD transforms; contact not verified"
    else:
        mating=cad["end_stop_mating_pose_in_base"]
        q=np.asarray(mating["quat_wxyz"],float)
        stop_R=base_R@Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
        stop_position=base_position+base_R@np.asarray(mating["position_m"])
        offset=cad["pin_hole_offsets_m"][("pin_left","pin_right").index(part)]
        entry=stop_position+stop_R@np.asarray(offset);outward=base_R[:,2]
        minimum=float(cad["pin_bridge_minimum_total_depth_m"])
        goal_source="predicted future mating pose from observed base and unchanged CAD; not an observed installed stop"
        stop_row=base_row
    outward=outward/np.linalg.norm(outward)
    q=Rotation.from_matrix(stop_R).as_quat()
    receivers={"guide_base":base_row,"end_stop":dict(valid=True,position_m=stop_position.tolist(),quat_wxyz=q[[3,0,1,2]].tolist())}
    # Other CAD parts may be present in metadata without a supplied future
    # pose. Do not invent those poses or advertise this as full-scene safety.
    query=GripperClearance({name:cad["collision_primitives"][name] for name in ("end_stop","guide_base")})
    uncertainty=max(.0005,float(source_row.get("fit_residual_m") or 0.))+max(.0005,float(stop_row.get("fit_residual_m") or 0.))
    required=float(3*uncertainty if required_clearance_m is None else required_clearance_m)
    if required<0 or not np.isfinite(required):raise ValueError("invalid gripper clearance requirement")
    # Along/normals to the actual receiver edges; no world-yaw hardcoding.
    base_yaw=float(np.arctan2(stop_R[1,0],stop_R[0,0]))
    goals=tuple(base_yaw+angle for angle in (0.,np.pi/2,np.pi,-np.pi/2)) if placement_yaws is None else tuple(placement_yaws)
    depths=tuple(command_depths) if command_depths is not None else (min(minimum+.001,float(cad["pin_head_seated_total_depth_m"])),)
    rows=[];reference=cad["parts"][part]["grasp_reference"]
    gap=float(reference["width_m"])/2+.0005
    tip=float(cad["pin_shaft_offsets_m"][0]);head_bottom=float(cad["pin_head_underside_body_z_m"])
    head_top=float(cad["parts"][part]["body_bounds_m"][1][2])
    for height in height_offsets:
        dz=float(reference["height_offset_m"])+float(height)
        # Current pad bottom is 4.4 mm below the EEF in the upright grasp.
        # This is a source-head overlap prerequisite, not proof of gripping.
        head_overlap=max(0.,min(head_top,dz+.0116)-max(head_bottom,dz-.0044))
        for yaw in goals:
            R=down(float(yaw))
            for depth in depths:
                for extra in press_extras:
                    maximum=float(depth)+float(extra)
                    valid_depth=0<float(depth) and extra>=0 and maximum<=float(cad["pin_head_seated_total_depth_m"])+1e-9
                    records=[]
                    if valid_depth and head_overlap>0:
                        # Query the full insertion envelope to the single
                        # maximum total depth; opening and retreat start there.
                        for d in np.linspace(-.030,maximum,int(np.ceil((maximum+.030)/.002))+1):
                            origin=entry-outward*(float(d)+tip)
                            result=query.query(origin+np.array([0,0,dz]),R,gap,receivers)
                            records.append(dict(phase="insertion",depth_m=float(d),**result))
                        endpoint=entry-outward*(maximum+tip)+[0,0,dz]
                        for jaw in np.linspace(gap,.04,9):
                            result=query.query(endpoint,R,float(jaw),receivers)
                            records.append(dict(phase="opening",jaw_gap_m=float(jaw),**result))
                        for lift in np.linspace(0,.10,11):
                            result=query.query(endpoint+[0,0,lift],R,.04,receivers)
                            records.append(dict(phase="retraction",lift_m=float(lift),**result))
                    worst=min(records,key=lambda r:r["min_clearance_m"]) if records else None
                    clearance=worst["min_clearance_m"] if worst else None
                    passed=bool(valid_depth and head_overlap>0 and clearance is not None and clearance>=required)
                    uncertain=bool(valid_depth and head_overlap>0 and clearance is not None and 0<=clearance<required)
                    reason=("sampled future full-gripper clearance necessary condition passed" if passed else
                        "outside CAD depth envelope" if not valid_depth else "no pad/head axial overlap" if head_overlap<=0 else
                        "positive nominal clearance below visual uncertainty reserve" if uncertain else "future gripper sweep collision")
                    for source_yaw in source_yaws:
                        rows.append(dict(part=part,yaw=float(source_yaw),height=float(height),placement_yaw=float(yaw),
                            pin_command_depth_m=float(depth),pin_press_extra_m=float(extra),min_clearance_m=clearance,
                            required_clearance_m=required,status="necessary_pass" if passed else "unknown" if uncertain else "rejected",reason=reason,
                            known=not uncertain,goal_source=goal_source,source_grasp_ik_checked=False,
                            source_full_shell_approach_withdrawal_check_required=True,
                            checked_receivers=["end_stop","guide_base"],
                            other_installed_parts_checked=False,
                            pin_base_bridge_required=True,minimum_final_depth_m=minimum,
                            gravity_completion_required=bool(float(depth)<minimum),
                            head_pad_axial_overlap_m=head_overlap,hole_entry_m=entry.tolist(),axis=(-outward).tolist(),
                            sampled_worst_phase=worst["phase"] if worst else None,
                            worst_pairs=worst["pairs"][:4] if worst else [],samples=len(records),
                            source_pin_axis_world=source_R[:,2].tolist(),
                            gripper_geometry_sha256=query.provenance["panda_xml_sha256"],
                            limitations="near-upright down(yaw) grasp; discrete base/stop CAD query; carriage, other pins and full-scene safety require separate checks; not physical feasibility proof"))
    return rows
