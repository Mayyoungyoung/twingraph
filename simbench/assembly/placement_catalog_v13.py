"""Observed-scene non-pin grasp/placement necessary geometry catalogue.

Only immutable CAD and caller-supplied RGB-D/declared goal poses are accepted.
Queries use the complete Panda finger/palm/pad collision model without physics
steps. Positive sampled clearance is not IK, continuous collision freedom,
grasp retention, contact insertion, or task success.
"""
from copy import deepcopy
from functools import lru_cache
import math
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from .gripper_clearance_v12 import GripperClearance, PANDA, PADS, SHELLS, pose_matrix

PARTS = ("carriage", "end_stop", "handle")
PRECEDING = {"carriage": (), "end_stop": ("carriage",),
             "handle": ("carriage", "end_stop", "pin_left", "pin_right")}
GEOMETRY_ROUNDOFF_M = 1.e-10  # floating-point zero, not a physical contact allowance


def _down(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    # Identical to assembly.control.down: local closing x points along world
    # y at yaw zero. Keep this pure helper independent of a live MjContext.
    return np.asarray([[-s, c, 0.], [c, s, 0.], [0., 0., -1.]])


def _angle(value):
    return float(math.atan2(math.sin(value), math.cos(value)))


def _goal_pose(row):
    # A goal is explicitly marked as conditional; never turn it into a
    # perception observation or feed it to the live state estimator.
    return {**row, "valid": row.get("position_m") is not None and row.get("quat_wxyz") is not None}


def _line(start, end, step):
    start, end = np.asarray(start, float), np.asarray(end, float)
    n = max(1, int(np.ceil(np.linalg.norm(end-start)/step)))
    return [start+(end-start)*t for t in np.linspace(0., 1., n+1)]


def _receiver_scenes(observation, cad, part, completed, planned_before):
    primitives = cad.get("collision_primitives", {})
    objects = observation.get("objects", {})
    goals = observation.get("assembly_targets", {})
    source, future = {}, {}
    missing = []
    predecessors = set(PRECEDING[part] if planned_before is None else planned_before)
    predecessors |= set(completed).intersection(objects)
    predecessors.discard(part)
    for name in primitives:
        if name == part:
            continue  # Held-part contact is intentional, not receiver collision.
        observed = (observation.get("fixtures", {}).get(name, {}) if name == "guide_base"
                    else objects.get(name, {}))
        try:
            pose_matrix(observed)
            source[name] = deepcopy(observed)
        except ValueError:
            pass
        candidate = (_goal_pose(goals.get(name, {})) if name in predecessors and name not in completed
                     else observed)
        try:
            pose_matrix(candidate)
            future[name] = deepcopy(candidate)
        except ValueError:
            if name in predecessors or name == "guide_base":
                missing.append(f"future receiver pose unknown: {name}")
    for name in {"guide_base", *predecessors}:
        if not primitives.get(name):
            missing.append(f"receiver CAD missing: {name}")
    if "guide_base" not in source or "guide_base" not in future:
        missing.append("observed guide_base pose/CAD unavailable")
    return source, future, sorted(set(missing)), sorted(predecessors)


def _future_body_path(observation, part, target, step):
    if part == "carriage":
        rail = observation.get("receiver_geometry", {}).get("rail", {})
        if any(rail.get(k) is None for k in ("entry_m", "entry_approach_m", "axis")):
            raise ValueError("observed rail entry/approach/axis unavailable")
        entry, approach = np.asarray(rail["entry_m"], float), np.asarray(rail["entry_approach_m"], float)
        axis = np.asarray(rail["axis"], float)
        if any(v.shape != (3,) or not np.isfinite(v).all() for v in (entry, approach, axis)) or np.linalg.norm(axis) < 1.e-8:
            raise ValueError("invalid declared rail path")
        # Same object-referenced segments as stage_v5.stage_calls. Do not
        # silently substitute a fixed world-x entry or a straight target drop.
        lifted_entry, slide_target = entry+[0., 0., .0015], target+[0., 0., .0015]
        waypoints = [("descent", approach, approach-[0., 0., .030]),
                     ("rail_entry", approach-[0., 0., .030], lifted_entry),
                     ("rail_push", lifted_entry, slide_target),
                     ("seat", slide_target, target)]
    else:
        approach = target+[0., 0., .045]
        align = target+[0., 0., .010 if part == "end_stop" else .023]
        waypoints = [("descent", approach, align), ("guarded_seat", align, target)]
    return [(phase, p) for phase, start, end in waypoints for p in _line(start, end, step)]


@lru_cache(maxsize=1)
def _pad_corners():
    """Original XML box corners relative to grip_site, without a simulator."""
    root = ET.parse(PANDA).getroot().find(".//body[@name='right_gripper']")
    found, eef = {}, []
    def walk(body, position, rotation, is_root=False, slide_direction=None):
        slide_direction = np.zeros(3) if slide_direction is None else slide_direction.copy()
        if not is_root:
            offset = np.fromstring(body.get("pos", "0 0 0"), sep=" ")
            q = np.fromstring(body.get("quat", "1 0 0 0"), sep=" ")
            position = position+rotation@offset
            rotation = rotation@Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
        for joint in body.findall("joint"):
            if joint.get("type") == "slide":
                sign = -1. if joint.get("name") == "finger_joint2" else 1.
                slide_direction += sign*rotation@np.fromstring(joint.get("axis", "0 0 1"), sep=" ")
        for site in body.findall("site"):
            if site.get("name") == "grip_site":
                eef.append(position+rotation@np.fromstring(site.get("pos", "0 0 0"), sep=" "))
        for geom in body.findall("geom"):
            if geom.get("name") not in PADS:
                continue
            if geom.get("type", "sphere") != "box":
                raise ValueError("Panda pad shape changed; re-audit grasp face query")
            size = np.fromstring(geom.get("size"), sep=" ")
            center = np.fromstring(geom.get("pos", "0 0 0"), sep=" ")
            q = np.fromstring(geom.get("quat", "1 0 0 0"), sep=" ")
            local_R = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
            corners = np.asarray([[a, b, c] for a in (-size[0], size[0])
                                  for b in (-size[1], size[1]) for c in (-size[2], size[2])])
            found[geom.get("name")] = (position+(rotation@(center+(local_R@corners.T).T).T).T, slide_direction)
        for child in body.findall("body"):
            walk(child, position, rotation, slide_direction=slide_direction)
    walk(root, np.zeros(3), np.eye(3), True)
    if len(eef) != 1 or set(found) != set(PADS):
        raise ValueError("Panda pad/EEF interface changed")
    return tuple((found[name][0]-eef[0], found[name][1]) for name in PADS)


def _grasp_geometry(spec, source_R, pickup_R, dz):
    """CAD face overlap and width; no successful grip is inferred."""
    region = spec.get("grasp_region", {})
    if not region or "body_z_interval_m" not in region:
        return None, None, None, "CAD grasp face region not declared; reference-height proposal only"
    lo, hi = map(float, region["body_z_interval_m"])
    closing = source_R.T@pickup_R[:, 0]
    axes = region.get("closing_axes_body", ())
    width = None
    if region.get("rotational_symmetry") and axes:
        width = float(axes[0]["width_m"])
    elif axes:
        matches = [a for a in axes if abs(np.dot(np.asarray(a["axis"], float), closing)) > 1.-1.e-6]
        if matches:
            width = float(matches[0]["width_m"])
    if width is None:
        return None, None, None, "closing direction not represented by declared CAD grasp faces"
    overlaps = []
    for corners, slide in _pad_corners():
        corners = corners+slide*(width/2+.0005)
        body = (source_R.T@(np.array([0., 0., dz])+(pickup_R@corners.T).T).T).T
        overlaps.append(max(0., min(hi, float(body[:, 2].max()))-max(lo, float(body[:, 2].min()))))
    minimum_overlap = min(overlaps)
    if minimum_overlap <= 0:
        return False, width, minimum_overlap, "both original finger pads must overlap a declared CAD side face"
    return True, width, minimum_overlap, "original pad axial extent overlaps declared CAD grasp face"


def _source_body_result(result, phase, spec, source_position, source_R, pickup_R, width):
    """Exclude only nonpenetrating closing-pad pairs on declared side faces.

    The full open approach has no intentional contact exemption. Negative
    shell or pad distances remain failures, even on an intended grasp face.
    A held object is not tested against its stale source pose during lift.
    """
    region = spec.get("grasp_region", {})
    closing = source_R.T @ pickup_R[:, 0]
    pairs, intentional, unknown = [], [], []
    for original in result.get("pairs", []):
        pair = deepcopy(original)
        distance = float(pair["distance_m"])
        names = pair.get("geoms", [])
        pad_only = any(n in PADS for n in names) and not any(n in SHELLS for n in names)
        face = None
        if pair.get("declared_face_support") is not None:
            face=bool(pair["declared_face_support"])
        elif width is not None and region.get("body_z_interval_m") is not None and pair.get("contact_world_m") is not None:
            point = source_R.T @ (np.asarray(pair["contact_world_m"],float)-source_position)
            lo, hi = region["body_z_interval_m"]
            # Contact positions are midpoints between the surfaces. Before
            # touching, half of the positive distance separates that point
            # from the object's actual side face.
            face = bool(lo-GEOMETRY_ROUNDOFF_M <= point[2] <= hi+GEOMETRY_ROUNDOFF_M and
                abs(abs(float(point@closing))-width/2) <= max(distance,0.)/2+GEOMETRY_ROUNDOFF_M)
        allowed = bool(phase=="source_close" and pad_only and face and distance>=-GEOMETRY_ROUNDOFF_M)
        pair["intentional_grasp_pad_pair"] = allowed
        if allowed:
            intentional.append(pair)
        else:
            if abs(distance)<=GEOMETRY_ROUNDOFF_M:
                pair["distance_m"]=0.
            pairs.append(pair)
            if phase=="source_close" and pad_only and distance<=GEOMETRY_ROUNDOFF_M and face is None:
                unknown.append("closing-pad contact cannot be associated with a declared source grasp face")
            elif phase=="source_close" and pad_only and abs(distance)<=GEOMETRY_ROUNDOFF_M and face is False:
                # A contact on another part surface cannot establish the
                # declared grasp even if it is exactly tangent.
                pair["invalid_grasp_face_contact"] = True
    raw_min = float(result["min_clearance_m"])
    # A query reporting penetration without its pair details is never an
    # empty-contact exemption (also supports conservative injected queries).
    unexplained = raw_min < -GEOMETRY_ROUNDOFF_M and not result.get("pairs")
    minimum = min((float(p["distance_m"]) for p in pairs),default=.02)
    if unexplained:
        minimum=raw_min
    if not result.get("pairs") and not unexplained:
        minimum=max(0.,raw_min)
    return {**result, "min_clearance_m":minimum, "raw_min_clearance_m":raw_min,
        "pairs":pairs, "intentional_pad_pairs":intentional,
        "invalid_grasp_face_contact":any(p.get("invalid_grasp_face_contact",False) for p in pairs),
        "unknown_contact_reasons":unknown, "collision":bool(minimum<0.)}


def _obb_separation(center_a, axes_a, half_a, center_b, axes_b, half_b):
    """Exact box intersection SAT; positive separation is a distance bound."""
    axes=np.concatenate((axes_a.T,axes_b.T,
        np.cross(axes_a.T[:,None,:],axes_b.T[None,:,:]).reshape(-1,3)))
    norm=np.linalg.norm(axes,axis=1);axes=axes[norm>1.e-12]/norm[norm>1.e-12,None]
    separations=np.abs(axes@(center_a-center_b))-np.abs(axes@axes_a)@half_a-np.abs(axes@axes_b)@half_b
    return float(np.max(separations))


def _verified_source_pad_boxes(result, primitives, eef, pickup_R, jaw, source_position, source_R, spec, width):
    """Replace native box/box pad distances with analytic unchanged boxes.

    MuJoCo 2.3.7 can report deep penetration for coplanar pad/body boxes.
    Exact SAT also checks pairs missing from that contact list. Original
    distances remain diagnostics; all palm/finger mesh pairs are untouched.
    """
    boxes={p["name"]:p for p in primitives if p["type"]=="box"}
    original=result.get("pairs",[])
    pairs=[p for p in original if not(any(n in PADS for n in p["geoms"]) and
                                    any(n in boxes for n in p["geoms"]))]
    closing=source_R.T@pickup_R[:,0]
    region=spec.get("grasp_region",{}).get("body_z_interval_m")
    for pad_name,(corners,slide) in zip(PADS,_pad_corners()):
        world=np.asarray(eef)+(pickup_R@(corners+slide*jaw).T).T
        points=(source_R.T@(world-source_position).T).T
        pc=points.mean(axis=0)
        vectors=np.column_stack(((points[4]-points[0])/2,(points[2]-points[0])/2,(points[1]-points[0])/2))
        ph=np.linalg.norm(vectors,axis=0);pr=vectors/ph
        for name,box in boxes.items():
            bc=np.asarray(box["pos"],float);bh=np.asarray(box["size"],float)
            quat=np.asarray(box.get("quat_wxyz",[1,0,0,0]),float)
            br=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
            distance=_obb_separation(pc,pr,ph,bc,br,bh)
            if distance>.02:
                continue
            face=None;point=(pc+bc)/2
            if width is not None and region is not None:
                sign=1. if float((pc-bc)@closing)>=0 else -1.
                face_position=float(bc@closing+sign*(np.abs(closing@br)@bh))
                p_extent=float(np.abs(np.array([0.,0.,1.])@pr)@ph)
                b_extent=float(np.abs(np.array([0.,0.,1.])@br)@bh)
                overlap_lo=max(pc[2]-p_extent,bc[2]-b_extent)
                overlap_hi=min(pc[2]+p_extent,bc[2]+b_extent)
                face=bool(np.max(np.abs(closing@br))>1.-1.e-9 and
                    abs(abs(face_position)-width/2)<=GEOMETRY_ROUNDOFF_M and
                    overlap_hi-overlap_lo>GEOMETRY_ROUNDOFF_M and
                    region[0]-GEOMETRY_ROUNDOFF_M<=overlap_lo and overlap_hi<=region[1]+GEOMETRY_ROUNDOFF_M)
                point[2]=(overlap_lo+overlap_hi)/2
            pairs.append(dict(geoms=[pad_name,name],distance_m=distance,shell_involved=False,
                contact_world_m=(source_position+source_R@point).tolist(),
                declared_face_support=face,distance_method="analytic_OBB_SAT_same_original_pad_and_CAD_box",
                native_contact_distances_m=[p["distance_m"] for p in original if set(p["geoms"])=={pad_name,name}]))
    minimum=min((float(p["distance_m"]) for p in pairs),default=.02)
    return {**result,"mujoco_raw_min_clearance_m":float(result["min_clearance_m"]),
        "min_clearance_m":minimum,"pairs":sorted(pairs,key=lambda p:p["distance_m"]),
        "collision":bool(minimum<-GEOMETRY_ROUNDOFF_M)}


def placement_clearance_catalog(observation, cad, part, *, completed=(), planned_before=None,
        source_axis_offsets=(0., math.pi/2), height_offsets=(0., .004, .007),
        required_clearance_m=None, sample_step_m=.002, query_factory=None):
    """Rows compatible with choices[yaw,height,placement_yaw].

    Unknown geometry or positive clearance below the uncertainty reserve is
    retained for DT. Rejection requires a known collision or an explicitly
    violated CAD grasp-face declaration. Source IK is always unchecked.
    """
    if part not in PARTS:
        raise ValueError("non-pin catalogue supports carriage/end_stop/handle only")
    if not 0 < sample_step_m <= .005:
        raise ValueError("invalid declared geometric sampling interval")
    source_row = observation.get("objects", {}).get(part, {})
    target_row = observation.get("assembly_targets", {}).get(part, {})
    try:
        source_position, source_R = pose_matrix(source_row)
        target_position, target_R = pose_matrix(_goal_pose(target_row))
    except ValueError as exc:
        return [dict(part=part, yaw=None, height=None, placement_yaw=None, status="unknown",
            reason=str(exc), min_clearance_m=None, executable_parameters_available=False,
            source_grasp_ik_checked=False)]
    source_yaw = math.atan2(source_R[1, 0], source_R[0, 0])
    target_yaw = math.atan2(target_R[1, 0], target_R[0, 0])
    spec = cad.get("parts", {}).get(part, {})
    reference = spec.get("grasp_reference", {})
    if any(k not in reference for k in ("height_offset_m", "width_m")):
        return [dict(part=part, yaw=None, height=None, placement_yaw=None, status="unknown",
            reason="declared CAD grasp reference missing", min_clearance_m=None,
            executable_parameters_available=False, source_grasp_ik_checked=False)]
    source_scene, future_scene, common_unknowns, predecessors = _receiver_scenes(
        observation, cad, part, set(completed), planned_before)
    primitive_map = cad.get("collision_primitives", {})
    factory = GripperClearance if query_factory is None else query_factory
    queries = {phase: factory({p: primitive_map[p] for p in scene}) if scene else None
               for phase, scene in (("source", source_scene), ("future", future_scene))}
    source_body_scene={part:deepcopy(source_row)}
    source_body_available=bool(primitive_map.get(part))
    queries["source_body"]=factory({part:primitive_map[part]}) if source_body_available else None
    if not source_body_available:
        common_unknowns.append(f"source body CAD missing: {part}; self-clearance unverified")
    try:
        future_path = _future_body_path(observation, part, target_position, sample_step_m)
    except ValueError as exc:
        common_unknowns.append(str(exc)); future_path = []
    pose_rows = [source_row, observation.get("fixtures", {}).get("guide_base", {})]
    pose_rows += [observation.get("objects", {}).get(p, {}) for p in completed if p in predecessors]
    residuals = [float(r.get("fit_residual_m") or 0.) for r in pose_rows]
    if any(not np.isfinite(v) or v < 0 for v in residuals):
        raise ValueError("invalid visual uncertainty indicator")
    reserve = float(3*sum(max(.0005, v) for v in residuals) if required_clearance_m is None else required_clearance_m)
    if not np.isfinite(reserve) or reserve < 0:
        raise ValueError("invalid clearance uncertainty reserve")
    rows = []
    for angle in source_axis_offsets:
        yaw = _angle(source_yaw+float(angle))
        placement_yaw = _angle(target_yaw+float(angle))
        pickup_R, placement_R = _down(yaw), _down(placement_yaw)
        actual_target_R = placement_R@pickup_R.T@source_R
        orientation_residual = float(Rotation.from_matrix(target_R.T@actual_target_R).magnitude())
        for height in height_offsets:
            height = float(height)
            if not np.isfinite(height) or abs(height) > .025:
                raise ValueError("grasp height outside skill command envelope")
            dz = float(reference["height_offset_m"])+height
            legal, width, pad_overlap, legal_reason = _grasp_geometry(spec, source_R, pickup_R, dz)
            gap = float(width if width is not None else reference["width_m"])/2+.0005
            if not 0 < gap <= .04:
                raise ValueError("CAD grasp width outside actual jaw range")
            unknowns = list(common_unknowns)
            if legal is None:
                unknowns.append(legal_reason)
            if orientation_residual > 1.e-6:
                unknowns.append("yaw-only placement retains source roll/pitch; declared goal orientation not exactly represented")
            source_offset = np.array([0., 0., dz])
            # Actual yaw-only EEF change applied to the source holding offset.
            # For upright source/goal this also equals target_R@source_R.T@offset.
            target_offset = placement_R@pickup_R.T@source_offset
            source_eef = source_position+source_offset
            target_eef = target_position+target_offset
            records = []
            def query(phase, eef, rotation, jaw, scene_key):
                q = queries[scene_key]
                if q is None:
                    return
                scene = source_body_scene if scene_key=="source_body" else source_scene if scene_key == "source" else future_scene
                result = q.query(eef, rotation, float(jaw), scene)
                if scene_key=="source_body":
                    if isinstance(q,GripperClearance):
                        result=_verified_source_pad_boxes(result,primitive_map[part],eef,pickup_R,float(jaw),
                            source_position,source_R,spec,width)
                    result=_source_body_result(result,phase,spec,source_position,source_R,pickup_R,width)
                    unknowns.extend(result["unknown_contact_reasons"])
                records.append(dict(phase=phase, scene=scene_key, eef_position_m=np.asarray(eef).tolist(),
                    finger_gap_m=float(jaw), command_geometry_known=bool(width is not None or jaw >= .04-1.e-9), **result))
            if legal is not False:
                for p in _line(source_eef+[0., 0., .10], source_eef, sample_step_m):
                    query("source_approach", p, pickup_R, .04, "source")
                    query("source_approach", p, pickup_R, .04, "source_body")
                for jaw in np.linspace(.04, gap, 9):
                    query("source_close", source_eef, pickup_R, jaw, "source")
                    query("source_close", source_eef, pickup_R, jaw, "source_body")
                for p in _line(source_eef, source_eef+[0., 0., .10], sample_step_m):
                    query("source_withdrawal", p, pickup_R, gap, "source")
                for phase, origin in future_path:
                    query(phase, origin+target_offset, placement_R, gap, "future")
                for jaw in np.linspace(gap, .04, 9):
                    query("opening", target_eef, placement_R, jaw, "future")
                for p in _line(target_eef, target_eef+[0., 0., .10], sample_step_m):
                    query("retraction", p, placement_R, .04, "future")
            worst = min(records, key=lambda r: r["min_clearance_m"]) if records else None
            clearance = float(worst["min_clearance_m"]) if worst else None
            if legal is False:
                status, reason = "rejected", legal_reason
            elif any(r.get("invalid_grasp_face_contact") and r["command_geometry_known"] for r in records):
                status, reason = "rejected", "closing-pad contact outside declared source grasp faces"
            elif any(r["min_clearance_m"] < 0 and r["command_geometry_known"] for r in records):
                status, reason = "rejected", "known sampled full-gripper/CAD penetration"
            elif clearance is None:
                status, reason = "unknown", "no valid receiver CAD query"
            else:
                if clearance < reserve:
                    unknowns.append("nonnegative clearance below visual uncertainty reserve")
                status = "unknown" if unknowns else "necessary_pass"
                reason = "; ".join(sorted(set(unknowns))) if unknowns else "sampled source and future gripper necessary geometry passed"
            rows.append(dict(part=part, yaw=yaw, height=height, placement_yaw=placement_yaw,
                status=status, reason=reason, min_clearance_m=clearance, required_clearance_m=reserve,
                executable_parameters_available=True, known=status != "unknown", source_grasp_ik_checked=False,
                source_full_shell_approach_withdrawal_check_required=True,
                source_pose_yaw_rad=source_yaw, target_pose_yaw_rad=target_yaw,
                source_axis_offset_rad=float(angle), goal_orientation_residual_rad=orientation_residual,
                grasp_face_status="unknown" if legal is None else "inside" if legal else "outside",
                grasp_width_m=width, pad_face_axial_overlap_m=pad_overlap,
                source_body_geometry_available=source_body_available,
                source_body_checked=any(r["scene"]=="source_body" for r in records),
                source_body_checked_phases=sorted({r["phase"] for r in records if r["scene"]=="source_body"}),
                source_body_min_raw_clearance_m=min((r["raw_min_clearance_m"] for r in records if r["scene"]=="source_body"),default=None),
                source_body_min_clearance_m=min((r["min_clearance_m"] for r in records if r["scene"]=="source_body"),default=None),
                source_body_min_mujoco_clearance_m=min((r["mujoco_raw_min_clearance_m"] for r in records if "mujoco_raw_min_clearance_m" in r),default=None),
                source_body_invalid_grasp_face_contact=any(
                    r.get("invalid_grasp_face_contact",False) for r in records if r["scene"]=="source_body"),
                environment_min_clearance_m=min((r["min_clearance_m"] for r in records if r["scene"]!="source_body"),default=None),
                source_body_pad_box_method="analytic OBB intersection/separation from original pad corners and unchanged CAD; native distances retained as diagnostics",
                source_body_intentional_pad_pair_count=sum(len(r.get("intentional_pad_pairs",[])) for r in records),
                source_body_withdrawal_policy="rigid grasp preserves checked gripper/object transform; no stale-source self query during withdrawal; attachment and retention unverified",
                planned_predecessors=predecessors, source_receivers=sorted(source_scene), future_receivers=sorted(future_scene),
                future_receiver_pose_modes={p: "predicted_mated" if p in predecessors and p not in completed else "observed"
                    for p in future_scene},
                sampled_worst_phase=worst["phase"] if worst else None,
                worst_pairs=worst.get("pairs", [])[:4] if worst else [], samples=len(records),
                sampled_phases=sorted({r["phase"] for r in records}), sample_step_m=sample_step_m,
                source_eef_m=source_eef.tolist(), target_eef_m=target_eef.tolist(),
                geometry_provenance={p: q.provenance for p, q in queries.items() if q is not None},
                limitations="discrete CAD necessary query; source-body approach/closure checked when its CAD exists; transfer IK, continuous sweep, robot self-collision, held-part/environment collision, physical contact and grasp retention remain unverified"))
    return rows


placement_catalog = placement_clearance_catalog
