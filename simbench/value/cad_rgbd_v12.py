"""Upright printed-part localization from calibrated depth and static CAD.

No color identities, simulation poses, expected task targets, or object IDs
are inputs. Horizontal depth patches propose poses; complete source-STL
surfaces must agree with visible depth in calibrated views. Unknown remains
unknown when occlusion leaves insufficient geometric evidence.
"""
from functools import lru_cache
import math

import cv2
import numpy as np
from scipy import ndimage

from simbench.assembly.printed_kit import body_triangles, planning_metadata
from .rgbd_perception import _as_calibration, _backproject, _depth_m, _quat_from_yaw


def templates(installed=False):
    cad = planning_metadata()
    specs = {
        "guide_base": dict(top=.028, feature="rectangle", feature_size=[.018,.052],
                           feature_center_body_m=[.108,0.,.028]),
        "carriage": dict(top=.062, feature="disk", feature_size=[.011, .011], yaw_secondary=dict(z=.040, size=[.044, .028])),
        "end_stop": dict(top=.031, feature="rectangle", feature_size=[.022, .026]),
        "pin_left": dict(top=.013, feature="disk", feature_size=[.018, .018]),
        "pin_right": dict(top=.013, feature="disk", feature_size=[.018, .018]),
        "handle": dict(top=.008, feature="annulus", feature_size=[.042, .042]),
        "wipe_tool": dict(top=.069, feature="rectangle", feature_size=[.024, .028]),
    }
    for part, spec in specs.items():
        spec.update(cad_part=part, size_m=cad["parts"][part]["dimensions_m"],
                    body_bounds_m=cad["parts"][part]["body_bounds_m"],
                    workspace_bounds_m=[[-.55, -.52, .796], [.32, .32, .94]])
    return specs


@lru_cache(maxsize=16)
def _surface(part):
    triangles = body_triangles(part)
    samples, normals = [], []
    for tri in triangles:
        normal = np.cross(tri[1]-tri[0], tri[2]-tri[0])
        length = np.linalg.norm(normal)
        if length < 1e-14: continue
        normal /= length
        count = max(1, int(np.ceil(max(np.linalg.norm(tri[(i+1)%3]-tri[i]) for i in range(3)) / .002)))
        # Interior barycentric samples avoid normal ambiguity at CAD edges.
        for i in range(count):
            for j in range(count-i):
                a, b = (i + 1/3) / count, (j + 1/3) / count
                if a+b >= 1.: continue
                samples.append(tri[0] + a * (tri[1]-tri[0]) + b * (tri[2]-tri[0]))
                normals.append(normal)
    points, normals = np.asarray(samples), np.asarray(normals)
    if len(points) > 4500:
        indices = np.linspace(0, len(points)-1, 4500).astype(int)
        points, normals = points[indices], normals[indices]
    return points, normals


def _horizontal_points(frames, calibrations, bounds):
    clouds = []
    lo, hi = np.asarray(bounds[0]), np.asarray(bounds[1])
    for view, frame in frames.items():
        depth = _depth_m(frame.get("depth_m", frame.get("depth_mm")))
        cal = _as_calibration(calibrations[view])
        ys, xs = np.indices(depth.shape)
        xyz = _backproject(xs, ys, depth, cal)
        # Crossed central differences estimate visible surface normals in
        # world coordinates. Calibration defines gravity, not a part pose.
        dx = xyz[1:-1, 2:] - xyz[1:-1, :-2]
        dy = xyz[2:, 1:-1] - xyz[:-2, 1:-1]
        n = np.cross(dx, dy)
        norm = np.linalg.norm(n, axis=2)
        flat = np.abs(n[..., 2]) / np.maximum(norm, 1e-15) > .75
        valid = np.isfinite(depth[1:-1, 1:-1]) & (depth[1:-1, 1:-1] > .08)
        cloud = xyz[1:-1, 1:-1][flat & valid]
        cloud = cloud[((cloud > lo) & (cloud < hi)).all(axis=1)]
        clouds.append(cloud)
    return np.concatenate(clouds) if clouds else np.zeros((0, 3))


def _patches(points, resolution=.001):
    if len(points) < 8: return []
    # Modes in observed height, rather than hard-coded assembly levels.
    bins = np.rint(points[:, 2] / .0005).astype(int)
    values, counts = np.unique(bins, return_counts=True)
    ranked = sorted(zip(values, counts), key=lambda v: -v[1])
    modes = []
    for value, count in ranked:
        if count < 8: continue
        if all(abs(value-prev) > 2 for prev in modes): modes.append(int(value))
    result = []
    xy_min = points[:, :2].min(axis=0) - .005
    shape = np.ceil((points[:, :2].max(axis=0) - xy_min + .005)/resolution).astype(int) + 1
    for mode in modes:
        pts = points[np.abs(points[:, 2] - mode*.0005) <= .0008]
        if len(pts) < 8: continue
        xy = np.floor((pts[:, :2]-xy_min)/resolution).astype(int)
        mask = np.zeros(tuple(shape), np.uint8); mask[xy[:, 0], xy[:, 1]] = 1
        # Connect sparse oblique-view samples (roughly 3 mm apart). Fit uses
        # original metric points, never the dilated pixels or their extents.
        mask = ndimage.binary_dilation(mask, structure=np.ones((5,5)))
        labels, number = ndimage.label(mask)
        assigned = labels[xy[:, 0], xy[:, 1]]
        for label in range(1, number+1):
            patch = pts[assigned == label]
            if len(patch) < 8: continue
            rect = cv2.minAreaRect(patch[:, :2].astype(np.float32))
            center, sizes, angle = rect
            size = np.asarray(sizes, float)
            if size.min() < .004 or size.max() > .120: continue
            result.append(dict(points=patch, center=np.asarray(center), size=size,
                               yaw=math.radians(angle), z=float(np.median(patch[:, 2])), count=len(patch)))
    return result


def _rotate(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c,-s,0],[s,c,0],[0,0,1.]])


def _depth_agreement(part, position, yaw, frames, calibrations, projection_radius=0):
    local, normals = _surface(part)
    R = _rotate(yaw); world = local @ R.T + position; normals = normals @ R.T
    matches, missing, occluded, visible = 0, 0, 0, 0
    abs_errors = []
    pixel_uncertainties = []
    for view, frame in frames.items():
        depth = _depth_m(frame.get("depth_m", frame.get("depth_mm")))
        cal = _as_calibration(calibrations[view]); tf = cal.matrix()
        camera = (world - tf[:3,3]) @ tf[:3,:3]
        distance = -camera[:,2]
        facing = np.einsum("ij,ij->i", normals, tf[:3,3]-world) > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.rint(cal.fx * camera[:,0] / distance + cal.cx).astype(int)
            v = np.rint(-cal.fy * camera[:,1] / distance + cal.cy).astype(int)
        good = facing & (distance > .08) & (u>=0)&(v>=0)&(u<depth.shape[1])&(v<depth.shape[0])
        expected = distance[good]
        # A projected CAD sample covers a pixel footprint, not an infinitely
        # precise ray. At a silhouette nearest-pixel depth may belong to the
        # background. Search only the adjacent 1-pixel footprint, retaining
        # the metric 3 mm depth gate and reporting this sampling uncertainty.
        alternatives = np.stack([depth[np.clip(v[good]+dy,0,depth.shape[0]-1),
                                       np.clip(u[good]+dx,0,depth.shape[1]-1)]
                                 for dx in range(-projection_radius, projection_radius+1)
                                 for dy in range(-projection_radius, projection_radius+1)])
        difference_options = np.abs(alternatives - expected[None,:])
        difference_options[~np.isfinite(alternatives) | (alternatives <= .08)] = np.inf
        closest = np.argmin(difference_options, axis=0)
        observed = alternatives[closest,np.arange(len(expected))]
        pixel_uncertainties.extend((expected / min(cal.fx,cal.fy)).tolist())
        good_depth = np.isfinite(observed)&(observed>.08)
        difference = observed[good_depth] - expected[good_depth]
        # Foreground occlusion cannot confirm a model, but does not mean the
        # hidden part is absent. Missing foreground geometry is penalized.
        agree = np.abs(difference) <= .003
        matches += int(agree.sum()); missing += int((difference > .003).sum())
        occluded += int((difference < -.003).sum()); visible += len(difference)
        abs_errors.extend(np.abs(difference[agree]).tolist())
    tested = matches + missing
    return dict(matched_samples=matches, missing_samples=missing, occluded_samples=occluded,
                tested_samples=tested, match_fraction=matches/max(tested,1),
                visible_fraction=matches/max(visible,1),
                residual_m=float(np.median(abs_errors)) if abs_errors else None,
                projection_tolerance_px=projection_radius,
                projected_pixel_size_m=float(np.median(pixel_uncertainties)) if pixel_uncertainties else None)


def _refine(part, candidate, frames, calibrations):
    """Bounded image-depth registration around an observed plane hypothesis."""
    best = dict(candidate)
    best["score"] = best["agreement"]["match_fraction"] + .05 * best["agreement"]["visible_fraction"] - .12 * best["shape_error"]/.005
    if part in ("guide_base", "end_stop", "carriage", "wipe_tool"):
        initial_yaw = best["yaw"]
        for angle in np.linspace(-.18, .18, 13):
            yaw = initial_yaw + angle
            agreement = _depth_agreement(part, best["position"], yaw, frames, calibrations)
            score = agreement["match_fraction"] + .05 * agreement["visible_fraction"] - .12 * best["shape_error"]/.005
            if score > best["score"]: best.update(yaw=yaw, agreement=agreement, score=score)
    # Search is in measured coordinates, independent of task destinations.
    for step in (.001, .0004):
        origin = best["position"].copy()
        for dx in (-step, 0., step):
            for dy in (-step, 0., step):
                position = origin + [dx, dy, 0.]
                agreement = _depth_agreement(part, position, best["yaw"], frames, calibrations)
                # Absolute support avoids rewarding additional occlusion.
                score = agreement["match_fraction"] + .05 * agreement["visible_fraction"] - .12 * best["shape_error"]/.005
                previous = best["agreement"]["match_fraction"] + .05 * best["agreement"]["visible_fraction"] - .12 * best["shape_error"]/.005
                if score > previous:
                    best.update(position=position, agreement=agreement, score=score)
    return best


def _hypotheses(part, spec, patch, patches):
    features = [spec]
    if part == "carriage":
        features.append({**spec, "top": .040, "feature": "rectangle", "feature_size": [.044,.028]})
    for feature in features:
        nominal = np.asarray(feature["feature_size"])
        dims = patch["size"]
        shape_error = float(np.max(np.abs(np.sort(dims)-np.sort(nominal))))
        if shape_error > .005: continue
        position = np.r_[patch["center"], patch["z"] - feature["top"]]
        low = position[2] + spec["body_bounds_m"][0][2]
        if low < .795 or low > .87: continue
        if feature["feature"] in ("disk", "annulus"):
            yaws = [0.]
        else:
            yaw = patch["yaw"] + (0. if np.linalg.norm(dims-nominal) <= np.linalg.norm(dims[::-1]-nominal) else math.pi/2)
            yaws = [yaw, yaw+math.pi]
        if part == "carriage" and feature["feature"] == "disk":
            near = [p for p in patches if abs(p["z"]-(position[2]+.040)) < .002 and np.linalg.norm(p["center"]-position[:2]) < .012]
            if near:
                p = min(near,key=lambda p:np.max(np.abs(np.sort(p["size"])-np.array([.028,.044]))))
                yaw = p["yaw"] + (0. if p["size"][0] > p["size"][1] else math.pi/2)
                yaws = [yaw,yaw+math.pi]
        for yaw in yaws:
            reference = np.asarray(feature.get("feature_center_body_m", [0.,0.,feature["top"]]))
            pose = np.r_[patch["center"],patch["z"]] - _rotate(yaw) @ reference
            yield pose, yaw, shape_error


def _observed_transform(row):
    """Validated perception pose only; there is no nominal/identity fallback."""
    if not row or not row.get("valid") or row.get("position_m") is None or row.get("quat_wxyz") is None:
        return None
    position, q = np.asarray(row["position_m"],float), np.asarray(row["quat_wxyz"],float)
    if position.shape != (3,) or q.shape != (4,) or not np.isfinite(position).all() or not np.isfinite(q).all() or np.linalg.norm(q)<1.e-9:
        return None
    w,x,y,z=q/np.linalg.norm(q)
    rotation=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                       [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                       [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
    return position, rotation


def _clip_halfplane(polygon, normal, bound):
    if len(polygon)==0: return polygon
    out=[]
    for first,last in zip(polygon,np.roll(polygon,-1,axis=0)):
        a,b=float(np.dot(normal,first)-bound),float(np.dot(normal,last)-bound)
        if a <= 1.e-12: out.append(first)
        if (a < 0 < b) or (b < 0 < a):
            out.append(first+(last-first)*a/(a-b))
    return np.asarray(out,float).reshape(-1,2)


def shared_shaft_corridor(stop_entry, stop_rotation, base_entry, base_rotation, *,
                          stop_depth_m=.036, base_depth_m=.012, half_width_m=.004, shaft_radius_m=.0033):
    """Straight shaft route through both physical square bores, in base XY.

    A common axis need not pass through either hole centre. Intersect the
    cylinder-centre halfplanes at both faces of each bore. Oblique cylinder
    cross-section support is accounted for; this is a geometric estimate,
    not a contact or uncertainty certificate. No pose-error tolerance is used.
    """
    stop_entry,base_entry=np.asarray(stop_entry,float),np.asarray(base_entry,float)
    stop_rotation,base_rotation=np.asarray(stop_rotation,float),np.asarray(base_rotation,float)
    axis=base_rotation[:,2]; basis=base_rotation[:,:2]
    clearance=float(half_width_m-shaft_radius_m)
    if clearance <= 0: raise ValueError("shaft does not fit the declared aperture")
    polygon=np.array([[-clearance,-clearance],[clearance,-clearance],[clearance,clearance],[-clearance,clearance]])
    constraints=[]
    for entry,R,depth in ((base_entry,base_rotation,base_depth_m),(stop_entry,stop_rotation,stop_depth_m)):
        v=R.T@axis
        if v[2] <= 1.e-6:
            return dict(geometric_route_exists=False,reason="receiver axis is not traversable along base axis")
        origin=R.T@(base_entry-entry); mapping=R.T@basis
        for z in (0.,-float(depth)):
            slope=mapping[:2]-np.outer(v[:2]/v[2],mapping[2])
            constant=origin[:2]+v[:2]*(z-origin[2])/v[2]
            support=shaft_radius_m*np.sqrt(1+(v[:2]/v[2])**2)
            for i in range(2):
                room=half_width_m-support[i]
                for sign in (-1.,1.):
                    normal=sign*slope[i]; bound=room-sign*constant[i]
                    constraints.append((normal,bound))
                    polygon=_clip_halfplane(polygon,normal,bound)
    if len(polygon)<3:
        return dict(geometric_route_exists=False,reason="no common cylinder-centre corridor through the two square holes",
                    centre_polygon_base_xy_m=polygon.tolist())
    area=abs(float(np.sum(polygon[:,0]*np.roll(polygon[:,1],-1)-polygon[:,1]*np.roll(polygon[:,0],-1))))/2
    if area <= 1.e-16:
        return dict(geometric_route_exists=False,reason="only tangent/zero-area common route",centre_polygon_base_xy_m=polygon.tolist())
    # Maximise residual distance to the real aperture constraints, not to a
    # desired world pose. HiGHS only solves this small static 2-D geometry LP.
    from scipy.optimize import linprog
    matrix=np.asarray([np.r_[a,np.linalg.norm(a)] for a,b in constraints])
    bounds=np.asarray([b for a,b in constraints])
    solution=linprog([0.,0.,-1.],A_ub=matrix,b_ub=bounds,bounds=[(None,None),(None,None),(0,None)],method="highs")
    if not solution.success:
        return dict(geometric_route_exists=False,reason="common-route centre could not be certified")
    xy=solution.x[:2]
    # Rectangular corridors can have a segment of equally optimal centres.
    # Prefer their geometric centre instead of an arbitrary LP endpoint.
    centre=polygon.mean(axis=0)
    centre_margin=min((bound-float(np.dot(normal,centre)))/np.linalg.norm(normal)
                      for normal,bound in constraints if np.linalg.norm(normal)>1.e-12)
    if centre_margin >= solution.x[2]-1.e-10: xy=centre
    point=base_entry+basis@xy
    return dict(geometric_route_exists=True,reason="nonempty shaft-centre corridor through both complete bores",
        axis=axis.tolist(),common_axis_point_m=point.tolist(),centre_polygon_base_xy_m=polygon.tolist(),
        geometric_margin_m=float(solution.x[2]),corridor_area_m2=area,
        corridor_model="base-axis straight cylinder, both faces of true square apertures")


def receiver_interface(objects, fixtures, cad=None):
    """Pure RGB-D/CAD receiver targets and downstream geometric relationships.

    Desired mating poses are CAD goals in an *observed* base frame, not
    claims that source objects are already assembled. A later observed
    end-stop/base relationship is a distinct insertion precondition.
    """
    cad=cad or planning_metadata()
    base_row=fixtures.get("guide_base",{})
    base=_observed_transform(base_row)
    unknown=dict(observable=False,geometric_route_exists=False,requires_guarded_verification=True,
                 reason="guide_base RGB-D/CAD pose unknown",holes={})
    if base is None:
        return dict(assembly_targets={},receiver_geometry={},fixture_relations={"end_stop_to_base":unknown})
    bp,bR=base; base_quat=list(base_row["quat_wxyz"])
    transform=lambda local:(bp+bR@np.asarray(local,float)).tolist()
    holes=[dict(id=name,entry_m=transform(offset),axis=bR[:,2].tolist(),transverse_axes=bR[:,:2].T.tolist(),
                depth_m=cad["base_receiver_depth_m"],half_width_m=cad["pin_hole_width_m"]/2)
           for name,offset in zip(("pin_left","pin_right"),cad["base_hole_offsets_m"])]
    rail=cad["rail_geometry"]
    receiver=dict(base_holes=holes,rail=dict(entry_m=transform(rail["entry_center_local_m"]),
        entry_face_m=transform(rail["entry_face_local_m"]),entry_approach_m=transform(rail["entry_approach_local_m"]),
        axis=(bR@np.asarray(rail["axis_local"])).tolist(),
        carriage_target_m=transform(cad["carriage_mating_pose_in_base"]["position_m"])))
    targets={}
    for part,key in (("carriage","carriage_mating_pose_in_base"),("end_stop","end_stop_mating_pose_in_base")):
        targets[part]=dict(position_m=transform(cad[key]["position_m"]),quat_wxyz=base_quat,
            source="observed_guide_base+static_CAD_mating",receiver="guide_base",goal_only=True)
    targets["handle"]=dict(position_m=(np.asarray(targets["carriage"]["position_m"])+bR[:,2]*cad["handle_seat_center_offset_from_carriage_m"]).tolist(),
        quat_wxyz=base_quat,source="observed_guide_base+planned_carriage+static_CAD_mating",receiver="carriage",goal_only=True,
        execution_rebinding="observed seated carriage post before handle placement")
    planned_stop=np.asarray(targets["end_stop"]["position_m"])
    for name,offset in zip(("pin_left","pin_right"),cad["pin_hole_offsets_m"]):
        entry=planned_stop+bR@np.asarray(offset)
        final_depth=cad["pin_bridge_minimum_total_depth_m"]
        shallow_depth=float(cad["legacy_isolated_stop_command_depth_m"])
        tip_offset=float(cad["pin_shaft_offsets_m"][0])
        targets[name]=dict(position_m=(entry-bR[:,2]*(final_depth+tip_offset)).tolist(),quat_wxyz=base_quat,
            source="observed_guide_base+planned_two_layer_CAD_holes",receiver="end_stop+guide_base",goal_only=True,
            hole_entry_m=entry.tolist(),axis=bR[:,2].tolist(),minimum_total_depth_m=final_depth,
            shallow_release_position_m=(entry-bR[:,2]*(shallow_depth+tip_offset)).tolist(),
            shallow_release_scope="legacy isolated stop-only reference, not a complete-task bridge command",
            execution_rebinding="observed shared end_stop/base shaft corridor; controller chooses collision-safe release")
    stop_row=objects.get("end_stop",{}); stop=_observed_transform(stop_row)
    if stop is None:
        unknown["reason"]="end_stop RGB-D/CAD pose unknown"
        relation=unknown
    else:
        sp,sR=stop; records={}
        error_bound=sum(float(r.get("fit_residual_m") or 0.) for r in (base_row,stop_row))
        error_bound+=sum(float(r.get("geometry_agreement",{}).get("projected_pixel_size_m") or 0.)/2 for r in (base_row,stop_row))
        for name,offset,base_hole in zip(("pin_left","pin_right"),cad["pin_hole_offsets_m"],holes):
            entry=sp+sR@np.asarray(offset); base_entry=np.asarray(base_hole["entry_m"])
            route=shared_shaft_corridor(entry,sR,base_entry,bR,stop_depth_m=cad["pin_receiver_depth_m"],
                base_depth_m=cad["base_receiver_depth_m"],half_width_m=cad["pin_hole_width_m"]/2,shaft_radius_m=cad["pin_shaft_radius_m"])
            dz=float(np.dot(entry-base_entry,bR[:,2]))
            minimum=dz+cad["pin_base_minimum_depth_m"]
            maximum=float(cad["pin_head_underside_body_z_m"]-cad["pin_shaft_offsets_m"][0])
            length_ok=bool(0 < dz and minimum <= maximum)
            route.update(stop_entry_m=entry.tolist(),base_entry_m=base_entry.tolist(),
                stop_axis=sR[:,2].tolist(),base_axis=bR[:,2].tolist(),
                stop_to_base_gap_m=dz-cad["pin_receiver_depth_m"],minimum_total_depth_m=minimum,
                head_seated_depth_limit_m=maximum,shaft_length_sufficient=length_ok,
                registration_uncertainty_indicator_m=error_bound,
                source="two observed receiver poses + unchanged square-hole CAD")
            route["geometric_route_exists"]=bool(route["geometric_route_exists"] and length_ok)
            if not length_ok:route["reason"]="observed layer order/separation exceeds usable shaft reach"
            route["uncertainty_clearance_certified"]=bool(route["geometric_route_exists"] and route.get("geometric_margin_m",0)>error_bound)
            if route["geometric_route_exists"]:
                axis=np.asarray(route["axis"]);point=np.asarray(route["common_axis_point_m"])
                along=float(np.dot(sR[:,2],entry-point)/np.dot(sR[:,2],axis))
                tip_offset=float(cad["pin_shaft_offsets_m"][0])
                shallow_depth=float(cad["legacy_isolated_stop_command_depth_m"])
                route.update(stop_entry_on_common_axis_m=(point+axis*along).tolist(),
                    bridge_goal_position_m=(point-axis*(cad["pin_base_minimum_depth_m"]+tip_offset)).tolist(),
                    shallow_release_position_m=(point+axis*(along-shallow_depth-tip_offset)).tolist(),
                    shallow_release_scope="legacy isolated stop-only reference, not a complete-task bridge command",
                    minimum_total_depth_m=along+cad["pin_base_minimum_depth_m"])
            records[name]=route
        relation=dict(observable=True,geometric_route_exists=all(r["geometric_route_exists"] for r in records.values()),
            requires_guarded_verification=True,
            reason="perception-derived two-layer route estimate; actual contact/capture and bridge require independent checks",
            holes=records,measurement_source="wrist_RGBD_depth+registered_static_CAD",
            contact_or_capture_verified=False,pose_error_threshold_used=False)
    return dict(assembly_targets=targets,receiver_geometry=receiver,fixture_relations={"end_stop_to_base":relation})


def estimate_scene(rgbd_frames, camera_calibration, object_templates=None, previous_estimates=None):
    object_templates = object_templates or templates()
    points = _horizontal_points(rgbd_frames, camera_calibration, [[-.55,-.52,.803],[.32,.32,.94]])
    patches = _patches(points)
    all_candidates = {}
    for part, spec in object_templates.items():
        candidates = []
        for patch in patches:
            for position, yaw, shape_error in _hypotheses(part, spec, patch, patches):
                result = _depth_agreement(part, position, yaw, rgbd_frames, camera_calibration)
                score = result["match_fraction"] - .12 * shape_error/.005
                candidates.append(dict(position=position, yaw=yaw, score=score, shape_error=shape_error, patch=patch, agreement=result))
        candidates.sort(key=lambda x:-x["score"])
        candidates = [_refine(part, row, rgbd_frames, camera_calibration) for row in candidates[:4]]
        candidates.sort(key=lambda x:-x["score"])
        for candidate in candidates:
            # Keep center-ray registration precise. Only the final validity
            # test accounts for the silhouette's one-pixel sampling footprint.
            candidate["registration_agreement"] = candidate["agreement"]
            candidate["agreement"] = _depth_agreement(part, candidate["position"], candidate["yaw"],
                rgbd_frames, camera_calibration, projection_radius=1)
        all_candidates[part] = candidates
    # Pins are geometrically identical. Their anonymous depth tracks are
    # associated by observed Y ordering, never by simulation instance IDs.
    pins = all_candidates.get("pin_left", all_candidates.get("pin_right", []))
    unique_pins = []
    for row in pins:
        if row["agreement"]["match_fraction"] < .83: continue
        if all(np.linalg.norm(row["position"]-p["position"]) > .015 for p in unique_pins): unique_pins.append(row)
    if "pin_left" in all_candidates and "pin_right" in all_candidates:
        names = ["pin_left", "pin_right"]
        associations = {}; remaining = list(unique_pins[:2])
        # Preserve an unchanged anonymous visual track first. A globally
        # nearest assignment can swap identical pins when the other moves
        # from its supply holder to the opposite side of the installed pair.
        for name in names:
            old = (previous_estimates or {}).get(name, {})
            if old.get("valid") and old.get("position_m") is not None and remaining:
                nearest = min(remaining, key=lambda r:np.linalg.norm(r["position"]-old["position_m"]))
                if np.linalg.norm(nearest["position"]-old["position_m"]) < .008:
                    associations[name] = nearest; remaining = [row for row in remaining if row is not nearest]
        missing = [n for n in names if n not in associations]
        if len(missing) == 1 and len(remaining) == 1:
            associations[missing[0]] = remaining[0]
        elif len(missing) == 2 and len(remaining) == 2:
            pair = sorted(remaining, key=lambda r:-r["position"][1])
            associations.update(dict(zip(names,pair)))
        for name in names: all_candidates[name] = [associations[name]] if name in associations else []
    objects = {}
    for part, candidates in all_candidates.items():
        best = candidates[0] if candidates else None
        agreement = best["agreement"] if best else {}
        valid = bool(best and agreement["match_fraction"] >= .83 and agreement["matched_samples"] >= 35 and agreement["visible_fraction"] >= .08)
        position = best["position"].tolist() if valid else None
        # The base is front/back asymmetric; do not erase its pi ambiguity.
        period=2*math.pi if part=="guide_base" else math.pi
        yaw = ((best["yaw"] + period/2) % period - period/2) if best else 0.
        objects[part] = dict(position_m=position, quat_wxyz=_quat_from_yaw(yaw) if valid else None,
            valid=valid, quality=float(best["score"]) if valid else 0., fit_residual_m=agreement.get("residual_m"),
            bbox_xyxy=None, mask_area_px=0,
            source_view=next(iter(rgbd_frames)) if len(rgbd_frames) == 1 else "multi_view_depth_CAD",
            occluded=not valid,
            track_id=f"cad-depth:{part}:0", category=part, geometry_agreement=agreement,
            diagnostic_hypotheses=[dict(position_m=r["position"].tolist(), yaw=r["yaw"], score=r["score"],
                                        shape_error_m=r["shape_error"], agreement=r["agreement"]) for r in candidates[:5]])
    fixtures={"guide_base":objects.pop("guide_base")} if "guide_base" in objects else {}
    interface=receiver_interface(objects,fixtures)
    return dict(schema="twingraph.rgbd_geometry.v1", backend="rgbd_geometry", detector="cad_geometry_v12",
        fixtures=fixtures,**interface,
        objects=objects, views=list(rgbd_frames), source="depth_pixels_calibration_static_STL_geometry",
        color_identity_used=False, simulator_pose_used=False, history_used=previous_estimates is not None,
        identity_association="identical-pin visual tracks; unchanged observed tracks preserved before moving track",
        assumptions="upright known printed parts; calibrated world gravity/workspace; uncertainty returns unknown",
        horizontal_patch_count=len(patches), detector_resolution={v:[np.asarray(f["rgb"]).shape[1],np.asarray(f["rgb"]).shape[0]] for v,f in rgbd_frames.items()})
