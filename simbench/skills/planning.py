"""Path-planning skill: waypoint styles for EEF moves.

Styles (pluggable, aligned with the old P1-P5 / A1-A5 planner semantics):

  direct    single hop to the goal (short moves in free space)
  safe_z    lift to a relative clearance above both endpoints, travel in
            xy, descend (the default for inter-object transfers)
  clearance travel at an absolute z height (e.g. above the tray walls)

``plan_path`` returns a :class:`PathPlan` -- waypoints plus a validity
flag and a failure reason, so the planner action (``plan_path`` in the
executor) can report WHY a plan failed (unreachable / collision) while
the execution action (``move_to``) simply consumes the waypoints.  The
old list-iteration interface is preserved via ``__iter__``.
"""
import numpy as np


class PathPlan:
    """A planned waypoint sequence + validity + failure reason."""

    def __init__(self, waypoints, style, valid=True, reason="ok"):
        self.waypoints = [np.asarray(w, dtype=float) for w in waypoints]
        self.style = style
        self.valid = bool(valid)
        self.reason = reason

    def __iter__(self):
        return iter(self.waypoints)

    def __len__(self):
        return len(self.waypoints)

    def __getitem__(self, i):
        return self.waypoints[i]

    def as_dict(self):
        return dict(waypoints=[w.tolist() for w in self.waypoints],
                    style=self.style, valid=self.valid,
                    reason=self.reason)


def plan_path(cur, goal, style="safe_z", lift=0.08, safe_z=None):
    """Waypoint sequence from ``cur`` to ``goal`` (both world xyz)."""
    cur = np.asarray(cur, dtype=float)
    goal = np.asarray(goal, dtype=float)
    if style == "direct":
        return PathPlan([goal], style)
    if style == "safe_z":
        z = max(cur[2], goal[2]) + lift
        return PathPlan([np.array([cur[0], cur[1], z]),
                         np.array([goal[0], goal[1], z]),
                         goal], style)
    if style == "clearance":
        z = safe_z if safe_z is not None else max(cur[2], goal[2]) + lift
        return PathPlan([np.array([cur[0], cur[1], z]),
                         np.array([goal[0], goal[1], z]),
                         goal], style)
    raise ValueError(f"unknown plan style {style!r}")


# ------------------------------------------------- candidate generation
# Planning skills generate MULTIPLE candidate branches (grasp poses /
# waypoint paths), each scored; the executor filters by the collision
# gate and picks the best feasible one.  The single-plan interfaces
# above stay as the legacy API (motion.move_eef consumes them).

def path_length(waypoints):
    """Total Euclidean length of a waypoint sequence (m)."""
    return float(sum(np.linalg.norm(np.asarray(waypoints[i + 1])
                                    - np.asarray(waypoints[i]))
                     for i in range(len(waypoints) - 1)))


def path_candidates(cur, goal, style="safe_z", lift=0.08, safe_z=None,
                    via_offsets=(0.0, 0.03, -0.03)):
    """Generate scored candidate waypoint plans for one *style*.

    Candidate space (safe_z/clearance): height variants {base-0.04,
    base, base+0.04} x lateral via-point offsets at the travel height.
    The FINAL DESCENT COLUMN is always the goal xy, so the descent-leg
    collision semantics are identical to the single-plan interface
    (the collision gate in skills.planning.check_path stays the
    caller's responsibility).  ``direct`` yields a single candidate.

    Score (lower = better): total path length + 0.02 per waypoint.
    Returns the candidate list sorted best-first; each PathPlan carries
    a ``score`` attribute.
    """
    cur = np.asarray(cur, dtype=float)
    goal = np.asarray(goal, dtype=float)
    if style == "direct":
        p = PathPlan([goal], style)
        p.score = path_length([cur, goal]) + 0.02
        return [p]
    if style not in ("safe_z", "clearance"):
        raise ValueError(f"unknown plan style {style!r}")
    if safe_z is not None:
        heights = [float(safe_z)]
    else:
        base = max(cur[2], goal[2]) + lift
        heights = [base - 0.04, base, base + 0.04]
    d = goal[:2] - cur[:2]
    n = float(np.linalg.norm(d))
    out = []
    for z in heights:
        for off in via_offsets:
            if off == 0.0 or n < 1e-9:
                wps = [np.array([cur[0], cur[1], z]),
                       np.array([goal[0], goal[1], z]), goal]
            else:
                perp = np.array([-d[1], d[0]]) / n
                via = 0.5 * (cur[:2] + goal[:2]) + perp * off
                wps = [np.array([cur[0], cur[1], z]),
                       np.array([via[0], via[1], z]),
                       np.array([goal[0], goal[1], z]), goal]
            p = PathPlan(wps, style)
            p.score = path_length(wps) + 0.02 * len(wps)
            out.append(p)
    out.sort(key=lambda p: p.score)
    return out


def grasp_pose_candidates(ctx, name, meta, detect_result=None,
                          noise_std=0.0, rng=None, n_yaw=8,
                          obstacles=None, margin=0.005):
    """Generate + score grasp-pose candidates for *name*.

    Candidates: n_yaw approach angles.  n_yaw=2 reproduces exactly the
    legacy two-axis candidate set (0, pi/2) of
    perception.estimate_grasp_pose; larger n_yaw adds intermediate
    approach angles for the planner to choose from.

    Each candidate carries the full grasp-artifact keys (pos/yaw/
    outer_d/type/grasp_dz/half_h/confidence) plus score/feasible/
    reason.

    Scoring (higher = better): -travel distance from the current EEF
    - yaw-mismatch penalty - approach-line collision penalty - unknown
    type penalty.  The best candidate sorts first.
    """
    if detect_result is None:
        from .perception import detect
        pos, yaw = detect(ctx, name, noise_std=noise_std, rng=rng)
    else:
        found, pos, yaw, _ = detect_result
        if not found:
            return []
    pos = np.asarray(pos, dtype=float)
    eef = ctx.eef_pos()
    if n_yaw == 2:
        yaws = [0.0, np.pi / 2.0]
    else:
        yaws = list(np.linspace(0.0, np.pi, max(2, n_yaw) + 1)[:-1])
        for ax in (0.0, np.pi / 2.0):
            if ax not in yaws:
                yaws.append(ax)
    cands = []
    for ay in yaws:
        gp = np.array([pos[0], pos[1], pos[2] + meta["grasp_dz"]])
        score = (-float(np.linalg.norm(gp[:2] - eef[:2]))
                 - 0.05 * abs(_wrap_pi(ay - yaw)))
        feasible = True
        reason = "ok"
        # approach-line collision: the hover->grasp descent (8cm above
        # the grasp target) must stay clear of static obstacles
        if obstacles:
            hover = gp + np.array([0.0, 0.0, 0.08])
            for ob in obstacles:
                if seg_aabb_hit(hover, gp, ob, margin=margin):
                    feasible = False
                    reason = f"approach collides with {ob['name']}"
                    score -= 100.0
                    break
        if meta["type"] not in ("cylinder", "ring", "box", "pin"):
            feasible = False
            reason = f"unknown part type {meta['type']!r}"
            score -= 50.0
        cands.append(dict(pos=gp, yaw=float(ay),
                          outer_d=meta["outer_d"], type=meta["type"],
                          grasp_dz=meta["grasp_dz"],
                          half_h=meta["half_h"], confidence=1.0,
                          score=float(score), feasible=bool(feasible),
                          reason=reason))
    cands.sort(key=lambda c: c["score"], reverse=True)
    return cands


def _wrap_pi(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


# ------------------------------------------------------------- collision

def seg_aabb_hit(p0, p1, aabb, margin=0.0):
    """Slab test: does segment p0->p1 intersect the axis-aligned box
    ``aabb`` = {"lo": (3,), "hi": (3,)} (optionally inflated by
    ``margin``)?"""
    lo = np.asarray(aabb["lo"], dtype=float) - margin
    hi = np.asarray(aabb["hi"], dtype=float) + margin
    p0 = np.asarray(p0, dtype=float)
    d = np.asarray(p1, dtype=float) - p0
    tmin, tmax = 0.0, 1.0
    for i in range(3):
        if abs(d[i]) < 1e-12:
            if p0[i] < lo[i] or p0[i] > hi[i]:
                return False
        else:
            t1 = (lo[i] - p0[i]) / d[i]
            t2 = (hi[i] - p0[i]) / d[i]
            if t1 > t2:
                t1, t2 = t2, t1
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmin > tmax:
                return False
    return True


def check_path(waypoints, obstacles, margin=0.005):
    """Collision-check a waypoint sequence against static obstacles.

    Returns a list of (segment_index, obstacle_name) hits; empty list =
    collision-free.  Every consecutive waypoint pair is checked.
    """
    hits = []
    n = len(waypoints)
    for i in range(n - 1):
        for ob in obstacles:
            if seg_aabb_hit(waypoints[i], waypoints[i + 1], ob,
                            margin=margin):
                hits.append((i, ob["name"]))
    return hits
