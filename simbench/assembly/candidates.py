"""Bound alternatives for a shared task template, without live rollouts.

Only the current prefix is materialized. A carrying trajectory is generated
after grasp feedback supplies the actual hand/object transform; it is never
borrowed from a different grasp branch. Future stages remain explicit slots.
"""

from dataclasses import dataclass, asdict
import copy
import hashlib
import json
import numpy as np
import mujoco
from .control import HOME, down


def fingerprint(session):
    return hashlib.sha256(
        session.ctx.data.qpos.tobytes()
        + session.ctx.data.qvel.tobytes()
        + session.ctx.data.ctrl.tobytes()
        + session.ctx.model.geom_size.tobytes()
        + session.ctx.model.geom_pos.tobytes()
        + session.ctx.model.geom_friction.tobytes()
    ).hexdigest()


def release_space_proxy(session, part, grasp, terminal):
    """Cheap terminal jaw-sweep penalty in scratch data, not a rollout label.

    The proposed object translation and nominal grasp offset are assumptions.
    Collision findings therefore penalize ranking without deleting the branch.
    """
    if "xyz" not in terminal:
        return dict(status="unknown", penalty=0.0, reason="terminal pose unspecified")
    ctx, m = session.ctx, session.ctx.model
    target = np.asarray(terminal["xyz"], float)
    offset = grasp["xyz"] - ctx.obj_pos(part)
    try:
        q = session.arm.ik(target + offset, down(grasp["yaw"]))
    except ValueError as exc:
        return dict(status="unknown", penalty=0.0, reason=str(exc))
    d = mujoco.MjData(m)
    d.qpos[:] = ctx.data.qpos
    d.qpos[ctx.arm_qadr] = q
    bid = ctx.body_id(part)
    qa = m.jnt_qposadr[m.body_jntadr[bid]]
    d.qpos[qa : qa + 3] = target
    fingers = {i for i in range(m.nbody) if "finger" in m.body(i).name}
    penetration, pairs = 0.0, set()
    for gap in np.linspace(grasp["width"] / 2, 0.04, 9):
        d.qpos[ctx.finger_qadr] = [gap, -gap]
        mujoco.mj_forward(m, d)
        for c in d.contact:
            b1, b2 = map(int, m.geom_bodyid[[c.geom1, c.geom2]])
            if (b1 in fingers) == (b2 in fingers) or bid in (b1, b2):
                continue
            if c.dist < -0.0008:
                penetration += -float(c.dist)
                pairs.add((m.geom(c.geom1).name, m.geom(c.geom2).name))
    return dict(
        status="proxy",
        penalty=1000.0 * penetration,
        summed_penetration_m=penetration,
        pairs=sorted(pairs),
        samples=9,
    )


def transfer_route_points(start, target, height):
    """Route through the requested world-height plane, not the HOME height.

    The Panda HOME pose is deliberately high.  Carrying that Z unchanged
    during a long lateral move can make an otherwise reachable supply pose
    fail IK.  Descending or rising at the current XY before translating keeps
    the declared clearance semantics and is checked like every other segment.
    """
    start=np.asarray(start,float);target=np.asarray(target,float);height=float(height)
    if start.shape!=(3,) or target.shape!=(3,) or not np.isfinite([*start,*target,height]).all():
        raise ValueError("transfer route requires finite 3D endpoints and height")
    plane=max(height,float(target[2]))
    return [np.r_[start[:2],plane],np.r_[target[:2],plane],target.copy()]


def transfer_routes(
    session, target, clearance=0.98, yaw=0.0, grasp=None, grasp_artifact="grasp"
):
    """Enumerate every configured route; collision witnesses reject only that route."""
    start = session.ctx.eef_pos().copy()
    target = np.asarray(target, float)
    binding = dict(
        held=session.held,
        grasp_artifact=grasp_artifact if grasp else None,
        grasp_id=grasp.get("id") if grasp else None,
        grasp_epoch=session.grasp_epoch,
        prefix_id=session.active_candidate_id,
    )
    routes, checks = [], []
    for index, height in enumerate((clearance, clearance + 0.07, clearance + 0.13)):
        route = dict(
            id=f"{binding['grasp_id'] or 'free'}:route:{index}",
            clearance=float(height),
            binding=copy.deepcopy(binding),
            target=target.copy(),
            yaw=float(yaw),
            status="unknown",
        )
        points = transfer_route_points(start,target,height)
        joints = []
        try:
            # Explore a fixed budget of arm postures for redundant IK. Every
            # alternative is checked against the same collision constraints.
            rng = np.random.default_rng(17)
            seeds = [session.ctx.arm_qpos.copy()] + [
                np.clip(HOME + rng.normal(0, .3, 7),
                        session.arm.limits[:, 0] + .02,
                        session.arm.limits[:, 1] - .02)
                for _ in range(12)
            ]
            if (getattr(session, "strict_rgbd_v12", False) and grasp
                    and "approach_joints_v12" in grasp):
                seeds = [np.asarray(grasp["q_hover"]).copy()] + session.arm.restart_seeds(24)
            verdict = dict(valid=None, status="unknown")
            for restart, seed in enumerate(seeds):
                q = seed.copy()
                attempt_joints = []
                try:
                    for point_index, point in enumerate(points):
                        bound_hover = (getattr(session, "strict_rgbd_v12", False) and not session.held
                            and grasp and "approach_joints_v12" in grasp
                            and np.linalg.norm(target-(np.asarray(grasp["xyz"])+[0,0,.10])) < 1e-6)
                        q = (np.asarray(grasp["q_hover"]).copy() if bound_hover and point_index==2
                             else session.arm.ik(point, down(yaw), seed=q))
                        attempt_joints.append(q.copy())
                    candidate_verdict = session.arm.check_joint_path(attempt_joints, session.held)
                except ValueError as exc:
                    candidate_verdict = dict(valid=None, reason=str(exc), status="unknown")
                verdict = candidate_verdict
                if candidate_verdict.get("valid"):
                    joints = attempt_joints
                    verdict = dict(candidate_verdict, ik_restart=restart)
                    break
            route["status"] = ("necessary_pass" if verdict["valid"] else
                               "unknown" if verdict["valid"] is None else "conflict")
            route["cost"] = float(
                height
                + 0.001
                * sum(
                    np.linalg.norm(b - a)
                    for a, b in zip([session.ctx.arm_qpos] + joints[:-1], joints)
                )
            ) if joints else float("inf")
            route["path"] = dict(
                type="joint_path",
                part=session.held,
                id=route["id"],
                start_q=session.ctx.arm_qpos.copy(),
                joints=joints,
                target=target.copy(),
                rotation=down(yaw),
                binding=copy.deepcopy(binding),
            ) if joints else None
        except ValueError as exc:
            verdict = dict(valid=None, reason=str(exc), status="unknown")
        route["check"] = verdict
        route["unknown"] = ["continuous swept collision", "closed-loop execution"]
        routes.append(route)
        checks.append(dict(route_id=route["id"], **verdict))
    return routes, checks


@dataclass
class Candidate:
    id: str
    part: str
    start_state: str
    grasp: dict
    path: dict | None
    control: dict
    terminal: dict
    bindings: dict
    status: str
    unknown: list
    cost: float
    steps: list

    def to_dict(self):
        return asdict(self)


def pick_template(part, lift=True):
    """Expandable macro, not another primitive or another controller."""
    rows = [
        dict(skill="move", params=dict(path="transfer")),
        dict(skill="move", params=dict(grasp="grasp", part=part)),
        dict(skill="grasp", params=dict(part=part)),
        dict(skill="inspect", params=dict(what="grasp", part=part)),
    ]
    if lift:
        rows.append(dict(skill="move", params=dict(delta=[0, 0, 0.10], part=part)))
    return rows


def release_template(height=0.10):
    return [
        dict(skill="gripper", params=dict(mode="open")),
        dict(skill="move", params=dict(delta=[0, 0, height])),
    ]


TEMPLATES = {"pick": pick_template, "release": release_template}


def build_pick_candidates(
    session, part, lift=True, terminal_targets=None, control_options=None,
    score_release=True
):
    """Bind grasp/approach route/control/terminal choices from one unchanged state.

    Terminal target alternatives are supplied by the task, not guessed here.
    Their transport paths remain bound deferred requests until grasp completion.
    No post-rollout outcome is available to a ranker at this point.
    """
    source = session.artifacts["grasps"]
    if source.get("part") != part:
        raise ValueError("grasp candidate source bound to wrong part")
    controls = control_options or [dict(strategy="feedback", force=3.0)]
    for control in controls:
        if control.get("strategy") != "feedback" or control.get("force", 0) <= 0:
            raise ValueError("pick supports positive-force feedback control only")
    terminals = terminal_targets or [
        dict(id="held", predicate="held", part=part, lift_m=0.10 if lift else 0.0)
    ]
    initial = fingerprint(session)
    candidates = []
    for grasp in source["candidates"]:
        routes, _ = transfer_routes(
            session, grasp["xyz"] + [0, 0, 0.10], yaw=grasp["yaw"], grasp=grasp
        )
        for route in routes:
            for ci, control in enumerate(controls):
                for ti, terminal in enumerate(terminals):
                    if terminal.get("part", part) != part:
                        raise ValueError("terminal target bound to wrong part")
                    steps = pick_template(part, lift)
                    steps[2]["params"]["force"] = control["force"]
                    cid = f"{route['id']}:control:{ci}:terminal:{ti}"
                    path = copy.deepcopy(route.get("path"))
                    if path is not None:
                        path["binding"]["prefix_id"] = cid
                    release = (release_space_proxy(session, part, grasp, terminal)
                               if score_release else dict(status="not_scored", penalty=0.0))
                    candidates.append(
                        Candidate(
                            cid,
                            part,
                            initial,
                            copy.deepcopy(grasp),
                            path,
                            copy.deepcopy(control),
                            copy.deepcopy(terminal),
                            dict(
                                part=part,
                                grasp_id=grasp["id"],
                                route_id=route["id"],
                                control_id=ci,
                                terminal_id=ti,
                                terminal_release_check=release,
                                transport=dict(
                                    status="deferred",
                                    after="verified grasp",
                                    target=copy.deepcopy(terminal),
                                    grasp_id=grasp["id"],
                                ),
                            ),
                            route["status"],
                            route["unknown"]
                            + ["grasp retention", "terminal suffix feasibility"],
                            float(
                                grasp["cost"]
                                + route.get("cost", 1.0e6)
                                + release["penalty"]
                            ),
                            steps,
                        )
                    )
    for grasp in source.get("unresolved", []):
        for ci, control in enumerate(controls):
            for ti, terminal in enumerate(terminals):
                steps = pick_template(part, lift)
                steps[2]["params"]["force"] = control["force"]
                candidates.append(
                    Candidate(
                        f"{grasp['id']}:unresolved:{ci}:{ti}",
                        part,
                        initial,
                        copy.deepcopy(grasp),
                        None,
                        copy.deepcopy(control),
                        copy.deepcopy(terminal),
                        dict(
                            part=part,
                            grasp_id=grasp["id"],
                            route_id=None,
                            control_id=ci,
                            terminal_id=ti,
                        ),
                        "unknown",
                        [grasp["reason"], "grasp IK and bound route remain unresolved"],
                        1.0e6,
                        steps,
                    )
                )
    return candidates


def choose_candidate(candidates, candidate_id=None):
    """Bootstrap ranking only; not a learned future-value model. Unknowns retained."""
    ready = [c for c in candidates if c.status == "necessary_pass"]
    if candidate_id:
        ready = [c for c in ready if c.id == candidate_id]
    if not ready:
        raise ValueError(
            "no materialized candidate; expand solver budget or verify unknowns"
        )
    return min(ready, key=lambda c: (c.cost, c.id))


def execute_pick_candidate(session, candidate):
    from .graph import execute_skeleton
    from .library import SkillFailure

    if candidate.status != "necessary_pass" or candidate.path is None:
        raise SkillFailure("candidate must be materialized before execution")
    if candidate.start_state != fingerprint(session):
        raise SkillFailure("stale candidate start state; rebuild from current state")
    if (
        candidate.grasp["id"] != candidate.bindings["grasp_id"]
        or candidate.path["binding"]["grasp_id"] != candidate.grasp["id"]
        or candidate.path["id"] != candidate.bindings["route_id"]
        or candidate.path["binding"].get("prefix_id") != candidate.id
    ):
        raise SkillFailure("candidate grasp/path binding mismatch")
    if (
        candidate.bindings["part"] != candidate.part
        or candidate.terminal.get("part", candidate.part) != candidate.part
        or candidate.control.get("strategy") != "feedback"
    ):
        raise SkillFailure("candidate object/control/terminal binding mismatch")
    if not np.allclose(
        candidate.path["target"],
        candidate.grasp["xyz"] + np.array([0, 0, 0.1]),
        atol=1.0e-8,
    ) or not np.allclose(
        candidate.path["rotation"], down(candidate.grasp["yaw"]), atol=1.0e-8
    ):
        raise SkillFailure("candidate path geometry disagrees with grasp")
    from .interfaces import resolve

    close_rows = [
        r
        for r in candidate.steps
        if resolve(r["skill"], r["params"])[0] == "close_gripper"
    ]
    if len(close_rows) != 1 or close_rows[0]["params"].get(
        "force"
    ) != candidate.control.get("force"):
        raise SkillFailure("candidate controller parameters disagree with template")
    if candidate.path["binding"]["grasp_epoch"] != session.grasp_epoch:
        raise SkillFailure("stale candidate grasp acquisition epoch")
    session.active_candidate_id = candidate.id
    session.artifact("grasp", "grasp", candidate.part, **copy.deepcopy(candidate.grasp))
    session.artifacts["transfer"] = copy.deepcopy(candidate.path)
    execute_skeleton(session, candidate.steps)


def save_batch(session, candidates, chosen):
    batch = dict(
        selected_id=chosen.id,
        ranker="bootstrap_geometric_cost",
        candidates=[c.to_dict() for c in candidates],
    )
    session.candidate_batches.append(batch)
    write_batches(session)


def write_batches(session):
    if session.out:
        session.out.mkdir(parents=True, exist_ok=True)
        (session.out / "candidates.json").write_text(
            json.dumps(
                session.candidate_batches,
                indent=2,
                default=lambda x: np.asarray(x).tolist(),
            )
        )


def record_continuation(session, routes, chosen):
    """Persist all alternatives for the actual measured grasp, linked to its prefix."""
    if session.candidate_batches and session.active_candidate_id:
        batch = next(
            (
                b
                for b in reversed(session.candidate_batches)
                if b["selected_id"] == session.active_candidate_id
            ),
            None,
        )
        if batch is not None:
            batch.setdefault("continuations", []).append(
                dict(
                    prefix_id=session.active_candidate_id,
                    selected_id=chosen["id"],
                    candidates=copy.deepcopy(routes),
                )
            )
            write_batches(session)
