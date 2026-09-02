"""Task executor: resolves semantic landmarks and dispatches atomic skills.

The ``Executor`` is the glue between the planner (skill-sequence plans with
semantic params) and the skills library (functions that take world
coordinates).  For each ``(skill, params)`` step in a plan it:

1. Resolves semantic landmarks (site names, body names, relative z refs)
   to world coordinates.
2. Calls the corresponding skill function with the resolved params.
3. Records ``{step, skill, params, ok, metrics}``.

Perception noise is injected via ``noise_std``: the grasp handler passes it
to ``perception.estimate_grasp_pose`` so detection is noisy.  Future
control-gain perturbation hooks can be added the same way.

State is shared between steps via ``self.state`` (e.g. the grasp offset
from a grasp step is used by a later detect_datum step).
"""
import numpy as np
import mujoco

from .core.controller import CartesianController, Gripper
from .faults import FailureModel
from .skills import perception
from .skills import manipulation
from .skills import planning
from .skills.base import REGISTRY
from .skills.actuator import drive_position, drive_force
from .skills.motion import move_eef, home, goto_ax, align_above
from .skills.settle import settle

# probe geometry (Task C specific — would be in scene meta in production)
PROBE_L = 0.030
DIMPLE_DEPTH = 0.002

# boss ring radius (Task A specific — would be in scene meta in production)
BOSS_RING_R = 0.01825


class Executor:
    """Runs a TaskPlan: resolves landmarks, dispatches skills, records results.

    Skills are dispatched through TWO tables (the two-class atomic-action
    taxonomy):

      _PLAN_SKILLS -- planning actions (plan_grasp_pose, plan_path): pure
          computation, no physics motion; they write plan artifacts into
          ``self.state["plans"]`` and can fail (missed detection,
          unreachable goal, predicted collision).
      _SKILLS      -- execution actions (detect_part, move_to, grasp,
          transport, place, insert, inspect, ...): drive the physics and
          consume the artifacts produced by the planning actions.

    A task that skips the planning steps (B/C nominal plans) keeps the
    legacy inline behavior: grasp/move fall back to detect/plan
    internally, so their semantics are unchanged.

    Fault injection goes through ``self.faults`` (a FailureModel): the
    new execution actions sample every stochastic source from it, so a
    run is deterministic for a given (profile, seed) and every failure
    is attributable via ``faults.log``.
    """

    def __init__(self, ctx, arm=None, gripper=None, noise_std=0.0,
                 faults=None):
        self.ctx = ctx
        self.arm = arm or CartesianController(ctx)
        self.gripper = gripper or Gripper(ctx)
        self.noise_std = noise_std
        self.faults = faults or FailureModel("none", seed=0)
        self.results = []
        self.state = {}          # shared state between steps (percepts,
                                 # plans, fault attribution log)
        self._custom_skills = {}
        self._step_idx = -1
        self.on_step = None      # optional per-step callback for the
                                 # data collector (i, skill, params, ok)
        self.on_stage = None     # optional video annotation callback
                                 # (i, skill, total) at step start

    # -------------------------------------------------- public API
    def register_skill(self, name, func):
        """Register a task-specific skill handler."""
        self._custom_skills[name] = func

    def run(self, plan, verbose=True):
        """Execute the plan; returns list of result dicts."""
        self.results = []
        self.state["fault_log"] = self.faults.log
        for i, (skill, params) in enumerate(plan):
            if self.on_stage is not None:
                self.on_stage(i, skill, len(plan))
            self._step_idx = i
            ok = self._dispatch(i, skill, params, verbose)
            kind = ("plan" if skill in self._PLAN_SKILLS else "exec")
            self.results.append(dict(step=i, skill=skill,
                                     params=dict(params), ok=ok,
                                     kind=kind))
            if self.on_step is not None:
                self.on_step(i, skill, params, ok)
            if not ok and plan.fail_mode == "abort":
                if verbose:
                    print(f"  [executor] abort at step {i} "
                          f"({skill} failed)")
                break
        return self.results

    @property
    def all_ok(self):
        return all(r["ok"] for r in self.results)

    # -------------------------------------------------- dispatch
    def _dispatch(self, i, skill, params, verbose):
        # planning actions first: the two-class taxonomy separates
        # plan-then-execute steps by table
        handler = self._PLAN_SKILLS.get(skill)
        if handler is None:
            handler = self._SKILLS.get(skill)
        if handler is None:
            handler = self._custom_skills.get(skill)
        self._current = params
        if handler is None and REGISTRY.has(skill):
            # registry fallback: the newly-registered composable skills
            # (transitions / extension / library) are runnable directly
            # in a plan through the contract-gated REGISTRY.run.  The
            # specialized tables above take precedence, so existing
            # A/B/C behavior is unchanged (purely additive).
            if verbose:
                print(f"  [step {i}] {skill} {params} (registry)")
            try:
                res = REGISTRY.run(skill, self.ctx, self.arm,
                                   self.gripper, verbose=verbose, **params)
                if res.metrics:
                    self.state[f"{skill}_metrics"] = res.metrics
                if verbose and res.reason:
                    print(f"  [step {i}] {skill} -> {res.reason}")
                return bool(res.ok)
            except Exception as exc:
                print(f"  [step {i}] {skill} EXCEPTION: {exc}")
                return False
        if handler is None:
            raise ValueError(f"unknown skill {skill!r} at step {i}")
        if verbose:
            print(f"  [step {i}] {skill} {params}")
        try:
            return handler(self, params, verbose)
        except Exception as exc:
            print(f"  [step {i}] {skill} EXCEPTION: {exc}")
            return False

    # -------------------------------------------------- resolution helpers
    def _resolve_xy(self, ref):
        """Resolve a semantic xy reference to world (x, y).

        ref can be: site name, body name, (x, y) tuple/list, or
        dict {'ref': body, 'mode': 'axis'} for live body xy, or
        dict {'ref': body, 'mode': 'offset', 'local': (lx, ly)} for a
        live body-frame point offset from the body centre.
        """
        if isinstance(ref, dict) and ref.get("mode") == "axis":
            return self.ctx.obj_pos(ref["ref"])[:2]
        if isinstance(ref, dict) and ref.get("mode") == "boss_axis":
            return self._boss_axis_xy(ref["ref"])
        if isinstance(ref, dict) and ref.get("mode") == "offset":
            body = ref["ref"]
            lx, ly = ref["local"]
            pos, quat = self.ctx.obj_pose(body)
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, quat)
            R = R.reshape(3, 3)
            return (pos + R @ np.array([lx, ly, 0.0]))[:2]
        if isinstance(ref, str):
            try:
                return self.ctx.site_pos(ref)[:2]
            except ValueError:
                return self.ctx.obj_pos(ref)[:2]
        return np.asarray(ref, dtype=float)[:2]

    def _resolve_z(self, ref):
        """Resolve a semantic z reference to world z.

        ref can be: float, site name, body name, or
        dict {'ref': body, 'mode': 'top'|'floor'|'seated', 'part': name}.
        """
        if isinstance(ref, (int, float)):
            return float(ref)
        if isinstance(ref, dict):
            body = ref["ref"]
            mode = ref.get("mode", "center")
            offset = ref.get("offset", 0.0)
            if mode == "site":
                return float(self.ctx.site_pos(body)[2]) + offset
            bz = self.ctx.obj_pos(body)[2]
            if mode == "center":
                return bz + offset
            meta = perception.part_meta(self.ctx, body)
            bh = meta["half_h"]
            if mode == "top":
                return bz + bh + offset
            if mode == "floor":
                return bz - bh + offset
            if mode == "seated":
                part = ref.get("part", self._current.get("part", ""))
                pm = perception.part_meta(self.ctx, part)
                return bz - bh + pm["half_h"] + offset
            return bz + offset
        if isinstance(ref, str):
            try:
                return float(self.ctx.site_pos(ref)[2])
            except ValueError:
                return float(self.ctx.obj_pos(ref)[2])
        return float(ref)

    def _resolve_ref_axis(self, ref):
        """Resolve a semantic ref_axis to a callable(ctx)->xy or fixed xy.

        ref can be: dict {'ref': body, 'mode': 'axis'|'boss_axis'|'offset'},
        body name, site name, or (x, y) tuple.
        """
        if isinstance(ref, dict) and ref.get("mode") == "axis":
            body = ref["ref"]
            return lambda ctx: ctx.obj_pos(body)[:2]
        if isinstance(ref, dict) and ref.get("mode") == "boss_axis":
            body = ref["ref"]
            return lambda ctx: self._boss_axis_xy(body)
        if isinstance(ref, dict) and ref.get("mode") == "offset":
            body = ref["ref"]
            lx, ly = ref["local"]

            def _off(ctx):
                pos, quat = ctx.obj_pose(body)
                R = np.zeros(9)
                mujoco.mju_quat2Mat(R, quat)
                R = R.reshape(3, 3)
                return (pos + R @ np.array([lx, ly, 0.0]))[:2]
            return _off
        if isinstance(ref, str):
            # body name → live callable; site name → fixed xy
            try:
                self.ctx.site_id(ref)
                return np.asarray(self.ctx.site_pos(ref)[:2],
                                  dtype=float)
            except ValueError:
                body = ref
                return lambda ctx: ctx.obj_pos(body)[:2]
        return np.asarray(ref, dtype=float)[:2]

    def _boss_axis_xy(self, body_name):
        """Live world xy of the boss axis of a body (boss sits at
        BOSS_RING_R in the body's local +x)."""
        pos, quat = self.ctx.obj_pose(body_name)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, quat)
        R = R.reshape(3, 3)
        return (pos + R @ np.array([BOSS_RING_R, 0.0, 0.0]))[:2]

    def _resolve_q_min(self, ref):
        """Resolve q_min for drive_force (float or semantic)."""
        if isinstance(ref, (int, float)):
            return float(ref)
        if isinstance(ref, str) and ref.endswith("_top"):
            # clamp bottom 4mm below the live plate top (demo: the 1mm
            # press only reached ~0.9N, under the 1.5N gate; 4mm keeps
            # ~1.8N of servo press on the -x edge).  half_h gives the
            # top, then -4mm press + clamp half-z - clamp root z
            body = ref[:-4]
            meta = perception.part_meta(self.ctx, body)
            return (self.ctx.obj_pos(body)[2] + meta["half_h"]
                    - 0.004 + 0.003 - 0.870)
        return float(ref)

    # ================================================================ SKILLS
    # -- grasp
    def _skill_grasp(self, p, verbose):
        name = p["part"]
        # consume the plan_grasp_pose artifact when the plan produced
        # one for this part (the two-class handshake); otherwise fall
        # back to the legacy inline detect+estimate with noise_std
        gp = None
        plans = self.state.get("plans", {})
        if (plans.get("grasp") is not None
                and plans.get("grasp_part") == name):
            gp = plans["grasp"]
        ok = manipulation.grasp(self.ctx, self.arm, self.gripper, name,
                               noise_std=(0.0 if gp is not None
                                          else self.noise_std),
                               grasp_pose=gp,
                               press=p.get("press", 0.0015),
                               lift=p.get("lift",
                                         manipulation.DEFAULT_LIFT),
                               grasp_tol=p.get("tol", manipulation.GRASP_TOL),
                               grasp_gain=p.get("gain", 6.0),
                               grasp_dz=p.get("grasp_dz"),
                               squeeze=p.get("squeeze") or 0.0,
                               approach_yaw=(
                                   gp.get("approach_yaw")
                                   if gp is not None else p.get("yaw")),
                               # the two-class move_to already travelled
                               # to a hover above the part: hop straight
                               # down (no safe_z re-lift)
                               approach_direct=(gp is not None),
                               repress=p.get("repress", True),
                               verify_lift=p.get("verify_lift", True),
                               verbose=verbose)
        if ok:
            self.state[f"{name}_offset"] = (
                self.ctx.eef_pos() - self.ctx.obj_pos(name))
            self.state["last_grasped"] = name
        return ok

    # -- place
    def _skill_place(self, p, verbose):
        name = p["part"]
        # held-part guard: a failed transport (slip-drop) must abort
        # BEFORE the open pads descend over the assembly -- the old
        # blind release could sweep a seated stack (measured on the
        # Task B probe).  Skipped when require_held=False.
        if p.get("require_held", True) and \
                np.linalg.norm(self.ctx.eef_pos()
                               - self.ctx.obj_pos(name)) > 0.06:
            if verbose:
                print(f"  [place {name}] part not held -- abort")
            return False
        at = p.get("at")
        target_xy = self._resolve_xy(at) if at else self.ctx.obj_pos(name)[:2]
        z_ref = p.get("z_ref")
        target_z = self._resolve_z(z_ref) if z_ref is not None else None
        return manipulation.place(self.ctx, self.arm, self.gripper, name,
                                 target_xy, target_z=target_z,
                                 carry_speed=p.get("carry_speed"),
                                 align=p.get("align", False),
                                 release=p.get("release", "slew"),
                                 settle_steps=p.get("settle_steps", 10),
                                 carry_style=p.get("carry_style", "safe_z"),
                                 live_align=p.get("live_align", False),
                                 low_carry=p.get("low_carry", False),
                                 level=p.get("level", False),
                                 carry_direct=p.get("carry_direct", False),
                                 verbose=verbose)

    # -- insert
    def _skill_insert(self, p, verbose):
        name = p["part"]
        mode = p.get("mode", "press")
        if mode == "press":
            target_eef = p.get("target_eef")
            if target_eef is not None:
                target_eef = np.asarray(target_eef, dtype=float)
            return manipulation.insert(self.ctx, self.arm, self.gripper,
                                      name=name, target_eef=target_eef,
                                      mode="press",
                                      tol=p.get("tol", 0.003),
                                      press_steps=p.get("press_steps", 15),
                                      verbose=verbose)
        if mode == "thread":
            into = p.get("into")
            ref_axis = self._resolve_ref_axis(into) if into else None
            to_z = self._resolve_z(p["to_z"])
            align_bottom_z = p.get("align_bottom_z")
            if align_bottom_z is not None:
                align_bottom_z = self._resolve_z(align_bottom_z)
            ok = manipulation.insert(self.ctx, self.arm, self.gripper,
                                     name=name, mode="thread",
                                     ref_axis=ref_axis, to_z=to_z,
                                     half=p.get("half"),
                                     align_bottom_z=align_bottom_z,
                                     max_steps=p.get("max_steps", 200),
                                     drop=p.get("drop", 0.004),
                                     steer=p.get("steer", 0.0008),
                                     steer_cap=p.get("steer_cap", 0.001),
                                     steer_drop=p.get("steer_drop", 0.0),
                                     max_wiggles=p.get("max_wiggles", 3),
                                     align_tol=p.get("align_tol", 0.0015),
                                     steer_bottom=p.get("steer_bottom",
                                                        False),
                                     level=p.get("level", False),
                                     force_seat=p.get("force_seat", True),
                                     release_lift=p.get("release_lift",
                                                         0.006),
                                     settle_steps=p.get("settle_steps", 10),
                                     verbose=verbose)
            # center_after: while the pads still grip, nudge the seated
            # part back over a live axis.  Threaded parts park on the
            # hole wall wherever the release dropped them (the Task B
            # connector measured 3.4mm eccentric on its 3.55mm rim
            # clearance); a flange-on-rim seat slides sideways under
            # the pads once the seat friction is unloaded.
            ca = p.get("center_after")
            if ok and ca is not None:
                axy = np.asarray(self._resolve_xy(ca), dtype=float)[:2]
                tol_c = p.get("center_tol", 0.0025)
                self.arm.move_eef(
                    self.ctx.eef_pos() + np.array([0.0, 0.0, 0.002]),
                    gain=10.0, tol=0.003)
                for _ in range(30):
                    err = self.ctx.obj_pos(name)[:2] - axy
                    if np.linalg.norm(err) < tol_c:
                        break
                    eef = self.ctx.eef_pos()
                    self.arm.move_eef(
                        eef - np.array([err[0], err[1], 0.0]),
                        gain=6.0, tol=0.0005, max_steps=1, stall=False)
                settle(self.ctx, max_steps=20)
                if verbose:
                    err = self.ctx.obj_pos(name)[:2] - axy
                    print(f"  [center_after] resid="
                          f"{np.linalg.norm(err)*1000:.2f}mm")
            return ok
        raise ValueError(f"unknown insert mode {mode!r}")

    # -- push
    def _skill_push(self, p, verbose):
        return manipulation.push(self.ctx, self.arm, self.gripper,
                                 p["part"],
                                 delta=p.get("delta"),
                                 to_target=(self._resolve_xy(p["to_target"])
                                            if "to_target" in p else None),
                                 standoff=p.get("standoff", 0.06),
                                 push_z=(self._resolve_z(p["push_z"])
                                         if "push_z" in p else None),
                                 speed=p.get("speed", 0.06),
                                 verbose=verbose)

    # -- resolve a ref_axis-ish param (callable / fixed xy) to world xy
    def _ref_to_xy(self, ref):
        if ref is None:
            return None
        r = self._resolve_ref_axis(ref)
        if callable(r):
            r = r(self.ctx)
        return np.asarray(r, dtype=float)[:2]

    def _tilt_of(self, name):
        """Body tilt in deg, or 0.0 when the scene has no such body
        (debug references to legacy bodies must not crash other scenes)."""
        try:
            return self.ctx.obj_tilt(name)
        except ValueError:
            return 0.0

    def _pos_of(self, name):
        """Body position, or zeros when the scene has no such body."""
        try:
            return self.ctx.obj_pos(name)
        except ValueError:
            return np.zeros(3)

    # -- screw_drive (hex bolt into a boss hole, controlled helix)
    def _skill_screw_drive(self, p, verbose):
        name = p["part"]
        # hold check: a failed grasp (the nut still in its feed spot)
        # must abort BEFORE the empty pads descend over the assembly
        # -- the blind screw loop once swept the whole stack over
        # (measured: every stage toppled during the empty helix)
        if np.linalg.norm(self.ctx.eef_pos()
                          - self.ctx.obj_pos(name)) > 0.03:
            if verbose:
                print(f"  [screw_drive {name}] part not held -- abort")
            return False
        axy = self._ref_to_xy(p.get("into"))
        to_z = self._resolve_z(p["to_z"])
        depth, tilt = manipulation.screw_drive(
            self.ctx, self.arm, self.gripper, name, axy, to_z,
            half=p.get("half"), pitch=p.get("pitch", 0.005),
            dz_step=p.get("dz_step", 0.0002),
            max_steps=p.get("max_steps", 600),
            settle_steps=p.get("settle_steps", 20), verbose=verbose)
        self.state[f"{name}_depth"] = depth
        self.state[f"{name}_tilt"] = tilt
        return depth >= p.get("depth_min", 4.0)

    # -- lateral_insert (horizontal rail insertion of a held module)
    def _skill_lateral_insert(self, p, verbose):
        name = p["part"]
        axy = self._ref_to_xy(p.get("into"))
        # level: plumb the held part before the drive.  A long box held
        # on the ~7deg tilted wrist sweeps +-9mm of height at its ends
        # and cannot enter a tight rail gap (measured: the tilted PSU
        # corner rammed the upper support bars); plumb it first.
        if p.get("level"):
            manipulation._level_axis(self.ctx, self.arm, name,
                                     verbose=verbose)
        # at_z: semantic z ref of the PART CENTRE at the rail height.
        # The horizontal drive keeps the current eef z, so the held
        # module must first descend to its slide height (otherwise the
        # drive pushes it through the air above the rails and the
        # release drops it from height -- measured).
        at_z = p.get("at_z")
        if at_z is not None:
            tz = self._resolve_z(at_z)
            off = self.ctx.eef_pos() - self.ctx.obj_pos(name)
            goal = np.array([self.ctx.eef_pos()[0], self.ctx.eef_pos()[1],
                             tz + off[2]])
            if verbose:
                print(f"  [lateral at_z] part "
                      f"{np.round(self.ctx.obj_pos(name), 4)} "
                      f"eef {np.round(self.ctx.eef_pos(), 4)} "
                      f"off_z {off[2]:+.4f} goal_z {goal[2]:.4f}")
            r = move_eef(self.arm, goal, style="direct", tol=0.006,
                         max_speed=0.12)
            if verbose:
                print(f"  [lateral at_z] after: part "
                      f"{np.round(self.ctx.obj_pos(name), 4)} "
                      f"eef {np.round(self.ctx.eef_pos(), 4)} "
                      f"moved={r}")
        ok, disp = manipulation.lateral_insert(
            self.ctx, self.arm, self.gripper, name, axy,
            depth=p["depth"], approach=p.get("approach", 0.02),
            carry_speed=p.get("carry_speed", 0.12),
            speed=p.get("speed", 0.03), wiggle=p.get("wiggle", 0.004),
            max_wiggles=p.get("max_wiggles", 2),
            stall_patience=p.get("stall_patience", 30),
            release=p.get("release", True), verbose=verbose)
        self.state[f"{name}_lat"] = disp
        return ok

    # -- snap_press (press a latch tab down with the closed pads)
    def _skill_snap_press(self, p, verbose):
        at = self._resolve_xy(p["at"])
        z_top = self._resolve_z(p["z_top"])
        f, pressed = manipulation.snap_press(
            self.ctx, self.arm, self.gripper, at, z_top,
            press=p.get("press", 0.004), f_stop=p.get("f_stop", 1.2),
            step=p.get("step", 0.0001),
            max_steps=p.get("max_steps", 250), verbose=verbose)
        tag = p.get("tag", p.get("part", "x"))
        self.state[f"snap_force_{tag}"] = f
        self.state[f"snap_press_{tag}"] = pressed
        # ok = a meaningful press force was reached (the exact snap gate
        # is the stage predicate's; same convention as drive_force)
        return f > 0.5 * p.get("f_stop", 1.2)

    # -- press_fit (force-controlled vertical press of a held part)
    def _skill_press_fit(self, p, verbose):
        name = p["part"]
        axy = self._ref_to_xy(p.get("into"))
        to_z = self._resolve_z(p["to_z"])
        ok, depth, f = manipulation.press_fit(
            self.ctx, self.arm, self.gripper, name, axy, to_z,
            half=p.get("half"), step=p.get("step", 0.0003),
            max_steps=p.get("max_steps", 300),
            contact_geom=p.get("contact_geom"), verbose=verbose)
        self.state[f"{name}_press_depth"] = depth
        self.state[f"{name}_press_force"] = f
        return ok

    # -- slide (long-travel rail slide of a held part)
    def _skill_slide(self, p, verbose):
        name = p["part"]
        d = np.asarray(p["direction"], dtype=float)[:2]
        disp = manipulation.slide(
            self.ctx, self.arm, self.gripper, name, d,
            float(p["dist"]), speed=p.get("speed", 0.05),
            stall_patience=p.get("stall_patience", 40),
            max_steps=p.get("max_steps", 600),
            release=p.get("release", False), verbose=verbose)
        self.state[f"{name}_slide"] = disp
        return disp >= 0.8 * float(p["dist"])

    # -- rotate
    def _skill_rotate(self, p, verbose):
        aa = np.asarray(p["axis_angle"], dtype=float)
        # Default mode verifies the arm's rotation EXECUTION (measured
        # eef quaternion change vs the command).  measure_part selects
        # PART-level torque transmission instead (Task C S8 spin check:
        # pads grip the square shaft flats and physically turn the
        # rotor -- a round shaft measurably slips, a square one cannot):
        # the judged quantity is then the part's yaw change, written to
        # state["{part}_rot_deg"].
        part = p.get("measure_part")
        yaw0 = float(self.ctx.obj_yaw(part)) if part else None
        q0 = self.ctx.eef_quat().copy()
        manipulation.rotate_eef(self.ctx, self.arm, self.gripper, aa,
                                steps=p.get("steps", 40))
        settle(self.ctx, max_steps=15)
        q1 = self.ctx.eef_quat()
        dot = abs(float(np.dot(q0, q1)))
        self.state["rotated_deg"] = float(np.degrees(
            2.0 * np.arccos(np.clip(dot, 0.0, 1.0))))
        if part is not None:
            dyaw = float(self.ctx.obj_yaw(part)) - yaw0
            dyaw = (dyaw + np.pi) % (2.0 * np.pi) - np.pi
            self.state[f"{part}_rot_deg"] = float(abs(np.degrees(dyaw)))
        if verbose:
            print(f"  [rotate] command={float(np.degrees(np.linalg.norm(aa))):.1f}deg "
                  f"eef={self.state['rotated_deg']:.1f}deg"
                  + (f" part={self.state[f'{part}_rot_deg']:.1f}deg"
                     if part is not None else ""))
        return True

    # -- drive_position
    def _skill_drive_position(self, p, verbose):
        return drive_position(self.ctx, p["actuator"], p["target"],
                             p.get("dur", 0.4),
                             settle_steps=p.get("settle", 0),
                             verbose=verbose)

    # -- drive_force
    def _skill_drive_force(self, p, verbose):
        q_min = self._resolve_q_min(p["q_min"]) \
            if isinstance(p["q_min"], str) else p["q_min"]
        f = drive_force(self.ctx, p["actuator"], p["q_start"],
                       q_min, p["v"], p["f_stop"], p["dur_max"],
                       p["geom"],
                       ctrl_offset=p.get("ctrl_offset", 0.0),
                       hold_s=p.get("hold", 0.3),
                       verbose=verbose)
        self.state[f"{p['actuator']}_force"] = f
        # ok = a meaningful press force was reached; the exact gate is
        # the stage predicate's.  drive_force stops at f_stop OR
        # dur_max, and the fixture yield lets the peak land short of
        # f_stop (measured: side 1.31N vs f_stop 1.5N, clamp 2.65N vs
        # 5.0N) while the stage thresholds (1.0 / 2.5N) still pass --
        # judging the step on f_stop alone flagged those as failures.
        return f > 0.5 * p["f_stop"]

    # -- settle
    def _skill_settle(self, p, verbose):
        settle(self.ctx, max_steps=p.get("max_steps", 20))
        return True

    # -- home
    def _skill_home(self, p, verbose):
        home(self.arm, self.gripper)
        return True

    # -- release
    def _skill_release(self, p, verbose):
        # open the gripper in place and let the held part drop where
        # it stands; used for wrap-ups where the exact landing spot
        # does not matter (the Task B probe park: placing the hanging
        # probe with the descend-and-seat place left it ~6cm off
        # target because the rod swings inside the pads, measured).
        self.gripper.open()
        settle(self.ctx, max_steps=p.get("settle_steps", 20))
        return True

    # -- move
    def _skill_move(self, p, verbose):
        target = p.get("to")
        if isinstance(target, str):
            target = self._resolve_xy(target)
        elif isinstance(target, dict):
            xy = self._resolve_xy(target.get("at", ""))
            z = self._resolve_z(target.get("z"))
            target = np.append(xy, z)
        else:
            target = np.asarray(target, dtype=float)
        return move_eef(self.arm, target,
                       style=p.get("style", "safe_z"),
                       tol=p.get("tol", 0.008),
                       max_speed=p.get("max_speed"))

    # -- settle_press (Task C S3: press plate top with open pads)
    def _skill_settle_press(self, p, verbose):
        part = p["part"]
        press = p.get("press", 0.001)
        pos0 = self.ctx.obj_pos(part)
        meta = perception.part_meta(self.ctx, part)
        top = pos0[2] + meta["half_h"]
        target = np.array([pos0[0], pos0[1], top - press])
        move_eef(self.arm, target + np.array([0.0, 0.0, 0.08]),
                tol=0.008, max_speed=0.2)
        self.gripper.open()
        goto_ax(self.arm, target, tol_xy=0.005, tol_z=0.0015,
               gain=8.0, max_steps=60)
        settle(self.ctx, max_steps=6)
        self.arm.move_eef(target + np.array([0.0, 0.0, 0.10]),
                         gain=12.0, tol=0.012)
        slip = (self.ctx.obj_pos(part) - pos0)[:2]
        self.state["slip_xy"] = slip
        if verbose:
            print(f"  [settle_press] slip=({slip[0]*1000:.2f},"
                  f"{slip[1]*1000:.2f})mm")
        return True

    # -- detect_datum (Task C S7b: dip probe into datum dimple)
    def _skill_detect_datum(self, p, verbose):
        datum_site = p["datum"]
        tip_geom = p["tip_geom"]
        part = self.state.get("last_grasped", "probe")
        offset = self.state.get(f"{part}_offset", np.zeros(3))

        nom = self.ctx.site_pos(datum_site)[:2]
        wp_pos = self.ctx.obj_pos("workpiece")
        wp_meta = perception.part_meta(self.ctx, "workpiece")
        wp_top = wp_pos[2] + wp_meta["half_h"]
        dimple_floor_z = wp_top - DIMPLE_DEPTH
        tip_target = dimple_floor_z + 0.0005

        eef_target = np.array([nom[0], nom[1],
                              tip_target + PROBE_L + offset[2]])
        move_eef(self.arm, eef_target + np.array([0.0, 0.0, 0.05]),
                tol=0.008, max_speed=0.05)
        # damp the hanging-rod swing BEFORE the dip: the 45cm carry
        # from the sleeve leaves the 60mm probe pendulum-swinging, the
        # tip enters the hole sideways and wedges on the +x wall
        # (measured: med=3.2mm with the tip parked on the wall)
        settle(self.ctx, max_steps=40)
        # align the LIVE tip over the nominal before the vertical dip:
        # the eef carries a ~7deg home-pose tilt, which offsets a 60mm
        # rod's tip ~7mm sideways -- close the loop on the tip
        gid = self.ctx.geom_id(tip_geom)
        for _ in range(60):
            tnow = np.array(self.ctx.data.geom_xpos[gid])[:2]
            err = nom - tnow
            if np.linalg.norm(err) < 0.001:
                break
            eef_now = self.ctx.eef_pos()
            self.arm.move_eef(np.array([eef_now[0] + err[0],
                                        eef_now[1] + err[1],
                                        eef_now[2]]),
                             gain=6.0, tol=0.0, max_steps=1,
                             stall=False)
        settle(self.ctx, max_steps=20)
        # descend on the ALIGNED eef xy -- not the nominal: the eef tilt
        # offsets the tip ~7mm, and going to the nominal eef xy would
        # throw the tip back at the hole wall (measured med=3.4mm)
        eef_target[:2] = self.ctx.eef_pos()[:2].copy()
        goto_ax(self.arm, eef_target, tol_xy=0.002, tol_z=0.0015,
               gain=6.0, max_steps=120)
        settle(self.ctx, max_steps=30)      # let the tip rest on the floor

        tip = np.array(self.ctx.data.geom_xpos[gid])
        med = tip[:2] - nom
        tz = float(tip[2])
        # detected: tip BOTTOM entered the plate top face (the tip
        # rides the funnel ramps ~0.3mm down; the old centre-vs-top
        # compare could never fire)
        detected = tz - 0.0045 < wp_top - 0.0002
        self.state["measured_offset"] = med
        self.state["datum_detected"] = bool(detected)
        if verbose:
            print(f"  [detect_datum] med={np.round(med*1000, 2)}mm "
                  f"tip_z={tz:.4f} detected={detected}")
        return bool(detected)

    # -- move_tool (Task C S8: compensated move to op point)
    def _skill_move_tool(self, p, verbose):
        target_site = p["target"]
        comp_key = p.get("compensate", "measured_offset")
        part = self.state.get("last_grasped", "probe")
        offset = self.state.get(f"{part}_offset", np.zeros(3))
        med = self.state.get(comp_key, np.zeros(2))

        tgt_xy = self.ctx.site_pos(target_site)[:2] + med
        wp_pos = self.ctx.obj_pos("workpiece")
        wp_meta = perception.part_meta(self.ctx, "workpiece")
        wp_top = wp_pos[2] + wp_meta["half_h"]
        zb = wp_top + 0.006
        eef_t = np.array([tgt_xy[0], tgt_xy[1],
                          zb + PROBE_L + offset[2]])
        move_eef(self.arm, eef_t + np.array([0.0, 0.0, 0.05]),
                tol=0.006, max_speed=0.2)
        # align the LIVE tip over the compensated op point (the ~7deg
        # eef tilt offsets the rod tip; the eef-level goto alone left
        # the tip ~3.2mm off -- measured resid)
        gid = self.ctx.geom_id("probe_tip")
        for _ in range(60):
            tnow = np.array(self.ctx.data.geom_xpos[gid])[:2]
            err = tgt_xy - tnow
            if np.linalg.norm(err) < 0.001:
                break
            eef_now = self.ctx.eef_pos()
            self.arm.move_eef(np.array([eef_now[0] + err[0],
                                        eef_now[1] + err[1],
                                        eef_now[2]]),
                             gain=6.0, tol=0.0, max_steps=1,
                             stall=False)
        settle(self.ctx, max_steps=20)
        # descend on the ALIGNED eef xy (the eef tilt offsets the tip;
        # the nominal eef xy would throw the tip ~7mm off -- measured)
        eef_t[:2] = self.ctx.eef_pos()[:2].copy()
        goto_ax(self.arm, eef_t, tol_xy=0.003, tol_z=0.004,
               gain=6.0, max_steps=100)
        # damp the hanging-rod pendulum before judging the residual
        # (the 4mm datum->op hop swings the 60mm rod; 8 steps left the
        # tip ~3.5mm off -- measured)
        settle(self.ctx, max_steps=30)

        tip = np.array(self.ctx.data.geom_xpos[gid])
        resid = float(np.linalg.norm(tip[:2] - tgt_xy))
        self.state["tool_resid"] = resid
        if verbose:
            print(f"  [move_tool] resid={resid*1000:.2f}mm")
        return resid < 0.005

    # -- pad_touch (EXEC): probe tip pressed onto a raised test pad.
    #    The Task C detect_datum recipe generalized: shallow-bite hang,
    #    live-tip servo alignment, then a rigid vertical press -- no
    #    long settles or slice servos (those age the bite out on a
    #    hanging rod; measured padF 1.39 -> 0 in ~300 hold steps).
    #    Judged on the pad contact force + tip residual.
    def _skill_pad_touch(self, p, verbose):
        tip_geom = p["tip_geom"]
        pad_geom = p["pad_geom"]
        part = self.state.get("last_grasped", "probe")
        offset = self.state.get(f"{part}_offset", np.zeros(3))

        pad_pos = np.array(self.ctx.data.geom_xpos[
            self.ctx.geom_id(pad_geom)])
        pad_top = pad_pos[2] + 0.001       # 1mm half-height bump
        tip_target = pad_top - 0.0015      # press INTO the bump
        nom_xy = pad_pos[:2]

        eef_target = np.array([nom_xy[0], nom_xy[1],
                               tip_target + 0.030 + offset[2]])
        move_eef(self.arm, eef_target + np.array([0.0, 0.0, 0.05]),
                tol=0.008, max_speed=0.05)
        # damp the hanging-rod swing BEFORE the alignment
        settle(self.ctx, max_steps=40)
        # align the LIVE tip over the live pad (the ~7deg eef tilt
        # offsets a 60mm rod's tip ~7mm sideways)
        gid = self.ctx.geom_id(tip_geom)
        for _ in range(60):
            tnow = np.array(self.ctx.data.geom_xpos[gid])[:2]
            err = nom_xy - tnow
            if np.linalg.norm(err) < 0.001:
                break
            eef_now = self.ctx.eef_pos()
            self.arm.move_eef(np.array([eef_now[0] + err[0],
                                        eef_now[1] + err[1],
                                        eef_now[2]]),
                             gain=6.0, tol=0.0, max_steps=1,
                             stall=False)
        settle(self.ctx, max_steps=20)
        # descend on the ALIGNED eef xy (rigid, no slice servo)
        eef_target[:2] = self.ctx.eef_pos()[:2].copy()
        goto_ax(self.arm, eef_target, tol_xy=0.002, tol_z=0.0015,
               gain=6.0, max_steps=120)
        # final re-centre + over-travel press: the descent swings the
        # hanging rod; re-servo the tip, then press down in small
        # slices until the pad reads force (the rod rides up through
        # the pads on the friction plateau ~0.26N, so the press must
        # go PAST touchdown)
        f_peak = 0.0
        for _ in range(40):
            tnow = np.array(self.ctx.data.geom_xpos[gid])[:2]
            err = nom_xy - tnow
            if np.linalg.norm(err) < 0.0015:
                break
            eef_now = self.ctx.eef_pos()
            self.arm.move_eef(np.array([eef_now[0] + err[0],
                                        eef_now[1] + err[1],
                                        eef_now[2]]),
                             gain=6.0, tol=0.0, max_steps=1,
                             stall=False)
        for _ in range(25):
            fnow = self.ctx.geom_contact_force(pad_geom)
            f_peak = max(f_peak, fnow)
            if fnow >= 0.1:
                break
            eef_now = self.ctx.eef_pos()
            self.arm.move_eef(eef_now + np.array([0.0, 0.0, -0.0002]),
                              gain=6.0, tol=0.0, max_steps=1,
                              stall=False)
        settle(self.ctx, max_steps=20)      # let the tip rest on the pad

        tip = np.array(self.ctx.data.geom_xpos[gid])
        resid = float(np.linalg.norm(tip[:2] - nom_xy))
        force = max(self.ctx.geom_contact_force(pad_geom), f_peak)
        quality = float(np.clip(force / 1.0, 0.0, 1.0))
        self.state["test_force"] = float(force)
        self.state["test_resid"] = resid
        self.state["contact_quality"] = quality
        ok = quality >= 0.1 and resid < 0.003
        if verbose:
            print(f"  [pad_touch] force={force:.2f}N "
                  f"resid={resid * 1000:.2f}mm "
                  f"tip_z={tip[2]:.4f} pad_top={pad_top:.4f} "
                  f"{'PASS' if ok else 'FAIL'}")
        return bool(ok)

    # -- continuity_test (Task B S8: probe tip pressed onto the live
    #    connector contact pad; detect_datum pattern generalised)
    def _skill_continuity_test(self, p, verbose):
        tip_geom = p["tip_geom"]
        pad_geom = p["pad_geom"]
        part = self.state.get("last_grasped", "probe")
        offset = self.state.get(f"{part}_offset", np.zeros(3))

        # live pad top: the pad geom centre + its half-height (1mm)
        pad_pos = np.array(self.ctx.data.geom_xpos[
            self.ctx.geom_id(pad_geom)])
        pad_top = pad_pos[2] + 0.001
        # press INTO the raised pad (unlike the datum dimple, the pad
        # is a bump: hovering +0.5mm above it measures zero force)
        tip_target = pad_top - 0.0015
        nom_xy = pad_pos[:2]

        eef_target = np.array([nom_xy[0], nom_xy[1],
                               tip_target + PROBE_L + offset[2]])
        if verbose:
            print(f"  [continuity] pad_top={pad_top:.4f} "
                  f"offset={np.round(offset, 4)} "
                  f"eef_target={np.round(eef_target, 4)}")
        # wrist leveling LAST, right before the descent: the rod relaxes
        # to its along-eef hang (~7deg) whenever the arm moves, so a
        # level servo issued BEFORE the align passes gets undone by
        # them (measured: 2.4deg right after the servo, 7.5deg after
        # the passes, eef left at 9.4deg).  Align first on the natural
        # hang, then one small final level (~7deg rotation) leaves the
        # rod plumb for an AXIAL press; the tip shifts ~1.3mm during
        # that rotation and the check pass re-centres it.  An isolated
        # press test shows the grip transmits >1N cleanly when the
        # press stays axial -- the rod's 12-15deg carry-in lean is the
        # pivot seed that skates the tip.
        rot0 = self.arm.rot_target
        if verbose:
            fq_ = self.ctx.finger_qpos
            ctrl_ = self.ctx.data.ctrl[self.ctx.finger_act_ids]
            pf_ = (self.ctx.geom_contact_force("finger1_pad_collision")
                   + self.ctx.geom_contact_force("finger2_pad_collision"))
            print(f"  [continuity] enter tilt="
                  f"{self.ctx.obj_tilt(part):.2f}deg "
                  f"eef_tilt={manipulation._eef_tilt(self.ctx):.2f}deg "
                  f"fq1={fq_[0]:.4f} "
                  f"ctrl1={ctrl_[0]:.4f} padF={pf_:.2f}N")

        # ---------------------------------------------------- grip safety
        # The squeeze bite relaxes with time (contact relaxation;
        # measured padF 2.3 -> 1.8N over ~300 hold steps) and the
        # hanging rod's pendulum swing drives it through the pads; once
        # it slides the padF collapses and the rod drops out.  The
        # alignment phase is LONG (hover settle + 2 coarse servo passes
        # + 3 refine passes + 2 level rotations + a 250-step settle),
        # so the bite must be topped up throughout, and a lost grip
        # must abort BEFORE the servo chases the fallen tip and sweeps
        # the assembly (measured: probe dropped -> eef flailed -> the
        # seated housing/connector chain toppled, all 8 stages failed).
        gid = self.ctx.geom_id(tip_geom)

        # SHALLOW bite maintenance (planner S8 now grasps with
        # press=0.002, no squeeze -- same recipe as task C's probe):
        # keep the bite in the 0.6-1.2N band -- enough to hold the
        # rod, low enough that it still rides through the pads on the
        # friction plateau during the press.
        _KEEP, _TOPPED = 0.6, 1.2

        def _pad_force():
            return (self.ctx.geom_contact_force("finger1_pad_collision")
                    + self.ctx.geom_contact_force("finger2_pad_collision"))

        def _grip_alive():
            """True while the probe still sits in the pads (contact
            present and the rod not laid over)."""
            return (_pad_force() > 0.3
                    and self.ctx.obj_tilt(part) < 25.0)

        def _grip_guard(min_keep=0.6, target=1.2):
            """Top up the squeeze bite with small conservative
            increments (an aggressive re-clench ejects the rod
            sideways); returns False only when the probe has really
            dropped (laid over) or the bite cannot be recovered.  A
            LOW bite with the rod still hanging is NOT a lost grip --
            top it up first (the bite relaxes with time; measured
            padF 2.3 -> 1.8N over ~300 hold steps)."""
            if self.ctx.obj_tilt(part) >= 25.0:
                return False
            if _pad_force() >= min_keep:
                return True
            c1 = float(self.ctx.data.ctrl[self.ctx.finger_act_ids[0]])
            for _ in range(80):
                if _pad_force() >= target:
                    break
                c1 = max(c1 - 0.0002, 0.0)
                if c1 <= 0.0003:
                    break
                self.ctx.set_finger_ctrl(c1, -c1)
                self.ctx.step()
            return _pad_force() > 0.3

        def _abort(reason):
            """Retreat the eef and report the continuity test failed
            cleanly -- a lost grip must never sweep the assembly."""
            if verbose:
                tz_ = np.array(self.ctx.data.geom_xpos[gid]) if gid is not None else None
                print(f"  [continuity] ABORT: {reason}")
                print(f"    tilt={self.ctx.obj_tilt(part):.2f}deg "
                      f"padF={_pad_force():.2f}N "
                      f"tip_z={tz_[2]:.4f} "
                      f"eef={np.round(self.ctx.eef_pos(), 4)} "
                      f"pad_top={pad_top:.4f}")
            # FULLY release the rod BEFORE retreating: a half-open
            # gripper (the 10-step slew still mid-open) drags the
            # wedged rod through the retreat and the rod-end sweeps the
            # table (measured: seated chain toppled; QACC NaN at DOF
            # 45-48 when the wedged rod hit the sleeve).  Command full
            # open in one shot and wait for the rod to drop.
            self.ctx.set_finger_ctrl(self.gripper.open_q,
                                     -self.gripper.open_q)
            settle(self.ctx, max_steps=15)
            e = self.ctx.eef_pos()
            self.arm.move_eef(e + np.array([0.0, 0.0, 0.08]),
                              gain=6.0, tol=0.02, max_steps=60,
                              stall=False)
            settle(self.ctx, max_steps=20)
            self.state["test_force"] = 0.0
            self.state["test_resid"] = 9.9
            self.state["contact_quality"] = 0.0
            self.arm.rot_target = rot0
            return False
        # carry FIRST, level LAST: the rod hangs along the ~7deg eef
        # axis right after the grasp, and the TILTED hang is the stable
        # pocket in the pads -- task C's probe carries 400mm tilted
        # without dropping, while the levelled vertical rod rolls out
        # mid-carry (measured: dropped at carry seg 1-8 across five
        # cuts, always with the wrist already plumb).  So translate
        # with the natural hang, and stand the rod plumb only at the
        # hover, where no further translation can excite the swing.
        hover = p.get("hover", 0.05)
        # ONE long slow move (task C's proven recipe): the short ~90mm
        # hop at 0.05m/s has a single start/stop pair, while the
        # segmented carry fired one impulse per 5mm leg (40 impulses
        # total) and the accumulated train rolled the rod out at seg
        # 15/20 even on the shallow bite.  The shallow bite holds the
        # tilted hang through one smooth move.
        self.arm.move_eef(eef_target + np.array([0.0, 0.0, hover]),
                          tol=0.008, max_speed=0.05)
        if not _grip_guard():
            return _abort("grip lost during the carry")
        # damp the hanging-rod swing BEFORE the dip (the long carry
        # from the sleeve leaves the probe pendulum-swinging)
        settle(self.ctx, max_steps=40)
        if not _grip_guard():
            return _abort("grip lost during hover settle")
        # NO wrist leveling: the rod cannot be plumbed in the pads --
        # the level rotation turns the wrist but the rod rolls back to
        # its ~4-7deg lean (measured: eef_tilt 0.35deg, rod 4.35deg
        # right after the level), and the rotation itself ages the
        # bite.  The press below goes ALONG THE ROD AXIS (the slice
        # loop's ``ax``); the seated connector leans up to ~1.5deg
        # under the press (bounded by the cover bore) and the pad
        # stays within the 3mm resid gate.
        # close the loop on the LIVE tip over the live pad xy: the eef
        # carries a ~7deg home-pose tilt, which offsets a 60mm rod's
        # tip ~7mm sideways.  Two passes with a settle between: the
        # first pass chases the still-swinging tip, the second
        # converges on the damped rest pose (single pass left a
        # ~1.8mm resid off the 2mm pad -- measured).

        def _tip_servo(tol, iters):
            for i in range(iters):
                if self.ctx.obj_tilt(part) >= 25.0:
                    return -1.0         # rod really dropped out
                if i % 5 == 0 and not _grip_guard():
                    return -1.0         # bite unrecoverable
                tnow = np.array(self.ctx.data.geom_xpos[gid])[:2]
                err = nom_xy - tnow
                if np.linalg.norm(err) < tol:
                    return float(np.linalg.norm(err))
                eef_now = self.ctx.eef_pos()
                self.arm.move_eef(np.array([eef_now[0] + err[0],
                                            eef_now[1] + err[1],
                                            eef_now[2]]),
                                  gain=6.0, tol=0.0, max_steps=1,
                                  stall=False)
            return float(np.linalg.norm(
                np.array(self.ctx.data.geom_xpos[gid])[:2] - nom_xy))

        def _servo_checked(tol, iters):
            """Run a servo pass; abort when the probe dropped out of
            the pads mid-pass (return value -1.0)."""
            r = _tip_servo(tol, iters)
            if r < 0:
                return _abort("probe lost from the grip during tip servo")
            return r

        # one coarse pass on the LIVE tip over the live pad xy (the
        # eef carries a ~7deg home-pose tilt, which offsets a 60mm
        # rod's tip ~7mm sideways), then REFINE until the SETTLED tip
        # rests <0.3mm off the pad centre.  The refine loop is cut to
        # two passes: the rod's bite decays with the hang time (the
        # deep 4N bite rolled out mid-carry; the shallow bite survives
        # ~900 hold steps -- measured), so the alignment must finish
        # before the bite ages out.
        _servo_checked(0.0007, 60)
        settle(self.ctx, max_steps=30)
        if not _grip_guard():
            return _abort("grip lost after coarse servo")
        for _ in range(2):
            r = _servo_checked(0.00015, 60)
            settle(self.ctx, max_steps=20)
            if r < 0.00015:
                break
        # final re-centre check pass on the settled tip, then let the
        # rod rest before descent.
        _servo_checked(0.00015, 30)
        settle(self.ctx, max_steps=15)
        # the tip creeps ~1.3mm through the pads between passes;
        # re-centre AGAIN and let the rod rest before descent.  The
        # descent must hit the 2mm pad TIP-FIRST: with rod tilt >1deg
        # the wider mid section (r1.5) catches the pad edge 5mm above
        # the surface and props the whole rod there (measured: tipF
        # plateau 0.27N, tip stranded 5.3mm above the pad, judgement
        # force 0).
        _servo_checked(0.00015, 30)
        settle(self.ctx, max_steps=20)
        # DAMP THE PENDULUM before descending: the re-centre servos
        # leave the rod swinging +-0.8deg (~35-step period); at a
        # swing peak >1.6deg the wider mid section catches the pad
        # edge 5mm above the surface and props the rod there
        # (measured twice: tip stranded at tip_z=0.8822, tipF plateau
        # 0.25-0.27N, judgement force ~0).  A settle lets the grip
        # friction damp the swing below the ~1deg catch angle -- cut
        # from 250 steps because the shallow bite ages out in ~900
        # hold steps (measured) and the alignment time budget is
        # consumed by the settle.
        settle(self.ctx, max_steps=100)
        # RE-SAMPLE the pad centre after the long settle: the leaning
        # seated connector relaxes sideways during the wait (measured:
        # pad xy drifted 1.2mm while the tip stayed on the stale
        # nominal), and an off-centre touchdown kicks the pad sideways
        # and levers the connector over.  One final servo pass centres
        # the LIVE tip on the LIVE pad before the descent.
        if not _grip_guard():
            return _abort("grip lost during the long settle")
        nom_xy = np.array(self.ctx.data.geom_xpos[
            self.ctx.geom_id(pad_geom)])[:2].copy()
        _servo_checked(0.00015, 30)
        settle(self.ctx, max_steps=20)
        # NO deep re-clench here: the deep bite rolls the rod out
        # under ANY motion (measured across six cuts) and the static
        # re-clench itself shoves the rod off-centre (measured: resid
        # 0.10 -> 1.22mm, dropped at slice k=17).  The shallow bite
        # holds through the press -- the rod rides up through the
        # pads on the friction plateau until the shoulder jams (see
        # the press loop below).
        if not _grip_guard():
            return _abort("grip lost at the pre-descent check")
        txy = np.array(self.ctx.data.geom_xpos[gid])[:2]
        # re-derive the descent z from the LIVE tip: the rod creeps a
        # few mm through the pads between the grasp-time offset
        # snapshot and the aligned hover (the deep-bite squeeze and the
        # level rotation unload/reload the hang), and a stale eef z
        # target stops the press short of the pad when the grip no
        # longer slides (measured with a high-friction rod: off drifted
        # 0.2 -> 4.9mm, the loop hit eef_zt with the tip 4mm above the
        # pad, tipF=0).
        tip_z0 = float(self.ctx.data.geom_xpos[gid][2])
        eef_zt_live = (self.ctx.eef_pos()[2] + tip_target - tip_z0)
        if verbose:
            fq_ = self.ctx.finger_qpos
            ctrl_ = self.ctx.data.ctrl[self.ctx.finger_act_ids]
            pf_ = (self.ctx.geom_contact_force("finger1_pad_collision")
                   + self.ctx.geom_contact_force("finger2_pad_collision"))
            print(f"  [continuity] post-align resid="
                  f"{np.linalg.norm(txy - nom_xy)*1000:.2f}mm "
                  f"tip_z={self.ctx.data.geom_xpos[gid][2]:.4f} "
                  f"tilt={self.ctx.obj_tilt(part):.2f}deg "
                  f"eef_tilt={manipulation._eef_tilt(self.ctx):.2f}deg "
                  f"fq1={fq_[0]:.4f} ctrl1={ctrl_[0]:.4f} "
                  f"padF={pf_:.2f}N")
        # segmented closed-loop descent on the LIVE tip: the rod slides
        # and swings in the loose pads during the ~55mm drop, and one
        # big rigid-lever descent let the tip drift 2.6mm onto the
        # connector grip top and lever the seated connector over
        # (11.5deg -- measured).  Descend in ~0.5mm slices, re-servoing
        # xy on the live tip after every slice, and break the moment
        # the tip LOADS the pad.
        #
        # descend="rigid" selects the Task C detect_datum recipe
        # instead: open-loop descent on the ALIGNED eef xy.  The slice
        # servo itself excites the hanging rod: chasing the swinging
        # tip while dropping phase-locks the swing and the tip skates
        # off the pad edge (measured: tilt 3 -> 17deg, resid 4-6mm,
        # force 0).  Pair it with a LOW hover: the rod accumulates
        # ~2deg of tilt per long descent (measured: 1.9mm tip drift
        # over a 55mm drop, 5.9mm over a second one), so the hover
        # standoff must be small enough that the final drop is ~2mm.
        if p.get("descend") == "rigid":
            # open-loop descent on the ALIGNED eef xy.  Everything
            # fancier failed: slice-wise live-tip servoing phase-locks
            # the hanging-rod swing (tip skates 4-6mm off the pad); a
            # fast/long drop excites a ~2deg touchdown swing;
            # re-launching the servo from the loaded state flicks the
            # rod sideways (measured).
            #
            # f_stop>0 selects the force-limited variant: single IK
            # steps down until the pad reads >= f_stop, then stop and
            # hold.  Needed with the squeeze grasp: the deep bite
            # blocks the slide-to-unload relief, so a fixed-depth push
            # keeps loading past contact and pivots the rod 6-12deg
            # about the tip until it skates off (measured at 0.001 and
            # 0.002 squeeze, 120 and 400 steps).  The deep bite makes
            # the load path stiff, so the force ramps to the stop in a
            # short travel before any tilt can build.  Without f_stop
            # (shallow bite) a fixed goto_ax of 120 steps is the
            # measured-stable recipe (resid 0.35mm, force 0.28N,
            # transient peak 0.47N -- the rod slides up through the
            # pads at ~0.3N and caps the push).
            f_stop = p.get("f_stop", 0.0)
            if f_stop > 0.0:
                def _roddbg(tag):
                    R_ = np.array(self.ctx.data.geom_xpos[gid])
                    q_ = self.ctx.obj_pose(part)[1]
                    Rm = np.zeros(9)
                    import mujoco as _mj
                    _mj.mju_quat2Mat(Rm, q_)
                    Rm = Rm.reshape(3, 3)
                    loc = Rm.T @ (self.ctx.eef_pos()
                                  - self.ctx.obj_pos(part))
                    pz_ = self.ctx.data.geom_xpos[
                        self.ctx.geom_id(pad_geom)]
                    print(f"  [RODDBG {tag}] eef_in_rod="
                          f"{np.round(loc*1000,1)}mm "
                          f"tip={np.round(R_,4)} "
                          f"pad={np.round(np.array(pz_),4)} "
                          f"ctilt={self._tilt_of('connector'):.2f}",
                          flush=True)
                _roddbg("start")
                # over_travel: keep pressing PAST tip touchdown.  With a
                # sliding grip the rod rides up through the pads at the
                # ~0.26N kinetic-friction plateau until the probe's
                # shoulder step jams against the pad-band bottom edge,
                # after which the press loads geometrically (>1N in the
                # isolated transmit test).  A floor AT touchdown quits
                # on the plateau before the jam can engage.
                #
                eef_zt = eef_zt_live - p.get("over_travel", 0.0)
                for k in range(400):
                    if not _grip_alive():
                        return _abort(f"grip lost during rigid press "
                                      f"(k={k})")
                    fpad = self.ctx.geom_contact_force(pad_geom)
                    if fpad >= f_stop:
                        if verbose:
                            print(f"  [press] break k={k} force "
                                  f"{fpad:.2f}N >= f_stop")
                        break
                    eef_now = self.ctx.eef_pos()
                    if eef_now[2] <= eef_zt:
                        if verbose:
                            tz_ = self.ctx.data.geom_xpos[gid][2]
                            print(f"  [press] break k={k} eef_z floor "
                                  f"(tip {tz_:.4f} vs target "
                                  f"{tip_target:.4f}, "
                                  f"tilt={self.ctx.obj_tilt(part):.1f})")
                        break
                    if verbose and k % 25 == 0:
                        pf_ = (self.ctx.geom_contact_force(
                            "finger1_pad_collision")
                            + self.ctx.geom_contact_force(
                                "finger2_pad_collision"))
                        off_ = eef_now[2] - self.ctx.obj_pos(part)[2]
                        tz_ = self.ctx.data.geom_xpos[gid]
                        rxy_ = np.linalg.norm(tz_[:2] - nom_xy)
                        print(f"  [press k={k}] tipF={fpad:.2f}N "
                              f"gripF={pf_:.2f}N off={off_*1000:.1f}mm "
                              f"tilt={self.ctx.obj_tilt(part):.1f} "
                              f"tip_xy_resid={rxy_*1000:.2f}mm "
                              f"tip_z={tz_[2]:.4f}")
                    d = np.array([0.0, 0.0,
                                  -min(0.0002,
                                       eef_now[2] - eef_zt)])
                    self.arm.move_eef(eef_now + d, gain=6.0, tol=0.0,
                                      max_steps=1, stall=False)
                _roddbg("end")
            else:
                eef_t = self.ctx.eef_pos().copy()
                eef_t[2] = eef_zt_live
                goto_ax(self.arm, eef_t, tol_xy=0.002, tol_z=0.001,
                        gain=6.0, max_steps=120)
        else:
            # slice servo with ACCUMULATED press depth: xy is closed
            # loop on the live tip (a z-only open-loop descent drifts
            # the tip ~0.8mm sideways over the 10mm drop -- measured),
            # while z commands accumulate against the measured position
            # so the contact load builds.  Clipping each step to the
            # CURRENT eef z caps the commanded penetration at one slice
            # (~0.1N equilibrium -- measured), so the press never loads
            # the pad.  f_stop sets the force target (0.8 default).
            #
            # THREE-PHASE descent.  The squeeze-hold bite BLEEDS with
            # time (measured: padF 2.3 -> 1.8N over ~300 hold steps,
            # then the rod drops out of the fingers), so the press
            # must reach its force target QUICKLY; but a fast touchdown
            # on the hard pad contact (solref 0.004) throws a collision
            # impulse that flings the seated connector over.
            # Phase A: 0.2mm slices until the LIVE tip is 1.5mm above
            # the pad top (fast, no contact yet).  Phase B: 0.02mm
            # creep until the pad first reads force.  Phase C: rebase
            # the command and accumulate 0.2mm slices to ramp the force.
            #
            # DESCEND ALONG THE ROD AXIS, not world z.  The rod hangs
            # 2.4deg off vertical in the pads and the wrist cannot
            # plumb it (singularity -- measured), so a vertical descent
            # lands the tip's tilted face EDGE-ON on the pad: the edge
            # shaves laterally through the contact (per-step trace:
            # cFy -0.1..-0.33N sustained, pad slides +1mm in y in ONE
            # physics step, conn angular velocity 2.7rad/s, ctilt 0.03
            # -> 1.28 -> 3.9deg at only 0.2N).  Pressing along the rod
            # axis loads the pad face-on; the same stack held a 1N
            # static press along a 2.5deg lean without rocking (xfrc
            # experiment, ctilt peak 0.058deg).
            ax = np.array(self.ctx.obj_axis(part), dtype=float)
            ax /= np.linalg.norm(ax)
            # f_target 0.1 (demo): a 0.1N tip-pad touch is a reliable
            # electrical contact; pressing harder levers the seated
            # 7.7g connector over in its seat (measured: ctilt 2.8deg
            # at ~0.1N, tip then skated off).  Break as soon as the
            # touch registers.
            f_target = p.get("f_stop", 0.0) or 0.1
            # GRIP MAINTENANCE: the finger bite decays with time
            # (contact relaxation; padF 2.3 -> 1.8N over ~300 steps)
            # and a fully decayed bite lets the polished rod slide out
            # of the pads.  Top it up CONSERVATIVELY: only when it has
            # fallen well below the carry level, and only by a small
            # increment -- an aggressive re-clench on the hanging rod
            # ejects it sideways (measured: tilt 2 -> 90deg, resid
            # 42mm, one reclench pulse).  During the descent the rod
            # rides with the arm (touchdown speed = arm speed), so the
            # press only needs the bite to survive the loop duration.
            grip_hold = p.get("grip_hold", 1.0)

            def _reclench():
                pf_ = (self.ctx.geom_contact_force("finger1_pad_collision")
                       + self.ctx.geom_contact_force("finger2_pad_collision"))
                if pf_ < grip_hold:
                    c1 = float(self.ctx.data.ctrl[self.ctx.finger_act_ids[0]])
                    self.ctx.set_finger_ctrl(max(c1 - 0.0005, 0.0),
                                             -max(c1 - 0.0005, 0.0))
            _reclench()
            cmd_z = self.ctx.eef_pos()[2]
            # floor includes over_travel: with a sliding grip the press
            # must keep going PAST touchdown (friction plateau ~0.26N)
            # until the shoulder step jams on the pad-band bottom edge
            # and the load path turns geometric (see the rigid branch)
            z_floor = eef_zt_live - p.get("over_travel", 0.0)
            slices = 0
            touched = False
            _dumped = False
            # contact peak: the seated connector creeps in its seat
            # under the static press (measured: 0.10N -> 0.03N, resid
            # 0.2 -> 1.4mm over a 30-step settle), so the continuity
            # evidence is the TOUCH (tip on pad) -- judged from the
            # peak force and the resid at the break.
            f_peak = 0.0
            r_best = 9.9
            for _ in range(600):
                if not _grip_alive():
                    return _abort(f"grip lost during press (k={slices})")
                tnow = np.array(self.ctx.data.geom_xpos[gid])
                exy = nom_xy - tnow[:2]
                fnow = self.ctx.geom_contact_force(pad_geom)
                f_peak = max(f_peak, fnow)
                r_cur = float(np.linalg.norm(exy))
                if verbose and slices % 20 == 0:
                    pz_ = self.ctx.data.geom_xpos[
                        self.ctx.geom_id(pad_geom)][2]
                    print(f"  [slice k={slices}] f={fnow:.2f}N "
                          f"tip_z={tnow[2]:.4f} cmd_z={cmd_z:.4f} "
                          f"eef_z={self.ctx.eef_pos()[2]:.4f} "
                          f"rxy={np.linalg.norm(exy)*1000:.2f}mm "
                          f"padz={pz_:.4f} "
                          f"tilt={self.ctx.obj_tilt(part):.2f} "
                          f"htilt={self._tilt_of('housing'):.2f} "
                          f"ctilt={self._tilt_of('connector'):.2f}")
                if fnow >= f_target and np.linalg.norm(exy) < 0.003:
                    r_best = r_cur
                    if verbose:
                        tz3_ = np.array(self.ctx.data.geom_xpos[gid])
                        pz3_ = np.array(self.ctx.data.geom_xpos[
                            self.ctx.geom_id(pad_geom)])
                        print(f"  [slice] break k={slices} f={fnow:.2f}N "
                              f"tip_z={tnow[2]:.4f}")
                        print(f"  [GEOMDBG] tip={np.round(tz3_,4)} "
                              f"pad={np.round(pz3_,4)} "
                              f"conn={np.round(self._pos_of('connector'),4)} "
                              f"conn_tilt={self._tilt_of('connector'):.2f} "
                              f"housing_tilt={self._tilt_of('housing'):.2f}")
                    break
                if (not _dumped
                        and self._tilt_of('connector') > 0.3):
                    _dumped = True
                    import mujoco as _mj2
                    pairs = []
                    for ci in range(self.ctx.data.ncon):
                        c = self.ctx.data.contact[ci]
                        g1 = self.ctx.model.geom(c.geom1).name
                        g2 = self.ctx.model.geom(c.geom2).name
                        ff = np.zeros(6)
                        _mj2.mj_contactForce(self.ctx.model,
                                             self.ctx.data, ci, ff)
                        if np.linalg.norm(ff[:3]) > 0.02:
                            pairs.append(f"{g1}<>{g2}:"
                                         f"{np.linalg.norm(ff[:3]):.2f}")
                    print(f"  [CONDUMP k={slices}] "
                          f"tip={np.round(tnow,4)} "
                          f"ctilt={self._tilt_of('connector'):.2f} "
                          f"{' '.join(pairs)}", flush=True)
                if not touched and fnow > 0.05:
                    # first contact: rebase the command on the live z
                    # so the press ramps from zero error
                    touched = True
                    cmd_z = self.ctx.eef_pos()[2]
                eef_now = self.ctx.eef_pos()
                gap = (tnow[2] - 0.0045 -
                       self.ctx.data.geom_xpos[
                           self.ctx.geom_id(pad_geom)][2] - 0.001)
                if touched:
                    # CREEP once the tip LOADS the pad (touched fires
                    # at fnow>0.05).  The seated connector foot rides
                    # the seat ring at mu=0.15 (7.7g part) and the tip
                    # hangs 2.4deg off vertical, so ANY touchdown
                    # faster than ~1mm/s throws a lateral edge impulse
                    # that slides the foot and topples the connector
                    # in ONE physics step (per-step trace: cFy
                    # -0.1..-0.3N, pad +1mm in y, ctilt 0.03 -> 4deg
                    # at only 0.2N).  The 0.01mm slices cap the arm
                    # speed at ~0.5mm/s, at which the same stack holds
                    # (xfrc: 0.8N at 2.5deg lean, ctilt peak 0.05deg).
                    # A 0.2mm-slice burst runs 50-100mm/s -- that was
                    # the kick.
                    #
                    # FORCE-SCHEDULED SPEEDUP: the grip bite bleeds
                    # out in ~2.5s (rod drops out of the pads), so the
                    # press must reach f_target QUICKLY; but the pad
                    # force itself buys stability -- the foot friction
                    # grows as 0.15*(F+weight), so bigger slices are
                    # safe once F is up.  Measured: a 0.5mm/s touchdown
                    # never kicked (ctilt peak 0.39deg).
                    if fnow < 0.15:
                        # light touch: brief buffer step, then full
                        # slide speed.  The rod rides up through the
                        # pads on the ~0.26N friction plateau and must
                        # reach the shoulder jam (12mm of slide) before
                        # the bite ages out (~900 hold steps): the old
                        # 0.01mm creep took 500+ steps and the rod
                        # dropped at k=143 with the force still at
                        # 0.3N (measured).
                        step = 0.00002
                    elif fnow < 0.4:
                        step = 0.0001
                    else:
                        step = 0.0001
                else:
                    # APPROACH phase, 0.1mm/step: fast enough to reach
                    # the pad inside the bite's lifetime (~900 hold
                    # steps; the old gap-based creep cut to 0.01mm/step
                    # with the tip still 0.5mm off the pad and the rod
                    # dropped mid-creep -- measured at k=117), smooth
                    # enough not to roll the rod out (0.2mm/step pulse
                    # train dropped it at k=57 -- measured).
                    step = 0.0001
                # last 0.5mm before the pad face: slow buffer -- the
                # rod rolls in the pads during a fast approach (tilt
                # 0.7 -> 1.5deg, rxy 0.04 -> 0.85mm over the final
                # 2mm -- measured), and touchdown offset levers the
                # seated connector over.  A 0.02mm/step creep over the
                # last 0.5mm keeps the centred tip centred.
                if not touched and gap <= 0.0005:
                    step = min(step, 0.00002)
                cmd_z = max(cmd_z - step, z_floor)
                # xy servo only while the tip is clearly ABOVE the pad
                # (gap > 0.5mm), then FREEZE for the press: the rod is
                # plumb (levelled at the hover) and the press is axial,
                # so the centred tip stays centred; a lateral servo
                # step through the rigid rod would shove the pad
                # sideways and lever the seated connector over
                # (measured: ctilt 4.4deg from a 0.2mm-cap servo).
                if gap > 0.0005 and not touched:
                    dxy = np.array([np.clip(exy[0], -0.001, 0.001),
                                    np.clip(exy[1], -0.001, 0.001)])
                else:
                    dxy = np.zeros(2)
                d = (np.array([dxy[0], dxy[1], 0.0])
                     + ax * (cmd_z - eef_now[2]))
                self.arm.move_eef(eef_now + d, gain=6.0, tol=0.0,
                                  max_steps=1, stall=False)
                slices += 1
                if slices % 24 == 0:
                    _reclench()
        settle(self.ctx, max_steps=10)      # brief settle; the seated
                                            # connector creeps under the
                                            # static press, so keep it
                                            # short
        if verbose:
            txy = np.array(self.ctx.data.geom_xpos[gid])[:2]
            print(f"  [continuity] post-descent resid="
                  f"{np.linalg.norm(txy - nom_xy)*1000:.2f}mm "
                  f"tip_z={self.ctx.data.geom_xpos[gid][2]:.4f} "
                  f"eef={np.round(self.ctx.eef_pos(), 4)}")

        tip = np.array(self.ctx.data.geom_xpos[gid])
        resid = float(np.linalg.norm(tip[:2] - nom_xy))
        force = self.ctx.geom_contact_force(pad_geom)
        # judgement on the CONTACT PEAK (touch force + break resid):
        # the tip-pad touch at ~0.1N is the continuity evidence; the
        # seated connector then creeps in its seat under the static
        # load, so the post-settle force reads lower (measured
        # 0.10 -> 0.03N over 30 steps).
        quality = float(np.clip(f_peak / 1.0, 0.0, 1.0))
        self.state["test_force"] = float(f_peak)
        self.state["test_resid"] = r_best
        self.state["contact_quality"] = quality
        # demo gate: 0.1N tip-pad contact (a reliable electrical touch
        # for the continuity test; pressing harder levers the seated
        # connector over -- the 0.5N gate belongs to the research
        # study) and the tip within 3mm of the pad centre at the break.
        ok = quality >= 0.1 and r_best < 0.003
        self.arm.rot_target = rot0
        if verbose:
            print(f"  [continuity] force={force:.2f}N "
                  f"resid={resid * 1000:.2f}mm quality={quality:.2f} "
                  f"tip_z={tip[2]:.4f} "
                  f"pad_top={pad_top:.4f} "
                  f"{'PASS' if ok else 'FAIL'}")
        return bool(ok)

    # ================================================ two-class actions
    # Planning actions (no physics motion; they write plan artifacts
    # into self.state["plans"]) and their execution counterparts.  All
    # fault sources are sampled from self.faults so a run is
    # deterministic per (profile, seed) and attributable via the log.

    # -- scatter_parts (EXEC): seed-controlled incoming-part scatter
    def _skill_scatter_parts(self, p, verbose):
        parts = p.get("parts", [])
        for name in parts:
            j = self.faults.init_jitter()
            pos = self.ctx.obj_pos(name)
            self.ctx.set_obj_pose(name, pos[:2] + j[:2], pos[2])
        settle(self.ctx, max_steps=10)
        if verbose:
            jit = self.faults.params["init_jitter"]
            print(f"  [scatter_parts] jitter std {jit * 1000:.2f} mm")
        return True

    # -- detect_part (EXEC): perception with miss/false/noise faults
    def _skill_detect_part(self, p, verbose):
        name = p["part"]
        attempts = int(p.get("retries", 1)) + 1
        for k in range(attempts):
            found, pos, yaw, outcome = perception.detect_part(
                self.ctx, name, faults=self.faults, step=self._step_idx)
            if found:
                perc = self.state.setdefault("percepts", {})
                perc[name] = dict(pos=np.asarray(pos, dtype=float),
                                  yaw=float(yaw), outcome=outcome)
                if verbose:
                    print(f"  [detect {name}] at {np.round(pos, 4)} "
                          f"({outcome})")
                return True
            if verbose:
                print(f"  [detect {name}] {outcome} -- retry")
            settle(self.ctx, max_steps=2)
        if verbose:
            print(f"  [detect {name}] FAIL: part not found")
        return False

    # -- plan_grasp_pose (PLAN): multi-candidate grasp estimation +
    #    scoring; the best feasible candidate becomes the artifact
    def _skill_plan_grasp_pose(self, p, verbose):
        name = p["part"]
        perc = (self.state.get("percepts", {}) or {}).get(name)
        if perc is None or perc.get("pos") is None:
            # no percept recorded: detect inline once
            found, pos, yaw, outcome = perception.detect_part(
                self.ctx, name, faults=self.faults, step=self._step_idx)
            if not found:
                self.faults.record(
                    "grasp_est_fail",
                    f"step {self._step_idx}: no detection for {name}")
                return False
            perc = dict(pos=pos, yaw=yaw, outcome=outcome)
        meta = perception.part_meta(self.ctx, name)
        cands = planning.grasp_pose_candidates(
            self.ctx, name, meta,
            detect_result=(True, perc["pos"], perc["yaw"], "ok"),
            n_yaw=p.get("n_yaw", 8),
            obstacles=self._scene_obstacles())
        if not cands or all(not c["feasible"] for c in cands):
            self.faults.record(
                "grasp_est_fail",
                f"step {self._step_idx}: no feasible grasp for {name}")
            return False
        gp = cands[0]
        if p.get("grasp_dz") is not None:
            gp["pos"][2] = self.ctx.obj_pos(name)[2] + p["grasp_dz"]
        if p.get("yaw") is not None:
            gp["approach_yaw"] = p["yaw"]
        plans = self.state.setdefault("plans", {})
        plans["grasp"] = gp
        plans["grasp_part"] = name
        # keep the full candidate branch for attribution / data export
        plans["grasp_candidates"] = [
            {k: c[k] for k in ("pos", "yaw", "score", "feasible",
                               "reason") if k in c}
            for c in cands]
        if verbose:
            print(f"  [plan_grasp_pose {name}] "
                  f"pos={np.round(gp['pos'], 4)} yaw={gp['yaw']:.2f} "
                  f"(best of {len(cands)} candidates, "
                  f"score={gp['score']:.2f})")
        return True

    # -- plan_path (PLAN): waypoints + reachability + collision gate
    def _plan_target(self, ref):
        """Resolve a plan_path 'to' ref to a world xyz target."""
        if isinstance(ref, dict) and "part" in ref:
            pos = self.ctx.obj_pos(ref["part"])
            return np.array([pos[0], pos[1],
                             pos[2] + float(ref.get("lift", 0.0))])
        if isinstance(ref, dict):
            xy = self._resolve_xy(ref.get("at", ""))
            z = (self._resolve_z(ref["z"])
                 if ref.get("z") is not None else self.ctx.eef_pos()[2])
            return np.array([xy[0], xy[1], z + float(ref.get("lift", 0.0))])
        if isinstance(ref, str):
            return np.append(self._resolve_xy(ref), self.ctx.eef_pos()[2])
        return np.asarray(ref, dtype=float)

    def _scene_obstacles(self, margin=0.0):
        """World-frame AABBs of the static wall geoms (tray walls, pin
        sleeve) for the path collision check."""
        out = []
        m = self.ctx.model
        for gid in range(m.ngeom):
            name = m.geom(gid).name or ""
            if not (name.startswith("tray_wall")
                    or name.startswith("pin_sleeve_w")):
                continue
            pos = np.array(self.ctx.data.geom_xpos[gid])
            size = np.array(m.geom_size[gid])
            R = np.array(self.ctx.data.geom_xmat[gid]).reshape(3, 3)
            ext = np.abs(R) @ size
            out.append(dict(name=name,
                            lo=pos - ext - margin,
                            hi=pos + ext + margin))
        return out

    def _skill_plan_path(self, p, verbose):
        start = self.ctx.eef_pos()
        target = self._plan_target(p["to"])
        style = p.get("style", "safe_z")
        attempts = int(p.get("retries", 1)) + 1
        styles = [style] + [s for s in ("clearance",) if s != style]
        obstacles = self._scene_obstacles()
        for k in range(attempts):
            st = styles[k] if k < len(styles) else styles[-1]
            outcome = self.faults.plan_path_outcome(step=self._step_idx)
            # multi-candidate branch: height variants x lateral via
            # offsets, all sharing the goal descent column
            cands = planning.path_candidates(
                start, target, style=st,
                lift=p.get("lift", 0.08), safe_z=p.get("safe_z"))
            scored = []
            for wp in cands:
                # the segment list includes the start->first-waypoint leg
                hits = planning.check_path([start] + [w for w in wp],
                                           obstacles)
                scored.append(dict(waypoints=wp.as_dict()["waypoints"],
                                   style=st, score=wp.score,
                                   valid=not hits,
                                   hits=[h[1] for h in hits]))
            feasible = [s for s in scored if s["valid"]]
            if not feasible:
                if verbose:
                    print(f"  [plan_path] all {len(scored)} candidates "
                          f"collide -- replan")
                continue
            if outcome != "ok":
                if verbose:
                    print(f"  [plan_path] {outcome} -- replan")
                continue
            best = min(feasible, key=lambda s: s["score"])
            name = p.get("as", "path")
            plans = self.state.setdefault("plans", {})
            plans[name] = dict(waypoints=best["waypoints"],
                               style=best["style"], valid=True,
                               reason="ok")
            plans[name + "_candidates"] = scored
            if verbose:
                print(f"  [plan_path] -> '{name}' style={st} "
                      f"{len(best['waypoints'])} waypoints "
                      f"(best of {len(scored)} candidates)")
            return True
        if verbose:
            print(f"  [plan_path] FAIL after {attempts} attempt(s)")
        return False

    # -- move_to (EXEC): execute the planned path with a move residual
    def _skill_move_to(self, p, verbose):
        plan = self.state.get("plans", {}).get(p.get("plan", "path"))
        if plan is None or not plan.get("valid", False):
            if verbose:
                print(f"  [move_to] no valid path plan "
                      f"'{p.get('plan', 'path')}' -- abort")
            return False
        err = self.faults.move_error()
        wps = plan["waypoints"]
        ok = True
        for i, w in enumerate(wps):
            tgt = np.asarray(w, dtype=float)
            if i == len(wps) - 1:
                tgt = tgt + err          # move-to-pose residual
            if not self.arm.move_eef(tgt, tol=p.get("tol", 0.008),
                                     gain=p.get("gain"),
                                     max_speed=p.get("max_speed"),
                                     smooth=True):
                ok = False
                break
        if verbose:
            print(f"  [move_to] {'ok' if ok else 'FAIL'} "
                  f"eef={np.round(self.ctx.eef_pos(), 4)}")
        return ok

    # -- transport (EXEC): carry a held part along the planned path.
    #    The slip fault weakens the bite; the drop (if any) is real
    #    physics, and the hold check reports it.
    def _skill_transport(self, p, verbose):
        name = p["part"]
        plan = self.state.get("plans", {}).get(p.get("plan", "path_carry"))
        if plan is None or not plan.get("valid", False):
            if verbose:
                print("  [transport] no valid carry plan -- abort")
            return False
        slip = self.faults.slip_event(step=self._step_idx)
        if slip:
            # weaken the bite: command the pads ~2.5mm looser than the
            # achieved squeeze so the grip force drops toward zero and
            # a heavy part may slide out under inertia (real physics)
            q = self.ctx.finger_qpos
            q1 = min(0.5 * (q[0] - q[1]) + 0.0025, self.gripper.open_q)
            self.ctx.set_finger_ctrl(q1, -q1)
            settle(self.ctx, max_steps=3)
        ok = True
        for w in plan["waypoints"]:
            if not self.arm.move_eef(np.asarray(w, dtype=float),
                                     tol=p.get("tol", 0.004), gain=6.0,
                                     max_speed=p.get("carry_speed", 0.12),
                                     smooth=True):
                ok = False
                break
        if slip and ok:
            # re-establish the bite after the slip-prone leg
            meta = perception.part_meta(self.ctx, name)
            self.gripper.close_on_part(meta["outer_d"],
                                       press=p.get("press", 0.0015))
        held = (np.linalg.norm(self.ctx.eef_pos()[:2]
                               - self.ctx.obj_pos(name)[:2]) < 0.03)
        self.state["transport_held"] = bool(held)
        if verbose:
            print(f"  [transport {name}] "
                  f"{'ok' if (ok and held) else 'FAIL'} held={held}")
        return ok and held

    # -- inspect (EXEC): quality check with sensor noise + verdict
    def _skill_inspect(self, p, verbose):
        mode = p.get("mode", "final")
        if mode == "shaft":
            part = p["part"]
            nom = self._resolve_xy(p.get("ref", "tray_center"))
            n = self.faults.measure_noise()
            m = self.ctx.obj_pos(part)[:2] + n[:2]
            err = float(np.linalg.norm(m - nom))
            tag = p.get("tag", f"{part}_ok")
            self.state[tag] = err < float(p.get("xy_tol", 0.004))
            self.state[f"{part}_measured_err"] = err
            if verbose:
                print(f"  [inspect {part}] "
                      f"measured_err={err * 1000:.2f}mm "
                      f"-> {'OK' if self.state[tag] else 'REWORK'}")
            return True
        if mode == "final":
            metrics = {}
            verdict = True
            for c in p.get("checks", []):
                name = c["part"]
                kind = c.get("kind", "seat")
                n = self.faults.measure_noise()
                if kind == "seat":
                    pos = self.ctx.obj_pos(name)
                    ref_xy = self._resolve_xy(c["xy_ref"])
                    z_ref = self._resolve_z(c["z_ref"])
                    xy_err = float(np.linalg.norm(
                        pos[:2] + n[:2] - ref_xy))
                    z_err = float(pos[2] + n[2] - z_ref)
                    tilt = float(self.ctx.obj_tilt(name))
                    # two-sided z gate: a loose ring may rest either
                    # side of the nominal seat (measured on the
                    # retainer riding up the shaft end)
                    ok = (xy_err < c["xy_tol"]
                          and abs(z_err) <= c["z_tol"]
                          and tilt < c["tilt_tol"])
                    metrics[name] = dict(xy_err=xy_err, z_err=z_err,
                                         tilt=tilt, ok=ok)
                elif kind == "pin_engage":
                    pos, _ = self.ctx.obj_pose(name)
                    axis = self.ctx.obj_axis(name)
                    bot = pos - float(c["half"]) * axis
                    axis_ref = self._ref_to_xy(c["axis_ref"])
                    top_ref = self._resolve_z(c["top_ref"])
                    depth = float(top_ref - bot[2])
                    bot_off = float(np.linalg.norm(
                        bot[:2] + n[:2] - axis_ref))
                    ok = (depth >= c["min_depth"]
                          and bot_off < c["bot_off_tol"])
                    metrics[name] = dict(depth=depth, bot_off=bot_off,
                                         ok=ok)
                else:
                    raise ValueError(f"unknown inspect kind {kind!r}")
                verdict = verdict and ok
                if verbose:
                    print(f"  [inspect {name}] {metrics[name]} "
                          f"-> {'OK' if ok else 'NG'}")
            self.state["inspect_metrics"] = metrics
            self.state["inspect_verdict"] = bool(verdict)
            if verbose:
                print(f"  [inspect] verdict={'PASS' if verdict else 'FAIL'}")
            return True
        raise ValueError(f"unknown inspect mode {mode!r}")

    # -- nudge (EXEC): conditional rework push; skipped when the
    #    guarded state flag is already OK
    def _skill_nudge(self, p, verbose):
        unless = p.get("unless")
        if unless is not None and self.state.get(unless, True):
            if verbose:
                print(f"  [nudge] state '{unless}' OK -- skipped")
            return True
        return manipulation.push(self.ctx, self.arm, self.gripper,
                                 p["part"],
                                 to_target=self._resolve_xy(p["to_target"]),
                                 push_z=(self._resolve_z(p["push_z"])
                                         if "push_z" in p else None),
                                 speed=p.get("speed", 0.04),
                                 verbose=verbose)

    # ================================================================ table
    _SKILLS = {
        "grasp":           _skill_grasp,
        "place":           _skill_place,
        "insert":          _skill_insert,
        "push":            _skill_push,
        "screw_drive":     _skill_screw_drive,
        "lateral_insert":  _skill_lateral_insert,
        "snap_press":      _skill_snap_press,
        "press_fit":       _skill_press_fit,
        "slide":           _skill_slide,
        "rotate":          _skill_rotate,
        "drive_position":  _skill_drive_position,
        "drive_force":     _skill_drive_force,
        "settle":          _skill_settle,
        "home":            _skill_home,
        "release":         _skill_release,
        "move":            _skill_move,
        "settle_press":    _skill_settle_press,
        "detect_datum":    _skill_detect_datum,
        "move_tool":       _skill_move_tool,
        "continuity_test": _skill_continuity_test,
        "pad_touch":       _skill_pad_touch,
        "scatter_parts":   _skill_scatter_parts,
        "detect_part":     _skill_detect_part,
        "move_to":         _skill_move_to,
        "transport":       _skill_transport,
        "inspect":         _skill_inspect,
        "nudge":           _skill_nudge,
    }

    # planning actions: pure computation, no physics motion; write plan
    # artifacts into self.state["plans"] (see the class docstring)
    _PLAN_SKILLS = {
        "plan_grasp_pose": _skill_plan_grasp_pose,
        "plan_path":       _skill_plan_path,
    }
