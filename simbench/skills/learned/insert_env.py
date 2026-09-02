"""Peg-in-hole insertion environment + observation/action contract.

This module defines the shared contract the learned insertion skill is
built on:

  ``build_obs(ctx, name, hole_xy, to_z, half)``  -> observation vector
  ``action_to_delta(act)``                        -> EEF xyz delta
  ``InsertEnv``                                   -> a gymnasium Env that
      trains a policy to "hold the peg and insert it into the given xyz
      hole", with contact / jam / tilt-deviation feedback in the reward.

The observation is SCENE-AGNOSTIC (it only reads the peg body, the hole
xy, the seat z and the peg half-height), so the SAME policy trained on
the minimal ``peg_in_hole.xml`` deploys through
``extension.peg_insert(mode='policy')`` in any scene that exposes a peg
body + a hole xy + a seat z (e.g. the Task A latch pin / gearbox rings).

gymnasium is preferred; a fall-back to the (already installed) ``gym``
0.26 API keeps the module importable without gymnasium.
"""
import os

import numpy as np
import mujoco

try:                                     # prefer the maintained API
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:                      # fall back to gym 0.26
    import gym
    from gym import spaces

# ------------------------------------------------------------- obs / act
OBS_DIM = 8
ACT_DIM = 3

# observation normalisation (raw SI -> O(1) inputs for the policy net)
_DIST_SCALE = 200.0        # 5 mm lateral / height error -> 1.0
_AXIS_SCALE = 5.0          # ~11 deg peg tilt -> 1.0
_FORCE_SCALE = 0.2         # 5 N contact -> 1.0
_FORCE_CLIP = 3.0

# action -> EEF delta (m per unit action, clipped to [-1, 1])
DELTA_SCALE = np.array([0.001, 0.001, 0.001])

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SCENE = os.path.join(os.path.dirname(os.path.dirname(_HERE)),
                             "scenes", "peg_in_hole.xml")


# ------------------------------------------------------------- contract
def _body_contact_force(ctx, name, exclude_gripper=True):
    """Contact-force magnitude (N) on body *name* from the ENVIRONMENT.

    With ``exclude_gripper`` the pad/finger/hand contacts are dropped, so
    the returned force is the peg-vs-hole (jam / wedge) signal -- NOT the
    constant grip bite (~4-7 N) that would otherwise dominate both the
    observation and the reward's contact penalty.
    """
    bid = ctx.body_id(name)
    f = 0.0
    buf = np.zeros(6)
    for i in range(ctx.data.ncon):
        c = ctx.data.contact[i]
        on_peg = (ctx.model.geom_bodyid[c.geom1] == bid
                  or ctx.model.geom_bodyid[c.geom2] == bid)
        if not on_peg:
            continue
        if exclude_gripper:
            other = (c.geom2 if ctx.model.geom_bodyid[c.geom1] == bid
                     else c.geom1)
            oname = ctx.model.geom(other).name or ""
            if any(k in oname for k in ("finger", "pad", "hand")):
                continue
        mujoco.mj_contactForce(ctx.model, ctx.data, i, buf)
        f += float(np.linalg.norm(buf[:3]))
    return f


def build_obs(ctx, name, hole_xy, to_z, half):
    """Scene-agnostic insertion observation (OBS_DIM,).

    Layout (all normalised to ~O(1)):
      [0:2] peg BOTTOM lateral error from the hole axis (x, y)
      [2]   peg bottom height above the seat (descent remaining)
      [3:5] peg axis deviation from world vertical (x, y) -- tilt
      [5]   contact force on the peg (jam / wedge signal)
      [6:8] EEF lateral offset from the hole axis (control reference)

    The peg bottom (centre - half*axis) is the insertion-relevant point:
    a peg hung on the ~7 deg EEF tilt carries its bottom several mm off
    the EEF axis, and it is the BOTTOM that must enter the bore.
    """
    hole_xy = np.asarray(hole_xy, dtype=float)[:2]
    pos = ctx.obj_pos(name)
    axis = ctx.obj_axis(name)
    bottom = pos - float(half) * axis
    lat = bottom[:2] - hole_xy
    height = float(bottom[2] - float(to_z))
    eef_lat = ctx.eef_pos()[:2] - hole_xy
    force = _body_contact_force(ctx, name)
    obs = np.array([
        lat[0] * _DIST_SCALE,
        lat[1] * _DIST_SCALE,
        height * _DIST_SCALE,
        float(axis[0]) * _AXIS_SCALE,
        float(axis[1]) * _AXIS_SCALE,
        float(np.clip(force * _FORCE_SCALE, 0.0, _FORCE_CLIP)),
        eef_lat[0] * _DIST_SCALE,
        eef_lat[1] * _DIST_SCALE,
    ], dtype=np.float32)
    return obs


def action_to_delta(act):
    """Map a policy action (ACT_DIM,) in [-1, 1] to an EEF xyz delta (m)."""
    a = np.clip(np.asarray(act, dtype=float).reshape(-1)[:ACT_DIM],
                -1.0, 1.0)
    if a.size < ACT_DIM:
        a = np.concatenate([a, np.zeros(ACT_DIM - a.size)])
    return a * DELTA_SCALE


# ---------------------------------------------------------------- env
class InsertEnv(gym.Env):
    """Gymnasium env: hold the peg and insert it into the hole.

    Reset grips the peg and hangs it above the bore with a randomised
    lateral offset + small tilt perturbation (on top of the natural ~7
    deg EEF tilt).  Each step applies ``action_to_delta`` as a single
    closed-loop EEF increment while holding the grip command, so the
    peg translates rigidly with the pads.  The reward is potential-based
    shaping on (lateral error + height) minus contact-force (jam),
    excess-tilt and step penalties, with a seating bonus -- explicitly
    covering the "contact / jam / pose-deviation" feedback the insertion
    task needs.

    Termination: seated (success), peg slipped out of the grip, or the
    peg tilted past ``tilt_fail_deg``.  Truncation: ``max_steps``.
    """

    metadata = {"render_modes": []}

    # reward weights (tunable)
    W_LAT = 50.0            # potential weight on lateral error (per m)
    W_HEIGHT = 20.0         # potential weight on height above seat (per m)
    C_STEP = 0.004          # per-step penalty
    C_FORCE = 0.06          # penalty per N of contact above f_free
    F_FREE = 1.0            # contact force below this is free (not jammed)
    C_TILT = 0.01           # penalty per deg of tilt above tilt_free
    TILT_FREE = 12.0        # tilt below this is not penalised (deg)
    R_SEAT = 1.0            # seating success bonus
    R_FAIL = -0.4           # slip / excessive-tilt failure penalty

    SEAT_HEIGHT_TOL = 0.0015    # bottom within 1.5mm of seat = seated
    SEAT_LAT_TOL = 0.0035       # bottom within 3.5mm of axis = seated
    TILT_FAIL_DEG = 32.0        # peg tilted past this = failed

    def __init__(self, scene_path=None, max_steps=200, hold_press=0.0025,
                 peg_below=0.012, start_clearance=0.016,
                 rand_lateral=0.0020, rand_tilt_deg=2.5, randomize=True,
                 render_mode=None):
        super().__init__()
        from ...core.sim_context import MjContext
        from ...core.controller import CartesianController, Gripper
        from ...skills.motion import align_above
        from ...skills.settle import settle
        self._MjContext = MjContext
        self._align_above = align_above
        self._settle = settle

        self.scene_path = scene_path or DEFAULT_SCENE
        self.ctx = MjContext(self.scene_path)
        self.ctx.reset()
        self.arm = CartesianController(self.ctx)
        self.gripper = Gripper(self.ctx)

        self.max_steps = int(max_steps)
        self.hold_press = float(hold_press)
        self.peg_below = float(peg_below)
        self.start_clearance = float(start_clearance)
        self.rand_lateral = float(rand_lateral)
        self.rand_tilt_deg = float(rand_tilt_deg)
        self.randomize = bool(randomize)
        self.render_mode = render_mode

        # hole geometry read from the scene (site = nominal axis)
        self.hole_xy = self.ctx.site_pos("hole_center")[:2].copy()
        self.to_z = float(self.ctx.site_pos("hole_center")[2])
        from ...skills import perception
        self.half = float(perception.part_meta(self.ctx, "peg")["half_h"])
        self.outer_d = float(
            perception.part_meta(self.ctx, "peg")["outer_d"])

        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(
            -1.0, 1.0, shape=(ACT_DIM,), dtype=np.float32)

        self._hover_q = None       # cached hover joint config (fast reset)
        self._grip_cmd = (0.0, 0.0)
        self._step = 0
        self._prev_pot = 0.0

    # ------------------------------------------------------------ helpers
    def _hover_z(self):
        return self.to_z + self.start_clearance + self.peg_below + self.half

    def _goto_hover(self):
        """Position + fine-align the EEF above the hole (cached qpos)."""
        ctx = self.ctx
        if self._hover_q is not None:
            for i, a in enumerate(ctx.arm_qadr):
                ctx.data.qpos[a] = self._hover_q[i]
            ctx.hold_arm()
            mujoco.mj_forward(ctx.model, ctx.data)
            self._settle(ctx, max_steps=8)
        else:
            tgt = np.array([self.hole_xy[0], self.hole_xy[1],
                            self._hover_z()])
            self.arm.move_eef(tgt, tol=0.005, max_steps=400)
            self._align_above(self.arm, self.hole_xy, tol=0.0012,
                              max_steps=60)
            self._settle(ctx, max_steps=8)
            self._hover_q = ctx.arm_qpos.copy()

    def _set_peg_pose(self, pos, quat_wxyz):
        """Teleport the free-jointed peg (pos + full quat), zero vel."""
        ctx = self.ctx
        bid = ctx.body_id("peg")
        jid = ctx.model.body_jntadr[bid]
        qa = ctx.model.jnt_qposadr[jid]
        ctx.data.qpos[qa:qa + 3] = np.asarray(pos, dtype=float)
        ctx.data.qpos[qa + 3:qa + 7] = np.asarray(quat_wxyz, dtype=float)
        da = ctx.model.jnt_dofadr[jid]
        ctx.data.qvel[da:da + 6] = 0.0
        mujoco.mj_forward(ctx.model, ctx.data)

    def _grip_peg(self):
        """Latch the pads onto the teleported peg WITHOUT an open slew.

        ``gripper.open()`` / ``close_on_part()`` step the physics while
        the pads travel, and a freshly-teleported free peg free-falls out
        of the gap before the pads reach it.  Instead we write the finger
        JOINT qpos straight to the touch span (pads just contacting the
        peg, no penetration) and command a deeper ctrl so kp*(cmd-q)
        builds the bite over a short settle -- the peg is captured in
        place and never falls.
        """
        from ...core.controller import PAD_HALF_Y
        ctx = self.ctx
        g = self.gripper
        q_touch = g.q_from_span(self.outer_d + 2.0 * PAD_HALF_Y)
        q_press = g.q_from_span(self.outer_d - self.hold_press
                                + 2.0 * PAD_HALF_Y)
        ctx.data.qpos[ctx.finger_qadr[0]] = q_touch
        ctx.data.qpos[ctx.finger_qadr[1]] = -q_touch
        for jd in ctx.finger_joint_ids:
            ctx.data.qvel[ctx.model.jnt_dofadr[jd]] = 0.0
        q_cmd = max(q_press - g.CLOSE_OVERSHOOT, 0.0)
        self._grip_cmd = (q_cmd, -q_cmd)
        ctx.set_finger_ctrl(*self._grip_cmd)
        mujoco.mj_forward(ctx.model, ctx.data)
        self._settle(ctx, max_steps=10)

    def _held(self):
        """Peg still in the pads (near the EEF and pads loaded)."""
        d = float(np.linalg.norm(self.ctx.eef_pos()
                                 - self.ctx.obj_pos("peg")))
        return d < 0.035

    def _potential(self):
        pos = self.ctx.obj_pos("peg")
        axis = self.ctx.obj_axis("peg")
        bottom = pos - self.half * axis
        lat = float(np.linalg.norm(bottom[:2] - self.hole_xy))
        height = float(bottom[2] - self.to_z)
        return self.W_LAT * lat + self.W_HEIGHT * max(height, 0.0), \
            lat, height

    # ------------------------------------------------------------ gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        ctx = self.ctx
        ctx.reset()
        self.arm.rot_target = ctx.eef_mat().copy()
        self._goto_hover()

        # hang the peg BELOW the EEF along the tool-down axis with a
        # small randomised lateral offset + tilt perturbation.  NOTE the
        # EEF local +z points DOWN (tool-down home pose), so the peg is
        # offset along +z_local and oriented with its OWN local +z UP
        # (bottom end = local -z points down into the hole) -- matching
        # the scene-wide convention bottom = centre - half*axis.
        R = ctx.eef_mat()
        eef = ctx.eef_pos()
        zax, xax, yax = R[:, 2], R[:, 0], R[:, 1]
        if self.randomize:
            off = self.np_random.uniform(-self.rand_lateral,
                                         self.rand_lateral, 2)
            tilt = self.np_random.uniform(-self.rand_tilt_deg,
                                          self.rand_tilt_deg, 2)
        else:
            off = np.zeros(2)
            tilt = np.zeros(2)
        peg_center = (eef + zax * self.peg_below
                      + xax * off[0] + yax * off[1])
        # peg orientation = EEF orientation flipped so peg +z is UP,
        # with a small tilt perturbation for episode variety
        tx, ty = np.radians(tilt[0]), np.radians(tilt[1])
        Rx = np.array([[1, 0, 0], [0, np.cos(tx), -np.sin(tx)],
                       [0, np.sin(tx), np.cos(tx)]])
        Ry = np.array([[np.cos(ty), 0, np.sin(ty)], [0, 1, 0],
                       [-np.sin(ty), 0, np.cos(ty)]])
        D = np.diag([1.0, -1.0, -1.0])       # flip eef z (down) -> up
        Rpeg = R @ D @ Ry @ Rx
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, Rpeg.reshape(9))
        # capture the peg (retry the teleport+grip if it slipped)
        for _ in range(3):
            self._set_peg_pose(peg_center, q)
            self._grip_peg()
            if self._held():
                break
        self._step = 0
        self._prev_pot, _, _ = self._potential()
        obs = build_obs(ctx, "peg", self.hole_xy, self.to_z, self.half)
        return obs.astype(np.float32), self._info(seated=False)

    def step(self, action):
        ctx = self.ctx
        delta = action_to_delta(action)
        eef = ctx.eef_pos()
        self.arm.move_eef(eef + delta, gripper=self._grip_cmd, gain=8.0,
                          tol=0.0, max_steps=1, stall=False)
        self._step += 1

        pot, lat, height = self._potential()
        force = _body_contact_force(ctx, "peg")
        tilt = float(ctx.obj_tilt("peg"))
        held = self._held()
        seated = (height <= self.SEAT_HEIGHT_TOL
                  and lat <= self.SEAT_LAT_TOL and held)

        # potential-based shaping (progress = decrease in cost-to-go)
        reward = (self._prev_pot - pot) - self.C_STEP
        reward -= self.C_FORCE * max(0.0, force - self.F_FREE)
        reward -= self.C_TILT * max(0.0, tilt - self.TILT_FREE)
        self._prev_pot = pot

        terminated = False
        if seated:
            reward += self.R_SEAT
            terminated = True
        elif not held:
            reward += self.R_FAIL
            terminated = True
        elif tilt >= self.TILT_FAIL_DEG:
            reward += self.R_FAIL
            terminated = True
        truncated = self._step >= self.max_steps

        info = self._info(seated=seated, lat=lat, height=height,
                          force=force, tilt=tilt, held=held)
        obs = build_obs(ctx, "peg", self.hole_xy, self.to_z, self.half)
        return obs.astype(np.float32), float(reward), bool(terminated), \
            bool(truncated), info

    def _info(self, seated, lat=None, height=None, force=None, tilt=None,
              held=None):
        if lat is None:
            _, lat, height = self._potential()
            force = _body_contact_force(self.ctx, "peg")
            tilt = float(self.ctx.obj_tilt("peg"))
            held = self._held()
        return dict(seated=bool(seated), lateral=float(lat),
                    height=float(height), force=float(force),
                    tilt=float(tilt), held=bool(held),
                    depth=float(max(0.0, (self.to_z + self.start_clearance)
                                    - height)))
