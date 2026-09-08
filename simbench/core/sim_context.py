"""Pure-MuJoCo simulation context for simbench (no robosuite / LIBERO).

One :class:`MjContext` owns the compiled scene model + data and exposes:

  - a control-step clock (default 20 Hz) on top of ``mj_step``;
  - name-based lookups for actuators / joints / bodies / sites / geoms;
  - robot convenience accessors (EEF pose, arm qpos, pad span);
  - full-physics state save / restore (``mj_getState``/``mj_setState``);
  - optional per-control-step hooks (recorder, viewer sync).

Scene XML files are self-contained: they ``<include>`` the Panda asset and
define the table, parts, nominal-target sites and cameras.  All target
coordinates the skills execute against are read from the scene (sites),
never hard-coded here.
"""
import os
import copy

import mujoco
import numpy as np

# robosuite Panda home configuration (identical to the old framework so
# the workspace matches: base at (-0.56, 0, 0.912), EEF home z ~ 1.011)
HOME_QPOS = np.array([0.0, np.pi / 16.0, 0.0,
                      -np.pi / 2.0 - np.pi / 3.0, 0.0, np.pi - 0.2,
                      np.pi / 4.0])
HOME_FINGER = 0.020833          # symmetric half-open grip at reset

_HERE = os.path.dirname(os.path.abspath(__file__))


class MjContext:
    """Compiled scene + stepping clock + name lookups."""

    def __init__(self, scene_path, control_freq=20.0):
        scene_path = os.path.abspath(scene_path)
        self.scene_path = scene_path
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)
        self.dt = float(self.model.opt.timestep)
        self.control_dt = 1.0 / float(control_freq)
        self.nsub = max(1, int(round(self.control_dt / self.dt)))

        # hooks: callable(ctx) executed after every control step
        self.on_control_step = None
        self.n_control_steps = 0
        self.n_physics_steps = 0
        self._hook_substeps = 0
        self._state_clients = {}

        # robot references (fail fast when the scene forgot the Panda)
        def jid(name):
            i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT,
                                  name)
            if i < 0:
                raise ValueError(f"scene {scene_path} lacks joint '{name}'")
            return i

        def aid(name):
            i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                  name)
            if i < 0:
                raise ValueError(f"scene {scene_path} lacks actuator "
                                 f"'{name}'")
            return i

        self.arm_joint_ids = [jid(f"joint{i}") for i in range(1, 8)]
        self.arm_qadr = [self.model.jnt_qposadr[i]
                         for i in self.arm_joint_ids]
        self.arm_act_ids = [aid(f"act_j{i}") for i in range(1, 8)]
        self.finger_joint_ids = [jid("finger_joint1"),
                                 jid("finger_joint2")]
        self.finger_qadr = [self.model.jnt_qposadr[i]
                            for i in self.finger_joint_ids]
        self.finger_act_ids = [aid("act_finger1"), aid("act_finger2")]
        self.eef_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "grip_site")
        if self.eef_site_id < 0:
            raise ValueError(f"scene {scene_path} lacks site 'grip_site'")

        self.n_arm = 7

    # ------------------------------------------------------------ stepping
    def phys_step(self, n=1):
        """Raw physics steps (no hooks)."""
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
        self.n_physics_steps += n

    def step(self, nsub=None):
        """One control step = ``nsub`` physics steps + hook dispatch."""
        if nsub is None:
            nsub = self.nsub
        self.step_physics_observed(nsub)

    def step_physics_observed(self, n=1):
        """Physics-rate feedback with the same sampling clock as normal motion."""
        for _ in range(int(n)):
            self.phys_step()
            self._hook_substeps += 1
            if self._hook_substeps >= self.nsub:
                self._hook_substeps = 0
                self.n_control_steps += 1
                if self.on_control_step is not None:
                    self.on_control_step(self)

    def register_state_client(self, name, getter, setter):
        self._state_clients[name] = (getter, setter)

    def snapshot(self):
        """Replay checkpoint: integration inputs AND controller memory.

        Model parameters must be identical on restore; randomised model arrays
        are captured too so candidate comparisons cannot inherit another run.
        Legacy get_state remains a kinematics-only compatibility interface.
        """
        fields = ('qpos', 'qvel', 'act', 'ctrl', 'qacc_warmstart',
                  'qfrc_applied', 'xfrc_applied', 'mocap_pos', 'mocap_quat',
                  'userdata', 'eq_active', 'plugin_state')
        model_fields = ('geom_friction', 'body_mass', 'body_inertia',
                        'dof_damping', 'eq_active')
        return dict(version=1, scene_path=self.scene_path,
                    numpy_rng=copy.deepcopy(np.random.get_state()),
                    data={k: getattr(self.data, k).copy() for k in fields
                          if hasattr(self.data, k)}, time=float(self.data.time),
                    model={k: getattr(self.model, k).copy() for k in model_fields
                           if hasattr(self.model, k)},
                    clock=(self.n_control_steps, self.n_physics_steps,
                           self._hook_substeps),
                    clients={k: copy.deepcopy(g()) for k, (g, _) in
                             self._state_clients.items()})

    def restore(self, state):
        if state.get('version') != 1 or state['scene_path'] != self.scene_path:
            raise ValueError('checkpoint version/scene mismatch')
        for name, value in state['model'].items():
            getattr(self.model, name)[:] = value
        for name, value in state['data'].items():
            getattr(self.data, name)[:] = value
        self.data.time = state['time']
        if 'numpy_rng' in state:
            np.random.set_state(state['numpy_rng'])
        self.n_control_steps, self.n_physics_steps, self._hook_substeps = state['clock']
        mujoco.mj_forward(self.model, self.data)
        # mj_forward may consume/overwrite solver warm starts. Preserve the
        # exact integration input from the branch point after the forward pass.
        self.data.qacc_warmstart[:] = state['data']['qacc_warmstart']
        for name, value in state['clients'].items():
            if name not in self._state_clients:
                raise ValueError(f'missing checkpoint client: {name}')
            self._state_clients[name][1](copy.deepcopy(value))

    # --------------------------------------------------------- name lookups
    def body_id(self, name):
        i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if i < 0:
            raise ValueError(f"no body named '{name}'")
        return i

    def site_id(self, name):
        i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if i < 0:
            raise ValueError(f"no site named '{name}'")
        return i

    def geom_id(self, name):
        i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if i < 0:
            raise ValueError(f"no geom named '{name}'")
        return i

    def act_id(self, name):
        i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                              name)
        if i < 0:
            raise ValueError(f"no actuator named '{name}'")
        return i

    def joint_id(self, name):
        i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if i < 0:
            raise ValueError(f"no joint named '{name}'")
        return i

    # ------------------------------------------------------------ robot IO
    @property
    def arm_qpos(self):
        return np.array([self.data.qpos[a] for a in self.arm_qadr])

    @property
    def finger_qpos(self):
        return np.array([self.data.qpos[a] for a in self.finger_qadr])

    def set_arm_ctrl(self, q, gravity_comp=True):
        """Command the 7 arm position actuators (rad).

        With ``gravity_comp`` the joint-gravity load (qfrc_bias) is fed
        forward as a ctrl offset ``bias/kp`` -- without it the finite
        servo stiffness leaves a ~cm steady-state EEF sag under gravity
        (measured in bring-up).
        """
        for i, a in enumerate(self.arm_act_ids):
            cmd = q[i]
            if gravity_comp:
                dof = self.model.jnt_dofadr[self.arm_joint_ids[i]]
                kp = self.model.actuator_gainprm[a][0]
                cmd += self.data.qfrc_bias[dof] / kp
            self.data.ctrl[a] = cmd

    def set_finger_ctrl(self, q1, q2=None):
        """Command the two finger position actuators (slide qpos)."""
        if q2 is None:
            q2 = -q1
        self.data.ctrl[self.finger_act_ids[0]] = q1
        self.data.ctrl[self.finger_act_ids[1]] = q2

    def hold_arm(self):
        self.set_arm_ctrl(self.arm_qpos)

    def hold_fingers(self):
        q = self.finger_qpos
        self.set_finger_ctrl(q[0], q[1])

    def eef_pos(self):
        return np.array(self.data.site_xpos[self.eef_site_id])

    def eef_quat(self):
        """EEF site orientation as a (w,x,y,z) quaternion."""
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, self.data.site_xmat[self.eef_site_id])
        return q

    def eef_mat(self):
        return np.array(self.data.site_xmat[self.eef_site_id]).reshape(3, 3)

    def pad_span(self):
        """Centre-to-centre distance of the two finger pads (m), or None."""
        ids = []
        for name in ("finger1_pad_collision", "finger2_pad_collision"):
            i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                                  name)
            if i >= 0:
                ids.append(i)
        if len(ids) < 2:
            return None
        return float(np.linalg.norm(
            self.data.geom_xpos[ids[0]] - self.data.geom_xpos[ids[1]]))

    # ------------------------------------------------------------ objects
    def obj_pose(self, name):
        """(pos, quat wxyz) of a body centre."""
        bid = self.body_id(name)
        return (np.array(self.data.xpos[bid]),
                np.array(self.data.xquat[bid]))

    def obj_pos(self, name):
        return self.obj_pose(name)[0]

    def obj_axis(self, name):
        """World-frame unit vector of the body's local +z axis."""
        _, quat = self.obj_pose(name)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, quat)
        return R.reshape(3, 3)[:, 2]

    def obj_tilt(self, name):
        """Tilt of the body local +z from world z (deg)."""
        axis = self.obj_axis(name)
        return float(np.degrees(np.arccos(np.clip(abs(axis[2]), 0.0, 1.0))))

    @staticmethod
    def yaw_from_quat(quat):
        w, x, y, z = quat
        return float(np.arctan2(2.0 * (w * z + x * y),
                                1.0 - 2.0 * (y * y + z * z)))

    def obj_yaw(self, name):
        _, quat = self.obj_pose(name)
        return self.yaw_from_quat(quat)

    def site_pos(self, name):
        return np.array(self.data.site_xpos[self.site_id(name)])

    def set_obj_pose(self, name, xy, z, yaw=0.0):
        """Teleport a free-joint body (pos + z-yaw, zero velocity)."""
        bid = self.body_id(name)
        jid = self.model.body_jntadr[bid]
        if jid < 0 or self.model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f"body '{name}' has no free joint")
        qa = self.model.jnt_qposadr[jid]
        self.data.qpos[qa:qa + 3] = (xy[0], xy[1], z)
        self.data.qpos[qa + 3:qa + 7] = (
            np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0))
        da = self.model.jnt_dofadr[jid]
        self.data.qvel[da:da + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    # -------------------------------------------------------- state / reset
    def get_state(self):
        """Full-physics state (time+qpos+qvel+act) as a flat array.

        Hand-rolled instead of mj_getState: mujoco 2.3.2 has no
        mjtState enum (that API landed in 3.x)."""
        return np.concatenate([np.atleast_1d(self.data.time),
                               self.data.qpos.copy(),
                               self.data.qvel.copy(),
                               self.data.act.copy()])

    def set_state(self, state):
        st = np.asarray(state, dtype=float)
        nq, nv, na = self.model.nq, self.model.nv, self.model.na
        self.data.time = float(st[0])
        self.data.qpos[:] = st[1:1 + nq]
        self.data.qvel[:] = st[1 + nq:1 + nq + nv]
        self.data.act[:] = st[1 + nq + nv:1 + nq + nv + na]
        mujoco.mj_forward(self.model, self.data)
        # position servos: re-anchor ctrl to the restored state so the
        # restore does not inject a transient (old Phase-5A lesson about
        # controller residue after set_state)
        self.hold_arm()
        self.hold_fingers()

    def reset(self, home=True, finger=HOME_FINGER):
        """Fresh data; servo the arm to HOME and let it settle."""
        mujoco.mj_resetData(self.model, self.data)
        if home:
            for i, a in enumerate(self.arm_qadr):
                self.data.qpos[a] = HOME_QPOS[i]
            self.data.qpos[self.finger_qadr[0]] = finger
            self.data.qpos[self.finger_qadr[1]] = -finger
        self.hold_arm()
        self.set_finger_ctrl(finger)
        mujoco.mj_forward(self.model, self.data)
        # RE-anchor the arm ctrl AFTER mj_forward: the first hold_arm
        # ran while qfrc_bias was still 0 (post-reset), so the gravity
        # feedforward (bias/kp) was missing and the arm oscillated
        # ~0.13 rad/s indefinitely -- the root of the historical
        # nondeterministic grasps (every run started from a different
        # swinging state).  With the forward-pass bias the servo sits
        # still from step 1.
        self.hold_arm()

    # -------------------------------------------------------------- forces
    def geom_contact_force(self, geom_name):
        """Total contact-force magnitude on a geom (N) via the per-contact
        loop -- cfrc_ext reads ~0 for joint-mounted bodies (measured pitfall).
        """
        gid = self.geom_id(geom_name)
        f = 0.0
        buf = np.zeros(6)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 == gid or c.geom2 == gid:
                mujoco.mj_contactForce(self.model, self.data, i, buf)
                f += float(np.linalg.norm(buf[:3]))
        return f

    def grasp_contacts(self, body_name, min_force=0.05):
        """Bilateral pad contact with the requested body, excluding table contact."""
        bid = self.body_id(body_name)
        pads = [self.geom_id(f'finger{i}_pad_collision') for i in (1, 2)]
        forces = np.zeros(2)
        buf = np.zeros(6)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            for j, pad in enumerate(pads):
                other = c.geom2 if c.geom1 == pad else c.geom1 if c.geom2 == pad else -1
                if other >= 0 and self.model.geom_bodyid[other] == bid:
                    mujoco.mj_contactForce(self.model, self.data, i, buf)
                    forces[j] += max(0.0, float(buf[0]))
        return dict(held=bool(np.all(forces >= min_force)),
                    left_n=float(forces[0]), right_n=float(forces[1]))
