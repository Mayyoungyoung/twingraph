"""Run data collection for simbench executions.

A ``DataCollector`` attaches as a control-step hook (same interface as
``VideoRecorder``) and snapshots the robot + contact state every
``every`` control steps.  The Executor's step notification
(``ex.on_step``) marks skill boundaries so the trace can be sliced per
skill.  On close it writes one directory per run:

    <out_dir>/<task>_s<scene>_seed<seed>_<tag>/
        plan.json      -- the skill sequence (VLM plan) + params
        steps.json     -- per-skill outcomes (params + ok)
        stages.json    -- per-stage predicate verdicts + metrics
        trace.npz      -- sampled qpos/eef/finger/contact time series
        meta.json      -- run config + wall time + step counts

Both successful and failed runs are saved identically -- the failure
samples are valid data (they carry the terminal states and the contact
traces that produced them).  The plan source is recorded so a future
LLM-planner run is distinguishable from the nominal baseline.
"""
import json
import os
import time

import numpy as np

MIN_CONTACT_F = 0.05      # N; contacts below this are dropped from the
                          # per-sample list (keeps trace.npz small)


def _pad_force(ctx, pad_geoms=("finger1_pad_collision",
                               "finger2_pad_collision")):
    """(f1, f2) pad contact forces in N."""
    out = []
    for g in pad_geoms:
        try:
            out.append(float(ctx.geom_contact_force(g)))
        except ValueError:
            out.append(0.0)
    return out


def _contacts(ctx):
    """List of (geom1, geom2, |force|) for active contacts above
    MIN_CONTACT_F; name resolution survives unnamed geoms."""
    import mujoco
    rows = []
    for ci in range(ctx.data.ncon):
        c = ctx.data.contact[ci]
        f = np.zeros(6)
        mujoco.mj_contactForce(ctx.model, ctx.data, ci, f)
        mag = float(np.linalg.norm(f[:3]))
        if mag < MIN_CONTACT_F:
            continue
        n1 = ctx.model.geom(c.geom1).name or f"g{c.geom1}"
        n2 = ctx.model.geom(c.geom2).name or f"g{c.geom2}"
        rows.append((n1, n2, round(mag, 3)))
    return rows


class DataCollector:
    """Per-run data logger: control-step trace + step/stage bookkeeping.

    Attach as ``ctx.on_control_step`` (fan-in with the video recorder)
    and forward the executor's step notifications to ``note_step``;
    call ``set_stages`` with the run_task stage verdicts and ``close``
    at the end.  Writing happens only in ``close``, so a run that
    crashes mid-way still leaves its partial trace.
    """

    def __init__(self, ctx, out_dir, task, scene=1, seed=0, noise=0.0,
                 every=5, tag=None):
        self.ctx = ctx
        self.every = max(1, every)
        self.n = 0               # control steps seen
        self.sample_n = 0        # samples taken
        self.boundaries = []     # (sample, step, skill, ok, params)
        self.rows = []           # per-sample dicts
        self.stages = {}
        self.plan_info = None
        self.faults = None
        self.t0 = time.time()

        tag = tag or f"{int(time.time())}"
        self.dir = os.path.join(
            out_dir, f"{task}_s{scene}_seed{seed}_noise{noise}_{tag}")
        os.makedirs(self.dir, exist_ok=True)
        self.meta = dict(task=task, scene=scene, seed=seed,
                         noise=float(noise), every=every, tag=tag)

    # ------------------------------------------------------------ hooks
    def __call__(self, ctx):
        self.n += 1
        if self.n % self.every:
            return
        d = ctx.data
        eef = ctx.eef_pos()
        self.rows.append(dict(
            n=self.n,
            t=float(d.time),
            qpos=np.array(ctx.arm_qpos, dtype=float).tolist(),
            eef=eef.tolist(),
            fingers=np.array(ctx.finger_qpos, dtype=float).tolist(),
            pad_f=_pad_force(ctx),
            ncon=int(d.ncon),
            contacts=_contacts(ctx),
        ))
        self.sample_n += 1

    def set_plan(self, plan):
        """Record the (VLM) plan: skill sequence + params + meta."""
        self.plan_info = dict(
            name=plan.name,
            fail_mode=plan.fail_mode,
            source="nominal",          # "llm" once generate_plan is live
            steps=[dict(skill=s, params={k: _jval(v)
                                         for k, v in p.items()})
                   for s, p in plan.steps],
        )

    def note_step(self, i, skill, params, ok):
        """Executor callback: mark the trace sample where skill i ends."""
        self.boundaries.append(dict(
            sample=self.sample_n, step=i, skill=skill,
            ok=bool(ok),
            params={k: _jval(v) for k, v in params.items()}))

    def set_stages(self, results):
        self.stages = {f"S{k}": dict(ok=bool(ok),
                                     metrics=_jval(m))
                       for k, (ok, m) in results.items()}

    def set_faults(self, faults):
        """Record the run's FailureModel -- serialized on close (after
        the run, when the attributed event log is complete)."""
        self.faults = faults

    # ------------------------------------------------------------ close
    def close(self, all_ok=None):
        """Flush plan/steps/stages/trace/meta to disk."""
        self.meta.update(
            wall_s=round(time.time() - self.t0, 2),
            n_control_steps=self.n,
            n_samples=self.sample_n,
            ok_all=bool(all_ok) if all_ok is not None else None,
        )
        with open(os.path.join(self.dir, "plan.json"), "w") as f:
            json.dump(self.plan_info or {}, f, indent=1, default=str)
        with open(os.path.join(self.dir, "steps.json"), "w") as f:
            json.dump(self.boundaries, f, indent=1, default=str)
        with open(os.path.join(self.dir, "stages.json"), "w") as f:
            json.dump(self.stages, f, indent=1, default=str)
        if self.faults is not None:
            with open(os.path.join(self.dir, "fault_log.json"), "w") as f:
                json.dump(dict(profile=self.faults.profile,
                               seed=self.faults.seed,
                               params=_jval(self.faults.params),
                               log=list(self.faults.log)),
                          f, indent=1, default=str)
        with open(os.path.join(self.dir, "meta.json"), "w") as f:
            json.dump(self.meta, f, indent=1, default=str)
        # trace: columnar arrays + per-sample contact lists (object
        # array, pickle allowed) -- row-aligned with self.rows
        if self.rows:
            cols = {k: [] for k in ("n", "t", "qpos", "eef", "fingers",
                                    "pad_f", "ncon", "contacts")}
            for r in self.rows:
                for k in cols:
                    cols[k].append(r[k])
            np.savez(os.path.join(self.dir, "trace.npz"),
                     allow_pickle=True,
                     **{k: np.array(v, dtype=object)
                        for k, v in cols.items()})
        return self.dir


def _jval(v):
    """JSON-safe conversion (numpy scalars/arrays -> python)."""
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return float(v)
    if isinstance(v, dict):
        return {k: _jval(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jval(x) for x in v]
    return v
