"""Skill-library base: contracts, registry, and composable interfaces.

A *skill* is a callable with the conventional signature
``fn(ctx, arm, gripper, **params)`` returning either a bool (legacy
skills) or a :class:`SkillResult`.  Every skill is registered with a
:class:`SkillSpec` that documents its contract:

  - category: ``exec`` | ``plan`` | ``trans`` | ``ext``
  - inputs / outputs: parameter and artifact field descriptions
  - preconditions / postconditions: declarative descriptions plus
    optional check callables (see the helpers below)
  - failure_policy: ``retry`` | ``abort`` | ``continue``
  - impl: ``script`` | ``motion_planning`` | ``optimization`` |
    ``imitation`` | ``rl``

``SkillRegistry.inventory()`` emits the structured skill list (the
skill-inventory deliverable).  The executor dispatches new skills
through ``registry.run(...)`` so the contract gates execute in one
place; legacy handlers keep their original behavior.
"""
from dataclasses import dataclass, field, asdict


# ------------------------------------------------------------- categories
CAT_EXEC = "exec"          # execution: drives the physics
CAT_PLAN = "plan"          # planning: computation only, emits artifacts
CAT_TRANS = "trans"        # transition: smooths skill boundaries
CAT_TRANS_CN = "过渡类"
CAT_EXEC_CN = "执行类"
CAT_PLAN_CN = "规划类"
CAT_EXT_CN = "扩展类"
CAT_CN = {CAT_EXEC: CAT_EXEC_CN, CAT_PLAN: CAT_PLAN_CN,
          CAT_TRANS: CAT_TRANS_CN, "ext": CAT_EXT_CN}

# implementation methods
IMPL_SCRIPT = "规则脚本"
IMPL_MOTION = "运动规划"
IMPL_OPT = "优化"
IMPL_IL = "模仿学习"
IMPL_RL = "强化学习"


# ------------------------------------------------------------------ result
@dataclass
class SkillResult:
    """Standard skill return value: outcome + measured metrics + reason."""

    ok: bool = True
    metrics: dict = field(default_factory=dict)
    reason: str = ""

    def as_dict(self):
        return dict(ok=bool(self.ok), metrics=dict(self.metrics),
                    reason=str(self.reason))


# -------------------------------------------------------------- skill spec
@dataclass
class SkillSpec:
    """Declarative contract of a registered skill."""

    name: str
    category: str                    # exec / plan / trans / ext
    description: str
    inputs: dict                     # param name -> description
    outputs: dict                    # artifact/state name -> description
    preconditions: list = field(default_factory=list)   # str | callable
    postconditions: list = field(default_factory=list)  # str | callable
    failure_policy: str = "continue"  # retry / abort / continue
    impl: str = IMPL_SCRIPT          # script / motion_planning / ...
    deps: list = field(default_factory=list)   # depended-on skills/parts

    def as_dict(self):
        d = asdict(self)
        d["category_cn"] = CAT_CN.get(self.category, self.category)
        return d


# ------------------------------------------------------------- check helpers
# A precondition check is callable(ctx, name, arm, gripper, params) ->
# (ok, reason).  A postcondition verifier is callable(ctx, name, arm,
# gripper, params) -> (ok, metric_key, metric_value).  ``name`` is the
# registered skill name; params is the dict passed to the skill.
import numpy as _np


def part_exists(ctx, name, arm=None, gripper=None, params=None):
    """Precondition: the scene has a body params['part'|'name'] (any pose)."""
    part = (params or {}).get("part") or (params or {}).get("name", "")
    try:
        ctx.obj_pos(part)
        return True, ""
    except ValueError:
        return False, f"part '{part}' not in scene"


def held_part(ctx, name, arm=None, gripper=None, params=None, tol=0.05):
    """Precondition: the EEF still holds params['part'|'name'] (within *tol*)."""
    part = (params or {}).get("part") or (params or {}).get("name", "")
    try:
        d = float(_np.linalg.norm(ctx.eef_pos() - ctx.obj_pos(part)))
    except ValueError:
        return False, f"part '{part}' not in scene"
    if d > tol:
        return False, f"part '{part}' not held (eef distance {d:.3f} m)"
    return True, ""


# ---------------------------------------------------------------- registry
class SkillRegistry:
    """Name -> (spec, fn) registry with contract-gated execution."""

    def __init__(self):
        self._skills = {}

    def register(self, name=None, **spec_kw):
        """Decorator: attach a SkillSpec to a skill function."""

        def deco(fn):
            skill_name = name or fn.__name__
            self._skills[skill_name] = (
                SkillSpec(name=skill_name, **spec_kw), fn)
            return fn

        return deco

    def add(self, spec, fn):
        self._skills[spec.name] = (spec, fn)

    def get(self, name):
        return self._skills.get(name)

    def has(self, name):
        return name in self._skills

    def names(self, category=None):
        out = []
        for name, (spec, _) in self._skills.items():
            if category is None or spec.category == category:
                out.append(name)
        return sorted(out)

    def inventory(self, category=None):
        """Structured skill list (the inventory deliverable)."""
        rows = []
        for name in self.names(category):
            spec, fn = self._skills[name]
            d = spec.as_dict()
            d["wrapped"] = bool(fn is not None)
            rows.append(d)
        return rows

    def run(self, skill, ctx, arm, gripper, check=True, **params):
        """Run a registered skill with contract gates.

        ``skill`` is the registered skill NAME (positional); it is kept
        distinct from the ``name``/``part`` params a skill takes for the
        scene body, so ``run('grasp', ctx, arm, gripper, name='gear')``
        does not collide.  check=True runs the preconditions first
        (callable entries) and the postcondition verifications after; a
        failed precondition returns SkillResult(ok=False) WITHOUT
        executing the skill.  Returns SkillResult always.
        """
        entry = self._skills.get(skill)
        if entry is None:
            raise ValueError(f"unknown skill {skill!r}")
        spec, fn = entry
        # precondition gate
        if check:
            for pre in spec.preconditions:
                if callable(pre):
                    ok_p, why = pre(ctx, skill, arm, gripper, params)
                    if not ok_p:
                        return SkillResult(ok=False, reason=why)
        out = fn(ctx, arm, gripper, **params)
        res = out if isinstance(out, SkillResult) else SkillResult(
            ok=bool(out))
        # postcondition verification (informational metrics)
        if check:
            for post in spec.postconditions:
                if callable(post):
                    ok_p, key, val = post(ctx, skill, arm, gripper, params)
                    res.metrics[key] = val
                    if not ok_p:
                        res.ok = False
                        if not res.reason:
                            res.reason = f"postcondition '{key}' failed"
        return res


REGISTRY = SkillRegistry()


def register(name=None, **spec_kw):
    """Module-level decorator alias for REGISTRY.register."""
    return REGISTRY.register(name=name, **spec_kw)


def inventory(category=None):
    return REGISTRY.inventory(category=category)


# ------------------------------------------------------------- composition
@dataclass
class ChainResult:
    """Aggregate outcome of a :func:`run_chain` skill sequence."""

    ok: bool = True
    results: list = field(default_factory=list)   # [(name, SkillResult)]
    stopped_at: int = -1                          # index aborted, -1 = done

    def metrics(self):
        """Flat {name: SkillResult.as_dict()} view of every step run."""
        return {name: res.as_dict() for name, res in self.results}


def run_chain(registry, ctx, arm, gripper, steps, check=True,
              stop_on_fail=True, verbose=False):
    """Compose a sequence of registered skills into one runnable unit.

    ``steps`` is an iterable of ``(skill_name, params_dict)``.  Each step
    runs through :meth:`SkillRegistry.run` (so its precondition /
    postcondition gates fire), and the per-step :class:`SkillResult` is
    collected.  With ``stop_on_fail`` the chain aborts at the first
    failed skill and records its index in ``stopped_at``.

    This is the composable-interface seam between the atomic skills:
    e.g. ``plan_grasp_pose -> approach -> grasp`` or
    ``place -> retreat_lift -> return_home``.
    """
    out = []
    ok_all = True
    stopped = -1
    for i, step in enumerate(steps):
        name, params = step
        res = registry.run(name, ctx, arm, gripper, check=check,
                           **(params or {}))
        out.append((name, res))
        if verbose:
            print(f"  [chain {i}] {name}: "
                  f"{'ok' if res.ok else 'FAIL'} {res.reason}")
        if not res.ok:
            ok_all = False
            if stop_on_fail:
                stopped = i
                break
    return ChainResult(ok=ok_all, results=out, stopped_at=stopped)
