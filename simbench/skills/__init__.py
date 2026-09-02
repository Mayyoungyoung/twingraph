"""simbench skill library: hierarchical, composable atomic robot skills.

Layering (bottom-up):

  L0  primitives      :mod:`simbench.core` -- ``MjContext`` (stepping +
      name lookups), ``CartesianController`` (closed-loop EEF motion),
      ``Gripper`` (force-bounded finger servo).
  L1  atomic skills   the four categories registered in ``REGISTRY``:
        exec  (:mod:`.library`, :mod:`.manipulation`, :mod:`.perception`)
        plan  (:mod:`.library`, :mod:`.planning`)   multi-candidate +
              scoring + selection
        trans (:mod:`.transitions`)                 approach / pre_align /
              retreat_lift / return_home
        ext   (:mod:`.extension`, :mod:`.learned`)  peg_insert (script +
              RL/IL policy) / wipe / pull
  L2  contracts       :class:`~.base.SkillSpec` gates (pre/post
      conditions, failure policy, impl method, deps) + one gated entry
      point :meth:`~.base.SkillRegistry.run`.
  L3  composition     :func:`~.base.run_chain` sequences skills; the
      planner + executor compose them into task chains via the
      plan->exec artifact handshake.

Importing this package populates ``REGISTRY`` (the ``@register``
decorators in the submodules run at import).  The learned layer
(:mod:`.learned`) is imported LAZILY by ``extension.peg_insert`` so a
plain ``import simbench.skills`` has no torch / gymnasium dependency.
"""
from . import base
from . import perception
from . import planning
from . import motion
from . import manipulation
from . import settle
from . import actuator
from . import transitions
from . import extension
from . import library

from .base import (REGISTRY, SkillSpec, SkillResult, ChainResult,  # noqa: F401
                   register, inventory, run_chain,
                   CAT_EXEC, CAT_PLAN, CAT_TRANS, CAT_CN,
                   IMPL_SCRIPT, IMPL_MOTION, IMPL_OPT, IMPL_IL, IMPL_RL,
                   GRAN_ATOMIC, GRAN_COMPOSITE, GRAN_CN)

__all__ = [
    "base", "perception", "planning", "motion", "manipulation", "settle",
    "actuator", "transitions", "extension", "library",
    "REGISTRY", "SkillSpec", "SkillResult", "ChainResult",
    "register", "inventory", "run_chain",
    "CAT_EXEC", "CAT_PLAN", "CAT_TRANS", "CAT_CN",
    "IMPL_SCRIPT", "IMPL_MOTION", "IMPL_OPT", "IMPL_IL", "IMPL_RL",
    "GRAN_ATOMIC", "GRAN_COMPOSITE", "GRAN_CN",
]
