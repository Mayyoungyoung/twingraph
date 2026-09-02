"""Learned skills: trainable policies for contact-rich primitives.

The insertion policy is trained to "hold the peg and insert it into the
given xyz hole" with contact, jam and tilt feedback in the reward (see
:mod:`.insert_env` + :mod:`.train_insert`).

Two interchangeable policy back-ends satisfy the ``policy.act(obs)``
interface the skill layer consumes:

  - :class:`.policy.LinearGaussianPolicy`  -- pure numpy (no deps)
  - :class:`.policy_torch.MLPGaussianPolicy` -- torch MLP (RL / BC)

``load_policy(path)`` dispatches on the checkpoint extension (.pt /
.npz).  The torch import is guarded so the learned package still imports
in a numpy-only environment (only .pt policies then require torch).
"""
from .policy import Policy, LinearGaussianPolicy
from .insert_env import (InsertEnv, build_obs, action_to_delta,
                         OBS_DIM, ACT_DIM)

try:                                             # torch back-end (RL / BC)
    from .policy_torch import MLPGaussianPolicy, MLPActor, load_policy
    TORCH_AVAILABLE = True
except ImportError:                              # numpy-only environment
    MLPGaussianPolicy = MLPActor = None
    TORCH_AVAILABLE = False

    def load_policy(path, device="cpu"):
        if str(path).endswith(".npz"):
            return LinearGaussianPolicy.from_file(path)
        raise ImportError(
            "torch is required to load .pt policies; install torch or "
            "use a .npz LinearGaussianPolicy checkpoint")

__all__ = ["Policy", "LinearGaussianPolicy", "InsertEnv",
           "build_obs", "action_to_delta", "OBS_DIM", "ACT_DIM",
           "MLPGaussianPolicy", "MLPActor", "load_policy",
           "TORCH_AVAILABLE"]
