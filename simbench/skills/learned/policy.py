"""Learned policy primitives (pure numpy).

``Policy`` is the interface a skill consumes (``policy.act(obs)``).
``LinearGaussianPolicy`` is the trainable workhorse: mean = W @ obs + b
with an annealable scalar variance for exploration (RL) or deterministic
evaluation.  Policies persist as .npz files and can be swapped for any
torch-based model later without touching the skill layer.
"""
import numpy as np


class Policy:
    """Policy interface: act(obs) -> action vector."""

    def act(self, obs, deterministic=True):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError


class LinearGaussianPolicy(Policy):
    """Gaussian linear policy: N(W @ obs + b, sigma^2 I)."""

    def __init__(self, obs_dim, act_dim, sigma=0.0005, seed=0,
                 W=None, b=None):
        self.W = (np.zeros((act_dim, obs_dim), dtype=float)
                  if W is None else np.asarray(W, dtype=float))
        self.b = (np.zeros(act_dim, dtype=float)
                  if b is None else np.asarray(b, dtype=float))
        self.sigma = float(sigma)
        self.rng = np.random.default_rng(seed)

    def act(self, obs, deterministic=True):
        mean = self.W @ np.asarray(obs, dtype=float) + self.b
        if deterministic:
            return mean
        return mean + self.rng.normal(0.0, self.sigma, size=mean.shape)

    def set_params(self, W, b):
        self.W = np.asarray(W, dtype=float)
        self.b = np.asarray(b, dtype=float)

    def save(self, path):
        np.savez(path, W=self.W, b=self.b, sigma=np.array(self.sigma))

    def load(self, path):
        d = np.load(path)
        self.W = np.asarray(d["W"], dtype=float)
        self.b = np.asarray(d["b"], dtype=float)
        self.sigma = float(np.asarray(d["sigma"]))
        return self

    @classmethod
    def from_file(cls, path):
        p = cls(1, 1)
        p.load(path)
        return p
