"""Torch MLP Gaussian policy for the learned insertion skill.

``MLPGaussianPolicy`` is the trainable workhorse for the RL / IL
insertion policy: a tanh MLP actor emitting a diagonal-Gaussian action
distribution with a state-independent log-std.  It implements the same
:class:`~.policy.Policy` interface the skill layer consumes
(``policy.act(obs)``), persists as a ``.pt`` checkpoint, and is
interchangeable with the pure-numpy ``LinearGaussianPolicy``.

``load_policy(path)`` dispatches on the file extension so a skill can
load either a torch (.pt) or numpy (.npz) checkpoint transparently.
"""
import numpy as np
import torch
import torch.nn as nn

from .policy import Policy


class MLPActor(nn.Module):
    """tanh-MLP mean head + state-independent log-std parameter."""

    def __init__(self, obs_dim, act_dim, hidden=(64, 64),
                 log_std_init=-1.0):
        super().__init__()
        layers = []
        d = int(obs_dim)
        for h in hidden:
            layers += [nn.Linear(d, int(h)), nn.Tanh()]
            d = int(h)
        layers += [nn.Linear(d, int(act_dim))]
        self.mean = nn.Sequential(*layers)
        self.log_std = nn.Parameter(torch.ones(int(act_dim))
                                    * float(log_std_init))

    def forward(self, obs):
        m = self.mean(obs)
        return m, self.log_std.exp()


class MLPGaussianPolicy(Policy):
    """Diagonal-Gaussian MLP policy: pi(a|o) = N(mu(o), sigma^2).

    ``act(obs, deterministic=True)`` returns the mean (deployment);
    ``deterministic=False`` samples for exploration during training.
    Actions are clipped to ``[act_low, act_high]`` (the env's action
    box) so a sampled tail never exceeds the delta budget.
    """

    def __init__(self, obs_dim, act_dim, hidden=(64, 64),
                 log_std_init=-1.0, seed=0, device="cpu",
                 act_low=-1.0, act_high=1.0):
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.hidden = tuple(int(h) for h in hidden)
        self.device = torch.device(device)
        self.act_low = float(act_low)
        self.act_high = float(act_high)
        torch.manual_seed(int(seed))
        self.net = MLPActor(self.obs_dim, self.act_dim, self.hidden,
                            log_std_init).to(self.device)

    # -------------------------------------------------------- inference
    def _to_tensor(self, obs):
        return torch.as_tensor(np.asarray(obs, dtype=np.float32),
                               device=self.device).float()

    def dist(self, obs):
        """torch.distributions.Normal for the given observation(s)."""
        mean, std = self.net(self._to_tensor(obs))
        return torch.distributions.Normal(mean, std)

    def act(self, obs, deterministic=True):
        with torch.no_grad():
            mean, std = self.net(self._to_tensor(obs))
            a = mean if deterministic else mean + std * torch.randn_like(
                mean)
        a = a.detach().cpu().numpy()
        return np.clip(a, self.act_low, self.act_high).astype(np.float32)

    # -------------------------------------------------------- (de)serialise
    def state_dict(self):
        return dict(obs_dim=self.obs_dim, act_dim=self.act_dim,
                    hidden=list(self.hidden), net=self.net.state_dict())

    def save(self, path):
        torch.save(self.state_dict(), path)
        return path

    def load(self, path):
        d = torch.load(path, map_location=self.device)
        self.obs_dim = int(d["obs_dim"])
        self.act_dim = int(d["act_dim"])
        self.hidden = tuple(d.get("hidden", (64, 64)))
        self.net = MLPActor(self.obs_dim, self.act_dim, self.hidden,
                            -1.0).to(self.device)
        self.net.load_state_dict(d["net"])
        return self

    @classmethod
    def from_file(cls, path, device="cpu"):
        d = torch.load(path, map_location=device)
        p = cls(d["obs_dim"], d["act_dim"],
                hidden=tuple(d.get("hidden", (64, 64))), device=device)
        p.net.load_state_dict(d["net"])
        return p


def load_policy(path, device="cpu"):
    """Load a policy checkpoint by extension: .pt -> torch MLP,
    .npz -> numpy linear (both satisfy the ``act(obs)`` interface)."""
    path = str(path)
    if path.endswith(".npz"):
        from .policy import LinearGaussianPolicy
        return LinearGaussianPolicy.from_file(path)
    return MLPGaussianPolicy.from_file(path, device=device)
