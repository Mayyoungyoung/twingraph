#!/usr/bin/env python3
"""Train the learned peg-insertion policy (RL or imitation).

    python -m simbench.skills.learned.train_insert --algo ppo \
        --episodes 30 --out simbench/skills/learned/checkpoints/peg_insert.pt
    python -m simbench.skills.learned.train_insert --algo bc --episodes 60

Two self-contained torch trainers on :class:`.insert_env.InsertEnv`:

  ppo  -- proximal policy optimization (GAE, clipped surrogate, entropy
          bonus).  The RL route the task calls for: the reward already
          encodes contact / jam / tilt-deviation feedback.
  bc   -- behaviour cloning of a scripted expert (proportional lateral
          alignment + descent) -- the imitation-learning route; a strong
         , fast baseline that also seeds a good policy.

Both save a ``.pt`` checkpoint consumable by
``extension.peg_insert(mode='policy')`` (via ``load_policy``) and print a
deterministic seat-rate evaluation.  ``--episodes`` is the single scale
knob (PPO iterations / BC demo episodes); everything is seeded for
reproducibility.
"""
import argparse
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import torch                                            # noqa: E402
import torch.nn as nn                                   # noqa: E402

from simbench.skills.learned.insert_env import (         # noqa: E402
    InsertEnv, OBS_DIM, ACT_DIM, _DIST_SCALE)
from simbench.skills.learned.policy_torch import (       # noqa: E402
    MLPGaussianPolicy, MLPActor)

DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "checkpoints", "peg_insert.pt")


# ------------------------------------------------------------------ utils
def _seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


class ValueNet(nn.Module):
    """tanh-MLP state-value critic for PPO."""

    def __init__(self, obs_dim, hidden=(64, 64)):
        super().__init__()
        layers = []
        d = obs_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.Tanh()]
            d = h
        layers += [nn.Linear(d, 1)]
        self.f = nn.Sequential(*layers)

    def forward(self, obs):
        return self.f(obs).squeeze(-1)


def expert_action(obs):
    """Scripted expert: proportional lateral alignment + full descent.

    Reads the peg-bottom lateral error straight from the observation
    (obs[0:2] = lat*_DIST_SCALE) and commands a correcting xy delta while
    descending at the max z rate.  This is the measured-stable insertion
    recipe; used to generate BC demonstrations and as an RL sanity floor.
    """
    lat = obs[0:2] / _DIST_SCALE
    ax = float(np.clip(-lat[0] * 500.0, -1.0, 1.0))
    ay = float(np.clip(-lat[1] * 500.0, -1.0, 1.0))
    return np.array([ax, ay, -1.0], dtype=np.float32)


def evaluate(policy, n_episodes=20, max_steps=200, seed=1000,
             randomize=True):
    """Deterministic seat-rate over n_episodes; returns (rate, steps)."""
    env = InsertEnv(max_steps=max_steps, randomize=randomize)
    seated = 0
    steps = []
    for e in range(n_episodes):
        obs, _ = env.reset(seed=seed + e)
        done = False
        k = 0
        while not done and k < max_steps:
            a = policy.act(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            done = term or trunc
            k += 1
        if info.get("seated"):
            seated += 1
            steps.append(k)
    rate = seated / max(1, n_episodes)
    msteps = float(np.mean(steps)) if steps else float("nan")
    return rate, msteps


# -------------------------------------------------------------------- ppo
def train_ppo(args):
    _seed_all(args.seed)
    dev = torch.device(args.device)
    policy = MLPGaussianPolicy(OBS_DIM, ACT_DIM, hidden=args.hidden,
                               log_std_init=args.log_std, seed=args.seed,
                               device=args.device)
    critic = ValueNet(OBS_DIM, args.hidden).to(dev)
    opt = torch.optim.Adam(
        list(policy.net.parameters()) + list(critic.parameters()),
        lr=args.lr)

    env = InsertEnv(max_steps=args.max_steps, randomize=True)
    iters = args.iters if args.iters is not None else args.episodes
    obs, _ = env.reset(seed=args.seed)
    best_rate = -1.0
    hist = []
    for it in range(iters):
        # ---- rollout
        b_obs, b_act, b_logp, b_rew, b_done, b_val = [], [], [], [], [], []
        ep_rets, ep_ret, ep_seated, ep_count = [], 0.0, 0, 0
        for _ in range(args.rollout):
            o_t = torch.as_tensor(obs, dtype=torch.float32, device=dev)
            with torch.no_grad():
                mean, std = policy.net(o_t)
                dist = torch.distributions.Normal(mean, std)
                a = dist.sample()
                a = torch.clamp(a, -1.0, 1.0)
                logp = dist.log_prob(a).sum()
                val = critic(o_t)
            b_obs.append(obs)
            b_act.append(a.cpu().numpy())
            b_logp.append(float(logp))
            b_val.append(float(val))
            nxt, rew, term, trunc, info = env.step(a.cpu().numpy())
            b_rew.append(float(rew))
            b_done.append(bool(term or trunc))
            ep_ret += float(rew)
            obs = nxt
            if term or trunc:
                ep_rets.append(ep_ret)
                ep_seated += int(bool(info.get("seated")))
                ep_count += 1
                ep_ret = 0.0
                obs, _ = env.reset()
        # ---- GAE
        with torch.no_grad():
            last_val = float(critic(torch.as_tensor(
                obs, dtype=torch.float32, device=dev)))
        T = len(b_rew)
        adv = np.zeros(T, dtype=np.float32)
        ret = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            nonterm = 0.0 if b_done[t] else 1.0
            next_val = last_val if t == T - 1 else b_val[t + 1]
            delta = b_rew[t] + args.gamma * next_val * nonterm - b_val[t]
            gae = delta + args.gamma * args.gae_lambda * nonterm * gae
            adv[t] = gae
            ret[t] = adv[t] + b_val[t]
        adv_t = torch.as_tensor(adv, device=dev)
        ret_t = torch.as_tensor(ret, device=dev)
        obs_t = torch.as_tensor(np.array(b_obs), dtype=torch.float32,
                                device=dev)
        act_t = torch.as_tensor(np.array(b_act), dtype=torch.float32,
                                device=dev)
        logp_t = torch.as_tensor(np.array(b_logp), dtype=torch.float32,
                                 device=dev)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        # ---- update
        n = T
        for _ in range(args.epochs):
            idx = torch.randperm(n, device=dev)
            for start in range(0, n, args.minibatch):
                mb = idx[start:start + args.minibatch]
                mean, std = policy.net(obs_t[mb])
                dist = torch.distributions.Normal(mean, std)
                logp_new = dist.log_prob(act_t[mb]).sum(-1)
                ratio = torch.exp(logp_new - logp_t[mb])
                s1 = ratio * adv_t[mb]
                s2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) \
                    * adv_t[mb]
                pi_loss = -torch.min(s1, s2).mean()
                ent = dist.entropy().sum(-1).mean()
                v_pred = critic(obs_t[mb])
                v_loss = nn.functional.mse_loss(v_pred, ret_t[mb])
                loss = pi_loss + args.vf_coef * v_loss - args.ent_coef * ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(policy.net.parameters())
                    + list(critic.parameters()), args.max_grad)
                opt.step()
        mean_ret = float(np.mean(ep_rets)) if ep_rets else float("nan")
        rate = ep_seated / max(1, ep_count)
        hist.append(mean_ret)
        print(f"[ppo {it+1}/{iters}] rollout_ret={mean_ret:7.3f} "
              f"train_seat={rate:.2f} ({ep_seated}/{ep_count}) "
              f"pi={float(pi_loss):.3f} v={float(v_loss):.3f} "
              f"ent={float(ent):.2f}", flush=True)
        if (it + 1) % args.eval_every == 0 or it == iters - 1:
            er, ms = evaluate(policy, args.eval_episodes, args.max_steps,
                              seed=args.seed + 5000)
            print(f"          EVAL seat_rate={er:.2f} "
                  f"mean_steps={ms:.1f}", flush=True)
            if er > best_rate:
                best_rate = er
                _save(policy, args.out)
    if best_rate < 0:
        _save(policy, args.out)
    print(f"[ppo] done. best eval seat_rate={best_rate:.2f} -> {args.out}")
    return policy, best_rate


# --------------------------------------------------------------------- bc
def gather_demos(args):
    """Collect (obs, action) demos from the scripted expert on SEATED
    episodes only."""
    env = InsertEnv(max_steps=args.max_steps, randomize=True)
    obs_buf, act_buf = [], []
    seated_eps = 0
    e = 0
    while seated_eps < args.episodes and e < args.episodes * 20:
        obs, _ = env.reset(seed=args.seed + e)
        e += 1
        traj_o, traj_a = [], []
        done = False
        k = 0
        while not done and k < args.max_steps:
            a = expert_action(obs)
            traj_o.append(obs.copy())
            traj_a.append(a.copy())
            obs, r, term, trunc, info = env.step(a)
            done = term or trunc
            k += 1
        if info.get("seated"):
            seated_eps += 1
            obs_buf.extend(traj_o)
            act_buf.extend(traj_a)
    return (np.array(obs_buf, dtype=np.float32),
            np.array(act_buf, dtype=np.float32), seated_eps)


def train_bc(args):
    _seed_all(args.seed)
    dev = torch.device(args.device)
    X, Y, n_eps = gather_demos(args)
    print(f"[bc] {n_eps} seated demos, {len(X)} transitions", flush=True)
    if len(X) == 0:
        raise SystemExit("bc: no successful expert demos collected")
    policy = MLPGaussianPolicy(OBS_DIM, ACT_DIM, hidden=args.hidden,
                               log_std_init=args.log_std, seed=args.seed,
                               device=args.device)
    opt = torch.optim.Adam(policy.net.parameters(), lr=args.lr)
    Xt = torch.as_tensor(X, device=dev)
    Yt = torch.as_tensor(Y, device=dev)
    n = len(Xt)
    best_rate = -1.0
    for ep in range(args.bc_epochs):
        idx = torch.randperm(n, device=dev)
        tot = 0.0
        for start in range(0, n, args.minibatch):
            mb = idx[start:start + args.minibatch]
            mean, _ = policy.net(Xt[mb])
            loss = nn.functional.mse_loss(mean, Yt[mb])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(mb)
        if (ep + 1) % args.eval_every == 0 or ep == args.bc_epochs - 1:
            er, ms = evaluate(policy, args.eval_episodes, args.max_steps,
                              seed=args.seed + 5000)
            print(f"[bc {ep+1}/{args.bc_epochs}] mse={tot/n:.5f} "
                  f"EVAL seat_rate={er:.2f} mean_steps={ms:.1f}",
                  flush=True)
            if er > best_rate:
                best_rate = er
                _save(policy, args.out)
    if best_rate < 0:
        _save(policy, args.out)
    print(f"[bc] done. best eval seat_rate={best_rate:.2f} -> {args.out}")
    return policy, best_rate


def _save(policy, out):
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    policy.save(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="ppo", choices=["ppo", "bc"])
    ap.add_argument("--episodes", type=int, default=30,
                    help="scale knob: PPO iterations / BC demo episodes")
    ap.add_argument("--iters", type=int, default=None,
                    help="explicit PPO iterations (defaults to --episodes)")
    ap.add_argument("--rollout", type=int, default=1024)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bc-epochs", type=int, default=60)
    ap.add_argument("--minibatch", type=int, default=64)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--max-grad", type=float, default=0.5)
    ap.add_argument("--log-std", type=float, default=-1.0)
    ap.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    args.hidden = tuple(args.hidden)

    if args.algo == "ppo":
        train_ppo(args)
    else:
        train_bc(args)


if __name__ == "__main__":
    main()
