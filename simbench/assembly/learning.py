"""Learn insertion from real robot-in-simulator expert rollouts.

This is behavior cloning, not a renamed hand-coded policy and not an RL claim.
Episodes restore a checkpoint produced by real grasp/transfer, then perturb the
held pin through robot motion. Recorded actions train the deployed actor.
"""

import argparse
import json
import pickle
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .control import make_context
from .library import Session
from .task import PIN_L

SCALE = np.array([0.0004, 0.0004, 0.00015], dtype=np.float32)
OBS_DIM = 11


def insert_observation(s, part, target):
    ctx = s.ctx
    bid = ctx.body_id(part)
    jid = int(ctx.model.body_jntadr[bid])
    da = int(ctx.model.jnt_dofadr[jid])
    contact = ctx.grasp_contacts(part)
    return np.r_[
        (np.asarray(target) - ctx.obj_pos(part)) * 100,
        ctx.data.qvel[da : da + 3] * 20,
        ctx.obj_axis(part)[:2] * 10,
        min(s.external_force(part), 20) / 10,
        contact["left_n"] / 10,
        contact["right_n"] / 10,
    ].astype(np.float32)


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, 48),
            nn.Tanh(),
            nn.Linear(48, 48),
            nn.Tanh(),
            nn.Linear(48, 3),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


def load_actor(path):
    torch.set_num_threads(2)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("algorithm") != "behavior_cloning"
        or checkpoint["obs_dim"] != OBS_DIM
    ):
        raise ValueError("incompatible learned insertion checkpoint")
    model = Actor()
    model.load_state_dict(checkpoint["weights"])
    model.eval()

    def act(obs):
        with torch.no_grad():
            return model(torch.as_tensor(obs)).numpy() * SCALE

    return act


def expert(obs):
    error = obs[:3] / 100
    command = np.clip(error * np.array([0.22, 0.22, 0.1]), -SCALE, SCALE)
    if obs[8] > 0.7:
        command[2] = 0.000025
    return command.astype(np.float32)


def rollout(s, checkpoint, rng, actor=None, record=False):
    s.restore(checkpoint)
    part = "pin_left"
    # Achieve the perturbed starting condition with a real held-part motion.
    offset = np.r_[rng.uniform(-0.002, 0.002, 2), rng.uniform(-0.006, 0.006)]
    s.arm.move(s.ctx.eef_pos() + offset, speed=0.025)
    rows = []
    peak = 0.0
    for k in range(1000):
        obs = insert_observation(s, part, PIN_L)
        action = expert(obs) if actor is None else actor(obs)
        if record:
            rows.append((obs.copy(), (action / SCALE).copy()))
        s.arm.servo(s.ctx.eef_pos() + action)
        peak = max(peak, s.external_force(part))
        if not s.ctx.grasp_contacts(part)["held"] or peak > 15.0:
            break
        error = s.ctx.obj_pos(part) - PIN_L
        if abs(error[2]) < 0.0007 and np.linalg.norm(error[:2]) < 0.0008:
            break
    error = s.ctx.obj_pos(part) - PIN_L
    ok = bool(
        np.linalg.norm(error) < 0.0012
        and s.ctx.grasp_contacts(part)["held"]
        and peak <= 15.0
    )
    return rows, dict(
        success=ok,
        steps=k + 1,
        error_m=float(np.linalg.norm(error)),
        peak_force_n=peak,
        initial_offset_m=offset.tolist(),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--out", default="results/tabletop/learned_insertion")
    ap.add_argument("--evaluate-only", action="store_true")
    ap.add_argument("--test-seed", type=int, default=98211)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(12)
    with open(args.checkpoint, "rb") as f:
        state = pickle.load(f)
    s = Session(make_context())
    rng = np.random.default_rng(421)
    if args.evaluate_only:
        actor = load_actor(out / "insert_bc.pt")
        reports = []
        test_rng = np.random.default_rng(args.test_seed)
        for i in range(12):
            _, report = rollout(s, state, test_rng, actor=actor)
            reports.append(report)
            print("evaluation", i, report, flush=True)
        report = dict(
            algorithm="behavior_cloning",
            frozen_checkpoint=str(out / "insert_bc.pt"),
            max_steps=1000,
            test_seed=args.test_seed,
            test_episodes=reports,
            test_successes=sum(x["success"] for x in reports),
            test_total=len(reports),
        )
        (out / f"evaluation_1000_steps_seed{args.test_seed}.json").write_text(
            json.dumps(report, indent=2)
        )
        print(json.dumps({k: v for k, v in report.items() if not isinstance(v, list)}))
        return
    episodes = []
    data = []
    for i in range(args.episodes):
        rows, report = rollout(s, state, rng, record=True)
        # Keep failure trajectories too; only use expert actions, never a final
        # success bit as an action target or an observation feature.
        data.extend((o, a, i) for o, a in rows)
        episodes.append(report)
        print("expert", i, report, flush=True)
    x = np.array([d[0] for d in data])
    y = np.array([d[1] for d in data])
    groups = np.array([d[2] for d in data])
    np.savez_compressed(
        out / "demonstrations.npz", observations=x, actions=y, episode_id=groups
    )
    # Whole episodes are held out, not individual adjacent trajectory frames.
    train = groups % 5 != 0
    val = ~train
    xt = torch.tensor(x[train])
    yt = torch.tensor(y[train])
    xv = torch.tensor(x[val])
    yv = torch.tensor(y[val])
    model = Actor()
    opt = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.00001)
    best = float("inf")
    weights = None
    for epoch in range(500):
        pred = model(xt)
        # Lateral correction examples are uncommon; retain them in minibatch
        # weighting instead of allowing straight descents to dominate entirely.
        weight = 1 + 5 * (yt[:, :2].abs().sum(1) > 0.12).float()
        loss = (((pred - yt) ** 2).mean(1) * weight).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            v = float(((model(xv) - yv) ** 2).mean())
        if v < best:
            best = v
            weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
    checkpoint = dict(
        algorithm="behavior_cloning",
        obs_dim=OBS_DIM,
        weights=weights,
        action_scale=SCALE,
        source_checkpoint=str(args.checkpoint),
        episodes=args.episodes,
        validation_mse=best,
        notes="Real Panda contact rollouts. No demonstration-time object teleport or weld.",
    )
    torch.save(checkpoint, out / "insert_bc.pt")
    actor = load_actor(out / "insert_bc.pt")
    evaluation = []
    test_rng = np.random.default_rng(98211)
    for i in range(12):
        _, report = rollout(s, state, test_rng, actor=actor)
        evaluation.append(report)
        print("evaluation", i, report, flush=True)
    report = dict(
        algorithm="behavior_cloning",
        samples=len(data),
        expert_episodes=episodes,
        validation_mse=best,
        test_episodes=evaluation,
        test_successes=sum(x["success"] for x in evaluation),
        test_total=len(evaluation),
    )
    (out / "training_report.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps({k: v for k, v in report.items() if not isinstance(v, list)}),
        flush=True,
    )


if __name__ == "__main__":
    main()
