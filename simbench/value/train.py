"""Grouped supervised training on a CUDA GPU or CPU."""
import argparse
from dataclasses import asdict
import json
import random
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from .collect import dump
from .dataset import load_groups, split_name
from .encode import collate
from .network import ModelConfig, PlanValueNet
from .losses import probability_loss, keep_loss
from .evaluate import evaluate
from .vision import ENCODER
from .plan import digest, plain


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="results/value/model")
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--batch-groups", type=int, default=2)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--keep-weight", type=float, default=0.2)
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--objective", choices=["dual", "direct"], default="direct")
    p.add_argument("--no-vision", action="store_true")
    a = p.parse_args()
    if min(a.epochs, a.batch_groups, a.k) < 1:
        p.error("epochs, batch-groups and k must be positive")
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.set_num_threads(4)
    groups = load_groups(a.data, vision=not a.no_vision)
    dataset_sha256 = digest(
        [
            dict(
                group_id=g.id,
                input_sha256=digest(g.inputs),
                trials=g.trials,
                prefix=g.prefix,
                full=g.full,
            )
            for g in sorted(groups, key=lambda g: g.id)
        ]
    )
    splits = {
        name: [g for g in groups if split_name(g.id, a.seed) == name]
        for name in ("train", "val", "test")
    }
    if any(not group for group in splits.values()):
        raise ValueError(
            "need nonempty group-held-out train/validation/test splits; collect more configurations"
        )
    if not any(g.full.max() > 0 for g in splits["train"]):
        raise ValueError(
            "all-zero training labels: repair underlying execution before training"
        )
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dump(
        out / "splits.json",
        {key: [g.id for g in value] for key, value in splits.items()},
    )
    config = ModelConfig(width=a.width, layers=a.layers, vision=not a.no_vision)
    model = PlanValueNet(config).to(a.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    best = (-float("inf"), -float("inf"), -float("inf"))
    history = []
    started = time.perf_counter()
    for epoch in range(a.epochs):
        model.train()
        order = np.random.permutation(len(splits["train"]))
        losses = []
        for start in range(0, len(order), a.batch_groups):
            batch_groups = [
                splits["train"][i] for i in order[start : start + a.batch_groups]
            ]
            enc = [x for g in batch_groups for x in g.encoded]
            vis = (
                [g.visual for g in batch_groups for _ in g.encoded]
                if config.vision
                else None
            )
            batch = collate(enc, vis, a.device)
            counts = [
                torch.as_tensor(
                    np.concatenate([getattr(g, key) for g in batch_groups]),
                    device=a.device,
                )
                for key in ("trials", "prefix", "full")
            ]
            group_ids = torch.tensor(
                [i for i, g in enumerate(batch_groups) for _ in g.plans],
                device=a.device,
            )
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            n, pa, full = counts
            if a.objective == "dual":
                loss = probability_loss(output, n, pa, full)
                if epoch >= a.warmup_epochs and a.keep_weight:
                    logq = F.logsigmoid(output["prefix_logit"]) + F.logsigmoid(
                        output["suffix_logit"]
                    )
                    loss = loss + a.keep_weight * keep_loss(
                        logq, full / n, group_ids, a.k, a.epsilon
                    )
            else:
                loss = F.binary_cross_entropy_with_logits(
                    output["direct_logit"], full / n
                )
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        val = evaluate(model, splits["val"], a.device, a.k, a.epsilon, a.objective)
        metrics = val["methods"]["learned"]
        selection = (
            metrics["hit"] if metrics["hit"] is not None else 0.0,
            -metrics["regret"],
            -val["brier_to_empirical_rate"],
        )
        row = dict(
            epoch=epoch + 1,
            loss=float(np.mean(losses)),
            val_hit=metrics["hit"],
            val_regret=metrics["regret"],
            seconds=time.perf_counter() - started,
        )
        history.append(row)
        print(json.dumps(row), flush=True)
        if selection > best:
            best = selection
            torch.save(
                dict(
                    schema="twingraph.value.v1",
                    dataset_sha256=dataset_sha256,
                    model_config=asdict(config),
                    state_dict=model.state_dict(),
                    objective=a.objective,
                    protocol=groups[0].inputs["protocol"],
                    vision_encoder=ENCODER if config.vision else None,
                    training=vars(a),
                    epoch=epoch + 1,
                    validation={k: v for k, v in val.items() if k != "predictions"},
                ),
                out / "best.pt",
            )
        dump(out / "history.json", history)
    checkpoint = torch.load(out / "best.pt", map_location=a.device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    # Test is evaluated ONCE, after checkpoint selection is complete.
    test = evaluate(model, splits["test"], a.device, a.k, a.epsilon, a.objective)
    dump(out / "test_metrics.json", test)
    dump(
        out / "training_summary.json",
        dict(
            config=vars(a),
            dataset_sha256=dataset_sha256,
            device=str(a.device),
            gpu=torch.cuda.get_device_name() if a.device.startswith("cuda") else None,
            torch=torch.__version__,
            parameters=sum(p.numel() for p in model.parameters()),
            splits={k: len(v) for k, v in splits.items()},
            trials=int(sum(g.trials.sum() for g in groups)),
            selected_epoch=checkpoint["epoch"],
            wall_seconds=time.perf_counter() - started,
            limitations=[
                "finite simulation repetitions",
                "privileged simulator pose/segmentation",
                "one parameterized pin subtask family; no cross-family or real-robot claim",
            ],
        ),
    )
    print(
        json.dumps(plain({k: v for k, v in test.items() if k != "predictions"})),
        flush=True,
    )


if __name__ == "__main__":
    main()
