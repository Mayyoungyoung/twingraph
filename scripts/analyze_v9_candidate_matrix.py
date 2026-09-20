"""Group-safe V9 screening replay and two actually trained numeric scorers."""

import argparse
import csv
import json
from pathlib import Path
import random
import time

import numpy as np
from simbench.value.v9_candidates import proposals
from scripts.collect_v9_candidate_matrix import CONDITIONS

PARTS = ("carriage", "end_stop", "pin_left", "pin_right", "handle", "wipe_tool")
KS = (1, 2, 4, 8, 12)


def load(root):
    candidates = proposals()
    names = [p["name"] for p in candidates]
    cases = {}
    seen_geometry = {}
    seen_observation = {}
    seen_seed_geometry = {}
    model_parameter_hashes = {}
    frozen_sources = None
    issues = []
    for path in sorted(Path(root).glob("seed_*/summary.json")):
        summary = json.loads(path.read_text())
        if frozen_sources is None:
            frozen_sources = summary.get("source_sha256")
        elif frozen_sources != summary.get("source_sha256"):
            issues.append(f"source hash mismatch in {path.parent.name}")
        seed = int(path.parent.name.split("_")[-1])
        rows = summary["rows"]
        if len(rows) != 36:
            issues.append(f"seed {seed}: {len(rows)} of 36 rows")
        for row in rows:
            if "infrastructure_error" in row:
                issues.append(f"seed {seed}: {row['candidate']} {row['condition']}: {row['infrastructure_error']}")
                continue
            condition = row["condition"]
            key = (seed, condition)
            case = cases.setdefault(key, {})
            if row["candidate"] in case:
                issues.append(f"duplicate {key} {row['candidate']}")
                continue
            result_path = path.parent / f"seed_{seed}" / condition / row["candidate"] / "result.json"
            detail = json.loads(result_path.read_text())
            geometry = detail["geometry_sha256"]
            old_geometry = seen_seed_geometry.setdefault(seed, geometry)
            if old_geometry != geometry:
                issues.append(f"seed {seed}: geometry changed between candidates or conditions")
            old_seed = seen_geometry.setdefault(geometry, seed)
            if old_seed != seed:
                issues.append(f"geometry duplicate seeds {old_seed}, {seed}")
            observation_hash = detail["pre_execution_observation"]["sha256"]
            old_observation = seen_observation.setdefault(seed, observation_hash)
            if old_observation != observation_hash:
                issues.append(f"seed {seed}: initial RGB-D input changed between rollouts")
            actual = detail["result"]["applied_parameters"]
            expected_friction, expected_gain = CONDITIONS[condition]
            if (abs(actual["friction_scale"] - expected_friction) > 1e-12
                    or abs(actual["actuator_gain_scale"] - expected_gain) > 1e-12):
                issues.append(f"seed {seed}: perturbation not applied for {condition}")
            hashes = (actual["geom_friction_sha256"], actual["actuator_gain_sha256"])
            previous_hashes = model_parameter_hashes.setdefault((seed, condition), hashes)
            if previous_hashes != hashes:
                issues.append(f"seed {seed}: physical condition differs between candidates")
            case[row["candidate"]] = dict(summary=row,
                observation=detail["pre_execution_observation"],
                proposal=detail["candidate"])
    for key, case in cases.items():
        if set(case) != set(names):
            issues.append(f"incomplete case {key}: {len(case)} of 12")
    for seed in seen_seed_geometry:
        hashes = [model_parameter_hashes.get((seed, condition)) for condition in CONDITIONS]
        if all(item is not None for item in hashes) and len(set(hashes)) != len(hashes):
            issues.append(f"seed {seed}: condition perturbations have identical model arrays")
    if issues:
        raise ValueError("matrix incomplete or duplicated:\n" + "\n".join(issues))
    return cases, candidates


def features(observation, proposal):
    output = []
    for part in PARTS:
        row = observation["objects"].get(part, {})
        output.extend(float(v) for v in (row.get("position_m") or [0., 0., 0.]))
        output.extend((float(row.get("quality") or 0.), float(bool(row.get("valid")))))
    for part in PARTS[:-1]:
        choice = proposal["choices"][part]
        output.extend(float(choice.get(key, 0.)) for key in ("yaw", "height", "clearance", "force", "speed"))
    output.extend((float(proposal["order"].index("pin_left") > proposal["order"].index("pin_right")),
                   float(proposal["wipe_variant"]), float(proposal["wipe_force"]),
                   float(proposal["wipe_duration"])))
    return np.asarray(output, dtype=np.float32)


def rule_score(observation, proposal):
    # Declared before examining labels: use only source camera positions and
    # the executable proposal.  No result, simulator truth, or condition ID.
    name = proposal["name"]
    objects = observation["objects"]
    pin_quality = min(float(objects[p].get("quality") or 0.) for p in ("pin_left", "pin_right"))
    left_x = float(objects["pin_left"]["position_m"][0])
    right_x = float(objects["pin_right"]["position_m"][0])
    score = {"reference": 1.0, "pin_slow": .7, "pin_order": .6,
             "pin_firm": .5, "pin_gentle": .4, "pin_high_grasp": .3,
             "pin_low_grasp": .2, "transfer_high": .15,
             "transfer_low": .1, "stop_slow": .05,
             "wipe_reverse": 0., "wipe_long": -.05}[name]
    if name == "pin_slow" and pin_quality < .83:
        score += .5
    if name == "pin_order" and right_x > left_x + .003:
        score += .5
    return score


def fit_models(cases, names, train_seeds, val_seeds, out, epochs=100):
    import torch
    from torch import nn
    from simbench.value.value_v6 import RobustProgramNet
    torch.manual_seed(51)
    torch.set_num_threads(2)
    def records(seeds):
        rows = []
        for seed in seeds:
            base = cases[(seed, "nominal")]
            for name in names:
                entry = base[name]
                outcomes = [float(cases[(seed, condition)][name]["summary"]["success"])
                            for condition in ("nominal", "light_low", "light_high")]
                rows.append((features(entry["observation"], entry["proposal"]), np.mean(outcomes)))
        return rows
    train, val = records(train_seeds), records(val_seeds)
    x = torch.as_tensor(np.stack([r[0] for r in train]))
    y = torch.as_tensor([r[1] for r in train], dtype=torch.float32)
    xv = torch.as_tensor(np.stack([r[0] for r in val]))
    yv = torch.as_tensor([r[1] for r in val], dtype=torch.float32)
    mean = x.mean(0); std = x.std(0, unbiased=False).clamp_min(1e-5)
    summaries = {}
    for name in ("numeric_logistic", "existing_value_v6_architecture"):
        started = time.perf_counter()
        if name == "numeric_logistic":
            model = nn.Linear(x.shape[1], 1)
            def forward(z): return model((z - mean) / std).squeeze(-1)
        else:
            model = RobustProgramNet(x.shape[1], "none")
            model.mean.copy_(mean); model.std.copy_(std)
            def forward(z): return model(z)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.03)
        best = float("inf"); chosen = None; history = []
        for epoch in range(epochs):
            model.train(); optimizer.zero_grad()
            z = forward(x)
            loss = nn.functional.binary_cross_entropy_with_logits(z, y)
            loss.backward(); optimizer.step()
            model.eval()
            with torch.no_grad():
                prediction = torch.sigmoid(forward(xv))
                brier = float(((prediction - yv) ** 2).mean())
            history.append(dict(epoch=epoch + 1, train_bce=float(loss), val_brier=brier))
            if brier < best:
                best = brier
                chosen = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        model.load_state_dict(chosen)
        checkpoint = out / f"{name}.pt"
        torch.save(dict(state_dict=chosen, mean=mean, std=std,
                        input_dim=x.shape[1], architecture=name,
                        train_seeds=train_seeds, val_seeds=val_seeds,
                        epochs=epochs, best_val_brier=best), checkpoint)
        (out / f"{name}_history.json").write_text(json.dumps(history, indent=2))
        summaries[name] = dict(checkpoint=str(checkpoint), best_val_brier=best,
                               training_seconds=time.perf_counter() - started,
                               train_rows=len(train), val_rows=len(val))
    return summaries


def rank_model(case, names, checkpoint):
    import torch
    from torch import nn
    from simbench.value.value_v6 import RobustProgramNet
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    x = torch.as_tensor(np.stack([features(case[name]["observation"], case[name]["proposal"])
                                   for name in names]))
    if saved["architecture"] == "numeric_logistic":
        model = nn.Linear(saved["input_dim"], 1)
        model.load_state_dict(saved["state_dict"])
        with torch.no_grad(): score = model((x - saved["mean"]) / saved["std"]).squeeze(-1).numpy()
    else:
        model = RobustProgramNet(saved["input_dim"], "none")
        model.load_state_dict(saved["state_dict"])
        model.eval()
        with torch.no_grad(): score = model(x).numpy()
    return [names[i] for i in np.argsort(-score, kind="stable")]


def outcome(case, order, k, overhead=0.):
    attempted = order[:k]
    selected = []
    for name in attempted:
        selected.append(name)
        if case[name]["summary"]["success"]:
            break
    seconds = sum(case[name]["summary"]["total_wall_seconds"] for name in selected)
    verification = sum(case[name]["summary"]["wall_seconds"] for name in selected)
    return dict(success=bool(case[selected[-1]]["summary"]["success"]),
                verification_count=len(selected), verification_wall_seconds=verification,
                full_system_wall_seconds=seconds + overhead,
                tried=selected)


def replay(cases, names, seeds, checkpoints, draws=256):
    rng = random.Random(5117)
    rows = []
    for (seed, condition), case in sorted(cases.items()):
        if seed not in seeds:
            continue
        observation = case[names[0]]["observation"]
        t0 = time.perf_counter(); generated = proposals(); generation_seconds = time.perf_counter() - t0
        assert [p["name"] for p in generated] == names
        t0 = time.perf_counter()
        ruled = sorted(names, key=lambda n: -rule_score(observation, case[n]["proposal"]))
        rule_seconds = time.perf_counter() - t0
        orders = {
            "fixed_reference": [names[0]],
            "simple_rule": ruled,
            "full_12": names,
        }
        overheads = {"fixed_reference": 0., "simple_rule": generation_seconds + rule_seconds,
                     "full_12": generation_seconds}
        for label, checkpoint in checkpoints.items():
            t0 = time.perf_counter()
            orders[label] = rank_model(case, names, checkpoint)
            overheads[label] = generation_seconds + time.perf_counter() - t0
        for method, order in orders.items():
            for k in (KS if method != "fixed_reference" else (1,)):
                row = outcome(case, order, k, overheads[method])
                rows.append(dict(seed=seed, condition=condition, method=method, k=k, **row))
        for _ in range(draws):
            t0 = time.perf_counter()
            order = rng.sample(names, len(names))
            overhead = generation_seconds + time.perf_counter() - t0
            for k in KS:
                row = outcome(case, order, k, overhead)
                rows.append(dict(seed=seed, condition=condition, method="random", k=k, **row))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cases, pool = load(args.matrix)
    names = [p["name"] for p in pool]
    seeds = sorted({key[0] for key in cases})
    if len(seeds) != 12:
        raise ValueError(f"expected 12 distinct development geometries, got {len(seeds)}")
    with (out / "candidate_matrix.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("seed", "condition", "candidate", "success",
            "error", "executed_steps", "verification_wall_seconds", "full_rollout_wall_seconds",
            "geometry_sha256", "initial_observation_sha256"))
        writer.writeheader()
        for (seed, condition), case in sorted(cases.items()):
            for name in names:
                record = case[name]["summary"]
                writer.writerow(dict(seed=seed, condition=condition, candidate=name,
                    success=record["success"], error=record["error"],
                    executed_steps=record["steps"], verification_wall_seconds=record["wall_seconds"],
                    full_rollout_wall_seconds=record["total_wall_seconds"],
                    geometry_sha256=record["geometry_sha256"],
                    initial_observation_sha256=case[name]["observation"]["sha256"]))
    train_seeds, val_seeds = seeds[:8], seeds[8:]
    labels = {seed: sum(int(cases[(seed, cond)][name]["summary"]["success"])
                         for cond in ("nominal", "light_low", "light_high") for name in names)
              for seed in seeds}
    (out / "split.json").write_text(json.dumps(dict(train=train_seeds, validation=val_seeds,
            independent_target=[], successes_by_seed=labels), indent=2))
    summary = dict(layouts=len(seeds), candidates=len(names), conditions=3,
                   random_replay_per_case=256, random_seed=5117,
                   successes=sum(labels.values()), successes_by_seed=labels,
                   successes_by_candidate={name: sum(int(cases[(seed, cond)][name]["summary"]["success"])
                       for seed in seeds for cond in CONDITIONS) for name in names},
                   successes_by_condition={cond: sum(int(cases[(seed, cond)][name]["summary"]["success"])
                       for seed in seeds for name in names) for cond in CONDITIONS},
                   reference_successes=sum(int(cases[(seed, cond)][names[0]]["summary"]["success"])
                       for seed in seeds for cond in CONDITIONS),
                   reference_fails_alternative_succeeds=[dict(seed=seed, condition=cond,
                       successful_alternatives=[name for name in names[1:]
                           if cases[(seed, cond)][name]["summary"]["success"]])
                       for seed in seeds for cond in CONDITIONS
                       if not cases[(seed, cond)][names[0]]["summary"]["success"]
                       and any(cases[(seed, cond)][name]["summary"]["success"] for name in names[1:])],
                   non_reference_successes=sum(int(cases[(seed, cond)][name]["summary"]["success"])
                       for seed in seeds for cond in ("nominal", "light_low", "light_high")
                       for name in names[1:]))
    checkpoints = {}
    if len(set(labels.values())) > 1 and summary["successes"] > 0:
        summary["model_training"] = fit_models(cases, names, train_seeds, val_seeds, out, args.epochs)
        checkpoints = {name: row["checkpoint"] for name, row in summary["model_training"].items()}
    else:
        summary["training_skipped"] = "no positive and varying layout outcomes"
    for split, selected in (("training", train_seeds), ("validation", val_seeds)):
        rows = replay(cases, names, selected, checkpoints)
        grouped = {}
        for row in rows:
            key = (row["method"], row["k"])
            acc = grouped.setdefault(key, [])
            acc.append(row)
        summary[split] = [dict(method=method, k=k, cases=len(group),
            success_rate=float(np.mean([r["success"] for r in group])),
            mean_verifications=float(np.mean([r["verification_count"] for r in group])),
            mean_verification_wall_seconds=float(np.mean([r["verification_wall_seconds"] for r in group])),
            mean_full_system_wall_seconds=float(np.mean([r["full_system_wall_seconds"] for r in group])))
            for (method, k), group in sorted(grouped.items())]
        (out / f"{split}_replay.json").write_text(json.dumps(rows, indent=2))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({key: summary[key] for key in ("layouts", "successes", "non_reference_successes")}), flush=True)


if __name__ == "__main__":
    main()
