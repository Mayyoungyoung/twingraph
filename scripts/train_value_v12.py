"""Train on recorded physical outcomes; split only by whole layout identity.

Historical V9 rows are development-transfer data, not V12 printed-kit tests.
Input graphs are recompiled from archived pre-execution proposals; this is
explicitly weaker provenance than an archived pre-execution graph hash.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch import nn

from simbench.value.graph_value_v12 import SCHEMA, STAGES, encode_graph
from simbench.value.plan import Call, PlanIR, argument, digest
from simbench.value.skill_graph import compile_graph
from simbench.value.stage_v5 import PARTS, PRECEDENCE, nominal_targets, stage_calls
from simbench.value.value_v12 import MODEL_KINDS, LOSS_CONFIG, StageValueNet, collate, supervised_loss

SELECTION_CRITERION = "maximum validation Hit@K; minimum normalized first-success calls; minimum Brier; earliest epoch/fixed model order"


def selection_key(report):
    """Predeclared lexicographic utility; never choose on test or cached time."""
    return (-float(report["success"]), float(report["normalized_first_success_calls"]), float(report["brier"]))


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def historical_graph(detail):
    """Offline schema migration; no MuJoCo scene or rollout outcome is read."""
    p = detail["candidate"]
    targets = nominal_targets()
    calls = [c for i, part in enumerate(p["order"]) for c in
             stage_calls(part, targets[part], p["choices"][part], i, v7=True, functional_clearance=True)]
    calls.append(Call("final_home", "move", dict(target=argument("home"))))
    cid = digest(dict(proposal=p, observation=detail["pre_execution_observation"]))[:20]
    prefix = dict(id=cid, part=p["order"][0], execution="program", start_state="unrecorded_historical_robot",
                  order=p["order"], choices=p["choices"], targets=targets,
                  steps=[dict(skill=c.skill, params={k: a.value for k, a in c.arguments.items()}) for c in calls[:10]])
    plan = PlanIR(cid, calls, 10, prefix, "unknown", protocol="assembly.program.feedback.v2")
    raw = detail["pre_execution_observation"]
    obs = dict(robot={}, objects={part: dict(position=r.get("position_m"), quaternion=r.get("quat_wxyz"),
               valid=r.get("valid"), quality=r.get("quality"), fit_residual_m=r.get("fit_residual_m"))
               for part, r in raw["objects"].items()}, goals=[dict(predicate="functional_task")])
    return dict(assembly=compile_graph(obs, plan), cleaning={k: p[k] for k in
                ("wipe_variant", "wipe_force", "wipe_duration")},
                completion=dict(completed=[]), phase_edges=[["cleaning", "carriage"], *map(list, PRECEDENCE),
                ["handle", "bidirectional_stroke"], ["bidirectional_stroke", "pin_retention"]])


def local_labels(result):
    """Only observed stage outcomes are supervised; unexecuted stages censored."""
    y = np.zeros(len(STAGES), np.float32)
    mask = np.zeros_like(y)
    passes = result.get("stage_passes", {})
    mask[0] = float("cleaning_pass" in passes)
    y[0] = bool(passes.get("cleaning_pass"))
    if bool(result["success"]):
        return np.ones_like(y), np.ones_like(y)
    steps = result.get("executed_parameters", [])
    for step in steps:
        params = step.get("params", {})
        if params.get("phase") == "retained_after_stroke":
            # Retention is a distinct post-stroke event. A failed recheck must
            # not relabel a previously successful insertion as never inserted.
            y[7], mask[7] = float(bool(step.get("ok"))), 1.
            continue
        part = step.get("params", {}).get("part")
        if part not in STAGES:
            continue
        i = STAGES.index(part)
        skill = step.get("skill", "")
        if step.get("ok") and skill in ("inspect_placement", "inspect_pin_inserted"):
            y[i], mask[i] = 1., 1.
        if not step.get("ok", True):
            y[i], mask[i] = 0., 1.
    if passes.get("assembly_pass"):
        y[1:6], mask[1:6] = 1., 1.
        mask[6] = 1.
        y[6] = bool(passes.get("functional_test_pass"))
    # An explicit completion boundary can label a successful phase without
    # inferring unexecuted negatives from initialized false summary flags.
    aliases = {"stroke": "bidirectional_stroke", "retention": "pin_retention"}
    for boundary in result.get("boundaries", []):
        for stage in boundary.get("completed", []):
            stage = aliases.get(stage, stage)
            if stage in STAGES:
                i = STAGES.index(stage)
                y[i], mask[i] = 1., 1.
    return y, mask


def collect_historical(roots):
    grouped = {}
    manifests = []
    for root in roots:
        for path in sorted(Path(root).glob("seed_*/seed_*/*/*/result.json")):
            raw = path.read_bytes()
            d = json.loads(raw)
            if d.get("task_version") != "functional_assembly_v9_funnel_r1":
                raise ValueError("unexpected historical task version")
            seed = int(d["seed"])
            source = d["candidate"]["source"]
            key = (seed, source, d["candidate"]["name"])
            entry = grouped.setdefault(key, dict(seed=seed, source=source, name=d["candidate"]["name"], records=[]))
            entry["records"].append((d, path))
            manifests.append(dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), layout=seed,
                                  condition=d["condition"], candidate=d["candidate"]["name"]))
    rows = []
    for key, group in sorted(grouped.items()):
        records = group.pop("records")
        conditions = {d["condition"] for d, _ in records}
        if conditions not in ({"nominal", "light_low", "light_high"}, {"nominal"}) or len(records) != len(conditions):
            raise ValueError(f"missing physical repeats: {key}")
        if len({d["pre_execution_observation"]["sha256"] for d, _ in records}) != 1:
            raise ValueError("observation changes within repeated physical conditions")
        graph = historical_graph(records[0][0])
        yy, mm = zip(*(local_labels(d["result"]) for d, _ in records))
        total_mask = np.stack(mm).sum(0)
        label = (np.stack(yy)*np.stack(mm)).sum(0)/np.maximum(total_mask, 1.)
        rows.append(dict(**group, encoded=encode_graph(graph), graph_sha256=digest(graph),
            y=np.mean([float(d["result"]["success"]) for d, _ in records]), local_y=label,
            local_mask=total_mask/len(records), outcomes=[dict(condition=d["condition"], success=d["result"]["success"],
            seconds=d["result"]["wall_seconds"]) for d, _ in records]))
    if not rows:
        raise ValueError("no historical records found")
    return rows, manifests


def random_metrics(success, seconds, k):
    n, f = len(success), sum(not y for y in success)
    choose = lambda a, b: math.comb(a, b) if 0 <= b <= a else 0
    hit = 1-choose(f, k)/choose(n, k)
    calls = sum(choose(f, i)/choose(n, i) for i in range(k))
    wall = 0.
    for y, duration in zip(success, seconds):
        available_fail = f - int(not y)
        wall += duration*sum(choose(available_fail, i)/choose(n-1, i)/n for i in range(k))
    return dict(success=hit, calls=calls, seconds=wall)


def evaluate(rows, predictions, k=4):
    cases = []
    for seed, source in sorted({(r["seed"], r["source"]) for r in rows}):
        indexes = [i for i, r in enumerate(rows) if (r["seed"], r["source"]) == (seed, source)]
        order = sorted(indexes, key=lambda i: -float(predictions[i]))
        conditions = sorted({o["condition"] for i in indexes for o in rows[i]["outcomes"]})
        for condition in conditions:
            data = {i: next(o for o in rows[i]["outcomes"] if o["condition"] == condition) for i in indexes}
            selected = []
            for i in order[:k]:
                selected.append(i)
                if data[i]["success"]:
                    break
            progressive=[]
            for i in order:
                progressive.append(i)
                if data[i]["success"]: break
            rand = random_metrics([data[i]["success"] for i in indexes], [data[i]["seconds"] for i in indexes], min(k, len(indexes)))
            cases.append(dict(layout=seed, source=source, condition=condition, success=bool(data[selected[-1]]["success"]),
                calls=len(selected), seconds=sum(data[i]["seconds"] for i in selected), random=rand,
                all_twin=dict(success=any(data[i]["success"] for i in indexes), calls=len(indexes),
                    seconds=sum(data[i]["seconds"] for i in indexes)),
                random_early_stop=random_metrics([data[i]["success"] for i in indexes], [data[i]["seconds"] for i in indexes], len(indexes)),
                value_early_stop=dict(success=bool(data[progressive[-1]]["success"]),calls=len(progressive),
                    seconds=sum(data[i]["seconds"] for i in progressive),
                    batches_opened=math.ceil(len(progressive)/min(k,len(indexes))),
                    same_pool_existence_coverage=True),
                normalized_first_success_calls=len(progressive)/len(indexes),
                first_success_rank=len(progressive) if data[progressive[-1]]["success"] else None,
                ranking=[rows[i]["name"] for i in order], pool_success=sum(data[i]["success"] for i in indexes)))
    return dict(layouts=len({r["seed"] for r in rows}), candidate_rows=len(rows), k=k,
        success=np.mean([r["success"] for r in cases]), random_success=np.mean([r["random"]["success"] for r in cases]),
        calls=np.mean([r["calls"] for r in cases]), random_calls=np.mean([r["random"]["calls"] for r in cases]),
        verification_seconds=np.mean([r["seconds"] for r in cases]),
        random_verification_seconds=np.mean([r["random"]["seconds"] for r in cases]),
        all_twin_success=np.mean([r["all_twin"]["success"] for r in cases]),
        all_twin_verification_seconds=np.mean([r["all_twin"]["seconds"] for r in cases]),
        random_early_stop_verification_seconds=np.mean([r["random_early_stop"]["seconds"] for r in cases]),
        random_early_stop_success=np.mean([r["random_early_stop"]["success"] for r in cases]),
        random_early_stop_calls=np.mean([r["random_early_stop"]["calls"] for r in cases]),
        value_early_stop_success=np.mean([r["value_early_stop"]["success"] for r in cases]),
        value_early_stop_calls=np.mean([r["value_early_stop"]["calls"] for r in cases]),
        value_early_stop_verification_seconds=np.mean([r["value_early_stop"]["seconds"] for r in cases]),
        normalized_first_success_calls=float(np.mean([r["normalized_first_success_calls"] for r in cases])),
        feasible_layout_cases=sum(r["pool_success"] > 0 for r in cases),
        hit_at_k_given_feasible=(float(np.mean([r["success"] for r in cases if r["pool_success"] > 0]))
                                if any(r["pool_success"] > 0 for r in cases) else None),
        brier=float(np.mean((np.asarray(predictions)-np.asarray([r["y"] for r in rows]))**2)), cases=cases)


@torch.inference_mode()
def predict(model, rows, device):
    model.eval()
    return model(collate([r["encoded"] for r in rows], device))["plan_logit"].sigmoid().cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--roots", nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    rows, manifest = collect_historical(args.roots)
    np.savez_compressed(args.out/"encoded_dataset.npz", x=np.stack([r["encoded"]["x"] for r in rows]),
        relations=np.stack([r["encoded"]["relations"] for r in rows]), active=np.stack([r["encoded"]["active"] for r in rows]),
        y=np.asarray([r["y"] for r in rows]), local_y=np.stack([r["local_y"] for r in rows]),
        local_mask=np.stack([r["local_mask"] for r in rows]), layout=np.asarray([r["seed"] for r in rows]))
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), root/"simbench/value/graph_value_v12.py", root/"simbench/value/value_v12.py",
               root/"simbench/value/stage_v5.py", root/"simbench/value/skill_graph.py", root/"simbench/assembly/ports.py"]
    dump(args.out/"runtime_provenance.json", dict(python=platform.python_version(), torch=torch.__version__,
         numpy=np.__version__, sources={str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest() for f in sources},
         encoded_dataset_sha256=hashlib.sha256((args.out/"encoded_dataset.npz").read_bytes()).hexdigest()))
    splits = dict(train=[*range(1302, 1310), 1320, 1321], validation=[*range(1310, 1314), 1322], historical_holdout=list(range(1314, 1320)))
    split_rows = {s: [r for r in rows if r["seed"] in ids] for s, ids in splits.items()}
    if set(splits["train"]) & set(splits["validation"]):
        raise ValueError("layout leakage")
    dump(args.out/"data_manifest.json", dict(schema=SCHEMA, splits=splits, records=manifest,
         graph_provenance="recompiled pre-execution proposal + RGB-D; original V9 did not archive graph hashes",
         labels="real MuJoCo V9/V10 full-task trial outcomes; no V12 printed-kit labels; no synthetic outcomes",
         evaluation_status="historical development replay, not a new unseen printed-kit test",
         predeclared_k=4, epochs=args.epochs, seed=1212,
         split_note="Original V9 layout split retained; already-published V10 1320/1321 added to train and 1322 to validation for retrospective transfer development. No V10 claim of unseen testing.",
         rows=[dict(layout=r["seed"], source=r["source"], candidate=r["name"], graph_sha256=r["graph_sha256"],
                    full_success=r["y"], local_y=r["local_y"].tolist(), local_mask=r["local_mask"].tolist()) for r in rows]))
    report = {}
    tr, va = split_rows["train"], split_rows["validation"]
    for kind in MODEL_KINDS:
        torch.manual_seed(1212)
        rng = np.random.default_rng(1212)
        model = StageValueNet(tr[0]["encoded"]["x"].shape[-1], kind=kind).to(args.device)
        allx = np.concatenate([r["encoded"]["x"] for r in tr])
        model.mean.copy_(torch.as_tensor(allx.mean(0), device=args.device))
        model.scale.copy_(torch.as_tensor(np.maximum(allx.std(0), .1), device=args.device))
        opt = torch.optim.AdamW(model.parameters(), lr=.003 if kind == "linear" else .001, weight_decay=.03)
        best, state, history = None, None, []
        started = time.perf_counter()
        for epoch in range(args.epochs):
            losses = []
            for seed in rng.permutation(splits["train"]):
                group = [r for r in tr if r["seed"] == seed]
                if not group:
                    continue
                model.train()
                opt.zero_grad()
                pred = model(collate([r["encoded"] for r in group], args.device))
                y = torch.as_tensor([r["y"] for r in group], dtype=torch.float32, device=args.device)
                ly = torch.as_tensor(np.stack([r["local_y"] for r in group]), device=args.device)
                lm = torch.as_tensor(np.stack([r["local_mask"] for r in group]), device=args.device)
                loss = supervised_loss(pred, y, ly, lm, local_weight=0. if kind == "linear" else LOSS_CONFIG["local_weight"])
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.)
                opt.step()
                losses.append(float(loss))
            score = predict(model, va, args.device)
            utility = evaluate(va, score, 4)
            key = selection_key(utility)
            history.append(dict(epoch=epoch+1, train_loss=float(np.mean(losses)), validation_brier=utility["brier"],
                validation_hit_at_k=utility["success"], validation_normalized_first_success_calls=utility["normalized_first_success_calls"]))
            if best is None or key < best:
                best = key
                chosen = epoch+1
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(state)
        path = args.out/f"{kind}.pt"
        torch.save(dict(schema=SCHEMA, model_config=model.config, state_dict=state, train_layouts=splits["train"],
             validation_layouts=splits["validation"], kind=kind, selected_epoch=chosen, selection_criterion=SELECTION_CRITERION,
             loss_config=LOSS_CONFIG, selected_validation_key=list(best),
             label_domain="historical_v9_functional_assembly", manifest_sha256=hashlib.sha256((args.out/"data_manifest.json").read_bytes()).hexdigest()), path)
        report[kind] = dict(selected_epoch=chosen, training_seconds=time.perf_counter()-started,
             checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
             **{s: evaluate(rr, predict(model, rr, args.device)) for s, rr in split_rows.items()})
        dump(args.out/f"{kind}_history.json", history)
        dump(args.out/"metrics.json", report)
        print(json.dumps({kind: {s: {k:v for k,v in report[kind][s].items() if k != "cases"} for s in splits}}), flush=True)
    selected = min(report, key=lambda name: selection_key(report[name]["validation"]))
    dump(args.out/"selection.json", dict(selected=selected, criterion=SELECTION_CRITERION, k=4,
         selection_did_not_use_historical_holdout=True,
         restriction="Printed-kit transfer is unvalidated; fresh V12 labels and layout-heldout evaluation required."))


if __name__ == "__main__":
    main()
