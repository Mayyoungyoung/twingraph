"""State + ordered executable program feasibility, with automatic input schema.

Training consumes direct physical outcomes. The small MLP is an implementation
choice; its input columns come from the complete execution interface, not an
authored product feature vector. No graph message passing or weighted reward.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn

from .collect import dump
from .plan import PlanIR, digest
from .program_input_v5 import InputSchema, encoding_hash, program_record, source_manifest
from .skill_graph import compile_graph, interface_hash

SCHEMA = "twingraph.value.v5"


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def request_dispositions(roots,splits=("train","val")):
    rows=[];seen={}
    for root in roots:
        for directory in sorted(Path(root).glob("group_*")):
            if not directory.is_dir():continue
            path=directory/"request.json"
            if not path.exists():raise ValueError(f"group has no immutable request: {directory}")
            request=json.loads(path.read_text());split=request["split"]
            if request["seed"] in seen and seen[request["seed"]]!=split:
                raise ValueError("scene seed crosses declared splits")
            seen[request["seed"]]=split
            if split not in splits:continue
            source=json.loads((directory/"source.json").read_text())
            if digest(source)!=request["source_sha256"]:
                raise ValueError("request/source binding mismatch")
            complete=(directory/"complete.json").is_file()
            failure=(directory/"failure.json").is_file()
            if complete and (directory/"censored.json").exists():
                raise ValueError("contradictory completion/censoring records")
            if complete and failure:raise ValueError("contradictory completion/failure records")
            if complete:status="completed"
            elif failure and not (directory/"inputs.json").exists():
                error=json.loads((directory/"failure.json").read_text())
                failed_request=error.get("request",{})
                if error.get("phase")!="before_inputs" or any(
                        failed_request.get(k)!=request[k] for k in ("seed","split","domain","n","repeats")) or \
                        failed_request.get("timeout")!=request["timeout_seconds"]:
                    raise ValueError("pre-input failure/request binding mismatch")
                kind=error.get("exception_type");message=error.get("message","")
                accepted=(kind=="CandidateGenerationError" or
                          kind=="ValueError" and message.startswith("IK unreachable:") or
                          kind=="SkillFailure" and not any(k in message for k in ("unknown skill","unexpected keyword","invalid parameter","missing a required")))
                if not accepted:raise ValueError(f"unclassified pre-input failure {directory}: {kind}: {message}")
                status="failed_before_inputs"
            else:raise ValueError(f"incomplete/censored/invalid group {directory}")
            rows.append(dict(directory=str(directory),request=request,source_sha256=digest(source),status=status))
    if len({(r["request"]["split"],r["request"]["seed"]) for r in rows})!=len(rows):
        raise ValueError("duplicate requested scene configuration")
    return rows


def load_groups(roots, splits=("train", "val")):
    dispositions=request_dispositions(roots,splits)
    by_directory={r["directory"]:r for r in dispositions}
    groups, seen, configurations = [], set(), {}
    for root in roots:
        for path in sorted(Path(root).glob("group_*/inputs.json")):
            inp = json.loads(path.read_text())
            split = inp["declared_split"]
            config = inp["split_group"]
            if config in configurations and configurations[config] != split:
                raise ValueError("configuration crosses training/validation/test boundaries")
            configurations[config] = split
            if split not in splits:
                continue  # Never open held-out outcomes while training.
            directory = path.parent
            if not (directory/"complete.json").is_file():
                raise ValueError(f"incomplete requested group {directory}")
            gid = inp["group_id"]
            if gid in seen:
                raise ValueError("duplicate group")
            seen.add(gid)
            complete = json.loads((directory/"complete.json").read_text())
            request=by_directory[str(directory)]["request"]
            collected_source=json.loads((directory/"source.json").read_text())
            if (complete["request_sha256"]!=digest(request) or
                    complete["source_sha256"]!=digest(collected_source) or
                    inp["source_sha256"]!=digest(collected_source)):
                raise ValueError("request/source/input/completion binding mismatch")
            if inp["task"]["seed"]!=request["seed"] or split!=request["split"]:
                raise ValueError("requested configuration differs from collected inputs")
            outcomes = json.loads((directory/"outcomes.json").read_text())
            ih = digest(inp)
            if complete["input_sha256"] != ih or outcomes["input_sha256"] != ih:
                raise ValueError("input/outcome binding mismatch")
            plans = [PlanIR.from_dict(p) for p in inp["candidates"]]
            if len({p.id for p in plans}) != len(plans):
                raise ValueError("duplicate candidate identity")
            if len(plans)!=request["n"] or complete["candidates"]!=len(plans):
                raise ValueError("collected pool differs from requested candidate budget")
            graphs = [compile_graph(inp["observation"], p) for p in plans]
            saved_graphs = json.loads((directory/"skill_graphs.json").read_text())
            if saved_graphs["input_sha256"] != ih or digest(saved_graphs["graphs"]) != digest(graphs):
                raise ValueError("pre-execution interface/graph drift")
            records = {p.id: {} for p in plans}
            disturbances = {}
            for trial in outcomes["trials"]:
                cid, repeat = trial["candidate_id"], trial["trial"]["repeat"]
                if cid not in records or repeat in records[cid]:
                    raise ValueError("unexpected or repeated trial")
                if trial.get("valid") is not True or type(trial["success"]) is not bool or trial.get("timeout"):
                    raise ValueError("invalid/censored trial is not a binary physical outcome")
                th = digest(trial["trial"])
                if trial["trial_sha256"] != th or (repeat in disturbances and disturbances[repeat] != th):
                    raise ValueError("unpaired perturbations")
                disturbances[repeat] = th
                records[cid][repeat] = trial
            if not disturbances or sum(map(len, records.values())) != complete["trials"]:
                raise ValueError("missing trials")
            if set(disturbances)!=set(range(request["repeats"])):
                raise ValueError("collected repeat budget differs from request")
            labels = []
            nominal = inp.get("nominal_index", 0)
            for p, graph in zip(plans, graphs):
                rs = records[p.id]
                if set(rs) != set(disturbances) or nominal not in rs:
                    raise ValueError("candidate trial coverage differs")
                if any(r["input_graph_sha256"] != digest(graph) for r in rs.values()):
                    raise ValueError("scored program differs from physical program")
                if rs[nominal]["trial"].get("domain") != "nominal":
                    raise ValueError("nominal training target must be explicitly identified")
                if rs[nominal]["trial"]["friction_scale"]!=1. or rs[nominal]["trial"]["actuator_gain_scale"]!=1.:
                    raise ValueError("nominal outcome cannot contain undisclosed physical noise")
                labels.append(float(rs[nominal]["success"]))
            groups.append(dict(id=gid, config=config, split=split, seed=inp["task"]["seed"],
                path=str(directory), inputs=inp, plans=plans, graphs=graphs,
                records=[program_record(g) for g in graphs], y=np.asarray(labels, np.float32),
                outcomes=[[int(records[p.id][r]["success"]) for r in sorted(disturbances)] for p in plans],
                nominal_index=sorted(disturbances).index(nominal), input_sha256=ih,
                collected_source_sha256=digest(collected_source)))
    expected={r["directory"] for r in dispositions if r["status"]=="completed"}
    if {g["path"] for g in groups}!=expected:
        raise ValueError("requested completed groups differ from loaded declared split")
    if not groups:
        raise ValueError("no complete groups")
    return groups


class ProgramMLP(nn.Module):
    def __init__(self, dim, kind="mlp"):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))
        if kind == "linear":
            self.net = nn.Linear(dim, 1)
        elif kind == "mlp":
            self.net = nn.Sequential(nn.Linear(dim, 64), nn.SiLU(), nn.Dropout(.1),
                                     nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1))
        else:
            raise ValueError(kind)

    def forward(self, x):
        return self.net((x-self.mean)/self.std).squeeze(-1)


@torch.inference_mode()
def infer(model, x, device):
    model.eval()
    return model(torch.as_tensor(x, device=device)).cpu().numpy().astype(float)


def sigmoid(z):
    return 1/(1+np.exp(-np.clip(z, -50, 50)))


def validation_summary(groups, zs):
    rows = []
    for g, z in zip(groups, zs):
        y = g["y"];p = sigmoid(z);rank = np.argsort(-z, kind="stable")
        rows.append(dict(group_id=g["id"], config_id=g["config"], candidate_ids=[p.id for p in g["plans"]],
            scores=p.tolist(), ranking_scores=z.tolist(), outcomes=g["outcomes"],
            nominal_index=g["nominal_index"], label_regime="nominal" if len(g["outcomes"][0]) == 1 else "repeated",
            brier=float(np.mean((p-y)**2)), log_loss=float(np.mean(np.logaddexp(0,z)-y*z)),
            accuracy_at_half=float(np.mean((z>=0)==y)),
            feasible_hit4=bool(y[rank[:4]].max())))
    return dict(brier=float(np.mean([r["brier"] for r in rows])),
                log_loss=float(np.mean([r["log_loss"] for r in rows])), rows=rows)


def train(args):
    start = time.perf_counter()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    groups = load_groups(args.data)
    tr = [g for g in groups if g["split"] == "train"]
    va = [g for g in groups if g["split"] == "val"]
    if not tr or not va:
        raise ValueError("training and validation configurations are required")
    encoder = InputSchema.fit([r for g in tr for r in g["records"]])
    for g in groups:
        g["x"], g["unseen"] = encoder.transform(g["records"])
    x = np.concatenate([g["x"] for g in tr]);y = np.concatenate([g["y"] for g in tr])
    model = ProgramMLP(x.shape[1], args.kind).to(args.device)
    xt = torch.as_tensor(x, device=args.device);yt = torch.as_tensor(y, device=args.device)
    model.mean.copy_(xt.mean(0));model.std.copy_(xt.std(0, unbiased=False).clamp_min(1e-6))
    # Each configuration has equal total training contribution, including when
    # necessary planning checks leave different numbers of unique candidates.
    weights = torch.as_tensor(np.concatenate([np.full(len(g["y"]), 1/len(g["y"])) for g in tr]),
                              dtype=torch.float32, device=args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    out = Path(args.out);out.mkdir(parents=True, exist_ok=True)
    source = source_manifest()
    if {g["collected_source_sha256"] for g in groups}!={digest(source)}:
        raise ValueError("training requires the frozen collection source; mixed/edited source needs a separate documented run")
    dispositions=request_dispositions(args.data)
    split = {s:[dict(group_id=g["id"],config_id=g["config"],input_sha256=g["input_sha256"]) for g in groups if g["split"]==s]
             for s in ("train","val")}
    saved = dict(schema=SCHEMA, kind=args.kind, seed=args.seed, input_schema=encoder.saved,
        encoding_sha256=encoding_hash(), interface_sha256=interface_hash(),
        source_sha256=digest(source), split_sha256=digest(split), training=vars(args),
        parameters=sum(p.numel() for p in model.parameters()), target="direct_nominal_full_program_success")
    dump(out/"source.json",source);dump(out/"split.json",split);dump(out/"input_schema.json",encoder.saved)
    dump(out/"request_dispositions.json",dispositions)
    best = float("inf");history=[]
    for epoch in range(args.epochs):
        model.train();order=np.random.permutation(len(x));losses=[]
        for begin in range(0,len(order),args.batch_size):
            ix=order[begin:begin+args.batch_size]
            optimizer.zero_grad()
            z=model(xt[ix]);loss=(nn.functional.binary_cross_entropy_with_logits(z,yt[ix],reduction="none")*weights[ix]).mean()
            loss=loss*len(x)/len(tr)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            losses.append(float(loss.detach()))
        val=validation_summary(va,[infer(model,g["x"],args.device) for g in va])
        row=dict(epoch=epoch+1,loss=float(np.mean(losses)),validation_brier=val["brier"],validation_log_loss=val["log_loss"])
        history.append(row)
        if val["brier"]<best:
            best=val["brier"]
            torch.save(saved|dict(state_dict=model.state_dict(),epoch=epoch+1,validation=val),out/"best.pt")
        if epoch%20==0:print(json.dumps(row),flush=True)
    dump(out/"history.json",history)
    chosen=torch.load(out/"best.pt",map_location="cpu",weights_only=False)
    summary={k:v for k,v in chosen.items() if k not in ("state_dict","input_schema")}
    summary.update(input_dim=len(encoder.keys),raw_dim=encoder.saved["raw_dim"],train_configurations=len(tr),
        validation_configurations=len(va),training_candidates=len(x),training_positive_candidates=int(y.sum()),
        requested_configurations=len(dispositions),failed_before_inputs=sum(r["status"]=="failed_before_inputs" for r in dispositions),
        wall_seconds=time.perf_counter()-start,checkpoint_sha256=file_sha(out/"best.pt"))
    dump(out/"summary.json",summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ("validation","training")}),flush=True)


class ValueScorer:
    """Public API: rank(current_state, candidate_plans, k). No label access."""
    def __init__(self, checkpoint, device="cpu"):
        start=time.perf_counter();self.device=device
        self.saved=torch.load(checkpoint,map_location=device,weights_only=False)
        if self.saved.get("schema") != SCHEMA or self.saved["encoding_sha256"]!=encoding_hash() or self.saved["interface_sha256"]!=interface_hash():
            raise ValueError("trained value input differs from current executable interface")
        self.encoder=InputSchema(self.saved["input_schema"])
        self.model=ProgramMLP(len(self.encoder.keys),self.saved["kind"]).to(device)
        self.model.load_state_dict(self.saved["state_dict"]);self.model.eval()
        self.checkpoint_sha256=file_sha(checkpoint);self.sync()
        self.load_seconds=time.perf_counter()-start

    def sync(self):
        if str(self.device).startswith("cuda"):torch.cuda.synchronize(self.device)

    def rank(self,state,plans,k=4):
        start=time.perf_counter()
        plans=[p if isinstance(p,PlanIR) else PlanIR.from_dict(p) for p in plans]
        if not plans or not 1<=k<=len(plans) or len({p.id for p in plans})!=len(plans):
            raise ValueError("invalid or duplicate candidate pool")
        graphs=[compile_graph(state,p) for p in plans]
        construction=time.perf_counter()-start
        t=time.perf_counter();records=[program_record(g) for g in graphs]
        x,unknown,diagnostics=self.encoder.transform(records,return_diagnostics=True);encoding=time.perf_counter()-t
        self.sync();t=time.perf_counter();z=infer(self.model,x,self.device);self.sync()
        inference=time.perf_counter()-t;t=time.perf_counter()
        p=sigmoid(z);order=np.argsort(-z,kind="stable")
        result=dict(schema="twingraph.topk.v5",checkpoint_sha256=self.checkpoint_sha256,
            input_schema_sha256=digest(self.encoder.saved),scores=p.tolist(),logits=z.tolist(),
            order=[plans[i].id for i in order],
            top_k=[dict(candidate_id=plans[i].id,score=float(p[i]),graph_sha256=digest(graphs[i]),
                        graph=graphs[i],plan=plans[i].to_dict()) for i in order[:k]],
            unknown_input_fields=[len(u) for u in unknown],
            input_diagnostics=diagnostics,
            seconds=dict(graph_construction=construction,encoding=encoding,inference=inference))
        result["seconds"]["export"]=time.perf_counter()-t
        result["seconds"]["total"]=time.perf_counter()-start
        return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",nargs="+",required=True);p.add_argument("--out",required=True)
    p.add_argument("--kind",choices=("mlp","linear"),default="mlp")
    p.add_argument("--seed",type=int,default=17);p.add_argument("--epochs",type=int,default=120)
    p.add_argument("--batch-size",type=int,default=48);p.add_argument("--lr",type=float,default=.001)
    p.add_argument("--device",default="cuda")
    train(p.parse_args())


if __name__=="__main__":main()
