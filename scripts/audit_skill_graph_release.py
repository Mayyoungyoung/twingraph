"""Verify packaged data, checkpoints, source snapshots and split isolation (stdlib only)."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tarfile


def sha(data):return hashlib.sha256(data).hexdigest()
def digest(row):return sha(json.dumps(row,sort_keys=True,allow_nan=False).encode())
def read(p):return json.loads(p.read_text(encoding="utf-8"))


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",default=str(Path(__file__).resolve().parents[1]));a=p.parse_args()
    root=Path(a.root);data=root/"datasets/value";manifest=read(data/"manifest_skill_graph_v3.json")
    for name,row in manifest["archives"].items():
        assert sha((data/name).read_bytes())==row["sha256"],name
    model_manifest=read(root/"models/value/v3/MANIFEST.json")
    for row in model_manifest["models"]:
        assert sha((root/row["file"]).read_bytes())==row["sha256"],row["name"]
    assert sha((root/"models/value/v3/best_graph_input.pt").read_bytes())==model_manifest["best_sha256"]
    inputs={};trial_count=0;successes=0;graphs_checked=0
    with tarfile.open(data/"skill_graph_v3.tar.gz") as tar:
        get=lambda name:json.load(tar.extractfile(name))
        for row in manifest["groups"]:
            base=f"group_sliding_stage_pin_{row['seed']}_0/"
            inp=get(base+"inputs.json");out=get(base+"outcomes.json");gr=get(base+"skill_graphs.json")
            ih=digest(inp);assert ih==row["input_sha256"]==out["input_sha256"]==gr["input_sha256"]
            inputs[inp["group_id"]]=inp
            mapping={g["plan"]["id"]:digest(g) for g in gr["graphs"]};assert len(mapping)==16
            assert len({p["id"] for p in inp["candidates"]})==16
            pairs=Counter()
            for trial in out["trials"]:
                assert trial["valid"] is True and trial["input_graph_sha256"]==mapping[trial["candidate_id"]]
                assert bool(trial["prefix_success"] and trial["suffix_success"])==trial["success"]
                assert digest(trial["trial"])==trial["trial_sha256"]
                pairs[(trial["candidate_id"],trial["trial"]["repeat"])]+=1
            assert len(pairs)==32 and set(pairs.values())=={1}
            trial_count+=len(out["trials"]);successes+=sum(t["success"] for t in out["trials"]);graphs_checked+=len(mapping)
    with tarfile.open(data/"two_family_v2.tar.gz") as tar:
        for member in tar:
            if member.name.startswith("group_sliding_stage_pin_") and member.name.endswith("/inputs.json"):
                inp=json.load(tar.extractfile(member));inputs[inp["group_id"]]=inp
    for row in model_manifest["models"]:
        split=read(root/Path(row["file"]).parent/"split.json")
        assert digest(split)==row["split_sha256"]
        ids=set()
        for name in ("train","val"):
            for sample in split[name]:
                assert sample["id"] not in ids;ids.add(sample["id"])
                inp=inputs[sample["id"]]
                assert inp["declared_split"]==name and digest(inp)==sample["input_sha256"]
    for label,expected in (("training",{r["source_sha256"] for r in model_manifest["models"]}),
                           ("collection",{r["source_sha256"] for r in manifest["groups"]})):
        with tarfile.open(data/f"skill_graph_v3_{label}_source.tar.gz") as tar:
            files={m.name:tar.extractfile(m).read() for m in tar if m.isfile()}
        if label=="training":result=digest({k:sha(v) for k,v in files.items() if not k.startswith("simbench/core/")})
        else:
            included={"research_scenarios.py","research_collect.py","physical.py","plan.py","scenarios.py","collect.py","skill_graph.py"};h=hashlib.sha256()
            for k,v in sorted(files.items()):
                if not k.startswith("simbench/value/") or Path(k).name in included:
                    h.update(k.removeprefix("simbench/").encode());h.update(v)
            result=h.hexdigest()
        assert expected=={result},label
    top=read(root/"docs/evidence/value_v3/scoring_top_k.json")
    assert top["n"]==16 and top["k"]==4 and top["checkpoint_sha256"]==model_manifest["best_sha256"]
    for row in top["top_k"]:
        assert row["candidate_id"]==row["plan"]["id"]==row["graph"]["plan"]["id"]
        assert digest(row["graph"])==row["input_graph_sha256"] and row["plan"]==row["graph"]["plan"]
    assert trial_count==manifest["new_label_rollouts"] and successes==manifest["new_full_successes"]
    result=dict(status="passed",fresh_groups=32,physical_labels=trial_count,full_successes=successes,
                executed_graphs_checked=graphs_checked,checkpoints_checked=len(model_manifest["models"]),
                source_snapshots_verified=["collection","training"],split_isolation="passed",complete_top_k=4,
                archives_checked=len(manifest["archives"]))
    (root/"docs/evidence/value_v3/release_audit.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result))


if __name__=="__main__":main()
