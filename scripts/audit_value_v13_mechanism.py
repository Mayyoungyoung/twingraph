"""Audit whether a fitted prefix ranker reproduces observed scenario reversals."""
import argparse
import json
from pathlib import Path

from scripts.train_value_v13_mechanism import load_rows
from scripts.train_value_v12 import dump
from simbench.value.graph_value_v12 import ValueRankerV12


def audit(roots, checkpoint, gate_path, out, device="cpu"):
    gate=json.loads(Path(gate_path).read_text())
    seeds=[int(c["seed"]) for c in gate["cases"]]
    rows,_,runtime=load_rows(roots,seeds,gate["stop_after"])
    ranker=ValueRankerV12(checkpoint,device);scores=ranker.rank([r["graph"] for r in rows])
    table={(r["seed"],r["name"]):float(score) for r,score in zip(rows,scores)}
    checks=[]
    for crossing in gate["crossovers"]:
        a,b=crossing["candidate_a"],crossing["candidate_b"]
        directions=[]
        for expected,seeds_for_direction in ((a,crossing["a_only_success_seeds"]),(b,crossing["b_only_success_seeds"])):
            for seed in seeds_for_direction:
                sa,sb=table[(seed,a)],table[(seed,b)]
                directions.append(dict(seed=seed,expected_above=expected,score_a=sa,score_b=sb,
                    correct=(sa>sb if expected==a else sb>sa)))
        checks.append(dict(candidate_a=a,candidate_b=b,directions=directions,
            directional_accuracy=sum(d["correct"] for d in directions)/len(directions),
            both_directions_learned=all(d["correct"] for d in directions)))
    report=dict(schema="twingraph.mechanism_value_reversal_audit.v13.v1",
        checkpoint_sha256=ranker.sha256,physical_runtime_sha256=runtime,
        crossover_pairs=len(checks),fully_learned_pairs=sum(c["both_directions_learned"] for c in checks),
        directional_comparisons=sum(len(c["directions"]) for c in checks),
        correct_directional_comparisons=sum(d["correct"] for c in checks for d in c["directions"]),
        all_observed_reversals_learned=bool(checks and all(c["both_directions_learned"] for c in checks)),
        checks=checks,limitations="development-prefix audit; not a held-out full-task or hardware claim")
    dump(out,report);return report


def main():
    p=argparse.ArgumentParser();p.add_argument("--roots",nargs="+",required=True)
    p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--gate",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True);p.add_argument("--device",default="cpu");a=p.parse_args()
    print(json.dumps(audit(a.roots,a.checkpoint,a.gate,a.out,a.device),ensure_ascii=False,indent=2))


if __name__=="__main__":main()
