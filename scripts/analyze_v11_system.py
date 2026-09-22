"""Audit paired inputs and summarize actual online V11 runs."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from math import comb
from pathlib import Path

import numpy as np


def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    summaries=[]; checks=[]; candidate_rows=[]; cases=[]
    for case in sorted(a.root.glob("seed_*")):
        main_methods=("all_twin","random_top_k","value_top_k")
        methods=[case / m / "summary.json" for m in main_methods]
        if not all(f.exists() for f in methods): raise ValueError(f"incomplete primary case {case}")
        pools=[];observations=[]
        for file in methods:
            result=read(file);request=read(file.parent / "request.json")
            pools.append(json.dumps(request["pool"],sort_keys=True))
            observations.append(request["source"]["observation_sha256"])
            # Each verification record must have a distinct physical run and
            # agree with its pre-execution request and summary.
            for trial in result["trials"]:
                detail=read(file.parent / "twins" / trial["name"] / "result.json")
                assert detail["proposal"]==request["pool"][trial["index"]]
                assert detail["success"]==trial["success"]
                assert detail["domain"]=="online"
            summaries.append(result)
        assert len(set(pools))==len(set(observations))==1
        checks.append(dict(case=case.name,identical_candidates=True,identical_initial_rgbd=True,methods=3))
        reference=read(case / "all_twin" / "summary.json")
        request=read(case / "all_twin" / "request.json")
        ns=sum(t["success"] for t in reference["trials"]);n=len(reference["trials"])
        random_hit=1-(comb(n-ns,4)/comb(n,4) if n-ns>=4 else 0)
        cases.append(dict(seed=reference["seed"],successful_candidates=ns,pool_size=n,
                          exact_random_top4_hit=random_hit))
        physical_prefixes=set()
        for trial in reference["trials"]:
            d=read(case / "all_twin" / "twins" / trial["name"] / "result.json")
            steps=d["executed_parameters"] or []
            failed=next((r for r in steps if not r["ok"]),None)
            physical_prefixes.add(hashlib.sha256(json.dumps(
                [{"skill":s["skill"],"params":s["params"]} for s in steps],sort_keys=True).encode()).hexdigest())
            candidate_rows.append(dict(seed=reference["seed"],name=trial["name"],
                success=trial["success"],wall_seconds=trial["wall_seconds"],sim_seconds=d["sim_seconds"],
                completed_stages=",".join(r["stage"] for r in d["boundaries"]),
                first_failed_skill=failed["skill"] if failed else "none",
                terminal_failure=d["error"].split(":",1)[0] if d["error"] else "none",
                error=d["error"]))
        cases[-1]["distinct_executed_command_traces"]=len(physical_prefixes)
    groups=defaultdict(list)
    for row in summaries: groups[row["method"]].append(row)
    aggregate=[]
    for method,rows in groups.items():
        aggregate.append(dict(method=method,cases=len(rows),
            twin_successes=sum(r["twin_success"] for r in rows),
            execution_successes=sum(r["execution_success"] for r in rows),
            mean_twin_calls=float(np.mean([len(r["trials"]) for r in rows])),
            mean_decision_seconds=float(np.mean([r["decision_seconds"] for r in rows])),
            mean_execution_seconds=float(np.mean([r["execution_seconds"] for r in rows])),
            mean_total_seconds=float(np.mean([r["total_wall_seconds"] for r in rows])),
            replan_events=sum(sum(e["action"].startswith("replan_verified") for e in r["events"]) for r in rows)))
    for name,rows in (("methods",summaries),("candidates",candidate_rows)):
        fields=[k for k in rows[0] if k not in ("trials","events")]
        with (a.out / f"{name}.csv").open("w",newline="",encoding="utf-8-sig") as f:
            w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(rows)
    result=dict(aggregate=aggregate,paired_input_checks=checks,cases=cases,
                failures=dict(Counter(r["terminal_failure"] for r in candidate_rows if not r["success"])),
                interpretation="Three-layout system smoke; not a noninferiority study or real-robot validation")
    (a.out / "summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    plot(aggregate,candidate_rows,a.out)
    if (a.out / "budget_sweep.json").exists():
        plot_budget(read(a.out / "budget_sweep.json")["aggregate"],a.out)
    print(json.dumps(result,indent=2))


def plot(aggregate,candidates,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels={"all_twin":"Full twin (12)","random_top_k":"Random top-4", "value_top_k":"Value top-4", "random_early_stop":"Random, stop on success"}
    colors={"all_twin":"#64748b","random_top_k":"#d59737","value_top_k":"#237ba5","random_early_stop":"#9566a9"}
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout="constrained")
    for i,r in enumerate(aggregate):
        axes[0].barh(i,r["mean_decision_seconds"],color=colors[r["method"]])
        axes[0].text(r["mean_decision_seconds"]+3,i,f'{r["mean_decision_seconds"]:.1f}s',va="center",fontsize=9)
        axes[1].bar(i-.17,r["twin_successes"],width=.32,label="Twin feasible" if i==0 else None,color="#97b6c9")
        axes[1].bar(i+.17,r["execution_successes"],width=.32,label="Independent execution" if i==0 else None,color="#214e69")
    axes[0].set_yticks(range(len(aggregate)),[labels[r["method"]] for r in aggregate]);axes[0].set_xlabel("Measured decision time (seconds)")
    axes[0].set_xlim(right=max(r["mean_decision_seconds"] for r in aggregate)*1.22)
    axes[1].set_xticks(range(len(aggregate)),[labels[r["method"]] for r in aggregate],rotation=25,ha="right")
    axes[1].set_ylim(0,3.5);axes[1].set_yticks([0,1,2,3]);axes[1].set_ylabel("Successful cases / 3");axes[1].legend()
    fig.suptitle("TwinGraph V11 — measured system smoke, independent perturbed simulation")
    fig.savefig(out / "system_comparison.png",dpi=180);plt.close(fig)
    seeds=sorted({r["seed"] for r in candidates});names=list(dict.fromkeys(r["name"] for r in candidates))
    matrix=np.array([[int(next(r["success"] for r in candidates if r["seed"]==s and r["name"]==n)) for n in names] for s in seeds])
    fig,ax=plt.subplots(figsize=(12,3.6),layout="constrained")
    ax.imshow(matrix,cmap=matplotlib.colors.ListedColormap(["#be645d","#4e9b82"]),vmin=0,vmax=1,aspect="auto")
    for i,s in enumerate(seeds):
        for j,n in enumerate(names): ax.text(j,i,"PASS" if matrix[i,j] else "FAIL",ha="center",va="center",color="white",fontsize=8)
    ax.set_xticks(range(len(names)),names,rotation=35,ha="right",fontsize=8);ax.set_yticks(range(len(seeds)),seeds)
    ax.set_title("All 12 candidates physically verified in every held-out layout")
    fig.savefig(out / "candidate_matrix.png",dpi=180);plt.close(fig)


def plot_budget(rows,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout="constrained")
    for method,label,color in (("frozen_value","Frozen value","#237ba5"),("exact_random","Exact random expectation","#d59737")):
        group=[r for r in rows if r["method"]==method]
        axes[0].plot([r["k"] for r in group],[100*r["success_rate"] for r in group],"o-",label=label,color=color)
        axes[1].plot([r["mean_full_system_wall_seconds"] for r in group],[100*r["success_rate"] for r in group],"o-",label=label,color=color)
    for ax in axes:
        ax.axhline(100/3,color="#64748b",linestyle="--",label="Full-pool coverage (1/3)")
        ax.set_ylabel("Retained feasible layouts (%)");ax.grid(alpha=.2)
    axes[0].set_xlabel("Maximum candidates verified (K)");axes[0].set_xticks([1,2,4,8,12]);axes[0].legend(fontsize=8)
    axes[1].set_xlabel("Estimated decision time from recorded rollouts (s)")
    fig.suptitle("Offline budget replay — not new online runs or deployment success")
    fig.savefig(Path(out) / "budget_sweep.png",dpi=180);plt.close(fig)


if __name__=="__main__": main()
