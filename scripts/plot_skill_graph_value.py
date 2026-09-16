"""Scientific summary figure from the frozen v3 measurements."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument("--evidence",required=True);a=p.parse_args()
    d=Path(a.evidence);results=json.loads((d/"test_metrics.json").read_text())["results"]
    timing=json.loads((d/"module_timing.json").read_text())["summary"]
    selected=json.loads((d/"frozen.json").read_text())["selected"]
    labels=["Geometry","MLP87 + proxy","Graph ports\n+ MLP","Ports\n+ Transformer","Relations\n+ Transformer"]
    families=[["geometry"],["mlp_17"],[f"port_mlp_{s}" for s in (17,29,43)],
              [f"sequence_{s}" for s in (17,29,43)],[f"graph_{s}" for s in (17,29,43)]]
    plt.rcParams.update({"font.size":10,"svg.fonttype":"none"})
    fig,axes=plt.subplots(1,2,figsize=(13,4.9));x=np.arange(len(labels));width=.34
    for offset,key,title,color in ((-.17,"hit1","Hit@1","#245e9d"),(.17,"quality4","Mean Top-4 empirical success","#3b967c")):
        values=[[results[n][key]*100 for n in names] for names in families]
        means=np.asarray([np.mean(v) for v in values]);low=means-[min(v) for v in values];high=[max(v) for v in values]-means
        axes[0].bar(x+offset,means,width,label=title,color=color,yerr=[low,high],capsize=3)
    axes[0].set_xticks(x,labels);axes[0].set_ylim(0,108);axes[0].set_ylabel("Problem-group mean (%)")
    axes[0].set_title("Ranking quality | 12 fresh configurations")
    axes[0].legend(loc="lower left",fontsize=8);axes[0].grid(axis="y",alpha=.15);axes[0].set_axisbelow(True)
    names=["geometry","mlp_17",selected,"sequence_17","graph_17"]
    bottom=np.zeros(5)
    phases=[("Shared graph build + integrity",["graph_construction","graph_integrity"],"#adb4be"),
            ("Optional geometry",["optional_geometry"],"#d99a39"),("Input encoding",["typed_encoding"],"#6d9bd2"),
            ("Network",["network"],"#876bac"),("Sort / export",["sort_export"],"#455c74")]
    for label,keys,color in phases:
        values=np.asarray([sum(timing[n][k]["median_seconds"] for k in keys)*1000 for n in names])
        axes[1].bar(x,values,.65,bottom=bottom,label=label,color=color);bottom+=values
    axes[1].set_xticks(x,labels);axes[1].set_ylabel("Milliseconds per N=16 candidate pool")
    axes[1].set_title("Value-module cost | resident RTX 2080 Ti")
    axes[1].legend(fontsize=8,loc="upper right");axes[1].set_ylim(0,bottom.max()*1.38)
    fig.text(.5,.01,"19-call pin assembly suffix; 2 reference rollouts/candidate. Whiskers: seed range (3 runs). No real robot. Phase medians are illustrative; exact totals/P95 in JSON.",ha="center",fontsize=8)
    fig.tight_layout(rect=[0,.05,1,1]);fig.savefig(d/"module_summary.png",dpi=180);fig.savefig(d/"module_summary.svg");plt.close(fig)


if __name__=="__main__":main()
