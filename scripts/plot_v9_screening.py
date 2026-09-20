"""Standalone success-versus-verification-cost figure for grouped validation."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    data = json.loads(Path(args.summary).read_text())
    rows = data["validation"]
    methods = sorted({row["method"] for row in rows})
    fig, ax = plt.subplots(figsize=(8.3, 5.2), dpi=160)
    for method in methods:
        subset = sorted((row for row in rows if row["method"] == method), key=lambda row: row["k"])
        x = [row["mean_verification_wall_seconds"] for row in subset]
        y = [row["success_rate"] for row in subset]
        ax.plot(x, y, marker="o", linewidth=1.6, label=method)
        for row, xx, yy in zip(subset, x, y):
            ax.annotate(str(row["k"]), (xx, yy), xytext=(3, 4),
                        textcoords="offset points", fontsize=7)
    ax.set(xlabel="Mean digital-twin verification wall time (s)",
           ylabel="Complete-task success probability",
           title="V9 grouped validation: success versus verification cost")
    ax.set_ylim(-.03, 1.03)
    ax.grid(alpha=.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(out)


if __name__ == "__main__":
    main()
