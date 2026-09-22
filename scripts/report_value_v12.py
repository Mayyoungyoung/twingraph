"""Compact retrospective evidence: paired layout uncertainty and model audit."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    args = p.parse_args()
    metrics = json.loads((args.run/"metrics.json").read_text())
    manifest = json.loads((args.run/"data_manifest.json").read_text())
    selection = json.loads((args.run/"selection.json").read_text())
    rows = manifest["rows"]
    mixed = {}
    for split, layouts in manifest["splits"].items():
        groups = [[r for r in rows if r["layout"] == seed] for seed in layouts]
        mixed[split] = dict(layouts=len(groups), mixed_layouts=sum(len({r["full_success"] for r in rs}) > 1 for rs in groups),
            supervised_ordered_pairs=sum(sum(a["full_success"] > b["full_success"]+.05 for a in rs for b in rs) for rs in groups))
    comparisons = {}
    for model, report in metrics.items():
        cases = report["historical_holdout"]["cases"]
        layouts = sorted({r["layout"] for r in cases})
        paired = np.asarray([np.mean([float(r["success"])-r["random"]["success"] for r in cases if r["layout"] == seed]) for seed in layouts])
        rng = np.random.default_rng(1221)
        boot = paired[rng.integers(len(paired), size=(10000, len(paired)))].mean(1)
        comparisons[model] = dict(paired_success_difference=float(paired.mean()),
            layout_bootstrap_95_percent_interval=np.quantile(boot, [.025, .975]).tolist(),
            layout_count=len(layouts), selected_by_validation_only=model == selection["selected"],
            interpretation="Retrospective old-geometry development; interval treats layouts, not physics repeats, as independent.")
    output = dict(primary_k=4, selected=selection["selected"], mixed_supervision=mixed,
        comparison=comparisons, training_physical_trial_count=len(manifest["records"]),
        model_files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.run.glob("*.pt")},
        restriction="No printed-kit generalization or prospective heldout accuracy claim is supported by this replay.")
    (args.run/"evidence_summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
