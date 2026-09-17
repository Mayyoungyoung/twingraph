"""Run paired robust-v6 Top-K and exhaustive digital-twin screening."""
import argparse
import json
from pathlib import Path

from simbench.value.collect import dump
from simbench.value.physical_v6 import RobustPhysicalRunner
from simbench.value.system_v5 import SystemConfig, run_pair
from simbench.value.value_v6 import ValueScorer
from simbench.value import stage_v6


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--planner", default="experiments/value_v5/planner_record.json")
    p.add_argument("--out", required=True); p.add_argument("--seed", type=int, required=True)
    p.add_argument("--n", type=int, default=12); p.add_argument("--k", type=int, default=4)
    p.add_argument("--validation-repeats", type=int, default=3)
    p.add_argument("--target-repeats", type=int, default=3)
    p.add_argument("--timeout", type=float, default=240.)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-render", action="store_true")
    a = p.parse_args()
    planner = json.loads(Path(a.planner).read_text())
    scorer = ValueScorer(a.checkpoint, a.device)
    config = SystemConfig(seed=a.seed, n=a.n, k=a.k,
        validation_repeats=a.validation_repeats, target_repeats=a.target_repeats,
        timeout=a.timeout, friction_span=.08, gain_span=.015, render=not a.no_render)
    result = run_pair(config, planner, scorer, a.out, stage=stage_v6,
                      runner_factory=RobustPhysicalRunner)
    dump(Path(a.out) / "v6_pair_summary.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
