"""Run the complete value/TopK/twin/independent-target assembly pipeline."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planner", required=True, help="Task and honest LLM/API/proxy provenance JSON")
    parser.add_argument("--checkpoint", help="Trained v5 value scorer; required for top_k/pair")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--method", choices=("top_k", "full", "pair"), default="pair")
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--validation-repeats", type=int, default=2)
    parser.add_argument("--target-repeats", type=int, default=1)
    parser.add_argument("--accept-rate", type=float, default=.5)
    parser.add_argument("--timeout", type=float, default=240.)
    parser.add_argument("--friction-span", type=float, default=.03,
                        help="Declared uncertainty half-width about nominal 1; scenario parameter, not a score weight")
    parser.add_argument("--gain-span", type=float, default=.005)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    if args.method != "full" and not args.checkpoint:
        parser.error("--checkpoint is required for top_k/pair")
    from simbench.value.system_v5 import SystemConfig, run_pair, run_system
    scorer = None
    if args.checkpoint:
        from simbench.value.value_v5 import ValueScorer
        scorer = ValueScorer(args.checkpoint, args.device)
    record = json.loads(Path(args.planner).read_text(encoding="utf-8"))
    config = SystemConfig(seed=args.seed, n=args.n, k=args.k,
        validation_repeats=args.validation_repeats, target_repeats=args.target_repeats,
        accept_rate=args.accept_rate, timeout=args.timeout, friction_span=args.friction_span,
        gain_span=args.gain_span, render=not args.no_render)
    if args.method == "pair":
        result = run_pair(config, record, scorer, args.out)
    else:
        result = run_system(config, record, scorer, args.out, args.method)
    print(json.dumps(dict(output=str(Path(args.out).resolve()),
                          status=result.get("status", "paired_actual_runs_complete"),
                          target_successes=result.get("target_successes"),
                          seconds=result.get("seconds", result.get("decision_seconds"))), ensure_ascii=False))


if __name__ == "__main__":
    main()
