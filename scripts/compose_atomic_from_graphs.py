#!/usr/bin/env python3
"""Let Codex compose atomic port alternatives from pre-execution graphs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simbench.value.codex_planner import generate_port_compositions_with_codex
from simbench.value.plan import digest


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def options_from_graphs(graphs, selected_ports=()):
    plans = [graph["assembly"]["plan"] for graph in graphs]
    base = plans[0]["calls"]
    selected = set(selected_ports)
    options = {}
    for plan in plans[1:]:
        calls = plan["calls"]
        if [(c["id"], c["skill"], c["roles"]) for c in calls] != [
            (c["id"], c["skill"], c["roles"]) for c in base]:
            raise ValueError("alternative graphs must share the same atomic call skeleton")
        for first, other in zip(base, calls):
            for argument, entry in first["arguments"].items():
                key = f"{first['id']}.{argument}"
                if selected and key not in selected:
                    continue
                alternative = other["arguments"].get(argument)
                if (alternative is None or entry["status"] != "known"
                        or alternative["status"] != "known"
                        or digest(entry["value"]) == digest(alternative["value"])):
                    continue
                values = options.setdefault(key, [entry["value"]])
                if digest(alternative["value"]) not in {digest(v) for v in values}:
                    values.append(alternative["value"])
    if selected and set(options) != selected:
        raise ValueError(f"selected ports lack observed alternatives: {sorted(selected-set(options))}")
    return options


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--graphs", type=Path, nargs="+", required=True,
                        help="First graph is the template; others provide solver-grounded port values")
    parser.add_argument("--ports", nargs="*", default=())
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    graphs = [read(path) for path in args.graphs]
    options = options_from_graphs(graphs, args.ports)
    task = read(args.task)
    template = graphs[0]["assembly"]["plan"]
    observation = graphs[0]["assembly"]["observation"]
    _, candidates = generate_port_compositions_with_codex(
        task, observation, template, options, candidate_count=args.n, out=args.out)
    print(json.dumps(dict(status="compiled", candidates=len(candidates),
                          options=list(options), names=[c["name"] for c in candidates]),
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
