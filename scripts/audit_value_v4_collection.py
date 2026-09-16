"""Require every requested configuration to have a complete, honest disposition."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_value_v4 import admissible_setup_error
from simbench.value.collect import dump
from simbench.value.schema_learning import load_groups


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True); p.add_argument("--seed", type=int, required=True)
    p.add_argument("--groups", type=int, required=True); p.add_argument("--split", required=True)
    a = p.parse_args(); rows = []
    for seed in range(a.seed, a.seed+a.groups):
        cp = 2 + seed % 2
        d = Path(a.data)/f"group_sliding_stage_assembly_{seed}_{cp}"
        if (d/"complete.json").exists() and not (d/"failure.json").exists():
            s = json.loads((d/"complete.json").read_text())
            if s["seed"] != seed or s["split"] != a.split or s.get("timeouts", 0):
                raise ValueError("invalid completed collection disposition")
            rows.append(dict(seed=seed, checkpoint=cp, disposition="reached", trials=s["trials"]))
        elif (d/"failure.json").exists() and not (d/"inputs.json").exists():
            failure = json.loads((d/"failure.json").read_text())
            request = failure["request"]
            if (request["seed"] != seed or request["split"] != a.split or request["checkpoint"] != cp
                    or not admissible_setup_error(failure["error"], require_trace=True)):
                raise ValueError("unresolved implementation error during collection")
            rows.append(dict(seed=seed, checkpoint=cp, disposition="setup_failed", failure=failure))
        else:
            raise ValueError(f"missing/incomplete configuration {seed}/{cp}")
    loaded = load_groups([a.data], splits=(a.split,))
    if {g["seed"] for g in loaded} != {r["seed"] for r in rows if r["disposition"] == "reached"}:
        raise ValueError("loaded graph dataset does not match requested configurations")
    result = dict(split=a.split, requested=len(rows), reached=len(loaded),
                  setup_failures=len(rows)-len(loaded), trials=sum(r.get("trials",0) for r in rows), rows=rows)
    dump(Path(a.data)/f"audit_{a.split}.json", result)
    print(json.dumps({k:v for k,v in result.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
