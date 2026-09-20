"""Verify and summarize the immutable paired candidate test archives."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tarfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = {}
    for seed in (1320, 1321, 1322):
        path = args.archives / f"seed_{seed}.tar.gz"
        with tarfile.open(path) as tar:
            summary = json.load(tar.extractfile("./summary.json"))
            assert len(summary["rows"]) == 24
            assert not any("infrastructure_error" in row for row in summary["rows"])
            seen = set()
            geometry = set()
            observations = set()
            physics = set()
            for row in summary["rows"]:
                key = (row["pool"], row["candidate"])
                assert key not in seen
                seen.add(key)
                name = f"./seed_{seed}/{row['pool']}/{row['candidate']}/result.json"
                detail = json.load(tar.extractfile(name))
                assert detail["task_version"] == "functional_assembly_v9_funnel_r1"
                assert detail["result"]["valid"]
                assert bool(detail["result"]["full_success"]) == bool(row["success"])
                geometry.add(detail["geometry_sha256"])
                observations.add(detail["pre_execution_observation"]["sha256"])
                applied = detail["result"]["applied_parameters"]
                assert applied["friction_scale"] == applied["actuator_gain_scale"] == 1.0
                physics.add((applied["geom_friction_sha256"], applied["actuator_gain_sha256"]))
            assert len(geometry) == len(observations) == len(physics) == 1
            pools = json.load(tar.extractfile("./pools.json"))
            assert all(len(pools[pool]) == 12 for pool in ("old_v9", "new_v10"))
            manifest[str(seed)] = dict(archive=path.name,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                rows=len(summary["rows"]), geometry_sha256=next(iter(geometry)),
                observation_sha256=next(iter(observations)),
                success_by_pool=dict(Counter(row["pool"] for row in summary["rows"] if row["success"])),
                failure_skill_by_pool={pool: dict(Counter((row["error"] or "").split(":", 1)[0]
                    for row in summary["rows"] if row["pool"] == pool and not row["success"]))
                    for pool in ("old_v9", "new_v10")})
    assert len({row["geometry_sha256"] for row in manifest.values()}) == 3
    args.out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
