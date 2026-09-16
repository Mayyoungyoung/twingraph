"""Compare frozen scorers across CPU/CUDA and candidate-order reversal.

Only the frozen validation input groups are opened. Outcomes, completion
summaries and test inputs are never read. Raw logits determine every diagnostic
ranking; this program does not fit, calibrate or select any model.

Example:
    python scripts/check_value_v4_devices.py --freeze results/freeze.json \
        --data datasets/value_v4 --out results/device_diagnostic.json

--help and the pure comparison helpers require only Python's standard library.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def logit_map(candidate_ids, result):
    """Align outputs by identities, including when input order was reversed."""
    ids, values = list(candidate_ids), result["logits"]
    if len(ids) != len(set(ids)) or len(ids) != len(values) or not ids:
        raise ValueError("invalid candidate/logit cardinality")
    mapped = dict(zip(ids, map(float, values)))
    if not all(math.isfinite(v) for v in mapped.values()):
        raise ValueError("nonfinite model logit")
    if len(result["order"]) != len(ids) or set(result["order"]) != set(ids):
        raise ValueError("scorer order does not cover the supplied candidates")
    return mapped


def canonical_rank(mapped):
    """Use candidate identity only as a deterministic exact-tie diagnostic."""
    return sorted(mapped, key=lambda cid: (-mapped[cid], cid))


def logit_range(mapped):
    values = list(mapped.values())
    low, high = min(values), max(values)
    return dict(minimum=low, maximum=high, within_pool_range=high-low,
                distinct_logits=len(set(values)))


def order_agreement(first, second):
    if len(first) != len(second) or set(first) != set(second) or not first:
        raise ValueError("orders must contain the same nonempty candidate set")
    k = min(4, len(first))
    a, b = first[:k], second[:k]
    return dict(effective_k=k, top1_agreement=first[0] == second[0],
                top4_set_agreement=set(a) == set(b), top4_order_agreement=a == b,
                top4_overlap_fraction=len(set(a) & set(b))/k,
                full_order_agreement=first == second)


def compare(first, second, first_order=None, second_order=None):
    if set(first) != set(second) or not first:
        raise ValueError("logit maps must cover the same nonempty candidate set")
    differences = [abs(first[cid]-second[cid]) for cid in first]
    scale = logit_range(first)["within_pool_range"]
    result = dict(max_logit_error=max(differences),
                  mean_logit_error=sum(differences)/len(differences),
                  error_over_reference_pool_range=max(differences)/scale if scale else None,
                  canonical=order_agreement(canonical_rank(first), canonical_rank(second)))
    if first_order is not None and second_order is not None:
        result["scorer_returned_order"] = order_agreement(first_order, second_order)
    return result


def validation_locations(frozen, split, saved):
    """Only identities/seeds/hashes are used from checkpoint validation metadata.

    Split manifests omit seeds, whereas validation summary rows record them.
    These metadata identify directories without opening other splits' inputs.
    No reference labels or prediction values in the checkpoint are consulted.
    """
    rows = saved["validation"]["rows"]
    metadata = {row["group_id"]: int(row["seed"]) for row in rows}
    expected = split["val"]
    if (not expected or len(metadata) != len(rows)
            or len({row["id"] for row in expected}) != len(expected)
            or set(metadata) != {row["id"] for row in expected}):
        raise ValueError("checkpoint validation metadata disagrees with split manifest")
    test_seeds = set(map(int, frozen.get("test", {}).get("seeds", [])))
    result = []
    for row in expected:
        gid = row["id"]
        try:
            family = row["config"][0]
            checkpoint = int(gid.rsplit("_cp", 1)[1])
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ValueError("expected v4 physical configuration/checkpoint metadata") from exc
        seed = metadata[gid]
        if seed in test_seeds:
            raise ValueError("validation metadata overlaps frozen test seeds")
        result.append(dict(id=gid, family=family, seed=seed, checkpoint=checkpoint,
                           input_sha256=row["input_sha256"],
                           directory=f"group_{family}_{seed}_{checkpoint}"))
    return sorted(result, key=lambda row: (row["seed"], row["checkpoint"], row["id"]))


def validation_inputs(roots, locations, digest_fn, compile_fn):
    groups = []
    for location in locations:
        matches = set()
        for root in map(Path, roots):
            if root.name == "inputs.json" and root.parent.name == location["directory"]:
                candidate = root
            elif root.name == location["directory"]:
                candidate = root/"inputs.json"
            else:
                candidate = root/location["directory"]/"inputs.json"
            if candidate.is_file():
                matches.add(candidate.resolve())
        if len(matches) != 1:
            raise ValueError(f"expected one validation inputs.json for {location['directory']}, found {len(matches)}")
        path = next(iter(matches))
        inp = json.loads(path.read_text(encoding="utf-8"))
        if (inp.get("declared_split") != "val" or inp["group_id"] != location["id"]
                or inp["task"]["family"] != location["family"]
                or inp["task"]["seed"] != location["seed"]
                or inp["checkpoint"] != location["checkpoint"]
                or digest_fn(inp) != location["input_sha256"]):
            raise ValueError("input does not match the frozen validation group")
        ids = [p["id"] for p in inp["candidates"]]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("validation pool has empty/duplicate candidate identity")
        graphs = [compile_fn(inp["observation"], p) for p in inp["candidates"]]
        groups.append(dict(**location, input_path=str(path), candidate_ids=ids, graphs=graphs))
    return groups


def summarize(rows):
    summary = {}
    for name in ("cpu_vs_cuda", "cpu_reversal", "cuda_reversal"):
        comparisons = [row["comparisons"][name] for row in rows]
        item = dict(max_logit_error=max(r["max_logit_error"] for r in comparisons))
        for ranking in ("canonical", "scorer_returned_order"):
            item[ranking] = {key: sum(float(r[ranking][key]) for r in comparisons)/len(comparisons)
                             for key in ("top1_agreement", "top4_set_agreement", "top4_order_agreement",
                                         "top4_overlap_fraction", "full_order_agreement")}
        summary[name] = item
    summary["pool_ranges"] = {
        name: dict(minimum=min(row["ranges"][name]["within_pool_range"] for row in rows),
                   maximum=max(row["ranges"][name]["within_pool_range"] for row in rows))
        for name in ("cpu", "cuda", "cpu_reversed", "cuda_reversed")}
    return summary


def run(args):
    # Heavy robot/Torch imports occur after argument parsing, so --help is portable.
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("CPU/CUDA comparison requires a Torch runtime with CUDA support") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no CPU/CUDA comparison was performed")
    if args.threads < 1 or not args.cuda_device.startswith("cuda"):
        raise ValueError("positive CPU thread count and a CUDA device are required")
    torch.set_num_threads(args.threads)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from simbench.value.plan import digest
    from simbench.value.schema_rank import SchemaScorer
    from simbench.value.skill_graph import compile_graph, interface_hash

    started = time.perf_counter()
    freeze_path = Path(args.freeze).resolve()
    frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (frozen.get("schema") != "twingraph.schema_experiment.freeze.v1"
            or frozen["interface_sha256"] != interface_hash()):
        raise ValueError("frozen schema/execution interface mismatch")
    records = frozen["models"]
    if (not records or len({row["name"] for row in records}) != len(records)
            or frozen["selected"] not in {row["name"] for row in records}
            or len({row["split_sha256"] for row in records}) != 1):
        raise ValueError("freeze must contain distinct models sharing one split and its selected model")
    paths = {}
    for row in records:
        path = Path(row["path"])
        path = path if path.is_absolute() else freeze_path.parent/path
        if sha(path) != row["sha256"]:
            raise ValueError(f"frozen checkpoint changed: {row['name']}")
        paths[row["name"]] = path
    first = records[0]
    split_path = paths[first["name"]].parent/"split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if digest(split) != first["split_sha256"]:
        raise ValueError("saved training split manifest differs from freeze")
    first_cpu = SchemaScorer(paths[first["name"]], "cpu")
    groups = validation_inputs(args.data, validation_locations(frozen, split, first_cpu.saved), digest, compile_graph)
    device = torch.device(args.cuda_device)
    gpu_index = device.index if device.index is not None else torch.cuda.current_device()
    result = dict(schema="twingraph.value.device_diagnostic.v1", freeze_sha256=digest(frozen),
        selected=frozen["selected"], model_count=len(records), validation_groups=len(groups),
        input_files=[dict(group_id=g["id"], path=g["input_path"], sha256=g["input_sha256"]) for g in groups],
        environment=dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
            cuda_device=args.cuda_device, gpu=torch.cuda.get_device_name(gpu_index), cpu_threads=args.threads,
            float32_matmul_precision=torch.get_float32_matmul_precision(),
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32),
        script_sha256=sha(__file__), models={}, selected_detailed_rows=[],
        interpretation=dict(inputs="Exactly the checkpoint validation groups, verified by frozen input hashes; no outcome/completion/test files read",
            rankings="Raw logits, no probability rounding or near-tie tolerance; canonical ranks use candidate ID only to resolve exact ties",
            returned_order="Actual SchemaScorer order retains its input-order tie break; reversal disagreements here can arise from exact ties",
            action="Diagnostic only; no fitting, calibration, freeze, model-selection or deployment-rule change"))
    for index, row in enumerate(records):
        cpu = first_cpu if index == 0 else SchemaScorer(paths[row["name"]], "cpu")
        cuda = SchemaScorer(paths[row["name"]], args.cuda_device)
        if cpu.saved["split_sha256"] != row["split_sha256"] or cuda.saved["split_sha256"] != row["split_sha256"]:
            raise ValueError("loaded checkpoint split differs from freeze")
        rows = []
        for group in groups:
            graphs, ids = group["graphs"], group["candidate_ids"]
            outputs = dict(cpu=cpu.rank(graphs, min(4, len(graphs)), calibrated=False),
                cuda=cuda.rank(graphs, min(4, len(graphs)), calibrated=False),
                cpu_reversed=cpu.rank(list(reversed(graphs)), min(4, len(graphs)), calibrated=False),
                cuda_reversed=cuda.rank(list(reversed(graphs)), min(4, len(graphs)), calibrated=False))
            maps = {name: logit_map(list(reversed(ids)) if name.endswith("reversed") else ids, output)
                    for name, output in outputs.items()}
            comparisons = {name: compare(maps[a], maps[b], outputs[a]["order"], outputs[b]["order"])
                for name, a, b in (("cpu_vs_cuda", "cpu", "cuda"),
                                    ("cpu_reversal", "cpu", "cpu_reversed"),
                                    ("cuda_reversal", "cuda", "cuda_reversed"))}
            detail = dict(group_id=group["id"], seed=group["seed"], checkpoint=group["checkpoint"],
                n=len(ids), ranges={name: logit_range(values) for name, values in maps.items()},
                comparisons=comparisons)
            rows.append(detail)
            if row["name"] == frozen["selected"]:
                result["selected_detailed_rows"].append(dict(detail,
                    candidate_ids=sorted(ids), logits_by_candidate_id=maps,
                    canonical_orders={name: canonical_rank(values) for name, values in maps.items()},
                    scorer_returned_orders={name: output["order"] for name, output in outputs.items()}))
        model_summary = summarize(rows)
        result["models"][row["name"]] = dict(kind=row["kind"], seed=row["seed"], checkpoint_sha256=row["sha256"],
                                              summary=model_summary, rows=rows)
        print(json.dumps(dict(model=row["name"], completed_models=index+1, total_models=len(records),
                              summary=model_summary)), flush=True)
        del cpu, cuda, outputs, maps
        if index == 0:
            del first_cpu
        torch.cuda.empty_cache()
    result["wall_seconds"] = time.perf_counter()-started
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(dict(output=str(output.resolve()), selected=frozen["selected"], models=len(records),
                          validation_groups=len(groups), wall_seconds=result["wall_seconds"])), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True, help="Existing frozen model manifest")
    parser.add_argument("--data", nargs="+", required=True, help="Dataset roots or validation-group directories")
    parser.add_argument("--out", required=True, help="Output diagnostic JSON path")
    parser.add_argument("--cuda-device", default="cuda", help="CUDA device, default: cuda")
    parser.add_argument("--threads", type=int, default=2, help="CPU Torch threads, default: 2")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
