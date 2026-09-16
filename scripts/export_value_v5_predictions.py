"""Score immutable v5 inputs before opening outcome files; freeze on validation.

The durable *.input_scores.json artifact contains no outcome arrays. Group
outcomes are opened only after every model has scored every reached input pool.
Model checkpoint metadata may contain training-time validation summaries; these
are not used to score candidates or to choose a test-time threshold/model.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simbench.value.plan import PlanIR, digest
from simbench.value.program_input_v5 import source_manifest
from scripts.evaluate_value_v5 import SCHEMA as PREDICTION_SCHEMA, validate_predictions


SELECTION_SCHEMA = "twingraph.value.model_selection.v5"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def scorer_factory(path, device):
    from simbench.value.value_v5 import ValueScorer
    return ValueScorer(path, device)


def dispositions_loader(roots, splits):
    from simbench.value.value_v5 import request_dispositions
    return request_dispositions(roots, splits)


def groups_loader(roots, splits):
    from simbench.value.value_v5 import load_groups
    return load_groups(roots, splits)


def scene_config_id(seed):
    from simbench.value.stage_v5 import StageV5Spec
    return StageV5Spec.sample(seed).config_id


def checkpoint_paths(values):
    result = {}
    for item in values:
        name, path = item.split("=", 1) if "=" in str(item) else (Path(item).parent.name, item)
        if not name or name in result or not Path(path).is_file():
            raise ValueError("checkpoint names must be unique and files must exist")
        result[name] = str(Path(path).resolve())
    if not result:
        raise ValueError("at least one checkpoint is required")
    return result


def load_input_manifest(roots, split, expected_seeds, n, repeats=None):
    """Inspect requests/source/input records and file existence, never outcomes."""
    dispositions = dispositions_loader(roots, (split,))
    if len(set(expected_seeds)) != len(expected_seeds) or not expected_seeds:
        raise ValueError("expected seeds must be a nonempty unique prospective list")
    if sorted(r["request"]["seed"] for r in dispositions) != sorted(expected_seeds):
        raise ValueError("requested configurations differ from prospective seed manifest")
    source_hash = digest(source_manifest())
    items = []
    for disposition in dispositions:
        directory = Path(disposition["directory"])
        request = disposition["request"]
        if request["n"] != n or (repeats is not None and request["repeats"] != repeats):
            raise ValueError("requested candidate/repeat budget differs from prospective manifest")
        if disposition["source_sha256"] != source_hash:
            raise ValueError("collected source differs from the current frozen source")
        item = dict(directory=str(directory), request=request, status=disposition["status"],
                    source_sha256=source_hash, request_file_sha256=sha(directory/"request.json"))
        if disposition["status"] == "completed":
            inputs = read(directory/"inputs.json")
            if (inputs.get("schema") != "twingraph.group.v5" or inputs["declared_split"] != split
                    or inputs["task"]["seed"] != request["seed"] or inputs["source_sha256"] != source_hash):
                raise ValueError("requested input identity/schema/source mismatch")
            if len(inputs["candidates"]) != n:
                raise ValueError("completed group does not contain the full candidate budget")
            plans = [PlanIR.from_dict(p) for p in inputs["candidates"]]
            if len({p.id for p in plans}) != n:
                raise ValueError("duplicate candidate identity")
            item.update(config_id=str(inputs["split_group"]), group_id=str(inputs["group_id"]),
                        inputs=inputs, input_sha256=digest(inputs), input_file_sha256=sha(directory/"inputs.json"))
        else:
            item.update(config_id=str(scene_config_id(request["seed"])), group_id=None,
                        failure=read(directory/"failure.json"), failure_file_sha256=sha(directory/"failure.json"))
        items.append(item)
    if len({r["config_id"] for r in items}) != len(items):
        raise ValueError("physical configurations must be distinct in this one-pool-per-configuration protocol")
    if not any(r["status"] == "completed" for r in items):
        raise ValueError("no reached candidate pool; preserve dispositions as a failed experiment")
    return sorted(items, key=lambda row:row["request"]["seed"]), source_hash


def _metadata(name, path, scorer, source_hash):
    saved = scorer.saved
    if saved.get("schema") != "twingraph.value.v5" or saved["source_sha256"] != source_hash:
        raise ValueError(f"{name}: checkpoint source/schema differs from frozen collection")
    checkpoint_sha = sha(path)
    if scorer.checkpoint_sha256 != checkpoint_sha:
        raise ValueError("checkpoint bytes changed while loading")
    split_path, source_path = Path(path).parent/"split.json", Path(path).parent/"source.json"
    if digest(read(split_path)) != saved["split_sha256"] or digest(read(source_path)) != source_hash:
        raise ValueError("checkpoint training-split/source sidecar binding mismatch")
    return dict(name=name, path=path, sha256=checkpoint_sha, source_sha256=source_hash,
        split_sha256=saved["split_sha256"], encoding_sha256=saved["encoding_sha256"],
        interface_sha256=saved["interface_sha256"], kind=saved["kind"], seed=saved["seed"], epoch=saved["epoch"],
        split_file_sha256=sha(split_path), model_load_seconds=float(scorer.load_seconds))


def verify_selection(selection, paths, source_hash):
    if selection.get("schema") != SELECTION_SCHEMA or selection["source_sha256"] != source_hash:
        raise ValueError("selection/source schema mismatch")
    if set(paths) != set(selection["models"]) or selection["selected"] not in paths:
        raise ValueError("checkpoint names differ from frozen selection")
    for name, path in paths.items():
        if sha(path) != selection["models"][name]["sha256"]:
            raise ValueError(f"{name}: checkpoint hash differs from validation freeze")


def export_predictions(roots, paths, output, *, split, expected_seeds=None, n=12,
                       selection=None, selection_output=None, test_seeds=None, test_repeats=1,
                       device="cpu", primary_k=4):
    """Validation may select; test must use an existing immutable selection."""
    if split not in {"val", "test"} or not 1 <= primary_k <= n:
        raise ValueError("invalid export split or primary K")
    output = Path(output)
    score_path = output.with_name(output.stem + ".input_scores.json")
    if output.exists() or score_path.exists() or selection_output and Path(selection_output).exists():
        raise ValueError("refuse to overwrite prediction/score/freeze evidence")
    if split == "test":
        if selection is None:
            raise ValueError("test requires a validation model-selection manifest")
        if selection_output is not None or test_seeds is not None:
            raise ValueError("test cannot request a new model selection")
        expected_seeds = selection["test"]["seeds"]
        n, test_repeats, primary_k = selection["test"]["n"], selection["test"]["repeats"], selection["primary_k"]
    elif (selection is not None or selection_output is None or not test_seeds or test_repeats < 1
          or len(set(test_seeds)) != len(test_seeds)):
        raise ValueError("validation requires a new selection output and prospective test specification")
    items, source_hash = load_input_manifest(roots, split, list(expected_seeds or []), n,
                                            repeats=test_repeats if split == "test" else None)
    if split == "test":
        verify_selection(selection, paths, source_hash)
        if {r["config_id"] for r in items} & set(selection["validation_config_ids"]):
            raise ValueError("test configuration overlaps validation")
    elif set(expected_seeds) & set(test_seeds):
        raise ValueError("prospective test seeds overlap validation")
    metadata, scores = {}, {}
    for name, path in sorted(paths.items()):
        scorer = scorer_factory(path, device)
        metadata[name] = _metadata(name, path, scorer, source_hash)
        training_split = read(Path(path).parent/"split.json")
        known_training_configs = {r["config_id"] for r in training_split["train"]}
        if {r["config_id"] for r in items} & known_training_configs:
            raise ValueError("evaluation configuration overlaps model training inputs")
        if split == "val":
            actual = {(r["group_id"],r["config_id"],r["input_sha256"]) for r in items if r["status"] == "completed"}
            expected = {(r["group_id"],r["config_id"],r["input_sha256"]) for r in training_split["val"]}
            if actual != expected:
                raise ValueError("validation inputs differ from checkpoint selection split")
        elif any(metadata[name][field] != selection["models"][name][field] for field in
                 ("source_sha256", "split_sha256", "encoding_sha256", "interface_sha256", "kind", "seed", "epoch")):
            raise ValueError("loaded checkpoint metadata differs from frozen selection")
        scores[name] = {}
        for item in items:
            if item["status"] != "completed":
                continue
            inp = item["inputs"]
            before = digest(inp)
            started = time.perf_counter()
            ranked = scorer.rank(copy.deepcopy(inp["observation"]), copy.deepcopy(inp["candidates"]), primary_k)
            elapsed = time.perf_counter()-started
            if digest(inp) != before or ranked["checkpoint_sha256"] != metadata[name]["sha256"]:
                raise ValueError("scorer mutated inputs or changed frozen checkpoint")
            ids = [p["id"] for p in inp["candidates"]]
            probabilities, logits = np.asarray(ranked["scores"]), np.asarray(ranked["logits"])
            if (probabilities.shape != (n,) or logits.shape != (n,) or not np.isfinite(probabilities).all()
                    or not np.isfinite(logits).all() or np.any((probabilities < 0) | (probabilities > 1))):
                raise ValueError("scorer output is not finite, complete candidate probabilities/logits")
            expected_order = [ids[i] for i in np.argsort(-logits, kind="stable")]
            if ranked["order"] != expected_order:
                raise ValueError("scorer ranking differs from its saved input-order logits")
            scores[name][item["group_id"]] = dict(group_id=item["group_id"], config_id=item["config_id"],
                seed=item["request"]["seed"], checkpoint=inp.get("checkpoint"), candidate_ids=ids,
                scores=probabilities.tolist(), ranking_scores=logits.tolist(), input_sha256=item["input_sha256"],
                plan_lengths=[len(p["calls"]) for p in inp["candidates"]],
                model_sha256=metadata[name]["sha256"], seconds=ranked.get("seconds"), measured_rank_wall_seconds=elapsed,
                input_diagnostics=ranked.get("input_diagnostics"), unknown_input_fields=ranked.get("unknown_input_fields"))
        del scorer
    if len({m["split_sha256"] for m in metadata.values()}) != 1:
        raise ValueError("candidate models were trained on different splits")
    input_only = dict(schema="twingraph.value.input_scores.v5", split=split, source_sha256=source_hash,
        models=metadata, scores=scores, device=device, created_unix=time.time(), exporter_sha256=sha(__file__),
        ordering="All model/pool scoring completed; no group outcomes.json or completion counts opened yet.",
        requested_dispositions=[{k:v for k,v in r.items() if k not in {"inputs","failure"}} for r in items])
    write_new(score_path, input_only)
    # The first outcome-file access is below this durable, label-free checkpoint.
    groups = groups_loader(roots, (split,))
    by_group = {g["id"]:g for g in groups}
    if set(by_group) != {r["group_id"] for r in items if r["status"] == "completed"}:
        raise ValueError("scored and outcome-bound groups differ")
    methods = {}
    for name in sorted(paths):
        rows = []
        for gid, scored in scores[name].items():
            group = by_group[gid]
            if (group["input_sha256"] != scored["input_sha256"] or group["config"] != scored["config_id"]
                    or [p.id for p in group["plans"]] != scored["candidate_ids"]
                    or group["collected_source_sha256"] != source_hash):
                raise ValueError("labels differ from the scored input/source/candidate binding")
            outcomes = group["outcomes"]
            rows.append(dict(scored, outcomes=outcomes, nominal_index=group["nominal_index"],
                label_regime="nominal" if len(outcomes[0]) == 1 else "repeated"))
        methods[name] = dict(model_sha256=metadata[name]["sha256"], rows=rows)
    if split == "val":
        for name, method in methods.items():
            per_config = {}
            for row in method["rows"]:
                y = np.asarray(row["outcomes"])[:,row["nominal_index"]]
                per_config.setdefault(row["config_id"], []).append(float(np.mean((np.asarray(row["scores"])-y)**2)))
            metadata[name]["validation_nominal_brier"] = float(np.mean([np.mean(v) for v in per_config.values()]))
        selected = min(metadata, key=lambda name:(metadata[name]["validation_nominal_brier"], name))
        selection = dict(schema=SELECTION_SCHEMA, selected=selected, models=metadata, source_sha256=source_hash,
            validation_config_ids=[r["config_id"] for r in items], validation_seeds=list(expected_seeds),
            validation_input_scores_sha256=sha(score_path), primary_k=primary_k,
            test=dict(seeds=sorted(test_seeds), n=n, repeats=test_repeats), created_unix=time.time(),
            selection_rule="minimum nominal validation Brier, equally weighted physical configurations; exact ties by model name",
            exporter_sha256=sha(__file__))
    payload = dict(schema=PREDICTION_SCHEMA, split="validation" if split == "val" else "test",
        selected=selection["selected"], source_sha256=source_hash, methods=methods,
        requested_config_ids=[r["config_id"] for r in items],
        setup_failures=[dict(config_id=r["config_id"], seed=r["request"]["seed"],
            reason=r["failure"].get("message"), exception_type=r["failure"].get("exception_type"),
            failure_sha256=r["failure_file_sha256"]) for r in items if r["status"] != "completed"],
        selection_metadata=dict(rule=selection["selection_rule"], model_selection_sha256=digest(selection)),
        input_scores_sha256=sha(score_path), prediction_device=device,
        source_file_hashes=source_manifest(), exporter_sha256=sha(__file__))
    validate_predictions(payload, "validation" if split == "val" else "test")
    if digest(source_manifest()) != source_hash or any(sha(path) != metadata[name]["sha256"] for name,path in paths.items()):
        raise ValueError("source or checkpoints changed during prediction export")
    write_new(output, payload)
    if split == "val":
        write_new(selection_output, selection)
    return payload, selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validation", "test"):
        p = commands.add_parser(name)
        p.add_argument("--data", nargs="+", required=True); p.add_argument("--out", required=True)
        p.add_argument("--device", default="cpu")
        p.add_argument("--checkpoints", nargs="+", required=name == "validation", help="paths, or stable_name=path; test override preserves frozen hashes")
        if name == "validation":
            p.add_argument("--selection-out", required=True)
            p.add_argument("--expected-seeds", type=int, nargs="+", required=True)
            p.add_argument("--test-seeds", type=int, nargs="+", required=True)
            p.add_argument("--n", type=int, default=12); p.add_argument("--test-repeats", type=int, default=1)
            p.add_argument("--primary-k", type=int, default=4)
        else:
            p.add_argument("--selection", required=True)
    args = parser.parse_args()
    if args.command == "validation":
        export_predictions(args.data, checkpoint_paths(args.checkpoints), args.out, split="val",
            expected_seeds=args.expected_seeds, n=args.n, selection_output=args.selection_out,
            test_seeds=args.test_seeds, test_repeats=args.test_repeats, device=args.device, primary_k=args.primary_k)
    else:
        selection = read(args.selection)
        paths = checkpoint_paths(args.checkpoints) if args.checkpoints else {name:m["path"] for name,m in selection["models"].items()}
        export_predictions(args.data, paths, args.out, split="test", selection=selection, device=args.device)


if __name__ == "__main__":
    main()
