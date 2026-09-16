"""Frozen, outcome-based classification and screening audits (no simulation).

Rows bind one score to each unique candidate and its raw binary executions.
The first execution is only nominal when explicitly declared by nominal_index.
This evaluator never trains a scorer or derives a runtime from a rollout budget.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


SCHEMA = "twingraph.value.predictions.v5"
FREEZE_SCHEMA = "twingraph.value.evaluation_freeze.v5"
INTERPRETATION = (
    "Primary classification and screening use the declared nominal binary execution. "
    "A single nominal outcome is not a success-probability estimate or real-robot evidence. "
    "Repeated-trial metrics reuse each pre-execution prediction across the recorded trials; "
    "empirical-fraction screening is secondary, with feasible meaning at least one observed "
    "success. Configuration bootstrap retains every candidate, repeat and sibling group "
    "together, but cannot estimate unobserved disturbance variability. Degenerate confidence "
    "intervals are not reliability guarantees."
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _method_payloads(payload):
    if payload.get("schema") != SCHEMA or not payload.get("methods"):
        raise ValueError("expected nonempty v5 predictions")
    result = {}
    for name, value in payload["methods"].items():
        metadata = value if isinstance(value, dict) else {}
        rows = metadata.get("rows") if metadata else value
        model_sha = metadata.get("model_sha256", payload.get("model_sha256s", {}).get(name,
                                          payload.get("model_sha256")))
        if not isinstance(model_sha, str) or len(model_sha) != 64 or any(c not in "0123456789abcdef" for c in model_sha):
            raise ValueError(f"{name}: require a SHA256 binding for model or deterministic baseline")
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"{name}: empty rows")
        result[name] = dict(model_sha256=model_sha, rows=rows,
                            probability_metrics=metadata.get("probability_metrics", True))
    return result


def validate_predictions(payload, split):
    if payload.get("split") not in ({"val", "validation"} if split == "validation" else {split}):
        raise ValueError(f"expected {split} predictions; never select from test")
    methods = _method_payloads(payload)
    shared = None
    for name, method in methods.items():
        checked, bindings = [], {}
        for raw in method["rows"]:
            row = dict(raw)
            gid, config = str(row["group_id"]), str(row["config_id"])
            ids = row["candidate_ids"]
            if gid in bindings or not ids or len(set(ids)) != len(ids):
                raise ValueError("duplicate group or candidate identity")
            scores = np.asarray(row["scores"], dtype=float)
            ranks = np.asarray(row.get("ranking_scores", scores), dtype=float)
            outcomes = np.asarray(row["outcomes"], dtype=float)
            if (scores.shape != (len(ids),) or ranks.shape != scores.shape or
                    not np.isfinite(scores).all() or not np.isfinite(ranks).all()):
                raise ValueError("candidate/score cardinality or finiteness mismatch")
            if outcomes.ndim != 2 or outcomes.shape[0] != len(ids) or not outcomes.shape[1] or not np.isin(outcomes, [0, 1]).all():
                raise ValueError("outcomes must be a rectangular candidate-by-trial binary matrix")
            regime = row.get("label_regime")
            if regime not in {"nominal", "repeated"} or (regime == "nominal" and outcomes.shape[1] != 1):
                raise ValueError("declare nominal R=1 or repeated raw outcomes")
            nominal = row.get("nominal_index", 0 if regime == "nominal" else None)
            if type(nominal) is not int or not 0 <= nominal < outcomes.shape[1]:
                raise ValueError("explicit nominal_index required for repeated outcomes")
            probabilistic = row.get("probability_metrics", method["probability_metrics"])
            if probabilistic and np.any((scores < 0) | (scores > 1)):
                raise ValueError("probability metrics require probabilities, not logits")
            row.update(group_id=gid, config_id=config, scores=scores, ranking_scores=ranks,
                       outcomes=outcomes.astype(int), nominal_index=nominal,
                       probability_metrics=bool(probabilistic))
            bindings[gid] = (config, ids, outcomes.tolist(), nominal)
            checked.append(row)
        if shared is not None and bindings != shared:
            raise ValueError("methods must share identical candidate pools, identities and outcomes")
        shared = bindings
        method["rows"] = checked
        if len({r["probability_metrics"] for r in checked}) != 1:
            raise ValueError("probability availability must be constant within a method")
        method["probability_metrics"] = checked[0]["probability_metrics"]
    return methods


def binary_metrics(labels, scores, threshold):
    """Trial micro metrics; grouped-score ROC/AP give exact half credit for ties."""
    y, p = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    if y.shape != p.shape or y.ndim != 1 or not len(y) or not np.isin(y, [0, 1]).all() or not np.isfinite(p).all():
        raise ValueError("invalid binary metric vectors")
    predicted = p >= threshold
    tp = int(np.sum(predicted & (y == 1))); fp = int(np.sum(predicted & (y == 0)))
    tn = int(np.sum(~predicted & (y == 0))); fn = int(np.sum(~predicted & (y == 1)))
    positive, negative = tp + fn, tn + fp
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / positive if positive else None
    specificity = tn / negative if negative else None
    auc = ap = None
    if positive and negative:
        order = np.argsort(-p, kind="stable")
        sorted_y, sorted_p = y[order], p[order]
        ends = np.r_[np.flatnonzero(np.diff(sorted_p)) + 1, len(y)]
        cumulative_positive = np.cumsum(sorted_y)[ends - 1]
        cumulative_negative = ends - cumulative_positive
        delta_positive = np.diff(np.r_[0, cumulative_positive])
        delta_negative = np.diff(np.r_[0, cumulative_negative])
        # Count each positive above a negative, plus half of each tied pair.
        auc = float(np.sum(delta_positive * (negative - cumulative_negative + .5 * delta_negative)) / (positive * negative))
        ap = float(np.sum(delta_positive / positive * cumulative_positive / ends))
    clipped = np.clip(p, 1e-12, 1 - 1e-12)
    return dict(trials=len(y), positive_trials=positive, negative_trials=negative,
        tp=tp, fp=fp, tn=tn, fn=fn, accuracy=(tp + tn) / len(y),
        balanced_accuracy=(recall + specificity) / 2 if positive and negative else None,
        precision=precision, recall=recall, specificity=specificity,
        f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        roc_auc=auc, average_precision=ap,
        brier=float(np.mean((p - y) ** 2)),
        log_loss=float(np.mean(-y * np.log(clipped) - (1 - y) * np.log1p(-clipped))))


def classification(rows, threshold, nominal=True):
    y, p = [], []
    for row in rows:
        outcomes = row["outcomes"][:, row["nominal_index"]:row["nominal_index"] + 1] if nominal else row["outcomes"]
        y.extend(outcomes.ravel())
        p.extend(np.repeat(row["scores"], outcomes.shape[1]))
    return binary_metrics(y, p, threshold)


def select_threshold(rows):
    y = np.concatenate([r["outcomes"][:, r["nominal_index"]] for r in rows])
    if len(np.unique(y)) != 2:
        return dict(threshold=.5, reason="single nominal validation class; fixed 0.5 fallback")
    p = np.concatenate([r["scores"] for r in rows])
    # Every distinct decision partition, including predicting no candidates positive.
    thresholds = np.unique(np.r_[0., .5, p, np.nextafter(float(p.max()), math.inf)])
    records = [(float(t), binary_metrics(y, p, t)) for t in thresholds]
    t, metric = max(records, key=lambda item: (item[1]["balanced_accuracy"], item[1]["recall"],
                                              -abs(item[0] - .5), -item[0]))
    return dict(threshold=t, reason="nominal validation balanced accuracy, recall, proximity to 0.5, then lower threshold",
                nominal_validation=metric)


def random_screening(reference, k, epsilon=.1):
    y = np.asarray(reference, dtype=float)
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all() or np.any((y < 0) | (y > 1)) or type(k) is not int or k < 1:
        raise ValueError("invalid reference or K")
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("invalid near-best epsilon")
    n, k = len(y), min(k, len(y))
    feasible = int(np.sum(y > 0))
    def hit(count):
        return 1. if n - count < k else 1. - math.comb(n - count, k) / math.comb(n, k)
    # P(max == v) is the difference of adjacent exact subset CDFs.
    expected_max, previous = 0., 0.
    for value in np.unique(y):
        count = int(np.sum(y <= value))
        cumulative = math.comb(count, k) / math.comb(n, k) if count >= k else 0.
        expected_max += value * (cumulative - previous)
        previous = cumulative
    return dict(effective_k=k, feasible_precision=feasible/n,
        feasible_recall=k/n if feasible else None, feasible_hit=hit(feasible),
        near_best_hit=hit(int(np.sum(y >= y.max() - epsilon))) if feasible else None,
        quality=float(y.mean()), best_quality=float(expected_max), regret=float(y.max() - expected_max))


def screening_row(row, k, epsilon=.1, nominal=True):
    y = row["outcomes"][:, row["nominal_index"]] if nominal else row["outcomes"].mean(axis=1)
    rank = np.argsort(-row["ranking_scores"], kind="stable")
    selected = y[rank[:k]]
    feasible, selected_feasible = int(np.sum(y > 0)), int(np.sum(selected > 0))
    random = random_screening(y, k, epsilon)
    result = dict(group_id=row["group_id"], config_id=row["config_id"], n=len(y),
        checkpoint=row.get("checkpoint"), k=k, effective_k=len(selected),
        feasible_candidates=feasible, selected_feasible=selected_feasible,
        selected_ids=[row["candidate_ids"][i] for i in rank[:k]],
        all_failure=not bool(feasible), all_success=bool(np.all(y == 1)),
        feasible_precision=selected_feasible/len(selected),
        feasible_recall=selected_feasible/feasible if feasible else None,
        feasible_hit=float(selected_feasible > 0),
        near_best_hit=float(selected.max() >= y.max() - epsilon) if feasible else None,
        quality=float(selected.mean()), best_quality=float(selected.max()),
        regret=float(y.max()-selected.max()))
    for key, value in random.items():
        if key == "effective_k":
            continue
        result[f"random_{key}"] = value
        result[f"excess_{key}"] = result[key] - value if value is not None else None
    return result


def _interval(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    return np.quantile(values, [.025, .975]).tolist() if values else None


def cluster_means(rows, keys, bootstrap_samples=2000, seed=17):
    result = {}
    for key in keys:
        clusters = {}
        for row in rows:
            if row.get(key) is not None:
                clusters.setdefault(row["config_id"], []).append(float(row[key]))
        values = np.array([np.mean(v) for _, v in sorted(clusters.items())])
        rng = np.random.default_rng(seed)
        means = [float(np.mean(values[rng.integers(len(values), size=len(values))])) for _ in range(bootstrap_samples)] if len(values) else []
        result[key] = dict(mean=float(values.mean()) if len(values) else None,
            bootstrap95=_interval(means), eligible_configurations=len(values),
            eligible_groups=sum(map(len, clusters.values())))
    return result


CLASS_METRICS = ("accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "average_precision", "brier", "log_loss")
POOL_METRICS = ("feasible_precision", "feasible_recall", "feasible_hit", "near_best_hit", "quality", "best_quality", "regret")


def classification_summary(rows, threshold, nominal, bootstrap_samples, seed):
    point = classification(rows, threshold, nominal)
    clusters = {}
    for row in rows:
        clusters.setdefault(row["config_id"], []).append(row)
    groups = [v for _, v in sorted(clusters.items())]
    rng = np.random.default_rng(seed)
    samples = {key: [] for key in CLASS_METRICS}
    for _ in range(bootstrap_samples):
        sample = [r for index in rng.integers(len(groups), size=len(groups)) for r in groups[index]]
        metric = classification(sample, threshold, nominal)
        for key in samples:
            samples[key].append(metric[key])
    return dict(point=point, bootstrap95={key:_interval(v) for key,v in samples.items()},
                eligible_bootstrap_samples={key:sum(v is not None for v in values) for key,values in samples.items()},
                configurations=len(groups), weighting="trial micro; resample entire physical configurations")


def freeze(validation, ks=(1, 2, 4), primary_k=4, epsilon=.1, tolerance=1e-6):
    methods = validate_predictions(validation, "validation")
    if not ks or any(type(k) is not int or k < 1 for k in ks) or primary_k not in ks:
        raise ValueError("invalid frozen K values")
    if epsilon < 0 or not math.isfinite(epsilon) or tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError("invalid frozen diagnostic settings")
    frozen_methods = {}
    for name, method in methods.items():
        threshold = select_threshold(method["rows"]) if method["probability_metrics"] else dict(threshold=None, reason="nonprobability baseline")
        pool = [screening_row(row, primary_k, epsilon) for row in method["rows"]]
        validation_pool = cluster_means(pool, ("feasible_hit", "quality"), bootstrap_samples=1)
        frozen_methods[name] = dict(model_sha256=method["model_sha256"], **threshold,
                                    validation_primary_screening=validation_pool)
    selected = validation.get("selected")
    if selected is not None and selected not in methods:
        raise ValueError("selected deployment model is absent from validation predictions")
    return dict(schema=FREEZE_SCHEMA, validation_sha256=digest(validation), selected=selected,
        source_sha256=validation.get("source_sha256"), selection_metadata=validation.get("selection_metadata"),
        validation_config_ids=sorted({r["config_id"] for m in methods.values() for r in m["rows"]}),
        methods=frozen_methods, ks=list(dict.fromkeys(ks)), primary_k=primary_k,
        epsilon=epsilon, tie_tolerance=tolerance,
        selection="Thresholds only; deployment model must be selected and hashed from validation before test.",
        ranking="descending ranking_scores if supplied, else scores; stable candidate input order breaks exact ties")


def analyze(test, frozen, bootstrap_samples=2000, seed=17):
    if frozen.get("schema") != FREEZE_SCHEMA or bootstrap_samples < 1:
        raise ValueError("invalid freeze or bootstrap count")
    methods = validate_predictions(test, "test")
    if set(methods) != set(frozen["methods"]):
        raise ValueError("test methods must match validation freeze")
    if frozen.get("selected") != test.get("selected"):
        raise ValueError("selected deployment model changed after validation freeze")
    if frozen.get("source_sha256") != test.get("source_sha256"):
        raise ValueError("prediction source binding changed after validation freeze")
    result = dict(schema="twingraph.value.evaluation.v5", interpretation=INTERPRETATION,
        freeze_sha256=digest(frozen), predictions_sha256=digest(test), methods={},
        bootstrap=dict(samples=bootstrap_samples, seed=seed, unit="physical configuration"),
        source_metadata={k:v for k,v in test.items() if k != "methods"})
    reached = {r["config_id"] for m in methods.values() for r in m["rows"]}
    failures = test.get("setup_failures", [])
    failed = [str(r["config_id"]) for r in failures]
    requested = [str(c) for c in test.get("requested_config_ids", sorted(reached | set(failed)))]
    if (len(set(requested)) != len(requested) or len(set(failed)) != len(failed)
            or reached & set(failed) or set(requested) != reached | set(failed)):
        raise ValueError("requested configurations must be exactly reached or explicit setup failures")
    result["disposition"] = dict(requested_configurations=len(requested), reached_configurations=len(reached),
        setup_failed_configurations=len(failed), setup_failures=failures,
        requested_manifest_supplied="requested_config_ids" in test,
        interpretation="Classification is conditional on candidate creation; setup failures have no score. Requested-denominator feasible Hit counts these workflow failures as zero.")
    for name, method in methods.items():
        locked = frozen["methods"][name]
        if method["model_sha256"] != locked["model_sha256"]:
            raise ValueError(f"{name}: frozen model hash mismatch")
        rows = method["rows"]
        if {r["config_id"] for r in rows} & set(frozen["validation_config_ids"]):
            raise ValueError("test physical configuration overlaps validation")
        threshold = locked["threshold"]
        if method["probability_metrics"] != (threshold is not None):
            raise ValueError("probability availability changed after freeze")
        strata = {"overall":rows}
        for cp in sorted({str(r["checkpoint"]) for r in rows if r.get("checkpoint") is not None}):
            strata[f"checkpoint_{cp}"] = [r for r in rows if str(r.get("checkpoint")) == cp]
        output = dict(model_sha256=method["model_sha256"], threshold=threshold, strata={})
        for stratum, subset in strata.items():
            section = dict(groups=len(subset), configurations=len({r["config_id"] for r in subset}),
                candidates=sum(len(r["scores"]) for r in subset),
                trial_counts=sorted({r["outcomes"].shape[1] for r in subset}),
                classification_nominal=None, classification_all_trials=None, screening={})
            if threshold is not None:
                section["classification_nominal"] = classification_summary(subset, threshold, True, bootstrap_samples, seed)
                section["classification_all_trials"] = classification_summary(subset, threshold, False, bootstrap_samples, seed)
                brier_rows = [dict(config_id=r["config_id"], brier=float(np.mean((r["scores"]-r["outcomes"].mean(axis=1))**2))) for r in subset]
                section["empirical_fraction_brier"] = cluster_means(brier_rows, ("brier",), bootstrap_samples, seed)["brier"]
            for view, nominal in (("nominal", True), ("empirical_repeats", False)):
                section["screening"][view] = {}
                for k in frozen["ks"]:
                    pool = [screening_row(r, k, frozen["epsilon"], nominal) for r in subset]
                    keys = [prefix+key for prefix in ("", "random_", "excess_") for key in POOL_METRICS]
                    section["screening"][view][str(k)] = dict(rows=pool,
                        all_failure_groups=sum(r["all_failure"] for r in pool),
                        all_success_groups=sum(r["all_success"] for r in pool),
                        summary=cluster_means(pool, keys, bootstrap_samples, seed))
                    if stratum == "overall":
                        with_failures = pool + [dict(config_id=c, feasible_hit=0.) for c in failed]
                        section["screening"][view][str(k)]["requested_feasible_hit"] = cluster_means(
                            with_failures, ("feasible_hit",), bootstrap_samples, seed)["feasible_hit"]
            output["strata"][stratum] = section
        output["score_diagnostics"] = [dict(group_id=r["group_id"],
            probability_spread=float(np.ptp(r["scores"])), rank_spread=float(np.ptp(r["ranking_scores"])),
            exactly_unique_rank_scores=int(len(np.unique(r["ranking_scores"]))),
            nearly_flat_rank=bool(np.ptp(r["ranking_scores"]) <= frozen["tie_tolerance"])) for r in rows]
        result["methods"][name] = output
    return result


def analyze_timing(payload, bootstrap_samples=2000, seed=17):
    """Compare clocks from separately executed full-pool and screened pipelines."""
    if payload.get("schema") != "twingraph.system_run.v5" or not payload.get("rows") or bootstrap_samples < 1:
        raise ValueError("expected measured v5 system runs")
    full, top = {}, []
    for raw in payload["rows"]:
        row = dict(raw)
        key = (str(row["config_id"]), str(row["group_id"]), digest(row["paired_namespace"]))
        if row["policy"] not in {"full", "top_k"} or not 1 <= row["k"] <= row["n"]:
            raise ValueError("invalid timing policy or budget")
        if row["policy"] == "full" and row["k"] != row["n"]:
            raise ValueError("full digital twin must validate the entire candidate pool")
        if type(row["success"]) is not bool:
            raise ValueError("target success must be an independently observed binary outcome")
        if not isinstance(row.get("simulation_calls"), dict) or "twin_validation" not in row["simulation_calls"]:
            raise ValueError("record actual twin-validation and target-execution call counts")
        status = row.get("status", "executed_independent_target")
        complete = status in {"executed_independent_target", "abstained_no_validated_plan"}
        if not complete and status not in {"twin_setup_failed", "no_materializable_candidates"}:
            raise ValueError("unexpected system-run status; incomplete/programming failures are not timing evidence")
        expected = min(row["k"], row.get("actual_n", row["n"])) * row.get("config", {}).get("validation_repeats", 1)
        if complete and row["simulation_calls"]["twin_validation"] != expected:
            raise ValueError("a timing strategy did not finish its declared candidate validation budget")
        row["timing_eligible"] = complete and row.get("actual_n", row["n"]) == row["n"]
        row["timing_exclusion"] = None if row["timing_eligible"] else "partial_candidate_pool" if complete else status
        requested = row.get("requested_target_trials", 1)
        successes = row.get("target_successes", int(row["success"]))
        if type(requested) is not int or requested < 1 or type(successes) is not int or not 0 <= successes <= requested:
            raise ValueError("invalid independent target success counts")
        row["target_rate"] = successes / requested
        seconds = row["seconds"]
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in seconds.values()):
            raise ValueError("record finite nonnegative measured times")
        if seconds.get("total", 0) <= 0:
            raise ValueError("actual end-to-end total time is required")
        if row["policy"] == "full":
            if key in full:
                raise ValueError("duplicate full-pool timing for a paired case")
            full[key] = row
        else:
            top.append((key, row))
    if not full or not top:
        raise ValueError("both actual full-pool and Top-K executions are required")
    comparisons, seen = {}, set()
    for key, row in top:
        if key not in full or row["n"] != full[key]["n"]:
            raise ValueError("timing strategies must use the same paired configuration and candidate count")
        identity = (key, row["method"])
        if identity in seen:
            raise ValueError("duplicate screened method in paired timing case")
        seen.add(identity)
        baseline = full[key]
        if row.get("inputs_sha256") != baseline.get("inputs_sha256"):
            raise ValueError("paired full and screened policies have different candidate input hashes")
        measured = dict(config_id=key[0], group_id=key[1], paired_namespace=key[2], n=row["n"], k=row["k"],
            full_success=baseline["target_rate"], screened_success=row["target_rate"],
            success_difference=row["target_rate"]-baseline["target_rate"],
            full_all_target_trials_success=float(baseline["success"]),
            screened_all_target_trials_success=float(row["success"]),
            full_twin_calls=baseline["simulation_calls"]["twin_validation"],
            screened_twin_calls=row["simulation_calls"]["twin_validation"],
            full_requested_target_trials=baseline.get("requested_target_trials",1),
            screened_requested_target_trials=row.get("requested_target_trials",1),
            full_actual_target_calls=baseline["simulation_calls"].get("target_execution",0),
            screened_actual_target_calls=row["simulation_calls"].get("target_execution",0),
            timing_eligible=row["timing_eligible"] and baseline["timing_eligible"],
            timing_exclusion=dict(full=baseline["timing_exclusion"], screened=row["timing_exclusion"]))
        for boundary in ("total", "decision"):
            f = baseline["seconds"].get(boundary, baseline.get(f"{boundary}_seconds"))
            s = row["seconds"].get(boundary, row.get(f"{boundary}_seconds"))
            if measured["timing_eligible"] and f is not None and s is not None:
                if not math.isfinite(f) or not math.isfinite(s) or min(f, s) <= 0:
                    raise ValueError("comparison boundary must have positive measured time")
                measured.update({f"{boundary}_full_seconds":f, f"{boundary}_screened_seconds":s,
                    f"{boundary}_saved_seconds":f-s, f"{boundary}_speedup":f/s,
                    f"{boundary}_reduction_fraction":1-s/f})
        comparisons.setdefault(row["method"], []).append(measured)
    for name, rows in comparisons.items():
        if {(r["config_id"],r["group_id"],r["paired_namespace"]) for r in rows} != set(full):
            raise ValueError(f"{name}: incomplete matched full-pool timing cases")
    return dict(schema="twingraph.value.system_timing_audit.v5", source_sha256=digest(payload),
        interpretation="Ratios and differences use actual separate pipeline clocks, not N/K. Interpret speed jointly with independent target success. End-to-end and decision boundaries are distinct; parallel batch wall time is not a sum of per-policy times.",
        time_boundaries=payload.get("time_boundaries",payload["rows"][0].get("timing_boundaries")),
        methods={name:dict(rows=rows, summary=cluster_means(rows,
            sorted({key for row in rows for key in row if key not in {"config_id","group_id","paired_namespace","n","k","timing_exclusion"}}),
            bootstrap_samples, seed)) for name,rows in comparisons.items()})


def save(result, directory):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    path = directory/"metrics.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    with (directory/"summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("method", "stratum", "family", "k", "metric", "value", "ci95_low", "ci95_high", "denominator"))
        writer.writeheader()
        for name, method in result["methods"].items():
            for stratum, section in method["strata"].items():
                for family in ("classification_nominal", "classification_all_trials"):
                    if section[family] is None:
                        continue
                    for key in CLASS_METRICS:
                        ci = section[family]["bootstrap95"][key] or [None, None]
                        writer.writerow(dict(method=name, stratum=stratum, family=family, k="", metric=key,
                            value=section[family]["point"][key], ci95_low=ci[0], ci95_high=ci[1], denominator=section[family]["point"]["trials"]))
                for view, settings in section["screening"].items():
                    for k, setting in settings.items():
                        for key, metric in setting["summary"].items():
                            ci = metric["bootstrap95"] or [None, None]
                            writer.writerow(dict(method=name, stratum=stratum, family=f"screening_{view}", k=k,
                                metric=key, value=metric["mean"], ci95_low=ci[0], ci95_high=ci[1], denominator=metric["eligible_configurations"]))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("freeze")
    p.add_argument("--validation", required=True); p.add_argument("--out", required=True)
    p.add_argument("--k", nargs="+", type=int, default=[1, 2, 4]); p.add_argument("--primary-k", type=int, default=4)
    p = commands.add_parser("analyze")
    p.add_argument("--predictions", required=True); p.add_argument("--freeze", required=True); p.add_argument("--out", required=True)
    p.add_argument("--bootstrap-samples", type=int, default=2000); p.add_argument("--seed", type=int, default=17)
    p = commands.add_parser("timing")
    p.add_argument("--runs", required=True); p.add_argument("--out", required=True)
    p.add_argument("--bootstrap-samples", type=int, default=2000); p.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.command == "freeze":
        path = Path(args.out)
        if path.exists():
            raise ValueError("refuse to overwrite a validation freeze")
        result = freeze(json.loads(Path(args.validation).read_text()), args.k, args.primary_k)
        result["validation_file_sha256"] = file_sha(args.validation)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    elif args.command == "analyze":
        result = analyze(json.loads(Path(args.predictions).read_text()), json.loads(Path(args.freeze).read_text()), args.bootstrap_samples, args.seed)
        result["prediction_file_sha256"] = file_sha(args.predictions)
        result["freeze_file_sha256"] = file_sha(args.freeze)
        save(result, args.out)
    else:
        result = analyze_timing(json.loads(Path(args.runs).read_text()), args.bootstrap_samples, args.seed)
        path = Path(args.out); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
