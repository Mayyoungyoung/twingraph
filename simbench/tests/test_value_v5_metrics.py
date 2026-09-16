"""Nominal correctness, noisy repetitions, screening and actual-clock controls."""
import copy
import itertools
import json

import numpy as np
import pytest

from scripts.evaluate_value_v5 import (
    SCHEMA, analyze, analyze_timing, binary_metrics, cluster_means, freeze,
    random_screening, save, select_threshold, validate_predictions,
)


def payload(split="validation", *, outcomes=None, scores=None, config=None):
    outcomes = outcomes or [[1], [0], [1], [0]]
    row = dict(group_id=f"{split}-g", config_id=config or split, candidate_ids=list("abcd"),
               scores=scores or [.8, .2, .6, .4], outcomes=outcomes,
               label_regime="nominal" if len(outcomes[0]) == 1 else "repeated", nominal_index=0,
               checkpoint=0)
    return dict(schema=SCHEMA, split=split, methods={"model":dict(model_sha256="a"*64,rows=[row])})


def test_binary_confusion_and_tie_correct_auc_average_precision():
    metric = binary_metrics([1, 0, 1, 0], [.8, .8, .2, .1], .5)
    assert [metric[k] for k in ("tp", "fp", "tn", "fn")] == [1, 1, 1, 1]
    assert metric["accuracy"] == metric["balanced_accuracy"] == .5
    assert metric["roc_auc"] == pytest.approx(.625)
    assert metric["average_precision"] == pytest.approx(.5*.5 + .5*2/3)
    tied = binary_metrics([1, 0, 1, 0], [.5]*4, .5)
    assert tied["roc_auc"] == tied["average_precision"] == .5
    assert tied["recall"] == 1


def test_one_class_undefined_metrics_are_null_not_perfect():
    m = binary_metrics([0, 0], [.2, .1], .5)
    assert m["accuracy"] == 1
    assert m["balanced_accuracy"] is m["roc_auc"] is m["average_precision"] is m["precision"] is m["recall"] is None
    rows = validate_predictions(payload(outcomes=[[0]]*4),"validation")["model"]["rows"]
    assert select_threshold(rows)["threshold"] == .5


def test_exact_random_all_screening_metrics_match_subsets():
    y = np.array([0, .5, 1, 0, 1.])
    for k in (1, 2, 4, 5, 8):
        selections = [y[list(indices)] for indices in itertools.combinations(range(len(y)),min(k,len(y))) ]
        result = random_screening(y, k)
        expected = dict(feasible_precision=np.mean([np.mean(s>0) for s in selections]),
            feasible_recall=np.mean([np.sum(s>0)/3 for s in selections]),
            feasible_hit=np.mean([np.any(s>0) for s in selections]),
            near_best_hit=np.mean([np.max(s)>=.9 for s in selections]),
            quality=np.mean([np.mean(s) for s in selections]),
            best_quality=np.mean([np.max(s) for s in selections]),
            regret=np.mean([1-np.max(s) for s in selections]))
        for key, value in expected.items():
            assert result[key] == pytest.approx(value)


def test_all_failure_denominators_and_requested_setup_failure():
    locked = freeze(payload())
    test = payload("test", outcomes=[[0]]*4)
    test.update(requested_config_ids=["test","failed"], setup_failures=[dict(config_id="failed",reason="physical setup failure")])
    result = analyze(test, locked, bootstrap_samples=20)
    pool = result["methods"]["model"]["strata"]["overall"]["screening"]["nominal"]["4"]
    assert pool["all_failure_groups"] == 1
    assert pool["summary"]["near_best_hit"]["mean"] is None
    assert pool["summary"]["feasible_recall"]["mean"] is None
    assert pool["summary"]["feasible_hit"]["mean"] == 0
    assert pool["summary"]["quality"]["eligible_groups"] == 1
    assert pool["requested_feasible_hit"]["eligible_configurations"] == 2
    assert result["disposition"]["setup_failed_configurations"] == 1


def test_nominal_and_repeated_correctness_are_separate():
    locked = freeze(payload())
    test = payload("test", outcomes=[[1,0], [0,1], [1,0], [0,1]])
    overall = analyze(test, locked, bootstrap_samples=20)["methods"]["model"]["strata"]["overall"]
    assert overall["classification_nominal"]["point"]["accuracy"] == 1
    assert overall["classification_nominal"]["point"]["trials"] == 4
    assert overall["classification_all_trials"]["point"]["accuracy"] == .5
    assert overall["classification_all_trials"]["point"]["trials"] == 8
    assert overall["screening"]["nominal"]["1"]["summary"]["quality"]["mean"] == 1
    assert overall["screening"]["empirical_repeats"]["1"]["summary"]["quality"]["mean"] == .5
    assert overall["empirical_fraction_brier"]["mean"] == pytest.approx(.05)
    assert overall["classification_all_trials"]["point"]["brier"] == pytest.approx(.30)


def test_repeats_require_explicit_nominal_index():
    p = payload(outcomes=[[1,0], [0,1], [1,0], [0,1]])
    del p["methods"]["model"]["rows"][0]["nominal_index"]
    with pytest.raises(ValueError, match="nominal_index"):
        freeze(p)


def test_freeze_cannot_use_test_or_overlap_and_hash_drift_fails():
    with pytest.raises(ValueError, match="never select from test"):
        freeze(payload("test"))
    locked = freeze(payload())
    with pytest.raises(ValueError, match="overlaps validation"):
        analyze(payload("test",config="validation"), locked)
    test = payload("test")
    test["methods"]["model"]["model_sha256"] = "b"*64
    with pytest.raises(ValueError, match="hash mismatch"):
        analyze(test, locked)


def test_threshold_frozen_nominal_validation_and_invariant_to_test_labels():
    val = payload(scores=[.4,.1,.3,.2])
    locked = freeze(val)
    assert locked["methods"]["model"]["threshold"] == .3
    good = analyze(payload("test",scores=[.4,.1,.3,.2]),locked,20)
    opposite = analyze(payload("test",scores=[.4,.1,.3,.2],outcomes=[[0],[1],[0],[1]]),locked,20)
    assert good["methods"]["model"]["threshold"] == opposite["methods"]["model"]["threshold"] == .3
    assert opposite["methods"]["model"]["strata"]["overall"]["classification_nominal"]["point"]["accuracy"] == 0


def test_saturated_probabilities_rank_by_logits_without_rounding():
    val = payload()
    locked = freeze(val)
    test = payload("test",scores=[1,1,1,1])
    test["methods"]["model"]["rows"][0]["ranking_scores"] = [30,40,31,35]
    result = analyze(test, locked,20)["methods"]["model"]
    pool = result["strata"]["overall"]["screening"]["nominal"]["1"]
    assert pool["rows"][0]["selected_ids"] == ["b"]
    assert pool["summary"]["feasible_hit"]["mean"] == 0
    assert result["score_diagnostics"][0]["probability_spread"] == 0


def test_same_physical_configuration_is_single_bootstrap_cluster():
    base = [dict(config_id="a", quality=1.), dict(config_id="b", quality=0.)]
    extra = base + [dict(config_id="a", quality=1.)]*10
    a, b = [cluster_means(rows,["quality"],100)["quality"] for rows in (base,extra)]
    assert a["mean"] == b["mean"] == .5
    assert a["bootstrap95"] == b["bootstrap95"]
    assert b["eligible_configurations"] == 2


def test_cross_method_candidate_binding_and_duplicates_rejected():
    p = payload()
    p["methods"]["other"] = copy.deepcopy(p["methods"]["model"])
    p["methods"]["other"]["rows"][0]["candidate_ids"] = list("abce")
    with pytest.raises(ValueError,match="identical candidate pools"):
        freeze(p)
    p = payload()
    p["methods"]["model"]["rows"][0]["candidate_ids"] = list("aabc")
    with pytest.raises(ValueError,match="duplicate"):
        freeze(p)


def timing_payload():
    rows = []
    for config in ("a", "b"):
        for method, policy, k, seconds, success in (("full","full",12,100.,True), ("model","top_k",4,60.,False)):
            rows.append(dict(config_id=config,group_id=config,method=method,policy=policy,n=12,k=k,
                simulation_calls=dict(twin_validation=k,target_execution=1),success=success,
                seconds=dict(total=seconds,decision=seconds-10),paired_namespace="paired_v5"))
    return dict(schema="twingraph.system_run.v5",rows=rows,
                time_boundaries=dict(total="scene creation through independent target terminal render"))


def test_actual_timing_ratio_is_not_candidate_count_ratio():
    result = analyze_timing(timing_payload(),20)["methods"]["model"]["summary"]
    assert result["total_speedup"]["mean"] == pytest.approx(100/60)
    assert result["total_speedup"]["mean"] != 12/4
    assert result["total_reduction_fraction"]["mean"] == pytest.approx(.4)
    assert result["success_difference"]["mean"] == -1
    assert result["full_twin_calls"]["mean"] == 12


def test_timing_missing_pair_or_incomplete_validation_is_error():
    p = timing_payload(); p["rows"].pop()
    with pytest.raises(ValueError,match="incomplete matched"):
        analyze_timing(p)
    p = timing_payload(); p["rows"][0]["simulation_calls"]["twin_validation"] = 4
    with pytest.raises(ValueError,match="declared candidate validation budget"):
        analyze_timing(p)


def test_timing_partial_pool_and_setup_failures_stay_in_success_denominator():
    p = timing_payload()
    for row in p["rows"][:2]:
        row.update(status="twin_setup_failed", success=False, actual_n=0,
                   target_successes=0, requested_target_trials=2)
        row["simulation_calls"] = dict(twin_validation=0,target_execution=0)
    result = analyze_timing(p,20)["methods"]["model"]
    assert result["summary"]["full_success"]["eligible_configurations"] == 2
    assert result["summary"]["full_success"]["mean"] == .5
    assert result["summary"]["total_speedup"]["eligible_configurations"] == 1
    assert result["rows"][0]["timing_exclusion"]["full"] == "twin_setup_failed"
    p = timing_payload()
    for row in p["rows"]:
        row["actual_n"] = 8
        row["simulation_calls"]["twin_validation"] = min(row["k"],8)
    result = analyze_timing(p,20)["methods"]["model"]
    assert result["summary"]["timing_eligible"]["mean"] == 0
    assert "total_speedup" not in result["summary"]


def test_repeated_independent_target_rate_not_all_success_flag():
    p = timing_payload()
    for row in p["rows"]:
        row.update(success=False, requested_target_trials=2, target_successes=1)
    result = analyze_timing(p,20)["methods"]["model"]["summary"]
    assert result["full_success"]["mean"] == result["screened_success"]["mean"] == .5
    assert result["full_all_target_trials_success"]["mean"] == 0


def test_selected_model_and_source_binding_cannot_change_after_freeze():
    val = payload(); val.update(selected="model",source_sha256="source")
    locked = freeze(val)
    test = payload("test"); test["source_sha256"] = "source"
    with pytest.raises(ValueError,match="selected deployment"):
        analyze(test,locked,20)
    test.update(selected="model",source_sha256="changed")
    with pytest.raises(ValueError,match="source binding"):
        analyze(test,locked,20)


def test_json_csv_outputs_preserve_nulls_and_confusion(tmp_path):
    result = analyze(payload("test"),freeze(payload()),20)
    path = save(result,tmp_path)
    restored = json.loads(path.read_text())
    assert restored["methods"]["model"]["strata"]["overall"]["classification_nominal"]["point"]["tp"] == 2
    assert "classification_nominal" in (tmp_path/"summary.csv").read_text()
