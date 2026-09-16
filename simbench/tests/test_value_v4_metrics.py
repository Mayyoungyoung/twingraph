"""Analytical ranking controls and configuration-level uncertainty regression."""
import itertools

import numpy as np
import pytest

from scripts.analyze_value_v4 import (
    analyze, cluster_summary, exact_uniform_random, save_outputs, score_diagnostics,
)


def test_exact_random_matches_enumeration_with_noisy_labels():
    y = np.array([0., .5, .9, 1.])
    for k in (1, 2, 3, 4, 8):
        subsets = list(itertools.combinations(range(len(y)), min(k, len(y))))
        expected = exact_uniform_random(y, k)
        assert expected["hit"] == pytest.approx(np.mean([y[list(ix)].max() >= .9 for ix in subsets]))
        assert expected["feasible"] == pytest.approx(np.mean([y[list(ix)].max() > 0 for ix in subsets]))
        assert expected["quality"] == pytest.approx(np.mean([y[list(ix)].mean() for ix in subsets]))


def test_all_failure_is_kept_except_near_best_denominator():
    payload = {"methods":{"model":[
        dict(group_id="failure", config_id="a", scores=[.8,.3], reference=[0,0]),
        dict(group_id="mixed", config_id="b", scores=[.8,.3], reference=[1,0]),
    ]}}
    result = analyze(payload, bootstrap_samples=100)["methods"]["model"]
    assert result["groups"] == 2
    assert result["all_failure_groups"] == 1
    assert result["summary"]["hit1"]["mean"] == 1.
    assert result["summary"]["hit1"]["eligible_configurations"] == 1
    assert result["summary"]["feasible1"]["mean"] == .5
    assert result["summary"]["quality1"]["mean"] == .5
    assert result["summary"]["random_hit1"]["mean"] == .5
    assert result["summary"]["random_feasible1"]["mean"] == .25
    assert len(result["rows"]) == 2


def test_checkpoint_siblings_are_not_independent_bootstrap_units():
    original = [dict(config_id="a", quality=1), dict(config_id="b", quality=0)]
    duplicated = [*original, *[dict(config_id="a", quality=1) for _ in range(12)]]
    a = cluster_summary(original, "quality", bootstrap_samples=500)
    b = cluster_summary(duplicated, "quality", bootstrap_samples=500)
    assert a["mean"] == b["mean"] == .5
    assert a["bootstrap95"] == b["bootstrap95"]
    assert a["eligible_configurations"] == b["eligible_configurations"] == 2


def test_tie_tolerance_is_diagnostic_and_never_changes_ranking():
    rows = [dict(group_id="g", seed=3, scores=[.5,.5000001], reference=[0,1])]
    result = analyze({"methods":{"m":rows}}, tolerance=1e-6, bootstrap_samples=100)
    row = result["methods"]["m"]["rows"][0]
    assert row["numerically_flat"] is True
    assert row["boundary_near_tie1"] is True
    assert row["hit1"] is True  # rounding/re-ranking would choose index zero
    assert row["score_spread"] == pytest.approx(1e-7)
    assert score_diagnostics([.5], 4)["boundary_near_tie"] is None


def test_v3_format_and_outputs(tmp_path):
    payload = {"freeze_sha256":"original", "results":{"m":{"rows":[
        dict(group_id="cp0",seed=3,scores=[1,0],reference=[1,0]),
        dict(group_id="cp1",seed=3,scores=[0,1],reference=[1,0]),
    ]}}}
    result = analyze(payload, bootstrap_samples=100)
    assert result["methods"]["m"]["configurations"] == 1
    assert result["methods"]["m"]["summary"]["hit1"]["mean"] == .5
    assert result["source_metadata"]["freeze_sha256"] == "original"
    json_path, csv_path = save_outputs(result, tmp_path)
    assert json_path.exists() and csv_path.exists()
    assert "eligible_configurations" in csv_path.read_text()


def test_mismatched_pool_and_invalid_scores_are_rejected():
    row = dict(group_id="g",config_id="a",scores=[1,0],reference=[1,0])
    with pytest.raises(ValueError, match="different reference pools"):
        analyze({"methods":{"a":[row], "b":[dict(row,reference=[0,1])]}})
    with pytest.raises(ValueError, match="cardinality"):
        analyze({"methods":{"a":[dict(row,scores=[1])]}})
    with pytest.raises(ValueError, match="finite"):
        exact_uniform_random([float("nan")], 1)


def test_no_successful_configuration_has_null_hit_interval():
    result = analyze({"methods":{"m":[dict(group_id="g",scores=[0],reference=[0])]}})
    summary = result["methods"]["m"]["summary"]["hit1"]
    assert summary == dict(mean=None,bootstrap95=None,eligible_configurations=0,eligible_groups=0)
