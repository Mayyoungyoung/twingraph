"""Analytical ranking controls and configuration-level uncertainty regression."""
import itertools
import json
from types import SimpleNamespace

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


def test_saturated_probabilities_use_logits_for_ranking_only():
    payload = {"methods":{"calibrated":[dict(group_id="g",scores=[1.,1.],
        ranking_scores=[30.,40.],reference=[0.,1.],probability_metrics=True)],
        "source_order":[dict(group_id="g",scores=[0.,-1.],reference=[0.,1.],probability_metrics=False)]}}
    result = analyze(payload,bootstrap_samples=100)
    learned = result["methods"]["calibrated"]
    assert learned["summary"]["hit1"]["mean"] == 1.
    assert learned["summary"]["brier"]["mean"] == .5
    assert learned["rows"][0]["numerically_flat"] is True
    assert learned["rows"][0]["ranking_score_spread"] == 10.
    assert result["methods"]["source_order"]["summary"]["brier"]["mean"] is None
    assert result["methods"]["source_order"]["summary"]["log_loss"]["mean"] is None


def _fake_freeze(monkeypatch, tmp_path, alternate=True):
    from scripts import evaluate_value_v4 as evaluator
    root = tmp_path/"models"
    for name in ("model_a","model_b"):
        path = root/name/"best.pt"
        path.parent.mkdir(parents=True)
        path.write_bytes(name.encode())
    def load(path, **kwargs):
        return dict(schema="twingraph.value.schema.v1",interface_sha256="interface",kind="compact",
                    seed=17,epoch=1,source_sha256="source",split_sha256="same-split",
                    validation=dict(hit4=None,quality4=0.,hit1=None,brier=.25),
                    validation_calibrated=dict(hit4=None,quality4=0.,hit1=None,
                        brier=.1 if path.parent.name == "model_b" else .2))
    monkeypatch.setattr(evaluator,"torch",SimpleNamespace(load=load))
    monkeypatch.setattr(evaluator,"interface_hash",lambda:"interface")
    monkeypatch.setattr(evaluator,"implementation",lambda:dict(sha256="script-source",files={}))
    freeze = evaluator.freeze(root,tmp_path/"frozen.json",list(range(51300,51308)),
        checkpoints=[2,3],n=8,alternate_checkpoints=alternate)
    return evaluator, freeze


def test_freeze_all_failure_validation_and_alternating_checkpoints(monkeypatch,tmp_path):
    evaluator, frozen = _fake_freeze(monkeypatch,tmp_path)
    assert frozen["selected"] == "model_b"
    assert frozen["test"]["n"] == 8 and frozen["test"]["repeats"] == 2
    assert [(t["seed"],t["checkpoint"]) for t in frozen["test"]["tasks"]] == [
        (seed,2+seed%2) for seed in range(51300,51308)]
    assert frozen["deployment"]["tasks"] == frozen["test"]["tasks"][:4]
    evaluator.verify_freeze(frozen)
    path = tmp_path/"models"/"model_a"/"best.pt"
    path.write_bytes(b"changed after freeze")
    with pytest.raises(ValueError,match="weights changed"):
        evaluator.verify_freeze(frozen)


def test_input_only_test_manifest_requires_exact_assigned_pairs(monkeypatch,tmp_path):
    evaluator, frozen = _fake_freeze(monkeypatch,tmp_path)
    root = tmp_path/"data"
    for task in frozen["test"]["tasks"]:
        path = root/f"group_{task['seed']}_{task['checkpoint']}"/"inputs.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(dict(declared_split="test",task=dict(
            family=frozen["test"]["family"],seed=task["seed"]),checkpoint=task["checkpoint"])))
        (path.parent/"complete.json").write_text("{}")
        # If the benchmark manifest reader touches a label this cannot parse.
        (path.parent/"outcomes.json").write_text("OUTCOMES MUST NOT BE OPENED")
    assert len(evaluator.test_inputs([root],frozen)) == 8
    changed = root/"group_51301_3"/"inputs.json"
    payload = json.loads(changed.read_text());payload["checkpoint"] = 2
    changed.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match="exactly match"):
        evaluator.test_inputs([root],frozen)


def test_non_alternating_freeze_preserves_cartesian_design(monkeypatch,tmp_path):
    _, frozen = _fake_freeze(monkeypatch,tmp_path,alternate=False)
    assert len(frozen["test"]["tasks"]) == 16
    assert frozen["deployment"]["tasks"] == [dict(seed=seed,checkpoint=2) for seed in range(51300,51304)]


def test_checkpoint_summary_exposes_length_strata():
    rows=[dict(group_id="cp2",config_id="a",checkpoint=2,plan_lengths=[58,58],scores=[.9,.1],reference=[1,0]),
          dict(group_id="cp3",config_id="b",checkpoint=3,plan_lengths=[40,40],scores=[.9,.1],reference=[0,1])]
    method=analyze({"methods":{"m":rows}},bootstrap_samples=100)["methods"]["m"]
    assert method["summary"]["hit1"]["mean"]==.5
    assert method["by_checkpoint"]["2"]["summary"]["hit1"]["mean"]==1.
    assert method["by_checkpoint"]["3"]["summary"]["hit1"]["mean"]==0.
    assert method["rows"][0]["plan_lengths"]==[58,58]


def test_manifest_retains_physical_setup_failure_but_rejects_missing_and_bugs(monkeypatch,tmp_path):
    evaluator,frozen=_fake_freeze(monkeypatch,tmp_path)
    root=tmp_path/"data"
    for task in frozen["test"]["tasks"]:
        directory=root/f"group_{task['seed']}_{task['checkpoint']}"
        directory.mkdir(parents=True)
        failed=dict(request=dict(seed=task["seed"],checkpoint=task["checkpoint"],split="test"),
            error='Traceback (most recent call last):\n  File "schema_collect.py", line 1, in create_scene\nValueError: IK unreachable: distance')
        (directory/"failure.json").write_text(json.dumps(failed))
    manifest=evaluator.test_inputs([root],frozen)
    assert len(manifest)==8 and all(r["disposition"]=="checkpoint_setup_failed" for r in manifest)
    path=root/"group_51301_3"/"failure.json"
    changed=json.loads(path.read_text());changed["error"]='File "schema_collect.py", line 1, in create_scene\nTypeError: bad argument'
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match="not an admissible"):
        evaluator.test_inputs([root],frozen)
    path.unlink()
    with pytest.raises(ValueError,match="exactly match"):
        evaluator.test_inputs([root],frozen)


def test_deployment_setup_failure_is_zero_success_and_never_resampled(monkeypatch,tmp_path):
    evaluator,frozen=_fake_freeze(monkeypatch,tmp_path)
    before=json.dumps(frozen,sort_keys=True)
    monkeypatch.setattr(evaluator,"make_scorer",lambda *args:object())
    attempts=[]
    def fail(seed,directory,checkpoint):
        attempts.append((seed,checkpoint))
        raise ValueError("IK unreachable: genuine robot setup")
    monkeypatch.setattr(evaluator,"create_scene",fail)
    task=frozen["deployment"]["tasks"][1]
    out=tmp_path/"deployment"
    result=evaluator.deployment(frozen,out,"cpu",selected_tasks=[task],write_summary=False)
    assert len(result["decisions"])==4
    for row in result["decisions"]:
        assert row["status"]=="checkpoint_setup_failed"
        assert row["deployment_success_rate"]==0. and row["requested_deployment_attempts"]==2
        assert row["validation_calls"]==0
        assert row["online_budget"]==(16 if row["method"]=="exhaustive" else 8)
    evaluator.deployment(frozen,out,"cpu",selected_tasks=[task],write_summary=False)
    assert attempts==[(51301,3)]
    assert json.dumps(frozen,sort_keys=True)==before


def test_deployment_does_not_swallow_implementation_errors(monkeypatch,tmp_path):
    evaluator,frozen=_fake_freeze(monkeypatch,tmp_path)
    monkeypatch.setattr(evaluator,"make_scorer",lambda *args:object())
    def broken(*args):
        raise TypeError("unexpected keyword in implementation")
    monkeypatch.setattr(evaluator,"create_scene",broken)
    with pytest.raises(TypeError,match="implementation"):
        evaluator.deployment(frozen,tmp_path/"deployment","cpu",selected_tasks=frozen["deployment"]["tasks"][:1])


def test_setup_solver_exhaustion_match_is_narrow():
    from scripts.evaluate_value_v4 import admissible_setup_error
    message="no materialized candidate; expand solver budget or verify unknowns"
    assert admissible_setup_error(ValueError(message))
    assert admissible_setup_error('File "schema_collect.py", line 1, in create_scene\nValueError: '+message,require_trace=True)
    assert not admissible_setup_error('ValueError: '+message,require_trace=True)
    assert not admissible_setup_error(ValueError("no materialized candidate; corrupt controller argument"))
