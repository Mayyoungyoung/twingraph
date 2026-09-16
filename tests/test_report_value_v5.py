"""All values in these tests are synthetic fixtures, never reported experiments."""
import copy
import json

import pytest

from scripts import evaluate_value_v5 as evaluator
from scripts import report_value_v5 as report


def prediction(split,y):
    scores=[.1,.2,.3,.4,.6,.8]
    row=dict(group_id=split+"-group",config_id=split+"-config",candidate_ids=list("abcdef"),
        outcomes=[[x] for x in y],scores=scores,ranking_scores=scores,nominal_index=0,label_regime="nominal",
        input_diagnostics=[dict(unseen_fields=i==0,changed_training_constants=i==1,
                                broken_duplicate_relations=0,examples=[]) for i in range(6)])
    return dict(schema=evaluator.SCHEMA,split=split,selected="model",
        requested_config_ids=[split+"-config"],methods={"model":dict(model_sha256="a"*64,rows=[row])})


def test_majority_is_fixed_only_from_validation_even_if_test_majority_reverses():
    validation=prediction("validation",[0,0,0,0,1,1])
    test=prediction("test",[1,1,1,1,1,0])
    value=report.controls(validation,test)
    assert value["majority"]["chosen_label"]==0
    assert value["majority"]["test"]["accuracy"]==pytest.approx(1/6)


def test_input_order_top4_and_exact_random_include_setup_failure_denominator():
    validation=prediction("validation",[0,0,0,0,1,1])
    test=prediction("test",[0,0,0,0,0,1])
    test["requested_config_ids"].append("failed-setup")
    test["setup_failures"]=[dict(config_id="failed-setup",reason="synthetic fixture")]
    value=report.controls(validation,test)
    assert value["source_order"]["rows"][0]["selected_ids"]==list("abcd")
    assert value["source_order"]["summary"]["feasible_hit"]["mean"]==0
    assert value["random"]["summary"]["mean"]==pytest.approx(2/3)
    assert value["random"]["requested_summary"]["mean"]==pytest.approx(1/3)


def test_program_identity_ignores_ids_but_retains_every_waypoint():
    plan=dict(protocol="fixture",boundary=1,calls=[dict(id="a",skill="move",arguments={},roles={})],
              prefix=dict(initial_artifacts={"path":dict(type="joint_path",part=None,start_q=[0.],
                  joints=[[.1],[.2]],target=[0,0,0],rotation=[[1,0,0],[0,1,0],[0,0,1]],id="trace_id")}))
    other=copy.deepcopy(plan);other["calls"][0]["id"]="renamed"
    other["prefix"]["initial_artifacts"]["path"]["id"]="new_trace_id"
    assert report.program_identity(plan)==report.program_identity(other)
    other["prefix"]["initial_artifacts"]["path"]["joints"][1][0]=.21
    assert report.program_identity(plan)!=report.program_identity(other)


def test_summary_binds_metrics_and_keeps_pending_sections_and_diagnostics(tmp_path):
    validation=prediction("validation",[0,0,0,0,1,1]);test=prediction("test",[0,0,0,1,1,1])
    frozen=evaluator.freeze(validation);metrics=evaluator.analyze(test,frozen,bootstrap_samples=10)
    summary=report.build_summary(validation=validation,test=test,thresholds=frozen,metrics=metrics)
    assert summary["status"]=="evidence_pending"
    assert "independent_system_results" in summary["pending"]
    assert summary["models"]["model"]["input_diagnostics"]["test"]["changed_training_constants"]["affected_candidates"]==1
    assert "未完成真机验证" in report.markdown(summary)
    report.plot(summary,tmp_path/"synthetic_fixture_overview.png")
    assert (tmp_path/"synthetic_fixture_overview.png").stat().st_size>1000
    bad=copy.deepcopy(metrics);bad["predictions_sha256"]="b"*64
    with pytest.raises(ValueError,match="do not bind supplied test predictions"):
        report.build_summary(validation=validation,test=test,thresholds=frozen,metrics=bad)


def test_partial_raw_pool_is_not_classified_as_all_failure(tmp_path):
    group=tmp_path/"group_7";group.mkdir()
    plan=dict(id="a",protocol="fixture",boundary=1,calls=[dict(id="a",skill="move",arguments={},roles={})],prefix={})
    inp=dict(schema="twingraph.group.v5",group_id="g7",declared_split="train",nominal_index=0,candidates=[plan,dict(plan,id="b")])
    request=dict(seed=7,split="train",n=2)
    outcomes=dict(input_sha256=report.digest(inp),trials=[dict(candidate_id="a",trial={"repeat":0},
        success=False,valid=True,timeout=False,executed_parameters=[dict(skill="grasp",ok=False)])])
    for name,value in (("inputs.json",inp),("request.json",request),("outcomes.json",outcomes)):
        (group/name).write_text(json.dumps(value),encoding="utf-8")
    result=report.collection_audit([tmp_path])
    assert result["groups"][0]["status"]=="incomplete"
    assert result["splits"]["train"]["all_failure_pools"]==0
    assert result["splits"]["train"]["first_failure_skills"]=={"grasp":1}


def test_system_report_uses_actual_clocks_and_requested_target_denominator(tmp_path):
    isolation=dict(snapshot_transfer=False,distinct_session=True,distinct_context=True,distinct_model=True,distinct_data=True)
    cold=[]
    for policy,k,time,success in (("full",4,120.,True),("top_k",2,90.,False)):
        directory=tmp_path/policy;directory.mkdir()
        row=dict(schema="twingraph.system_run.v5",method=policy,policy=policy,config_id="case",group_id="case",
            paired_namespace={"twin":5107,"target":7901},config={"seed":10,"validation_repeats":1},
            n=4,k=k,actual_n=4,status="executed_independent_target",success=success,
            inputs_sha256="a"*64,seconds=dict(total=time+10,decision=time),requested_target_trials=1,
            target_successes=int(success),simulation_calls=dict(twin_validation=k,target_execution=1),
            validation=[{} for _ in range(k)],target_trials=[dict(status="executed",success=success,isolation=isolation)])
        (directory/"result.json").write_text(json.dumps(row),encoding="utf-8")
        updated=copy.deepcopy(row);load=10. if policy=="top_k" else 0.
        updated["model_initialization_wall_seconds"]=load
        updated["seconds"]["total"]+=load;updated["seconds"]["decision"]+=load
        cold.append(updated)
    (tmp_path/"timing_rows_cold_start.json").write_text(json.dumps(dict(rows=cold)),encoding="utf-8")
    result=report.system_audit([tmp_path])
    metrics=result["paired"]["methods"]["top_k"]["summary"]
    assert metrics["decision_speedup"]["mean"]==pytest.approx(120/90)
    assert metrics["decision_speedup"]["mean"]!=2  # Never use N/K as measured speed.
    assert result["methods"]["top_k"]["target_successes"]==0
    assert result["methods"]["top_k"]["requested_target_trials"]==1
    assert result["cold_start"]["paired"]["methods"]["top_k"]["summary"]["decision_speedup"]["mean"]==pytest.approx(1.2)
    cold[1]["seconds"]["total"]+=1
    (tmp_path/"timing_rows_cold_start.json").write_text(json.dumps(dict(rows=cold)),encoding="utf-8")
    with pytest.raises(ValueError,match="resident clock plus measured initialization"):
        report.system_audit([tmp_path])


def test_model_dimensions_come_from_bound_neighbor_or_relocated_summary(tmp_path):
    directory=tmp_path/"models/mlp_17";directory.mkdir(parents=True)
    summary=dict(parameters=3456,raw_dim=890, input_dim=123,checkpoint_sha256="a"*64)
    (directory/"summary.json").write_text(json.dumps(summary),encoding="utf-8")
    details=report.model_details("mlp_17",dict(path="/old/missing/mlp_17/best.pt",sha256="a"*64),[tmp_path/"models"])
    assert details["parameters"]==3456 and details["raw_input_dim"]==890 and details["retained_input_dim"]==123
    assert report.model_details("missing",dict(sha256="a"*64),[tmp_path/"models"])["status"]=="pending_summary_detail"
    with pytest.raises(ValueError,match="checkpoint binding mismatch"):
        report.model_details("mlp_17",dict(sha256="b"*64),[tmp_path/"models"])


def test_value_call_latency_uses_whole_pool_measurements_and_observed_phase_counts():
    rows=[dict(config_id="a",candidate_ids=list(range(12)),measured_rank_wall_seconds=.12,
               seconds=dict(graph_construction=.05,encoding=.03,inference=.01,export=.01,total=.10)),
          dict(config_id="b",candidate_ids=list(range(12)),measured_rank_wall_seconds=.20,
               seconds=dict(graph_construction=.09,encoding=.07,inference=.01,total=.18)),
          dict(config_id="c",candidate_ids=list(range(12)))]
    measured=report.value_call_latency(rows)
    assert measured["candidate_counts"]==[12] and measured["available_pool_rows"]==3
    assert measured["whole_call"]["observed_calls"]==2
    assert measured["whole_call"]["mean_seconds"]==pytest.approx(.16)
    assert measured["phases"]["export"]["observed_calls"]==1
    assert measured["phases"]["total"]["mean_seconds"]==pytest.approx(.14)
    assert report.value_call_latency([rows[-1]])["whole_call"]["mean_seconds"] is None
    rows[0]["measured_rank_wall_seconds"]=-1
    with pytest.raises(ValueError,match="finite nonnegative measured times"):
        report.value_call_latency(rows)
