"""Archive integrity and physical attrition tests; no assembly rollout needed."""
import copy
import json
from pathlib import Path

import pytest

from simbench.tests.test_materialized_graph import example
from simbench.value.plan import digest, materialize_initial_path
from simbench.value.skill_graph import compile_graph


def put(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def get(path):
    return json.loads(path.read_text())


def archive(root, seed=1, split="val", repeats=2, failed=False):
    directory = root/f"group_{seed}"
    directory.mkdir(parents=True)
    source = {"simbench/value/controller.py":"unchanged-controller-sha"}
    request = dict(seed=seed, split=split, domain="train", n=1, repeats=repeats,
                   timeout_seconds=240., source_sha256=digest(source), friction_span=.03, gain_span=.005)
    put(directory/"source.json",source); put(directory/"request.json",request)
    if failed:
        failure_request = {k:request[k] for k in ("seed","split","domain","n","repeats")}
        failure_request.update(out=str(root),timeout=request["timeout_seconds"])
        put(directory/"failure.json", dict(request=failure_request, phase="before_inputs",
            exception_type="ValueError",message="IK unreachable: physical approach",error="preserved traceback"))
        return directory
    session, observation, plan, path = example()
    plan = materialize_initial_path(session,plan,path)
    inputs = dict(schema="twingraph.group.v5",group_id=f"group-{seed}",split_group=f"config-{seed}",
        declared_split=split,task=dict(seed=seed),observation=observation,candidates=[plan.to_dict()],
        source_sha256=digest(source),nominal_index=0,checkpoint=0)
    graph = compile_graph(observation,plan)
    trials = []
    for repeat in range(repeats):
        trial = dict(repeat=repeat,domain="nominal" if repeat==0 else "train",
                     friction_scale=1. if repeat==0 else 1.01,actuator_gain_scale=1.)
        success = repeat==0
        trials.append(dict(candidate_id=plan.id,trial=trial,trial_sha256=digest(trial),
            valid=True,timeout=False,success=success,prefix_success=True,suffix_success=success,
            full_success=success,input_graph_sha256=digest(graph)))
    put(directory/"inputs.json",inputs)
    put(directory/"skill_graphs.json",dict(input_sha256=digest(inputs),graphs=[graph]))
    put(directory/"outcomes.json",dict(input_sha256=digest(inputs),trials=trials))
    put(directory/"complete.json",dict(request_sha256=digest(request),source_sha256=digest(source),
        input_sha256=digest(inputs),trials=len(trials),candidates=1))
    return directory


@pytest.fixture
def loader():
    pytest.importorskip("torch",reason="actual value_v5 loader shares a module with Torch model definitions")
    from simbench.value import value_v5
    return value_v5


def test_nominal_target_preserves_repeated_raw_outcomes_and_attrition(tmp_path,loader):
    archive(tmp_path,1);archive(tmp_path,2,failed=True)
    groups = loader.load_groups([tmp_path],("val",))
    assert len(groups)==1 and groups[0]["y"].tolist()==[1.]
    assert groups[0]["outcomes"]==[[1,0]]
    statuses = loader.request_dispositions([tmp_path],("val",))
    assert [r["status"] for r in statuses]==["completed","failed_before_inputs"]


@pytest.mark.parametrize("field",["input_hash","source_hash","request_hash","candidate_budget","repeat_budget","nominal_noise","graph_hash"])
def test_loader_rejects_corrupted_binding_budget_and_nominal_noise(tmp_path,loader,field):
    d=archive(tmp_path)
    if field in {"input_hash","source_hash","request_hash"}:
        complete=get(d/"complete.json")
        key={"input_hash":"input_sha256","source_hash":"source_sha256","request_hash":"request_sha256"}[field]
        complete[key]="corrupted";put(d/"complete.json",complete)
    elif field in {"candidate_budget","repeat_budget"}:
        request=get(d/"request.json")
        request["n" if field=="candidate_budget" else "repeats"]+=1
        complete=get(d/"complete.json");complete["request_sha256"]=digest(request)
        put(d/"request.json",request);put(d/"complete.json",complete)
    else:
        outcomes=get(d/"outcomes.json")
        if field=="nominal_noise":
            outcomes["trials"][0]["trial"]["friction_scale"]=1.001
            outcomes["trials"][0]["trial_sha256"]=digest(outcomes["trials"][0]["trial"])
        else:outcomes["trials"][0]["input_graph_sha256"]="another-plan"
        put(d/"outcomes.json",outcomes)
    with pytest.raises(ValueError):loader.load_groups([tmp_path],("val",))


@pytest.mark.parametrize("marker",["failure.json","censored.json"])
def test_completed_archive_cannot_also_be_failed_or_censored(tmp_path,loader,marker):
    d=archive(tmp_path);put(d/marker,{})
    with pytest.raises(ValueError,match="contradictory"):
        loader.request_dispositions([tmp_path],("val",))


@pytest.mark.parametrize("mutation",["phase","seed","programming_error"])
def test_preinput_failure_requires_physical_kind_phase_and_request_binding(tmp_path,loader,mutation):
    d=archive(tmp_path,failed=True);failure=get(d/"failure.json")
    if mutation=="phase":failure["phase"]="after_inputs"
    elif mutation=="seed":failure["request"]["seed"]=987
    else:failure.update(exception_type="TypeError",message="unexpected controller argument")
    put(d/"failure.json",failure)
    with pytest.raises(ValueError):loader.request_dispositions([tmp_path],("val",))


def test_requested_group_without_inputs_or_classified_failure_is_error(tmp_path,loader):
    d=archive(tmp_path,failed=True);(d/"failure.json").unlink()
    with pytest.raises(ValueError,match="incomplete"):
        loader.request_dispositions([tmp_path],("val",))


def test_training_loader_does_not_open_heldout_outcomes(tmp_path,loader):
    archive(tmp_path,1,split="train")
    heldout=archive(tmp_path,2,split="test")
    (heldout/"outcomes.json").write_text("HELDOUT OUTCOMES MUST NOT BE OPENED")
    assert len(loader.load_groups([tmp_path],("train",)))==1


def test_collection_worker_preserves_existing_archive_on_request_error(tmp_path,monkeypatch):
    from simbench.value import collect_v5
    d=archive(tmp_path);before={p.name:p.read_bytes() for p in d.iterdir()}
    def rejected(**kwargs):raise ValueError("existing collection request/source differs")
    monkeypatch.setattr(collect_v5,"collect_one",rejected)
    result=collect_v5.worker(dict(seed=1,out=str(tmp_path)))
    assert result["exception_type"]=="ValueError"
    assert {p.name:p.read_bytes() for p in d.iterdir()}==before


def test_collection_worker_preserves_original_failure_on_retry(tmp_path,monkeypatch):
    from simbench.value import collect_v5
    d=archive(tmp_path,failed=True);before=(d/"failure.json").read_bytes()
    def rejected(**kwargs):raise FileExistsError("preserve incomplete/failed attempt")
    monkeypatch.setattr(collect_v5,"collect_one",rejected)
    collect_v5.worker(dict(seed=1,out=str(tmp_path)))
    assert (d/"failure.json").read_bytes()==before


def test_allfail_validation_feasible_hit_has_zero_not_null(loader):
    from types import SimpleNamespace
    import numpy as np
    group=dict(id="g",config="c",plans=[SimpleNamespace(id="p")],y=np.asarray([0.]),
               outcomes=[[0]],nominal_index=0)
    summary=loader.validation_summary([group],[np.asarray([.8])])
    assert summary["rows"][0]["feasible_hit4"] is False
