"""Prediction ordering and freeze integration with an input-only fake scorer."""
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import export_value_v5_predictions as exporter
from scripts.evaluate_value_v5 import analyze, freeze
from simbench.tests.test_value_v5_data import archive, get, put
from simbench.value.plan import PlanIR, digest


@pytest.fixture
def setup(tmp_path,monkeypatch):
    root=tmp_path/"data"
    val=archive(root,1,split="val",repeats=1)
    test=archive(root,3,split="test",repeats=1)
    failed=archive(root,4,split="test",repeats=1,failed=True)
    source=get(val/"source.json")
    monkeypatch.setattr(exporter,"source_manifest",lambda:copy.deepcopy(source))
    monkeypatch.setattr(exporter,"scene_config_id",lambda seed:f"config-{seed}")
    events=[]
    def dispositions(roots,splits):
        rows=[]
        for d in sorted(root.glob("group_*")):
            request=get(d/"request.json")
            if request["split"] in splits:
                rows.append(dict(directory=str(d),request=request,source_sha256=digest(source),
                    status="completed" if (d/"complete.json").exists() else "failed_before_inputs"))
        return rows
    monkeypatch.setattr(exporter,"dispositions_loader",dispositions)
    def load_groups(roots,splits):
        events.append("labels")
        expected=sum(1 for d in root.glob("group_*") if get(d/"request.json")["split"] in splits and (d/"inputs.json").exists())*2
        assert sum(e.startswith("score:") for e in events)==expected
        scores=list(tmp_path.glob("*.input_scores.json"))
        assert scores,"input-only scores must be durable before labels"
        rows=[]
        for r in dispositions(roots,splits):
            if r["status"]!="completed":continue
            d=Path(r["directory"]);inp=get(d/"inputs.json")
            rows.append(dict(id=inp["group_id"],config=inp["split_group"],input_sha256=digest(inp),
                plans=[PlanIR.from_dict(p) for p in inp["candidates"]],outcomes=[[1]],nominal_index=0,
                collected_source_sha256=digest(source)))
        return rows
    monkeypatch.setattr(exporter,"groups_loader",load_groups)
    paths={}
    split=dict(train=[dict(group_id="train",config_id="train-config",input_sha256="train-input")],
        val=[dict(group_id="group-1",config_id="config-1",input_sha256=digest(get(val/"inputs.json")))])
    for name in ("model_b","model_a"):
        d=tmp_path/name;d.mkdir();path=d/"best.pt";path.write_bytes(name.encode())
        put(d/"source.json",source);put(d/"split.json",split);paths[name]=str(path)
    def scorer(path,device):
        name=Path(path).parent.name
        saved=dict(schema="twingraph.value.v5",source_sha256=digest(source),split_sha256=digest(split),
            encoding_sha256="encoder",interface_sha256="interface",kind="mlp",seed=17,epoch=1)
        checkpoint_sha=exporter.sha(path)
        def rank(state,plans,k):
            events.append("score:"+name)
            return dict(scores=[.8],logits=[float(np.log(4))],order=[plans[0]["id"]],
                checkpoint_sha256=checkpoint_sha,seconds=dict(total=.002),input_diagnostics=[])
        return SimpleNamespace(saved=saved,checkpoint_sha256=checkpoint_sha,load_seconds=.001,rank=rank)
    monkeypatch.setattr(exporter,"scorer_factory",scorer)
    return SimpleNamespace(root=root,paths=paths,events=events,tmp=tmp_path)


def validation(setup):
    return exporter.export_predictions([setup.root],setup.paths,setup.tmp/"validation.json",split="val",
        expected_seeds=[1],n=1,primary_k=1,test_seeds=[3,4],selection_output=setup.tmp/"selection.json")


def test_scores_all_models_before_labels_and_validation_tie_is_stable(setup):
    predictions,selection=validation(setup)
    assert setup.events==["score:model_a","score:model_b","labels"]
    assert selection["selected"]==predictions["selected"]=="model_a"
    raw=get(setup.tmp/"validation.input_scores.json")
    assert all("outcomes" not in row for pools in raw["scores"].values() for row in pools.values())
    assert selection["test"]["seeds"]==[3,4]


def test_test_export_preserves_selection_and_all_requested_denominators(setup):
    predictions,selection=validation(setup);setup.events.clear()
    frozen=freeze(predictions,ks=(1,),primary_k=1)
    test,_=exporter.export_predictions([setup.root],setup.paths,setup.tmp/"test.json",split="test",selection=selection)
    assert setup.events==["score:model_a","score:model_b","labels"]
    assert test["requested_config_ids"]==["config-3","config-4"]
    assert test["setup_failures"][0]["config_id"]=="config-4"
    result=analyze(test,frozen,bootstrap_samples=20)
    pool=result["methods"]["model_a"]["strata"]["overall"]["screening"]["nominal"]["1"]
    assert pool["summary"]["feasible_hit"]["mean"]==1
    assert pool["requested_feasible_hit"]["mean"]==.5


def test_missing_entire_requested_configuration_fails_before_scoring(setup):
    with pytest.raises(ValueError,match="prospective seed manifest"):
        exporter.export_predictions([setup.root],setup.paths,setup.tmp/"bad.json",split="val",
            expected_seeds=[1,2],n=1,primary_k=1,test_seeds=[3],selection_output=setup.tmp/"bad_selection.json")
    assert setup.events==[]


def test_changed_model_fails_test_before_scoring(setup):
    _,selection=validation(setup);setup.events.clear()
    Path(setup.paths["model_a"]).write_bytes(b"changed")
    with pytest.raises(ValueError,match="checkpoint hash"):
        exporter.export_predictions([setup.root],setup.paths,setup.tmp/"badtest.json",split="test",selection=selection)
    assert setup.events==[]


def test_training_configuration_overlap_rejected_before_labels(setup):
    path=Path(setup.paths["model_a"]).parent/"split.json"
    split=get(path);split["train"][0]["config_id"]="config-1";put(path,split)
    # This also changes the sidecar hash; either mismatch must fail closed.
    with pytest.raises(ValueError,match="binding mismatch|overlaps"):
        validation(setup)
    assert "labels" not in setup.events


def test_no_overwrite_of_durable_scores_after_export_failure(setup,monkeypatch):
    def corrupted(*args):raise ValueError("corrupted outcomes")
    monkeypatch.setattr(exporter,"groups_loader",corrupted)
    with pytest.raises(ValueError,match="corrupted outcomes"):
        validation(setup)
    score_path=setup.tmp/"validation.input_scores.json"
    assert score_path.exists() and not (setup.tmp/"selection.json").exists()
    original=score_path.read_bytes()
    with pytest.raises(ValueError,match="overwrite"):
        validation(setup)
    assert score_path.read_bytes()==original


def test_real_torch_scorer_loader_and_exporter_smoke(tmp_path):
    """Synthetic integrity fixture, not a physical result or training experiment."""
    torch=pytest.importorskip("torch")
    from simbench.value.program_input_v5 import InputSchema, encoding_hash, program_record, source_manifest
    from simbench.value.skill_graph import interface_hash
    from simbench.value.value_v5 import ProgramMLP
    root=tmp_path/"real_api_data"
    source=source_manifest()
    directories=[archive(root,seed,split=split,repeats=1) for seed,split in ((1,"val"),(3,"test"))]
    for d in directories:
        request=get(d/"request.json");request["source_sha256"]=digest(source)
        inputs=get(d/"inputs.json");inputs["source_sha256"]=digest(source)
        complete=get(d/"complete.json");complete.update(source_sha256=digest(source),
            request_sha256=digest(request),input_sha256=digest(inputs))
        outcomes=get(d/"outcomes.json");outcomes["input_sha256"]=digest(inputs)
        graphs=get(d/"skill_graphs.json");graphs["input_sha256"]=digest(inputs)
        for name,value in (("request",request),("source",source),("inputs",inputs),
                           ("complete",complete),("outcomes",outcomes),("skill_graphs",graphs)):
            put(d/f"{name}.json",value)
    record=program_record(get(directories[0]/"skill_graphs.json")["graphs"][0])
    changed=copy.deepcopy(record);changed["state"]["robot"]["joints"][0]=.001
    schema=InputSchema.fit([record,changed])
    model=ProgramMLP(len(schema.keys),"linear")
    with torch.no_grad():
        for parameter in model.parameters():parameter.zero_()
    d=tmp_path/"actual_api_model";d.mkdir()
    split=dict(train=[dict(group_id="synthetic-training",config_id="independent-training",input_sha256="synthetic")],
        val=[dict(group_id="group-1",config_id="config-1",input_sha256=digest(get(directories[0]/"inputs.json")))])
    put(d/"source.json",source);put(d/"split.json",split)
    checkpoint=d/"best.pt"
    torch.save(dict(schema="twingraph.value.v5",kind="linear",seed=17,epoch=1,
        source_sha256=digest(source),split_sha256=digest(split),encoding_sha256=encoding_hash(),
        interface_sha256=interface_hash(),input_schema=schema.saved,state_dict=model.state_dict()),checkpoint)
    paths={"actual_api_model":str(checkpoint)}
    validation,selection=exporter.export_predictions([root],paths,tmp_path/"val_real.json",split="val",
        expected_seeds=[1],n=1,primary_k=1,test_seeds=[3],selection_output=tmp_path/"real_selection.json")
    test,_=exporter.export_predictions([root],paths,tmp_path/"test_real.json",split="test",selection=selection)
    assert validation["methods"]["actual_api_model"]["rows"][0]["scores"]==[.5]
    result=analyze(test,freeze(validation,ks=(1,),primary_k=1),bootstrap_samples=20)
    assert result["methods"]["actual_api_model"]["strata"]["overall"]["classification_nominal"]["point"]["accuracy"]==1.
