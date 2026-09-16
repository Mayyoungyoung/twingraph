import copy
import numpy as np
import pytest

from simbench.tests.test_materialized_graph import example
from simbench.value.plan import materialize_initial_path
from simbench.value.program_input_v5 import InputSchema, leaves, program_record
from simbench.value.skill_graph import compile_graph


def record():
    session, observation, plan, path = example()
    plan = materialize_initial_path(session, plan, path)
    return observation, plan, program_record(compile_graph(observation, plan))


def test_complete_declared_ports_and_all_waypoints_survive_serialization():
    observation, plan, encoded = record()
    graph = compile_graph(observation, plan)
    for raw, call in zip(graph["nodes"], encoded["plan"]["calls"]):
        assert set(call["ports"]) == {p["name"] for p in raw["ports"]}
    before = leaves(encoded)
    for i in range(35):
        changed = copy.deepcopy(plan)
        changed.prefix["initial_artifacts"]["transfer"]["joints"][i][4] += .003
        after = leaves(program_record(compile_graph(observation, changed)))
        assert before != after, i


def test_trace_ids_and_consistent_object_renaming_are_not_predictor_inputs():
    observation, plan, before = record()
    graph = compile_graph(observation, plan)
    # Test canonical binding transform independently of executable PlanIR hash.
    def rename(value):
        if isinstance(value,dict):return {k:rename(v) for k,v in value.items()}
        if isinstance(value,list):return [rename(x) for x in value]
        if value=="part":return "renamed"
        return value
    changed=copy.deepcopy(graph)
    changed["observation"]["objects"]={"renamed":changed["observation"]["objects"]["part"]}
    changed["observation"]["goals"]=rename(changed["observation"]["goals"])
    for node in changed["nodes"]:
        node["roles"]=rename(node["roles"])
        for port in node["ports"]:port["value"]=rename(port["value"])
        for direction in ("reads","writes"):
            for ref in node[direction]:
                if ref["state"]=="object:part":ref["state"]="object:renamed"
    assert program_record(changed)==before
    renamed=copy.deepcopy(plan)
    renamed.id=renamed.prefix["id"]="new_trace_id"
    artifact=renamed.prefix["initial_artifacts"]["transfer"]
    artifact["binding"]["prefix_id"]=renamed.id;artifact["id"]="unrelated_trace"
    assert program_record(compile_graph(observation,renamed))==before


def test_input_schema_fit_has_no_label_or_heldout_dependency():
    training=[dict(x=float(i),alias=float(i),constant=7.,category="known") for i in range(3)]
    schema=InputSchema.fit(training)
    assert schema.saved["raw_dim"]==4
    assert schema.saved["dim"]==1
    x,unknown=schema.transform([dict(x=100.,alias=100.,constant=9.,category="unseen")])
    assert x.shape==(1,1) and x[0,0]==100.
    assert unknown and "unseen" in unknown[0][0]
    saved=copy.deepcopy(schema.saved)
    schema.transform([dict(x=-100.,validation_only=3.)])
    assert schema.saved==saved
    _,_,diagnostics=schema.transform([dict(x=101.,alias=100.,constant=9.,category="known")],return_diagnostics=True)
    assert diagnostics[0]["changed_training_constants"]==1
    assert diagnostics[0]["broken_duplicate_relations"]==1


def test_numeric_fields_are_raw_and_call_order_is_preserved():
    _,_,before=record()
    swapped=copy.deepcopy(before)
    swapped["plan"]["calls"][1],swapped["plan"]["calls"][2]=swapped["plan"]["calls"][2],swapped["plan"]["calls"][1]
    assert leaves(before)!=leaves(swapped)
    assert leaves(dict(force=3.,position=[.005]))["/force|number"]==3.
    assert leaves(dict(force=3.,position=[.005]))["/position/0|number"]==.005
    with pytest.raises(ValueError):leaves(dict(x=float("nan")))
