"""Contract tests only; mock twin results are not physical dataset labels."""
import copy

import pytest

from simbench.value.atomic_flow import (
    RESPONSE_SCHEMA, compile_candidates, compose_from_template, planner_request, screen_and_verify,
)
from simbench.value.generic_graph_value_v15 import encode_graph
from simbench.value.codex_planner import decode_model_response
from simbench.value.plan import digest


def fixture():
    observation = dict(robot=dict(joints=[0.] * 7, eef=[0., 0., .9]),
        objects=dict(block=dict(position=[.1, .2, .8], valid=True,
                                capabilities=[])),
        goals=[dict(predicate="supported_on", object="block", target="platform")])
    calls = [dict(skill="detect", params=dict(required_parts=["block"]),
                  roles=dict(manipulated="block")),
             dict(skill="estimate_pose", params=dict(part="block", as_="pose"),
                  roles=dict(manipulated="block"))]
    response = dict(schema=RESPONSE_SCHEMA,
                    observation_sha256=digest(observation),
                    plans=[dict(name="first", calls=calls),
                           dict(name="second", calls=[calls[0],
                               dict(skill="estimate_pose", params=dict(part="block", as_="alternate_pose"),
                                    roles=dict(manipulated="block"))])])
    return observation, response


def test_request_exposes_generic_registered_skills_without_task_templates():
    observation, _ = fixture()
    request = planner_request(dict(instruction="Place block on platform"),
                              observation, candidate_count=2)
    assert "estimate_grasp" in request["atomic_skills"]
    assert "plan_path" in request["atomic_skills"]
    assert "place" in request["atomic_skills"]
    assert request["atomic_skills"]["estimate_pose"]["implementations"].keys() == {"default"}
    assert "carriage" not in str(request["atomic_skills"])


def test_llm_atomic_calls_compile_to_the_same_validated_value_input():
    observation, response = fixture()
    candidates = compile_candidates(response, observation, expected_count=2)
    assert len(candidates) == 2
    assert candidates[0]["plan"].protocol == "atomic.program.v1"
    assert candidates[0]["graph"]["assembly"]["plan_sha256"] == candidates[0]["plan_sha256"]
    assert encode_graph(candidates[0]["graph"])["x"].shape[0] == 2


def test_reject_stale_observation_unknown_skill_and_duplicate_executable_plan():
    observation, response = fixture()
    stale = copy.deepcopy(response); stale["observation_sha256"] = "old"
    with pytest.raises(ValueError, match="observation binding"):
        compile_candidates(stale, observation)
    unknown = copy.deepcopy(response); unknown["plans"][0]["calls"][0]["skill"] = "assemble_carriage"
    with pytest.raises(ValueError, match="unregistered public"):
        compile_candidates(unknown, observation)
    repeated = copy.deepcopy(response); repeated["plans"][1]["calls"] = repeated["plans"][0]["calls"]
    with pytest.raises(ValueError, match="duplicate executable"):
        compile_candidates(repeated, observation)


def test_value_top_k_only_reaches_verifier_and_all_failed_remains_explicit():
    observation, response = fixture()
    candidates = compile_candidates(response, observation)
    class Ranker:
        def score(self, graphs):
            assert len(graphs) == 2
            return [.1, .9]
    visited = []
    def twin(plan, graph):
        visited.append(plan.id)
        return dict(valid=True, success=False, evaluation_scope="supported_on",
                    input_graph_sha256=digest(graph))
    result = screen_and_verify(candidates, Ranker(), twin, k=1,
                               evaluation_scope="supported_on")
    assert result["ranking"][0]["name"] == "second"
    assert visited == [candidates[1]["plan"].id]
    assert result["selected"] is None
    assert result["status"] == "top_k_all_failed"


def test_model_transport_decodes_full_calls_without_task_specific_slots():
    observation, response = fixture()
    raw = dict(observation_sha256=digest(observation), plans=[
        dict(name=row["name"], calls=[dict(skill=call["skill"],
            manipulated=call["roles"]["manipulated"],
            params_json=__import__("json").dumps(call["params"]))
            for call in row["calls"]]) for row in response["plans"]])
    decoded, candidates = decode_model_response(raw, observation, expected_count=2)
    assert decoded == response
    assert len(candidates) == 2


def test_model_can_compose_grounded_port_alternatives_on_generic_template():
    observation, response = fixture()
    base = compile_candidates(response, observation)[0]["plan"]
    composed = compose_from_template(base, observation,
        [dict(name="pose_a", edits=[]),
         dict(name="pose_b", edits=[dict(call_id="call_001", argument="as_", value="pose_b")])],
        {"call_001.as_": ["pose_b"]})
    assert len(composed) == 2
    assert composed[0]["plan_sha256"] != composed[1]["plan_sha256"]
    with pytest.raises(ValueError, match="not a grounded alternative"):
        compose_from_template(base, observation,
            [dict(name="bad", edits=[dict(call_id="call_001", argument="as_", value="invented")])],
            {"call_001.as_": ["pose_b"]})
