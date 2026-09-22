"""Auditable Codex-as-LLM strategy contract and deterministic grounding."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from simbench.assembly.interfaces import PUBLIC_SKILLS
from simbench.value.plan import digest


SCHEMA = "twingraph.codex_planner_response.v15.r1"


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_response(path, expected_round):
    response = json.loads(Path(path).read_text(encoding="utf-8"))
    if response.get("schema") != SCHEMA or response.get("round") != expected_round:
        raise ValueError("wrong Codex planner response schema/round")
    if response.get("authorship", {}).get("external_api_call") is not False:
        raise ValueError("this experiment requires an honest no-external-API record")
    slots = response.get("selection_slots", ())
    if len(slots) != int(response.get("candidate_count", -1)) or not slots:
        raise ValueError("candidate_count must exactly bind all strategy slots")
    if len({slot.get("id") for slot in slots}) != len(slots):
        raise ValueError("strategy slot identifiers must be unique")
    return response


def _angle_distance(a, b):
    return abs(math.atan2(math.sin(float(a) - float(b)), math.cos(float(a) - float(b))))


def _candidate_cost(proposal, slot):
    order = proposal["order"]
    requested_left = slot["pin_order"] == "left_first"
    actual_left = order.index("pin_left") < order.index("pin_right")
    order_cost = 100. * float(requested_left != actual_left)
    target_yaw = int(slot["pin_yaw_quadrant"]) * math.pi / 2
    yaws = [proposal["choices"][part].get("yaw", 0.) for part in ("pin_left", "pin_right")]
    yaw_cost = sum(_angle_distance(yaw, target_yaw) for yaw in yaws)
    requested_joint = slot["approach"] == "joint_checked"
    strategy = proposal["choices"]["pin_left"].get("approach_strategy")
    approach_cost = 10. * float(requested_joint != (strategy == "joint_checked_v10"))
    limits = [float(proposal["choices"][part].get("force_limit", 10.))
              for part in ("pin_left", "pin_right")]
    profile = slot.get("contact_profile", "balanced")
    if profile not in ("gentle", "balanced", "high_force"):
        raise ValueError(f"unknown contact profile: {profile}")
    profile_cost = {"gentle": .05 * np_mean(limits), "balanced": 0.,
                    "high_force": -.05 * np_mean(limits)}[profile]
    # A small deterministic tie breaker spreads otherwise equivalent choices
    # over contact-speed/clearance alternatives without reading outcomes.
    speeds = [float(proposal["choices"][part].get("speed", 0.)) for part in proposal["order"]]
    clearance = [float(proposal["choices"][part].get("clearance", 1.)) for part in proposal["order"]]
    tie = .01 * (sum(speeds) + sum(clearance))
    return order_cost + approach_cost + yaw_cost + profile_cost + tie


def np_mean(values):
    return sum(values) / max(len(values), 1)


def compile_response(pool, response, *, excluded_names=()):
    """Bind language strategy slots to distinct, already grounded plans."""
    excluded = set(excluded_names); selected = []; bindings = []
    for slot in response["selection_slots"]:
        choices = [(float(_candidate_cost(proposal, slot)), proposal["name"], i, proposal)
                   for i, proposal in enumerate(pool)
                   if proposal["name"] not in excluded and proposal["name"] not in {p["name"] for p in selected}]
        if not choices:
            raise ValueError("not enough distinct grounded candidates for Codex strategy slots")
        cost, _, index, proposal = min(choices)
        selected.append(deepcopy(proposal))
        bindings.append(dict(slot_id=slot["id"], search_pool_index=index,
                             grounded_name=proposal["name"], match_cost=cost,
                             proposal_sha256=digest(proposal)))
    return selected, bindings


def planner_request(task_spec, observation, cad, *, round_index, failure_summaries=()):
    return dict(schema="twingraph.codex_planner_request.v15.r1", round=round_index,
        task_spec=task_spec, observation_sha256=observation.get("sha256"),
        observation=observation, planning_cad=cad, planning_cad_sha256=digest(cad),
        atomic_skills=PUBLIC_SKILLS,
        constraints=dict(output="complete strategy slots", numeric_grounding="RGB-D/CAD/registered ports",
                         forbidden=("rollout labels", "simulator object-pose oracle", "unregistered skills")),
        failure_summaries=list(failure_summaries))


def failure_summary(result, rank, score):
    """Keep observations and LLM inferences in different fields."""
    error = result.get("error")
    executed = result.get("executed_parameters", ())
    failed = next((step for step in reversed(executed) if not step.get("ok", True)), None)
    completed = result.get("boundaries", ())[-1].get("completed", ()) if result.get("boundaries") else ()
    facts = dict(valid=result.get("valid"), success=result.get("success"), error=error,
        timeout=result.get("timeout"), resource_censored=result.get("resource_censored"),
        completed=list(completed), failed_skill=failed.get("skill") if failed else None,
        failed_parameters=failed.get("params") if failed else None,
        stage_passes=result.get("stage_passes"), wall_seconds=result.get("total_wall_seconds"),
        input_graph_sha256=result.get("input_graph_sha256"))
    return dict(schema="twingraph.failure_summary.v15.r1", candidate=result["proposal"]["name"],
                value_rank=rank, value_score=float(score), observed_facts=facts,
                inferred_causes=[], inference_status="not inferred by deterministic extractor")
