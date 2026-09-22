import json
import math

from scripts.v15_llm_contract import compile_response, failure_summary, load_response


def pool():
    rows = []
    for left in (True, False):
        for quadrant in range(4):
            for joint in (False, True):
                pins = ("pin_left", "pin_right") if left else ("pin_right", "pin_left")
                order = ["carriage", "end_stop", *pins, "handle"]
                choices = {part: dict(yaw=quadrant * math.pi / 2, speed=.004, clearance=1.)
                           for part in order}
                if joint:
                    choices["pin_left"]["approach_strategy"] = "joint_checked_v10"
                rows.append(dict(name=f"plan_{len(rows):02d}", order=order, choices=choices,
                                 wipe_variant=0, wipe_force=1.5, wipe_duration=14., stroke_minimum=.02))
    return rows


def test_frozen_codex_response_compiles_to_sixteen_distinct_complete_plans():
    response = load_response(
        "docs/evidence/value_v15_full_system/protocol/llm_round0_response.json", 0)
    plans, bindings = compile_response(pool(), response)
    assert len(plans) == len({plan["name"] for plan in plans}) == 16
    assert len(bindings) == 16
    assert all(set(("order", "choices", "stroke_minimum")) <= set(plan) for plan in plans)


def test_exclusion_for_feedback_round_never_reuses_a_plan():
    response = load_response(
        "docs/evidence/value_v15_full_system/protocol/llm_round0_response.json", 0)
    larger = pool() + [{**row, "name": row["name"] + "_alternative"} for row in pool()]
    first, _ = compile_response(larger, response)
    second, _ = compile_response(larger, response, excluded_names=[row["name"] for row in first])
    assert {row["name"] for row in first}.isdisjoint(row["name"] for row in second)


def test_failure_summary_does_not_present_inference_as_observation():
    result = dict(valid=True, success=False, error="insert: overload", timeout=False,
        resource_censored=False, total_wall_seconds=2., input_graph_sha256="a",
        proposal=dict(name="plan_00"), stage_passes=dict(assembly_pass=False),
        boundaries=[dict(completed=["cleaning", "carriage"])],
        executed_parameters=[dict(skill="insert", params=dict(part="end_stop"), ok=False)])
    summary = failure_summary(result, 1, .2)
    assert summary["observed_facts"]["failed_skill"] == "insert"
    assert summary["inferred_causes"] == []
    assert "overload" in summary["observed_facts"]["error"]
