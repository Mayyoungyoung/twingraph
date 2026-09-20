from pathlib import Path
from tempfile import TemporaryDirectory
import pytest

from simbench.value import stage_v5, stage_v9
from simbench.value.v9_candidates import proposals as old_proposals
from simbench.value.v10_candidates import proposals as new_proposals


def test_versioned_approach_changes_executable_plan_without_rewriting_v9():
    scratch = Path(__file__).resolve().parents[1] / "results"
    scratch.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=scratch) as temporary:
        _, session, _, targets = stage_v9.make_scene(1311, Path(temporary))
    old = old_proposals()[0]
    new = next(row for row in new_proposals() if row["name"] == "handle_joint_approach")
    old_plan = stage_v5.program(session, targets, old["order"], old["choices"], v7=True)
    new_plan = stage_v5.program(session, targets, new["order"], new["choices"], v7=True)
    old_approach = [call for call in old_plan.calls if call.skill == "move"
                    and call.roles.get("manipulated") == "handle" and "grasp" in call.arguments]
    new_approach = [call for call in new_plan.calls if call.skill == "move"
                    and call.roles.get("manipulated") == "handle" and "grasp" in call.arguments]
    assert len(old_approach) == len(new_approach) == 1
    assert "strategy" not in old_approach[0].arguments
    assert new_approach[0].arguments["strategy"].value == "joint_checked_v10"
    assert old_plan.id != new_plan.id


def test_staged_pin_exit_is_two_executable_lifts():
    scratch = Path(__file__).resolve().parents[1] / "results"
    scratch.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=scratch) as temporary:
        _, session, _, targets = stage_v9.make_scene(1306, Path(temporary))
    proposal = next(row for row in new_proposals() if row["name"] == "pin_left_staged_lift")
    plan = stage_v5.program(session, targets, proposal["order"], proposal["choices"], v7=True)
    pin_lifts = [call.arguments["delta"].value[2] for call in plan.calls
                 if call.skill == "move" and call.roles.get("manipulated") == "pin_left"
                 and "delta" in call.arguments and 0 < call.arguments["delta"].value[2] < .1]
    assert pin_lifts[:2] == pytest.approx([.025, .075])
