from types import SimpleNamespace
import numpy as np
import pytest

from simbench.value.schema_learning import compact_columns, calibrate, sigmoid, score_metrics
from simbench.value.goal_check import evaluate_goals, validate_goals


def test_compaction_fits_inputs_only_and_preserves_distinct_columns():
    x = np.asarray([[1, 2, 2, 7, 4], [1, 3, 3, 7, 5], [1, 4, 4, 7, 9]], np.float32)
    assert compact_columns(x) == [1, 4]
    # No labels or test rows can affect the saved transform.
    assert compact_columns(x[::-1]) == [1, 4]
    with pytest.raises(ValueError):
        compact_columns(np.ones((3, 5)))


def test_calibration_corrects_offset_without_reversing_plan_order():
    z = np.asarray([-3., -2., -1., 0., 1., 2.])
    y = sigmoid(z + 2).astype(np.float32)
    c = calibrate([z], [dict(y=y, config=("task", "a"))])
    prediction = sigmoid(c["scale"]*z + c["bias"])
    assert c["scale"] > 0
    assert np.all(np.diff(prediction) > 0)
    assert np.mean((prediction-y)**2) < np.mean((sigmoid(z)-y)**2) / 10


def test_final_goal_rejects_held_part_even_at_correct_target():
    ctx = SimpleNamespace(obj_pos=lambda part: np.array([0., 0., .8]),
                          obj_axis=lambda part: np.array([0., 0., 1.]),
                          eef_pos=lambda: np.array([0., 0., .9]), body_id=lambda part: 1,
                          data=SimpleNamespace(contact=[]))
    session = SimpleNamespace(ctx=ctx, held="pin")
    goals = [dict(predicate="seated_released_retracted", manipulated="pin", position=[0, 0, .8])]
    assert not evaluate_goals(session, goals)["success"]
    session.held = None
    assert evaluate_goals(session, goals)["success"]
    goals[0]["position"][0] = .02
    assert not evaluate_goals(session, goals)["success"]


def test_unknown_task_predicate_is_not_implicitly_success():
    with pytest.raises(ValueError, match="unsupported"):
        validate_goals([dict(predicate="assembled", manipulated="pin", position=[0, 0, 0])])


def test_saturated_probabilities_do_not_change_deployed_ranking():
    groups = [dict(id="a", seed=1, y=np.array([0., 1.]))]
    result = score_metrics(groups, [np.array([41., 42.])])
    assert result["hit1"] == 1.
    assert result["brier"] == .5


def test_single_class_validation_does_not_fit_fake_calibration():
    c = calibrate([np.array([-1., 1.])], [dict(y=np.zeros(2), config=("a", 1))])
    assert c["scale"] == 1. and c["bias"] == 0.
    assert "skip_reason" in c


@pytest.mark.parametrize("key,value", [("position_tolerance", float("inf")),
                                      ("tilt_tolerance_deg", float("nan")),
                                      ("minimum_eef_clearance_m", -.1)])
def test_task_tolerances_must_be_finite_and_physically_valid(key, value):
    goal = dict(predicate="seated", manipulated="pin", position=[0, 0, 0])
    goal[key] = value
    with pytest.raises(ValueError):
        validate_goals([goal])
