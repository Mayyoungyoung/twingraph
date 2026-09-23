"""Exact random-screening time must agree with exhaustive small permutations."""
from itertools import permutations

import pytest

from scripts.train_and_compare_stage_value import random_expected_time, ordered_result


@pytest.mark.parametrize("labels", ((1, 0, 0), (1, 1, 0), (0, 0, 0), (1, 0, 1, 0)))
@pytest.mark.parametrize("k", (1, 2, 3))
def test_random_expected_time_matches_exhaustive(labels, k):
    rows = [dict(y=float(y), seconds=float(i + 2)) for i, y in enumerate(labels)]
    observed = [ordered_result(rows, order, k)["twin_seconds"]
                for order in permutations(range(len(rows)))]
    assert random_expected_time(rows, k) == pytest.approx(sum(observed)/len(observed))
