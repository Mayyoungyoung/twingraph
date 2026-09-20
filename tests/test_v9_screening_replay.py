from scripts.analyze_v9_candidate_matrix import outcome


def test_screening_stops_at_first_verified_success_and_charges_attempts():
    case = {
        "reference": {"summary": {"success": False, "wall_seconds": 2., "total_wall_seconds": 3.}},
        "alternative": {"summary": {"success": True, "wall_seconds": 4., "total_wall_seconds": 5.}},
        "unused": {"summary": {"success": True, "wall_seconds": 8., "total_wall_seconds": 9.}},
    }
    order = ["reference", "alternative", "unused"]
    one = outcome(case, order, 1)
    assert one["success"] is False and one["verification_count"] == 1
    two = outcome(case, order, 3, overhead=.25)
    assert two["success"] is True and two["tried"] == order[:2]
    assert two["verification_wall_seconds"] == 6.
    assert two["full_system_wall_seconds"] == 8.25
