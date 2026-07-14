import numpy as np
import pytest

from mm_plan.ompl_base_planner import OMPLBasePlanner
from mm_plan.ompl_ee_planner import OMPLEEPlanner


@pytest.mark.parametrize(
    ("planner_class", "goal_key", "goal"),
    [
        (OMPLBasePlanner, "base_pose", [1.0, 0.0, 0.0]),
        (OMPLEEPlanner, "ee_pose", [1.0, 0.0, 0.5, 0.0, 0.0, 0.0]),
    ],
)
def test_ompl_succeeds_before_exhausting_attempts(
    monkeypatch, planner_class, goal_key, goal
):
    planner = planner_class(
        {
            "name": "test_planner",
            goal_key: goal,
            "ompl": {"solve_attempts": 3},
            "esdf": {
                "enabled": False,
                "allow_straight_line_fallback": True,
            },
        }
    )
    expected_path = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    call_count = 0

    def solve(_start, _goal):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise RuntimeError(f"failure {call_count}")
        return expected_path

    monkeypatch.setattr(planner, "_solve_ompl", solve)
    monkeypatch.setattr(
        planner,
        "_straight_line_path",
        lambda _start, _goal: pytest.fail("fallback should not be used"),
    )

    path = planner._plan_with_ompl(np.zeros(3), np.ones(3))

    assert call_count == 3
    assert path is expected_path


@pytest.mark.parametrize(
    ("planner_class", "goal_key", "goal"),
    [
        (OMPLBasePlanner, "base_pose", [1.0, 0.0, 0.0]),
        (OMPLEEPlanner, "ee_pose", [1.0, 0.0, 0.5, 0.0, 0.0, 0.0]),
    ],
)
def test_ompl_falls_back_only_after_all_attempts_fail(
    monkeypatch, planner_class, goal_key, goal
):
    planner = planner_class(
        {
            "name": "test_planner",
            goal_key: goal,
            "ompl": {"solve_attempts": 3},
            "esdf": {
                "enabled": False,
                "allow_straight_line_fallback": True,
            },
        }
    )
    fallback_path = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    solve_count = 0
    fallback_count = 0

    def solve(_start, _goal):
        nonlocal solve_count
        solve_count += 1
        raise RuntimeError(f"failure {solve_count}")

    def fallback(_start, _goal):
        nonlocal fallback_count
        fallback_count += 1
        return fallback_path

    monkeypatch.setattr(planner, "_solve_ompl", solve)
    monkeypatch.setattr(planner, "_straight_line_path", fallback)

    path = planner._plan_with_ompl(np.zeros(3), np.ones(3))

    assert solve_count == 3
    assert fallback_count == 1
    assert path is fallback_path


@pytest.mark.parametrize("planner_class", [OMPLBasePlanner, OMPLEEPlanner])
def test_ompl_solve_attempts_must_be_positive(planner_class):
    goal_config = (
        {"base_pose": [1.0, 0.0, 0.0]}
        if planner_class is OMPLBasePlanner
        else {"ee_pose": [1.0, 0.0, 0.5, 0.0, 0.0, 0.0]}
    )

    with pytest.raises(ValueError, match="solve_attempts"):
        planner_class(
            {
                "name": "test_planner",
                **goal_config,
                "ompl": {"solve_attempts": 0},
            }
        )
