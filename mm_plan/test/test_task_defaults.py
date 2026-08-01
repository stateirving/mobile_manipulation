import pytest

from mm_plan.TaskManager import resolve_task_configs


def test_resolve_task_configs_applies_defaults_and_nested_overrides():
    config = {
        "task_defaults": {
            "base": {
                "planner_type": "OMPLBasePlanner",
                "ompl": {"planner": "RRTstar", "solve_time": 0.5},
                "esdf": {"enabled": True, "d_safe": 0.2},
            }
        },
        "tasks": [
            {"defaults": "base", "name": "First", "base_pose": [1, 0, 0]},
            {
                "defaults": "base",
                "name": "Second",
                "base_pose": [2, 0, 0],
                "esdf": {"d_safe": 0.1},
            },
        ],
    }

    tasks = resolve_task_configs(config)

    assert tasks[0]["planner_type"] == "OMPLBasePlanner"
    assert tasks[0]["ompl"] == {"planner": "RRTstar", "solve_time": 0.5}
    assert tasks[1]["esdf"] == {"enabled": True, "d_safe": 0.1}
    assert "defaults" not in tasks[0]
    assert config["task_defaults"]["base"]["esdf"]["d_safe"] == 0.2


def test_resolve_task_configs_keeps_legacy_tasks_unchanged():
    task = {
        "name": "Legacy",
        "planner_type": "WaypointPlanner",
        "base_pose": [0, 0, 0],
    }

    assert resolve_task_configs({"tasks": [task]}) == [task]


def test_resolve_task_configs_rejects_unknown_defaults():
    with pytest.raises(KeyError, match="Unknown planner task defaults"):
        resolve_task_configs({"task_defaults": {}, "tasks": [{"defaults": "missing"}]})
