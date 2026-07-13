import numpy as np

from mm_plan.ompl_base_planner import OMPLBasePlanner
from mm_plan.ompl_ee_planner import OMPLEEPlanner


class _FakeRobotModel:
    def getEE(self, q):
        return np.asarray(q[:3], dtype=np.float64), np.array(
            [0.0, 0.0, 0.0, 1.0], dtype=np.float64
        )


def _robot_states(position):
    q = np.asarray(position, dtype=np.float64)
    return q, np.zeros_like(q)


def _base_planner():
    planner = OMPLBasePlanner(
        {
            "name": "test_base",
            "base_pose": [1.0, 1.0, np.pi / 2.0],
            "path": {"linear_speed": 0.25, "dt": 0.1},
            "esdf": {"enabled": False},
        }
    )
    planner.base_plan = planner._make_plan(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, np.pi / 2.0],
            ]
        )
    )
    planner.planned = True
    return planner


def _ee_planner():
    planner = OMPLEEPlanner(
        {
            "name": "test_ee",
            "ee_pose": [1.0, 1.0, 0.0, 0.0, 0.0, np.pi / 2.0],
            "path": {"linear_speed": 0.2, "dt": 0.1},
            "esdf": {"enabled": False},
        },
        resources={"robot_model": _FakeRobotModel()},
    )
    planner.ee_plan = planner._make_plan(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        ),
        np.zeros(3),
        np.array([0.0, 0.0, np.pi / 2.0]),
    )
    planner.planned = True
    return planner


def test_base_horizon_starts_at_projected_progress_and_ignores_elapsed_time():
    planner = _base_planner()
    states = _robot_states([0.4, 0.2, 0.0])

    positions, velocities = planner.getBaseTrackingPointArray(
        states, num_pts=3, dt=0.1, time_offset=20.0
    )

    np.testing.assert_allclose(
        positions[:, :2], [[0.4, 0.0], [0.425, 0.0], [0.45, 0.0]]
    )
    np.testing.assert_allclose(velocities[:, :2], [[0.25, 0.0]] * 3)
    assert np.isclose(planner.path_progress, 0.4)

    positions_later, _ = planner.getBaseTrackingPointArray(
        states, num_pts=3, dt=0.1, time_offset=200.0
    )
    np.testing.assert_allclose(positions_later, positions)


def test_base_progress_is_monotonic_and_advances_around_corner():
    planner = _base_planner()
    planner.getBaseTrackingPointArray(
        _robot_states([0.4, 0.0, 0.0]), num_pts=1, dt=0.1
    )
    planner.getBaseTrackingPointArray(
        _robot_states([0.2, 0.0, 0.0]), num_pts=1, dt=0.1
    )
    assert np.isclose(planner.path_progress, 0.4)

    positions, _ = planner.getBaseTrackingPointArray(
        _robot_states([1.0, 0.2, 0.0]), num_pts=2, dt=0.1
    )
    assert np.isclose(planner.path_progress, 1.2)
    np.testing.assert_allclose(positions[:, :2], [[1.0, 0.2], [1.0, 0.225]])


def test_base_terminal_pose_snaps_directly_to_goal_yaw():
    planner = _base_planner()
    planner.base_plan = planner._make_plan(
        np.array([[0.0, 0.0, 0.0], [1.0, 1.0, np.pi / 4.0]])
    )
    states = _robot_states([0.9, 0.9, 0.0])

    positions, velocities = planner.getBaseTrackingPointArray(
        states, num_pts=3, dt=0.1
    )

    assert planner.terminal_pose_active
    np.testing.assert_allclose(positions, np.tile(planner.base_target, (3, 1)))
    np.testing.assert_allclose(velocities, np.zeros((3, 3)))

    # Terminal mode is latched even if the base drifts outside XY tolerance.
    positions_later, _ = planner.getBaseTrackingPointArray(
        _robot_states([0.0, 0.0, 0.0]), num_pts=1, dt=0.1
    )
    np.testing.assert_allclose(positions_later[0], planner.base_target)


def test_tangent_path_does_not_blend_goal_yaw_before_xy_is_reached():
    planner = _base_planner()
    path = planner._densify_path(
        np.array([[0.0, 0.0, 0.0], [1.0, 1.0, np.pi / 2.0]])
    )
    planner.base_plan = planner._make_plan(path)

    positions, _ = planner.getBaseTrackingPointArray(
        _robot_states([0.0, 0.0, 0.0]), num_pts=2, dt=20.0
    )

    tangent_yaw = np.pi / 4.0
    assert not planner.terminal_pose_active
    assert np.isclose(path[-1, 2], tangent_yaw)
    assert np.isclose(positions[-1, 2], tangent_yaw)
    assert not np.isclose(positions[-1, 2], planner.base_target[2])


def test_ee_horizon_uses_projected_cartesian_progress():
    planner = _ee_planner()
    states = _robot_states([1.0, 0.3, 0.0])

    positions, velocities = planner.getEETrackingPointArray(
        states, num_pts=3, dt=0.1, time_offset=100.0
    )

    assert np.isclose(planner.path_progress, 1.3)
    np.testing.assert_allclose(
        positions[:, :3], [[1.0, 0.3, 0.0], [1.0, 0.32, 0.0], [1.0, 0.34, 0.0]]
    )
    np.testing.assert_allclose(velocities[:, :3], [[0.0, 0.2, 0.0]] * 3)


def test_successful_replan_resets_progress_for_new_path():
    planner = _base_planner()
    planner.path_progress = 1.2
    planner.terminal_pose_active = True
    planner._plan_with_ompl = lambda start, goal: np.asarray([start, goal])

    planner._set_plan_from_current_state(
        _robot_states([0.3, 0.0, 0.0]), t=5.0, reason="test"
    )

    assert planner.path_progress == 0.0
    assert not planner.terminal_pose_active
    assert planner.base_plan["s"][0] == 0.0
    np.testing.assert_allclose(planner.base_plan["p"][0], [0.3, 0.0, 0.0])
