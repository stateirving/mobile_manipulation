import numpy as np

from mm_simulator.robot import (
    OmnidirectionalBaseMapping,
    PyBulletInputMapping,
    SimulatedRobot,
)


def test_mapping_factory_keeps_mobile_robot_coordinates_unchanged():
    mapping = PyBulletInputMapping.from_string("omnidirectional")
    assert mapping is OmnidirectionalBaseMapping

    q = np.arange(9, dtype=float)
    v = np.arange(9, dtype=float) + 10.0
    q_mapped, v_mapped = mapping.forward(q, v)

    np.testing.assert_allclose(q_mapped, q)
    np.testing.assert_allclose(v_mapped, v)


def test_simulator_aggregate_actuator_preserves_total_extension():
    robot = SimulatedRobot.__new__(SimulatedRobot)
    joint_names = ["base", "arm_3", "arm_2", "arm_1", "arm_0"]
    robot.joints = {name: (index,) for index, name in enumerate(joint_names)}
    robot.q_lb = np.array([-np.inf, 0.0, 0.0, 0.0, 0.0])
    robot.q_ub = np.array([np.inf, 0.13, 0.13, 0.13, 0.13])
    robot.v_lb = np.array([-np.inf, -0.3, -0.3, -0.3, -0.3])
    robot.v_ub = np.array([np.inf, 0.3, 0.3, 0.3, 0.3])
    config = {
        "joint_names": joint_names,
        "aggregate_actuators": {
            "enabled": True,
            "groups": [
                {
                    "name": "wrist_extension",
                    "joints": ["arm_3", "arm_2", "arm_1", "arm_0"],
                    "geometry_distribution": [0.25, 0.25, 0.25, 0.25],
                    "velocity_limits": [-1.0, 1.0],
                }
            ],
        },
    }
    robot.aggregate_actuators = robot._parse_aggregate_actuators(config)
    q = robot._set_aggregate_actuator_states_from_configuration(
        [0.0, 0.04, 0.01, 0.03, 0.02]
    )
    projected = robot._project_aggregate_actuator_velocity([0.0, 0.3, 0.3, 0.3, 0.3])

    np.testing.assert_allclose(q, [0.0, 0.025, 0.025, 0.025, 0.025])
    np.testing.assert_allclose(projected, [0.0, 0.25, 0.25, 0.25, 0.25])
    assert robot.aggregate_actuators[0]["position"] == 0.1
    assert robot.aggregate_actuators[0]["requested_velocity"] == 1.0


def test_simulator_aggregate_actuator_applies_configured_delay(monkeypatch):
    robot = SimulatedRobot.__new__(SimulatedRobot)
    robot.aggregate_actuators = [
        {
            "pybullet_indices": np.array([4, 5, 6, 7]),
            "distribution": np.full(4, 0.25),
            "position_limits": np.array([0.0, 0.52]),
            "command_delay": 0.12,
            "position": 0.1,
            "requested_velocity": 0.1,
            "applied_velocity": 0.0,
            "command_queue": [],
        }
    ]
    resets = []
    monkeypatch.setattr(
        "mm_simulator.robot.pyb.resetJointState",
        lambda uid, joint, position, targetVelocity: resets.append(
            (joint, position, targetVelocity)
        ),
    )
    robot.uid = 1

    for _ in range(3):
        robot.advance_aggregate_actuators(0.03)
    assert robot.aggregate_actuators[0]["position"] == 0.1

    robot.advance_aggregate_actuators(0.03)
    assert np.isclose(robot.aggregate_actuators[0]["position"], 0.103)
    np.testing.assert_allclose([value[1] for value in resets[-4:]], 0.02575)
