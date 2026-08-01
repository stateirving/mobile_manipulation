import numpy as np

from mm_simulator.robot import OmnidirectionalBaseMapping, PyBulletInputMapping


def test_mapping_factory_keeps_mobile_robot_coordinates_unchanged():
    mapping = PyBulletInputMapping.from_string("omnidirectional")
    assert mapping is OmnidirectionalBaseMapping

    q = np.arange(9, dtype=float)
    v = np.arange(9, dtype=float) + 10.0
    q_mapped, v_mapped = mapping.forward(q, v)

    np.testing.assert_allclose(q_mapped, q)
    np.testing.assert_allclose(v_mapped, v)
