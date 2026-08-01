import casadi as cs
import numpy as np

from mm_utils.math import casadi_SO3_log


def _rotz(theta):
    return cs.vertcat(
        cs.horzcat(cs.cos(theta), -cs.sin(theta), 0),
        cs.horzcat(cs.sin(theta), cs.cos(theta), 0),
        cs.horzcat(0, 0, 1),
    )


def test_casadi_so3_log_identity_has_finite_value_and_jacobian():
    angle = cs.MX.sym("angle")
    omega = casadi_SO3_log(_rotz(angle))
    function = cs.Function("so3_log_and_jacobian", [angle], [omega, cs.jacobian(omega, angle)])

    value, jacobian = function(0.0)

    np.testing.assert_allclose(value.full().reshape(-1), np.zeros(3), atol=1e-12)
    assert np.all(np.isfinite(jacobian.full()))
    np.testing.assert_allclose(jacobian.full().reshape(-1), [0.0, 0.0, 1.0], atol=1e-9)


def test_casadi_so3_log_recovers_small_rotation():
    angle = cs.MX.sym("angle")
    function = cs.Function("so3_log_small_angle", [angle], [casadi_SO3_log(_rotz(angle))])

    value = function(1.0e-4).full().reshape(-1)

    np.testing.assert_allclose(value, [0.0, 0.0, 1.0e-4], rtol=1e-7, atol=1e-12)
