import numpy as np
import casadi as cs

from mm_control.local_esdf_grid import LocalESDFGridSampler
from mm_control.MPCConstraints import CasadiLocalGridESDFConstraint


class FakeESDFMap:
    def query(self, points):
        points = np.asarray(points, dtype=np.float64)
        distances = points[:, 0] + points[:, 1] + points[:, 2]
        gradients = np.ones((points.shape[0], 3), dtype=np.float64)
        valid = np.ones(points.shape[0], dtype=bool)
        return distances, gradients, valid


class FakeRobotModel:
    def __init__(self):
        self.q_sym = cs.MX.sym("q", 3)
        self.ssSymMdl = {"nx": 6, "nu": 3}
        self.collisionLinkKinSymMdls = {
            "sphere": cs.Function(
                "sphere_fk",
                [self.q_sym],
                [self.q_sym, cs.DM.eye(3)],
            )
        }


def test_local_esdf_grid_sampler_centers_grid_on_base():
    sampler = LocalESDFGridSampler(
        {
            "size_xy": [1.0, 1.0],
            "z_range": [0.0, 1.0],
            "voxel_size": 0.5,
        }
    )
    q_bar = np.array([[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]])

    grid = sampler.sample(FakeESDFMap(), q_bar)

    np.testing.assert_allclose(grid["x_grid"], [0.5, 1.0, 1.5])
    np.testing.assert_allclose(grid["y_grid"], [1.5, 2.0, 2.5])
    np.testing.assert_allclose(grid["z_grid"], [0.0, 0.5, 1.0])
    assert grid["value"].shape == (27, 1)
    assert grid["valid_count"] == 27

    values = grid["value"].reshape(sampler.shape, order="F")
    assert np.isclose(values[1, 1, 1], 3.5)


def test_casadi_local_grid_esdf_constraint_interpolates_distance():
    robot = FakeRobotModel()
    constraint = CasadiLocalGridESDFConstraint(
        robot,
        sphere_names=["sphere"],
        sphere_radii=[0.1],
        d_safe=0.2,
        grid_shape=(2, 2, 2),
    )

    x_grid = np.array([0.0, 1.0])
    y_grid = np.array([0.0, 1.0])
    z_grid = np.array([0.0, 1.0])
    x_mesh, y_mesh, z_mesh = np.meshgrid(x_grid, y_grid, z_grid, indexing="ij")
    values = x_mesh + y_mesh + z_mesh

    p_map = constraint.p_struct(0)
    p_map["x_grid"] = x_grid
    p_map["y_grid"] = y_grid
    p_map["z_grid"] = z_grid
    p_map["value"] = values.ravel(order="F").reshape((-1, 1))

    x = np.array([0.25, 0.25, 0.25, 0.0, 0.0, 0.0])
    u = np.zeros(3)
    g = constraint.check(x, u, p_map.cat.full().flatten())

    assert np.isclose(float(g), -0.45)
