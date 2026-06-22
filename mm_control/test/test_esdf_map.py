import numpy as np
import pytest

from mm_control.esdf_map import ESDFMap, OnlineNvbloxESDFMap


def test_esdf_map_trilinear_query(tmp_path):
    xs = np.array([0.0, 1.0], dtype=np.float32)
    ys = np.array([0.0, 1.0], dtype=np.float32)
    zs = np.array([0.0, 1.0], dtype=np.float32)
    grid = np.meshgrid(xs, ys, zs, indexing="ij")
    distance = grid[0] + grid[1] + grid[2]
    gradient = np.ones(distance.shape + (3,), dtype=np.float32)
    valid = np.ones(distance.shape, dtype=bool)

    path = tmp_path / "esdf_grid.npz"
    np.savez(
        path,
        xs=xs,
        ys=ys,
        zs=zs,
        distance=distance,
        gradient=gradient,
        valid=valid,
    )

    esdf = ESDFMap(path)
    distance_q, gradient_q, valid_q = esdf.query([0.25, 0.25, 0.25])

    assert valid_q
    assert np.isclose(distance_q, 0.75)
    np.testing.assert_allclose(gradient_q, [1.0, 1.0, 1.0])


def test_online_nvblox_from_config_merges_online_config(monkeypatch):
    init_kwargs = {}

    def fake_init(self, **kwargs):
        init_kwargs.update(kwargs)

    monkeypatch.setattr(OnlineNvbloxESDFMap, "__init__", fake_init)

    OnlineNvbloxESDFMap.from_config(
        {
            "source": "online_nvblox",
            "query_radius": 0.10,
            "unknown_distance_threshold": 50.0,
            "online_nvblox": {
                "voxel_size": 0.03,
                "query_radius": 0.20,
                "camera": {
                    "fx": 525.0,
                    "fy": 525.0,
                    "cx": 319.5,
                    "cy": 239.5,
                    "width": 640,
                    "height": 480,
                },
            },
        }
    )

    assert init_kwargs["voxel_size"] == 0.03
    assert init_kwargs["query_radius"] == 0.20
    assert init_kwargs["unknown_distance_threshold"] == 50.0
    assert init_kwargs["camera"]["width"] == 640


def test_online_nvblox_esdf_grad_query_tensor_has_radius_column():
    torch = pytest.importorskip("torch")
    esdf = object.__new__(OnlineNvbloxESDFMap)
    esdf._torch = torch
    esdf.device = torch.device("cpu")
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    esdf.query_radius = 0.0
    query = esdf._make_query_tensor(points)
    assert tuple(query.shape) == (2, 4)
    np.testing.assert_allclose(query[:, :3].numpy(), points)
    np.testing.assert_allclose(query[:, 3].numpy(), [0.0, 0.0])

    esdf.query_radius = 0.15
    query = esdf._make_query_tensor(points)
    assert tuple(query.shape) == (2, 4)
    np.testing.assert_allclose(query[:, 3].numpy(), [0.15, 0.15])
