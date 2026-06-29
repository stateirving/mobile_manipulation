import time

import numpy as np

from mm_utils.parsing import parse_array


def _parse_vector(value, expected_size, name):
    if isinstance(value, (int, float)):
        arr = np.full(expected_size, float(value), dtype=np.float64)
    elif isinstance(value, str):
        arr = parse_array([value]).astype(np.float64)
    else:
        arr = parse_array(value).astype(np.float64)

    if arr.size == 1 and expected_size != 1:
        arr = np.full(expected_size, float(arr[0]), dtype=np.float64)
    if arr.size != expected_size:
        raise ValueError(
            f"{name} must have length {expected_size}, got {arr.size}"
        )
    return arr


class LocalESDFGridSampler:
    """Sample a fixed-size local ESDF grid from an ESDF provider.

    The sampled grid is world-axis aligned. Its x/y axes are recentered each
    control tick, while z is an absolute world-frame interval.
    """

    def __init__(self, config=None, invalid_distance=-1.0):
        config = {} if config is None else dict(config)
        self.size_xy = _parse_vector(
            config.get("size_xy", [2.0, 2.0]), 2, "size_xy"
        )
        self.z_range = _parse_vector(
            config.get("z_range", [0.0, 1.2]), 2, "z_range"
        )
        self.voxel_size = float(config.get("voxel_size", 0.10))
        self.center = str(config.get("center", "base"))
        self.invalid_distance = float(
            config.get("invalid_distance", invalid_distance)
        )
        self.max_distance = config.get("max_distance", None)
        self.max_distance = (
            None if self.max_distance is None else float(self.max_distance)
        )

        if np.any(self.size_xy <= 0.0):
            raise ValueError("local ESDF grid size_xy must be positive")
        if self.z_range[1] <= self.z_range[0]:
            raise ValueError("local ESDF grid z_range must be increasing")
        if self.voxel_size <= 0.0:
            raise ValueError("local ESDF grid voxel_size must be positive")
        if self.center not in {"base", "warm_start_mean"}:
            raise ValueError(
                "local ESDF grid center must be 'base' or 'warm_start_mean'"
            )

        self.num_x = self._num_samples(self.size_xy[0])
        self.num_y = self._num_samples(self.size_xy[1])
        self.num_z = self._num_samples(self.z_range[1] - self.z_range[0])
        self.shape = (self.num_x, self.num_y, self.num_z)
        self.num_values = int(np.prod(self.shape))

    def sample(self, esdf_map, q_bar):
        """Sample the ESDF map on the configured local grid.

        Args:
            esdf_map: Object with query(points) returning ESDF values.
            q_bar: Warm-start generalized-coordinate trajectory.

        Returns:
            dict: x_grid, y_grid, z_grid, value, valid and timing/stat fields.
        """
        t1 = time.perf_counter()
        q_bar = np.asarray(q_bar, dtype=np.float64)
        center_xy = self._center_xy(q_bar)
        x_grid, y_grid, z_grid = self._axes(center_xy)

        x_mesh, y_mesh, z_mesh = np.meshgrid(
            x_grid, y_grid, z_grid, indexing="ij"
        )
        points = np.column_stack(
            (
                x_mesh.ravel(order="F"),
                y_mesh.ravel(order="F"),
                z_mesh.ravel(order="F"),
            )
        )

        distances, _, valid = esdf_map.query(points)
        distances = np.asarray(distances, dtype=np.float64).reshape(-1)
        valid = np.asarray(valid, dtype=bool).reshape(-1)
        valid &= np.isfinite(distances)

        values = np.full(
            distances.shape, self.invalid_distance, dtype=np.float64
        )
        values[valid] = distances[valid]
        if self.max_distance is not None:
            values[valid] = np.minimum(values[valid], self.max_distance)

        t2 = time.perf_counter()
        valid_values = values[valid]
        return {
            "x_grid": x_grid,
            "y_grid": y_grid,
            "z_grid": z_grid,
            "value": values.reshape((-1, 1), order="F"),
            "valid": valid.reshape(self.shape, order="F"),
            "raw_distance": distances.reshape(self.shape, order="F"),
            "center_xy": center_xy,
            "valid_count": int(np.count_nonzero(valid)),
            "total_count": int(valid.size),
            "min_distance": (
                float(np.min(valid_values)) if valid_values.size else np.nan
            ),
            "max_distance": (
                float(np.max(valid_values)) if valid_values.size else np.nan
            ),
            "sample_time": t2 - t1,
        }

    def _center_xy(self, q_bar):
        if q_bar.ndim != 2 or q_bar.shape[1] < 2:
            raise ValueError("q_bar must have shape (N, nq) with nq >= 2")
        if self.center == "warm_start_mean":
            return np.mean(q_bar[:, :2], axis=0)
        return q_bar[0, :2].copy()

    def _axes(self, center_xy):
        x_offsets = np.linspace(
            -0.5 * self.size_xy[0],
            0.5 * self.size_xy[0],
            self.num_x,
        )
        y_offsets = np.linspace(
            -0.5 * self.size_xy[1],
            0.5 * self.size_xy[1],
            self.num_y,
        )
        x_grid = center_xy[0] + x_offsets
        y_grid = center_xy[1] + y_offsets
        z_grid = np.linspace(self.z_range[0], self.z_range[1], self.num_z)
        return x_grid, y_grid, z_grid

    def _num_samples(self, length):
        return max(2, int(round(float(length) / self.voxel_size)) + 1)
