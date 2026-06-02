from pathlib import Path

import numpy as np


class ESDFMap:
    """Trilinear query helper for exported nvblox ESDF grids."""

    def __init__(self, path, require_all_corners_valid=True):
        self.path = Path(path)
        self.require_all_corners_valid = bool(require_all_corners_valid)

        with np.load(self.path) as data:
            self.xs = np.asarray(data["xs"], dtype=np.float64)
            self.ys = np.asarray(data["ys"], dtype=np.float64)
            self.zs = np.asarray(data["zs"], dtype=np.float64)
            self.distance = np.asarray(data["distance"], dtype=np.float64)
            self.gradient = np.asarray(data["gradient"], dtype=np.float64)
            self.valid = np.asarray(data["valid"], dtype=bool)

        self._validate()
        self.bounds = np.array(
            [
                self.xs[0],
                self.ys[0],
                self.zs[0],
                self.xs[-1],
                self.ys[-1],
                self.zs[-1],
            ],
            dtype=np.float64,
        )
        self.resolution = np.array(
            [
                self._axis_resolution(self.xs),
                self._axis_resolution(self.ys),
                self._axis_resolution(self.zs),
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_config(cls, config):
        return cls(
            config["map_path"],
            require_all_corners_valid=config.get("require_all_corners_valid", True),
        )

    def query(self, point):
        """Query one point or a batch of points in world frame.

        Args:
            point: Array-like with shape (3,) or (N, 3).

        Returns:
            For a single point: (distance, gradient, valid).
            For a batch: (distances, gradients, valid), with shapes (N,), (N, 3), (N,).
        """
        points, single = self._as_points(point)
        distances, gradients, valid = self.query_batch(points)
        if single:
            return float(distances[0]), gradients[0], bool(valid[0])
        return distances, gradients, valid

    def query_batch(self, points):
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")

        n = points.shape[0]
        distances = np.full(n, np.nan, dtype=np.float64)
        gradients = np.full((n, 3), np.nan, dtype=np.float64)

        ix, tx, inside_x = self._axis_indices(self.xs, points[:, 0])
        iy, ty, inside_y = self._axis_indices(self.ys, points[:, 1])
        iz, tz, inside_z = self._axis_indices(self.zs, points[:, 2])
        valid = inside_x & inside_y & inside_z

        if not np.any(valid):
            return distances, gradients, valid

        interp_distance = np.zeros(n, dtype=np.float64)
        interp_gradient = np.zeros((n, 3), dtype=np.float64)
        interp_valid = valid.copy()

        for dx in (0, 1):
            wx = tx if dx else 1.0 - tx
            for dy in (0, 1):
                wy = ty if dy else 1.0 - ty
                for dz in (0, 1):
                    wz = tz if dz else 1.0 - tz
                    weight = wx * wy * wz
                    corner_valid = self.valid[ix + dx, iy + dy, iz + dz]
                    if self.require_all_corners_valid:
                        interp_valid &= corner_valid
                    interp_distance += weight * self.distance[ix + dx, iy + dy, iz + dz]
                    interp_gradient += (
                        weight[:, None] * self.gradient[ix + dx, iy + dy, iz + dz]
                    )

        if not self.require_all_corners_valid:
            interp_valid &= np.isfinite(interp_distance)

        distances[interp_valid] = interp_distance[interp_valid]
        gradients[interp_valid] = interp_gradient[interp_valid]
        return distances, gradients, interp_valid

    def _validate(self):
        if self.xs.ndim != 1 or self.ys.ndim != 1 or self.zs.ndim != 1:
            raise ValueError("ESDF axes xs, ys, zs must be one-dimensional")
        if min(len(self.xs), len(self.ys), len(self.zs)) < 2:
            raise ValueError("ESDF axes must contain at least two samples")
        for axis_name, axis in (("xs", self.xs), ("ys", self.ys), ("zs", self.zs)):
            if not np.all(np.diff(axis) > 0.0):
                raise ValueError(f"ESDF axis {axis_name} must be strictly increasing")

        expected_shape = (len(self.xs), len(self.ys), len(self.zs))
        if self.distance.shape != expected_shape:
            raise ValueError(
                f"distance shape {self.distance.shape} does not match {expected_shape}"
            )
        if self.valid.shape != expected_shape:
            raise ValueError(
                f"valid shape {self.valid.shape} does not match {expected_shape}"
            )
        if self.gradient.shape != expected_shape + (3,):
            raise ValueError(
                f"gradient shape {self.gradient.shape} does not match {expected_shape + (3,)}"
            )

    def _axis_indices(self, axis, values):
        inside = (values >= axis[0]) & (values <= axis[-1])
        idx = np.searchsorted(axis, values, side="right") - 1
        idx = np.clip(idx, 0, len(axis) - 2)
        t = (values - axis[idx]) / (axis[idx + 1] - axis[idx])
        t = np.clip(t, 0.0, 1.0)
        return idx, t, inside

    def _axis_resolution(self, axis):
        diffs = np.diff(axis)
        return float(np.median(diffs))

    def _as_points(self, point):
        points = np.asarray(point, dtype=np.float64)
        if points.ndim == 1:
            if points.shape[0] != 3:
                raise ValueError("point must have shape (3,)")
            return points.reshape(1, 3), True
        if points.ndim == 2 and points.shape[1] == 3:
            return points, False
        raise ValueError("point must have shape (3,) or (N, 3)")
