import threading
import time
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
            For a batch: (distances, gradients, valid), with shapes
            (N,), (N, 3), (N,).
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
                    corner_distance = self.distance[ix + dx, iy + dy, iz + dz]
                    corner_gradient = self.gradient[ix + dx, iy + dy, iz + dz]
                    interp_distance += weight * corner_distance
                    interp_gradient += weight[:, None] * corner_gradient

        if not self.require_all_corners_valid:
            interp_valid &= np.isfinite(interp_distance)

        distances[interp_valid] = interp_distance[interp_valid]
        gradients[interp_valid] = interp_gradient[interp_valid]
        return distances, gradients, interp_valid

    def query_diagnostics(self, points):
        """Explain offline-grid validity for a batch of query points.

        A trilinear query is valid only when the point is inside the grid and,
        with the conservative default policy, all eight interpolation corners
        were observed when the ESDF was exported.
        """
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")

        ix, _, inside_x = self._axis_indices(self.xs, points[:, 0])
        iy, _, inside_y = self._axis_indices(self.ys, points[:, 1])
        iz, _, inside_z = self._axis_indices(self.zs, points[:, 2])
        inside_bounds = inside_x & inside_y & inside_z
        valid_corner_count = np.zeros(points.shape[0], dtype=np.int8)
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    valid_corner_count += self.valid[ix + dx, iy + dy, iz + dz].astype(
                        np.int8
                    )

        reason = np.full(points.shape[0], "valid", dtype=object)
        reason[~inside_bounds] = "outside_grid_bounds"
        if self.require_all_corners_valid:
            reason[inside_bounds & (valid_corner_count < 8)] = (
                "unobserved_interpolation_corner"
            )
        else:
            distances, _, query_valid = self.query_batch(points)
            reason[inside_bounds & (~query_valid | ~np.isfinite(distances))] = (
                "invalid_interpolation"
            )
        return {
            "inside_bounds": inside_bounds,
            "valid_corner_count": valid_corner_count,
            "reason": reason,
        }

    def _validate(self):
        if self.xs.ndim != 1 or self.ys.ndim != 1 or self.zs.ndim != 1:
            raise ValueError("ESDF axes xs, ys, zs must be one-dimensional")
        if min(len(self.xs), len(self.ys), len(self.zs)) < 2:
            raise ValueError("ESDF axes must contain at least two samples")
        for axis_name, axis in (
            ("xs", self.xs),
            ("ys", self.ys),
            ("zs", self.zs),
        ):
            if not np.all(np.diff(axis) > 0.0):
                raise ValueError(f"ESDF axis {axis_name} must be strictly increasing")

        expected_shape = (len(self.xs), len(self.ys), len(self.zs))
        if self.distance.shape != expected_shape:
            raise ValueError(
                f"distance shape {self.distance.shape} does not match "
                f"{expected_shape}"
            )
        if self.valid.shape != expected_shape:
            raise ValueError(
                f"valid shape {self.valid.shape} does not match " f"{expected_shape}"
            )
        if self.gradient.shape != expected_shape + (3,):
            expected_gradient_shape = expected_shape + (3,)
            raise ValueError(
                f"gradient shape {self.gradient.shape} does not match "
                f"{expected_gradient_shape}"
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


class NvbloxUnavailableError(RuntimeError):
    """Raised when the online nvblox backend cannot be initialized."""


def _load_nvblox_torch():
    try:
        import torch
        from nvblox_torch.constants import constants
        from nvblox_torch.mapper import Mapper, QueryType
        from nvblox_torch.projective_integrator_types import ProjectiveIntegratorType
        from nvblox_torch.sensor import Sensor
    except ImportError as exc:
        raise NvbloxUnavailableError(
            "OnlineNvbloxESDFMap requires nvblox_torch and torch. "
            "Install pixi/default dependencies before enabling online ESDF."
        ) from exc

    return (
        torch,
        constants,
        Mapper,
        QueryType,
        ProjectiveIntegratorType,
        Sensor,
    )


class OnlineNvbloxESDFMap:
    """Online nvblox-torch ESDF provider with the same query API as ESDFMap."""

    def __init__(
        self,
        voxel_size=0.05,
        integrator_type="tsdf",
        device="cuda",
        mapper_id=0,
        query_radius=0.0,
        unknown_distance_threshold=None,
        auto_update_esdf=True,
        update_esdf_on_depth=False,
        camera=None,
        initial_map_path=None,
    ):
        (
            self._torch,
            constants,
            Mapper,
            self._QueryType,
            ProjectiveIntegratorType,
            self._Sensor,
        ) = _load_nvblox_torch()

        self.device = self._torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("OnlineNvbloxESDFMap requires a CUDA device")
        if not self._torch.cuda.is_available():
            raise NvbloxUnavailableError(
                "OnlineNvbloxESDFMap requires torch.cuda.is_available()"
            )

        self.voxel_size = float(voxel_size)
        self.mapper_id = int(mapper_id)
        self.query_radius = float(query_radius)
        self.unknown_distance_threshold = float(
            constants.esdf_unknown_distance()
            if unknown_distance_threshold is None
            else unknown_distance_threshold
        )
        self.auto_update_esdf = bool(auto_update_esdf)
        self.update_esdf_on_depth = bool(update_esdf_on_depth)
        self.sensor = self._make_sensor(camera) if camera is not None else None

        integrator = self._parse_integrator_type(
            integrator_type, ProjectiveIntegratorType
        )
        self.mapper = Mapper(self.voxel_size, integrator)
        self.bounds = None
        self.resolution = np.array([self.voxel_size] * 3, dtype=np.float64)

        self._lock = threading.RLock()
        self._has_depth = False
        self._has_esdf = False
        self._pending_esdf_update = False
        self._timing = {
            "add_depth_frame_time": 0.0,
            "decay_time": 0.0,
            "decay_count": 0,
            "update_esdf_time": 0.0,
            "query_layer_time": 0.0,
            "query_total_time": 0.0,
            "query_count": 0,
        }

        if initial_map_path is not None:
            self.load_map(initial_map_path)

    @classmethod
    def from_config(cls, config):
        online_config = dict(config.get("online_nvblox", {}))
        for key in (
            "voxel_size",
            "integrator_type",
            "device",
            "mapper_id",
            "query_radius",
            "unknown_distance_threshold",
            "auto_update_esdf",
            "update_esdf_on_depth",
            "camera",
            "initial_map_path",
        ):
            if key in config and key not in online_config:
                online_config[key] = config[key]
        return cls(**online_config)

    def add_depth_frame(
        self,
        depth_frame,
        t_w_c,
        sensor=None,
        mask_frame=None,
        mapper_id=None,
        update_esdf=None,
    ):
        """Integrate one depth frame into the online TSDF/ESDF map."""
        mapper_id = self.mapper_id if mapper_id is None else int(mapper_id)
        sensor = self._resolve_sensor(sensor)
        depth = self._as_device_tensor(depth_frame, self._torch.float32, "depth_frame")
        pose = self._as_pose_tensor(t_w_c)
        mask = (
            None
            if mask_frame is None
            else self._as_device_tensor(mask_frame, self._torch.uint8, "mask_frame")
        )

        with self._lock:
            t1 = time.perf_counter()
            self.mapper.add_depth_frame(depth, pose, sensor, mask, mapper_id)
            self._timing["add_depth_frame_time"] = time.perf_counter() - t1
            self._has_depth = True
            self._pending_esdf_update = True
            if update_esdf is None:
                should_update = self.update_esdf_on_depth
            else:
                should_update = update_esdf
            if should_update:
                self._update_esdf_locked(mapper_id)

    integrate_depth_frame = add_depth_frame

    def add_color_frame(
        self,
        color_frame,
        t_w_c,
        sensor=None,
        mask_frame=None,
        mapper_id=None,
    ):
        mapper_id = self.mapper_id if mapper_id is None else int(mapper_id)
        sensor = self._resolve_sensor(sensor)
        color = self._as_device_tensor(color_frame, self._torch.uint8, "color_frame")
        pose = self._as_pose_tensor(t_w_c)
        mask = (
            None
            if mask_frame is None
            else self._as_device_tensor(mask_frame, self._torch.uint8, "mask_frame")
        )

        with self._lock:
            self.mapper.add_color_frame(color, pose, sensor, mask, mapper_id)

    def update_esdf(self, mapper_id=None):
        mapper_id = self.mapper_id if mapper_id is None else int(mapper_id)
        with self._lock:
            return self._update_esdf_locked(mapper_id)

    def query(self, point):
        """Query one point or a batch of points in world frame."""
        points, single = self._as_points(point)
        distances, gradients, valid = self.query_batch(points)
        if single:
            return float(distances[0]), gradients[0], bool(valid[0])
        return distances, gradients, valid

    def query_batch(self, points):
        query_total_start = time.perf_counter()
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")

        with self._lock:
            self._timing["query_count"] = int(points.shape[0])
            if self.auto_update_esdf and self._pending_esdf_update:
                self._update_esdf_locked(self.mapper_id)
            if not self._has_esdf:
                self._timing["query_layer_time"] = 0.0
                self._timing["query_total_time"] = (
                    time.perf_counter() - query_total_start
                )
                return self._invalid_result(points.shape[0])

            query = self._make_query_tensor(points)
            query_layer_start = time.perf_counter()
            out = self.mapper.query_layer(
                self._QueryType.ESDF_GRAD, query, mapper_id=self.mapper_id
            )
            self._timing["query_layer_time"] = time.perf_counter() - query_layer_start

        out_np = out.detach().cpu().numpy()
        raw_gradients = out_np[:, :3].astype(np.float64, copy=False)
        raw_distances = out_np[:, 3].astype(np.float64, copy=False)
        valid = (
            np.isfinite(raw_distances)
            & np.all(np.isfinite(raw_gradients), axis=1)
            & (np.abs(raw_distances) < self.unknown_distance_threshold)
        )

        distances, gradients, _ = self._invalid_result(points.shape[0])
        distances[valid] = raw_distances[valid]
        gradients[valid] = raw_gradients[valid]
        with self._lock:
            self._timing["query_total_time"] = time.perf_counter() - query_total_start
        return distances, gradients, valid

    def save_map(self, path, mapper_id=None):
        mapper_id = self.mapper_id if mapper_id is None else int(mapper_id)
        with self._lock:
            self.mapper.save_map(str(path), mapper_id)

    def load_map(self, path, mapper_id=None, update_esdf=True):
        mapper_id = self.mapper_id if mapper_id is None else int(mapper_id)
        with self._lock:
            self.mapper.load_from_file(str(path), mapper_id)
            self._has_depth = True
            self._pending_esdf_update = True
            if update_esdf:
                self._update_esdf_locked(mapper_id)

    def clear(self, mapper_id=None):
        mapper_id = self.mapper_id if mapper_id is None else int(mapper_id)
        with self._lock:
            self.mapper.clear(mapper_id)
            self._has_depth = False
            self._has_esdf = False
            self._pending_esdf_update = False

    def decay(self, mapper_id=None):
        mapper_id = self.mapper_id if mapper_id is None else int(mapper_id)
        with self._lock:
            if not self._has_depth:
                self._timing["decay_time"] = 0.0
                self._timing["decay_count"] = 0
                return False
            t1 = time.perf_counter()
            self.mapper.decay(mapper_id)
            self._timing["decay_time"] = time.perf_counter() - t1
            self._timing["decay_count"] = 1
            self._pending_esdf_update = True
            return True

    @property
    def last_timing(self):
        with self._lock:
            return dict(self._timing)

    @property
    def has_esdf(self):
        return self._has_esdf

    def _update_esdf_locked(self, mapper_id):
        if not self._has_depth:
            self._timing["update_esdf_time"] = 0.0
            return False
        t1 = time.perf_counter()
        self.mapper.update_esdf(mapper_id)
        self._timing["update_esdf_time"] = time.perf_counter() - t1
        self._has_esdf = True
        self._pending_esdf_update = False
        return True

    def _make_query_tensor(self, points):
        query_np = np.empty((points.shape[0], 4), dtype=np.float32)
        query_np[:, :3] = points.astype(np.float32, copy=False)
        query_np[:, 3] = self.query_radius
        return self._torch.as_tensor(
            query_np, device=self.device, dtype=self._torch.float32
        )

    def _as_device_tensor(self, value, dtype, field_name):
        tensor = self._torch.as_tensor(value, device=self.device, dtype=dtype)
        if field_name in ("depth_frame", "mask_frame") and tensor.ndim != 2:
            raise ValueError(f"{field_name} must have shape (H, W)")
        if field_name == "color_frame" and (tensor.ndim != 3 or tensor.shape[2] != 3):
            raise ValueError("color_frame must have shape (H, W, 3)")
        return tensor

    def _as_pose_tensor(self, t_w_c):
        pose = self._torch.as_tensor(t_w_c, dtype=self._torch.float32, device="cpu")
        if pose.shape != (4, 4):
            raise ValueError("t_w_c must have shape (4, 4)")
        return pose

    def _resolve_sensor(self, sensor):
        if sensor is None:
            sensor = self.sensor
        elif isinstance(sensor, dict):
            sensor = self._make_sensor(sensor)
        if sensor is None:
            raise ValueError(
                "A nvblox Sensor or camera config must be supplied before "
                "integrating frames"
            )
        return sensor

    def _make_sensor(self, camera):
        if hasattr(camera, "get_c_sensor"):
            return camera
        if not isinstance(camera, dict):
            raise TypeError("camera must be a nvblox Sensor or a dict")

        fu = camera.get("fu", camera.get("fx"))
        fv = camera.get("fv", camera.get("fy"))
        cu = camera.get("cu", camera.get("cx"))
        cv = camera.get("cv", camera.get("cy"))
        width = camera.get("width")
        height = camera.get("height")
        missing = [
            name
            for name, value in (
                ("fu/fx", fu),
                ("fv/fy", fv),
                ("cu/cx", cu),
                ("cv/cy", cv),
                ("width", width),
                ("height", height),
            )
            if value is None
        ]
        if missing:
            raise ValueError(f"camera config is missing: {', '.join(missing)}")
        return self._Sensor.from_camera(
            float(fu), float(fv), float(cu), float(cv), int(width), int(height)
        )

    def _invalid_result(self, n):
        distances = np.full(n, np.nan, dtype=np.float64)
        gradients = np.full((n, 3), np.nan, dtype=np.float64)
        valid = np.zeros(n, dtype=bool)
        return distances, gradients, valid

    def _as_points(self, point):
        points = np.asarray(point, dtype=np.float64)
        if points.ndim == 1:
            if points.shape[0] != 3:
                raise ValueError("point must have shape (3,)")
            return points.reshape(1, 3), True
        if points.ndim == 2 and points.shape[1] == 3:
            return points, False
        raise ValueError("point must have shape (3,) or (N, 3)")

    def _parse_integrator_type(self, integrator_type, ProjectiveIntegratorType):
        if isinstance(integrator_type, ProjectiveIntegratorType):
            return integrator_type
        value = str(integrator_type).lower()
        if value == "tsdf":
            return ProjectiveIntegratorType.TSDF
        if value == "occupancy":
            return ProjectiveIntegratorType.OCCUPANCY
        raise ValueError(
            "integrator_type must be 'tsdf' or 'occupancy', " f"got {integrator_type!r}"
        )
