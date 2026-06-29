import argparse
import datetime
import logging
import math
import os
import time

import numpy as np
import pybullet as pyb
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as Rot

import mm_control.MPC as MPC
from mm_plan.TaskManager import TaskManager
from mm_simulator import simulation
from mm_utils import parsing
from mm_utils.logging import DataLogger

from export_nvblox_esdf import make_base_spin_camera_poses, render_camera_pose


class CameraPreview:
    def __init__(self, config):
        preview_config = config.get("preview", {})
        self.enabled = bool(preview_config.get("enabled", False))
        self.show_initial_scan = bool(
            preview_config.get("show_initial_scan", True)
        )
        self.show_realtime_scan = bool(
            preview_config.get("show_realtime_scan", True)
        )
        self.mode = str(preview_config.get("mode", "rgb_depth")).lower()
        self.window_name = str(
            preview_config.get("window_name", "online nvblox camera")
        )
        self.wait_ms = int(preview_config.get("wait_ms", 1))
        self.scale = float(preview_config.get("scale", 1.0))
        self._cv2 = None

        if not self.enabled:
            return
        display_available = (
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        )
        if not display_available:
            logging.getLogger("Controller").warning(
                "online_nvblox_sim.preview.enabled is true, but no display "
                "server is available; disabling camera preview"
            )
            self.enabled = False
            return

        try:
            import cv2
        except ImportError:
            logging.getLogger("Controller").warning(
                "online_nvblox_sim.preview.enabled is true, but OpenCV is not "
                "available; disabling camera preview"
            )
            self.enabled = False
            return

        self._cv2 = cv2

    def show(self, phase, name, rgb, depth, mask, far):
        if not self.enabled:
            return
        if phase == "initial" and not self.show_initial_scan:
            return
        if phase == "realtime" and not self.show_realtime_scan:
            return

        image = self._compose_image(rgb, depth, mask, far)
        if self.scale > 0.0 and self.scale != 1.0:
            image = self._cv2.resize(
                image,
                None,
                fx=self.scale,
                fy=self.scale,
                interpolation=self._cv2.INTER_NEAREST,
            )

        self._cv2.imshow(self.window_name, image)
        self._cv2.setWindowTitle(self.window_name, f"{phase}: {name}")
        self._cv2.waitKey(max(1, self.wait_ms))

    def close(self):
        if self.enabled and self._cv2 is not None:
            self._cv2.destroyWindow(self.window_name)

    def _compose_image(self, rgb, depth, mask, far):
        rgb_bgr = self._cv2.cvtColor(rgb, self._cv2.COLOR_RGB2BGR)
        if self.mode == "rgb":
            return rgb_bgr

        depth_vis = self._depth_to_color(depth, mask, far)
        if self.mode == "depth":
            return depth_vis
        if self.mode != "rgb_depth":
            raise ValueError(
                "online_nvblox_sim.preview.mode must be 'rgb', 'depth', "
                "or 'rgb_depth'"
            )
        return np.concatenate([rgb_bgr, depth_vis], axis=1)

    def _depth_to_color(self, depth, mask, far):
        max_depth = max(float(far), 1.0e-6)
        depth_norm = np.clip(depth / max_depth, 0.0, 1.0)
        depth_u8 = np.asarray(255.0 * (1.0 - depth_norm), dtype=np.uint8)
        depth_u8 = np.where(mask > 0, depth_u8, 0).astype(np.uint8)
        return self._cv2.applyColorMap(depth_u8, self._cv2.COLORMAP_TURBO)


class OnlineNvbloxDiagnostics:
    def __init__(self, sim, config, controller):
        diag_config = config.get("diagnostics", {})
        self.enabled = bool(diag_config.get("enabled", False))
        self.unknown_distance = float(
            diag_config.get("unknown_distance", 100.0)
        )
        self.unknown_tolerance = float(
            diag_config.get("unknown_distance_tolerance", 1.0e-3)
        )
        self.base_sphere_name = str(
            diag_config.get("base_sphere_name", "base_body_collision")
        )
        target_names = diag_config.get(
            "target_link_names",
            [
                "desk_right_leg_fr",
                "desk_right_leg_fl",
                "desk_right_leg_br",
                "desk_right_leg_bl",
            ],
        )
        self.target_names = [str(name) for name in target_names]
        self.target_specs = self._resolve_targets(sim)
        self.sphere_names = list(getattr(controller, "esdf_sphere_names", []))

    @property
    def target_count(self):
        return len(self.target_names)

    def metadata(self):
        if not self.enabled:
            return {}
        return {
            "target_names": np.asarray(self.target_names, dtype=str),
            "target_body_ids": np.asarray(
                [body_uid for body_uid, _ in self.target_specs],
                dtype=np.int64,
            ),
            "target_link_indices": np.asarray(
                [link_idx for _, link_idx in self.target_specs],
                dtype=np.int64,
            ),
            "sphere_names": np.asarray(self.sphere_names, dtype=str),
        }

    def zero_stats(self):
        if not self.enabled:
            return {}
        return {
            "enabled": 1,
            "target_resolved": np.asarray(
                [
                    1 if body_uid >= 0 and link_idx >= -1 else 0
                    for body_uid, link_idx in self.target_specs
                ],
                dtype=np.int64,
            ),
            "target_visible_pixels": np.zeros(
                self.target_count, dtype=np.int64
            ),
            "target_integrated_pixels": np.zeros(
                self.target_count, dtype=np.int64
            ),
            "target_visible_pixels_total": 0,
            "target_integrated_pixels_total": 0,
            "target_visible_pixels_max": 0,
            "target_integrated_pixels_max": 0,
            "target_visible_min_z": np.full(
                self.target_count, np.nan, dtype=np.float64
            ),
            "target_visible_mean_z": np.full(
                self.target_count, np.nan, dtype=np.float64
            ),
            "target_visible_max_z": np.full(
                self.target_count, np.nan, dtype=np.float64
            ),
            "target_integrated_min_z": np.full(
                self.target_count, np.nan, dtype=np.float64
            ),
            "target_integrated_mean_z": np.full(
                self.target_count, np.nan, dtype=np.float64
            ),
            "target_integrated_max_z": np.full(
                self.target_count, np.nan, dtype=np.float64
            ),
            "base_esdf_node0_distance": np.nan,
            "base_esdf_node0_margin": np.nan,
            "base_esdf_node0_unknown": False,
            "base_esdf_min_horizon_distance": np.nan,
            "base_esdf_min_horizon_margin": np.nan,
            "min_sphere_node0_margin": np.nan,
            "min_sphere_horizon_margin": np.nan,
            "min_sphere_horizon_index": -1,
        }

    def add_render_stats(
        self,
        stats,
        segmentation,
        visible_mask,
        integrated_mask,
        world_z,
    ):
        if not self.enabled or segmentation is None:
            return

        visible_counts, visible_z = self._target_pixel_stats(
            segmentation, visible_mask, world_z
        )
        integrated_counts, integrated_z = self._target_pixel_stats(
            segmentation, integrated_mask, world_z
        )
        old_visible_counts = stats["target_visible_pixels"].copy()
        old_integrated_counts = stats["target_integrated_pixels"].copy()
        stats["target_visible_pixels"] += visible_counts
        stats["target_integrated_pixels"] += integrated_counts
        stats["target_visible_pixels_total"] = int(
            np.sum(stats["target_visible_pixels"])
        )
        stats["target_integrated_pixels_total"] = int(
            np.sum(stats["target_integrated_pixels"])
        )
        stats["target_visible_pixels_max"] = int(
            np.max(stats["target_visible_pixels"])
            if self.target_count
            else 0
        )
        stats["target_integrated_pixels_max"] = int(
            np.max(stats["target_integrated_pixels"])
            if self.target_count
            else 0
        )
        self._update_target_z_stats(
            stats,
            "target_visible",
            old_visible_counts,
            visible_counts,
            visible_z,
        )
        self._update_target_z_stats(
            stats,
            "target_integrated",
            old_integrated_counts,
            integrated_counts,
            integrated_z,
        )

    def add_controller_stats(self, stats, controller):
        if not self.enabled:
            return

        log = getattr(controller, "log", {})
        node0_distances = np.asarray(
            log.get("esdf_node0_distances", []), dtype=float
        )
        node0_margins = np.asarray(
            log.get("esdf_node0_margins", []), dtype=float
        )
        min_distances = np.asarray(
            log.get("esdf_min_distance_per_sphere", []), dtype=float
        )
        min_margins = np.asarray(
            log.get("esdf_min_margin_per_sphere", []), dtype=float
        )
        sphere_names = list(getattr(controller, "esdf_sphere_names", []))
        if self.base_sphere_name in sphere_names:
            base_idx = sphere_names.index(self.base_sphere_name)
        else:
            base_idx = 0 if sphere_names else -1

        if base_idx >= 0 and base_idx < node0_distances.size:
            base_distance = float(node0_distances[base_idx])
            stats["base_esdf_node0_distance"] = base_distance
            if base_idx < node0_margins.size:
                stats["base_esdf_node0_margin"] = float(
                    node0_margins[base_idx]
                )
            stats["base_esdf_node0_unknown"] = bool(
                np.isfinite(base_distance)
                and abs(base_distance - self.unknown_distance)
                <= self.unknown_tolerance
            )

        if base_idx >= 0 and base_idx < min_distances.size:
            stats["base_esdf_min_horizon_distance"] = float(
                min_distances[base_idx]
            )
        if base_idx >= 0 and base_idx < min_margins.size:
            stats["base_esdf_min_horizon_margin"] = float(
                min_margins[base_idx]
            )
        if node0_margins.size and np.any(np.isfinite(node0_margins)):
            stats["min_sphere_node0_margin"] = float(
                np.nanmin(node0_margins)
            )
        if min_margins.size and np.any(np.isfinite(min_margins)):
            min_idx = int(np.nanargmin(min_margins))
            stats["min_sphere_horizon_margin"] = float(min_margins[min_idx])
            stats["min_sphere_horizon_index"] = min_idx

    def _resolve_targets(self, sim):
        if not self.enabled:
            return []

        link_map = {}
        for body_uid in self._diagnostic_body_ids(sim):
            link_map.update(self._body_link_map(body_uid))

        specs = []
        for name in self.target_names:
            specs.append(link_map.get(name, (-1, -2)))
        unresolved = [
            name
            for name, (body_uid, link_idx) in zip(self.target_names, specs)
            if body_uid < 0 or link_idx < -1
        ]
        if unresolved:
            logging.getLogger("Controller").warning(
                "online nvblox diagnostics could not resolve target links: %s",
                ", ".join(unresolved),
            )
        return specs

    @staticmethod
    def _diagnostic_body_ids(sim):
        body_ids = []
        static_uid = getattr(sim, "static_obstacles_uid", None)
        if static_uid is not None:
            body_ids.append(int(static_uid))
        for body_idx in range(pyb.getNumBodies()):
            body_uid = pyb.getBodyUniqueId(body_idx)
            if body_uid not in body_ids:
                body_ids.append(body_uid)
        return body_ids

    @staticmethod
    def _body_link_map(body_uid):
        link_map = {}
        try:
            base_name = pyb.getBodyInfo(body_uid)[0].decode("utf-8")
        except Exception:
            base_name = ""
        if base_name:
            link_map[base_name] = (int(body_uid), -1)

        for link_idx in range(pyb.getNumJoints(body_uid)):
            info = pyb.getJointInfo(body_uid, link_idx)
            link_name = info[12].decode("utf-8")
            link_map[link_name] = (int(body_uid), int(link_idx))
        return link_map

    @staticmethod
    def _combine_mean_z(old_mean, old_count, new_sum, new_count):
        total_count = int(old_count) + int(new_count)
        if total_count <= 0:
            return np.nan
        old_sum = 0.0
        if int(old_count) > 0 and np.isfinite(old_mean):
            old_sum = float(old_mean) * int(old_count)
        return (old_sum + float(new_sum)) / total_count

    @staticmethod
    def _combine_min_z(old_min, new_min):
        if not np.isfinite(old_min):
            return new_min
        if not np.isfinite(new_min):
            return old_min
        return min(float(old_min), float(new_min))

    @staticmethod
    def _combine_max_z(old_max, new_max):
        if not np.isfinite(old_max):
            return new_max
        if not np.isfinite(new_max):
            return old_max
        return max(float(old_max), float(new_max))

    def _update_target_z_stats(
        self,
        stats,
        prefix,
        old_counts,
        new_counts,
        new_z_stats,
    ):
        if self.target_count == 0:
            return
        min_key = f"{prefix}_min_z"
        mean_key = f"{prefix}_mean_z"
        max_key = f"{prefix}_max_z"
        for idx in range(self.target_count):
            count = int(new_counts[idx])
            if count <= 0:
                continue
            stats[min_key][idx] = self._combine_min_z(
                stats[min_key][idx], new_z_stats["min"][idx]
            )
            stats[max_key][idx] = self._combine_max_z(
                stats[max_key][idx], new_z_stats["max"][idx]
            )
            stats[mean_key][idx] = self._combine_mean_z(
                stats[mean_key][idx],
                old_counts[idx],
                new_z_stats["sum"][idx],
                count,
            )

    def _target_pixel_stats(self, segmentation, mask, world_z):
        counts = np.zeros(self.target_count, dtype=np.int64)
        z_stats = {
            "min": np.full(self.target_count, np.nan, dtype=np.float64),
            "sum": np.zeros(self.target_count, dtype=np.float64),
            "max": np.full(self.target_count, np.nan, dtype=np.float64),
        }
        if self.target_count == 0:
            return counts, z_stats

        seg = np.asarray(segmentation, dtype=np.int64)
        valid = np.asarray(mask).astype(bool)
        z_w = np.asarray(world_z, dtype=np.float64)
        body_uid = seg & ((1 << 24) - 1)
        link_idx = (seg >> 24) - 1
        for idx, (target_body_uid, target_link_idx) in enumerate(
            self.target_specs
        ):
            if target_body_uid < 0 or target_link_idx < -1:
                continue
            target_mask = (
                valid
                & (body_uid == int(target_body_uid))
                & (link_idx == int(target_link_idx))
            )
            counts[idx] = int(np.count_nonzero(target_mask))
            if counts[idx] <= 0:
                continue
            target_z = z_w[target_mask]
            target_z = target_z[np.isfinite(target_z)]
            if target_z.size == 0:
                continue
            z_stats["min"][idx] = float(np.min(target_z))
            z_stats["sum"][idx] = float(np.sum(target_z))
            z_stats["max"][idx] = float(np.max(target_z))
        return counts, z_stats


def _get_command_velocity_limits(config, expected_dim):
    state_limits = config.get("robot", {}).get("limits", {}).get("state")
    if state_limits is None:
        return None, None

    nq = config["robot"]["dims"]["q"]
    lower = parsing.parse_array(state_limits["lower"])[nq:]
    upper = parsing.parse_array(state_limits["upper"])[nq:]
    if lower.shape[0] != expected_dim or upper.shape[0] != expected_dim:
        raise ValueError(
            "Command velocity limit dimension does not match robot velocity "
            "dimension"
        )
    return lower, upper


def _renderer_from_config(config):
    renderer = str(config.get("renderer", "tiny")).lower()
    if renderer == "tiny":
        return pyb.ER_TINY_RENDERER
    if renderer == "hardware":
        return pyb.ER_BULLET_HARDWARE_OPENGL
    raise ValueError(f"Unsupported online_nvblox_sim.renderer: {renderer}")


def _camera_from_render_config(config):
    width = int(config.get("width", 320))
    height = int(config.get("height", 240))
    fov_y_deg = float(config.get("fov_y_deg", 60.0))
    fy = height / (2.0 * math.tan(math.radians(fov_y_deg) / 2.0))
    return {
        "fx": float(config.get("fx", fy)),
        "fy": float(config.get("fy", fy)),
        "cx": float(config.get("cx", (width - 1.0) / 2.0)),
        "cy": float(config.get("cy", (height - 1.0) / 2.0)),
        "width": width,
        "height": height,
    }


def _online_nvblox_stats(prefix, stats):
    return {
        f"online_nvblox_{prefix}_{key}": value
        for key, value in stats.items()
    }


def _zero_integration_stats():
    return {
        "decays": 0,
        "frames": 0,
        "valid_depth_pixels": 0,
        "filtered_depth_pixels": 0,
        "decay_time": 0.0,
        "render_time": 0.0,
        "filter_time": 0.0,
        "preview_time": 0.0,
        "integrate_time": 0.0,
        "update_time": 0.0,
        "total_time": 0.0,
    }


def _zero_map_timing():
    return {
        "add_depth_frame_time": 0.0,
        "decay_time": 0.0,
        "decay_count": 0,
        "update_esdf_time": 0.0,
        "query_layer_time": 0.0,
        "query_total_time": 0.0,
        "query_count": 0,
    }


def _online_map_timing(controller):
    esdf_map = getattr(controller, "esdf_map", None)
    if esdf_map is None or not hasattr(esdf_map, "last_timing"):
        return _zero_map_timing()
    timing = _zero_map_timing()
    timing.update(esdf_map.last_timing)
    return timing


def _append_prefixed_stats(logger, prefix, stats):
    for key, value in stats.items():
        logger.append(f"{prefix}_{key}", value)


def _exclude_body_ids(sim, config):
    body_ids = []
    if config.get("exclude_robot", True):
        body_ids.append(sim.robot.uid)
    if config.get("exclude_collision_sphere_markers", True):
        body_ids.extend(getattr(sim, "collision_sphere_marker_bodies", []))
    return body_ids


def _depth_world_z(depth, t_w_c, camera):
    height, width = depth.shape
    u = np.arange(width, dtype=np.float32)[None, :]
    v = np.arange(height, dtype=np.float32)[:, None]
    z_c = depth.astype(np.float32, copy=False)
    x_c = ((u - float(camera["cx"])) / float(camera["fx"])) * z_c
    y_c = ((v - float(camera["cy"])) / float(camera["fy"])) * z_c

    return (
        float(t_w_c[2, 0]) * x_c
        + float(t_w_c[2, 1]) * y_c
        + float(t_w_c[2, 2]) * z_c
        + float(t_w_c[2, 3])
    )


def _filter_depth_by_world_z(depth, mask, t_w_c, camera, min_z, world_z=None):
    if min_z is None:
        return depth, mask, 0

    min_z = float(min_z)
    if not np.isfinite(min_z):
        return depth, mask, 0

    valid = mask.astype(bool) & (depth > 0.0)
    if not np.any(valid):
        return depth, mask, 0

    if world_z is None:
        world_z = _depth_world_z(depth, t_w_c, camera)
    keep = valid & (world_z > min_z)
    filtered = int(np.count_nonzero(valid) - np.count_nonzero(keep))
    if filtered <= 0:
        return depth, mask, 0

    depth = np.where(keep, depth, 0.0).astype(np.float32)
    mask = keep.astype(np.uint8)
    return depth, mask, filtered


def _online_esdf_map(controller):
    esdf_map = getattr(controller, "esdf_map", None)
    if esdf_map is None or not hasattr(esdf_map, "add_depth_frame"):
        raise RuntimeError(
            "online_nvblox_sim is enabled, but controller.esdf_map is not an "
            "OnlineNvbloxESDFMap. Set controller.esdf_collision.source to "
            "'online_nvblox'."
        )
    return esdf_map


def _run_online_decay(controller, config, step_idx):
    decay_config = config.get("decay", {})
    if not bool(decay_config.get("enabled", False)):
        return 0.0, 0

    interval = int(decay_config.get("interval_steps", 1))
    if interval <= 0 or step_idx % interval != 0:
        return 0.0, 0

    esdf_map = _online_esdf_map(controller)
    decay_start = time.perf_counter()
    decayed = esdf_map.decay()
    decay_time = time.perf_counter() - decay_start
    return decay_time, 1 if decayed else 0


def _integrate_camera_poses(
    controller,
    sim,
    config,
    camera_poses,
    update_esdf,
    preview,
    phase,
    diagnostics,
):
    total_start = time.perf_counter()
    esdf_map = _online_esdf_map(controller)
    renderer = _renderer_from_config(config)
    width = int(config.get("width", 320))
    height = int(config.get("height", 240))
    fov_y_deg = float(config.get("fov_y_deg", 60.0))
    near = float(config.get("near", 0.05))
    far = float(config.get("far", 4.0))
    exclude_body_ids = _exclude_body_ids(sim, config)
    camera = _camera_from_render_config(config)
    sensor = None if getattr(esdf_map, "sensor", None) is not None else camera
    ground_filter_min_z = config.get("ground_filter_min_z")

    stats = _zero_integration_stats()
    diagnostic_stats = diagnostics.zero_stats()
    for name, t_w_c in camera_poses:
        render_start = time.perf_counter()
        if diagnostics.enabled:
            rgb, depth, mask, segmentation = render_camera_pose(
                width,
                height,
                fov_y_deg,
                near,
                far,
                t_w_c,
                renderer,
                exclude_body_ids,
                return_segmentation=True,
            )
        else:
            rgb, depth, mask = render_camera_pose(
                width,
                height,
                fov_y_deg,
                near,
                far,
                t_w_c,
                renderer,
                exclude_body_ids,
            )
            segmentation = None
        visible_mask = mask
        world_z = _depth_world_z(depth, t_w_c, camera)
        stats["render_time"] += time.perf_counter() - render_start
        filter_start = time.perf_counter()
        depth, mask, filtered = _filter_depth_by_world_z(
            depth, mask, t_w_c, camera, ground_filter_min_z, world_z
        )
        diagnostics.add_render_stats(
            diagnostic_stats, segmentation, visible_mask, mask, world_z
        )
        stats["filter_time"] += time.perf_counter() - filter_start
        stats["filtered_depth_pixels"] += filtered
        stats["valid_depth_pixels"] += int(np.count_nonzero(depth > 0.0))
        if preview is not None:
            preview_start = time.perf_counter()
            preview.show(phase, name, rgb, depth, mask, far)
            stats["preview_time"] += time.perf_counter() - preview_start

        integrate_start = time.perf_counter()
        esdf_map.add_depth_frame(
            depth,
            t_w_c,
            sensor=sensor,
            mask_frame=mask,
            update_esdf=False,
        )
        stats["integrate_time"] += time.perf_counter() - integrate_start
        stats["frames"] += 1

    if update_esdf and stats["frames"] > 0:
        update_start = time.perf_counter()
        esdf_map.update_esdf()
        stats["update_time"] = time.perf_counter() - update_start

    stats["total_time"] = time.perf_counter() - total_start
    return stats, diagnostic_stats


def _initial_scan_origins(scan_config, robot):
    origins = scan_config.get("origins")
    if origins is not None:
        return np.asarray(origins, dtype=float).reshape((-1, 3))
    q, _ = robot.joint_states(add_noise=False)
    return q[:3].reshape((1, 3))


def _pose_matrix(pos, quat_xyzw):
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = Rot.from_quat(quat_xyzw).as_matrix()
    pose[:3, 3] = pos
    return pose


def _local_rpy_transform(rpy):
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = Rot.from_euler(
        "xyz", np.asarray(rpy, dtype=float)
    ).as_matrix()
    return transform


def _camera_link_pose(robot, link_name, frame_convention, correction_rpy):
    if link_name not in robot.links:
        available = ", ".join(sorted(robot.links))
        raise ValueError(
            f"Unknown camera link '{link_name}'. Available links: {available}"
        )
    link_idx = robot.links[link_name][0]
    pos, quat_xyzw = robot.link_pose(link_idx)
    t_w_c = _pose_matrix(pos, quat_xyzw)

    convention = str(frame_convention).lower()
    if convention in {"optical", "ros_optical", "depth_optical"}:
        pass
    elif convention in {"ros_link", "camera_link", "link"}:
        t_w_c = t_w_c @ _local_rpy_transform(
            [-math.pi / 2.0, 0.0, -math.pi / 2.0]
        )
    else:
        raise ValueError(
            "camera_frame_convention must be 'optical' or 'ros_link'"
        )

    t_w_c = t_w_c @ _local_rpy_transform(correction_rpy)
    return t_w_c


def _realtime_camera_configs(scan_config):
    camera_entries = scan_config.get("cameras")
    if camera_entries is None:
        return [scan_config]
    if not isinstance(camera_entries, list):
        raise TypeError("online_nvblox_sim.realtime_scan.cameras must be a list")

    shared_config = {
        key: value for key, value in scan_config.items() if key != "cameras"
    }
    camera_configs = []
    for idx, entry in enumerate(camera_entries):
        if not isinstance(entry, dict):
            raise TypeError(
                "Each online_nvblox_sim.realtime_scan.cameras entry must "
                "be a mapping"
            )
        camera_config = dict(shared_config)
        camera_config.update(entry)
        camera_config.setdefault("name", f"camera{idx}")
        if not bool(camera_config.get("enabled", True)):
            continue
        camera_configs.append(camera_config)
    return camera_configs


def _realtime_camera_poses_from_config(
    sim, scan_config, base_pose, step_idx, interval
):
    pose_source = str(scan_config.get("pose_source", "camera_link")).lower()
    camera_name = str(scan_config.get("name", pose_source))
    name_prefix = f"realtime_{step_idx:06d}_{camera_name}"

    if pose_source in {"camera_link", "link", "onboard"}:
        link_name = scan_config.get(
            "camera_link_name", "camera_depth_optical_frame"
        )
        frame_convention = scan_config.get(
            "camera_frame_convention", "optical"
        )
        correction_rpy = scan_config.get(
            "pose_correction_rpy", [0.0, 0.0, 0.0]
        )
        return [
            (
                f"{name_prefix}_{link_name}",
                _camera_link_pose(
                    sim.robot, link_name, frame_convention, correction_rpy
                ),
            )
        ]

    if pose_source in {"base_spin", "base"}:
        num_views = int(scan_config.get("num_views", 1))
        yaw_offset = math.radians(
            float(scan_config.get("yaw_offset_deg", 0.0))
        )
        if scan_config.get("cycle_yaw_offset", False):
            cycle = max(1, int(scan_config.get("cycle_length", 8)))
            yaw_offset += (
                2.0 * math.pi * ((step_idx // interval) % cycle) / cycle
            )
        return make_base_spin_camera_poses(
            np.asarray(base_pose, dtype=float),
            num_views,
            float(scan_config.get("camera_height", 1.0)),
            yaw_offset,
            name_prefix=name_prefix,
        )

    raise ValueError(
        "online_nvblox_sim.realtime_scan.pose_source must be "
        "'camera_link' or 'base_spin'"
    )


def _run_initial_online_scan(controller, sim, config, preview, diagnostics):
    scan_config = config.get("initial_scan", {})
    if not scan_config.get("enabled", True):
        return _zero_integration_stats(), diagnostics.zero_stats()

    num_views = int(scan_config.get("num_views", 8))
    yaw_offset = math.radians(float(scan_config.get("yaw_offset_deg", 0.0)))
    camera_heights = scan_config.get("camera_heights", [1.0])
    update_esdf = bool(scan_config.get("update_esdf", True))
    origins = _initial_scan_origins(scan_config, sim.robot)

    camera_poses = []
    for origin_idx, origin in enumerate(origins):
        for height_idx, camera_height in enumerate(camera_heights):
            name_prefix = f"initial_p{origin_idx:02d}_h{height_idx:02d}"
            camera_poses.extend(
                make_base_spin_camera_poses(
                    origin,
                    num_views,
                    float(camera_height),
                    yaw_offset,
                    name_prefix=name_prefix,
                )
            )

    return _integrate_camera_poses(
        controller,
        sim,
        config,
        camera_poses,
        update_esdf=update_esdf,
        preview=preview,
        phase="initial",
        diagnostics=diagnostics,
    )


def _run_realtime_online_scan(
    controller, sim, config, base_pose, step_idx, preview, diagnostics
):
    scan_config = config.get("realtime_scan", {})
    if not scan_config.get("enabled", True):
        return _zero_integration_stats(), diagnostics.zero_stats()

    stats = _zero_integration_stats()
    diagnostic_stats = diagnostics.zero_stats()
    decay_time, decays = _run_online_decay(controller, config, step_idx)
    stats["decay_time"] = decay_time
    stats["decays"] = decays
    stats["total_time"] = decay_time

    interval = int(scan_config.get("render_interval_steps", 1))
    if interval <= 0 or step_idx % interval != 0:
        return stats, diagnostic_stats

    camera_poses = []
    for camera_config in _realtime_camera_configs(scan_config):
        camera_poses.extend(
            _realtime_camera_poses_from_config(
                sim, camera_config, base_pose, step_idx, interval
            )
        )
    if not camera_poses:
        return stats, diagnostic_stats

    scan_stats, diagnostic_stats = _integrate_camera_poses(
        controller,
        sim,
        config,
        camera_poses,
        update_esdf=bool(scan_config.get("update_esdf", False)),
        preview=preview,
        phase="realtime",
        diagnostics=diagnostics,
    )
    scan_stats["decay_time"] += stats["decay_time"]
    scan_stats["decays"] += stats["decays"]
    scan_stats["total_time"] += stats["total_time"]
    return scan_stats, diagnostic_stats


def _configure_loggers(config):
    ch = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)

    for name in ("Planner", "Controller", "Simulator"):
        logger = logging.getLogger(name)
        logger.setLevel(config["logging"]["log_level"])
        logger.addHandler(ch)


def main():
    np.set_printoptions(precision=3, suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", required=True, help="Path to configuration file."
    )
    parser.add_argument(
        "--video",
        nargs="?",
        default=None,
        const="",
        help="Record video. Optionally specify prefix for video directory.",
    )
    parser.add_argument(
        "--ctrl_config",
        type=str,
        default="default",
        help="controller config. This overwrites the yaml settings",
    )
    parser.add_argument(
        "--planner_config",
        type=str,
        default="default",
        help="planner config. This overwrites the yaml settings",
    )
    parser.add_argument(
        "--logging_sub_folder",
        type=str,
        default="default",
        help="save data in a sub folder of logging directory",
    )
    parser.add_argument(
        "--GUI",
        action="store_true",
        help="Pybullet GUI. This overwrites the yaml settings",
    )
    args = parser.parse_args()

    config = parsing.load_config(args.config)
    if args.ctrl_config != "default":
        ctrl_config = parsing.load_config(args.ctrl_config)
        config = parsing.recursive_dict_update(config, ctrl_config)
    if args.planner_config != "default":
        planner_config = parsing.load_config(args.planner_config)
        config = parsing.recursive_dict_update(config, planner_config)

    if args.logging_sub_folder != "default":
        config["logging"]["log_dir"] = os.path.join(
            config["logging"]["log_dir"], args.logging_sub_folder
        )

    if args.GUI:
        config["simulation"]["gui"] = True

    sim_config = config["simulation"]
    ctrl_config = config["controller"]
    planner_config = config["planner"]
    online_config = config.get("online_nvblox_sim", {})
    _configure_loggers(config)
    preview = CameraPreview(online_config)

    if (
        "limits" not in sim_config.get("robot", {})
        and "limits" in ctrl_config.get("robot", {})
    ):
        sim_config["robot"]["limits"] = ctrl_config["robot"]["limits"]
    if (
        "collision_model" not in sim_config.get("robot", {})
        and "collision_model" in ctrl_config.get("robot", {})
    ):
        sim_config["robot"]["collision_model"] = ctrl_config["robot"][
            "collision_model"
        ]

    timestamp = datetime.datetime.now()
    sim = simulation.BulletSimulation(
        config=sim_config, timestamp=timestamp, cli_args=args
    )
    robot = sim.robot

    control_class = getattr(MPC, ctrl_config["type"], None)
    if control_class is None:
        raise ValueError(f"Unknown controller type: {ctrl_config['type']}")
    controller = control_class(ctrl_config)
    diagnostics = OnlineNvbloxDiagnostics(sim, online_config, controller)

    online_enabled = bool(
        online_config.get(
            "enabled",
            hasattr(getattr(controller, "esdf_map", None), "add_depth_frame"),
        )
    )
    initial_stats = _zero_integration_stats()
    initial_diagnostic_stats = diagnostics.zero_stats()
    if online_enabled:
        _online_esdf_map(controller)
        initial_stats, initial_diagnostic_stats = _run_initial_online_scan(
            controller, sim, online_config, preview, diagnostics
        )
        controller.log.update(_online_nvblox_stats("initial", initial_stats))
        logging.getLogger("Controller").info(
            "Initial online nvblox scan integrated %d frames in %.3fs "
            "(render %.3fs, update %.3fs)",
            initial_stats["frames"],
            initial_stats["integrate_time"],
            initial_stats["render_time"],
            initial_stats["update_time"],
        )

    planner_resources = {"esdf_map": getattr(controller, "esdf_map", None)}
    sot = TaskManager(planner_config, resources=planner_resources)
    base_targets, ee_targets = sot.getVisualizationTargets()
    sim.import_target_markers(base_targets=base_targets, ee_targets=ee_targets)

    t = 0.0
    logger = DataLogger(config, name="combined")
    logger.add("sim_timestep", sim.timestep)
    logger.add("duration", sim.duration)
    logger.add("nq", sim_config["robot"]["dims"]["q"])
    logger.add("nv", sim_config["robot"]["dims"]["v"])
    logger.add("nx", sim_config["robot"]["dims"]["x"])
    logger.add("nu", sim_config["robot"]["dims"]["u"])
    for key, value in initial_stats.items():
        logger.add(f"online_nvblox_initial_{key}", value)
    for key, value in initial_diagnostic_stats.items():
        logger.add(f"online_nvblox_initial_diag_{key}", value)
    for key, value in diagnostics.metadata().items():
        logger.add(f"online_nvblox_diag_{key}", value)

    sot.activatePlanners()
    u = np.zeros(sim_config["robot"]["dims"]["v"])
    cmd_vel_lower, cmd_vel_upper = _get_command_velocity_limits(
        ctrl_config, sim_config["robot"]["dims"]["v"]
    )
    cmd_vel_clip_count = 0
    step_idx = 0
    timing_log_interval = int(
        online_config.get("timing_log_interval_steps", 0)
    )

    while t <= sim.duration:
        robot_states = robot.joint_states(add_noise=False)

        realtime_stats = _zero_integration_stats()
        diagnostic_stats = diagnostics.zero_stats()
        if online_enabled:
            realtime_stats, diagnostic_stats = _run_realtime_online_scan(
                controller,
                sim,
                online_config,
                robot_states[0][:3],
                step_idx,
                preview,
                diagnostics,
            )
            controller.log.update(
                _online_nvblox_stats("realtime", realtime_stats)
            )

        references = sot.getReferences(
            t, robot_states, controller.N + 1, controller.dt
        )

        t0 = time.perf_counter()
        v_bar, u_bar = controller.control(t, robot_states, references)
        t1 = time.perf_counter()
        map_timing = _online_map_timing(controller)
        diagnostics.add_controller_stats(diagnostic_stats, controller)
        controller.log.update(
            _online_nvblox_stats(
                "enabled", {"value": 1 if online_enabled else 0}
            )
        )
        logging.getLogger("Controller").log(
            20, f"Controller Run Time: {t1 - t0}"
        )
        if (
            online_enabled
            and timing_log_interval > 0
            and step_idx % timing_log_interval == 0
        ):
            logging.getLogger("Controller").info(
                "online nvblox step=%d frames=%d render=%.4fs "
                "decays=%d decay=%.4fs filter=%.4fs "
                "preview=%.4fs integrate=%.4fs "
                "map_add=%.4fs map_update_esdf=%.4fs "
                "map_query=%.4fs valid_depth=%d filtered_depth=%d "
                "target_visible=%d target_integrated=%d "
                "base_esdf_d=%.3f base_esdf_margin=%.3f "
                "base_unknown=%s",
                step_idx,
                realtime_stats["frames"],
                realtime_stats["render_time"],
                realtime_stats["decays"],
                realtime_stats["decay_time"],
                realtime_stats["filter_time"],
                realtime_stats["preview_time"],
                realtime_stats["integrate_time"],
                map_timing["add_depth_frame_time"],
                map_timing["update_esdf_time"],
                map_timing["query_layer_time"],
                realtime_stats["valid_depth_pixels"],
                realtime_stats["filtered_depth_pixels"],
                diagnostic_stats.get("target_visible_pixels_total", 0),
                diagnostic_stats.get("target_integrated_pixels_total", 0),
                diagnostic_stats.get("base_esdf_node0_distance", np.nan),
                diagnostic_stats.get("base_esdf_node0_margin", np.nan),
                diagnostic_stats.get("base_esdf_node0_unknown", False),
            )

        solver_fallback = bool(
            getattr(controller, "log", {}).get("solver_fallback", False)
        )
        if solver_fallback:
            u = np.zeros_like(u)
        elif ctrl_config["cmd_vel_type"] == "integration":
            u += u_bar[0] * sim.timestep
        elif ctrl_config["cmd_vel_type"] == "interpolation":
            n_nodes = v_bar.shape[0]
            t_v_bar = np.arange(n_nodes) * controller.dt
            v_interp = interp1d(
                t_v_bar,
                v_bar,
                axis=0,
                bounds_error=False,
                fill_value="extrapolate",
            )
            u = v_interp(sim.timestep)
        else:
            raise ValueError(
                f"Unknown cmd_vel_type: {ctrl_config['cmd_vel_type']}"
            )

        u_raw = np.asarray(u, dtype=float).copy()
        cmd_vel_clipped = False
        if cmd_vel_lower is not None:
            u = np.clip(u_raw, cmd_vel_lower, cmd_vel_upper)
            cmd_vel_clipped = not np.allclose(u, u_raw)
            if cmd_vel_clipped:
                cmd_vel_clip_count += 1
                if cmd_vel_clip_count <= 5:
                    logging.getLogger("Controller").warning(
                        "Command velocity clipped at t=%.3f: "
                        "raw=%s clipped=%s",
                        t,
                        u_raw,
                        u,
                    )
                elif cmd_vel_clip_count == 6:
                    logging.getLogger("Controller").warning(
                        "Further command velocity clipping messages suppressed"
                    )

        robot.command_velocity(u)
        t, _ = sim.step(t)

        ee_curr_pos, ee_cur_orn = robot.link_pose()
        ee_euler = Rot.from_quat(ee_cur_orn).as_euler("xyz")
        ee_pose = np.hstack([ee_curr_pos, ee_euler])

        ee_lin_vel, ee_ang_vel = robot.link_velocity()
        ee_vel = np.hstack([ee_lin_vel, ee_ang_vel])
        base_pose = robot_states[0][:3]
        base_vel = robot_states[1][:3]

        states = {
            "base": {"pose": base_pose, "velocity": base_vel},
            "EE": {"pose": ee_pose, "velocity": ee_vel},
        }
        sot.update(t, states)

        ee_linear_vel_world, ee_angular_vel_world = robot.link_velocity()
        ee_ref_pos = None
        base_ref_pose = None
        ee_ref_vel = None
        base_ref_vel = None

        if references.get("ee_pose") is not None:
            ee_ref_pos = references["ee_pose"][0][:3]
            if references.get("ee_velocity") is not None:
                ee_ref_vel = references["ee_velocity"][0][:3]

        if references.get("base_pose") is not None:
            base_ref_pose = references["base_pose"][0]
            if references.get("base_velocity") is not None:
                base_ref_vel = references["base_velocity"][0]

        logger.append("ts", t)
        logger.append("xs", np.hstack(robot_states))
        logger.append("controller_run_time", t1 - t0)
        logger.append("cmd_vels_raw", u_raw)
        logger.append("cmd_vels", u)
        logger.append("cmd_vel_clipped", cmd_vel_clipped)
        logger.append("r_ew_ws", ee_curr_pos)
        logger.append("Q_wes", ee_cur_orn)
        logger.append("v_ew_ws", ee_linear_vel_world)
        logger.append("omega_ew_ws", ee_angular_vel_world)
        logger.append("r_bw_ws", robot_states[0][:2])
        _append_prefixed_stats(
            logger, "online_nvblox_realtime", realtime_stats
        )
        _append_prefixed_stats(logger, "online_nvblox_map", map_timing)
        _append_prefixed_stats(logger, "online_nvblox_diag", diagnostic_stats)

        if base_ref_pose is not None:
            if base_ref_pose.shape[0] == 2:
                logger.append("r_bw_w_ds", base_ref_pose)
            elif base_ref_pose.shape[0] == 3:
                logger.append("r_bw_w_ds", base_ref_pose[:2])
                logger.append("yaw_bw_w_ds", base_ref_pose[2])
                logger.append("yaw_bw_ws", robot_states[0][2])
        if base_ref_vel is not None:
            if base_ref_vel.shape[0] == 2:
                logger.append("v_bw_w_ds", base_ref_vel)
            elif base_ref_vel.shape[0] == 3:
                logger.append("v_bw_w_ds", base_ref_vel[:2])
                logger.append("omega_bw_w_ds", base_ref_vel[2])
        if ee_ref_pos is not None:
            logger.append("r_ew_w_ds", ee_ref_pos)
        if ee_ref_vel is not None:
            logger.append("v_ew_w_ds", ee_ref_vel)
        if "MPC" in ctrl_config["type"]:
            for key, val in controller.log.items():
                logger.append("_".join(["mpc", key]) + "s", val)

        step_idx += 1
        time.sleep(sim.timestep)

    session_timestamp = timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    logger.save(session_timestamp=session_timestamp)
    preview.close()


if __name__ == "__main__":
    main()
