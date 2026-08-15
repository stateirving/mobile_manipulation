# flake8: noqa: E402
"""Build and render an nvblox ESDF from a recorded real-robot ROS 2 bag.

The converter intentionally runs offline.  It reads synchronized Spectacular AI
depth keyframes, camera intrinsics, and the recorded TF tree, reconstructs
``T_map_camera`` at every depth timestamp, integrates the frames with
nvblox-torch, and writes the same ``esdf_grid.npz`` schema consumed by
``mm_control.esdf_map.ESDFMap``.
"""

import argparse
import bisect
import collections
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import rosbag2_py
import torch
from export_nvblox_esdf import query_esdf_grid, save_slice_images, save_surface_band_ply
from nvblox_torch.constants import constants
from nvblox_torch.mapper import Mapper, QueryType
from nvblox_torch.projective_integrator_types import ProjectiveIntegratorType
from nvblox_torch.sensor import Sensor
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from mm_control.esdf_map import ESDFMap

DEPTH_TOPIC = "/spectacular_ai/depth_image"
CAMERA_INFO_TOPIC = "/spectacular_ai/camera_info"
TF_TOPIC = "/tf"
TF_STATIC_TOPIC = "/tf_static"


def _stamp_ns(header):
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


def _quaternion_to_rotation(quaternion):
    q = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm <= 1.0e-12:
        raise ValueError("Transform contains a zero-length quaternion")
    x, y, z, w = q / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _make_transform(translation, quaternion):
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = _quaternion_to_rotation(quaternion)
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64)
    return matrix


def _transform_record(transform):
    translation = transform.transform.translation
    rotation = transform.transform.rotation
    return (
        _stamp_ns(transform.header),
        np.asarray([translation.x, translation.y, translation.z], dtype=np.float64),
        np.asarray([rotation.x, rotation.y, rotation.z, rotation.w], dtype=np.float64),
    )


class OfflineTFGraph:
    """Small TF2-compatible graph with linear/nlerp interpolation."""

    def __init__(self, static_records, dynamic_records, max_age_s):
        self.static = dict(static_records)
        self.dynamic = {}
        self.dynamic_stamps = {}
        self.duplicate_records_discarded = {}
        self.max_age_ns = int(float(max_age_s) * 1.0e9)

        for pair, records in dynamic_records.items():
            # SLAM can republish a corrected transform with the same header
            # timestamp.  The bag preserves all revisions in receive order, so
            # retain the last (newest) revision before interpolating.  Keeping
            # duplicate stamps would make bisect join an old estimate to a new
            # one and create large, physically impossible pose jumps.
            latest_by_stamp = {}
            for record in records:
                latest_by_stamp[record[0]] = record
            discarded = len(records) - len(latest_by_stamp)
            if discarded:
                self.duplicate_records_discarded[pair] = discarded
            ordered = sorted(latest_by_stamp.values(), key=lambda record: record[0])
            self.dynamic[pair] = ordered
            self.dynamic_stamps[pair] = [record[0] for record in ordered]

    def _dynamic_at(self, pair, query_ns):
        records = self.dynamic[pair]
        stamps = self.dynamic_stamps[pair]
        index = bisect.bisect_left(stamps, query_ns)

        if index < len(stamps) and stamps[index] == query_ns:
            stamp, translation, quaternion = records[index]
            return _make_transform(translation, quaternion), abs(stamp - query_ns)

        if 0 < index < len(records):
            before = records[index - 1]
            after = records[index]
            max_age = max(query_ns - before[0], after[0] - query_ns)
            if max_age > self.max_age_ns:
                return None
            alpha = float(query_ns - before[0]) / float(after[0] - before[0])
            translation = (1.0 - alpha) * before[1] + alpha * after[1]
            q0 = before[2]
            q1 = after[2]
            if np.dot(q0, q1) < 0.0:
                q1 = -q1
            quaternion = (1.0 - alpha) * q0 + alpha * q1
            return _make_transform(translation, quaternion), max_age

        nearest = records[0] if index == 0 else records[-1]
        age = abs(query_ns - nearest[0])
        if age > self.max_age_ns:
            return None
        return _make_transform(nearest[1], nearest[2]), age

    def lookup(self, target_frame, source_frame, query_ns):
        """Return ``T_target_source`` and the path used at ``query_ns``."""
        edges = collections.defaultdict(list)
        for (parent, child), matrix in self.static.items():
            edges[parent].append((child, matrix, 0))
            edges[child].append((parent, np.linalg.inv(matrix), 0))

        for pair in self.dynamic:
            sample = self._dynamic_at(pair, query_ns)
            if sample is None:
                continue
            matrix, age = sample
            parent, child = pair
            edges[parent].append((child, matrix, age))
            edges[child].append((parent, np.linalg.inv(matrix), age))

        queue = collections.deque([(target_frame, np.eye(4), [], 0)])
        visited = {target_frame}
        while queue:
            frame, target_to_frame, path, max_age = queue.popleft()
            if frame == source_frame:
                return target_to_frame, path, max_age
            for neighbor, frame_to_neighbor, edge_age in edges.get(frame, []):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append(
                    (
                        neighbor,
                        target_to_frame @ frame_to_neighbor,
                        path + [(frame, neighbor)],
                        max(max_age, edge_age),
                    )
                )
        raise KeyError(
            f"No recorded TF path from {target_frame!r} to {source_frame!r} "
            f"at {query_ns}"
        )


def _read_bag(bag_path):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_path), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("", ""),
    )
    topic_types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    required = {DEPTH_TOPIC, CAMERA_INFO_TOPIC, TF_TOPIC, TF_STATIC_TOPIC}
    missing = sorted(required.difference(topic_types))
    if missing:
        raise KeyError(f"Bag is missing required topics: {', '.join(missing)}")
    message_types = {topic: get_message(topic_types[topic]) for topic in required}

    depth_records = []
    camera_info_by_stamp = {}
    static_records = {}
    dynamic_records = collections.defaultdict(list)

    while reader.has_next():
        topic, raw, _ = reader.read_next()
        if topic not in message_types:
            continue
        msg = deserialize_message(raw, message_types[topic])
        if topic == DEPTH_TOPIC:
            depth_records.append((_stamp_ns(msg.header), msg))
        elif topic == CAMERA_INFO_TOPIC:
            camera_info_by_stamp[_stamp_ns(msg.header)] = msg
        else:
            for transform in msg.transforms:
                parent = str(transform.header.frame_id).lstrip("/")
                child = str(transform.child_frame_id).lstrip("/")
                record = _transform_record(transform)
                pair = (parent, child)
                if topic == TF_STATIC_TOPIC:
                    static_records[pair] = _make_transform(record[1], record[2])
                else:
                    dynamic_records[pair].append(record)

    if not depth_records:
        raise RuntimeError(f"Bag contains no messages on {DEPTH_TOPIC}")
    return depth_records, camera_info_by_stamp, static_records, dynamic_records


def _decode_depth(message, depth_scale):
    if message.encoding != "16UC1":
        raise ValueError(f"Expected 16UC1 depth, got {message.encoding!r}")
    dtype = np.dtype(">u2" if message.is_bigendian else "<u2")
    raw = np.frombuffer(bytes(message.data), dtype=dtype)
    expected = int(message.height) * int(message.width)
    if raw.size != expected or int(message.step) != 2 * int(message.width):
        raise ValueError(
            "Only tightly packed 16UC1 images are supported: "
            f"size={raw.size}, expected={expected}, step={message.step}"
        )
    return raw.reshape((message.height, message.width)).astype(np.float32) * float(
        depth_scale
    )


def _camera_from_info(message):
    return {
        "fx": float(message.k[0]),
        "fy": float(message.k[4]),
        "cx": float(message.k[2]),
        "cy": float(message.k[5]),
        "width": int(message.width),
        "height": int(message.height),
    }


def _depth_range_mask(depth, min_depth, max_depth):
    return (
        np.isfinite(depth) & (depth >= float(min_depth)) & (depth <= float(max_depth))
    ).astype(np.uint8)


def _ground_aware_mask(depth, camera, t_map_camera, min_depth, max_depth, ground_min_z):
    valid = _depth_range_mask(depth, min_depth, max_depth).astype(bool)
    if ground_min_z is None:
        return valid.astype(np.uint8)

    u = (np.arange(camera["width"], dtype=np.float32) - camera["cx"]) / camera["fx"]
    v = (np.arange(camera["height"], dtype=np.float32) - camera["cy"]) / camera["fy"]
    rotation = t_map_camera[:3, :3]
    endpoint_z = (
        rotation[2, 0] * depth * u[None, :]
        + rotation[2, 1] * depth * v[:, None]
        + rotation[2, 2] * depth
        + t_map_camera[2, 3]
    )
    valid &= endpoint_z >= float(ground_min_z)
    return valid.astype(np.uint8)


def _query_occupancy_grid(mapper, xs, ys, zs, chunk_size):
    """Query occupancy log-odds on the ESDF grid in bounded CUDA batches."""
    shape = (len(xs), len(ys), len(zs))
    total_points = int(np.prod(shape))
    occupancy = np.empty(shape, dtype=np.float32)
    occupancy_flat = occupancy.reshape(-1)
    ny, nz = shape[1], shape[2]
    yz_size = ny * nz

    num_chunks = int(np.ceil(total_points / chunk_size))
    print(f"Querying observed-space occupancy ({num_chunks} chunks)", flush=True)
    for chunk_idx, start in enumerate(range(0, total_points, chunk_size), start=1):
        stop = min(start + chunk_size, total_points)
        flat_indices = np.arange(start, stop, dtype=np.int64)
        ix = flat_indices // yz_size
        iy = (flat_indices // nz) % ny
        iz = flat_indices % nz
        points = np.column_stack((xs[ix], ys[iy], zs[iz])).astype(np.float32)
        query = torch.as_tensor(points, device="cuda", dtype=torch.float32)
        output = mapper.query_layer(QueryType.OCCUPANCY, query, mapper_id=-1)
        occupancy_flat[start:stop] = output.detach().cpu().numpy()[:, 0]
        if chunk_idx == 1 or chunk_idx == num_chunks or chunk_idx % 10 == 0:
            print(f"  occupancy query chunk {chunk_idx}/{num_chunks}", flush=True)
    return occupancy


def _fuse_observed_free_space(
    distances,
    gradients,
    valid,
    occupancy,
    resolution,
    free_log_odds_threshold,
    obstacle_site_distance,
    max_fill_distance,
):
    """Fill obstacle-ESDF gaps only where occupancy rays observed free space."""
    observed_free = np.isfinite(occupancy) & (
        occupancy < float(free_log_odds_threshold)
    )
    fill_mask = observed_free & ~valid
    obstacle_sites = (
        valid & np.isfinite(distances) & (distances <= float(obstacle_site_distance))
    )
    fill_count = int(np.count_nonzero(fill_mask))
    obstacle_count = int(np.count_nonzero(obstacle_sites))
    if fill_count and obstacle_count == 0:
        raise RuntimeError(
            "Observed free space exists, but the obstacle ESDF has no surface sites"
        )

    if fill_count:
        from scipy.ndimage import distance_transform_edt

        print(
            f"Propagating obstacle distances into {fill_count} observed-free voxels",
            flush=True,
        )
        propagated = distance_transform_edt(
            ~obstacle_sites, sampling=(float(resolution),) * 3
        ).astype(np.float32)
        np.minimum(propagated, float(max_fill_distance), out=propagated)
        distances[fill_mask] = propagated[fill_mask]
        for axis in range(3):
            component = np.gradient(propagated, float(resolution), axis=axis).astype(
                np.float32, copy=False
            )
            gradients[..., axis][fill_mask] = component[fill_mask]
        del propagated

    valid |= observed_free
    return {
        "observed_free_points": int(np.count_nonzero(observed_free)),
        "filled_free_points": fill_count,
        "obstacle_site_points": obstacle_count,
    }


def _surface_points(xs, ys, zs, distances, valid, band, max_points):
    mask = valid & np.isfinite(distances) & (np.abs(distances) <= band)
    indices = np.flatnonzero(mask.reshape(-1))
    total = len(indices)
    if total > max_points:
        indices = indices[np.linspace(0, total - 1, max_points, dtype=np.int64)]
    ny, nz = len(ys), len(zs)
    yz = ny * nz
    ix = indices // yz
    iy = (indices // nz) % ny
    iz = indices % nz
    return np.column_stack((xs[ix], ys[iy], zs[iz])), total


def _equalize_3d_axes(ax, bounds):
    lower = np.asarray(bounds[:3], dtype=float)
    upper = np.asarray(bounds[3:], dtype=float)
    center = 0.5 * (lower + upper)
    radius = 0.5 * float(np.max(upper - lower))
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(lower[2], upper[2])
    ax.set_box_aspect((1.0, 1.0, max((upper[2] - lower[2]) / (2.0 * radius), 0.2)))


def _save_static_preview(
    path,
    xs,
    ys,
    zs,
    distances,
    valid,
    bounds,
    surface_band,
    camera_positions,
):
    points, total = _surface_points(
        xs, ys, zs, distances, valid, surface_band, max_points=120000
    )
    if not len(points):
        raise RuntimeError("No ESDF zero-surface samples available for rendering")

    colors = points[:, 2]
    fig = plt.figure(figsize=(15, 7), constrained_layout=True)
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    scatter = ax3d.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=colors,
        cmap="turbo",
        s=0.8,
        alpha=0.8,
    )
    if len(camera_positions):
        trajectory = np.asarray(camera_positions)
        ax3d.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            trajectory[:, 2],
            "k.-",
            linewidth=1.2,
            markersize=3,
        )
    ax3d.set_title(f"Real-bag ESDF surface (showing {len(points)}/{total})")
    ax3d.set_xlabel("map x [m]")
    ax3d.set_ylabel("map y [m]")
    ax3d.set_zlabel("map z [m]")
    ax3d.view_init(elev=27, azim=-58)
    _equalize_3d_axes(ax3d, bounds)
    fig.colorbar(scatter, ax=ax3d, shrink=0.65, label="height [m]")

    ax_top = fig.add_subplot(1, 2, 2)
    ax_top.scatter(points[:, 0], points[:, 1], c=colors, cmap="turbo", s=0.8, alpha=0.7)
    if len(camera_positions):
        trajectory = np.asarray(camera_positions)
        ax_top.plot(
            trajectory[:, 0], trajectory[:, 1], "k.-", linewidth=1.4, markersize=4
        )
        ax_top.scatter(
            trajectory[0, 0],
            trajectory[0, 1],
            c="lime",
            edgecolors="black",
            s=70,
            label="start",
        )
        ax_top.scatter(
            trajectory[-1, 0],
            trajectory[-1, 1],
            c="red",
            edgecolors="black",
            s=70,
            label="end",
        )
        ax_top.legend()
    ax_top.set_title("Top view; black line is the camera trajectory")
    ax_top.set_xlabel("map x [m]")
    ax_top.set_ylabel("map y [m]")
    ax_top.set_aspect("equal")
    ax_top.set_xlim(bounds[0], bounds[3])
    ax_top.set_ylim(bounds[1], bounds[4])
    ax_top.grid(alpha=0.2)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _planner_quality_report(
    map_path,
    bounds_xy,
    start_xy,
    query_z,
    base_radius,
    d_safe,
    lattice_resolution,
):
    """Evaluate the exported map with the offline base planner predicate."""
    from scipy.ndimage import label

    esdf_map = ESDFMap(map_path)
    query_z = np.asarray(query_z, dtype=np.float64)
    required_distance = float(base_radius) + float(d_safe)
    xs = np.arange(
        float(bounds_xy[0]),
        float(bounds_xy[1]) + 0.5 * float(lattice_resolution),
        float(lattice_resolution),
    )
    ys = np.arange(
        float(bounds_xy[2]),
        float(bounds_xy[3]) + 0.5 * float(lattice_resolution),
        float(lattice_resolution),
    )
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    xy = np.column_stack((xx.reshape(-1), yy.reshape(-1)))
    points = np.vstack([np.column_stack((xy, np.full(len(xy), z))) for z in query_z])
    distances, _, valid = esdf_map.query(points)
    distances = np.asarray(distances).reshape((len(query_z), -1))
    valid = np.asarray(valid).reshape((len(query_z), -1))
    known = np.all(valid, axis=0)
    planner_valid = known & np.all(distances >= required_distance, axis=0)
    components, component_count = label(
        planner_valid.reshape((len(xs), len(ys))),
        structure=np.ones((3, 3), dtype=np.uint8),
    )

    start_points = np.column_stack(
        (
            np.full(len(query_z), float(start_xy[0])),
            np.full(len(query_z), float(start_xy[1])),
            query_z,
        )
    )
    start_distances, _, start_valid = esdf_map.query(start_points)
    start_planner_valid = bool(
        np.all(start_valid) & np.all(start_distances >= required_distance)
    )
    start_component = 0
    start_lattice_distance = None
    if start_planner_valid:
        valid_indices = np.argwhere(components > 0)
        if len(valid_indices):
            lattice_xy = np.column_stack(
                (xs[valid_indices[:, 0]], ys[valid_indices[:, 1]])
            )
            delta = lattice_xy - np.asarray(start_xy, dtype=np.float64)
            nearest = int(np.argmin(np.sum(delta * delta, axis=1)))
            ix, iy = valid_indices[nearest]
            start_component = int(components[ix, iy])
            start_lattice_distance = float(np.linalg.norm(delta[nearest]))

    return {
        "bounds_xy_m": [float(value) for value in bounds_xy],
        "query_z_m": query_z.tolist(),
        "base_radius_m": float(base_radius),
        "d_safe_m": float(d_safe),
        "required_distance_m": required_distance,
        "lattice_resolution_m": float(lattice_resolution),
        "known_ratio": float(np.mean(known)),
        "planner_valid_ratio": float(np.mean(planner_valid)),
        "component_count": int(component_count),
        "start": {
            "xy_m": [float(start_xy[0]), float(start_xy[1])],
            "distances_m": np.asarray(start_distances).tolist(),
            "known": bool(np.all(start_valid)),
            "planner_valid": start_planner_valid,
            "component": start_component,
            "nearest_planner_lattice_distance_m": start_lattice_distance,
        },
    }


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a real Stretch Spectacular-AI rosbag into an nvblox ESDF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("bag", help="Path to a rosbag2 directory")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--world-frame", default="map")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--camera-frame", default="camera_color_optical_frame")
    parser.add_argument("--depth-scale", type=float, default=0.001)
    parser.add_argument("--min-depth", type=float, default=0.25)
    parser.add_argument("--max-depth", type=float, default=4.0)
    parser.add_argument(
        "--ground-min-z",
        type=float,
        default=0.08,
        help=(
            "Discard rays whose measured endpoint is below this map-z height; "
            "use a negative value to retain the ground."
        ),
    )
    parser.add_argument(
        "--disable-observed-space",
        action="store_true",
        help="Disable the secondary unfiltered occupancy map (not recommended).",
    )
    parser.add_argument(
        "--observed-voxel-size",
        type=float,
        default=None,
        help="Occupancy voxel size; defaults to --voxel-size.",
    )
    parser.add_argument("--free-log-odds-threshold", type=float, default=0.0)
    parser.add_argument("--obstacle-site-distance", type=float, default=0.0)
    parser.add_argument("--max-fill-distance", type=float, default=2.0)
    parser.add_argument("--voxel-size", type=float, default=0.05)
    parser.add_argument("--grid-resolution", type=float, default=0.05)
    parser.add_argument(
        "--bounds",
        nargs=6,
        type=float,
        default=[-4.2, -4.2, -0.2, 4.2, 4.2, 2.2],
        metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"),
    )
    parser.add_argument("--max-tf-age", type=float, default=0.25)
    parser.add_argument("--query-chunk-size", type=int, default=131072)
    parser.add_argument(
        "--unknown-distance-threshold",
        type=float,
        default=float(constants.esdf_unknown_distance()),
        help="Absolute nvblox distance at or above which a sample is unknown.",
    )
    parser.add_argument("--surface-band", type=float, default=0.08)
    parser.add_argument("--surface-max-points", type=int, default=200000)
    parser.add_argument(
        "--planner-query-z", nargs="+", type=float, default=[0.15, 0.35]
    )
    parser.add_argument("--planner-base-radius", type=float, default=0.20)
    parser.add_argument("--planner-d-safe", type=float, default=0.20)
    parser.add_argument("--planner-lattice-resolution", type=float, default=0.05)
    parser.add_argument(
        "--planner-bounds-xy",
        nargs=4,
        type=float,
        default=[-4.0, 4.0, -4.0, 4.0],
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
    )
    parser.add_argument(
        "--slice-z", nargs="*", type=float, default=[0.10, 0.50, 1.00, 1.50]
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("nvblox-torch requires a CUDA device")
    bag_path = Path(args.bag).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    print(f"Reading bag: {bag_path}", flush=True)
    depth_records, camera_infos, static_records, dynamic_records = _read_bag(bag_path)
    print(
        f"Found {len(depth_records)} depth frames, {len(camera_infos)} CameraInfo messages, "
        f"{len(static_records)} static TF edges and {len(dynamic_records)} dynamic TF edges.",
        flush=True,
    )
    tf_graph = OfflineTFGraph(
        static_records, dynamic_records, max_age_s=args.max_tf_age
    )
    duplicate_tf_count = sum(tf_graph.duplicate_records_discarded.values())
    if duplicate_tf_count:
        details = ", ".join(
            f"{parent}->{child}: {count}"
            for (parent, child), count in sorted(
                tf_graph.duplicate_records_discarded.items()
            )
        )
        print(
            f"Discarded {duplicate_tf_count} superseded TF records with duplicate "
            f"header stamps ({details}).",
            flush=True,
        )

    first_info = next(
        (camera_infos[stamp] for stamp, _ in depth_records if stamp in camera_infos),
        None,
    )
    if first_info is None:
        raise KeyError("No depth frame has an exactly synchronized CameraInfo")
    camera = _camera_from_info(first_info)
    sensor = Sensor.from_camera(
        camera["fx"],
        camera["fy"],
        camera["cx"],
        camera["cy"],
        camera["width"],
        camera["height"],
    )
    mapper = Mapper(float(args.voxel_size), ProjectiveIntegratorType.TSDF)
    observed_voxel_size = (
        float(args.voxel_size)
        if args.observed_voxel_size is None
        else float(args.observed_voxel_size)
    )
    observed_mapper = (
        None
        if args.disable_observed_space
        else Mapper(observed_voxel_size, ProjectiveIntegratorType.OCCUPANCY)
    )

    integrated = 0
    total_valid_pixels = 0
    total_observed_pixels = 0
    camera_positions = []
    camera_poses = []
    base_positions = []
    base_poses = []
    tf_paths = collections.Counter()
    tf_ages_ms = []
    ground_min_z = None if args.ground_min_z < -1.0e6 else args.ground_min_z

    for index, (stamp, message) in enumerate(depth_records, start=1):
        info = camera_infos.get(stamp)
        if info is None:
            print(f"Skipping frame {index}: no exact CameraInfo match", flush=True)
            continue
        current_camera = _camera_from_info(info)
        if current_camera != camera:
            raise ValueError("Camera intrinsics changed within the bag")

        try:
            t_map_camera, path, max_age_ns = tf_graph.lookup(
                args.world_frame, args.camera_frame, stamp
            )
        except KeyError as exc:
            print(f"Skipping frame {index}: {exc}", flush=True)
            continue
        if not np.all(np.isfinite(t_map_camera)) or not np.isclose(
            np.linalg.det(t_map_camera[:3, :3]), 1.0, atol=1.0e-3
        ):
            raise ValueError(f"Invalid camera pose for frame {index}")
        try:
            t_map_base, _, _ = tf_graph.lookup(args.world_frame, args.base_frame, stamp)
        except KeyError as exc:
            raise KeyError(
                f"Cannot evaluate planner start without {args.base_frame!r}: {exc}"
            ) from exc

        depth = _decode_depth(message, args.depth_scale)
        observed_mask = _depth_range_mask(depth, args.min_depth, args.max_depth)
        if observed_mapper is not None:
            observed_depth = np.where(observed_mask != 0, depth, 0.0).astype(
                np.float32, copy=False
            )
            observed_mapper.add_depth_frame(
                torch.as_tensor(observed_depth, device="cuda", dtype=torch.float32),
                torch.as_tensor(t_map_camera, device="cpu", dtype=torch.float32),
                sensor,
                torch.as_tensor(observed_mask, device="cuda", dtype=torch.uint8),
            )
            total_observed_pixels += int(np.count_nonzero(observed_mask))

        mask = _ground_aware_mask(
            depth,
            camera,
            t_map_camera,
            args.min_depth,
            args.max_depth,
            ground_min_z,
        )
        depth = np.where(mask != 0, depth, 0.0).astype(np.float32, copy=False)
        valid_pixels = int(np.count_nonzero(mask))
        if valid_pixels == 0:
            print(f"Skipping frame {index}: no valid depth pixels", flush=True)
            continue

        mapper.add_depth_frame(
            torch.as_tensor(depth, device="cuda", dtype=torch.float32),
            torch.as_tensor(t_map_camera, device="cpu", dtype=torch.float32),
            sensor,
            torch.as_tensor(mask, device="cuda", dtype=torch.uint8),
        )
        integrated += 1
        total_valid_pixels += valid_pixels
        camera_positions.append(t_map_camera[:3, 3].copy())
        camera_poses.append(t_map_camera.tolist())
        base_positions.append(t_map_base[:3, 3].copy())
        base_poses.append(t_map_base.tolist())
        tf_paths[tuple(path)] += 1
        tf_ages_ms.append(max_age_ns / 1.0e6)
        print(
            f"Integrated frame {index}/{len(depth_records)}: valid={valid_pixels}, "
            f"camera_xyz={np.round(t_map_camera[:3, 3], 3).tolist()}, "
            f"max_tf_age={max_age_ns / 1.0e6:.1f} ms",
            flush=True,
        )

    if integrated == 0:
        raise RuntimeError("No depth frames could be integrated")

    print("Updating ESDF and saving native nvblox map...", flush=True)
    mapper.update_esdf()
    mapper.save_map(str(output / "map.nvblox"), 0)
    observed_path = None
    if observed_mapper is not None:
        observed_path = output / "observed_space.nvblox"
        observed_mapper.save_map(str(observed_path), 0)

    xs, ys, zs, distances, gradients, valid = query_esdf_grid(
        mapper,
        args.bounds,
        args.grid_resolution,
        args.query_chunk_size,
        args.unknown_distance_threshold,
        0.0,
    )
    obstacle_esdf_valid_count = int(np.count_nonzero(valid))
    if observed_mapper is not None:
        occupancy = _query_occupancy_grid(
            observed_mapper, xs, ys, zs, args.query_chunk_size
        )
        fusion_stats = _fuse_observed_free_space(
            distances,
            gradients,
            valid,
            occupancy,
            args.grid_resolution,
            args.free_log_odds_threshold,
            args.obstacle_site_distance,
            args.max_fill_distance,
        )
        del occupancy
    else:
        fusion_stats = {
            "observed_free_points": 0,
            "filled_free_points": 0,
            "obstacle_site_points": 0,
        }
    valid_count = int(np.count_nonzero(valid))
    if valid_count == 0:
        raise RuntimeError("The sampled ESDF contains no valid points")
    np.savez_compressed(
        output / "esdf_grid.npz",
        bounds=np.asarray(args.bounds, dtype=np.float32),
        resolution=np.asarray(args.grid_resolution, dtype=np.float32),
        xs=xs,
        ys=ys,
        zs=zs,
        distance=distances.astype(np.float32),
        gradient=gradients.astype(np.float32),
        valid=valid,
    )
    quality_report = _planner_quality_report(
        output / "esdf_grid.npz",
        args.planner_bounds_xy,
        base_positions[0][:2],
        args.planner_query_z,
        args.planner_base_radius,
        args.planner_d_safe,
        args.planner_lattice_resolution,
    )
    save_slice_images(
        output / "slices", xs, ys, zs, distances, valid, args.slice_z, 0.75
    )
    surface_saved = save_surface_band_ply(
        output / "esdf_surface_band.ply",
        xs,
        ys,
        zs,
        distances,
        valid,
        args.surface_band,
        args.surface_max_points,
    )
    _save_static_preview(
        output / "esdf_surface_preview.png",
        xs,
        ys,
        zs,
        distances,
        valid,
        args.bounds,
        args.surface_band,
        camera_positions,
    )

    metadata = {
        "source_bag": str(bag_path),
        "world_frame": args.world_frame,
        "camera_frame": args.camera_frame,
        "base_frame": args.base_frame,
        "depth_scale_m_per_unit": args.depth_scale,
        "min_depth_m": args.min_depth,
        "max_depth_m": args.max_depth,
        "ground_min_z_m": ground_min_z,
        "voxel_size_m": args.voxel_size,
        "observed_space": {
            "enabled": observed_mapper is not None,
            "map_path": None if observed_path is None else str(observed_path),
            "voxel_size_m": None if observed_mapper is None else observed_voxel_size,
            "free_log_odds_threshold": args.free_log_odds_threshold,
            "obstacle_site_distance_m": args.obstacle_site_distance,
            "max_fill_distance_m": args.max_fill_distance,
            "valid_depth_pixels_integrated": total_observed_pixels,
            **fusion_stats,
        },
        "grid_resolution_m": args.grid_resolution,
        "unknown_distance_threshold_m": args.unknown_distance_threshold,
        "bounds": args.bounds,
        "camera_intrinsics": camera,
        "depth_frames_in_bag": len(depth_records),
        "depth_frames_integrated": integrated,
        "valid_depth_pixels_integrated": total_valid_pixels,
        "obstacle_esdf_valid_points": obstacle_esdf_valid_count,
        "grid_valid_points": valid_count,
        "grid_total_points": int(valid.size),
        "surface_band_ply_saved": surface_saved,
        "max_tf_age_ms": {
            "max": float(np.max(tf_ages_ms)),
            "p95": float(np.percentile(tf_ages_ms, 95)),
        },
        "tf_paths": [
            {"path": list(path), "frame_count": count}
            for path, count in tf_paths.items()
        ],
        "duplicate_tf_records_discarded": {
            f"{parent}->{child}": count
            for (parent, child), count in sorted(
                tf_graph.duplicate_records_discarded.items()
            )
        },
        "camera_poses_t_map_camera": camera_poses,
        "base_poses_t_map_base": base_poses,
        "planner_quality": quality_report,
        "provisional_assumptions": [
            (
                "16UC1 depth uses depth_scale_m_per_unit; 0.001 is provisional "
                "until independently measured."
            ),
            (
                "Ground filtering removes rays by endpoint map-z and is intended "
                "for visualization/first-pass obstacle ESDF."
            ),
        ],
    }
    with (output / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)

    print(
        f"Saved real-bag ESDF to {output} ({valid_count}/{valid.size} sampled points valid).",
        flush=True,
    )
    print(
        "Planner quality: "
        f"known={100.0 * quality_report['known_ratio']:.2f}%, "
        f"valid={100.0 * quality_report['planner_valid_ratio']:.2f}%, "
        f"start_valid={quality_report['start']['planner_valid']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
