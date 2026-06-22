import argparse
import datetime
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pybullet as pyb
import pybullet_data
import torch
from nvblox_torch.mapper import Mapper, QueryType
from nvblox_torch.projective_integrator_types import ProjectiveIntegratorType
from nvblox_torch.sensor import Sensor

from mm_utils import parsing


def _normalize(v):
    norm = np.linalg.norm(v)
    if norm < 1e-9:
        raise ValueError("Cannot normalize a near-zero vector")
    return v / norm


def look_at_pose(eye, target, up_hint=np.array([0.0, 0.0, 1.0])):
    """Return camera-to-world pose using the pinhole x-right, y-down, z-forward frame."""
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    up_hint = np.asarray(up_hint, dtype=float)

    forward = _normalize(target - eye)
    right = np.cross(forward, up_hint)
    if np.linalg.norm(right) < 1e-6:
        right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
    right = _normalize(right)
    down = _normalize(np.cross(forward, right))

    t_w_c = np.eye(4, dtype=np.float32)
    t_w_c[:3, :3] = np.column_stack([right, down, forward])
    t_w_c[:3, 3] = eye
    return t_w_c


def pybullet_up_vector(t_w_c):
    """Convert the camera y-down pose convention to PyBullet's image-up vector."""
    return -t_w_c[:3, 1]


def render_camera_pose(
    width,
    height,
    fov_y_deg,
    near,
    far,
    t_w_c,
    renderer,
    exclude_body_ids,
    return_segmentation=False,
):
    """Render from a camera-to-world pose using x-right, y-down, z-forward axes."""
    eye = t_w_c[:3, 3]
    target = eye + t_w_c[:3, 2]
    up = pybullet_up_vector(t_w_c)

    view = pyb.computeViewMatrix(
        cameraEyePosition=list(eye),
        cameraTargetPosition=list(target),
        cameraUpVector=list(up),
    )
    proj = pyb.computeProjectionMatrixFOV(
        fov=fov_y_deg,
        aspect=float(width) / float(height),
        nearVal=near,
        farVal=far,
    )

    result = pyb.getCameraImage(
        width,
        height,
        viewMatrix=view,
        projectionMatrix=proj,
        renderer=renderer,
        flags=pyb.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
    )

    rgba = np.asarray(result[2], dtype=np.uint8).reshape(height, width, 4)
    rgb = np.ascontiguousarray(rgba[:, :, :3])
    z_buffer = np.asarray(result[3], dtype=np.float32).reshape(height, width)
    depth = far * near / (far - (far - near) * z_buffer)

    valid = np.isfinite(depth) & (z_buffer < 0.999999) & (depth > near) & (depth < far)
    if exclude_body_ids:
        seg = np.asarray(result[4], dtype=np.int64).reshape(height, width)
        body_uid = seg & ((1 << 24) - 1)
        for excluded_uid in exclude_body_ids:
            valid &= body_uid != int(excluded_uid)

    depth = np.where(valid, depth, 0.0).astype(np.float32)
    mask = valid.astype(np.uint8)
    if return_segmentation:
        segmentation = np.asarray(result[4], dtype=np.int64).reshape(height, width)
        return rgb, depth, mask, segmentation
    return rgb, depth, mask


def render_camera(
    width,
    height,
    fov_y_deg,
    near,
    far,
    eye,
    target,
    renderer,
    exclude_body_ids,
):
    t_w_c = look_at_pose(eye, target)
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
    return rgb, depth, mask, t_w_c


def make_orbit_views(bounds, num_views, height_offset):
    bounds = np.asarray(bounds, dtype=float)
    xyz_min = bounds[:3]
    xyz_max = bounds[3:]
    target = 0.5 * (xyz_min + xyz_max)
    target[2] = min(max(target[2], 0.6), xyz_max[2])

    span_xy = np.maximum(xyz_max[:2] - xyz_min[:2], 1.0)
    radius = 0.75 * float(np.linalg.norm(span_xy)) + 1.0
    camera_z = xyz_max[2] + height_offset

    views = []
    for i in range(num_views):
        theta = 2.0 * math.pi * float(i) / float(num_views)
        eye = np.array(
            [
                target[0] + radius * math.cos(theta),
                target[1] + radius * math.sin(theta),
                camera_z,
            ]
        )
        views.append((eye, target.copy()))
    return views


def make_base_spin_camera_poses(
    base_xy_yaw,
    num_views,
    camera_height,
    yaw_offset,
    name_prefix="base_spin",
):
    """Spin a virtual horizontal camera around a fixed world-frame point."""
    eye = np.array([base_xy_yaw[0], base_xy_yaw[1], camera_height], dtype=float)

    camera_poses = []
    for view_idx in range(num_views):
        yaw = (
            base_xy_yaw[2]
            + yaw_offset
            + 2.0 * math.pi * float(view_idx) / float(num_views)
        )
        forward = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=float)
        target = eye + forward
        camera_poses.append(
            (
                f"{name_prefix}_{view_idx:03d}",
                look_at_pose(eye, target),
            )
        )
    return camera_poses


def resolve_base_spin_origins(args):
    """Return one or more world-frame base-spin origins as rows [x, y, yaw]."""
    if args.base_spin_origins is None:
        return np.asarray([args.base_spin_origin], dtype=float)

    origins = np.asarray(args.base_spin_origins, dtype=float)
    if origins.size % 3 != 0:
        raise ValueError(
            "--base-spin-origins expects a multiple of 3 values: "
            "X Y YAW [X Y YAW ...]"
        )
    return origins.reshape((-1, 3))


def load_environment_scene(config, gui):
    """Load only the ground plane and static obstacles into PyBullet."""
    if gui:
        pyb.connect(pyb.GUI, options="--width=1280 --height=720")
    else:
        pyb.connect(pyb.DIRECT)

    pyb.setGravity(*config.get("gravity", [0.0, 0.0, 0.0]))
    timestep = float(config.get("timestep", 0.03))
    pyb.setTimeStep(timestep)

    pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
    pyb.loadURDF("plane.urdf", [0, 0, 0])

    static_obstacles = config.get("static_obstacles", {})
    if static_obstacles.get("enabled", False):
        urdf_path = parsing.parse_and_compile_urdf(static_obstacles["urdf"])
        obstacles_uid = pyb.loadURDF(parsing.parse_path(urdf_path))
        pyb.changeDynamics(obstacles_uid, -1, mass=0)

    return timestep


def make_grid_axes(bounds, resolution):
    xmin, ymin, zmin, xmax, ymax, zmax = [float(v) for v in bounds]
    xs = np.arange(xmin, xmax + 0.5 * resolution, resolution, dtype=np.float32)
    ys = np.arange(ymin, ymax + 0.5 * resolution, resolution, dtype=np.float32)
    zs = np.arange(zmin, zmax + 0.5 * resolution, resolution, dtype=np.float32)
    return xs, ys, zs


def make_grid(bounds, resolution):
    xs, ys, zs = make_grid_axes(bounds, resolution)
    grid = np.meshgrid(xs, ys, zs, indexing="ij")
    points = np.column_stack([axis.reshape(-1) for axis in grid]).astype(np.float32)
    return xs, ys, zs, points


def query_esdf_grid(
    mapper,
    bounds,
    resolution,
    chunk_size,
    unknown_distance_threshold,
    query_radius,
):
    xs, ys, zs = make_grid_axes(bounds, resolution)
    shape = (len(xs), len(ys), len(zs))
    total_points = int(np.prod(shape))
    distances = np.empty(shape, dtype=np.float32)
    gradients = np.empty(shape + (3,), dtype=np.float32)
    distances_flat = distances.reshape(-1)
    gradients_flat = gradients.reshape((-1, 3))

    num_chunks = int(math.ceil(total_points / chunk_size))
    print(
        f"Querying ESDF grid: {total_points} points "
        f"({len(xs)} x {len(ys)} x {len(zs)}, {num_chunks} chunks)",
        flush=True,
    )

    ny, nz = shape[1], shape[2]
    yz_size = ny * nz
    for chunk_idx, start in enumerate(range(0, total_points, chunk_size), start=1):
        stop = min(start + chunk_size, total_points)
        flat_indices = np.arange(start, stop, dtype=np.int64)
        ix = flat_indices // yz_size
        iy = (flat_indices // nz) % ny
        iz = flat_indices % nz

        query_np = np.empty((stop - start, 4), dtype=np.float32)
        query_np[:, 0] = xs[ix]
        query_np[:, 1] = ys[iy]
        query_np[:, 2] = zs[iz]
        query_np[:, 3] = query_radius

        query = torch.as_tensor(query_np, device="cuda", dtype=torch.float32)
        out = mapper.query_layer(QueryType.ESDF_GRAD, query)
        out_np = out.detach().cpu().numpy()
        gradients_flat[start:stop] = out_np[:, :3]
        distances_flat[start:stop] = out_np[:, 3]
        if chunk_idx == 1 or chunk_idx == num_chunks or chunk_idx % 10 == 0:
            print(f"  ESDF query chunk {chunk_idx}/{num_chunks}", flush=True)

    valid = np.isfinite(distances) & (np.abs(distances) < unknown_distance_threshold)
    return xs, ys, zs, distances, gradients, valid


def save_slice_images(out_dir, xs, ys, zs, distances, valid, slice_zs, max_abs_distance):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    for z in slice_zs:
        k = int(np.argmin(np.abs(zs - float(z))))
        data = np.ma.masked_where(~valid[:, :, k], distances[:, :, k])

        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
        im = ax.imshow(
            data.T,
            origin="lower",
            extent=[float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1])],
            cmap="coolwarm_r",
            vmin=-max_abs_distance,
            vmax=max_abs_distance,
            interpolation="nearest",
        )
        ax.set_title(f"ESDF z={float(zs[k]):.2f} m")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, label="signed distance [m]")
        fig.savefig(out_dir / f"esdf_slice_z_{float(zs[k]):.2f}.png", dpi=160)
        plt.close(fig)


def write_ply(path, points, colors):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            f.write(
                f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                f"{int(c[0])} {int(c[1])} {int(c[2])}\n"
            )


def save_surface_band_ply(
    path,
    xs,
    ys,
    zs,
    distances,
    valid,
    band,
    max_points,
):
    if max_points <= 0:
        return False

    distances_flat = distances.reshape(-1)
    valid_flat = valid.reshape(-1)
    total_points = distances_flat.shape[0]
    chunk_size = 4_000_000

    band_count = 0
    for start in range(0, total_points, chunk_size):
        stop = min(start + chunk_size, total_points)
        mask = valid_flat[start:stop] & (np.abs(distances_flat[start:stop]) <= band)
        band_count += int(np.count_nonzero(mask))

    if band_count == 0:
        return False

    stride = max(1, int(math.ceil(band_count / max_points)))
    ny, nz = len(ys), len(zs)
    yz_size = ny * nz
    points_chunks = []
    distance_chunks = []
    seen_band_points = 0

    for start in range(0, total_points, chunk_size):
        stop = min(start + chunk_size, total_points)
        mask = valid_flat[start:stop] & (np.abs(distances_flat[start:stop]) <= band)
        local_indices = np.flatnonzero(mask)
        local_count = local_indices.shape[0]
        if local_count == 0:
            continue

        keep = (seen_band_points + np.arange(local_count, dtype=np.int64)) % stride == 0
        selected = local_indices[keep] + start
        seen_band_points += local_count
        if selected.shape[0] == 0:
            continue

        ix = selected // yz_size
        iy = (selected // nz) % ny
        iz = selected % nz
        points_chunks.append(
            np.column_stack([xs[ix], ys[iy], zs[iz]]).astype(np.float32)
        )
        distance_chunks.append(distances_flat[selected].astype(np.float32))

    if not points_chunks:
        return False

    points = np.concatenate(points_chunks, axis=0)
    d = np.concatenate(distance_chunks, axis=0)

    t = np.clip((d + band) / (2.0 * band), 0.0, 1.0)
    colors = np.column_stack(
        [
            255.0 * (1.0 - t),
            255.0 * (1.0 - np.abs(2.0 * t - 1.0)),
            255.0 * t,
        ]
    ).astype(np.uint8)
    write_ply(path, points, colors)
    return True


def save_base_navigation_esdf(
    out_dir,
    xs,
    ys,
    zs,
    distances,
    valid,
    z_min,
    z_max,
    inflation_radius,
    max_abs_distance,
):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    z_mask = (zs >= z_min) & (zs <= z_max)
    if not np.any(z_mask):
        print(
            f"Warning: base navigation z band [{z_min}, {z_max}] does not overlap ESDF grid.",
            flush=True,
        )
        return False

    band_distances = distances[:, :, z_mask]
    band_valid = valid[:, :, z_mask]
    projected = np.min(np.where(band_valid, band_distances, np.inf), axis=2)
    projected_valid = np.isfinite(projected)
    inflated = projected - float(inflation_radius)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "base_navigation_esdf.npz",
        xs=xs,
        ys=ys,
        z_min=np.asarray(z_min, dtype=np.float32),
        z_max=np.asarray(z_max, dtype=np.float32),
        inflation_radius=np.asarray(inflation_radius, dtype=np.float32),
        distance=projected.astype(np.float32),
        inflated_distance=inflated.astype(np.float32),
        valid=projected_valid,
    )

    for name, data, title in [
        (
            "base_navigation_esdf.png",
            projected,
            f"Base navigation ESDF z=[{z_min:.2f}, {z_max:.2f}] m",
        ),
        (
            "base_navigation_esdf_inflated.png",
            inflated,
            f"Inflated base ESDF radius={inflation_radius:.2f} m",
        ),
    ]:
        masked = np.ma.masked_where(~projected_valid, data)
        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
        im = ax.imshow(
            masked.T,
            origin="lower",
            extent=[float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1])],
            cmap="coolwarm_r",
            vmin=-max_abs_distance,
            vmax=max_abs_distance,
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, label="signed distance [m]")
        fig.savefig(out_dir / name, dpi=160)
        plt.close(fig)

    return True


def save_debug_frame(out_dir, idx, rgb, depth, max_depth):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.imsave(out_dir / f"rgb_{idx:03d}.png", rgb)
    depth_vis = np.ma.masked_where(depth <= 0.0, depth)
    plt.imsave(
        out_dir / f"depth_{idx:03d}.png",
        depth_vis,
        cmap="viridis",
        vmin=0.0,
        vmax=max_depth,
    )
    np.save(out_dir / f"depth_{idx:03d}.npy", depth)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build an nvblox-torch map from a PyBullet scene and export sampled ESDF data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-c", "--config", required=True, help="Experiment YAML file.")
    parser.add_argument(
        "-o",
        "--output",
        default="mm_run/results/nvblox_esdf",
        help="Output directory. A timestamped subdirectory is created inside it unless --no-timestamp is set.",
    )
    parser.add_argument("--no-timestamp", action="store_true")
    parser.add_argument("--gui", action="store_true", help="Open the PyBullet GUI.")
    parser.add_argument(
        "--keep-open",
        action="store_true",
        help="Keep the PyBullet GUI open after exporting. Use with --gui.",
    )

    # Virtual depth camera used to scan the PyBullet scene. base-spin is the
    # default for environment-only export; orbit is useful for full-scene scans.
    parser.add_argument(
        "--scan-mode",
        choices=["base-spin", "orbit"],
        default="base-spin",
        help="Camera placement mode: base-spin spins a virtual camera at one or more base origins, orbit scans around --bounds.",
    )
    parser.add_argument(
        "--base-spin-camera-height",
        type=float,
        default=0.5,
        help="Virtual camera z height in world frame when --scan-mode base-spin.",
    )
    parser.add_argument(
        "--base-spin-camera-heights",
        nargs="+",
        type=float,
        default=None,
        metavar="Z",
        help="Multiple virtual camera z heights for --scan-mode base-spin. Overrides --base-spin-camera-height.",
    )
    parser.add_argument(
        "--base-spin-origin",
        nargs=3,
        type=float,
        default=[0.0, 0.0, 0.0],
        metavar=("X", "Y", "YAW"),
        help="Single world-frame x, y, yaw used by --scan-mode base-spin. Yaw is in radians.",
    )
    parser.add_argument(
        "--base-spin-origins",
        nargs="+",
        type=float,
        default=None,
        metavar="VALUE",
        help=(
            "Flat list of multiple world-frame base-spin origins: "
            "X Y YAW [X Y YAW ...]. Yaw is in radians. Overrides --base-spin-origin."
        ),
    )
    parser.add_argument(
        "--base-spin-yaw-offset-deg",
        type=float,
        default=15.0,
        help="Yaw offset for --scan-mode base-spin, in degrees. Avoids PyBullet renderer hangs at exact cardinal directions.",
    )
    parser.add_argument("--width", type=int, default=640, help="Rendered image width in pixels.")
    parser.add_argument("--height", type=int, default=480, help="Rendered image height in pixels.")
    parser.add_argument(
        "--fov-y-deg",
        type=float,
        default=65.0,
        help="Vertical field of view of the virtual camera, in degrees.",
    )
    parser.add_argument(
        "--near",
        type=float,
        default=0.05,
        help="Near clipping plane for the depth camera, in meters.",
    )
    parser.add_argument(
        "--far",
        type=float,
        default=12.0,
        help="Far clipping plane for the depth camera, in meters.",
    )
    parser.add_argument(
        "--num-views",
        type=int,
        default=24,
        help="Number of camera poses in the orbit scan, or per base-spin origin.",
    )
    parser.add_argument(
        "--camera-height-offset",
        type=float,
        default=1.2,
        help="Camera height above the top of --bounds during the orbit scan, in meters.",
    )

    # nvblox map resolution and the dense ESDF export grid.
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.05,
        help="nvblox voxel size used for TSDF/ESDF integration, in meters.",
    )
    parser.add_argument(
        "--grid-resolution",
        type=float,
        default=0.05,
        help="Resolution of the exported ESDF sampling grid, in meters.",
    )
    parser.add_argument(
        "--bounds",
        nargs=6,
        type=float,
        default=[-1.0, -2.5, 0.0, 7.0, 2.5, 2.0],
        metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"),
        help="World-frame box sampled for ESDF export, in meters.",
    )

    # Visualization/export filters for derived files.
    parser.add_argument(
        "--slice-z",
        nargs="*",
        type=float,
        default=[0.05, 0.5, 1.0],
        help="Z heights where 2D ESDF slice images are exported.",
    )
    parser.add_argument(
        "--slice-max-abs-distance",
        type=float,
        default=1.0,
        help="Color scale limit for ESDF slice images, in meters.",
    )
    parser.add_argument(
        "--base-nav-esdf",
        action="store_true",
        help="Export a 2D base-navigation ESDF by vertically projecting the 3D ESDF over --base-nav-z-min/max.",
    )
    parser.add_argument(
        "--base-nav-z-min",
        type=float,
        default=0.05,
        help="Minimum z height included in the base-navigation ESDF projection.",
    )
    parser.add_argument(
        "--base-nav-z-max",
        type=float,
        default=1.2,
        help="Maximum z height included in the base-navigation ESDF projection.",
    )
    parser.add_argument(
        "--base-nav-inflation-radius",
        type=float,
        default=0.35,
        help="Radius subtracted from the projected ESDF to form an inflated base-navigation distance field.",
    )
    parser.add_argument(
        "--base-nav-max-abs-distance",
        type=float,
        default=0.5,
        help="Color scale limit for base-navigation ESDF images, in meters.",
    )
    parser.add_argument(
        "--surface-band",
        type=float,
        default=0.08,
        help="Export ESDF point-cloud voxels whose absolute distance is below this value.",
    )
    parser.add_argument(
        "--surface-max-points",
        type=int,
        default=200000,
        help="Maximum number of points written to esdf_surface_band.ply.",
    )
    parser.add_argument(
        "--skip-color-mesh",
        action="store_true",
        help="Skip nvblox color mesh update/export. This saves GPU memory when only ESDF output is needed.",
    )
    parser.add_argument(
        "--query-radius",
        type=float,
        default=0.0,
        help="Sphere radius subtracted by nvblox ESDF queries. Keep 0.0 for raw ESDF export.",
    )
    parser.add_argument(
        "--query-chunk-size",
        type=int,
        default=131072,
        help="Number of ESDF grid points queried from nvblox per CUDA batch.",
    )
    parser.add_argument(
        "--unknown-distance-threshold",
        type=float,
        default=1.0e5,
        help="Distances with absolute value above this threshold are marked invalid.",
    )
    parser.add_argument(
        "--renderer",
        choices=["tiny", "hardware"],
        default="tiny",
        help="PyBullet renderer. tiny is deterministic and works in DIRECT mode.",
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Save rendered RGB/depth frames used as nvblox input.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("nvblox-torch requires a CUDA device for this export script.")

    config = parsing.load_config(args.config)

    timestamp = datetime.datetime.now()
    timestep = load_environment_scene(config["simulation"], bool(args.gui))

    root = Path(args.output)
    out_dir = root if args.no_timestamp else root / timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = out_dir / "frames"

    renderer = (
        pyb.ER_TINY_RENDERER
        if args.renderer == "tiny"
        else pyb.ER_BULLET_HARDWARE_OPENGL
    )
    exclude_body_ids = []

    fy = args.height / (2.0 * math.tan(math.radians(args.fov_y_deg) / 2.0))
    fx = fy
    cx = (args.width - 1.0) / 2.0
    cy = (args.height - 1.0) / 2.0
    sensor = Sensor.from_camera(fx, fy, cx, cy, args.width, args.height)
    mapper = Mapper(args.voxel_size, ProjectiveIntegratorType.TSDF)

    camera_records = []
    base_spin_origins = None
    base_spin_camera_heights = None
    if args.scan_mode == "orbit":
        camera_poses = [
            (f"orbit_{idx:03d}", look_at_pose(eye, target))
            for idx, (eye, target) in enumerate(
                make_orbit_views(args.bounds, args.num_views, args.camera_height_offset)
            )
        ]
    else:
        base_spin_origins = resolve_base_spin_origins(args)
        base_spin_camera_heights = (
            args.base_spin_camera_heights
            if args.base_spin_camera_heights is not None
            else [args.base_spin_camera_height]
        )
        camera_poses = []
        for origin_idx, base_spin_origin in enumerate(base_spin_origins):
            for height_idx, camera_height in enumerate(base_spin_camera_heights):
                name_prefix = (
                    "base_spin"
                    if len(base_spin_origins) == 1
                    else f"base_spin_p{origin_idx:03d}"
                )
                if len(base_spin_camera_heights) > 1:
                    name_prefix = f"{name_prefix}_h{height_idx:02d}"
                camera_poses.extend(
                    make_base_spin_camera_poses(
                        base_spin_origin,
                        args.num_views,
                        camera_height,
                        math.radians(args.base_spin_yaw_offset_deg),
                        name_prefix,
                    )
                )

    print(f"Rendering and integrating {len(camera_poses)} camera views...", flush=True)
    for idx, (camera_name, t_w_c) in enumerate(camera_poses):
        view_start = time.time()
        print(
            f"  View {idx + 1}/{len(camera_poses)} {camera_name}: rendering...",
            flush=True,
        )
        rgb, depth, mask = render_camera_pose(
            args.width,
            args.height,
            args.fov_y_deg,
            args.near,
            args.far,
            t_w_c,
            renderer,
            exclude_body_ids,
        )
        render_elapsed = time.time() - view_start
        valid_depth_pixels = int(np.count_nonzero(depth > 0.0))
        print(
            f"  View {idx + 1}/{len(camera_poses)} {camera_name}: "
            f"rendered in {render_elapsed:.2f}s, valid depth pixels={valid_depth_pixels}",
            flush=True,
        )
        if args.save_frames:
            save_debug_frame(frame_dir, idx, rgb, depth, args.far)

        integrate_start = time.time()
        depth_cuda = torch.as_tensor(depth, device="cuda", dtype=torch.float32)
        rgb_cuda = torch.as_tensor(rgb, device="cuda", dtype=torch.uint8)
        mask_cuda = torch.as_tensor(mask, device="cuda", dtype=torch.uint8)
        t_w_c_cpu = torch.as_tensor(t_w_c, dtype=torch.float32)
        mapper.add_depth_frame(depth_cuda, t_w_c_cpu, sensor, mask_cuda)
        mapper.add_color_frame(rgb_cuda, t_w_c_cpu, sensor, mask_cuda)
        integrate_elapsed = time.time() - integrate_start

        camera_records.append(
            {
                "name": camera_name,
                "eye": t_w_c[:3, 3].tolist(),
                "forward": t_w_c[:3, 2].tolist(),
                "t_w_c": t_w_c.tolist(),
            }
        )
        print(
            f"  View {idx + 1}/{len(camera_poses)} {camera_name}: "
            f"integrated in {integrate_elapsed:.2f}s",
            flush=True,
        )

    print("Updating nvblox ESDF...", flush=True)
    mapper.update_esdf()

    print("Saving nvblox map...", flush=True)
    mapper.save_map(str(out_dir / "map.nvblox"), 0)
    color_mesh_saved = False
    if args.skip_color_mesh:
        print("Skipping nvblox color mesh export.", flush=True)
    else:
        print("Updating nvblox color mesh...", flush=True)
        mapper.update_color_mesh()
        try:
            mapper.get_color_mesh(0).save(str(out_dir / "tsdf_mesh.ply"))
            color_mesh_saved = True
        except Exception as exc:
            print(f"Warning: failed to save color mesh: {exc}")

    xs, ys, zs, distances, gradients, valid = query_esdf_grid(
        mapper,
        args.bounds,
        args.grid_resolution,
        args.query_chunk_size,
        args.unknown_distance_threshold,
        args.query_radius,
    )
    print("Saving ESDF grid and visualizations...", flush=True)
    np.savez_compressed(
        out_dir / "esdf_grid.npz",
        bounds=np.asarray(args.bounds, dtype=np.float32),
        resolution=np.asarray(args.grid_resolution, dtype=np.float32),
        xs=xs,
        ys=ys,
        zs=zs,
        distance=distances.astype(np.float32),
        gradient=gradients.astype(np.float32),
        valid=valid,
    )

    save_slice_images(
        out_dir / "slices",
        xs,
        ys,
        zs,
        distances,
        valid,
        args.slice_z,
        args.slice_max_abs_distance,
    )
    base_nav_esdf_saved = False
    if args.base_nav_esdf:
        base_nav_esdf_saved = save_base_navigation_esdf(
            out_dir / "base_navigation",
            xs,
            ys,
            zs,
            distances,
            valid,
            args.base_nav_z_min,
            args.base_nav_z_max,
            args.base_nav_inflation_radius,
            args.base_nav_max_abs_distance,
        )
    surface_saved = save_surface_band_ply(
        out_dir / "esdf_surface_band.ply",
        xs,
        ys,
        zs,
        distances,
        valid,
        args.surface_band,
        args.surface_max_points,
    )

    metadata = {
        "config": args.config,
        "bounds": args.bounds,
        "voxel_size": args.voxel_size,
        "grid_resolution": args.grid_resolution,
        "query_radius": args.query_radius,
        "image_width": args.width,
        "image_height": args.height,
        "fov_y_deg": args.fov_y_deg,
        "near": args.near,
        "far": args.far,
        "scan_mode": args.scan_mode,
        "num_views": args.num_views,
        "base_spin_camera_height": args.base_spin_camera_height,
        "base_spin_camera_heights": base_spin_camera_heights,
        "base_spin_origin": args.base_spin_origin,
        "base_spin_origins": (
            None if base_spin_origins is None else base_spin_origins.tolist()
        ),
        "base_spin_yaw_offset_deg": args.base_spin_yaw_offset_deg,
        "camera_intrinsics": {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "width": args.width,
            "height": args.height,
        },
        "surface_band_ply_saved": surface_saved,
        "base_nav_esdf_saved": base_nav_esdf_saved,
        "base_nav_z_min": args.base_nav_z_min,
        "base_nav_z_max": args.base_nav_z_max,
        "base_nav_inflation_radius": args.base_nav_inflation_radius,
        "color_mesh_saved": color_mesh_saved,
        "cameras": camera_records,
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved nvblox map and ESDF export to: {out_dir}", flush=True)

    if args.keep_open:
        if not args.gui:
            print("--keep-open was set, but --gui was not. Nothing to keep open.")
        else:
            print("PyBullet GUI is open. Press Ctrl-C in this terminal to close it.")
            try:
                while True:
                    pyb.stepSimulation()
                    time.sleep(timestep)
            except KeyboardInterrupt:
                pass

    pyb.disconnect()


if __name__ == "__main__":
    main()

# Example usage:
# 
# python mm_run/scripts/export_nvblox_esdf.py \
#   --config mm_run/config/aws_small_warehouse_esdf.yaml \
#   --output mm_run/results/nvblox_esdf/aws_small_warehouse_env \
#   --bounds -7 -10 0 7 10 3 \
#   --far 20 \
#   --base-spin-origins 0 0 0  2 0 0  4 0 0 \
#   --base-spin-camera-height 1.2 \
#   --base-spin-yaw-offset-deg 15 \
#   --num-views 12 \
#   --width 320 \
#   --height 240 \
#   --grid-resolution 0.2 \
#   --renderer hardware \
#   --save-frames
# 
# python mm_run/scripts/export_nvblox_esdf.py \
#   --config mm_run/config/aws_small_warehouse_esdf.yaml \
#   --output /tmp/nvblox_scene_view \
#   --bounds -7 -10 0 7 10 3 \
#   --scan-mode base-spin \
#   --base-spin-origin 0 0 0 \
#   --base-spin-camera-height 1.2 \
#   --num-views 1 \
#   --width 320 \
#   --height 240 \
#   --grid-resolution 0.5 \
#   --gui --keep-open
