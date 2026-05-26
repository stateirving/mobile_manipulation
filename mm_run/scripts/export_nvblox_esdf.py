import argparse
import copy
import datetime
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pybullet as pyb
import torch
from nvblox_torch.mapper import Mapper, QueryType
from nvblox_torch.projective_integrator_types import ProjectiveIntegratorType
from nvblox_torch.sensor import Sensor

from mm_simulator import simulation
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


def robot_link_pose(sim, link_name):
    """Return a robot link pose as a 4x4 camera-to-world matrix."""
    if link_name not in sim.robot.links:
        available = ", ".join(sorted(sim.robot.links.keys()))
        raise ValueError(
            f"Robot camera link '{link_name}' does not exist. Available links: {available}"
        )
    link_idx = sim.robot.links[link_name][0]
    pos, quat = sim.robot.link_pose(link_idx)
    t_w_c = np.eye(4, dtype=np.float32)
    t_w_c[:3, :3] = np.asarray(pyb.getMatrixFromQuaternion(quat)).reshape(3, 3)
    t_w_c[:3, 3] = pos
    return t_w_c


def make_robot_spin_camera_poses(sim, link_names, num_views):
    """Spin the mobile base in place and return robot camera poses around 360 deg."""
    q_home, _ = sim.robot.joint_states()
    camera_poses = []
    for view_idx in range(num_views):
        yaw = q_home[2] + 2.0 * math.pi * float(view_idx) / float(num_views)
        q = q_home.copy()
        q[2] = yaw
        sim.robot.reset_joint_configuration(q)
        pyb.stepSimulation()
        for link_name in link_names:
            camera_poses.append(
                (
                    f"robot_spin_{view_idx:03d}_{link_name}",
                    robot_link_pose(sim, link_name),
                )
            )

    sim.robot.reset_joint_configuration(q_home)
    pyb.stepSimulation()
    return camera_poses


def make_grid(bounds, resolution):
    xmin, ymin, zmin, xmax, ymax, zmax = [float(v) for v in bounds]
    xs = np.arange(xmin, xmax + 0.5 * resolution, resolution, dtype=np.float32)
    ys = np.arange(ymin, ymax + 0.5 * resolution, resolution, dtype=np.float32)
    zs = np.arange(zmin, zmax + 0.5 * resolution, resolution, dtype=np.float32)
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
    xs, ys, zs, points = make_grid(bounds, resolution)
    values = np.empty((points.shape[0], 4), dtype=np.float32)

    for start in range(0, points.shape[0], chunk_size):
        stop = min(start + chunk_size, points.shape[0])
        query_np = np.column_stack(
            [
                points[start:stop],
                np.full(stop - start, query_radius, dtype=np.float32),
            ]
        )
        query = torch.as_tensor(query_np, device="cuda", dtype=torch.float32)
        out = mapper.query_layer(QueryType.ESDF_GRAD, query)
        values[start:stop] = out.detach().cpu().numpy()

    shape = (len(xs), len(ys), len(zs))
    gradients = values[:, :3].reshape(shape + (3,))
    distances = values[:, 3].reshape(shape)
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
            cmap="coolwarm",
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
    mask = valid & (np.abs(distances) <= band)
    if not np.any(mask):
        return False

    grid = np.meshgrid(xs, ys, zs, indexing="ij")
    points = np.column_stack([axis[mask] for axis in grid]).astype(np.float32)
    d = distances[mask].astype(np.float32)

    if points.shape[0] > max_points:
        stride = int(math.ceil(points.shape[0] / max_points))
        points = points[::stride]
        d = d[::stride]

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
    parser.add_argument(
        "--include-robot",
        action="store_true",
        help="Do not mask the robot from rendered depth images.",
    )

    # Virtual depth camera used to scan the PyBullet scene. The default orbit
    # mode is useful for offline map export; robot-spin uses URDF camera links
    # while rotating the mobile base in place.
    parser.add_argument(
        "--scan-mode",
        choices=["orbit", "robot-spin"],
        default="orbit",
        help="Camera placement mode: orbit scans around --bounds, robot-spin rotates the robot base in place.",
    )
    parser.add_argument(
        "--robot-camera-links",
        nargs="+",
        default=["camera_base_color_optical_frame"],
        help="Robot link names used as camera optical frames when --scan-mode robot-spin.",
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
        help="Number of camera poses in the orbit scan.",
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
    sim_config = copy.deepcopy(config["simulation"])
    sim_config["gui"] = bool(args.gui)

    timestamp = datetime.datetime.now()
    sim = simulation.BulletSimulation(config=sim_config, timestamp=timestamp, cli_args=None)

    root = Path(args.output)
    out_dir = root if args.no_timestamp else root / timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = out_dir / "frames"

    renderer = (
        pyb.ER_TINY_RENDERER
        if args.renderer == "tiny"
        else pyb.ER_BULLET_HARDWARE_OPENGL
    )
    exclude_body_ids = [] if args.include_robot else [sim.robot.uid]

    fy = args.height / (2.0 * math.tan(math.radians(args.fov_y_deg) / 2.0))
    fx = fy
    cx = (args.width - 1.0) / 2.0
    cy = (args.height - 1.0) / 2.0
    sensor = Sensor.from_camera(fx, fy, cx, cy, args.width, args.height)
    mapper = Mapper(args.voxel_size, ProjectiveIntegratorType.TSDF)

    camera_records = []
    if args.scan_mode == "orbit":
        camera_poses = [
            (f"orbit_{idx:03d}", look_at_pose(eye, target))
            for idx, (eye, target) in enumerate(
                make_orbit_views(args.bounds, args.num_views, args.camera_height_offset)
            )
        ]
    else:
        camera_poses = make_robot_spin_camera_poses(
            sim, args.robot_camera_links, args.num_views
        )

    for idx, (camera_name, t_w_c) in enumerate(camera_poses):
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
        if args.save_frames:
            save_debug_frame(frame_dir, idx, rgb, depth, args.far)

        depth_cuda = torch.as_tensor(depth, device="cuda", dtype=torch.float32)
        rgb_cuda = torch.as_tensor(rgb, device="cuda", dtype=torch.uint8)
        mask_cuda = torch.as_tensor(mask, device="cuda", dtype=torch.uint8)
        t_w_c_cpu = torch.as_tensor(t_w_c, dtype=torch.float32)
        mapper.add_depth_frame(depth_cuda, t_w_c_cpu, sensor, mask_cuda)
        mapper.add_color_frame(rgb_cuda, t_w_c_cpu, sensor, mask_cuda)

        camera_records.append(
            {
                "name": camera_name,
                "eye": t_w_c[:3, 3].tolist(),
                "forward": t_w_c[:3, 2].tolist(),
                "t_w_c": t_w_c.tolist(),
            }
        )

    mapper.update_esdf()
    mapper.update_color_mesh()

    mapper.save_map(str(out_dir / "map.nvblox"), 0)
    try:
        mapper.get_color_mesh(0).save(str(out_dir / "tsdf_mesh.ply"))
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
        "robot_camera_links": args.robot_camera_links,
        "camera_intrinsics": {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "width": args.width,
            "height": args.height,
        },
        "include_robot": args.include_robot,
        "surface_band_ply_saved": surface_saved,
        "cameras": camera_records,
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if args.keep_open:
        if not args.gui:
            print("--keep-open was set, but --gui was not. Nothing to keep open.")
        else:
            print("PyBullet GUI is open. Press Ctrl-C in this terminal to close it.")
            try:
                while True:
                    pyb.stepSimulation()
                    time.sleep(sim.timestep)
            except KeyboardInterrupt:
                pass

    pyb.disconnect()
    print(f"Saved nvblox map and ESDF export to: {out_dir}")


if __name__ == "__main__":
    main()
