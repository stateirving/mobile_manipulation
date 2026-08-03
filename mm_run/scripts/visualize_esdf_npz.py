"""Visualize the zero-level surface of an exported ESDF NPZ in PyBullet."""

import argparse
import time
from pathlib import Path

import numpy as np
import pybullet as pyb


def _load_esdf_npz(path):
    path = Path(path).expanduser().resolve()
    with np.load(path) as data:
        required = {"xs", "ys", "zs", "distance", "valid"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise KeyError(f"ESDF NPZ is missing fields: {', '.join(missing)}")
        xs = np.asarray(data["xs"], dtype=np.float32)
        ys = np.asarray(data["ys"], dtype=np.float32)
        zs = np.asarray(data["zs"], dtype=np.float32)
        distance = np.asarray(data["distance"], dtype=np.float32)
        valid = np.asarray(data["valid"], dtype=bool)

    if xs.ndim != 1 or ys.ndim != 1 or zs.ndim != 1:
        raise ValueError("xs, ys, and zs must be one-dimensional")
    expected_shape = (len(xs), len(ys), len(zs))
    if distance.shape != expected_shape:
        raise ValueError(
            f"distance shape {distance.shape} does not match {expected_shape}"
        )
    if valid.shape != expected_shape:
        raise ValueError(f"valid shape {valid.shape} does not match {expected_shape}")
    for name, axis in (("xs", xs), ("ys", ys), ("zs", zs)):
        if len(axis) < 2 or not np.all(np.diff(axis) > 0.0):
            raise ValueError(f"{name} must contain at least two increasing samples")

    resolution = float(
        max(
            np.median(np.diff(xs)),
            np.median(np.diff(ys)),
            np.median(np.diff(zs)),
        )
    )
    return path, xs, ys, zs, distance, valid, resolution


def _surface_points(xs, ys, zs, distance, valid, band, max_points):
    """Extract a deterministic subset of valid voxels close to distance zero."""
    surface_mask = valid & np.isfinite(distance) & (np.abs(distance) <= band)
    surface_indices = np.flatnonzero(surface_mask.reshape(-1))
    total_surface_points = int(surface_indices.size)
    if total_surface_points == 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            0,
        )

    if max_points > 0 and total_surface_points > max_points:
        sample_positions = np.linspace(
            0, total_surface_points - 1, max_points, dtype=np.int64
        )
        surface_indices = surface_indices[sample_positions]

    ny, nz = len(ys), len(zs)
    yz_size = ny * nz
    ix = surface_indices // yz_size
    iy = (surface_indices // nz) % ny
    iz = surface_indices % nz
    points = np.column_stack((xs[ix], ys[iy], zs[iz])).astype(np.float32)
    signed_distances = distance.reshape(-1)[surface_indices].astype(np.float32)
    return points, signed_distances, total_surface_points


def _distance_colors(distances, band):
    """Map negative/zero/positive distance to red/white/blue."""
    normalized = np.clip(distances / max(float(band), 1.0e-9), -1.0, 1.0)
    colors = np.ones((len(distances), 3), dtype=np.float32)
    negative = normalized < 0.0
    positive = normalized > 0.0
    colors[negative, 1] = 1.0 + normalized[negative]
    colors[negative, 2] = 1.0 + normalized[negative]
    colors[positive, 0] = 1.0 - normalized[positive]
    colors[positive, 1] = 1.0 - normalized[positive]
    return colors


def _height_colors(points, z_min, z_max):
    """Map height to a compact blue-cyan-yellow-red palette."""
    span = max(float(z_max - z_min), 1.0e-9)
    value = np.clip((points[:, 2] - z_min) / span, 0.0, 1.0)
    colors = np.empty((len(points), 3), dtype=np.float32)
    colors[:, 0] = np.clip(2.0 * value - 0.5, 0.0, 1.0)
    colors[:, 1] = np.clip(1.5 - np.abs(2.0 * value - 1.0), 0.0, 1.0)
    colors[:, 2] = np.clip(1.5 - 2.0 * value, 0.0, 1.0)
    return colors


def _add_line(client_id, start, end, color, width=1.0):
    pyb.addUserDebugLine(
        start,
        end,
        lineColorRGB=color,
        lineWidth=width,
        lifeTime=0.0,
        physicsClientId=client_id,
    )


def _draw_bounds(client_id, xyz_min, xyz_max):
    xmin, ymin, zmin = xyz_min
    xmax, ymax, zmax = xyz_max
    corners = [
        [xmin, ymin, zmin],
        [xmax, ymin, zmin],
        [xmax, ymax, zmin],
        [xmin, ymax, zmin],
        [xmin, ymin, zmax],
        [xmax, ymin, zmax],
        [xmax, ymax, zmax],
        [xmin, ymax, zmax],
    ]
    for start_idx, end_idx in (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ):
        _add_line(client_id, corners[start_idx], corners[end_idx], [0.4] * 3)


def _draw_axes(client_id, length):
    origin = [0.0, 0.0, 0.0]
    _add_line(client_id, origin, [length, 0.0, 0.0], [1.0, 0.0, 0.0], 3.0)
    _add_line(client_id, origin, [0.0, length, 0.0], [0.0, 1.0, 0.0], 3.0)
    _add_line(client_id, origin, [0.0, 0.0, length], [0.0, 0.4, 1.0], 3.0)


def _draw_points(client_id, points, colors, point_size, batch_size):
    item_ids = []
    for start in range(0, len(points), batch_size):
        stop = min(start + batch_size, len(points))
        item_ids.append(
            pyb.addUserDebugPoints(
                pointPositions=points[start:stop].tolist(),
                pointColorsRGB=colors[start:stop].tolist(),
                pointSize=point_size,
                lifeTime=0.0,
                physicsClientId=client_id,
            )
        )
    return item_ids


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct an approximate obstacle surface from esdf_grid.npz "
            "and display it as a 3D point cloud."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("npz", help="Path to esdf_grid.npz")
    parser.add_argument(
        "--surface-band",
        type=float,
        default=None,
        help="Show valid samples with abs(distance) below this value; defaults to 1.5 grid cells",
    )
    parser.add_argument("--max-points", type=int, default=250000)
    parser.add_argument("--point-size", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=50000)
    parser.add_argument(
        "--color-mode", choices=("distance", "height"), default="height"
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.max_points <= 0:
        raise ValueError("max_points must be positive")
    if args.point_size <= 0.0:
        raise ValueError("point_size must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    path, xs, ys, zs, distance, valid, resolution = _load_esdf_npz(args.npz)
    band = 1.5 * resolution if args.surface_band is None else float(args.surface_band)
    if not np.isfinite(band) or band <= 0.0:
        raise ValueError("surface_band must be a positive finite number")

    points, signed_distances, total_surface_points = _surface_points(
        xs, ys, zs, distance, valid, band, args.max_points
    )
    if len(points) == 0:
        raise RuntimeError(
            f"No valid ESDF samples found within {band:.4f} m of zero; "
            "try a larger --surface-band"
        )

    if args.color_mode == "distance":
        colors = _distance_colors(signed_distances, band)
    else:
        colors = _height_colors(points, float(zs[0]), float(zs[-1]))

    xyz_min = np.array([xs[0], ys[0], zs[0]], dtype=float)
    xyz_max = np.array([xs[-1], ys[-1], zs[-1]], dtype=float)
    center = 0.5 * (xyz_min + xyz_max)
    span = xyz_max - xyz_min

    print(f"Loaded: {path}", flush=True)
    print(
        f"Grid: {len(xs)} x {len(ys)} x {len(zs)}, " f"resolution≈{resolution:.4f} m",
        flush=True,
    )
    print(
        f"Surface band: ±{band:.4f} m, displaying {len(points)}/"
        f"{total_surface_points} surface samples",
        flush=True,
    )
    print("Mouse: orbit/zoom/pan. Press Q in the PyBullet window to close.", flush=True)

    client_id = pyb.connect(pyb.GUI, options="--width=1280 --height=800")
    try:
        pyb.configureDebugVisualizer(pyb.COV_ENABLE_GUI, 0, physicsClientId=client_id)
        pyb.configureDebugVisualizer(
            pyb.COV_ENABLE_SHADOWS, 0, physicsClientId=client_id
        )
        pyb.resetDebugVisualizerCamera(
            cameraDistance=max(2.0, 0.85 * float(np.linalg.norm(span))),
            cameraYaw=45.0,
            cameraPitch=-35.0,
            cameraTargetPosition=center.tolist(),
            physicsClientId=client_id,
        )
        _draw_bounds(client_id, xyz_min, xyz_max)
        _draw_axes(client_id, max(0.25, 0.08 * float(np.max(span))))
        _draw_points(
            client_id,
            points,
            colors,
            float(args.point_size),
            int(args.batch_size),
        )
        pyb.addUserDebugText(
            f"ESDF zero surface | band=±{band:.3f}m | points={len(points)}",
            (xyz_min + np.array([0.0, 0.0, 0.05])).tolist(),
            textColorRGB=[1.0, 1.0, 1.0],
            textSize=1.2,
            lifeTime=0.0,
            physicsClientId=client_id,
        )

        while pyb.isConnected(client_id):
            events = pyb.getKeyboardEvents(physicsClientId=client_id)
            if events.get(ord("q"), 0) & pyb.KEY_WAS_TRIGGERED:
                break
            time.sleep(0.02)
    finally:
        if pyb.isConnected(client_id):
            pyb.disconnect(client_id)


if __name__ == "__main__":
    main()
