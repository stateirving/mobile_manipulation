"""Empty-scene PyBullet viewer for live low-resolution ESDF reconstructions."""

import argparse
import math
import time
from multiprocessing.connection import Client

import numpy as np
import pybullet as pyb


def _height_colors(points, z_min, z_max):
    span = max(float(z_max - z_min), 1.0e-9)
    value = np.clip((points[:, 2] - z_min) / span, 0.0, 1.0)
    colors = np.empty((len(points), 3), dtype=np.float32)
    colors[:, 0] = np.clip(2.0 * value - 0.5, 0.0, 1.0)
    colors[:, 1] = np.clip(1.5 - np.abs(2.0 * value - 1.0), 0.0, 1.0)
    colors[:, 2] = np.clip(1.5 - 2.0 * value, 0.0, 1.0)
    return colors


def _add_line(client_id, start, end, color, width=1.0):
    return pyb.addUserDebugLine(
        start,
        end,
        lineColorRGB=color,
        lineWidth=width,
        lifeTime=0.0,
        physicsClientId=client_id,
    )


def _draw_bounds(client_id, bounds):
    xmin, ymin, zmin, xmax, ymax, zmax = bounds
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
    item_ids = []
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
        item_ids.append(
            _add_line(client_id, corners[start_idx], corners[end_idx], [0.35] * 3)
        )
    axis_length = max(0.25, 0.08 * max(xmax - xmin, ymax - ymin, zmax - zmin))
    item_ids.extend(
        [
            _add_line(client_id, [0, 0, 0], [axis_length, 0, 0], [1, 0, 0], 3),
            _add_line(client_id, [0, 0, 0], [0, axis_length, 0], [0, 1, 0], 3),
            _add_line(client_id, [0, 0, 0], [0, 0, axis_length], [0, 0.4, 1], 3),
        ]
    )
    return item_ids


def _add_points(client_id, points, colors, point_size, batch_size):
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


def _remove_items(client_id, item_ids):
    for item_id in item_ids:
        pyb.removeUserDebugItem(item_id, physicsClientId=client_id)


def _render_update(client_id, update, dynamic_item_ids, batch_size):
    _remove_items(client_id, dynamic_item_ids)
    dynamic_item_ids.clear()

    bounds = np.asarray(update["bounds"], dtype=float)
    surface = np.asarray(update["surface"], dtype=np.float32).reshape((-1, 3))
    frontier = np.asarray(update["frontier"], dtype=np.float32).reshape((-1, 3))
    robot_points = np.asarray(update["robot_points"], dtype=np.float32).reshape((-1, 3))
    robot_status = np.asarray(update["robot_status"], dtype=np.int8)
    base_pose = np.asarray(update["base_pose"], dtype=float)

    if len(surface):
        colors = _height_colors(surface, bounds[2], bounds[5])
        dynamic_item_ids.extend(
            _add_points(client_id, surface, colors, 3.0, batch_size)
        )
    if len(frontier):
        colors = np.tile([1.0, 0.0, 0.8], (len(frontier), 1))
        dynamic_item_ids.extend(
            _add_points(client_id, frontier, colors, 4.0, batch_size)
        )
    if len(robot_points):
        status_colors = np.asarray(
            [
                [1.0, 0.0, 0.8],  # invalid/unknown
                [1.0, 0.0, 0.0],  # valid but unsafe clearance
                [1.0, 1.0, 0.0],  # valid and safe
            ],
            dtype=np.float32,
        )
        colors = status_colors[np.clip(robot_status, 0, 2)]
        dynamic_item_ids.extend(
            _add_points(client_id, robot_points, colors, 10.0, batch_size)
        )

    arrow_start = [float(base_pose[0]), float(base_pose[1]), 0.05]
    arrow_end = [
        arrow_start[0] + 0.35 * math.cos(float(base_pose[2])),
        arrow_start[1] + 0.35 * math.sin(float(base_pose[2])),
        arrow_start[2],
    ]
    dynamic_item_ids.append(
        _add_line(client_id, arrow_start, arrow_end, [1.0, 1.0, 0.0], 5.0)
    )

    all_safe = bool(len(robot_status) and np.all(robot_status == 2))
    status_text = "SAFE" if all_safe else "INVALID / UNSAFE"
    status_color = [0.2, 1.0, 0.2] if all_safe else [1.0, 0.1, 0.8]
    text_position = [float(bounds[0]), float(bounds[1]), float(bounds[5])]
    dynamic_item_ids.append(
        pyb.addUserDebugText(
            (
                f"LIVE ESDF ONLY | surface={len(surface)} | "
                f"invalid frontier={len(frontier)} | "
                f"known={100.0 * float(update['known_ratio']):.1f}% | "
                f"robot={status_text}"
            ),
            text_position,
            textColorRGB=status_color,
            textSize=1.2,
            lifeTime=0.0,
            physicsClientId=client_id,
        )
    )


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--authkey", required=True)
    parser.add_argument("--batch-size", type=int, default=50000)
    parser.add_argument("--mode", choices=("GUI", "DIRECT"), default="GUI")
    return parser.parse_args()


def main():
    args = _parse_args()
    connection = Client((args.host, args.port), authkey=bytes.fromhex(args.authkey))
    connection_mode = pyb.GUI if args.mode == "GUI" else pyb.DIRECT
    options = "--width=1100 --height=760" if args.mode == "GUI" else ""
    client_id = pyb.connect(connection_mode, options=options)
    connection.send({"ready": client_id >= 0})
    if client_id < 0:
        connection.close()
        raise RuntimeError("Could not open live ESDF PyBullet viewer")

    dynamic_item_ids = []
    static_item_ids = []
    current_bounds = None
    try:
        pyb.configureDebugVisualizer(pyb.COV_ENABLE_GUI, 0, physicsClientId=client_id)
        pyb.configureDebugVisualizer(
            pyb.COV_ENABLE_SHADOWS, 0, physicsClientId=client_id
        )
        while pyb.isConnected(client_id):
            latest = None
            while connection.poll(0.02 if latest is None else 0.0):
                message = connection.recv()
                if message is None:
                    return
                latest = message

            if latest is not None:
                bounds = tuple(float(value) for value in latest["bounds"])
                if bounds != current_bounds:
                    _remove_items(client_id, static_item_ids)
                    static_item_ids = _draw_bounds(client_id, bounds)
                    current_bounds = bounds
                    xyz_min = np.asarray(bounds[:3])
                    xyz_max = np.asarray(bounds[3:])
                    center = 0.5 * (xyz_min + xyz_max)
                    span = xyz_max - xyz_min
                    pyb.resetDebugVisualizerCamera(
                        cameraDistance=max(2.0, 0.85 * float(np.linalg.norm(span))),
                        cameraYaw=45.0,
                        cameraPitch=-35.0,
                        cameraTargetPosition=center.tolist(),
                        physicsClientId=client_id,
                    )
                _render_update(
                    client_id, latest, dynamic_item_ids, int(args.batch_size)
                )

            events = pyb.getKeyboardEvents(physicsClientId=client_id)
            if events.get(ord("q"), 0) & pyb.KEY_WAS_TRIGGERED:
                return
            time.sleep(0.01)
    except (EOFError, BrokenPipeError, ConnectionResetError):
        pass
    finally:
        connection.close()
        if pyb.isConnected(client_id):
            pyb.disconnect(client_id)


if __name__ == "__main__":
    main()
