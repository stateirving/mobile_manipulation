"""Keyboard teleoperation and offline-compatible ESDF NPZ export.

This script intentionally stops after exporting the map.  The exported
``esdf_grid.npz`` has the same schema consumed by ``mm_control.esdf_map.ESDFMap``
and can therefore be passed to the existing offline OMPL/WB-MPC profile.
"""

import argparse
import datetime
import json
import math
import os
import queue
import secrets
import subprocess
import sys
import threading
import time
from multiprocessing.connection import Listener
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pybullet as pyb
from experiment_online_nvblox import _run_realtime_online_scan
from export_nvblox_esdf import make_grid_axes

from mm_control.esdf_map import ESDFMap, OnlineNvbloxESDFMap
from mm_simulator import simulation
from mm_utils import parsing


class _NoopDiagnostics:
    """Minimal diagnostics adapter required by the shared camera pipeline."""

    enabled = False

    @staticmethod
    def zero_stats():
        return {}

    @staticmethod
    def add_render_stats(*_args, **_kwargs):
        return None


def _invalid_frontier(valid):
    """Return invalid voxels adjacent to at least one 6-connected valid voxel."""
    valid = np.asarray(valid, dtype=bool)
    adjacent_valid = np.zeros_like(valid)
    adjacent_valid[1:, :, :] |= valid[:-1, :, :]
    adjacent_valid[:-1, :, :] |= valid[1:, :, :]
    adjacent_valid[:, 1:, :] |= valid[:, :-1, :]
    adjacent_valid[:, :-1, :] |= valid[:, 1:, :]
    adjacent_valid[:, :, 1:] |= valid[:, :, :-1]
    adjacent_valid[:, :, :-1] |= valid[:, :, 1:]
    return (~valid) & adjacent_valid


def _subsample_points(points, max_points):
    if len(points) <= max_points:
        return points
    indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
    return points[indices]


class _ViewerPublisher:
    """Latest-only asynchronous publisher for the isolated viewer process."""

    def __init__(self, batch_size, mode="GUI"):
        self._listener = None
        self._process = None
        self._connection = None
        self._updates = None
        self._worker_error = None
        self._worker = None
        try:
            authkey = secrets.token_bytes(24)
            self._listener = Listener(("127.0.0.1", 0), authkey=authkey)
            host, port = self._listener.address
            viewer_path = Path(__file__).with_name("live_esdf_viewer.py")
            command = [
                sys.executable,
                str(viewer_path),
                "--host",
                str(host),
                "--port",
                str(port),
                "--authkey",
                authkey.hex(),
                "--batch-size",
                str(batch_size),
                "--mode",
                str(mode).upper(),
            ]
            self._process = subprocess.Popen(command)
            accepted = queue.Queue(maxsize=1)

            def accept_connection():
                try:
                    accepted.put((self._listener.accept(), None))
                except Exception as exc:
                    accepted.put((None, exc))

            accept_thread = threading.Thread(target=accept_connection, daemon=True)
            accept_thread.start()
            while True:
                try:
                    self._connection, error = accepted.get(timeout=0.1)
                    break
                except queue.Empty:
                    if self._process.poll() is not None:
                        raise RuntimeError("Live ESDF viewer exited during startup")
            self._listener.close()
            self._listener = None
            if error is not None:
                raise RuntimeError("Live ESDF viewer connection failed") from error
            if not self._connection.poll(15.0):
                raise RuntimeError("Timed out opening the live ESDF viewer")
            ready = self._connection.recv()
            if not ready.get("ready", False):
                raise RuntimeError("Live ESDF viewer could not open a GUI window")

            self._updates = queue.Queue(maxsize=1)
            self._worker = threading.Thread(target=self._send_updates, daemon=True)
            self._worker.start()
        except Exception:
            if self._listener is not None:
                self._listener.close()
                self._listener = None
            self.close()
            raise

    def _send_updates(self):
        try:
            while True:
                update = self._updates.get()
                self._connection.send(update)
                if update is None:
                    return
        except Exception as exc:
            self._worker_error = exc

    def publish(self, update):
        if self._process.poll() is not None:
            raise RuntimeError("Live ESDF viewer has closed")
        if self._worker_error is not None:
            raise RuntimeError(
                "Live ESDF viewer connection failed"
            ) from self._worker_error
        try:
            self._updates.put_nowait(update)
        except queue.Full:
            try:
                self._updates.get_nowait()
            except queue.Empty:
                pass
            self._updates.put_nowait(update)

    def close(self):
        updates = getattr(self, "_updates", None)
        if updates is not None:
            try:
                updates.get_nowait()
            except queue.Empty:
                pass
            try:
                updates.put_nowait(None)
            except queue.Full:
                pass
        worker = getattr(self, "_worker", None)
        if worker is not None:
            worker.join(timeout=2.0)
        connection = getattr(self, "_connection", None)
        if connection is not None:
            connection.close()
        process = getattr(self, "_process", None)
        if process is not None and process.poll() is None:
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=3.0)


class _LiveESDFReconstruction:
    """Query a coarse online ESDF and publish only reconstructed map data."""

    def __init__(self, config, bounds):
        live_config = config.get("live_reconstruction", {})
        self.enabled = bool(live_config.get("enabled", True))
        self.update_interval_steps = int(live_config.get("update_interval_steps", 30))
        self.resolution = float(live_config.get("resolution", 0.10))
        self.surface_band = float(
            live_config.get("surface_band", 1.5 * self.resolution)
        )
        self.max_surface_points = int(live_config.get("max_surface_points", 80000))
        self.max_frontier_points = int(live_config.get("max_frontier_points", 30000))
        self.robot_query_z = np.asarray(
            live_config.get("robot_query_z", [0.15, 0.35]), dtype=np.float32
        )
        self.robot_required_distance = float(
            live_config.get("robot_required_distance", 0.40)
        )
        self.bounds = np.asarray(bounds, dtype=np.float32)
        self._publisher = None

        if not self.enabled:
            return
        if self.update_interval_steps <= 0 or self.resolution <= 0.0:
            raise ValueError(
                "live_reconstruction interval and resolution must be positive"
            )
        if self.surface_band <= 0.0:
            raise ValueError("live_reconstruction.surface_band must be positive")
        if min(self.max_surface_points, self.max_frontier_points) <= 0:
            raise ValueError("live_reconstruction point limits must be positive")

        self.xs, self.ys, self.zs = make_grid_axes(bounds, self.resolution)
        self.shape = (len(self.xs), len(self.ys), len(self.zs))
        grid = np.meshgrid(self.xs, self.ys, self.zs, indexing="ij")
        self.query_points = np.column_stack([axis.reshape(-1) for axis in grid]).astype(
            np.float32
        )
        self._batch_size = int(live_config.get("viewer_batch_size", 50000))
        self._viewer_mode = str(live_config.get("viewer_mode", "GUI")).upper()

    def start(self):
        if not self.enabled:
            return
        self._publisher = _ViewerPublisher(self._batch_size, self._viewer_mode)

    def update(self, esdf_map, base_pose, step_idx):
        if not self.enabled or step_idx % self.update_interval_steps != 0:
            return None
        distances, _, valid = esdf_map.query(self.query_points)
        distances = np.asarray(distances, dtype=np.float32)
        valid = np.asarray(valid, dtype=bool)
        valid_grid = valid.reshape(self.shape)

        surface_mask = (
            valid & np.isfinite(distances) & (np.abs(distances) <= self.surface_band)
        )
        surface = _subsample_points(
            self.query_points[surface_mask], self.max_surface_points
        )
        frontier_mask = _invalid_frontier(valid_grid).reshape(-1)
        frontier = _subsample_points(
            self.query_points[frontier_mask], self.max_frontier_points
        )

        base_pose = np.asarray(base_pose, dtype=np.float32)
        robot_points = np.column_stack(
            [
                np.full(len(self.robot_query_z), base_pose[0], dtype=np.float32),
                np.full(len(self.robot_query_z), base_pose[1], dtype=np.float32),
                self.robot_query_z,
            ]
        )
        robot_distances, _, robot_valid = esdf_map.query(robot_points)
        robot_distances = np.asarray(robot_distances, dtype=np.float32)
        robot_valid = np.asarray(robot_valid, dtype=bool)
        robot_status = np.zeros(len(robot_points), dtype=np.int8)
        robot_status[robot_valid] = 1
        robot_status[
            robot_valid & (robot_distances >= self.robot_required_distance)
        ] = 2

        update = {
            "bounds": self.bounds,
            "surface": surface,
            "frontier": frontier,
            "robot_points": robot_points,
            "robot_status": robot_status,
            "base_pose": base_pose,
            "known_ratio": float(np.count_nonzero(valid) / valid.size),
        }
        self._publisher.publish(update)
        return {
            "surface": len(surface),
            "frontier": len(frontier),
            "known_ratio": update["known_ratio"],
            "robot_valid": bool(np.all(robot_status == 2)),
        }

    def close(self):
        if self._publisher is not None:
            self._publisher.close()
            self._publisher = None


def _teleop_command(key_events, command_dim, linear_speed, angular_speed):
    """Convert held PyBullet GUI keys into a body-frame velocity command."""
    command = np.zeros(command_dim, dtype=float)

    def is_down(key):
        return bool(key_events.get(ord(key), 0) & pyb.KEY_IS_DOWN)

    # Avoid W/A/S/D: PyBullet reserves those keys for visualizer options such
    # as wireframe and shadows, and getKeyboardEvents cannot consume them.
    forward = float(is_down("i")) - float(is_down("k"))
    yaw = float(is_down("j")) - float(is_down("l"))
    command[0] = linear_speed * forward
    command[2] = angular_speed * yaw

    if key_events.get(ord(" "), 0) & (pyb.KEY_IS_DOWN | pyb.KEY_WAS_TRIGGERED):
        command[:3] = 0.0

    finish = bool(key_events.get(ord("x"), 0) & pyb.KEY_WAS_TRIGGERED)
    return command, finish


def _query_esdf_grid(esdf_map, bounds, resolution, chunk_size):
    """Sample an online ESDF through its public query API in bounded chunks."""
    xs, ys, zs = make_grid_axes(bounds, resolution)
    shape = (len(xs), len(ys), len(zs))
    total_points = int(np.prod(shape))
    distances = np.empty(shape, dtype=np.float32)
    gradients = np.empty(shape + (3,), dtype=np.float32)
    valid = np.empty(shape, dtype=bool)
    distances_flat = distances.reshape(-1)
    gradients_flat = gradients.reshape((-1, 3))
    valid_flat = valid.reshape(-1)

    num_chunks = int(math.ceil(total_points / chunk_size))
    print(
        f"Querying {total_points} ESDF samples "
        f"({len(xs)} x {len(ys)} x {len(zs)}, {num_chunks} chunks)...",
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
        points = np.column_stack((xs[ix], ys[iy], zs[iz]))

        chunk_distances, chunk_gradients, chunk_valid = esdf_map.query(points)
        distances_flat[start:stop] = np.asarray(chunk_distances, dtype=np.float32)
        gradients_flat[start:stop] = np.asarray(chunk_gradients, dtype=np.float32)
        valid_flat[start:stop] = np.asarray(chunk_valid, dtype=bool)

        if chunk_idx == 1 or chunk_idx == num_chunks or chunk_idx % 10 == 0:
            print(f"  ESDF query chunk {chunk_idx}/{num_chunks}", flush=True)

    return xs, ys, zs, distances, gradients, valid


def _export_npz(esdf_map, out_dir, bounds, resolution, chunk_size):
    """Finalize, sample, atomically save, and reload-validate an ESDF grid."""
    print("Updating final nvblox ESDF...", flush=True)
    if not esdf_map.update_esdf():
        raise RuntimeError("No depth frames were integrated; refusing to export")

    native_path = out_dir / "map.nvblox"
    print(f"Saving native nvblox checkpoint: {native_path}", flush=True)
    esdf_map.save_map(native_path)

    xs, ys, zs, distances, gradients, valid = _query_esdf_grid(
        esdf_map, bounds, resolution, chunk_size
    )
    valid_count = int(np.count_nonzero(valid))
    if valid_count == 0:
        raise RuntimeError("The sampled ESDF contains no valid grid points")

    final_path = out_dir / "esdf_grid.npz"
    temporary_path = out_dir / "esdf_grid.tmp.npz"
    try:
        np.savez_compressed(
            temporary_path,
            bounds=np.asarray(bounds, dtype=np.float32),
            resolution=np.asarray(resolution, dtype=np.float32),
            xs=xs,
            ys=ys,
            zs=zs,
            distance=distances,
            gradient=gradients,
            valid=valid,
        )
        # Exercise the exact loader used by the offline controller before the
        # artifact is made visible under its final name.
        ESDFMap(temporary_path)
        os.replace(temporary_path, final_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return final_path, native_path, valid_count, int(valid.size)


def _accumulate_stats(totals, current):
    for key, value in current.items():
        totals[key] = totals.get(key, 0) + value


def _positive_float(value, field_name):
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{field_name} must be a positive finite number")
    return value


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Teleoperate the simulated Stretch with the PyBullet GUI keyboard, "
            "integrate onboard depth frames, and export an offline ESDF NPZ."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-c", "--config", required=True, help="Experiment YAML")
    parser.add_argument("-o", "--output", default=None, help="Output root directory")
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Write directly into --output instead of a timestamped subdirectory",
    )
    parser.add_argument(
        "--bounds",
        nargs=6,
        type=float,
        default=None,
        metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"),
    )
    parser.add_argument("--grid-resolution", type=float, default=None)
    parser.add_argument("--query-chunk-size", type=int, default=None)
    parser.add_argument("--linear-speed", type=float, default=None)
    parser.add_argument("--angular-speed", type=float, default=None)
    parser.add_argument(
        "--max-duration",
        type=float,
        default=0.0,
        help="Automatically stop and export after this many simulated seconds; 0 disables",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    config = parsing.load_config(args.config)
    sim_config = config["simulation"]
    online_config = config["online_nvblox_sim"]
    export_config = config.get("teleop_esdf_export", {})

    # Keyboard events are read from the PyBullet GUI, so this mode always owns
    # a GUI connection regardless of the included simulation profile.
    sim_config["gui"] = True

    bounds = args.bounds or export_config.get(
        "bounds", [-4.2, -4.2, 0.0, 4.2, 4.2, 2.0]
    )
    if len(bounds) != 6 or np.any(np.asarray(bounds[:3]) >= np.asarray(bounds[3:])):
        raise ValueError("bounds must be XMIN YMIN ZMIN XMAX YMAX ZMAX")
    resolution = _positive_float(
        (
            args.grid_resolution
            if args.grid_resolution is not None
            else export_config.get("grid_resolution", 0.02)
        ),
        "grid_resolution",
    )
    chunk_size = int(
        args.query_chunk_size
        if args.query_chunk_size is not None
        else export_config.get("query_chunk_size", 131072)
    )
    if chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive")
    linear_speed = _positive_float(
        (
            args.linear_speed
            if args.linear_speed is not None
            else export_config.get("linear_speed", 0.25)
        ),
        "linear_speed",
    )
    angular_speed = _positive_float(
        (
            args.angular_speed
            if args.angular_speed is not None
            else export_config.get("angular_speed", 0.6)
        ),
        "angular_speed",
    )

    output_root = Path(
        args.output
        or export_config.get("output", "mm_run/results/nvblox_esdf/stretch_teleop")
    ).expanduser()
    timestamp = datetime.datetime.now()
    out_dir = (
        output_root
        if args.no_timestamp
        else output_root / timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    esdf_config = config["controller"]["esdf_collision"]
    if str(esdf_config.get("source", "")).lower() not in {
        "online",
        "online_nvblox",
        "nvblox",
    }:
        raise ValueError(
            "teleop export requires controller.esdf_collision.source=online_nvblox"
        )

    sim = None
    esdf_map = None
    live_reconstruction = None
    totals = {}
    t = 0.0
    step_idx = 0
    interrupted = False
    last_robot_configuration = None
    try:
        sim = simulation.BulletSimulation(
            config=sim_config, timestamp=timestamp, cli_args=args
        )
        esdf_map = OnlineNvbloxESDFMap.from_config(esdf_config)
        map_owner = SimpleNamespace(esdf_map=esdf_map)
        diagnostics = _NoopDiagnostics()
        live_reconstruction = _LiveESDFReconstruction(export_config, bounds)
        if live_reconstruction.enabled:
            try:
                live_reconstruction.start()
            except Exception as exc:
                live_reconstruction.enabled = False
                live_reconstruction.close()
                print(f"Warning: live ESDF viewer could not start: {exc}", flush=True)
        command_dim = int(sim_config["robot"]["dims"]["v"])
        last_robot_configuration, _ = sim.robot.joint_states(add_noise=False)

        print("\nKeyboard teleoperation is active in the PyBullet window:", flush=True)
        print("  hold I/K : forward/backward", flush=True)
        print("  hold J/L : turn left/right", flush=True)
        print("  SPACE    : stop", flush=True)
        print("  X        : stop, finalize, and export ESDF\n", flush=True)
        if live_reconstruction.enabled:
            print(
                "A separate empty PyBullet window shows only the reconstructed ESDF:",
                flush=True,
            )
            print("  height color : reconstructed zero surface", flush=True)
            print("  MAGENTA      : invalid-space frontier", flush=True)
            print("  YELLOW/RED   : current robot ESDF checks\n", flush=True)

        last_status_time = time.perf_counter()
        while pyb.isConnected():
            loop_start = time.perf_counter()
            key_events = pyb.getKeyboardEvents()
            command, finish = _teleop_command(
                key_events, command_dim, linear_speed, angular_speed
            )
            sim.robot.command_velocity(command, bodyframe=True)
            if finish:
                break

            q, _ = sim.robot.joint_states(add_noise=False)
            last_robot_configuration = q.copy()
            frame_stats, _ = _run_realtime_online_scan(
                map_owner,
                sim,
                online_config,
                q[:3],
                step_idx,
                preview=None,
                diagnostics=diagnostics,
            )
            _accumulate_stats(totals, frame_stats)
            if live_reconstruction.enabled:
                try:
                    live_stats = live_reconstruction.update(esdf_map, q[:3], step_idx)
                    if live_stats is not None:
                        print(
                            "live reconstruction "
                            f"surface={live_stats['surface']} "
                            f"frontier={live_stats['frontier']} "
                            f"known={100.0 * live_stats['known_ratio']:.1f}% "
                            f"robot_valid={live_stats['robot_valid']}",
                            flush=True,
                        )
                except Exception as exc:
                    live_reconstruction.enabled = False
                    live_reconstruction.close()
                    print(f"Warning: disabling live ESDF viewer: {exc}", flush=True)

            t, _ = sim.step(t)
            step_idx += 1
            if args.max_duration > 0.0 and t >= args.max_duration:
                print("Maximum duration reached; exporting ESDF.", flush=True)
                break

            now = time.perf_counter()
            if now - last_status_time >= float(
                export_config.get("status_interval", 2.0)
            ):
                print(
                    f"mapping t={t:.1f}s base=({q[0]:.2f}, {q[1]:.2f}, "
                    f"{q[2]:.2f}) frames={int(totals.get('frames', 0))}",
                    flush=True,
                )
                last_status_time = now

            sleep_time = sim.timestep - (time.perf_counter() - loop_start)
            if sleep_time > 0.0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        interrupted = True
        print("\nCtrl-C received; stopping and exporting collected ESDF.", flush=True)
    finally:
        if live_reconstruction is not None:
            live_reconstruction.close()
        if sim is not None and pyb.isConnected():
            zero = np.zeros(int(sim_config["robot"]["dims"]["v"]), dtype=float)
            sim.robot.command_velocity(zero, bodyframe=True)
            sim.step(t)

    if esdf_map is None:
        if pyb.isConnected():
            pyb.disconnect()
        raise RuntimeError("Online ESDF map was not initialized")

    try:
        npz_path, native_path, valid_count, total_count = _export_npz(
            esdf_map, out_dir, bounds, resolution, chunk_size
        )
        metadata = {
            "format_version": 1,
            "created_at": timestamp.isoformat(),
            "config": str(args.config),
            "npz_path": str(npz_path.resolve()),
            "native_map_path": str(native_path.resolve()),
            "bounds": [float(value) for value in bounds],
            "grid_resolution": resolution,
            "voxel_size": float(esdf_map.voxel_size),
            "query_radius": float(esdf_map.query_radius),
            "integrated_frames": int(totals.get("frames", 0)),
            "valid_grid_points": valid_count,
            "total_grid_points": total_count,
            "valid_ratio": float(valid_count / total_count),
            "simulation_time": float(t),
            "final_robot_configuration": last_robot_configuration.tolist(),
            "ended_by_keyboard_interrupt": interrupted,
        }
        with (out_dir / "metadata.json").open("w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2)

        print(f"\nOffline-compatible ESDF saved to: {npz_path.resolve()}", flush=True)
        print(f"Native nvblox checkpoint saved to: {native_path.resolve()}", flush=True)
        print(
            f"Valid grid points: {valid_count}/{total_count} "
            f"({100.0 * valid_count / total_count:.2f}%)",
            flush=True,
        )
    finally:
        if pyb.isConnected():
            pyb.disconnect()


if __name__ == "__main__":
    main()
