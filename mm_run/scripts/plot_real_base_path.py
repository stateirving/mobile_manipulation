#!/usr/bin/env python3
"""Plot WB-MPC's generated base path and the measured real trajectory."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mm_plot_real_base_path_mpl")

import matplotlib.pyplot as plt
import numpy as np


def _jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {error}"
                ) from error


def _load_plan(path: Path, replan_count: int | None):
    available = set()
    first = None
    for record in _jsonl(path):
        waypoints = record.get("ompl_path_waypoints_map")
        if not waypoints:
            continue
        count = record.get("ompl_replan_count")
        if count is not None:
            available.add(int(count))
        if first is None:
            first = record
        if replan_count is None or count == replan_count:
            return record
    if replan_count is not None:
        raise ValueError(
            f"replan count {replan_count} not found in {path}; "
            f"available: {sorted(available)}"
        )
    if first is None:
        raise ValueError(f"no OMPL path was recorded in {path}")
    return first


def _load_measured_trajectory(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    points = [record.get("base_pose_map") for record in _jsonl(path)]
    points = [point for point in points if point is not None]
    if not points:
        raise ValueError(f"no base_pose_map records found in {path}")
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1] < 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"invalid base_pose_map data in {path}")
    return array


def _limits(values: np.ndarray, minimum_padding: float):
    low = float(np.min(values))
    high = float(np.max(values))
    padding = max(minimum_padding, 0.08 * max(high - low, minimum_padding))
    return low - padding, high + padding


def plot(args) -> dict:
    solver = _load_plan(args.wbmpc_log, args.replan_count)
    path = np.asarray(solver["ompl_path_waypoints_map"], dtype=float)
    if path.ndim != 2 or path.shape[1] != 3 or not np.all(np.isfinite(path)):
        raise ValueError("ompl_path_waypoints_map must be a finite Nx3 array")
    measured = _load_measured_trajectory(args.adapter_log)

    logged_target = solver.get("base_target_map")
    target = (
        np.asarray(args.target, dtype=float)
        if args.target is not None
        else (
            np.asarray(logged_target, dtype=float)
            if logged_target is not None
            else path[-1].copy()
        )
    )
    target_source = (
        "command line"
        if args.target is not None
        else "WB-MPC log" if logged_target is not None else "path endpoint fallback"
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), constrained_layout=True)
    start = path[0]
    endpoint = path[-1]
    for axis in axes:
        axis.plot(
            path[:, 0],
            path[:, 1],
            color="#1565c0",
            linewidth=3.2,
            label="Generated OMPL path",
        )
        axis.plot(
            [start[0], target[0]],
            [start[1], target[1]],
            color="#555555",
            linestyle="--",
            linewidth=1.8,
            label="Straight line to exact target",
        )
        if measured is not None:
            axis.plot(
                measured[:, 0],
                measured[:, 1],
                color="#d32f2f",
                linewidth=2.2,
                label="Measured real trajectory",
            )
        axis.scatter(start[0], start[1], s=90, color="#2e7d32", zorder=5, label="Start")
        axis.scatter(
            endpoint[0],
            endpoint[1],
            s=90,
            color="#ef6c00",
            marker="D",
            zorder=5,
            label="OMPL endpoint",
        )
        axis.scatter(
            target[0],
            target[1],
            s=170,
            color="#6a1b9a",
            marker="*",
            zorder=6,
            label="Exact target",
        )
        axis.axhline(0.0, color="0.7", linewidth=1.0)
        axis.grid(True, alpha=0.3)
        axis.set_xlabel("map X [m] (forward)")
        axis.set_ylabel("map Y [m] (left + / right -)")

    full_sets = [path[:, :2], target[None, :2]]
    if measured is not None:
        full_sets.append(measured[:, :2])
    full = np.vstack(full_sets)
    axes[0].set_xlim(*_limits(full[:, 0], 0.05))
    axes[0].set_ylim(*_limits(full[:, 1], 0.05))
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_title("True-scale top view")

    plan_detail = np.vstack((path[:, :2], target[None, :2]))
    axes[1].set_xlim(*_limits(plan_detail[:, 0], 0.05))
    axes[1].set_ylim(*_limits(plan_detail[:, 1], 0.005))
    axes[1].set_title("Generated path detail (Y deviation magnified)")
    axes[1].legend(loc="best", fontsize=8)

    endpoint_error = float(np.linalg.norm(endpoint[:2] - target[:2]))
    tangent_yaw = float(np.arctan2(endpoint[1] - start[1], endpoint[0] - start[0]))
    title = args.title or (
        f"Base path top view: task {solver.get('task_index', '?')}, "
        f"replan {solver.get('ompl_replan_count', '?')}"
    )
    fig.suptitle(title, fontsize=14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi)
    plt.close(fig)
    return {
        "output": str(args.output.resolve()),
        "waypoint_count": int(len(path)),
        "start_map": start.tolist(),
        "ompl_endpoint_map": endpoint.tolist(),
        "target_map": target.tolist(),
        "target_source": target_source,
        "endpoint_xy_error_m": endpoint_error,
        "start_to_endpoint_tangent_yaw_deg": float(np.degrees(tangent_yaw)),
        "measured_point_count": 0 if measured is None else int(len(measured)),
    }


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wbmpc-log",
        type=Path,
        default=Path("/tmp/stretch_sim_esdf_wbmpc_execute.jsonl"),
    )
    parser.add_argument(
        "--adapter-log",
        type=Path,
        default=Path("/tmp/stretch_sim_esdf_adapter_execute.jsonl"),
        help="Adapter JSONL containing measured base_pose_map records.",
    )
    parser.add_argument(
        "--no-measured",
        action="store_true",
        help="Plot only the generated path, without adapter measurements.",
    )
    parser.add_argument(
        "--replan-count",
        type=int,
        help="Select an OMPL replan count; default is the first recorded path.",
    )
    parser.add_argument(
        "--target",
        nargs=3,
        type=float,
        metavar=("X", "Y", "YAW"),
        help="Override the exact map-frame target for old logs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/diagnostics/latest_base_path_topdown.png"),
    )
    parser.add_argument("--title")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args(argv)
    if args.no_measured:
        args.adapter_log = None
    return args


def main(argv=None):
    summary = plot(_parse_args(argv))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
