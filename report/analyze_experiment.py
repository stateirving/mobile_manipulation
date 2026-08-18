#!/usr/bin/env python3
"""Compute report metrics and regenerate all data-derived figures.

The script deliberately reads only the archived ``data.npz`` file.  This keeps
the reported numbers reproducible without importing the controller or simulator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Avoid Type-3 bitmap fonts in the generated PDFs so the final IEEE-style
# document keeps searchable, embedded vector fonts.
plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
    }
)


DEADLINE_S = 0.12


def _rmse_norm(error: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(np.square(error), axis=1))))


def _stats_seconds(values: np.ndarray) -> dict[str, float]:
    return {
        "mean_ms": float(np.mean(values) * 1e3),
        "median_ms": float(np.median(values) * 1e3),
        "p95_ms": float(np.percentile(values, 95) * 1e3),
        "p99_ms": float(np.percentile(values, 99) * 1e3),
        "max_ms": float(np.max(values) * 1e3),
    }


def calculate_metrics(
    data: np.lib.npyio.NpzFile,
    source: Path,
    *,
    base_target: np.ndarray,
    ee_target: np.ndarray,
    safety_distance: float,
) -> dict:
    sample_count = int(data["ts"].size)
    base_count = int(data["r_bw_w_ds"].shape[0])
    ee_count = int(data["r_ew_w_ds"].shape[0])
    if base_count + ee_count != sample_count:
        raise ValueError(
            "Expected exactly one active reference per sample; got "
            f"{base_count} base + {ee_count} EE for {sample_count} samples"
        )

    base_error = data["r_bw_ws"][:base_count] - data["r_bw_w_ds"]
    yaw_error = np.arctan2(
        np.sin(data["yaw_bw_ws"] - data["yaw_bw_w_ds"]),
        np.cos(data["yaw_bw_ws"] - data["yaw_bw_w_ds"]),
    )
    ee_error = data["r_ew_ws"][base_count:] - data["r_ew_w_ds"]

    valid_node0 = data["mpc_esdf_node0_valids"]
    node0_margins = np.where(valid_node0, data["mpc_esdf_node0_marginss"], np.nan)
    node0_min_margin = np.nanmin(node0_margins, axis=1)

    controller_time = data["controller_run_time"]
    statuses, status_counts = np.unique(data["mpc_solver_statuss"], return_counts=True)
    hash_value = hashlib.sha256(source.read_bytes()).hexdigest()

    return {
        "source": str(source),
        "sha256": hash_value,
        "sample_count": sample_count,
        "simulated_duration_s": float(data["ts"][-1]),
        "simulation_timestep_s": float(data["sim_timestep"]),
        "base_reference_samples": base_count,
        "ee_reference_samples": ee_count,
        "base_task_end_s": float(data["ts"][base_count - 1]),
        "tracking": {
            "base_position_rmse_m": _rmse_norm(base_error),
            "base_x_rmse_m": float(np.sqrt(np.mean(np.square(base_error[:, 0])))),
            "base_y_rmse_m": float(np.sqrt(np.mean(np.square(base_error[:, 1])))),
            "base_yaw_rmse_rad": float(np.sqrt(np.mean(np.square(yaw_error)))),
            "base_final_reference_error_m": float(np.linalg.norm(base_error[-1])),
            "base_final_yaw_reference_error_rad": float(abs(yaw_error[-1])),
            "ee_position_rmse_m": _rmse_norm(ee_error),
            "ee_x_rmse_m": float(np.sqrt(np.mean(np.square(ee_error[:, 0])))),
            "ee_y_rmse_m": float(np.sqrt(np.mean(np.square(ee_error[:, 1])))),
            "ee_z_rmse_m": float(np.sqrt(np.mean(np.square(ee_error[:, 2])))),
            "ee_final_reference_error_m": float(np.linalg.norm(ee_error[-1])),
            "ee_final_task_target_error_m": float(
                np.linalg.norm(data["r_ew_ws"][-1] - ee_target)
            ),
            "base_drift_from_completed_target_at_end_m": float(
                np.linalg.norm(data["r_bw_ws"][-1] - base_target)
            ),
        },
        "collision": {
            "all_esdf_queries_valid": bool(np.all(data["mpc_esdf_all_valids"])),
            "invalid_query_count": int(data["mpc_esdf_invalid_query_counts"].sum()),
            "minimum_node0_safety_margin_m": float(np.nanmin(node0_min_margin)),
            "node0_margin_p05_m": float(np.nanpercentile(node0_min_margin, 5)),
            "minimum_warm_start_horizon_margin_m": float(
                np.nanmin(data["mpc_esdf_min_margins"])
            ),
            "configured_safety_distance_m": safety_distance,
            "minimum_node0_surface_clearance_m": float(
                np.nanmin(node0_min_margin) + safety_distance
            ),
        },
        "timing": {
            "controller": _stats_seconds(controller_time),
            "ocp_solve": _stats_seconds(data["mpc_time_ocp_solves"]),
            "parameter_update": _stats_seconds(data["mpc_time_ocp_set_paramss"]),
            "esdf_linearization": _stats_seconds(data["mpc_time_esdf_linearizations"]),
            "logging_overhead": _stats_seconds(data["mpc_time_ocp_overheads"]),
            "deadline_s": DEADLINE_S,
            "deadline_miss_count": int(np.count_nonzero(controller_time > DEADLINE_S)),
            "deadline_miss_fraction": float(np.mean(controller_time > DEADLINE_S)),
        },
        "solver": {
            "status_counts": {
                str(int(status)): int(count)
                for status, count in zip(statuses, status_counts, strict=True)
            },
            "accepted_nonzero_status_count": int(
                np.count_nonzero(data["mpc_solver_status_accepteds"])
            ),
            "fallback_count": int(np.count_nonzero(data["mpc_solver_fallbacks"])),
            "command_clip_count": int(np.count_nonzero(data["cmd_vel_clipped"])),
            "sqp_iterations_mean": float(np.mean(data["mpc_sqp_iters"])),
            "sqp_iterations_p95": float(np.percentile(data["mpc_sqp_iters"], 95)),
            "sqp_iterations_max": int(np.max(data["mpc_sqp_iters"])),
        },
    }


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_base_tracking(data: np.lib.npyio.NpzFile, output: Path) -> None:
    n = data["r_bw_w_ds"].shape[0]
    time_s = data["ts"][:n]
    error = data["r_bw_ws"][:n] - data["r_bw_w_ds"]
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.45))
    axes[0].plot(
        data["r_bw_w_ds"][:, 0], data["r_bw_w_ds"][:, 1], "--", label="reference"
    )
    axes[0].plot(data["r_bw_ws"][:n, 0], data["r_bw_ws"][:n, 1], label="measured")
    axes[0].scatter([0, 2], [0, 0], marker="x", color="black", zorder=3)
    axes[0].set(xlabel="$x$ (m)", ylabel="$y$ (m)", title="Base path")
    axes[0].axis("equal")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)
    axes[1].plot(time_s, 100 * error[:, 0], label="$e_x$")
    axes[1].plot(time_s, 100 * error[:, 1], label="$e_y$")
    axes[1].plot(time_s, 100 * np.linalg.norm(error, axis=1), label="$\\|e_p\\|$")
    axes[1].set(
        xlabel="simulation time (s)", ylabel="error (cm)", title="Reference error"
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, ncol=3, fontsize=8)
    fig.tight_layout()
    _save(fig, output / "base_tracking.pdf")


def plot_ee_tracking(data: np.lib.npyio.NpzFile, output: Path) -> None:
    offset = data["r_bw_w_ds"].shape[0]
    ref = data["r_ew_w_ds"]
    measured = data["r_ew_ws"][offset:]
    time_s = data["ts"][offset:]
    error_norm = np.linalg.norm(measured - ref, axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.45))
    labels = ("x", "y", "z")
    for axis, label in enumerate(labels):
        axes[0].plot(time_s, measured[:, axis], label=f"{label} measured")
        axes[0].plot(time_s, ref[:, axis], "--", linewidth=1.0, label=f"{label} ref.")
    axes[0].set(
        xlabel="simulation time (s)",
        ylabel="position (m)",
        title="End-effector position",
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, ncol=2, fontsize=7)
    axes[1].plot(time_s, 100 * error_norm, color="tab:red")
    axes[1].axhline(
        10, color="black", linestyle="--", linewidth=1, label="task tolerance"
    )
    axes[1].set(
        xlabel="simulation time (s)",
        ylabel="position error (cm)",
        title="End-effector reference error",
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, output / "ee_tracking.pdf")


def plot_timing_clearance(data: np.lib.npyio.NpzFile, output: Path) -> None:
    timing_ms = data["controller_run_time"] * 1e3
    valid = data["mpc_esdf_node0_valids"]
    margins = np.where(valid, data["mpc_esdf_node0_marginss"], np.nan)
    minimum_margin_cm = 100 * np.nanmin(margins, axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.45))
    axes[0].hist(timing_ms, bins=np.linspace(0, 800, 41), color="tab:blue", alpha=0.8)
    axes[0].axvline(120, color="tab:red", linestyle="--", label="120 ms deadline")
    axes[0].set(
        xlabel="controller time (ms)", ylabel="cycles", title="Runtime distribution"
    )
    axes[0].set_xlim(0, 800)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(data["ts"], minimum_margin_cm, color="tab:green")
    axes[1].axhline(
        0, color="tab:red", linestyle="--", linewidth=1, label="configured boundary"
    )
    axes[1].set(
        xlabel="simulation time (s)",
        ylabel="minimum margin (cm)",
        title="Current-state ESDF margin",
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, output / "timing_clearance.pdf")


def _box(ax, xy, width, height, text, color="#e9f2fb", fontsize=8, pad=0.02):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad={pad},rounding_size=0.02",
        facecolor=color,
        edgecolor="#29465b",
        linewidth=1,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
    )


def _arrow(ax, start, end, text=""):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            shrinkA=0,
            shrinkB=0,
            color="#29465b",
            linewidth=1,
        )
    )
    if text:
        ax.text(
            (start[0] + end[0]) / 2,
            (start[1] + end[1]) / 2 + 0.035,
            text,
            ha="center",
            fontsize=7,
        )


def plot_architecture(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.15, 2.25))
    ax.set(xlim=(0, 1), ylim=(-0.06, 1))
    ax.axis("off")
    box_pad = 0.01
    ax.text(0.02, 0.94, "Offline robot specialization", fontsize=8.3, weight="bold")
    _box(
        ax,
        (0.03, 0.67),
        0.17,
        0.17,
        "URDF + YAML\nrobot profile",
        fontsize=7.3,
        pad=box_pad,
    )
    _box(
        ax,
        (0.28, 0.67),
        0.20,
        0.17,
        "Pinocchio + CasADi\nmodel and OCP",
        fontsize=7.3,
        pad=box_pad,
    )
    _box(
        ax,
        (0.56, 0.67),
        0.18,
        0.17,
        "acados\ncode generation",
        fontsize=7.3,
        pad=box_pad,
    )
    _box(
        ax,
        (0.83, 0.67),
        0.14,
        0.17,
        "Robot-specific\nsolver",
        fontsize=7.3,
        color="#e8f4e5",
        pad=box_pad,
    )
    offline = [(0.03, 0.17), (0.28, 0.20), (0.56, 0.18), (0.83, 0.14)]
    for left, right in zip(offline[:-1], offline[1:], strict=True):
        _arrow(
            ax,
            (left[0] + left[1] + box_pad, 0.755),
            (right[0] - box_pad, 0.755),
        )

    ax.text(0.02, 0.54, "Online feedback control", fontsize=8.3, weight="bold")
    runtime = [
        (0.03, 0.12, "Control\ntarget"),
        (0.20, 0.19, "Task manager + OMPL\nreference horizon"),
        (0.45, 0.17, "ESDF-aware\nWB-MPC"),
        (0.68, 0.14, "Platform\nadapter"),
        (0.87, 0.10, "Robot /\nSimulator"),
    ]
    for x, width, label in runtime:
        _box(ax, (x, 0.15), width, 0.20, label, fontsize=7.3, pad=box_pad)
    for left, right in zip(runtime[:-1], runtime[1:], strict=True):
        _arrow(
            ax,
            (left[0] + left[1] + box_pad, 0.25),
            (right[0] - box_pad, 0.25),
        )

    _box(
        ax,
        (0.42, 0.49),
        0.12,
        0.08,
        "ESDF",
        fontsize=7.1,
        color="#fff1cf",
        pad=box_pad,
    )
    _arrow(ax, (0.48, 0.48), (0.48, 0.36))
    _arrow(ax, (0.90, 0.66), (0.59, 0.36))
    ax.add_patch(
        FancyArrowPatch(
            (0.93, 0.14),
            (0.54, 0.14),
            arrowstyle="-|>",
            mutation_scale=10,
            shrinkA=0,
            shrinkB=0,
            connectionstyle="arc3,rad=-0.13",
            color="#29465b",
            linewidth=1,
        )
    )
    ax.text(
        0.74,
        -0.015,
        "state feedback",
        ha="center",
        fontsize=7,
    )
    _save(fig, output / "system_overview.pdf")


def plot_mapping(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.15, 2.0))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    _box(ax, (0.02, 0.40), 0.22, 0.34, "Shared pipeline\nOMPL + ESDF +\nWB-MPC")
    _box(
        ax,
        (0.34, 0.55),
        0.27,
        0.28,
        "Stretch configuration\n8 optimized DoF\nnonholonomic dynamics",
        color="#f4f0df",
    )
    _box(
        ax,
        (0.34, 0.16),
        0.27,
        0.28,
        "Mobile UR10 configuration\n9 optimized DoF\nholonomic double integrator",
        color="#e9f2fb",
    )
    _box(
        ax,
        (0.72, 0.40),
        0.25,
        0.34,
        "Robot boundary\nURDF + limits + spheres\nstate/command mapping",
        color="#e8f4e5",
    )
    _arrow(ax, (0.24, 0.57), (0.34, 0.69), "YAML/URDF")
    _arrow(ax, (0.24, 0.53), (0.34, 0.30))
    _arrow(ax, (0.61, 0.69), (0.72, 0.60))
    _arrow(ax, (0.61, 0.30), (0.72, 0.50))
    _save(fig, output / "coordinate_mapping.pdf")


def plot_platform_comparison(metrics: dict, output: Path) -> None:
    names = ["Stretch\nnonholonomic", "Mobile UR10\nholonomic"]
    platform_metrics = [metrics["stretch"], metrics["ur10"]]
    tracking_mm = [m["tracking"]["ee_position_rmse_m"] * 1e3 for m in platform_metrics]
    controller_ms = [m["timing"]["controller"]["mean_ms"] for m in platform_metrics]
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.45))
    axes[0].bar(names, tracking_mm, color=["#4c78a8", "#72b7b2"])
    axes[0].set(ylabel="EE position RMSE (mm)", title="Reference tracking")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(names, controller_ms, color=["#4c78a8", "#72b7b2"])
    axes[1].axhline(
        120, color="tab:red", linestyle="--", linewidth=1, label="120 ms benchmark"
    )
    axes[1].set(ylabel="mean controller time (ms)", title="Complete control call")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, output / "platform_comparison.pdf")


def plot_platform_tracking(
    stretch: np.lib.npyio.NpzFile,
    ur10: np.lib.npyio.NpzFile,
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.7))
    for row, (name, data) in enumerate((("Stretch", stretch), ("Mobile UR10", ur10))):
        base_count = data["r_bw_w_ds"].shape[0]
        axes[row, 0].plot(
            data["r_bw_w_ds"][:, 0],
            data["r_bw_w_ds"][:, 1],
            "--",
            label="reference",
        )
        axes[row, 0].plot(
            data["r_bw_ws"][:base_count, 0],
            data["r_bw_ws"][:base_count, 1],
            label="measured",
        )
        axes[row, 0].set(
            xlabel="$x$ (m)",
            ylabel="$y$ (m)",
            title=f"{name}: base task",
        )
        axes[row, 0].axis("equal")
        axes[row, 0].grid(alpha=0.25)
        axes[row, 0].legend(frameon=False, fontsize=8)

        ee_error = np.linalg.norm(
            data["r_ew_ws"][base_count:] - data["r_ew_w_ds"], axis=1
        )
        axes[row, 1].plot(data["ts"][base_count:], 100 * ee_error)
        axes[row, 1].set(
            xlabel="simulation time (s)",
            ylabel="position error (cm)",
            title=f"{name}: EE reference error",
        )
        axes[row, 1].grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, output / "platform_tracking.pdf")


def plot_platform_timing_clearance(
    stretch: np.lib.npyio.NpzFile,
    ur10: np.lib.npyio.NpzFile,
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.7))
    for row, (name, data) in enumerate((("Stretch", stretch), ("Mobile UR10", ur10))):
        timing_ms = data["controller_run_time"] * 1e3
        axes[row, 0].hist(
            timing_ms,
            bins=np.linspace(0, 800, 41),
            color="tab:blue" if row == 0 else "#72b7b2",
            alpha=0.8,
        )
        axes[row, 0].axvline(
            120, color="tab:red", linestyle="--", linewidth=1, label="120 ms reference"
        )
        axes[row, 0].set(
            xlabel="controller time (ms)",
            ylabel="cycles",
            title=f"{name}: runtime distribution",
            xlim=(0, 800),
        )
        axes[row, 0].grid(axis="y", alpha=0.25)
        axes[row, 0].legend(frameon=False, fontsize=8)

        valid = data["mpc_esdf_node0_valids"]
        margins = np.where(valid, data["mpc_esdf_node0_marginss"], np.nan)
        axes[row, 1].plot(data["ts"], 100 * np.nanmin(margins, axis=1))
        axes[row, 1].axhline(
            0, color="tab:red", linestyle="--", linewidth=1, label="configured boundary"
        )
        axes[row, 1].set(
            xlabel="simulation time (s)",
            ylabel="minimum margin (cm)",
            title=f"{name}: executed-state margin",
        )
        axes[row, 1].grid(alpha=0.25)
        axes[row, 1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, output / "platform_timing_clearance.pdf")


def plot_safety_states(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.15, 2.1))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    states = [
        (0.03, "shadow", "#e9f2fb"),
        (0.29, "WB-MPC", "#e8f4e5"),
        (0.55, "hold", "#fff1cf"),
        (0.81, "latched", "#f8dfdf"),
    ]
    for x, name, color in states:
        _box(ax, (x, 0.43), 0.15, 0.25, name, color=color, fontsize=9)
    _arrow(ax, (0.18, 0.56), (0.29, 0.56), "preflight")
    _arrow(ax, (0.44, 0.56), (0.55, 0.56), "stale/deadline")
    _arrow(ax, (0.70, 0.56), (0.81, 0.56), "hard fault")
    _arrow(ax, (0.55, 0.38), (0.44, 0.25), "fresh valid plan")
    ax.text(0.50, 0.88, "Adapter state machine", ha="center", fontsize=9)
    ax.text(
        0.50,
        0.10,
        "Shadow creates no hardware publishers; latched stop requires restart and preflight.",
        ha="center",
        fontsize=8,
    )
    _save(fig, output / "safety_states.pdf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path, help="Archived Stretch raw data.npz")
    parser.add_argument(
        "--ur10-data", type=Path, help="Archived mobile-UR10 raw data.npz"
    )
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "figures"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    with np.load(args.data, allow_pickle=True) as data:
        stretch_metrics = calculate_metrics(
            data,
            args.data,
            base_target=np.array([2.0, 0.0]),
            ee_target=np.array([3.0, 0.5, 0.4]),
            safety_distance=0.07,
        )
        plot_base_tracking(data, args.output)
        plot_ee_tracking(data, args.output)
        plot_timing_clearance(data, args.output)
    metrics = {"stretch": stretch_metrics}
    if args.ur10_data is not None:
        with np.load(args.ur10_data, allow_pickle=True) as data:
            metrics["ur10"] = calculate_metrics(
                data,
                args.ur10_data,
                base_target=np.array([2.0, 0.0]),
                ee_target=np.array([3.5, 0.5, 0.4]),
                safety_distance=0.10,
            )
        plot_platform_comparison(metrics, args.output)
        with (
            np.load(args.data, allow_pickle=True) as stretch_data,
            np.load(args.ur10_data, allow_pickle=True) as ur10_data,
        ):
            plot_platform_tracking(stretch_data, ur10_data, args.output)
            plot_platform_timing_clearance(stretch_data, ur10_data, args.output)
    plot_architecture(args.output)
    plot_mapping(args.output)
    plot_safety_states(args.output)

    metrics_path = args.output.parent / "experiment_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
