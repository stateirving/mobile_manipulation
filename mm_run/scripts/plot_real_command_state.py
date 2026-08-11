#!/usr/bin/env python3
"""Plot real-deploy base and joint command/state signals from JSONL logs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mm_plot_real_command_state_mpl")

import matplotlib.pyplot as plt
import numpy as np

MODEL_JOINT_NAMES = (
    "joint_lift",
    "joint_arm_l3",
    "joint_arm_l2",
    "joint_arm_l1",
    "joint_arm_l0",
    "joint_wrist_yaw",
    "joint_wrist_pitch",
    "joint_wrist_roll",
)
SG3_QPOS_NAMES = (
    "wrist_extension",
    "joint_lift",
    "joint_wrist_yaw",
    "joint_wrist_pitch",
    "joint_wrist_roll",
    "joint_head_pan",
    "joint_head_tilt",
    "joint_gripper_finger_left",
    "base_translate_increment",
    "base_rotate_increment",
)


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {error}"
                ) from error
    return records


def _series(records, key, width=None):
    selected = [
        record
        for record in records
        if record.get("monotonic_time") is not None and record.get(key) is not None
    ]
    if not selected:
        return None, None
    values = np.asarray([record[key] for record in selected], dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if width is not None and values.shape[1] != width:
        raise ValueError(f"{key} has width {values.shape[1]}, expected {width}")
    return np.asarray([record["monotonic_time"] for record in selected]), values


def _relative(times, origin):
    return None if times is None else times - origin


def _legend(axis):
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(loc="best", fontsize=7, ncol=2)


def _save(fig, path: Path, dpi: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _plot_base(wbmpc, adapter, origin, window, output, dpi):
    ta, pose = _series(adapter, "base_pose_map", 3)
    _, measured_velocity = _series(adapter, "base_velocity_map", 3)
    tw, wb_velocity = _series(
        [record for record in wbmpc if record.get("record_type") == "command"],
        "velocity_command",
        11,
    )
    _, wb_acceleration = _series(
        [record for record in wbmpc if record.get("record_type") == "command"],
        "acceleration",
        11,
    )
    ts, safe_velocity = _series(adapter, "safe_driver_velocity", 7)
    if pose is None:
        raise ValueError("adapter log has no base_pose_map records")
    ta = _relative(ta, origin)
    tw = _relative(tw, origin)
    ts = _relative(ts, origin)

    fig, axes = plt.subplots(
        5, 1, figsize=(14, 15), sharex=True, constrained_layout=True
    )
    axes[0].plot(ta, pose[:, 0], label="state x")
    axes[0].plot(ta, pose[:, 1], label="state y")
    axes[0].set_ylabel("position [m]")
    axes[1].plot(ta, np.degrees(pose[:, 2]), label="state yaw")
    axes[1].set_ylabel("yaw [deg]")
    if measured_velocity is not None:
        axes[2].plot(ta, measured_velocity[:, 0], label="state vx map")
        axes[2].plot(ta, measured_velocity[:, 1], label="state vy map")
        axes[3].plot(ta, measured_velocity[:, 2], label="state yaw rate")
    if wb_velocity is not None:
        axes[2].plot(tw, wb_velocity[:, 0], "--", label="WB command vx map")
        axes[2].plot(tw, wb_velocity[:, 1], "--", label="WB command vy map")
        axes[3].plot(tw, wb_velocity[:, 2], "--", label="WB command yaw rate")
    if safe_velocity is not None:
        axes[2].plot(
            ts, safe_velocity[:, 0], ":", linewidth=2, label="adapter cmd forward"
        )
        axes[3].plot(
            ts, safe_velocity[:, 1], ":", linewidth=2, label="adapter cmd yaw rate"
        )
    axes[2].set_ylabel("linear velocity [m/s]")
    axes[3].set_ylabel("angular velocity [rad/s]")
    if wb_acceleration is not None:
        axes[4].plot(tw, wb_acceleration[:, 0], label="MPC ax map")
        axes[4].plot(tw, wb_acceleration[:, 1], label="MPC ay map")
        axes[4].plot(tw, wb_acceleration[:, 2], label="MPC yaw acceleration")
    axes[4].set_ylabel("MPC acceleration")
    axes[4].set_xlabel("time since first logged sample [s]")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.set_xlim(*window)
        _legend(axis)
    fig.suptitle("Base command and measured state", fontsize=15)
    _save(fig, output, dpi)


def _plot_localization(adapter, origin, window, output, dpi):
    tf_time, tf_pose = _series(adapter, "map_base_tf_pose_raw", 3)
    odom_time, odom_pose = _series(adapter, "odom_pose_raw", 3)
    _, odom_twist = _series(adapter, "odom_twist_body_raw", 3)
    fused_time, fused_pose = _series(adapter, "base_pose_map", 3)
    _, fused_velocity = _series(adapter, "base_velocity_map", 3)
    skew_time, tf_odom_skew = _series(adapter, "map_base_tf_odom_skew", 1)
    if tf_pose is None or odom_pose is None or fused_pose is None:
        raise ValueError(
            "adapter log needs map_base_tf_pose_raw, odom_pose_raw, and "
            "base_pose_map records; run once with the updated adapter"
        )

    # Put raw odometry into map coordinates using only the first TF/odom pair.
    # Any later divergence is therefore visible instead of being hidden by
    # continuously re-anchoring odometry.
    heading_offset = float(tf_pose[0, 2] - odom_pose[0, 2])
    cosine, sine = np.cos(heading_offset), np.sin(heading_offset)
    rotation = np.asarray([[cosine, -sine], [sine, cosine]])
    odom_aligned = np.empty_like(odom_pose)
    odom_aligned[:, :2] = (odom_pose[:, :2] - odom_pose[0, :2]) @ rotation.T + tf_pose[
        0, :2
    ]
    odom_aligned[:, 2] = odom_pose[:, 2] + heading_offset

    tf_time = _relative(tf_time, origin)
    odom_time = _relative(odom_time, origin)
    fused_time = _relative(fused_time, origin)
    skew_time = _relative(skew_time, origin)
    tf_yaw = np.degrees(np.unwrap(tf_pose[:, 2]))
    odom_yaw = np.degrees(np.unwrap(odom_aligned[:, 2]))
    fused_yaw = np.degrees(np.unwrap(fused_pose[:, 2]))

    fig, axes = plt.subplots(
        5, 1, figsize=(14, 17), sharex=True, constrained_layout=True
    )
    for index, label in enumerate(("x", "y")):
        axes[index].plot(
            fused_time, fused_pose[:, index], label=f"adapter fused {label}"
        )
        axes[index].plot(
            tf_time,
            tf_pose[:, index],
            ".",
            markersize=2,
            label=f"raw map→base TF {label}",
        )
        axes[index].plot(
            odom_time,
            odom_aligned[:, index],
            "--",
            label=f"raw odom aligned once {label}",
        )
        axes[index].set_ylabel(f"map {label} [m]")
    axes[2].plot(fused_time, fused_yaw, label="adapter fused yaw")
    axes[2].plot(tf_time, tf_yaw, ".", markersize=2, label="raw map→base TF yaw")
    axes[2].plot(odom_time, odom_yaw, "--", label="raw odom yaw aligned once")
    axes[2].set_ylabel("yaw [deg]")
    if odom_twist is not None:
        axes[3].plot(odom_time, odom_twist[:, 0], label="raw odom body vx")
        axes[3].plot(odom_time, odom_twist[:, 1], label="raw odom body vy")
        axes[3].plot(odom_time, odom_twist[:, 2], label="raw odom yaw rate")
    if fused_velocity is not None:
        axes[3].plot(fused_time, fused_velocity[:, 0], "--", label="adapter map vx")
        axes[3].plot(fused_time, fused_velocity[:, 1], "--", label="adapter map vy")
        axes[3].plot(fused_time, fused_velocity[:, 2], "--", label="adapter yaw rate")
    axes[3].set_ylabel("velocity")
    if tf_odom_skew is not None:
        axes[4].plot(
            skew_time, 1000.0 * tf_odom_skew[:, 0], label="TF stamp − odom stamp"
        )
    axes[4].axhline(0.0, color="0.5", linewidth=1)
    axes[4].set_ylabel("timestamp skew [ms]")
    axes[4].set_xlabel("time since first logged sample [s]")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.set_xlim(*window)
        _legend(axis)
    fig.suptitle("Raw odometry, raw map TF, and adapter-fused base state", fontsize=15)
    _save(fig, output, dpi)


def _plot_model_joints(wbmpc, adapter, origin, window, output, dpi):
    ta, measured_velocity = _series(adapter, "measured_model_joint_velocity", 8)
    tr, realized_velocity = _series(adapter, "realized_model_velocity", 11)
    commands = [record for record in wbmpc if record.get("record_type") == "command"]
    tw, wb_velocity = _series(commands, "velocity_command", 11)
    _, acceleration = _series(commands, "acceleration", 11)
    ta = _relative(ta, origin)
    tr = _relative(tr, origin)
    tw = _relative(tw, origin)

    fig, axes = plt.subplots(
        8, 2, figsize=(16, 24), sharex=True, constrained_layout=True
    )
    for index, name in enumerate(MODEL_JOINT_NAMES):
        velocity_axis, acceleration_axis = axes[index]
        if measured_velocity is not None:
            velocity_axis.plot(ta, measured_velocity[:, index], label="measured state")
        if wb_velocity is not None:
            velocity_axis.plot(
                tw, wb_velocity[:, index + 3], "--", label="WB velocity command"
            )
        if realized_velocity is not None:
            velocity_axis.plot(
                tr,
                realized_velocity[:, index + 3],
                ":",
                linewidth=2,
                label="adapter realized command",
            )
        if acceleration is not None:
            acceleration_axis.plot(
                tw,
                acceleration[:, index + 3],
                color="#6a1b9a",
                label="MPC acceleration",
            )
        velocity_axis.set_ylabel(f"{name}\nvelocity")
        acceleration_axis.set_ylabel(f"{name}\nacceleration")
        for axis in (velocity_axis, acceleration_axis):
            axis.grid(True, alpha=0.3)
            axis.set_xlim(*window)
            _legend(axis)
    axes[-1, 0].set_xlabel("time since first logged sample [s]")
    axes[-1, 1].set_xlabel("time since first logged sample [s]")
    fig.suptitle("Every WB-MPC model joint: command versus measured state", fontsize=15)
    _save(fig, output, dpi)


def _plot_streaming_positions(adapter, origin, window, output, dpi):
    tc, commanded = _series(adapter, "streaming_qpos", 10)
    tm, measured = _series(adapter, "measured_streaming_qpos", 10)
    if commanded is None:
        raise ValueError("adapter log has no streaming_qpos command records")
    tc = _relative(tc, origin)
    tm = _relative(tm, origin)

    fig, axes = plt.subplots(
        4, 2, figsize=(16, 14), sharex=True, constrained_layout=True
    )
    for index, axis in enumerate(axes.flat):
        name = SG3_QPOS_NAMES[index]
        if measured is not None:
            axis.plot(tm, measured[:, index], label="measured position")
        axis.plot(tc, commanded[:, index], "--", label="streaming position command")
        axis.set_ylabel(name)
        axis.grid(True, alpha=0.3)
        axis.set_xlim(*window)
        _legend(axis)
    axes[-1, 0].set_xlabel("time since first logged sample [s]")
    axes[-1, 1].set_xlabel("time since first logged sample [s]")
    fig.suptitle(
        "Every physical SG3 joint: streaming-position command versus state",
        fontsize=15,
    )
    _save(fig, output, dpi)


def plot(args):
    wbmpc = _load_jsonl(args.wbmpc_log)
    adapter = _load_jsonl(args.adapter_log)
    timestamps = [
        float(record["monotonic_time"])
        for record in wbmpc + adapter
        if record.get("monotonic_time") is not None
    ]
    if not timestamps:
        raise ValueError("logs contain no monotonic_time records")
    origin = min(timestamps)
    adapter_timestamps = [
        float(record["monotonic_time"])
        for record in adapter
        if record.get("monotonic_time") is not None
    ]
    if not adapter_timestamps:
        raise ValueError("adapter log contains no monotonic_time records")
    window = (
        min(adapter_timestamps) - origin,
        max(adapter_timestamps) - origin,
    )
    outputs = {
        "base": args.output_dir / "base_command_state.png",
        "localization": args.output_dir / "base_localization_state.png",
        "model_joints": args.output_dir / "model_joint_command_state.png",
        "streaming_positions": args.output_dir / "streaming_position_command_state.png",
    }
    _plot_base(wbmpc, adapter, origin, window, outputs["base"], args.dpi)
    _plot_localization(adapter, origin, window, outputs["localization"], args.dpi)
    _plot_model_joints(
        wbmpc, adapter, origin, window, outputs["model_joints"], args.dpi
    )
    _plot_streaming_positions(
        adapter, origin, window, outputs["streaming_positions"], args.dpi
    )
    return {name: str(path.resolve()) for name, path in outputs.items()}


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
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/diagnostics/command_state"),
    )
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args(argv)


def main(argv=None):
    print(json.dumps(plot(_parse_args(argv)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
