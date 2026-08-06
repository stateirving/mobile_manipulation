"""Stretch streaming-position command contract and safety helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

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


@dataclass(frozen=True)
class StreamingTestLimits:
    """Hard safety bounds for the dedicated physical test helper."""

    max_wrist_yaw_delta: float = 0.03
    max_base_speed: float = 0.015
    max_base_motion_duration: float = 1.5
    max_publish_rate: float = 15.0


def sg3_qpos_from_joint_state(
    names: Sequence[str], positions: Sequence[float]
) -> np.ndarray:
    """Build the official 10-element SG3 streaming qpos vector."""

    if len(names) != len(positions):
        raise ValueError("JointState name and position lengths differ")
    values: Mapping[str, float] = dict(zip(names, positions))
    required = SG3_QPOS_NAMES[:8]
    missing = [name for name in required if name not in values]
    if missing:
        raise ValueError(f"JointState is missing streaming joints: {missing}")
    qpos = np.array([float(values[name]) for name in required] + [0.0, 0.0])
    if not np.all(np.isfinite(qpos)):
        raise ValueError("Streaming qpos contains non-finite values")
    return qpos


def smoothstep(fraction: float) -> float:
    """C1-continuous interpolation fraction clamped to [0, 1]."""

    value = min(max(float(fraction), 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)


def validate_streaming_test_request(
    yaw_delta: float,
    publish_rate: float,
    base_speed: float,
    base_motion_duration: float,
    limits: StreamingTestLimits = StreamingTestLimits(),
) -> None:
    """Reject physical-test inputs outside deliberately narrow bounds."""

    values = (yaw_delta, publish_rate, base_speed, base_motion_duration)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Streaming test inputs must be finite")
    if yaw_delta <= 0.0 or yaw_delta > limits.max_wrist_yaw_delta:
        raise ValueError(f"yaw_delta must be in (0, {limits.max_wrist_yaw_delta}]")
    if publish_rate <= 0.0 or publish_rate > limits.max_publish_rate:
        raise ValueError(f"publish_rate must be in (0, {limits.max_publish_rate}]")
    if abs(base_speed) > limits.max_base_speed:
        raise ValueError(f"abs(base_speed) must be <= {limits.max_base_speed}")
    if base_speed != 0.0 and not (
        0.0 < base_motion_duration <= limits.max_base_motion_duration
    ):
        raise ValueError(
            "base_motion_duration must be positive and <= "
            f"{limits.max_base_motion_duration} when base motion is enabled"
        )
