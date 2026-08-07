"""Pure helpers for driving a WB-MPC controller from validated real state."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from stretch_runtime.real_full_state import FULL_STATE_NAMES


@dataclass(frozen=True)
class WBMPCState:
    """Validated 11-position/11-velocity controller feedback."""

    position: np.ndarray
    velocity: np.ndarray


def validate_wbmpc_state(
    names: Sequence[str],
    position: Sequence[float],
    velocity: Sequence[float],
) -> WBMPCState:
    """Validate and reorder a named state into the controller contract."""

    names = tuple(str(name) for name in names)
    if len(names) != len(set(names)):
        raise ValueError("WB-MPC state contains duplicate names")
    missing = [name for name in FULL_STATE_NAMES if name not in names]
    extra = [name for name in names if name not in FULL_STATE_NAMES]
    if missing or extra:
        raise ValueError(
            f"WB-MPC state names mismatch: missing={missing}, extra={extra}"
        )
    positions = np.asarray(position, dtype=float).reshape(-1)
    velocities = np.asarray(velocity, dtype=float).reshape(-1)
    if positions.size != len(names) or velocities.size != len(names):
        raise ValueError("WB-MPC state position/velocity lengths do not match names")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
        raise ValueError("WB-MPC state contains NaN or Inf")
    indices = [names.index(name) for name in FULL_STATE_NAMES]
    return WBMPCState(
        position=positions[indices].copy(),
        velocity=velocities[indices].copy(),
    )


def sample_acceleration_plan(
    plan: Sequence[Sequence[float]], elapsed: float, plan_dt: float
) -> np.ndarray:
    """Linearly sample an MPC acceleration plan, returning zero after its end."""

    values = np.asarray(plan, dtype=float)
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError("acceleration plan must be a non-empty matrix")
    if not np.all(np.isfinite(values)):
        raise ValueError("acceleration plan contains NaN or Inf")
    elapsed = float(elapsed)
    plan_dt = float(plan_dt)
    if not math.isfinite(elapsed) or not math.isfinite(plan_dt) or plan_dt <= 0.0:
        raise ValueError("plan sampling time must be finite and plan_dt positive")
    if elapsed < 0.0:
        return values[0].copy()
    if elapsed > (values.shape[0] - 1) * plan_dt:
        return np.zeros(values.shape[1], dtype=float)
    scaled = elapsed / plan_dt
    lower = int(math.floor(scaled))
    if lower >= values.shape[0]:
        return np.zeros(values.shape[1], dtype=float)
    upper = lower + 1
    if upper >= values.shape[0]:
        return values[lower].copy()
    alpha = scaled - lower
    return (1.0 - alpha) * values[lower] + alpha * values[upper]


def integrate_acceleration_velocity(
    previous_velocity: Sequence[float],
    acceleration: Sequence[float],
    dt: float,
    position: Sequence[float],
    measured_velocity: Sequence[float],
    *,
    nonholonomic: bool,
) -> np.ndarray:
    """Integrate acceleration while anchoring nonholonomic base speed to feedback."""

    command = np.asarray(previous_velocity, dtype=float).reshape(-1).copy()
    acceleration = np.asarray(acceleration, dtype=float).reshape(-1)
    position = np.asarray(position, dtype=float).reshape(-1)
    measured = np.asarray(measured_velocity, dtype=float).reshape(-1)
    dt = float(dt)
    if command.size != acceleration.size or command.size != measured.size:
        raise ValueError("velocity, acceleration and feedback dimensions must match")
    if position.size != command.size:
        raise ValueError("position and velocity dimensions must match")
    if (
        not math.isfinite(dt)
        or dt <= 0.0
        or not np.all(np.isfinite(command))
        or not np.all(np.isfinite(acceleration))
        or not np.all(np.isfinite(position))
        or not np.all(np.isfinite(measured))
    ):
        raise ValueError("velocity integration inputs must be finite and dt positive")
    command += acceleration * dt
    if not nonholonomic:
        return command

    yaw = float(position[2])
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    measured_forward = cosine * measured[0] + sine * measured[1]
    forward_acceleration = cosine * acceleration[0] + sine * acceleration[1]
    forward = measured_forward + forward_acceleration * dt
    command[0] = cosine * forward
    command[1] = sine * forward
    command[2] = measured[2] + acceleration[2] * dt
    return command


def controller_velocity_limits(
    controller_config: Mapping, expected_dimension: int
) -> tuple[np.ndarray, np.ndarray]:
    """Extract velocity bounds from the controller state-limit vector."""

    robot = controller_config["robot"]
    nq = int(robot["dims"]["q"])
    state_limits = robot["limits"]["state"]
    lower = np.asarray(
        [_parse_numeric_limit(value) for value in state_limits["lower"]],
        dtype=float,
    )[nq:]
    upper = np.asarray(
        [_parse_numeric_limit(value) for value in state_limits["upper"]],
        dtype=float,
    )[nq:]
    if lower.size != expected_dimension or upper.size != expected_dimension:
        raise ValueError("controller velocity limit dimension mismatch")
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError("controller velocity limits contain NaN or Inf")
    if np.any(lower >= upper):
        raise ValueError("controller velocity limit interval is empty")
    return lower, upper


def _parse_numeric_limit(value) -> float:
    if not isinstance(value, str):
        return float(value)
    text = value.strip()
    if text in ("pi", "+pi"):
        return math.pi
    if text == "-pi":
        return -math.pi
    match = re.fullmatch(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\.pi", text)
    if match is not None:
        return float(match.group(1)) * math.pi
    return float(text)
