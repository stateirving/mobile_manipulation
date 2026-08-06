"""Validation and composition of the complete real Stretch WB-MPC state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from stretch_runtime.real_base_state import MappedBaseState
from stretch_runtime.real_state import (
    DEFAULT_ARM_JOINT_NAMES,
    MappedJointState,
    StateValidationError,
)

BASE_STATE_NAMES = (
    "x_to_world_joint",
    "y_to_x_joint",
    "base_to_y_joint",
)
FULL_STATE_NAMES = BASE_STATE_NAMES + DEFAULT_ARM_JOINT_NAMES


@dataclass(frozen=True)
class FullStateConfig:
    """Synchronization and freshness rules for the complete state."""

    max_state_skew: float = 0.01
    max_age: float = 0.25
    future_tolerance: float = 0.05

    @classmethod
    def from_dict(cls, data: Mapping) -> "FullStateConfig":
        return cls(
            max_state_skew=float(data.get("max_state_skew", 0.01)),
            max_age=float(data.get("max_age", 0.25)),
            future_tolerance=float(data.get("future_tolerance", 0.05)),
        )


@dataclass(frozen=True)
class MappedFullState:
    """Complete 11-position/11-velocity WB-MPC feedback state."""

    stamp: float
    oldest_stamp: float
    base_stamp: float
    joint_stamp: float
    skew: float
    age: float | None
    names: tuple[str, ...]
    position: np.ndarray
    velocity: np.ndarray
    joint_velocity_source: str


class StretchFullStateCombiner:
    """Combine validated base and arm states without changing their values."""

    def __init__(self, config: FullStateConfig):
        self.config = config

    def combine(
        self,
        base: MappedBaseState,
        joints: MappedJointState,
        *,
        now: float | None = None,
    ) -> MappedFullState:
        errors = []
        base_position = np.asarray(base.position, dtype=float).reshape(-1)
        base_velocity = np.asarray(base.velocity_world, dtype=float).reshape(-1)
        joint_position = np.asarray(joints.position, dtype=float).reshape(-1)
        joint_velocity = np.asarray(joints.velocity, dtype=float).reshape(-1)

        for label, values, expected in (
            ("base position", base_position, 3),
            ("base velocity", base_velocity, 3),
            ("joint position", joint_position, len(DEFAULT_ARM_JOINT_NAMES)),
            ("joint velocity", joint_velocity, len(DEFAULT_ARM_JOINT_NAMES)),
        ):
            if values.size != expected:
                errors.append(f"{label} must have length {expected}, got {values.size}")
            elif not np.all(np.isfinite(values)):
                errors.append(f"{label} contains NaN or Inf")

        base_stamp = float(base.odom_stamp)
        joint_stamp = float(joints.stamp)
        if not np.isfinite(base_stamp) or not np.isfinite(joint_stamp):
            errors.append("base/joint timestamp contains NaN or Inf")
        skew = abs(base_stamp - joint_stamp)
        if skew > self.config.max_state_skew:
            errors.append(
                f"base/joint timestamp skew={skew:.6f}s > "
                f"{self.config.max_state_skew:.6f}s"
            )

        oldest_stamp = min(base_stamp, joint_stamp)
        stamp = max(base_stamp, joint_stamp)
        age = None if now is None else float(now - oldest_stamp)
        if age is not None:
            if not np.isfinite(age):
                errors.append(f"non-finite full-state age: {age}")
            elif age > self.config.max_age:
                errors.append(
                    f"stale full state: age={age:.6f}s > " f"{self.config.max_age:.6f}s"
                )
            elif age < -self.config.future_tolerance:
                errors.append(
                    f"full-state timestamp is in the future: age={age:.6f}s < "
                    f"-{self.config.future_tolerance:.6f}s"
                )

        if errors:
            raise StateValidationError(errors)

        position = np.concatenate((base_position, joint_position))
        velocity = np.concatenate((base_velocity, joint_velocity))
        return MappedFullState(
            stamp=stamp,
            oldest_stamp=oldest_stamp,
            base_stamp=base_stamp,
            joint_stamp=joint_stamp,
            skew=skew,
            age=age,
            names=FULL_STATE_NAMES,
            position=position,
            velocity=velocity,
            joint_velocity_source=joints.velocity_source,
        )
