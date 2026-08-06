"""Pure-Python joint-state validation and mapping for the real Stretch robot."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

DEFAULT_ARM_JOINT_NAMES = (
    "joint_lift",
    "joint_arm_l3",
    "joint_arm_l2",
    "joint_arm_l1",
    "joint_arm_l0",
    "joint_wrist_yaw",
    "joint_wrist_pitch",
    "joint_wrist_roll",
)

DEFAULT_EXTENSION_MEMBERS = (
    "joint_arm_l3",
    "joint_arm_l2",
    "joint_arm_l1",
    "joint_arm_l0",
)


class StateValidationError(ValueError):
    """Raised when a feedback sample cannot safely enter the controller state."""

    def __init__(self, reasons: Sequence[str]):
        self.reasons = tuple(reasons)
        super().__init__("; ".join(self.reasons))


@dataclass(frozen=True)
class JointMappingConfig:
    """Validation rules for mapping a ROS JointState-like sample."""

    joint_names: tuple[str, ...] = DEFAULT_ARM_JOINT_NAMES
    aggregate_name: str = "wrist_extension"
    aggregate_members: tuple[str, ...] = DEFAULT_EXTENSION_MEMBERS
    aggregate_tolerance: float = 0.001
    position_limits: Mapping[str, tuple[float, float]] | None = None
    limit_tolerance: float = 1.0e-6
    joint_limit_tolerances: Mapping[str, float] | None = None
    max_age: float = 0.25
    future_tolerance: float = 0.05
    allow_velocity_fallback: bool = True
    velocity_filter_time_constant: float = 0.10
    max_differentiation_dt: float = 0.25

    @classmethod
    def from_dict(cls, data: Mapping) -> "JointMappingConfig":
        limits = {
            name: (float(bounds[0]), float(bounds[1]))
            for name, bounds in data.get("position_limits", {}).items()
        }
        joint_limit_tolerances = {
            name: float(tolerance)
            for name, tolerance in data.get("joint_limit_tolerances", {}).items()
        }
        return cls(
            joint_names=tuple(data.get("joint_names", DEFAULT_ARM_JOINT_NAMES)),
            aggregate_name=str(data.get("aggregate_name", "wrist_extension")),
            aggregate_members=tuple(
                data.get("aggregate_members", DEFAULT_EXTENSION_MEMBERS)
            ),
            aggregate_tolerance=float(data.get("aggregate_tolerance", 0.001)),
            position_limits=limits,
            limit_tolerance=float(data.get("limit_tolerance", 1.0e-6)),
            joint_limit_tolerances=joint_limit_tolerances,
            max_age=float(data.get("max_age", 0.25)),
            future_tolerance=float(data.get("future_tolerance", 0.05)),
            allow_velocity_fallback=bool(data.get("allow_velocity_fallback", True)),
            velocity_filter_time_constant=float(
                data.get("velocity_filter_time_constant", 0.10)
            ),
            max_differentiation_dt=float(data.get("max_differentiation_dt", 0.25)),
        )


@dataclass(frozen=True)
class MappedJointState:
    """Validated arm state in the exact order required by WB-MPC."""

    stamp: float
    position: np.ndarray
    velocity: np.ndarray
    velocity_source: str
    age: float | None
    aggregate_error: float


class StretchJointStateMapper:
    """Map name-indexed Stretch feedback without relying on message order."""

    def __init__(self, config: JointMappingConfig):
        self.config = config
        self._previous_stamp: float | None = None
        self._previous_position: np.ndarray | None = None
        self._previous_velocity: np.ndarray | None = None

    def reset(self) -> None:
        """Clear differentiation and timestamp history."""

        self._previous_stamp = None
        self._previous_position = None
        self._previous_velocity = None

    def map(
        self,
        names: Sequence[str],
        positions: Sequence[float],
        velocities: Sequence[float],
        stamp: float,
        now: float | None = None,
    ) -> MappedJointState:
        """Validate and map one JointState-like sample.

        Args:
            names: Message joint names.
            positions: Positions parallel to ``names``.
            velocities: Velocities parallel to ``names`` or an empty sequence.
            stamp: Message timestamp in seconds.
            now: Current ROS time in seconds, used for the age gate.
        """

        errors: list[str] = []
        names = tuple(str(name) for name in names)
        positions_array = np.asarray(positions, dtype=float).reshape(-1)
        velocities_array = np.asarray(velocities, dtype=float).reshape(-1)

        if len(names) != positions_array.size:
            errors.append(
                "name/position length mismatch: "
                f"{len(names)} names, {positions_array.size} positions"
            )
        duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
        if duplicates:
            errors.append(f"duplicate joint names: {duplicates}")
        if errors:
            raise StateValidationError(errors)

        index = {name: i for i, name in enumerate(names)}
        required = set(self.config.joint_names)
        required.add(self.config.aggregate_name)
        missing = sorted(required - index.keys())
        if missing:
            errors.append(f"missing required joints: {missing}")

        if not np.isfinite(stamp):
            errors.append(f"non-finite timestamp: {stamp}")
        elif self._previous_stamp is not None and stamp <= self._previous_stamp:
            errors.append(
                "non-monotonic timestamp: "
                f"current={stamp:.9f}, previous={self._previous_stamp:.9f}"
            )

        age = None if now is None else float(now - stamp)
        if age is not None:
            if not np.isfinite(age):
                errors.append(f"non-finite sample age: {age}")
            elif age > self.config.max_age:
                errors.append(
                    f"stale JointState: age={age:.6f}s > {self.config.max_age:.6f}s"
                )
            elif age < -self.config.future_tolerance:
                errors.append(
                    "JointState timestamp is in the future: "
                    f"age={age:.6f}s < -{self.config.future_tolerance:.6f}s"
                )

        if errors:
            raise StateValidationError(errors)

        mapped_position = np.array(
            [positions_array[index[name]] for name in self.config.joint_names],
            dtype=float,
        )
        aggregate_position = float(positions_array[index[self.config.aggregate_name]])
        if not np.all(np.isfinite(mapped_position)):
            errors.append("required joint positions contain NaN or Inf")
        if not np.isfinite(aggregate_position):
            errors.append(
                f"aggregate joint {self.config.aggregate_name!r} contains NaN or Inf"
            )

        member_indices = [
            self.config.joint_names.index(name)
            for name in self.config.aggregate_members
        ]
        aggregate_error = float(
            aggregate_position - np.sum(mapped_position[member_indices])
        )
        if abs(aggregate_error) > self.config.aggregate_tolerance:
            errors.append(
                f"{self.config.aggregate_name} mismatch: "
                f"error={aggregate_error:.9f}m, "
                f"tolerance={self.config.aggregate_tolerance:.9f}m"
            )

        limits = self.config.position_limits or {}
        joint_tolerances = self.config.joint_limit_tolerances or {}
        for joint_index, name in enumerate(self.config.joint_names):
            if name not in limits:
                continue
            lower, upper = limits[name]
            value = mapped_position[joint_index]
            tolerance = joint_tolerances.get(name, self.config.limit_tolerance)
            if value < lower - tolerance or value > upper + tolerance:
                errors.append(
                    f"joint limit violation for {name}: "
                    f"{value:.9f} not in [{lower:.9f}, {upper:.9f}]"
                )

        mapped_velocity, velocity_source = self._map_velocity(
            index,
            velocities_array,
            mapped_position,
            stamp,
            errors,
        )
        if errors:
            raise StateValidationError(errors)

        self._previous_stamp = float(stamp)
        self._previous_position = mapped_position.copy()
        self._previous_velocity = mapped_velocity.copy()
        return MappedJointState(
            stamp=float(stamp),
            position=mapped_position,
            velocity=mapped_velocity,
            velocity_source=velocity_source,
            age=age,
            aggregate_error=aggregate_error,
        )

    def _map_velocity(
        self,
        index: Mapping[str, int],
        velocities: np.ndarray,
        position: np.ndarray,
        stamp: float,
        errors: list[str],
    ) -> tuple[np.ndarray, str]:
        if velocities.size:
            if velocities.size != len(index):
                errors.append(
                    "name/velocity length mismatch: "
                    f"{len(index)} names, {velocities.size} velocities"
                )
                return np.zeros(len(self.config.joint_names)), "invalid"
            mapped = np.array(
                [velocities[index[name]] for name in self.config.joint_names],
                dtype=float,
            )
            if not np.all(np.isfinite(mapped)):
                errors.append("required joint velocities contain NaN or Inf")
            return mapped, "measured"

        if not self.config.allow_velocity_fallback:
            errors.append("JointState velocity is empty and fallback is disabled")
            return np.zeros(len(self.config.joint_names)), "invalid"
        if self._previous_stamp is None or self._previous_position is None:
            errors.append("velocity differentiation history is not ready")
            return np.zeros(len(self.config.joint_names)), "invalid"

        dt = stamp - self._previous_stamp
        if dt <= 0.0 or dt > self.config.max_differentiation_dt:
            errors.append(
                "invalid velocity differentiation interval: "
                f"dt={dt:.9f}s, max={self.config.max_differentiation_dt:.9f}s"
            )
            return np.zeros(len(self.config.joint_names)), "invalid"

        raw_velocity = (position - self._previous_position) / dt
        time_constant = max(0.0, self.config.velocity_filter_time_constant)
        alpha = 1.0 if time_constant == 0.0 else dt / (time_constant + dt)
        previous_velocity = (
            np.zeros_like(raw_velocity)
            if self._previous_velocity is None
            else self._previous_velocity
        )
        filtered_velocity = alpha * raw_velocity + (1.0 - alpha) * previous_velocity
        return filtered_velocity, "filtered_difference"
