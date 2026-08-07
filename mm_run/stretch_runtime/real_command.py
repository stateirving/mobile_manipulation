"""Pure-Python safety mapping from WB-MPC velocity to Stretch commands."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from stretch_runtime.streaming_position import SG3_QPOS_NAMES

WB_MPC_VELOCITY_SIZE = 11

DRIVER_CHANNEL_NAMES = (
    "base_forward",
    "base_yaw",
    "joint_lift",
    "wrist_extension",
    "joint_wrist_yaw",
    "joint_wrist_pitch",
    "joint_wrist_roll",
)

POSITION_CHANNEL_NAMES = DRIVER_CHANNEL_NAMES[2:]

_POSITION_TO_SG3_INDEX = {
    "wrist_extension": SG3_QPOS_NAMES.index("wrist_extension"),
    "joint_lift": SG3_QPOS_NAMES.index("joint_lift"),
    "joint_wrist_yaw": SG3_QPOS_NAMES.index("joint_wrist_yaw"),
    "joint_wrist_pitch": SG3_QPOS_NAMES.index("joint_wrist_pitch"),
    "joint_wrist_roll": SG3_QPOS_NAMES.index("joint_wrist_roll"),
}


class CommandSafetyError(ValueError):
    """A command cannot be mapped safely to the deployed Stretch interface."""


def command_owner_errors(
    publisher_counts: Mapping[str, int], expected_count: int
) -> tuple[str, ...]:
    """Report command topics whose publisher count violates ownership."""

    if expected_count not in (0, 1):
        raise ValueError("expected command publisher count must be zero or one")
    errors = []
    for topic, count in publisher_counts.items():
        count = int(count)
        if count != expected_count:
            owner = "no publisher" if expected_count == 0 else "one WB-MPC owner"
            errors.append(
                f"command topic {topic} requires {owner}, found {count} publishers"
            )
    return tuple(errors)


@dataclass(frozen=True)
class AxisLimits:
    """Effective limits for one physical driver command channel."""

    position_lower: float | None = None
    position_upper: float | None = None
    velocity: float | None = None
    acceleration: float | None = None

    @classmethod
    def from_dict(cls, data: Mapping) -> "AxisLimits":
        position = data.get("position")
        if position is None:
            lower = upper = None
        else:
            if len(position) != 2:
                raise CommandSafetyError("position limit must contain [lower, upper]")
            lower, upper = (float(value) for value in position)
        return cls(
            position_lower=lower,
            position_upper=upper,
            velocity=(
                None if data.get("velocity") is None else float(data["velocity"])
            ),
            acceleration=(
                None
                if data.get("acceleration") is None
                else float(data["acceleration"])
            ),
        )


@dataclass(frozen=True)
class CommandCoreConfig:
    """Validated configuration for :class:`StretchCommandCore`."""

    limits: Mapping[str, AxisLimits]
    position_margin: Mapping[str, float]
    soft_following_error: Mapping[str, float]
    hard_following_error: Mapping[str, float]
    lateral_warn_threshold: float = 0.005
    lateral_stop_threshold: float = 0.02
    arm_projection_warn_threshold: float = 0.01
    arm_projection_stop_threshold: float = 0.01
    min_dt: float = 0.005
    max_dt: float = 0.25

    @classmethod
    def from_dict(cls, data: Mapping) -> "CommandCoreConfig":
        limits_config = data.get("limits", {})
        layers = limits_config.get("layers", {})
        margins = {
            str(name): float(value)
            for name, value in limits_config.get("position_margin", {}).items()
        }
        effective_limits = merge_limit_layers(layers, margins)
        following = data.get("following_error", {})
        soft = {
            str(name): float(value) for name, value in following.get("soft", {}).items()
        }
        hard = {
            str(name): float(value) for name, value in following.get("hard", {}).items()
        }
        projection_stop = float(data.get("arm_projection_stop_threshold", 0.01))
        config = cls(
            limits=effective_limits,
            position_margin=margins,
            soft_following_error=soft,
            hard_following_error=hard,
            lateral_warn_threshold=float(data.get("lateral_warn_threshold", 0.005)),
            lateral_stop_threshold=float(data.get("lateral_stop_threshold", 0.02)),
            arm_projection_warn_threshold=float(
                data.get("arm_projection_warn_threshold", min(0.01, projection_stop))
            ),
            arm_projection_stop_threshold=projection_stop,
            min_dt=float(data.get("min_dt", 0.005)),
            max_dt=float(data.get("max_dt", 0.25)),
        )
        _validate_core_config(config)
        return config


@dataclass(frozen=True)
class SafeCommand:
    """A complete command that is feasible for the real Stretch interface."""

    base_linear_x: float
    base_angular_z: float
    streaming_qpos: np.ndarray
    requested_driver_velocity: np.ndarray
    safe_driver_velocity: np.ndarray
    realized_model_velocity: np.ndarray
    lateral_velocity: float
    arm_projection_residual: float
    clipped_channels: tuple[str, ...]


def merge_limit_layers(
    layers: Mapping[str, Mapping], position_margins: Mapping[str, float] | None = None
) -> dict[str, AxisLimits]:
    """Intersect model, driver and commissioning limits by physical channel."""

    if not layers:
        raise CommandSafetyError("at least one command limit layer is required")
    margins = {} if position_margins is None else position_margins
    effective: dict[str, AxisLimits] = {}
    for channel in DRIVER_CHANNEL_NAMES:
        lower_values = []
        upper_values = []
        velocity_values = []
        acceleration_values = []
        for layer_name, layer in layers.items():
            channel_data = layer.get(channel)
            if channel_data is None:
                continue
            parsed = AxisLimits.from_dict(channel_data)
            for label, value in (
                ("position lower", parsed.position_lower),
                ("position upper", parsed.position_upper),
                ("velocity", parsed.velocity),
                ("acceleration", parsed.acceleration),
            ):
                if value is not None and not math.isfinite(value):
                    raise CommandSafetyError(
                        f"{layer_name}.{channel} {label} is not finite"
                    )
            if parsed.position_lower is not None:
                lower_values.append(parsed.position_lower)
                upper_values.append(parsed.position_upper)
            if parsed.velocity is not None:
                if parsed.velocity <= 0.0:
                    raise CommandSafetyError(
                        f"{layer_name}.{channel} velocity must be positive"
                    )
                velocity_values.append(parsed.velocity)
            if parsed.acceleration is not None:
                if parsed.acceleration <= 0.0:
                    raise CommandSafetyError(
                        f"{layer_name}.{channel} acceleration must be positive"
                    )
                acceleration_values.append(parsed.acceleration)

        if not velocity_values or not acceleration_values:
            raise CommandSafetyError(
                f"{channel} needs at least one velocity and acceleration limit"
            )
        lower = max(lower_values) if lower_values else None
        upper = min(upper_values) if upper_values else None
        if channel in POSITION_CHANNEL_NAMES:
            if lower is None or upper is None:
                raise CommandSafetyError(f"{channel} needs position limits")
            margin = float(margins.get(channel, 0.0))
            if not math.isfinite(margin) or margin < 0.0:
                raise CommandSafetyError(
                    f"{channel} position margin must be finite and nonnegative"
                )
            lower += margin
            upper -= margin
            if lower >= upper:
                raise CommandSafetyError(
                    f"{channel} position limit intersection is empty"
                )
        effective[channel] = AxisLimits(
            position_lower=lower,
            position_upper=upper,
            velocity=min(velocity_values),
            acceleration=min(acceleration_values),
        )
    return effective


def world_to_body_nonholonomic(
    velocity_world: Sequence[float], yaw: float
) -> tuple[float, float, float]:
    """Project a planar world-frame velocity into Stretch body coordinates."""

    velocity = np.asarray(velocity_world, dtype=float).reshape(-1)
    if velocity.size != 3 or not np.all(np.isfinite(velocity)):
        raise CommandSafetyError("world base velocity must contain three finite values")
    if not math.isfinite(yaw):
        raise CommandSafetyError("base yaw is not finite")
    c = math.cos(yaw)
    s = math.sin(yaw)
    forward = c * velocity[0] + s * velocity[1]
    lateral = -s * velocity[0] + c * velocity[1]
    return float(forward), float(lateral), float(velocity[2])


class StretchCommandCore:
    """Stateful velocity shaper and position target generator without ROS."""

    def __init__(self, config: CommandCoreConfig):
        self.config = config
        self._target_qpos: np.ndarray | None = None
        self._previous_driver_velocity = np.zeros(len(DRIVER_CHANNEL_NAMES))

    def reset(self, measured_qpos: Sequence[float]) -> None:
        """Anchor all absolute targets to fresh measured driver positions."""

        qpos = _validated_qpos(measured_qpos)
        for channel in POSITION_CHANNEL_NAMES:
            index = _POSITION_TO_SG3_INDEX[channel]
            limit = self.config.limits[channel]
            value = float(qpos[index])
            if not limit.position_lower <= value <= limit.position_upper:
                raise CommandSafetyError(
                    f"measured {channel}={value:.6f} is outside effective position "
                    f"limits [{limit.position_lower:.6f}, {limit.position_upper:.6f}]"
                )
        qpos[-2:] = 0.0
        self._target_qpos = qpos
        self._previous_driver_velocity = np.zeros(len(DRIVER_CHANNEL_NAMES))

    def step(
        self,
        *,
        yaw: float,
        measured_qpos: Sequence[float],
        requested_velocity_world: Sequence[float],
        dt: float,
        enforce_tracking: bool = True,
    ) -> SafeCommand:
        """Return a limited Twist and full SG3 qpos for one control cycle."""

        measured = _validated_qpos(measured_qpos)
        requested = np.asarray(requested_velocity_world, dtype=float).reshape(-1)
        if requested.size != WB_MPC_VELOCITY_SIZE:
            raise CommandSafetyError(
                f"WB-MPC velocity must have length {WB_MPC_VELOCITY_SIZE}"
            )
        if not np.all(np.isfinite(requested)):
            raise CommandSafetyError("WB-MPC velocity contains NaN or Inf")
        dt = float(dt)
        if not math.isfinite(dt) or not self.config.min_dt <= dt <= self.config.max_dt:
            raise CommandSafetyError(
                f"command dt={dt!r} is outside "
                f"[{self.config.min_dt}, {self.config.max_dt}]"
            )
        if self._target_qpos is None:
            self.reset(measured)

        forward, lateral, yaw_rate = world_to_body_nonholonomic(requested[:3], yaw)
        if abs(lateral) > self.config.lateral_stop_threshold:
            raise CommandSafetyError(
                f"body lateral command {lateral:.6f} m/s exceeds stop threshold "
                f"{self.config.lateral_stop_threshold:.6f} m/s"
            )

        segment_velocity = requested[4:8]
        extension_velocity = float(np.sum(segment_velocity))
        equal_segment_velocity = extension_velocity / 4.0
        projection_residual = float(
            np.max(np.abs(segment_velocity - equal_segment_velocity))
        )
        if projection_residual > self.config.arm_projection_stop_threshold:
            raise CommandSafetyError(
                f"arm velocity projection residual {projection_residual:.6f} m/s "
                f"exceeds stop threshold "
                f"{self.config.arm_projection_stop_threshold:.6f} m/s"
            )

        driver_requested = np.array(
            [
                forward,
                yaw_rate,
                requested[3],
                extension_velocity,
                requested[8],
                requested[9],
                requested[10],
            ],
            dtype=float,
        )
        safe_velocity = np.empty_like(driver_requested)
        clipped = []
        for index, channel in enumerate(DRIVER_CHANNEL_NAMES):
            limit = self.config.limits[channel]
            velocity_limited = float(
                np.clip(driver_requested[index], -limit.velocity, limit.velocity)
            )
            max_delta = limit.acceleration * dt
            acceleration_limited = float(
                np.clip(
                    velocity_limited,
                    self._previous_driver_velocity[index] - max_delta,
                    self._previous_driver_velocity[index] + max_delta,
                )
            )
            safe_velocity[index] = acceleration_limited
            if not math.isclose(
                acceleration_limited,
                driver_requested[index],
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ):
                clipped.append(channel)

        target = self._target_qpos.copy()
        for driver_index, channel in enumerate(DRIVER_CHANNEL_NAMES[2:], start=2):
            qpos_index = _POSITION_TO_SG3_INDEX[channel]
            error = float(target[qpos_index] - measured[qpos_index])
            soft_error = self.config.soft_following_error[channel]
            hard_error = self.config.hard_following_error[channel]
            if enforce_tracking and abs(error) > hard_error:
                raise CommandSafetyError(
                    f"{channel} following error {error:.6f} exceeds "
                    f"{hard_error:.6f}"
                )
            channel_velocity = float(safe_velocity[driver_index])
            if (
                enforce_tracking
                and abs(error) > soft_error
                and channel_velocity * error > 0.0
            ):
                channel_velocity = 0.0
                safe_velocity[driver_index] = 0.0
                if channel not in clipped:
                    clipped.append(channel)

            limit = self.config.limits[channel]
            candidate = target[qpos_index] + channel_velocity * dt
            limited_candidate = float(
                np.clip(candidate, limit.position_lower, limit.position_upper)
            )
            if not math.isclose(
                limited_candidate, candidate, rel_tol=1.0e-12, abs_tol=1.0e-12
            ):
                safe_velocity[driver_index] = (
                    limited_candidate - target[qpos_index]
                ) / dt
                if channel not in clipped:
                    clipped.append(channel)
            target[qpos_index] = limited_candidate

        # Head and gripper are not controlled by WB-MPC. Always hold their latest
        # measured positions; the two base-increment entries remain unused because
        # the base is owned by /stretch/cmd_vel.
        for name in (
            "joint_head_pan",
            "joint_head_tilt",
            "joint_gripper_finger_left",
        ):
            index = SG3_QPOS_NAMES.index(name)
            target[index] = measured[index]
        target[-2:] = 0.0

        safe_forward = float(safe_velocity[0])
        safe_yaw_rate = float(safe_velocity[1])
        c = math.cos(yaw)
        s = math.sin(yaw)
        realized_model_velocity = np.array(
            [
                c * safe_forward,
                s * safe_forward,
                safe_yaw_rate,
                safe_velocity[2],
                safe_velocity[3] / 4.0,
                safe_velocity[3] / 4.0,
                safe_velocity[3] / 4.0,
                safe_velocity[3] / 4.0,
                safe_velocity[4],
                safe_velocity[5],
                safe_velocity[6],
            ]
        )
        self._target_qpos = target
        self._previous_driver_velocity = safe_velocity.copy()
        return SafeCommand(
            base_linear_x=safe_forward,
            base_angular_z=safe_yaw_rate,
            streaming_qpos=target.copy(),
            requested_driver_velocity=driver_requested,
            safe_driver_velocity=safe_velocity.copy(),
            realized_model_velocity=realized_model_velocity,
            lateral_velocity=lateral,
            arm_projection_residual=projection_residual,
            clipped_channels=tuple(clipped),
        )


def _validated_qpos(values: Sequence[float]) -> np.ndarray:
    qpos = np.asarray(values, dtype=float).reshape(-1).copy()
    if qpos.size != len(SG3_QPOS_NAMES):
        raise CommandSafetyError(f"SG3 qpos must have length {len(SG3_QPOS_NAMES)}")
    if not np.all(np.isfinite(qpos)):
        raise CommandSafetyError("SG3 qpos contains NaN or Inf")
    return qpos


def _validate_core_config(config: CommandCoreConfig) -> None:
    if not 0.0 <= config.lateral_warn_threshold <= config.lateral_stop_threshold:
        raise CommandSafetyError("lateral thresholds are inconsistent")
    if not (
        0.0
        <= config.arm_projection_warn_threshold
        <= config.arm_projection_stop_threshold
    ):
        raise CommandSafetyError("arm projection thresholds are inconsistent")
    if not 0.0 < config.min_dt <= config.max_dt:
        raise CommandSafetyError("command dt limits are inconsistent")
    for channel in POSITION_CHANNEL_NAMES:
        if channel not in config.soft_following_error:
            raise CommandSafetyError(f"missing soft following error for {channel}")
        if channel not in config.hard_following_error:
            raise CommandSafetyError(f"missing hard following error for {channel}")
        soft = config.soft_following_error[channel]
        hard = config.hard_following_error[channel]
        if not math.isfinite(soft) or not math.isfinite(hard):
            raise CommandSafetyError(f"non-finite following error for {channel}")
        if not 0.0 < soft < hard:
            raise CommandSafetyError(
                f"following errors for {channel} must satisfy 0 < soft < hard"
            )
