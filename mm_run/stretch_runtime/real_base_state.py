"""Pure-Python base pose/velocity validation for the real Stretch robot."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from stretch_runtime.real_state import StateValidationError


@dataclass(frozen=True)
class BaseMappingConfig:
    """Frame and timing rules for constructing the WB-MPC base state."""

    pose_source: str = "map_tf"
    global_frame: str = "map"
    odom_frame: str = "odom"
    base_frame: str = "base_link"
    max_pose_age: float = 0.25
    max_odom_age: float = 0.25
    max_pose_odom_skew: float = 0.25
    future_tolerance: float = 0.05
    quaternion_norm_tolerance: float = 1.0e-3
    max_abs_body_lateral_velocity: float | None = 0.02
    odom_history_duration: float = 2.0
    max_anchor_odom_skew: float = 0.10
    max_anchor_translation_jump: float = 0.03
    max_anchor_yaw_jump: float = 0.05
    max_anchor_propagation_translation: float = 0.10
    max_anchor_propagation_yaw: float = 0.20

    @classmethod
    def from_dict(cls, data: Mapping) -> "BaseMappingConfig":
        lateral_limit = data.get("max_abs_body_lateral_velocity", 0.02)
        return cls(
            pose_source=str(data.get("pose_source", "map_tf")),
            global_frame=str(data.get("global_frame", "map")),
            odom_frame=str(data.get("odom_frame", "odom")),
            base_frame=str(data.get("base_frame", "base_link")),
            max_pose_age=float(data.get("max_pose_age", 0.25)),
            max_odom_age=float(data.get("max_odom_age", 0.25)),
            max_pose_odom_skew=float(data.get("max_pose_odom_skew", 0.25)),
            future_tolerance=float(data.get("future_tolerance", 0.05)),
            quaternion_norm_tolerance=float(
                data.get("quaternion_norm_tolerance", 1.0e-3)
            ),
            max_abs_body_lateral_velocity=(
                None if lateral_limit is None else float(lateral_limit)
            ),
            odom_history_duration=float(data.get("odom_history_duration", 2.0)),
            max_anchor_odom_skew=float(data.get("max_anchor_odom_skew", 0.10)),
            max_anchor_translation_jump=float(
                data.get("max_anchor_translation_jump", 0.03)
            ),
            max_anchor_yaw_jump=float(data.get("max_anchor_yaw_jump", 0.05)),
            max_anchor_propagation_translation=float(
                data.get("max_anchor_propagation_translation", 0.10)
            ),
            max_anchor_propagation_yaw=float(
                data.get("max_anchor_propagation_yaw", 0.20)
            ),
        )


@dataclass(frozen=True)
class MappedBaseState:
    """Validated planar base state with explicit body/world velocity."""

    pose_stamp: float
    odom_stamp: float
    position: np.ndarray
    velocity_world: np.ndarray
    velocity_body: np.ndarray
    pose_age: float | None
    odom_age: float | None
    pose_odom_skew: float
    pose_frame: str


@dataclass(frozen=True)
class PropagatedMapPose:
    """Map pose propagated from a sparse global anchor with local odometry."""

    position: np.ndarray
    stamp: float
    anchor_stamp: float
    anchor_age: float
    translation_since_anchor: float
    yaw_since_anchor: float
    last_translation_jump: float
    last_yaw_jump: float


def yaw_from_quaternion_xyzw(quaternion: Sequence[float]) -> float:
    """Return planar yaw from an xyzw quaternion."""

    x, y, z, w = (float(value) for value in quaternion)
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""

    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def compose_pose2(left: Sequence[float], right: Sequence[float]) -> np.ndarray:
    """Compose two planar poses represented as [x, y, yaw]."""

    left_pose = np.asarray(left, dtype=float)
    right_pose = np.asarray(right, dtype=float)
    cosine, sine = math.cos(left_pose[2]), math.sin(left_pose[2])
    return np.array(
        [
            left_pose[0] + cosine * right_pose[0] - sine * right_pose[1],
            left_pose[1] + sine * right_pose[0] + cosine * right_pose[1],
            wrap_angle(left_pose[2] + right_pose[2]),
        ]
    )


def inverse_pose2(pose: Sequence[float]) -> np.ndarray:
    """Invert a planar pose represented as [x, y, yaw]."""

    value = np.asarray(pose, dtype=float)
    cosine, sine = math.cos(value[2]), math.sin(value[2])
    return np.array(
        [
            -cosine * value[0] - sine * value[1],
            sine * value[0] - cosine * value[1],
            wrap_angle(-value[2]),
        ]
    )


class SparseMapOdomPropagator:
    """Propagate sparse map poses with timestamped odometry poses."""

    def __init__(self, config: BaseMappingConfig):
        self.config = config
        self._odom_history: deque[tuple[float, np.ndarray]] = deque()
        self._map_from_odom: np.ndarray | None = None
        self._anchor_stamp: float | None = None
        self._anchor_odom_pose: np.ndarray | None = None
        self._last_translation_jump = 0.0
        self._last_yaw_jump = 0.0

    @property
    def anchor_stamp(self) -> float | None:
        return self._anchor_stamp

    def add_odom_pose(self, stamp: float, pose: Sequence[float]) -> None:
        value = self._validated_pose("odom pose", pose)
        if not np.isfinite(stamp):
            raise StateValidationError([f"non-finite odom pose timestamp: {stamp}"])
        if self._odom_history and stamp <= self._odom_history[-1][0]:
            raise StateValidationError(
                [
                    "odom pose timestamp is not increasing: "
                    f"current={stamp:.9f}, previous={self._odom_history[-1][0]:.9f}"
                ]
            )
        self._odom_history.append((float(stamp), value))
        oldest_allowed = float(stamp) - self.config.odom_history_duration
        while len(self._odom_history) > 2 and self._odom_history[1][0] < oldest_allowed:
            self._odom_history.popleft()

    def update_anchor(
        self,
        *,
        map_pose: Sequence[float],
        anchor_stamp: float,
        parent_frame: str,
        child_frame: str,
    ) -> bool:
        """Install a new global keyframe anchor; return False if already seen."""

        errors = []
        if parent_frame != self.config.global_frame:
            errors.append(
                f"anchor parent frame is {parent_frame!r}, expected "
                f"{self.config.global_frame!r}"
            )
        if child_frame != self.config.base_frame:
            errors.append(
                f"anchor child frame is {child_frame!r}, expected "
                f"{self.config.base_frame!r}"
            )
        if not np.isfinite(anchor_stamp):
            errors.append(f"non-finite anchor timestamp: {anchor_stamp}")
        if errors:
            raise StateValidationError(errors)

        if self._anchor_stamp is not None:
            if anchor_stamp == self._anchor_stamp:
                return False
            if anchor_stamp < self._anchor_stamp:
                raise StateValidationError(
                    [
                        "anchor timestamp moved backwards: "
                        f"current={anchor_stamp:.9f}, previous={self._anchor_stamp:.9f}"
                    ]
                )

        map_base = self._validated_pose("map anchor pose", map_pose)
        odom_base = self._interpolate_odom_pose(float(anchor_stamp))
        candidate_map_odom = compose_pose2(map_base, inverse_pose2(odom_base))

        translation_jump = 0.0
        yaw_jump = 0.0
        if self._map_from_odom is not None:
            predicted_map_base = compose_pose2(self._map_from_odom, odom_base)
            correction = compose_pose2(inverse_pose2(predicted_map_base), map_base)
            translation_jump = float(np.linalg.norm(correction[:2]))
            yaw_jump = abs(float(correction[2]))
            errors = []
            if translation_jump > self.config.max_anchor_translation_jump:
                errors.append(
                    f"map anchor translation jump={translation_jump:.6f}m > "
                    f"{self.config.max_anchor_translation_jump:.6f}m"
                )
            if yaw_jump > self.config.max_anchor_yaw_jump:
                errors.append(
                    f"map anchor yaw jump={yaw_jump:.6f}rad > "
                    f"{self.config.max_anchor_yaw_jump:.6f}rad"
                )
            if errors:
                raise StateValidationError(errors)

        self._map_from_odom = candidate_map_odom
        self._anchor_stamp = float(anchor_stamp)
        self._anchor_odom_pose = odom_base
        self._last_translation_jump = translation_jump
        self._last_yaw_jump = yaw_jump
        return True

    def propagate(self, stamp: float) -> PropagatedMapPose:
        """Return the current map pose using the latest accepted anchor."""

        if self._map_from_odom is None or self._anchor_odom_pose is None:
            raise StateValidationError(["no map keyframe anchor received"])
        odom_pose = self._interpolate_odom_pose(float(stamp))
        relative = compose_pose2(inverse_pose2(self._anchor_odom_pose), odom_pose)
        translation = float(np.linalg.norm(relative[:2]))
        yaw = abs(float(relative[2]))
        errors = []
        if translation > self.config.max_anchor_propagation_translation:
            errors.append(
                f"odom propagation translation={translation:.6f}m > "
                f"{self.config.max_anchor_propagation_translation:.6f}m"
            )
        if yaw > self.config.max_anchor_propagation_yaw:
            errors.append(
                f"odom propagation yaw={yaw:.6f}rad > "
                f"{self.config.max_anchor_propagation_yaw:.6f}rad"
            )
        if errors:
            raise StateValidationError(errors)
        return PropagatedMapPose(
            position=compose_pose2(self._map_from_odom, odom_pose),
            stamp=float(stamp),
            anchor_stamp=float(self._anchor_stamp),
            anchor_age=float(stamp - self._anchor_stamp),
            translation_since_anchor=translation,
            yaw_since_anchor=yaw,
            last_translation_jump=self._last_translation_jump,
            last_yaw_jump=self._last_yaw_jump,
        )

    def _interpolate_odom_pose(self, stamp: float) -> np.ndarray:
        if not self._odom_history:
            raise StateValidationError(["no odom pose history available"])
        samples = list(self._odom_history)
        if stamp <= samples[0][0]:
            nearest_stamp, nearest_pose = samples[0]
            return self._nearest_odom_pose(stamp, nearest_stamp, nearest_pose)
        if stamp >= samples[-1][0]:
            nearest_stamp, nearest_pose = samples[-1]
            return self._nearest_odom_pose(stamp, nearest_stamp, nearest_pose)

        for (before_stamp, before), (after_stamp, after) in zip(samples, samples[1:]):
            if before_stamp <= stamp <= after_stamp:
                nearest_skew = min(stamp - before_stamp, after_stamp - stamp)
                if nearest_skew > self.config.max_anchor_odom_skew:
                    raise StateValidationError(
                        [
                            f"anchor/odom nearest timestamp skew={nearest_skew:.6f}s > "
                            f"{self.config.max_anchor_odom_skew:.6f}s"
                        ]
                    )
                ratio = (stamp - before_stamp) / (after_stamp - before_stamp)
                yaw_delta = wrap_angle(after[2] - before[2])
                return np.array(
                    [
                        before[0] + ratio * (after[0] - before[0]),
                        before[1] + ratio * (after[1] - before[1]),
                        wrap_angle(before[2] + ratio * yaw_delta),
                    ]
                )
        raise RuntimeError("failed to bracket odom timestamp")

    def _nearest_odom_pose(
        self, requested_stamp: float, nearest_stamp: float, pose: np.ndarray
    ) -> np.ndarray:
        skew = abs(requested_stamp - nearest_stamp)
        if skew > self.config.max_anchor_odom_skew:
            raise StateValidationError(
                [
                    f"anchor/odom nearest timestamp skew={skew:.6f}s > "
                    f"{self.config.max_anchor_odom_skew:.6f}s"
                ]
            )
        return pose.copy()

    @staticmethod
    def _validated_pose(label: str, pose: Sequence[float]) -> np.ndarray:
        value = np.asarray(pose, dtype=float).reshape(-1).copy()
        errors = []
        if value.size != 3:
            errors.append(f"{label} must have length 3, got {value.size}")
        elif not np.all(np.isfinite(value)):
            errors.append(f"{label} contains NaN or Inf")
        if errors:
            raise StateValidationError(errors)
        value[2] = wrap_angle(value[2])
        return value


class StretchBaseStateMapper:
    """Validate pose/odometry frames and construct world-frame WB-MPC state."""

    VALID_POSE_SOURCES = ("map_tf_odom", "map_tf", "odom_pose")

    def __init__(self, config: BaseMappingConfig):
        if config.pose_source not in self.VALID_POSE_SOURCES:
            raise ValueError(
                f"pose_source must be one of {self.VALID_POSE_SOURCES}, "
                f"got {config.pose_source!r}"
            )
        self.config = config
        self._previous_pose_stamp: float | None = None
        self._previous_odom_stamp: float | None = None

    def reset(self) -> None:
        self._previous_pose_stamp = None
        self._previous_odom_stamp = None

    def map(
        self,
        *,
        pose_translation: Sequence[float],
        pose_quaternion_xyzw: Sequence[float],
        pose_stamp: float,
        pose_parent_frame: str,
        pose_child_frame: str,
        odom_linear_velocity: Sequence[float],
        odom_yaw_rate: float,
        odom_stamp: float,
        odom_frame_id: str,
        odom_child_frame_id: str,
        now: float | None = None,
    ) -> MappedBaseState:
        errors: list[str] = []
        translation = np.asarray(pose_translation, dtype=float).reshape(-1)
        quaternion = np.asarray(pose_quaternion_xyzw, dtype=float).reshape(-1)
        linear_body = np.asarray(odom_linear_velocity, dtype=float).reshape(-1)

        if translation.size != 3:
            errors.append(
                f"pose translation must have length 3, got {translation.size}"
            )
        if quaternion.size != 4:
            errors.append(f"pose quaternion must have length 4, got {quaternion.size}")
        if linear_body.size != 3:
            errors.append(
                f"odom linear velocity must have length 3, got {linear_body.size}"
            )
        if errors:
            raise StateValidationError(errors)

        expected_pose_parent = (
            self.config.odom_frame
            if self.config.pose_source == "odom_pose"
            else self.config.global_frame
        )
        if pose_parent_frame != expected_pose_parent:
            errors.append(
                f"pose parent frame is {pose_parent_frame!r}, expected "
                f"{expected_pose_parent!r} for {self.config.pose_source}"
            )
        if pose_child_frame != self.config.base_frame:
            errors.append(
                f"pose child frame is {pose_child_frame!r}, expected "
                f"{self.config.base_frame!r}"
            )
        if odom_frame_id != self.config.odom_frame:
            errors.append(
                f"Odometry frame_id is {odom_frame_id!r}, expected "
                f"{self.config.odom_frame!r}"
            )
        if odom_child_frame_id != self.config.base_frame:
            errors.append(
                f"Odometry child_frame_id is {odom_child_frame_id!r}, expected "
                f"{self.config.base_frame!r}"
            )

        for label, stamp, previous in (
            ("pose", pose_stamp, self._previous_pose_stamp),
            ("odom", odom_stamp, self._previous_odom_stamp),
        ):
            if not np.isfinite(stamp):
                errors.append(f"non-finite {label} timestamp: {stamp}")
            elif previous is not None:
                if label == "pose" and stamp < previous:
                    errors.append(
                        f"pose timestamp moved backwards: current={stamp:.9f}, "
                        f"previous={previous:.9f}"
                    )
                elif label == "odom" and stamp <= previous:
                    errors.append(
                        f"odom timestamp is not increasing: current={stamp:.9f}, "
                        f"previous={previous:.9f}"
                    )

        if not np.all(np.isfinite(translation)):
            errors.append("pose translation contains NaN or Inf")
        if not np.all(np.isfinite(quaternion)):
            errors.append("pose quaternion contains NaN or Inf")
        if not np.all(np.isfinite(linear_body)) or not np.isfinite(odom_yaw_rate):
            errors.append("odom twist contains NaN or Inf")

        quaternion_norm = float(np.linalg.norm(quaternion))
        if np.isfinite(quaternion_norm) and abs(quaternion_norm - 1.0) > (
            self.config.quaternion_norm_tolerance
        ):
            errors.append(
                f"quaternion norm is {quaternion_norm:.9f}, expected 1 within "
                f"{self.config.quaternion_norm_tolerance:.9f}"
            )

        pose_age = None if now is None else float(now - pose_stamp)
        odom_age = None if now is None else float(now - odom_stamp)
        self._validate_age("pose", pose_age, self.config.max_pose_age, errors)
        self._validate_age("odom", odom_age, self.config.max_odom_age, errors)

        pose_odom_skew = abs(float(pose_stamp - odom_stamp))
        if pose_odom_skew > self.config.max_pose_odom_skew:
            errors.append(
                f"pose/odom timestamp skew={pose_odom_skew:.6f}s > "
                f"{self.config.max_pose_odom_skew:.6f}s"
            )

        lateral_limit = self.config.max_abs_body_lateral_velocity
        if lateral_limit is not None and abs(linear_body[1]) > lateral_limit:
            errors.append(
                f"body lateral velocity={linear_body[1]:.6f}m/s exceeds "
                f"{lateral_limit:.6f}m/s"
            )

        if errors:
            raise StateValidationError(errors)

        normalized_quaternion = quaternion / quaternion_norm
        yaw = yaw_from_quaternion_xyzw(normalized_quaternion)
        cosine, sine = math.cos(yaw), math.sin(yaw)
        velocity_world = np.array(
            [
                cosine * linear_body[0] - sine * linear_body[1],
                sine * linear_body[0] + cosine * linear_body[1],
                float(odom_yaw_rate),
            ]
        )
        velocity_body = np.array([linear_body[0], linear_body[1], float(odom_yaw_rate)])
        position = np.array([translation[0], translation[1], yaw])

        self._previous_pose_stamp = float(pose_stamp)
        self._previous_odom_stamp = float(odom_stamp)
        return MappedBaseState(
            pose_stamp=float(pose_stamp),
            odom_stamp=float(odom_stamp),
            position=position,
            velocity_world=velocity_world,
            velocity_body=velocity_body,
            pose_age=pose_age,
            odom_age=odom_age,
            pose_odom_skew=pose_odom_skew,
            pose_frame=expected_pose_parent,
        )

    def _validate_age(
        self,
        label: str,
        age: float | None,
        maximum: float,
        errors: list[str],
    ) -> None:
        if age is None:
            return
        if not np.isfinite(age):
            errors.append(f"non-finite {label} age: {age}")
        elif age > maximum:
            errors.append(f"stale {label}: age={age:.6f}s > {maximum:.6f}s")
        elif age < -self.config.future_tolerance:
            errors.append(
                f"{label} timestamp is in the future: age={age:.6f}s < "
                f"-{self.config.future_tolerance:.6f}s"
            )
