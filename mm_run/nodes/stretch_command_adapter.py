#!/usr/bin/env python3
"""Default-shadow ROS 2 adapter from WB-MPC velocity to real Stretch commands."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64MultiArray, String
from std_srvs.srv import Trigger
from stretch_runtime.real_base_state import (
    BaseMappingConfig,
    SparseMapOdomPropagator,
    StretchBaseStateMapper,
    yaw_from_quaternion_xyzw,
)
from stretch_runtime.real_command import (
    WB_MPC_VELOCITY_SIZE,
    CommandCoreConfig,
    CommandSafetyError,
    SafeCommand,
    StretchCommandCore,
    command_owner_errors,
)
from stretch_runtime.real_full_state import FullStateConfig, StretchFullStateCombiner
from stretch_runtime.real_state import (
    JointMappingConfig,
    StateValidationError,
    StretchJointStateMapper,
)
from stretch_runtime.streaming_position import sg3_qpos_from_joint_state
from tf2_ros import Buffer, TransformException, TransformListener


def _reliable_qos() -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )


def _stamp_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9


class StretchCommandAdapter(Node):
    """Validate feedback, shape velocity, and optionally own both command topics."""

    def __init__(self, adapter_config: dict, *, log_path: str | None = None):
        super().__init__("stretch_command_adapter")
        config = adapter_config["stretch_command_adapter"]
        state_config = config["state"]
        self.adapter_config = config
        self.base_mapper = StretchBaseStateMapper(
            BaseMappingConfig.from_dict(state_config["base"])
        )
        if self.base_mapper.config.pose_source != "map_tf_odom":
            raise ValueError("real command adapter requires pose_source=map_tf_odom")
        self.propagator = SparseMapOdomPropagator(self.base_mapper.config)
        self.joint_mapper = StretchJointStateMapper(
            JointMappingConfig.from_dict(state_config["joint"])
        )
        self.full_combiner = StretchFullStateCombiner(
            FullStateConfig.from_dict(state_config["full"])
        )
        self.core = StretchCommandCore(CommandCoreConfig.from_dict(config["command"]))

        self.publish_rate = float(config.get("publish_rate", 10.0))
        if not math.isfinite(self.publish_rate) or self.publish_rate <= 0.0:
            raise ValueError("publish_rate must be positive and finite")
        self.period = 1.0 / self.publish_rate
        self.state_receive_timeout = float(config.get("state_receive_timeout", 0.10))
        self.command_receive_timeout = float(
            config.get("command_receive_timeout", 0.20)
        )
        self.status_receive_timeout = float(config.get("status_receive_timeout", 0.50))
        self.enable_zero_velocity_tolerance = float(
            config.get("enable_zero_velocity_tolerance", 1.0e-6)
        )

        self.velocity_command_topic = str(
            config.get("velocity_command_topic", "/wbmpc/velocity_command")
        )
        self.state_topic = str(config.get("state_topic", "/wbmpc/state"))
        self.base_command_topic = str(
            config.get("base_command_topic", "/stretch/cmd_vel")
        )
        self.joint_command_topic = str(
            config.get("joint_command_topic", "/joint_pose_cmd")
        )

        self._latest_base_state = None
        self._latest_joint_state = None
        self._latest_full_state = None
        self._latest_full_receive_time = None
        self._latest_qpos = None
        self._latest_qpos_receive_time = None
        self._latest_velocity_request = None
        self._latest_command_receive_time = None
        self._mode = None
        self._homed = None
        self._runstopped = None
        self._streaming = None
        self._status_receive_times: dict[str, float] = {}
        self._last_tick_time = time.monotonic()
        self._last_warning_time = 0.0
        self._last_state_error = ""
        self._enabled = False
        self._latched = False
        self._stop_reason = ""
        self._streaming_activated_by_us = False
        self._deactivate_future = None
        self._last_combined_pair = None

        self._base_publisher = None
        self._joint_publisher = None

        self._tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self._tf_listener = TransformListener(self._tf_buffer, self)
        qos = _reliable_qos()
        self._odom_subscription = self.create_subscription(
            Odometry,
            str(config.get("odom_topic", "/odom")),
            self._odom_callback,
            qos,
        )
        self._joint_subscription = self.create_subscription(
            JointState,
            str(config.get("joint_state_topic", "/stretch/joint_states")),
            self._joint_state_callback,
            qos,
        )
        self._velocity_subscription = self.create_subscription(
            Float64MultiArray,
            self.velocity_command_topic,
            self._velocity_command_callback,
            qos,
        )
        self._mode_subscription = self.create_subscription(
            String,
            str(config.get("mode_topic", "/mode")),
            self._mode_callback,
            qos,
        )
        self._homed_subscription = self.create_subscription(
            Bool,
            str(config.get("homed_topic", "/is_homed")),
            self._homed_callback,
            qos,
        )
        self._runstop_subscription = self.create_subscription(
            Bool,
            str(config.get("runstop_topic", "/is_runstopped")),
            self._runstop_callback,
            qos,
        )
        self._streaming_subscription = self.create_subscription(
            Bool,
            str(config.get("streaming_topic", "/is_streaming_position")),
            self._streaming_callback,
            qos,
        )
        self._status_publisher = self.create_publisher(
            String,
            str(config.get("status_topic", "/stretch_command_adapter/status")),
            qos,
        )
        self._state_publisher = self.create_publisher(
            JointState,
            self.state_topic,
            qos,
        )
        self._activate_client = self.create_client(
            Trigger,
            str(
                config.get("activate_streaming_service", "/activate_streaming_position")
            ),
        )
        self._deactivate_client = self.create_client(
            Trigger,
            str(
                config.get(
                    "deactivate_streaming_service", "/deactivate_streaming_position"
                )
            ),
        )
        self._timer = self.create_timer(self.period, self._command_tick)

        configured_log_path = config.get("shadow_log_path", "")
        selected_log_path = log_path if log_path is not None else configured_log_path
        self._command_log_file = None
        if selected_log_path:
            path = Path(selected_log_path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            self._command_log_file = path.open("w", encoding="utf-8")
            self.get_logger().info(f"Writing command records to {path.resolve()}")

        self.get_logger().info(
            "Started in shadow mode; no /stretch/cmd_vel or /joint_pose_cmd "
            "publisher has been created"
        )

    def _odom_callback(self, message: Odometry) -> None:
        stamp = _stamp_seconds(message.header.stamp)
        pose = message.pose.pose
        quaternion = [
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ]
        now_ros = self.get_clock().now().nanoseconds * 1.0e-9
        try:
            quaternion_norm = float(np.linalg.norm(quaternion))
            if abs(quaternion_norm - 1.0) > (
                self.base_mapper.config.quaternion_norm_tolerance
            ):
                raise StateValidationError(
                    [
                        f"odom pose quaternion norm is {quaternion_norm:.9f}, "
                        "expected 1"
                    ]
                )
            self.propagator.add_odom_pose(
                stamp,
                [
                    pose.position.x,
                    pose.position.y,
                    yaw_from_quaternion_xyzw(quaternion),
                ],
            )
            try:
                transform = self._tf_buffer.lookup_transform(
                    self.base_mapper.config.global_frame,
                    self.base_mapper.config.base_frame,
                    Time(),
                )
            except TransformException:
                transform = None
            if transform is not None:
                rotation = transform.transform.rotation
                self.propagator.update_anchor(
                    map_pose=[
                        transform.transform.translation.x,
                        transform.transform.translation.y,
                        yaw_from_quaternion_xyzw(
                            [rotation.x, rotation.y, rotation.z, rotation.w]
                        ),
                    ],
                    anchor_stamp=_stamp_seconds(transform.header.stamp),
                    parent_frame=transform.header.frame_id,
                    child_frame=transform.child_frame_id,
                )
            propagated = self.propagator.propagate(stamp)
            yaw = float(propagated.position[2])
            mapped = self.base_mapper.map(
                pose_translation=[
                    propagated.position[0],
                    propagated.position[1],
                    0.0,
                ],
                pose_quaternion_xyzw=[
                    0.0,
                    0.0,
                    math.sin(yaw / 2.0),
                    math.cos(yaw / 2.0),
                ],
                pose_stamp=stamp,
                pose_parent_frame=self.base_mapper.config.global_frame,
                pose_child_frame=self.base_mapper.config.base_frame,
                odom_linear_velocity=[
                    message.twist.twist.linear.x,
                    message.twist.twist.linear.y,
                    message.twist.twist.linear.z,
                ],
                odom_yaw_rate=message.twist.twist.angular.z,
                odom_stamp=stamp,
                odom_frame_id=message.header.frame_id,
                odom_child_frame_id=message.child_frame_id,
                now=now_ros,
            )
        except StateValidationError as error:
            self._latest_base_state = None
            self._handle_state_error("base", error)
            return
        self._latest_base_state = mapped
        self._try_combine_state()

    def _joint_state_callback(self, message: JointState) -> None:
        receive_time = time.monotonic()
        stamp = _stamp_seconds(message.header.stamp)
        now_ros = self.get_clock().now().nanoseconds * 1.0e-9
        try:
            qpos = sg3_qpos_from_joint_state(message.name, message.position)
            mapped = self.joint_mapper.map(
                message.name,
                message.position,
                message.velocity,
                stamp=stamp,
                now=now_ros,
            )
        except (StateValidationError, ValueError) as error:
            self._latest_joint_state = None
            self._handle_state_error("joint", error)
            return
        self._latest_joint_state = mapped
        self._latest_qpos = qpos
        self._latest_qpos_receive_time = receive_time
        self._try_combine_state()

    def _try_combine_state(self) -> None:
        if self._latest_base_state is None or self._latest_joint_state is None:
            return
        pair = (
            self._latest_base_state.odom_stamp,
            self._latest_joint_state.stamp,
        )
        if pair == self._last_combined_pair:
            return
        if abs(pair[0] - pair[1]) > self.full_combiner.config.max_state_skew:
            return
        now_ros = self.get_clock().now().nanoseconds * 1.0e-9
        try:
            combined = self.full_combiner.combine(
                self._latest_base_state,
                self._latest_joint_state,
                now=now_ros,
            )
        except StateValidationError as error:
            self._handle_state_error("full", error)
            return
        self._latest_full_state = combined
        self._latest_full_receive_time = time.monotonic()
        self._last_combined_pair = pair
        self._last_state_error = ""
        state_message = JointState()
        state_message.header.stamp = Time(
            nanoseconds=int(round(combined.stamp * 1.0e9))
        ).to_msg()
        state_message.header.frame_id = self.base_mapper.config.global_frame
        state_message.name = list(combined.names)
        state_message.position = combined.position.tolist()
        state_message.velocity = combined.velocity.tolist()
        self._state_publisher.publish(state_message)

    def _handle_state_error(self, source: str, error: Exception) -> None:
        reason = f"invalid {source} state: {error}"
        self._last_state_error = reason
        self._warn_throttled(reason)
        if self._enabled:
            self._latch_stop(reason)

    def _velocity_command_callback(self, message: Float64MultiArray) -> None:
        values = np.asarray(message.data, dtype=float).reshape(-1)
        if values.size != WB_MPC_VELOCITY_SIZE or not np.all(np.isfinite(values)):
            reason = (
                f"velocity request must contain {WB_MPC_VELOCITY_SIZE} finite values"
            )
            self._warn_throttled(reason)
            if self._enabled:
                self._latch_stop(reason)
            return
        self._latest_velocity_request = values.copy()
        self._latest_command_receive_time = time.monotonic()

    def _record_status(self, name: str, value) -> None:
        setattr(self, f"_{name}", value)
        self._status_receive_times[name] = time.monotonic()

    def _mode_callback(self, message: String) -> None:
        self._record_status("mode", message.data)

    def _homed_callback(self, message: Bool) -> None:
        self._record_status("homed", bool(message.data))

    def _runstop_callback(self, message: Bool) -> None:
        self._record_status("runstopped", bool(message.data))

    def _streaming_callback(self, message: Bool) -> None:
        self._record_status("streaming", bool(message.data))

    def _health_errors(self, *, expect_streaming: bool | None) -> list[str]:
        now = time.monotonic()
        errors = []
        for label, receive_time, timeout in (
            ("full state", self._latest_full_receive_time, self.state_receive_timeout),
            (
                "SG3 JointState",
                self._latest_qpos_receive_time,
                self.state_receive_timeout,
            ),
            (
                "velocity command",
                self._latest_command_receive_time,
                self.command_receive_timeout,
            ),
        ):
            if receive_time is None:
                errors.append(f"{label} has not been received")
            elif now - receive_time > timeout:
                errors.append(f"{label} receive timeout: {now - receive_time:.3f}s")
        for name in ("mode", "homed", "runstopped", "streaming"):
            receive_time = self._status_receive_times.get(name)
            if receive_time is None:
                errors.append(f"{name} status has not been received")
            elif now - receive_time > self.status_receive_timeout:
                errors.append(
                    f"{name} status receive timeout: {now - receive_time:.3f}s"
                )
        if self._mode != "navigation":
            errors.append(f"mode is {self._mode!r}, expected 'navigation'")
        if self._homed is not True:
            errors.append(f"homed is {self._homed!r}, expected true")
        if self._runstopped is not False:
            errors.append(f"runstopped is {self._runstopped!r}, expected false")
        if expect_streaming is not None and self._streaming is not expect_streaming:
            errors.append(
                f"streaming is {self._streaming!r}, expected {expect_streaming}"
            )
        return errors

    def _command_owner_errors(self, expected_count: int) -> list[str]:
        counts = {
            topic: self.count_publishers(topic)
            for topic in (self.base_command_topic, self.joint_command_topic)
        }
        return list(command_owner_errors(counts, expected_count))

    def _command_tick(self) -> None:
        now = time.monotonic()
        dt = now - self._last_tick_time
        self._last_tick_time = now
        if self._latched:
            self._request_deactivate()
            # Never publish qpos while the driver transitions to (or is already
            # in) streaming-position=false. Continue sending only the
            # independent base zero command until process teardown.
            self._publish_stop_once(include_joint=False)
            self._publish_status(None)
            return
        if (
            self._latest_full_state is None
            or self._latest_qpos is None
            or self._latest_velocity_request is None
        ):
            self._publish_status(None)
            return

        if self._enabled:
            errors = self._health_errors(expect_streaming=True)
            errors.extend(self._command_owner_errors(expected_count=1))
            if errors:
                self._latch_stop("; ".join(errors))
                return
            step_dt = dt
            enforce_tracking = True
        else:
            errors = self._health_errors(expect_streaming=False)
            if errors:
                reason = "; ".join(errors)
                self._warn_throttled(reason)
                try:
                    self.core.reset(self._latest_qpos)
                except CommandSafetyError as error:
                    reason = f"{reason}; {error}"
                self._publish_status(None, error=reason)
                return
            step_dt = self.period
            enforce_tracking = False

        try:
            command = self.core.step(
                yaw=float(self._latest_full_state.position[2]),
                measured_qpos=self._latest_qpos,
                requested_velocity_world=self._latest_velocity_request,
                dt=step_dt,
                enforce_tracking=enforce_tracking,
            )
        except CommandSafetyError as error:
            if self._enabled:
                self._latch_stop(str(error))
            else:
                self._warn_throttled(str(error))
                self._publish_status(None, error=str(error))
            return

        if abs(command.lateral_velocity) > (self.core.config.lateral_warn_threshold):
            self._warn_throttled(
                f"projecting body lateral command {command.lateral_velocity:.6f} m/s "
                "to zero"
            )
        if command.arm_projection_residual > (
            self.core.config.arm_projection_warn_threshold
        ):
            self._warn_throttled(
                "projecting independent arm-segment velocities to aggregate "
                f"wrist_extension; residual={command.arm_projection_residual:.6f} m/s"
            )
        if self._enabled:
            self._publish_command(command)
        self._publish_status(command)
        self._write_command_record(command)

    def wait_and_enable_wbmpc(self, timeout: float) -> None:
        """Perform physical preflight and explicitly enter WB-MPC mode."""

        if self._latched:
            raise RuntimeError(f"adapter is latched: {self._stop_reason}")
        deadline = time.monotonic() + timeout
        errors = ["waiting for preflight data"]
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            errors = self._health_errors(expect_streaming=False)
            if not errors:
                break
        if errors:
            raise RuntimeError("preflight failed: " + "; ".join(errors))
        if np.max(np.abs(self._latest_velocity_request)) > (
            self.enable_zero_velocity_tolerance
        ):
            raise RuntimeError(
                "preflight requires a zero WB-MPC velocity request before enable"
            )
        ownership_errors = self._command_owner_errors(expected_count=0)
        if ownership_errors:
            raise RuntimeError("preflight failed: " + "; ".join(ownership_errors))

        self.core.reset(self._latest_qpos)
        self.core.step(
            yaw=float(self._latest_full_state.position[2]),
            measured_qpos=self._latest_qpos,
            requested_velocity_world=self._latest_velocity_request,
            dt=self.period,
            enforce_tracking=False,
        )
        self.core.reset(self._latest_qpos)
        self._call_trigger(
            self._activate_client,
            str(
                self.adapter_config.get(
                    "activate_streaming_service", "/activate_streaming_position"
                )
            ),
        )
        self._streaming_activated_by_us = True

        deadline = time.monotonic() + timeout
        while (
            rclpy.ok() and time.monotonic() < deadline and self._streaming is not True
        ):
            rclpy.spin_once(self, timeout_sec=0.05)
        if self._streaming is not True:
            self._request_deactivate()
            raise RuntimeError("streaming-position did not become active")

        qos = _reliable_qos()
        self._base_publisher = self.create_publisher(
            Twist, self.base_command_topic, qos
        )
        self._joint_publisher = self.create_publisher(
            Float64MultiArray, self.joint_command_topic, qos
        )
        self.core.reset(self._latest_qpos)
        # Activation may finish immediately before the already-scheduled ROS
        # timer fires. Seed one nominal period so that first enabled step cannot
        # spuriously violate CommandCoreConfig.min_dt.
        self._last_tick_time = time.monotonic() - self.period
        self._enabled = True
        self._publish_stop_once()
        self.get_logger().warning(
            "WB-MPC OUTPUT ENABLED after explicit --execute and physical preflight"
        )

    def _publish_command(self, command: SafeCommand) -> None:
        if self._base_publisher is None or self._joint_publisher is None:
            self._latch_stop("command publishers are unavailable")
            return
        twist = Twist()
        twist.linear.x = command.base_linear_x
        twist.linear.y = 0.0
        twist.angular.z = command.base_angular_z
        pose = Float64MultiArray()
        pose.data = [float(value) for value in command.streaming_qpos]
        self._base_publisher.publish(twist)
        self._joint_publisher.publish(pose)

    def _publish_stop_once(self, *, include_joint: bool = True) -> None:
        if self._base_publisher is not None:
            self._base_publisher.publish(Twist())
        if (
            include_joint
            and self._joint_publisher is not None
            and self._latest_qpos is not None
        ):
            pose = Float64MultiArray()
            hold_qpos = [float(value) for value in self._latest_qpos]
            hold_qpos[-2:] = [0.0, 0.0]
            pose.data = hold_qpos
            self._joint_publisher.publish(pose)

    def _latch_stop(self, reason: str) -> None:
        if self._latched:
            return
        self._enabled = False
        self._latched = True
        self._stop_reason = str(reason)
        self.get_logger().error(f"LATCHED STOP: {self._stop_reason}")
        # Deactivation is the arm stop for a latched fault. Publishing a hold
        # qpos immediately before the asynchronous service call can be reordered
        # at the driver and rejected after streaming-position becomes false.
        self._publish_stop_once(include_joint=False)
        self._request_deactivate()
        self._publish_status(None, error=self._stop_reason)

    def _request_deactivate(self) -> None:
        if self._deactivate_future is not None and self._deactivate_future.done():
            try:
                response = self._deactivate_future.result()
            except Exception:
                response = None
            if response is not None and response.success:
                self._streaming_activated_by_us = False
                return
            self._deactivate_future = None
        if (
            self._streaming_activated_by_us
            and self._deactivate_future is None
            and self._deactivate_client.service_is_ready()
        ):
            self._deactivate_future = self._deactivate_client.call_async(
                Trigger.Request()
            )

    def _call_trigger(self, client, name: str, timeout: float = 5.0) -> None:
        if not client.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(f"service {name} is unavailable")
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        if not future.done() or future.result() is None:
            raise RuntimeError(f"service {name} timed out")
        response = future.result()
        if not response.success:
            raise RuntimeError(f"service {name} failed: {response.message}")

    def _publish_status(
        self, command: SafeCommand | None, *, error: str | None = None
    ) -> None:
        state = "latched" if self._latched else "wbmpc" if self._enabled else "shadow"
        record = {
            "state": state,
            "stop_reason": self._stop_reason,
            "error": error if error is not None else self._last_state_error or None,
            "state_ready": self._latest_full_state is not None,
            "qpos_ready": self._latest_qpos is not None,
            "command_ready": self._latest_velocity_request is not None,
            "mode": self._mode,
            "homed": self._homed,
            "runstopped": self._runstopped,
            "streaming": self._streaming,
        }
        if command is not None:
            record.update(
                {
                    "base_linear_x": command.base_linear_x,
                    "base_angular_z": command.base_angular_z,
                    "lateral_velocity": command.lateral_velocity,
                    "arm_projection_residual": command.arm_projection_residual,
                    "clipped_channels": list(command.clipped_channels),
                    "safe_driver_velocity": command.safe_driver_velocity.tolist(),
                    "streaming_qpos": command.streaming_qpos.tolist(),
                }
            )
        message = String()
        message.data = json.dumps(record, sort_keys=True)
        self._status_publisher.publish(message)

    def _write_command_record(self, command: SafeCommand) -> None:
        if self._command_log_file is None:
            return
        record = {
            "monotonic_time": time.monotonic(),
            "wbmpc_enabled": self._enabled,
            "base_linear_x": command.base_linear_x,
            "base_angular_z": command.base_angular_z,
            "lateral_velocity": command.lateral_velocity,
            "arm_projection_residual": command.arm_projection_residual,
            "requested_driver_velocity": command.requested_driver_velocity.tolist(),
            "safe_driver_velocity": command.safe_driver_velocity.tolist(),
            "realized_model_velocity": command.realized_model_velocity.tolist(),
            "streaming_qpos": command.streaming_qpos.tolist(),
            "clipped_channels": list(command.clipped_channels),
        }
        self._command_log_file.write(json.dumps(record, sort_keys=True) + "\n")
        self._command_log_file.flush()

    def _warn_throttled(self, text: str) -> None:
        now = time.monotonic()
        if now - self._last_warning_time >= 1.0:
            self.get_logger().warning(text)
            self._last_warning_time = now

    def shutdown_commands(self) -> None:
        """Best-effort zero/hold/deactivate sequence for every exit path."""

        self._enabled = False
        include_joint = (
            not self._latched
            and self._streaming_activated_by_us
            and self._streaming is True
        )
        for _ in range(5):
            self._publish_stop_once(include_joint=include_joint)
            if rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.02)
        if self._streaming_activated_by_us and rclpy.ok():
            try:
                self._call_trigger(
                    self._deactivate_client,
                    str(
                        self.adapter_config.get(
                            "deactivate_streaming_service",
                            "/deactivate_streaming_position",
                        )
                    ),
                )
            except Exception as error:  # Best effort during process teardown.
                self.get_logger().error(
                    f"failed to deactivate streaming-position: {error}"
                )
        self._streaming_activated_by_us = False

    def close(self) -> None:
        if self._command_log_file is not None:
            self._command_log_file.close()
            self._command_log_file = None


def _load_adapter_config(path_text: str) -> dict:
    path = Path(path_text).expanduser().resolve()
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def main(argv=None) -> None:
    argv = sys.argv if argv is None else argv
    # Keep the ROS context alive while Python handles Ctrl-C so the finally
    # block can publish zero/hold and deactivate streaming-position first.
    rclpy.init(args=argv, signal_handler_options=SignalHandlerOptions.NO)
    parsed_argv = remove_ros_args(args=argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--log", help="Override shadow/enabled JSONL output path.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Explicitly pass physical preflight and enter WB-MPC hardware mode.",
    )
    parser.add_argument("--duration", type=float, default=0.0)
    args = parser.parse_args(parsed_argv[1:])

    config = _load_adapter_config(args.config)
    node = StretchCommandAdapter(config, log_path=args.log)
    started = time.monotonic()
    exit_error = None
    try:
        if args.execute:
            timeout = float(
                config["stretch_command_adapter"].get("startup_timeout", 10.0)
            )
            node.wait_and_enable_wbmpc(timeout)
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.2)
            if args.duration > 0.0 and time.monotonic() - started >= args.duration:
                break
    except KeyboardInterrupt:
        pass
    except Exception as error:
        if rclpy.ok():
            exit_error = error
            node.get_logger().error(str(error))
    finally:
        node.shutdown_commands()
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    if exit_error is not None:
        raise exit_error


if __name__ == "__main__":
    main()
