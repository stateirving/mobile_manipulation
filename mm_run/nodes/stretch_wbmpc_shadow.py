#!/usr/bin/env python3
"""Run the real WB-MPC stack against Stretch feedback without hardware output."""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.utilities import remove_ros_args
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String
from stretch_runtime.stretch_mimic import (
    STRETCH_EXTERNAL_DIMENSION,
    STRETCH_MIMIC_DIMENSION,
    expand_stretch_mimic_vector,
    reduce_stretch_mimic_vector,
)
from stretch_runtime.wbmpc_shadow import (
    WBMPCState,
    controller_velocity_limits,
    integrate_acceleration_velocity,
    sample_acceleration_plan,
    validate_wbmpc_state,
)

import mm_control.MPC as MPC
from mm_plan.TaskManager import TaskManager
from mm_utils import parsing


class StretchWBMPCShadow(Node):
    """Own WB-MPC planning and publish only the adapter's internal velocity input."""

    def __init__(self, runner_config: dict, controller_config: dict, *, log_path=None):
        super().__init__("stretch_wbmpc_shadow")
        config = runner_config["stretch_wbmpc_shadow"]
        self.config = config
        self.controller_config = controller_config["controller"]

        controller_type = str(self.controller_config["type"])
        controller_class = getattr(MPC, controller_type, None)
        if controller_class is None:
            raise ValueError(f"unknown controller type: {controller_type}")
        self.get_logger().info("Initializing WB-MPC controller and ESDF resources")
        self.controller = controller_class(self.controller_config)
        self._task_resources = {
            "esdf_map": getattr(self.controller, "esdf_map", None),
            "robot_model": getattr(self.controller, "robot", None),
        }
        self._planner_config = deepcopy(controller_config["planner"])
        self.task_manager = self._create_task_manager(self._planner_config)

        self.control_rate = float(
            config.get("control_rate", self.controller_config["ctrl_rate"])
        )
        self.publish_rate = float(
            config.get("publish_rate", self.controller_config["cmd_vel_pub_rate"])
        )
        self.state_timeout = float(config.get("state_timeout", 0.25))
        self.plan_timeout = float(config.get("plan_timeout", 0.50))
        for name, value in (
            ("control_rate", self.control_rate),
            ("publish_rate", self.publish_rate),
            ("state_timeout", self.state_timeout),
            ("plan_timeout", self.plan_timeout),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")

        self.dimension = int(self.controller_config["robot"]["dims"]["v"])
        self.mimic = bool(self.controller_config["robot"].get("mimic", False))
        expected_dimension = (
            STRETCH_MIMIC_DIMENSION if self.mimic else STRETCH_EXTERNAL_DIMENSION
        )
        if self.dimension != expected_dimension:
            raise ValueError(
                "real Stretch WB-MPC dimension must be "
                f"{expected_dimension} when mimic={self.mimic}, got {self.dimension}"
            )
        self.velocity_lower, self.velocity_upper = controller_velocity_limits(
            self.controller_config, self.dimension
        )
        robot_config = self.controller_config["robot"]
        self.nonholonomic = (
            str(robot_config.get("base_type", "")).lower() == "nonholonomic"
            and str(robot_config.get("nonholonomic_mode", "")).lower() == "dynamics"
        )
        if str(self.controller_config["cmd_vel_type"]).lower() != "integration":
            raise ValueError(
                "real Stretch runner currently requires cmd_vel_type=integration"
            )

        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.VOLATILE
        self.state_topic = str(config.get("state_topic", "/wbmpc/state"))
        self.velocity_topic = str(
            config.get("velocity_command_topic", "/wbmpc/velocity_command")
        )
        self.status_topic = str(config.get("status_topic", "/wbmpc/status"))
        self.expected_frame = str(config.get("global_frame", "map"))
        self._state_subscription = self.create_subscription(
            JointState, self.state_topic, self._state_callback, qos
        )
        self._velocity_publisher = self.create_publisher(
            Float64MultiArray, self.velocity_topic, qos
        )
        self._status_publisher = self.create_publisher(String, self.status_topic, qos)

        self._lock = threading.Lock()
        self._latest_state: WBMPCState | None = None
        self._latest_state_receive_time: float | None = None
        self._latest_state_stamp: float | None = None
        self._state_error = "state has not been received"
        self._acceleration_plan: np.ndarray | None = None
        self._plan_time: float | None = None
        self._velocity_command = np.zeros(self.dimension)
        self._last_publish_time = time.monotonic()
        self._control_started = time.monotonic()
        self._solver_count = 0
        self._solver_failure_count = 0
        self._solver_fallback_count = 0
        self._latest_status = {
            "mode": "shadow",
            "state": "initializing",
            "error": None,
        }
        self._stop_event = threading.Event()

        configured_log = str(config.get("log_path", ""))
        selected_log = configured_log if log_path is None else str(log_path)
        self._log_file = None
        self._log_lock = threading.Lock()
        if selected_log:
            path = Path(selected_log).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = path.open("w", encoding="utf-8")
            self.get_logger().info(f"Writing WB-MPC shadow records to {path.resolve()}")

        self._publish_timer = self.create_timer(
            1.0 / self.publish_rate, self._publish_velocity
        )
        self._status_timer = self.create_timer(0.5, self._publish_status)
        self._worker = threading.Thread(
            target=self._control_worker,
            name="stretch-wbmpc-control",
            daemon=True,
        )
        self._worker.start()
        self.get_logger().warning(
            "Started WB-MPC shadow runner; only the internal adapter velocity topic "
            "is published"
        )

    def _state_callback(self, message: JointState) -> None:
        try:
            if message.header.frame_id != self.expected_frame:
                raise ValueError(
                    f"state frame is {message.header.frame_id!r}, expected "
                    f"{self.expected_frame!r}"
                )
            state = validate_wbmpc_state(
                message.name, message.position, message.velocity
            )
            if self.mimic:
                state = WBMPCState(
                    reduce_stretch_mimic_vector(state.position),
                    reduce_stretch_mimic_vector(state.velocity),
                )
            stamp = (
                float(message.header.stamp.sec)
                + float(message.header.stamp.nanosec) * 1.0e-9
            )
            if not math.isfinite(stamp):
                raise ValueError("state timestamp is not finite")
        except ValueError as error:
            with self._lock:
                self._latest_state = None
                self._state_error = str(error)
            return

        with self._lock:
            if (
                self._latest_state_stamp is not None
                and stamp <= self._latest_state_stamp
            ):
                self._latest_state = None
                self._state_error = (
                    f"state timestamp is not increasing: {stamp:.9f} <= "
                    f"{self._latest_state_stamp:.9f}"
                )
                return
            self._latest_state = state
            self._latest_state_stamp = stamp
            self._latest_state_receive_time = time.monotonic()
            self._state_error = ""

    def _state_snapshot(self) -> tuple[WBMPCState | None, float, str]:
        with self._lock:
            state = self._latest_state
            receive_time = self._latest_state_receive_time
            error = self._state_error
            if state is not None:
                state = WBMPCState(state.position.copy(), state.velocity.copy())
        age = math.inf if receive_time is None else time.monotonic() - receive_time
        if not error and age > self.state_timeout:
            error = f"state receive timeout: {age:.3f}s"
        return state, age, error

    def _control_worker(self) -> None:
        period = 1.0 / self.control_rate
        next_tick = time.monotonic()
        while not self._stop_event.is_set():
            delay = next_tick - time.monotonic()
            if delay > 0.0 and self._stop_event.wait(delay):
                break
            next_tick = max(next_tick + period, time.monotonic())
            state, state_age, state_error = self._state_snapshot()
            if state is None or state_error:
                self._clear_plan("waiting_state", state_error or "state unavailable")
                continue

            control_time = time.monotonic() - self._control_started
            robot_states = (state.position, state.velocity)
            started = time.perf_counter()
            with self._lock:
                self._latest_status = {
                    "mode": "shadow",
                    "state": "solving",
                    "error": None,
                    "state_age": state_age,
                }
            try:
                references = self.task_manager.getReferences(
                    control_time,
                    robot_states,
                    self.controller.N + 1,
                    self.controller.dt,
                )
                _, acceleration_plan = self.controller.control(
                    control_time, robot_states, references
                )
                acceleration_plan = np.asarray(acceleration_plan, dtype=float)
                if acceleration_plan.shape != (self.controller.N, self.dimension):
                    raise ValueError(
                        "solver acceleration plan shape is "
                        f"{acceleration_plan.shape}, expected "
                        f"({self.controller.N}, {self.dimension})"
                    )
                if not np.all(np.isfinite(acceleration_plan)):
                    raise ValueError("solver acceleration plan contains NaN or Inf")
                solver_log = getattr(self.controller, "log", {})
                fallback = bool(solver_log.get("solver_fallback", False))
                elapsed = time.perf_counter() - started
                with self._lock:
                    self._solver_count += 1
                    if fallback:
                        self._solver_fallback_count += 1
                        self._acceleration_plan = None
                        self._plan_time = None
                    else:
                        self._acceleration_plan = acceleration_plan.copy()
                        self._plan_time = time.monotonic()
                    status_record = self._solver_status_record(
                        elapsed, state_age, fallback, solver_log
                    )
                    self._latest_status = status_record
                self._write_solver_record(status_record)
                self._update_task_manager(control_time, state)
            except Exception as error:
                elapsed = time.perf_counter() - started
                with self._lock:
                    self._solver_failure_count += 1
                    self._acceleration_plan = None
                    self._plan_time = None
                    self._velocity_command.fill(0.0)
                    self._latest_status = {
                        "mode": "shadow",
                        "state": "solver_error",
                        "error": str(error),
                        "solver_time": elapsed,
                        "solver_count": self._solver_count,
                        "solver_failure_count": self._solver_failure_count,
                        "solver_fallback_count": self._solver_fallback_count,
                    }
                    status_record = dict(self._latest_status)
                self._write_solver_record(status_record)

    def _create_task_manager(self, planner_config: dict) -> TaskManager:
        task_manager = TaskManager(planner_config, resources=self._task_resources)
        task_manager.activatePlanners()
        return task_manager

    def _solver_status_record(self, elapsed, state_age, fallback, solver_log):
        planner = self.task_manager.getPlanner()
        return {
            "mode": "shadow",
            "state": "fallback" if fallback else "ready",
            "error": "solver fallback" if fallback else None,
            "task_index": self.task_manager.curr_task_id,
            "task_name": planner.name,
            "solver_time": elapsed,
            "solver_status": int(solver_log.get("solver_status", 0)),
            "solver_count": self._solver_count,
            "solver_failure_count": self._solver_failure_count,
            "solver_fallback_count": self._solver_fallback_count,
            "state_age": state_age,
            "esdf_all_valid": bool(solver_log.get("esdf_all_valid", False)),
            "esdf_valid_count": int(solver_log.get("esdf_valid_count", 0)),
            "esdf_total_count": int(solver_log.get("esdf_total_count", 0)),
            "esdf_min_distance": _finite_or_none(solver_log.get("esdf_min_distance")),
            "esdf_min_margin": _finite_or_none(solver_log.get("esdf_min_margin")),
            "esdf_invalid_queries": list(solver_log.get("esdf_invalid_queries", [])),
        }

    def _write_solver_record(self, status_record: dict) -> None:
        record = dict(status_record)
        record["record_type"] = "solver"
        record["monotonic_time"] = time.monotonic()
        self._write_record(record)

    def _update_task_manager(self, control_time: float, state: WBMPCState) -> None:
        ee_position, ee_quaternion = self.controller.robot.getEE(state.position)
        ee_euler = Rotation.from_quat(ee_quaternion).as_euler("xyz")
        jacobian = self.controller.robot.jacSymMdls[
            self.controller.robot.tool_link_name + "_spatial"
        ](state.position)
        ee_velocity = np.asarray(jacobian @ state.velocity).reshape(-1)
        states = {
            "base": {
                "pose": state.position[:3].copy(),
                "velocity": state.velocity[:3].copy(),
            },
            "EE": {
                "pose": np.concatenate((ee_position, ee_euler)),
                "velocity": ee_velocity,
            },
        }
        self.task_manager.update(control_time, states)

    def _clear_plan(self, state_name: str, error: str) -> None:
        with self._lock:
            self._acceleration_plan = None
            self._plan_time = None
            self._velocity_command.fill(0.0)
            self._latest_status = {
                "mode": "shadow",
                "state": state_name,
                "error": error,
                "solver_count": self._solver_count,
                "solver_failure_count": self._solver_failure_count,
                "solver_fallback_count": self._solver_fallback_count,
            }

    def _publish_velocity(self) -> None:
        now = time.monotonic()
        state, state_age, state_error = self._state_snapshot()
        with self._lock:
            plan = (
                None
                if self._acceleration_plan is None
                else self._acceleration_plan.copy()
            )
            plan_time = self._plan_time
            previous = self._velocity_command.copy()
        plan_age = math.inf if plan_time is None else now - plan_time
        valid = (
            state is not None
            and not state_error
            and plan is not None
            and plan_age <= self.plan_timeout
        )
        if valid:
            acceleration = sample_acceleration_plan(
                plan, plan_age, float(self.controller.dt)
            )
            dt = float(np.clip(now - self._last_publish_time, 1.0e-4, 0.25))
            command = integrate_acceleration_velocity(
                previous,
                acceleration,
                dt,
                state.position,
                state.velocity,
                nonholonomic=self.nonholonomic,
            )
            command = np.clip(command, self.velocity_lower, self.velocity_upper)
            source = "solver"
        else:
            acceleration = np.zeros(self.dimension)
            command = np.zeros(self.dimension)
            source = "zero"
        self._last_publish_time = now
        with self._lock:
            self._velocity_command = command.copy()

        external_acceleration = (
            expand_stretch_mimic_vector(acceleration) if self.mimic else acceleration
        )
        external_command = (
            expand_stretch_mimic_vector(command) if self.mimic else command
        )
        message = Float64MultiArray()
        message.data = external_command.tolist()
        self._velocity_publisher.publish(message)
        self._write_record(
            {
                "record_type": "command",
                "monotonic_time": now,
                "mode": "shadow",
                "source": source,
                "state_age": None if not math.isfinite(state_age) else state_age,
                "plan_age": None if not math.isfinite(plan_age) else plan_age,
                "acceleration": external_acceleration.tolist(),
                "velocity_command": external_command.tolist(),
                "mimic_velocity_command": command.tolist() if self.mimic else None,
            }
        )

    def _publish_status(self) -> None:
        with self._lock:
            record = dict(self._latest_status)
            command = self._velocity_command.copy()
        record["velocity_command"] = (
            expand_stretch_mimic_vector(command).tolist()
            if self.mimic
            else command.tolist()
        )
        record["record_type"] = "status"
        record["monotonic_time"] = time.monotonic()
        message = String()
        message.data = json.dumps(record, sort_keys=True)
        self._status_publisher.publish(message)
        self._write_record(record)

    def _write_record(self, record: dict) -> None:
        if self._log_file is None:
            return
        with self._log_lock:
            self._log_file.write(json.dumps(record, sort_keys=True) + "\n")
            self._log_file.flush()

    def close(self) -> None:
        """Stop the solver thread and close the optional shadow log."""

        self._stop_event.set()
        self._worker.join(timeout=10.0)
        if self._worker.is_alive():
            self.get_logger().warning("WB-MPC worker did not stop within 10 seconds")
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None


def _finite_or_none(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _load_configs(path_text: str) -> tuple[dict, dict]:
    path = Path(path_text).expanduser().resolve()
    with path.open(encoding="utf-8") as stream:
        runner = yaml.safe_load(stream)
    controller_path = Path(runner["stretch_wbmpc_shadow"]["controller_config"])
    if not controller_path.is_absolute():
        controller_path = path.parent / controller_path
    controller = parsing.load_config(str(controller_path.resolve()))
    return runner, controller


def main(argv=None) -> None:
    """Run the WB-MPC source node; hardware output is intentionally impossible."""

    argv = sys.argv if argv is None else argv
    rclpy.init(args=argv)
    parsed_argv = remove_ros_args(args=argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--log", help="Override the JSONL shadow log path.")
    parser.add_argument("--duration", type=float, default=0.0)
    args = parser.parse_args(parsed_argv[1:])

    runner_config, controller_config = _load_configs(args.config)
    node = StretchWBMPCShadow(runner_config, controller_config, log_path=args.log)
    started = time.monotonic()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.2)
            if args.duration > 0.0 and time.monotonic() - started >= args.duration:
                break
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
