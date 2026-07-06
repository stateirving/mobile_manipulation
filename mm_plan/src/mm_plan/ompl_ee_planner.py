"""OMPL-backed end-effector path planner for MPC reference tracking."""

import math

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from mm_plan.Planners import Planner
from mm_utils import parsing
from mm_utils.enums import RefType
from mm_utils.math import interpolate


def _wrap_pi(values):
    values = np.asarray(values, dtype=np.float64)
    return (values + np.pi) % (2.0 * np.pi) - np.pi


class OMPLEEPlanner(Planner):
    """Plan a Cartesian EE path with OMPL and expose it as MPC references.

    This planner is intentionally only a reference generator. OMPL searches in
    Cartesian EE position space, while the existing MPC still tracks the path
    with the full robot model and applies whole-body ESDF costs/constraints.
    """

    def __init__(self, config, resources=None):
        super().__init__(name=config["name"], ref_type=RefType.PATH)

        resources = {} if resources is None else resources
        self.esdf_map = resources.get("esdf_map")
        self.robot_model = resources.get("robot_model")

        self.has_base_ref = False
        self.base_mask = np.ones(3, dtype=bool)
        self.base_target = None

        self.has_ee_ref = True
        self.ee_target = self._parse_ee_target(config)
        self.ee_mask = np.array(config.get("ee_mask", [True] * 6), dtype=bool)
        if self.ee_mask.shape != (6,):
            raise ValueError("ee_mask must have shape (6,)")

        self.tracking_pos_err_tol = float(
            config.get("tracking_pos_err_tol", 0.05)
        )
        self.tracking_ori_err_tol = float(
            config.get("tracking_ori_err_tol", 0.2)
        )
        self.hold_period = float(config.get("hold_period", 0.0))
        self.end_stop = bool(config.get("end_stop", False))

        self.bounds_xyz = self._parse_bounds_xyz(config)
        self.auto_expand_bounds = bool(config.get("auto_expand_bounds", True))
        self.bounds_margin = float(config.get("bounds_margin", 0.25))

        ompl_config = dict(config.get("ompl", {}))
        self.planner_name = str(ompl_config.get("planner", "RRTConnect"))
        self.solve_time = float(ompl_config.get("solve_time", 0.5))
        self.goal_tolerance = float(ompl_config.get("goal_tolerance", 0.04))
        self.planner_range = ompl_config.get("range")
        self.simplify = bool(ompl_config.get("simplify", True))
        self.simplify_time = float(ompl_config.get("simplify_time", 0.05))
        self.state_validity_resolution = ompl_config.get(
            "state_validity_resolution", 0.01
        )

        path_config = dict(config.get("path", {}))
        self.dt = float(path_config.get("dt", config.get("dt", 0.1)))
        self.linear_speed = float(path_config.get("linear_speed", 0.12))
        self.angular_speed = float(path_config.get("angular_speed", 0.8))
        self.interpolation_resolution = float(
            path_config.get("interpolation_resolution", 0.04)
        )

        esdf_config = dict(config.get("esdf", {}))
        self.esdf_enabled = bool(esdf_config.get("enabled", True))
        self.tool_radius = float(
            esdf_config.get("tool_radius", esdf_config.get("ee_radius", 0.08))
        )
        self.d_safe = float(esdf_config.get("d_safe", 0.05))
        self.unknown_is_valid = bool(esdf_config.get("unknown_is_valid", True))
        self.allow_fallback = bool(
            esdf_config.get("allow_straight_line_fallback", True)
        )
        self.collision_offsets = self._parse_collision_offsets(esdf_config)
        self._warned_no_esdf = False

        replan_config = dict(config.get("replan", {}))
        self.replan_enabled = bool(replan_config.get("enabled", False))
        self.replan_min_interval = float(
            replan_config.get("min_interval", 1.0)
        )
        self.replan_check_horizon = float(
            replan_config.get("check_horizon", 2.0)
        )
        self.replan_check_dt = float(replan_config.get("check_dt", 0.2))
        self.replan_min_clearance = float(
            replan_config.get("min_clearance", 0.0)
        )
        self.replan_validate_every_step = bool(
            replan_config.get("validate_every_step", False)
        )
        self.replan_validate_remaining_path = bool(
            replan_config.get("validate_remaining_path", False)
        )
        self.replan_deviation_threshold = float(
            replan_config.get("deviation_threshold", 0.25)
        )
        self.replan_force_periodic = bool(
            replan_config.get("force_periodic", False)
        )
        self.keep_previous_on_replan_failure = bool(
            replan_config.get("keep_previous_on_failure", True)
        )
        self.last_replan_time = None
        self.last_replan_reason = "not_planned"
        self.replan_count = 0

        self.ee_plan = self._make_plan(
            np.asarray([self.ee_target[:3]], dtype=np.float64),
            self.ee_target[3:],
            self.ee_target[3:],
        )
        self.finished = False
        self.target_reached = False
        self.t_reached = 0.0
        self.planned = False
        self.start_time = 0.0

    def set_resources(self, resources):
        """Attach runtime resources after construction."""
        if resources is None:
            return
        if self.esdf_map is None and resources.get("esdf_map") is not None:
            self.esdf_map = resources["esdf_map"]
        if self.robot_model is None and resources.get("robot_model") is not None:
            self.robot_model = resources["robot_model"]

    def ready(self):
        return True

    def activate(self):
        self.started = True

    def updateRobotStates(self, robot_states):
        super().updateRobotStates(robot_states)

    def updatePlanningContext(self, t, robot_states, num_pts=None, dt=None):
        """Plan or replan against the latest ESDF before references."""
        if not self.planned:
            self._ensure_plan(robot_states, t=t, reason="initial")
            return
        if not self.replan_enabled:
            return
        if self.replan_validate_every_step:
            reason = self._replan_reason(
                t, robot_states, num_pts, dt, allow_periodic=False
            )
            if reason is not None:
                self._attempt_replan(robot_states, t, reason)
            return
        if self.last_replan_time is not None:
            elapsed = t - self.last_replan_time
            if elapsed < self.replan_min_interval:
                return

        reason = self._replan_reason(t, robot_states, num_pts, dt)
        if reason is None:
            return
        self._attempt_replan(robot_states, t, reason)

    def _attempt_replan(self, robot_states, t, reason):
        try:
            self._set_plan_from_current_state(robot_states, t, reason)
        except Exception as exc:
            if not self.keep_previous_on_replan_failure:
                raise
            self.last_replan_time = t
            self.py_logger.warning(
                "%s replan failed (%s); keeping previous EE plan",
                self.name,
                exc,
            )

    def getEETrackingPoint(self, t, robot_states=None):
        if robot_states is not None:
            self._ensure_plan(robot_states, reason="direct_point_request")
        te = t - self.start_time if self.started else 0.0
        return interpolate(te, self.ee_plan)

    def getEETrackingPointArray(
        self, robot_states, num_pts, dt, time_offset=0
    ):
        self._ensure_plan(robot_states, reason="direct_array_request")
        times = time_offset + np.arange(num_pts) * dt
        positions = np.array(
            [interpolate(t, self.ee_plan)[0] for t in times]
        )
        velocities = np.array(
            [interpolate(t, self.ee_plan)[1] for t in times]
        )
        return positions, velocities

    def checkFinished(self, t, states):
        ee_pose = np.asarray(states["EE"]["pose"], dtype=np.float64)
        end_pose = self.ee_target

        pos_mask = self.ee_mask[:3]
        pos_err = np.linalg.norm((ee_pose[:3] - end_pose[:3])[pos_mask])
        pos_finished = pos_err < self.tracking_pos_err_tol

        ori_mask = self.ee_mask[3:]
        ori_err = np.linalg.norm(_wrap_pi(ee_pose[3:] - end_pose[3:])[ori_mask])
        ori_finished = ori_err < self.tracking_ori_err_tol

        ee_finished = pos_finished and ori_finished
        if self.end_stop:
            ee_vel = states["EE"].get("velocity")
            ee_finished = (
                ee_finished
                and ee_vel is not None
                and np.linalg.norm(ee_vel) < 1e-2
            )

        if ee_finished:
            if not self.target_reached:
                self.target_reached = True
                self.t_reached = float(t)
                self.py_logger.info("%s reached target", self.name)
            if self.hold_period > 0.0 and (t - self.t_reached) < self.hold_period:
                self.finished = False
                return False
        else:
            self.target_reached = False
            self.t_reached = 0.0

        self.finished = ee_finished
        if self.finished:
            self.py_logger.info("%s finished", self.name)
        return self.finished

    def reset(self):
        self.finished = False
        self.target_reached = False
        self.t_reached = 0.0
        self.started = False
        self.planned = False
        self.start_time = 0.0
        self.ee_plan = self._make_plan(
            np.asarray([self.ee_target[:3]], dtype=np.float64),
            self.ee_target[3:],
            self.ee_target[3:],
        )
        self.last_replan_time = None
        self.last_replan_reason = "reset"
        self.replan_count = 0

    def _ensure_plan(self, robot_states, t=None, reason="initial"):
        if self.planned:
            return
        if t is None:
            t = self.start_time
        self._set_plan_from_current_state(robot_states, t, reason)

    def _set_plan_from_current_state(self, robot_states, t, reason):
        start_pose = self._ee_pose_from_robot_states(robot_states)
        path = self._plan_with_ompl(start_pose[:3], self.ee_target[:3])
        self.ee_plan = self._make_plan(path, start_pose[3:], self.ee_target[3:])
        self.planned = True
        self.start_time = float(t)
        self.last_replan_time = float(t)
        self.last_replan_reason = str(reason)
        self.replan_count += 1
        self.py_logger.info(
            "%s planned %d EE waypoints to %s (%s, count=%d)",
            self.name,
            len(self.ee_plan["p"]),
            np.array2string(self.ee_target, precision=3),
            reason,
            self.replan_count,
        )

    def _replan_reason(
        self, t, robot_states, num_pts=None, dt=None, allow_periodic=True
    ):
        if allow_periodic and self.replan_force_periodic:
            return "periodic"

        current_ee = self._ee_pose_from_robot_states(robot_states)
        deviation = self._path_deviation(current_ee[:3])
        if deviation > self.replan_deviation_threshold:
            return f"ee_path_deviation_{deviation:.3f}"

        if self.replan_validate_remaining_path:
            min_clearance = self._remaining_path_min_clearance(t)
        else:
            min_clearance = self._future_path_min_clearance(t, num_pts, dt)
        if min_clearance < self.replan_min_clearance:
            return f"ee_path_clearance_{min_clearance:.3f}"
        return None

    def _path_deviation(self, current_position):
        if self.ee_plan is None or len(self.ee_plan["p"]) == 0:
            return np.inf
        path_xyz = np.asarray(self.ee_plan["p"], dtype=np.float64)[:, :3]
        distances = np.linalg.norm(path_xyz - current_position[:3], axis=1)
        return float(np.min(distances))

    def _future_path_min_clearance(self, t, num_pts=None, dt=None):
        if not self.esdf_enabled:
            return np.inf
        if self.esdf_map is None:
            return np.inf
        if self.ee_plan is None or len(self.ee_plan["p"]) == 0:
            return -np.inf

        time_offset = t - self.start_time if self.started else 0.0
        horizon = self.replan_check_horizon
        if num_pts is not None and dt is not None:
            horizon = min(horizon, max(0.0, (int(num_pts) - 1) * float(dt)))
        check_dt = max(self.replan_check_dt, 1.0e-3)
        times = time_offset + np.arange(0.0, horizon + 1.0e-9, check_dt)
        if times.size == 0:
            times = np.array([time_offset], dtype=np.float64)

        min_clearance = np.inf
        for query_time in times:
            state = interpolate(query_time, self.ee_plan)[0]
            min_clearance = min(
                min_clearance, self._state_clearance(state[:3])
            )
        return float(min_clearance)

    def _remaining_path_min_clearance(self, t):
        if not self.esdf_enabled:
            return np.inf
        if self.esdf_map is None:
            return np.inf
        if self.ee_plan is None or len(self.ee_plan["p"]) == 0:
            return -np.inf

        time_offset = t - self.start_time if self.started else 0.0
        path_end = float(self.ee_plan["t"][-1])
        time_offset = min(max(0.0, time_offset), path_end)
        check_dt = max(self.replan_check_dt, 1.0e-3)
        times = np.arange(time_offset, path_end + 1.0e-9, check_dt)
        if times.size == 0 or times[-1] < path_end:
            times = np.append(times, path_end)

        min_clearance = np.inf
        for query_time in times:
            state = interpolate(query_time, self.ee_plan)[0]
            min_clearance = min(
                min_clearance, self._state_clearance(state[:3])
            )
        return float(min_clearance)

    def _plan_with_ompl(self, start, goal):
        try:
            return self._solve_ompl(start, goal)
        except Exception as exc:
            if not self.allow_fallback:
                raise
            self.py_logger.warning(
                "%s OMPL EE planning failed (%s); using straight-line fallback",
                self.name,
                exc,
            )
            return self._straight_line_path(start, goal)

    def _solve_ompl(self, start, goal):
        from ompl import base as ob
        from ompl import geometric as og

        space = ob.RealVectorStateSpace(3)
        bounds_xyz = self._bounds_containing(start, goal)
        bounds = ob.RealVectorBounds(3)
        for idx in range(3):
            bounds.setLow(idx, float(bounds_xyz[idx, 0]))
            bounds.setHigh(idx, float(bounds_xyz[idx, 1]))
        space.setBounds(bounds)

        setup = og.SimpleSetup(space)
        checker = _EEStateValidityChecker(setup.getSpaceInformation(), self)
        setup.setStateValidityChecker(checker)
        if self.state_validity_resolution is not None:
            setup.getSpaceInformation().setStateValidityCheckingResolution(
                float(self.state_validity_resolution)
            )

        start_state = space.allocState()
        goal_state = space.allocState()
        for idx in range(3):
            start_state[idx] = float(start[idx])
            goal_state[idx] = float(goal[idx])
        setup.setStartAndGoalStates(
            start_state, goal_state, self.goal_tolerance
        )

        planner = self._make_ompl_planner(og, setup.getSpaceInformation())
        if self.planner_range is not None and hasattr(planner, "setRange"):
            planner.setRange(float(self.planner_range))
        setup.setPlanner(planner)

        solved = setup.solve(self.solve_time)
        if not bool(solved):
            raise RuntimeError(f"OMPL {self.planner_name} found no EE solution")

        if self.simplify:
            setup.simplifySolution(self.simplify_time)
        raw_path = self._extract_solution_path(setup.getSolutionPath())
        if len(raw_path) < 2:
            raise RuntimeError("OMPL EE solution path has fewer than two states")
        self._ensure_goal_reached(raw_path[-1], goal)
        return self._densify_path(raw_path)

    def _make_ompl_planner(self, og, space_information):
        planner_cls = getattr(og, self.planner_name, None)
        if planner_cls is None:
            available = ["RRTConnect", "RRT", "RRTstar", "PRM", "BITstar"]
            raise ValueError(
                f"Unknown OMPL geometric planner '{self.planner_name}'. "
                f"Try one of {available}."
            )
        return planner_cls(space_information)

    def _extract_solution_path(self, solution_path):
        points = []
        for idx in range(solution_path.getStateCount()):
            state = solution_path.getState(idx)
            points.append([state[0], state[1], state[2]])
        return np.asarray(points, dtype=np.float64)

    def _ensure_goal_reached(self, endpoint, goal):
        endpoint = np.asarray(endpoint, dtype=np.float64)
        goal = np.asarray(goal, dtype=np.float64)
        position_error = float(np.linalg.norm(endpoint[:3] - goal[:3]))
        if position_error <= self.goal_tolerance:
            return
        raise RuntimeError(
            f"OMPL {self.planner_name} EE solution endpoint is "
            f"{position_error:.3f} m from goal, exceeding "
            f"goal_tolerance {self.goal_tolerance:.3f}"
        )

    def _is_state_valid(self, state):
        return bool(self._state_clearance(state) >= 0.0)

    def _state_clearance(self, state):
        if not self.esdf_enabled:
            return np.inf
        if self.esdf_map is None:
            if not self._warned_no_esdf:
                self.py_logger.warning(
                    "%s has no ESDF map resource; OMPL EE validity is free-space",
                    self.name,
                )
                self._warned_no_esdf = True
            return np.inf

        points = self._collision_points_world(state)
        distances, _, valid = self.esdf_map.query(points)
        distances = np.asarray(distances, dtype=np.float64)
        valid = np.asarray(valid, dtype=bool)

        if not np.all(valid):
            if not self.unknown_is_valid:
                return -np.inf
            if not np.any(valid):
                return np.inf

        clearance = distances[valid] - self.tool_radius - self.d_safe
        return float(np.min(clearance))

    def _collision_points_world(self, state):
        position = np.asarray(state, dtype=np.float64).reshape(3)
        return position[None, :] + self.collision_offsets

    def _densify_path(self, raw_path):
        if len(raw_path) <= 1:
            return raw_path

        points = [raw_path[0].copy()]
        for start, goal in zip(raw_path[:-1], raw_path[1:]):
            dist = float(np.linalg.norm(goal[:3] - start[:3]))
            steps = max(
                1, int(math.ceil(dist / self.interpolation_resolution))
            )
            for idx in range(1, steps + 1):
                alpha = idx / steps
                point = (1.0 - alpha) * start[:3] + alpha * goal[:3]
                points.append(point)
        return np.asarray(points, dtype=np.float64)

    def _straight_line_path(self, start, goal):
        raw_path = np.asarray([start, goal], dtype=np.float64)
        return self._densify_path(raw_path)

    def _make_plan(self, path_xyz, start_orientation, goal_orientation):
        path_xyz = np.asarray(path_xyz, dtype=np.float64)
        if path_xyz.ndim != 2 or path_xyz.shape[1] != 3:
            raise ValueError("EE position path must have shape (N, 3)")

        start_orientation = np.asarray(start_orientation, dtype=np.float64)
        goal_orientation = np.asarray(goal_orientation, dtype=np.float64)
        if start_orientation.shape != (3,) or goal_orientation.shape != (3,):
            raise ValueError("EE orientations must have shape (3,)")

        if len(path_xyz) == 1:
            pose = np.hstack((path_xyz, goal_orientation[None, :]))
            return {
                "t": np.array([0.0], dtype=np.float64),
                "p": pose,
                "v": np.zeros_like(pose),
            }

        cumulative = np.zeros(len(path_xyz), dtype=np.float64)
        if len(path_xyz) > 1:
            cumulative[1:] = np.cumsum(
                np.linalg.norm(np.diff(path_xyz, axis=0), axis=1)
            )
        if cumulative[-1] > 1.0e-9:
            alpha = cumulative / cumulative[-1]
        else:
            alpha = np.linspace(0.0, 1.0, len(path_xyz))
        orientation_delta = _wrap_pi(goal_orientation - start_orientation)
        orientations = (
            start_orientation[None, :] + alpha[:, None] * orientation_delta[None, :]
        )
        orientations = _wrap_pi(orientations)
        path = np.hstack((path_xyz, orientations))

        times = [0.0]
        for p0, p1 in zip(path[:-1], path[1:]):
            dist = float(np.linalg.norm(p1[:3] - p0[:3]))
            ori_dist = float(np.linalg.norm(_wrap_pi(p1[3:] - p0[3:])))
            duration = max(
                dist / max(self.linear_speed, 1.0e-6),
                ori_dist / max(self.angular_speed, 1.0e-6),
                self.dt,
            )
            times.append(times[-1] + duration)
        times = np.asarray(times, dtype=np.float64)

        velocities = np.zeros_like(path)
        dt = np.diff(times)
        dp = np.diff(path, axis=0)
        dp[:, 3:] = _wrap_pi(dp[:, 3:])
        velocities[:-1] = dp / dt[:, None]
        return {"t": times, "p": path.copy(), "v": velocities}

    def _bounds_containing(self, start, goal):
        bounds = self.bounds_xyz.copy()
        if self.auto_expand_bounds:
            for idx in range(3):
                bounds[idx, 0] = (
                    min(bounds[idx, 0], start[idx], goal[idx]) - self.bounds_margin
                )
                bounds[idx, 1] = (
                    max(bounds[idx, 1], start[idx], goal[idx]) + self.bounds_margin
                )
        return bounds

    def _ee_pose_from_robot_states(self, robot_states):
        if self.robot_model is None:
            raise RuntimeError(
                "OMPLEEPlanner requires a robot_model resource for current EE FK"
            )
        if robot_states is None:
            raise RuntimeError("OMPLEEPlanner requires current robot_states")
        q = np.asarray(robot_states[0], dtype=np.float64)
        ee_position, ee_quat = self.robot_model.getEE(q)
        ee_euler = Rot.from_quat(
            np.asarray(ee_quat, dtype=np.float64)
        ).as_euler("xyz")
        return np.hstack((np.asarray(ee_position, dtype=np.float64), ee_euler))

    def _parse_ee_target(self, config):
        target = config.get("ee_pose", config.get("ee_goal"))
        if target is None:
            raise ValueError("OMPLEEPlanner requires ee_pose or ee_goal")
        target = parsing.parse_array(target)
        if len(target) != 6:
            raise ValueError("ee_pose must be SE3 [x, y, z, roll, pitch, yaw]")
        return np.asarray(target, dtype=np.float64)

    def _parse_bounds_xyz(self, config):
        raw = config.get(
            "bounds_xyz", [[0.0, 3.5], [-2.5, 1.0], [0.25, 1.3]]
        )
        bounds = np.asarray(raw, dtype=np.float64)
        if bounds.shape != (3, 2):
            raise ValueError(
                "bounds_xyz must be [[xmin, xmax], [ymin, ymax], [zmin, zmax]]"
            )
        if np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("bounds_xyz lower limits must be < upper limits")
        return bounds

    def _parse_collision_offsets(self, esdf_config):
        raw_offsets = esdf_config.get(
            "collision_offsets", esdf_config.get("collision_points")
        )
        if raw_offsets is not None:
            offsets = np.asarray(raw_offsets, dtype=np.float64)
            if offsets.ndim != 2 or offsets.shape[1] != 3:
                raise ValueError("collision_offsets must have shape (N, 3)")
            return offsets
        return np.zeros((1, 3), dtype=np.float64)


class _EEStateValidityChecker:
    """Small adapter around OMPL's Python StateValidityChecker base class."""

    def __new__(cls, space_information, planner):
        from ompl import base as ob

        class Checker(ob.StateValidityChecker):
            def __init__(self, si, parent):
                super().__init__(si)
                self.parent = parent

            def isValid(self, state):
                return self.parent._is_state_valid(
                    [state[0], state[1], state[2]]
                )

        return Checker(space_information, planner)
