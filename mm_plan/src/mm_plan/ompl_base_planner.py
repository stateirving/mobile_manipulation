"""OMPL-backed base path planner for the existing MPC reference interface."""

import math

import numpy as np

from mm_plan.Planners import Planner
from mm_utils import parsing
from mm_utils.enums import RefType
from mm_utils.math import interpolate, wrap_pi_array, wrap_pi_scalar


class OMPLBasePlanner(Planner):
    """Plan a base SE(2) path with OMPL and expose it as MPC references.

    The planner deliberately stays on the `mm_plan` side of the existing
    architecture: it produces a base path, while MPC still tracks the path with
    the full robot model and applies local ESDF collision costs/constraints.
    """

    def __init__(self, config, resources=None):
        super().__init__(name=config["name"], ref_type=RefType.PATH)

        resources = {} if resources is None else resources
        self.esdf_map = resources.get("esdf_map")
        self.has_base_ref = True
        self.base_target = self._parse_base_target(config)
        self.base_mask = np.array(
            config.get("base_mask", [True, True, False]), dtype=bool
        )
        if self.base_mask.shape != (3,):
            raise ValueError("base_mask must have shape (3,)")

        self.ee_target = None
        self.ee_mask = np.ones(6, dtype=bool)
        self.has_ee_ref = False
        if "ee_pose" in config:
            self.ee_target = parsing.parse_array(config["ee_pose"])
            if len(self.ee_target) != 6:
                raise ValueError("ee_pose must be SE3 [x, y, z, r, p, y]")
            self.ee_mask = np.array(
                config.get("ee_mask", [True] * 6), dtype=bool
            )
            if self.ee_mask.shape != (6,):
                raise ValueError("ee_mask must have shape (6,)")
            self.has_ee_ref = True

        self.tracking_pos_err_tol = float(
            config.get("tracking_pos_err_tol", 0.15)
        )
        self.tracking_ori_err_tol = float(
            config.get("tracking_ori_err_tol", 0.2)
        )
        self.end_stop = bool(config.get("end_stop", False))

        self.bounds_xy = self._parse_bounds_xy(config)
        self.auto_expand_bounds = bool(config.get("auto_expand_bounds", True))
        self.bounds_margin = float(config.get("bounds_margin", 0.25))

        ompl_config = dict(config.get("ompl", {}))
        self.planner_name = str(ompl_config.get("planner", "RRTConnect"))
        self.solve_time = float(ompl_config.get("solve_time", 0.5))
        self.goal_tolerance = float(ompl_config.get("goal_tolerance", 0.05))
        self.planner_range = ompl_config.get("range")
        self.simplify = bool(ompl_config.get("simplify", True))
        self.simplify_time = float(ompl_config.get("simplify_time", 0.05))
        self.state_validity_resolution = ompl_config.get(
            "state_validity_resolution", 0.01
        )

        path_config = dict(config.get("path", {}))
        self.dt = float(path_config.get("dt", config.get("dt", 0.1)))
        self.linear_speed = float(path_config.get("linear_speed", 0.25))
        self.angular_speed = float(path_config.get("angular_speed", 0.8))
        self.interpolation_resolution = float(
            path_config.get("interpolation_resolution", 0.05)
        )
        self.yaw_mode = str(path_config.get("yaw_mode", "tangent")).lower()

        esdf_config = dict(config.get("esdf", {}))
        self.esdf_enabled = bool(esdf_config.get("enabled", True))
        self.base_radius = float(esdf_config.get("base_radius", 0.35))
        self.d_safe = float(esdf_config.get("d_safe", 0.05))
        self.unknown_is_valid = bool(esdf_config.get("unknown_is_valid", True))
        self.allow_fallback = bool(
            esdf_config.get("allow_straight_line_fallback", True)
        )
        self.collision_points = self._parse_collision_points(esdf_config)
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
        self.replan_deviation_threshold = float(
            replan_config.get("deviation_threshold", 0.4)
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

        self.base_plan = self._make_plan(
            np.asarray([self.base_target], dtype=np.float64)
        )
        self.finished = False
        self.planned = False
        self.start_time = 0.0

    def set_resources(self, resources):
        """Attach runtime resources after construction."""
        if resources is None:
            return
        if self.esdf_map is None and resources.get("esdf_map") is not None:
            self.esdf_map = resources["esdf_map"]

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
        if self.last_replan_time is not None:
            elapsed = t - self.last_replan_time
            if elapsed < self.replan_min_interval:
                return

        reason = self._replan_reason(t, robot_states, num_pts, dt)
        if reason is None:
            return
        try:
            self._set_plan_from_current_state(robot_states, t, reason)
        except Exception as exc:
            if not self.keep_previous_on_replan_failure:
                raise
            self.last_replan_time = t
            self.py_logger.warning(
                "%s replan failed (%s); keeping previous plan",
                self.name,
                exc,
            )

    def getBaseTrackingPoint(self, t, robot_states=None):
        if robot_states is not None:
            self._ensure_plan(robot_states, reason="direct_point_request")
        te = t - self.start_time if self.started else 0.0
        return interpolate(te, self.base_plan)

    def getBaseTrackingPointArray(
        self, robot_states, num_pts, dt, time_offset=0
    ):
        self._ensure_plan(robot_states, reason="direct_array_request")
        times = time_offset + np.arange(num_pts) * dt
        positions = np.array(
            [interpolate(t, self.base_plan)[0] for t in times]
        )
        velocities = np.array(
            [interpolate(t, self.base_plan)[1] for t in times]
        )
        return positions, velocities

    def getEETrackingPoint(self, t, robot_states=None):
        if not self.has_ee_ref:
            return None, None
        return self.ee_target.copy(), np.zeros(6)

    def getEETrackingPointArray(
        self, robot_states, num_pts, dt, time_offset=0
    ):
        if not self.has_ee_ref:
            return None, None
        return (
            np.tile(self.ee_target, (num_pts, 1)),
            np.zeros((num_pts, 6), dtype=np.float64),
        )

    def checkFinished(self, t, states):
        base_pose = np.asarray(states["base"]["pose"], dtype=np.float64)
        end_pose = self.base_target

        pos_mask = self.base_mask[:2]
        pos_err = np.linalg.norm((base_pose[:2] - end_pose[:2])[pos_mask])
        pos_finished = pos_err < self.tracking_pos_err_tol
        if self.base_mask[2]:
            yaw_err = abs(wrap_pi_scalar(base_pose[2] - end_pose[2]))
            yaw_finished = yaw_err < self.tracking_ori_err_tol
        else:
            yaw_finished = True

        base_finished = pos_finished and yaw_finished
        if self.end_stop:
            base_vel = states["base"].get("velocity")
            base_finished = (
                base_finished
                and base_vel is not None
                and np.linalg.norm(base_vel) < 1e-2
            )

        ee_finished = True
        if self.has_ee_ref:
            ee_pose = np.asarray(states["EE"]["pose"], dtype=np.float64)
            pos_mask = self.ee_mask[:3]
            ori_mask = self.ee_mask[3:]
            pos_err = np.linalg.norm(
                (ee_pose[:3] - self.ee_target[:3])[pos_mask]
            )
            ori_err = np.linalg.norm(
                wrap_pi_array(ee_pose[3:] - self.ee_target[3:])[ori_mask]
            )
            ee_finished = (
                pos_err < self.tracking_pos_err_tol
                and ori_err < self.tracking_ori_err_tol
            )

        self.finished = base_finished and ee_finished
        if self.finished:
            self.py_logger.info("%s finished", self.name)
        return self.finished

    def reset(self):
        self.finished = False
        self.started = False
        self.planned = False
        self.start_time = 0.0
        self.base_plan = self._make_plan(
            np.asarray([self.base_target], dtype=np.float64)
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
        start = self._base_pose_from_robot_states(robot_states)
        path = self._plan_with_ompl(start, self.base_target)
        self.base_plan = self._make_plan(path)
        self.planned = True
        self.start_time = float(t)
        self.last_replan_time = float(t)
        self.last_replan_reason = str(reason)
        self.replan_count += 1
        self.py_logger.info(
            "%s planned %d base waypoints to %s (%s, count=%d)",
            self.name,
            len(self.base_plan["p"]),
            np.array2string(self.base_target, precision=3),
            reason,
            self.replan_count,
        )

    def _replan_reason(self, t, robot_states, num_pts=None, dt=None):
        if self.replan_force_periodic:
            return "periodic"

        current_base = self._base_pose_from_robot_states(robot_states)
        deviation = self._path_deviation(current_base)
        if deviation > self.replan_deviation_threshold:
            return f"path_deviation_{deviation:.3f}"

        min_clearance = self._future_path_min_clearance(t, num_pts, dt)
        if min_clearance < self.replan_min_clearance:
            return f"path_clearance_{min_clearance:.3f}"
        return None

    def _path_deviation(self, current_base):
        if self.base_plan is None or len(self.base_plan["p"]) == 0:
            return np.inf
        path_xy = np.asarray(self.base_plan["p"], dtype=np.float64)[:, :2]
        distances = np.linalg.norm(path_xy - current_base[:2], axis=1)
        return float(np.min(distances))

    def _future_path_min_clearance(self, t, num_pts=None, dt=None):
        if not self.esdf_enabled:
            return np.inf
        if self.esdf_map is None:
            return np.inf
        if self.base_plan is None or len(self.base_plan["p"]) == 0:
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
            state = interpolate(query_time, self.base_plan)[0]
            min_clearance = min(
                min_clearance, self._state_clearance(state)
            )
        return float(min_clearance)

    def _plan_with_ompl(self, start, goal):
        try:
            return self._solve_ompl(start, goal)
        except Exception as exc:
            if not self.allow_fallback:
                raise
            self.py_logger.warning(
                "%s OMPL planning failed (%s); using straight-line fallback",
                self.name,
                exc,
            )
            return self._straight_line_path(start, goal)

    def _solve_ompl(self, start, goal):
        from ompl import base as ob
        from ompl import geometric as og

        space = ob.SE2StateSpace()
        bounds_xy = self._bounds_containing(start, goal)
        bounds = ob.RealVectorBounds(2)
        bounds.setLow(0, float(bounds_xy[0, 0]))
        bounds.setHigh(0, float(bounds_xy[0, 1]))
        bounds.setLow(1, float(bounds_xy[1, 0]))
        bounds.setHigh(1, float(bounds_xy[1, 1]))
        space.setBounds(bounds)

        setup = og.SimpleSetup(space)
        checker = _ESDFStateValidityChecker(setup.getSpaceInformation(), self)
        setup.setStateValidityChecker(checker)
        if self.state_validity_resolution is not None:
            setup.getSpaceInformation().setStateValidityCheckingResolution(
                float(self.state_validity_resolution)
            )

        start_state = space.allocState()
        start_state.setX(float(start[0]))
        start_state.setY(float(start[1]))
        start_state.setYaw(float(start[2]))
        goal_state = space.allocState()
        goal_state.setX(float(goal[0]))
        goal_state.setY(float(goal[1]))
        goal_state.setYaw(float(goal[2]))
        setup.setStartAndGoalStates(
            start_state, goal_state, self.goal_tolerance
        )

        planner = self._make_ompl_planner(og, setup.getSpaceInformation())
        if self.planner_range is not None and hasattr(planner, "setRange"):
            planner.setRange(float(self.planner_range))
        setup.setPlanner(planner)

        solved = setup.solve(self.solve_time)
        if not bool(solved):
            raise RuntimeError(f"OMPL {self.planner_name} found no solution")

        if self.simplify:
            setup.simplifySolution(self.simplify_time)
        raw_path = self._extract_solution_path(setup.getSolutionPath())
        if len(raw_path) < 2:
            raise RuntimeError("OMPL solution path has fewer than two states")
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
            points.append([state.getX(), state.getY(), state.getYaw()])
        return np.asarray(points, dtype=np.float64)

    def _is_state_valid(self, state):
        return bool(self._state_clearance(state) >= 0.0)

    def _state_clearance(self, state):
        if not self.esdf_enabled:
            return np.inf
        if self.esdf_map is None:
            if not self._warned_no_esdf:
                self.py_logger.warning(
                    "%s has no ESDF map resource; OMPL validity is free-space",
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

        clearance = distances[valid] - self.base_radius - self.d_safe
        return float(np.min(clearance))

    def _collision_points_world(self, state):
        x, y, yaw = float(state[0]), float(state[1]), float(state[2])
        c, s = math.cos(yaw), math.sin(yaw)
        points = []
        for ox, oy, z in self.collision_points:
            points.append([x + c * ox - s * oy, y + s * ox + c * oy, z])
        return np.asarray(points, dtype=np.float64)

    def _densify_path(self, raw_path):
        if len(raw_path) <= 1:
            return raw_path

        points = [raw_path[0].copy()]
        for start, goal in zip(raw_path[:-1], raw_path[1:]):
            dist = float(np.linalg.norm(goal[:2] - start[:2]))
            steps = max(
                1, int(math.ceil(dist / self.interpolation_resolution))
            )
            yaw_delta = wrap_pi_scalar(goal[2] - start[2])
            for idx in range(1, steps + 1):
                alpha = idx / steps
                point = np.empty(3, dtype=np.float64)
                point[:2] = (1.0 - alpha) * start[:2] + alpha * goal[:2]
                point[2] = start[2] + alpha * yaw_delta
                points.append(point)

        path = np.asarray(points, dtype=np.float64)
        if self.yaw_mode == "tangent":
            path[:, 2] = self._path_tangent_yaw(path, self.base_target[2])
        path[:, 2] = wrap_pi_array(path[:, 2])
        return path

    def _straight_line_path(self, start, goal):
        raw_path = np.asarray([start, goal], dtype=np.float64)
        return self._densify_path(raw_path)

    def _make_plan(self, path):
        path = np.asarray(path, dtype=np.float64)
        if path.ndim != 2 or path.shape[1] != 3:
            raise ValueError("base path must have shape (N, 3)")
        if len(path) == 1:
            return {
                "t": np.array([0.0], dtype=np.float64),
                "p": path.copy(),
                "v": np.zeros_like(path),
            }

        times = [0.0]
        for p0, p1 in zip(path[:-1], path[1:]):
            dist = float(np.linalg.norm(p1[:2] - p0[:2]))
            yaw_dist = abs(wrap_pi_scalar(p1[2] - p0[2]))
            duration = max(
                dist / max(self.linear_speed, 1.0e-6),
                yaw_dist / max(self.angular_speed, 1.0e-6),
                self.dt,
            )
            times.append(times[-1] + duration)
        times = np.asarray(times, dtype=np.float64)

        velocities = np.zeros_like(path)
        dt = np.diff(times)
        dp = np.diff(path, axis=0)
        dp[:, 2] = [wrap_pi_scalar(v) for v in dp[:, 2]]
        velocities[:-1] = dp / dt[:, None]
        return {"t": times, "p": path.copy(), "v": velocities}

    def _path_tangent_yaw(self, path, goal_yaw):
        yaw = np.zeros(len(path), dtype=np.float64)
        for idx in range(len(path) - 1):
            delta = path[idx + 1, :2] - path[idx, :2]
            if np.linalg.norm(delta) > 1.0e-9:
                yaw[idx] = math.atan2(delta[1], delta[0])
            elif idx > 0:
                yaw[idx] = yaw[idx - 1]
            else:
                yaw[idx] = path[idx, 2]
        yaw[-1] = goal_yaw if self.base_mask[2] else yaw[-2]
        return yaw

    def _bounds_containing(self, start, goal):
        bounds = self.bounds_xy.copy()
        if self.auto_expand_bounds:
            bounds[0, 0] = (
                min(bounds[0, 0], start[0], goal[0]) - self.bounds_margin
            )
            bounds[0, 1] = (
                max(bounds[0, 1], start[0], goal[0]) + self.bounds_margin
            )
            bounds[1, 0] = (
                min(bounds[1, 0], start[1], goal[1]) - self.bounds_margin
            )
            bounds[1, 1] = (
                max(bounds[1, 1], start[1], goal[1]) + self.bounds_margin
            )
        return bounds

    def _base_pose_from_robot_states(self, robot_states):
        if robot_states is None:
            raise RuntimeError("OMPLBasePlanner requires current robot_states")
        q = np.asarray(robot_states[0], dtype=np.float64)
        if q.size < 3:
            raise ValueError("robot q must contain base [x, y, yaw]")
        return q[:3].copy()

    def _parse_base_target(self, config):
        target = config.get("base_pose", config.get("base_goal"))
        if target is None:
            raise ValueError("OMPLBasePlanner requires base_pose or base_goal")
        target = parsing.parse_array(target)
        if len(target) != 3:
            raise ValueError("base_pose must be SE2 [x, y, yaw]")
        return np.asarray(target, dtype=np.float64)

    def _parse_bounds_xy(self, config):
        raw = config.get("bounds_xy", [[-1.0, 4.0], [-2.5, 2.5]])
        bounds = np.asarray(raw, dtype=np.float64)
        if bounds.shape != (2, 2):
            raise ValueError("bounds_xy must be [[xmin, xmax], [ymin, ymax]]")
        if np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("bounds_xy lower limits must be < upper limits")
        return bounds

    def _parse_collision_points(self, esdf_config):
        raw_points = esdf_config.get("collision_points")
        if raw_points is not None:
            points = np.asarray(raw_points, dtype=np.float64)
            if points.ndim != 2 or points.shape[1] != 3:
                raise ValueError("collision_points must have shape (N, 3)")
            return points

        z_samples = np.asarray(
            esdf_config.get("query_z", [0.15, 0.35]), dtype=np.float64
        ).reshape(-1)
        return np.column_stack(
            (
                np.zeros_like(z_samples),
                np.zeros_like(z_samples),
                z_samples,
            )
        )


class _ESDFStateValidityChecker:
    """Small adapter around OMPL's Python StateValidityChecker base class."""

    def __new__(cls, space_information, planner):
        from ompl import base as ob

        class Checker(ob.StateValidityChecker):
            def __init__(self, si, parent):
                super().__init__(si)
                self.parent = parent

            def isValid(self, state):
                return self.parent._is_state_valid(
                    [state.getX(), state.getY(), state.getYaw()]
                )

        return Checker(space_information, planner)
