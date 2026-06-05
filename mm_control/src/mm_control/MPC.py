import time
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d

from mm_control.esdf_map import ESDFMap
from mm_control.MPCConstraints import (
    LinearizedESDFConstraint,
    NonholonomicBaseConstraint,
)
from mm_control.MPCBase import MPCBase
from mm_control.MPCCostFunctions import CostFunctionRegistry
from mm_utils.math import wrap_pi_array
from mm_utils.parsing import parse_path, parse_ros_path


class MPC(MPCBase):
    """Single-task Model Predictive Controller using Acados"""

    def __init__(self, config):
        """Initialize MPC controller.

        Args:
            config (dict): Configuration dictionary with MPC parameters.
        """
        super().__init__(config)
        self._setup_esdf_collision()

        num_terminal_cost = 2
        cost_params = config["cost_params"]

        # Create cost functions using simplified parameterized registry
        costs = []

        # Base costs - always use SE2 (yaw tracking controlled via weights, set yaw weight to 0 to disable)
        costs.append(
            CostFunctionRegistry.create(
                "BasePose", self.robot, cost_params.get("BasePose", {}), dimension="SE2"
            )
        )
        costs.append(
            CostFunctionRegistry.create(
                "BaseVel", self.robot, cost_params.get("BaseVel", {}), dimension=3
            )
        )

        # EE costs - always use SE3 (orientation weights can be set to 0 if not needed)
        costs.append(
            CostFunctionRegistry.create(
                "EEPose",
                self.robot,
                cost_params.get("EEPose", {}),
                pose_type="SE3",
                frame="world",
            )
        )
        costs.append(
            CostFunctionRegistry.create(
                "EEVel", self.robot, cost_params.get("EEVel", {})
            )
        )

        # Control effort (always included)
        costs.append(
            CostFunctionRegistry.create(
                "ControlEffort", self.robot, cost_params.get("Effort", {})
            )
        )

        # Add collision costs/constraints
        constraints = []
        if (
            self.robot.base_type == "nonholonomic"
            and self.robot.nonholonomic_mode != "dynamics"
        ):
            constraints.append(NonholonomicBaseConstraint(self.robot))

        for name in self.collision_link_names:
            # fmt: off
            is_static = name in self.model_interface.scene.collision_link_names["static_obstacles"]
            softened = self.params["collision_constraints_softened"]["static_obstacles" if is_static else name]
            # fmt: on
            if softened:
                costs.append(self.collisionSoftCsts[name])
            else:
                constraints.append(self.collisionCsts[name])

        if self.esdf_collision_enabled:
            constraints.append(self.esdfCollisionCst)

        name = self.params["acados"].get("name", "MM")
        self.ocp, self.ocp_solver, self.p_struct = self._construct(
            costs, constraints, num_terminal_cost, name
        )

        self.cost = costs
        self.constraints = constraints + [self.controlCst, self.stateCst]

        # Initialize masks to None (will be set from references in control())
        self.base_mask = None
        self.ee_mask = None

    def _get_config_key_for_cost_name(self, cost_name):
        """Map cost function name to simplified config parameter key.

        Args:
            cost_name (str): Cost function name (e.g., "EEPoseSE3").

        Returns:
            str: Simplified config key (e.g., "EEPose").
        """
        name_mapping = {
            "EEPoseSE3": "EEPose",
            "EEVel6": "EEVel",
            "BasePoseSE2": "BasePose",
            "BaseVel3": "BaseVel",
        }
        return name_mapping.get(cost_name, cost_name)

    def _set_control_effort_params(self, curr_p_map):
        """Set ControlEffort cost function parameters in the parameter map.

        Args:
            curr_p_map (casadi.struct_MX): Current parameter map to update.
        """
        effort_params = self.params["cost_params"]["Effort"]
        for param_name in ["Qqa", "Qqb", "Qva", "Qvb", "Qua", "Qub"]:
            curr_p_map[f"{param_name}_ControlEffort"] = effort_params[param_name]

    def _setup_esdf_collision(self):
        """Configure optional ESDF linearization data used outside the solver."""
        self.esdf_collision_config = self.params.get("esdf_collision", {})
        self.esdf_collision_enabled = bool(
            self.esdf_collision_config.get("enabled", False)
        )
        self.esdf_map = None
        self.esdf_sphere_names = []
        self.esdf_sphere_radii = np.zeros(0)
        self.esdf_safety_margin = float(
            self.esdf_collision_config.get("d_safe", 0.05)
        )
        self.esdf_invalid_distance = float(
            self.esdf_collision_config.get("invalid_distance", -1.0)
        )
        self.esdf_accept_status2_min_margin = float(
            self.esdf_collision_config.get("accept_status2_min_margin", 0.0)
        )
        self.esdf_constraint_name = self.esdf_collision_config.get("name", "esdf")
        self.esdfCollisionCst = None
        self.esdf_linearization = None

        if not self.esdf_collision_enabled:
            return

        map_path = self._resolve_esdf_map_path(self.esdf_collision_config["map_path"])
        self.esdf_map = ESDFMap(
            map_path,
            require_all_corners_valid=self.esdf_collision_config.get(
                "require_all_corners_valid", True
            ),
        )

        sphere_names = list(self.esdf_collision_config.get("spheres", []))
        if (
            not sphere_names
            and "base_body_collision" in self.robot.collisionLinkKinSymMdls
        ):
            sphere_names = ["base_body_collision"]
        if not sphere_names:
            raise ValueError(
                "esdf_collision.enabled is true, but no ESDF collision spheres "
                "were configured"
            )

        self.esdf_sphere_names = sphere_names
        self.esdf_sphere_radii = np.asarray(
            [self._get_collision_sphere_radius(name) for name in sphere_names],
            dtype=float,
        )
        self.esdfCollisionCst = LinearizedESDFConstraint(
            self.robot,
            self.esdf_sphere_names,
            self.esdf_sphere_radii,
            self.esdf_safety_margin,
            self.esdf_constraint_name,
        )

    def _resolve_esdf_map_path(self, map_path):
        if isinstance(map_path, dict):
            return parse_ros_path(map_path)
        return parse_path(str(map_path))

    def _get_collision_sphere_radius(self, name):
        if name not in self.robot.collisionLinkKinSymMdls:
            raise KeyError(f"Unknown robot collision link '{name}' for ESDF collision")

        spec = self.robot.collision_object_specs.get(name)
        if spec is not None:
            if spec.get("type", "").lower() != "sphere":
                raise ValueError(
                    f"ESDF collision link '{name}' must be a sphere, got {spec.get('type')}"
                )
            return float(spec["radius"])

        geom = self.model_interface.pinocchio_interface.getGeometryObject(name)
        if geom is None or not hasattr(geom.geometry, "radius"):
            raise ValueError(
                f"Could not infer sphere radius for ESDF collision link '{name}'"
            )
        return float(geom.geometry.radius)

    def control(
        self,
        t: float,
        robot_states: Tuple[NDArray[np.float64], NDArray[np.float64]],
        references: dict,
    ):
        """
        Args:
            t (float): Current control time.
            robot_states (tuple): (q, v) generalized coordinates and velocities.
            references (dict): Dictionary with reference trajectories from TaskManager:
                {
                    "base_pose": array of shape (N+1, 3) or None,
                    "base_velocity": array of shape (N+1, 3) or None,
                    "ee_pose": array of shape (N+1, 6) or None,
                    "ee_velocity": array of shape (N+1, 6) or None,
                    "base_mask": array of shape (3,) or None,  # [x, y, yaw] - True means dimension matters
                    "ee_mask": array of shape (6,) or None,  # [x, y, z, roll, pitch, yaw] - True means dimension matters
                }

        Returns:
            tuple: (v_bar, u_bar) where:
                - v_bar: velocity trajectory, shape (N+1, nu)
                - u_bar: control input trajectory, shape (N, nu)
        """
        self.py_logger.debug(f"control time {t}")
        self.curr_control_time = t
        q, v = robot_states
        q[2:9] = wrap_pi_array(q[2:9])
        xo = np.hstack((q, v))

        x_bar_initial, u_bar_initial = self._prepare_warm_start(t, xo)
        r_bar_map = self._convert_references_to_r_bar_map(references, xo)
        curr_p_map_bar = self._setup_horizon_parameters(
            r_bar_map, x_bar_initial, u_bar_initial
        )
        self._solve_and_extract(xo, t, curr_p_map_bar, x_bar_initial, u_bar_initial)
        self._update_logging(curr_p_map_bar)

        velocity_traj = self.x_bar[:, self.DoF :].copy()
        return velocity_traj, self.u_bar.copy()

    def _prepare_warm_start(self, t, xo):
        """Prepare warm start trajectories from previous solution or zeros.

        Args:
            t (float): Current control time.
            xo (ndarray): Current state vector.

        Returns:
            tuple: (x_bar_initial, u_bar_initial) initial guess trajectories.
        """
        if self.t_bar is not None:
            self.u_t = interp1d(
                self.t_bar,
                self.u_bar,
                axis=0,
                bounds_error=False,
                fill_value="extrapolate",
            )
            t_bar_new = t + np.arange(self.N) * self.dt
            self.u_bar = self.u_t(t_bar_new)
            self.x_bar = self._predictTrajectories(xo, self.u_bar)
        else:
            self.u_bar = np.zeros_like(self.u_bar)
            self.x_bar = self._predictTrajectories(xo, self.u_bar)

        return self.x_bar.copy(), self.u_bar.copy()

    def _convert_references_to_r_bar_map(self, references, xo):
        """Convert references from TaskManager format to MPC cost function format.

        Args:
            references (dict): Dictionary from TaskManager with keys:
                - "base_pose": array of shape (N+1, 3) or None
                - "base_velocity": array of shape (N+1, 3) or None
                - "ee_pose": array of shape (N+1, 6) or None
                - "ee_velocity": array of shape (N+1, 6) or None
                - "base_mask": array of shape (3,) or None, [x, y, yaw] - True means dimension matters
                - "ee_mask": array of shape (6,) or None, [x, y, z, roll, pitch, yaw] - True means dimension matters
            xo (ndarray): Current state vector [q, v] to compute current EE pose if needed

        Returns:
            Dictionary with cost function names as keys:
                - "BasePoseSE2": list of arrays (N+1, 3)
                - "BaseVel3": list of arrays (N+1, 3)
                - "EEPoseSE3": list of arrays (N+1, 6)
                - "EEVel6": list of arrays (N+1, 6)
        """
        r_bar_map = {}

        # Store masks as instance variables for use in _set_tracking_params
        self.base_mask = references.get("base_mask")
        self.ee_mask = references.get("ee_mask")

        # Convert base pose reference - only if provided
        if references.get("base_pose") is not None:
            base_pose = references["base_pose"]
            r_bar_map["BasePoseSE2"] = [base_pose[i] for i in range(self.N + 1)]
            self.rbase_bar = (
                base_pose.tolist() if hasattr(base_pose, "tolist") else base_pose
            )
        else:
            # No base reference: set empty list for visualization
            self.rbase_bar = []

        # Convert base velocity reference - only if provided
        if references.get("base_velocity") is not None:
            base_vel = references["base_velocity"]
            r_bar_map["BaseVel3"] = [base_vel[i] for i in range(self.N + 1)]

        # Convert EE pose reference - only if provided
        if references.get("ee_pose") is not None:
            ee_pose = references["ee_pose"]
            # EE reference is in world frame
            r_bar_map["EEPoseSE3"] = [ee_pose[i] for i in range(self.N + 1)]
            self.ree_bar = ee_pose.tolist() if hasattr(ee_pose, "tolist") else ee_pose
        else:
            # No EE reference: set empty list for visualization
            self.ree_bar = []

        # Convert EE velocity reference - only if provided
        if references.get("ee_velocity") is not None:
            ee_vel = references["ee_velocity"]
            r_bar_map["EEVel6"] = [ee_vel[i] for i in range(self.N + 1)]

        return r_bar_map

    def _setup_horizon_parameters(self, r_bar_map, x_bar_initial, u_bar_initial):
        """Setup OCP parameters for each horizon step.

        Args:
            r_bar_map (dict): Dictionary mapping cost function names to reference trajectories.
            x_bar_initial (ndarray): Initial state trajectory guess, shape (N+1, nx).
            u_bar_initial (ndarray): Initial control trajectory guess, shape (N, nu).

        Returns:
            list: List of parameter maps for each horizon step.
        """
        tp1 = time.perf_counter()
        curr_p_map_bar = []

        # Reset time logging
        for key in self.log.keys():
            if "time" in key:
                self.log[key] = 0

        self._update_esdf_linearization(x_bar_initial)

        for i in range(self.N + 1):
            curr_p_map = self.p_struct(0)
            self._set_initial_guess(curr_p_map, i, x_bar_initial, u_bar_initial)
            self._set_tracking_params(curr_p_map, r_bar_map, i)
            self._set_control_effort_params(curr_p_map)
            self._set_esdf_params(curr_p_map, i)
            self._set_ocp_params(curr_p_map, i)
            curr_p_map_bar.append(curr_p_map)

        tp2 = time.perf_counter()
        self.log["time_ocp_set_params"] = tp2 - tp1
        return curr_p_map_bar

    def _set_esdf_params(self, curr_p_map, i):
        if not self.esdf_collision_enabled:
            return
        if self.esdf_linearization is None:
            raise RuntimeError("ESDF linearization data was not initialized")

        suffix = self.esdf_constraint_name
        curr_p_map[f"c0_{suffix}"] = self.esdf_linearization["c0"][i].T
        curr_p_map[f"n0_{suffix}"] = self.esdf_linearization["n0"][i].T
        curr_p_map[f"d0_{suffix}"] = self.esdf_linearization["d0"][i].reshape((-1, 1))

    def _update_esdf_linearization(self, x_bar_initial):
        """Linearize ESDF distances at the current warm-start trajectory."""
        t1 = time.perf_counter()
        if not self.esdf_collision_enabled:
            self.esdf_linearization = None
            return

        q_bar = x_bar_initial[:, : self.DoF]
        num_nodes = q_bar.shape[0]
        num_spheres = len(self.esdf_sphere_names)
        centers = self._compute_esdf_sphere_centers(q_bar)

        flat_centers = centers.reshape((-1, 3))
        distances, gradients, valid = self.esdf_map.query(flat_centers)
        distances = distances.reshape((num_nodes, num_spheres))
        gradients = gradients.reshape((num_nodes, num_spheres, 3))
        valid = valid.reshape((num_nodes, num_spheres))

        solver_distances = np.where(valid, distances, self.esdf_invalid_distance)
        solver_gradients = np.where(valid[:, :, None], gradients, 0.0)

        clearance = distances - self.esdf_sphere_radii[None, :]
        margin = clearance - self.esdf_safety_margin

        self.esdf_linearization = {
            "sphere_names": list(self.esdf_sphere_names),
            "radii": self.esdf_sphere_radii.copy(),
            "c0": centers,
            "raw_d0": distances,
            "raw_n0": gradients,
            "d0": solver_distances,
            "n0": solver_gradients,
            "valid": valid,
            "clearance": clearance,
            "margin": margin,
        }

        valid_distances = distances[valid]
        valid_clearance = clearance[valid]
        valid_margin = margin[valid]
        self.log["esdf_all_valid"] = bool(np.all(valid))
        self.log["esdf_valid_count"] = int(np.count_nonzero(valid))
        self.log["esdf_total_count"] = int(valid.size)
        self.log["esdf_min_distance"] = (
            float(np.min(valid_distances)) if valid_distances.size else np.nan
        )
        self.log["esdf_min_clearance"] = (
            float(np.min(valid_clearance)) if valid_clearance.size else np.nan
        )
        self.log["esdf_min_margin"] = (
            float(np.min(valid_margin)) if valid_margin.size else np.nan
        )
        t2 = time.perf_counter()
        self.log["time_esdf_linearization"] = t2 - t1

    def _compute_esdf_sphere_centers(self, q_bar):
        """Evaluate configured collision sphere centers for a q trajectory."""
        num_nodes = q_bar.shape[0]
        num_spheres = len(self.esdf_sphere_names)
        centers = np.zeros((num_nodes, num_spheres, 3), dtype=float)

        for sphere_idx, name in enumerate(self.esdf_sphere_names):
            fk_fcn = self.robot.collisionLinkKinSymMdls[name]
            for node_idx in range(num_nodes):
                pos, _ = fk_fcn(q_bar[node_idx])
                pos_np = pos.toarray() if hasattr(pos, "toarray") else pos
                centers[node_idx, sphere_idx] = np.asarray(
                    pos_np, dtype=float
                ).reshape(3)
        return centers

    def _evaluate_solution_esdf_margins(self, x_bar):
        """Query actual ESDF margins for a candidate state trajectory."""
        if not self.esdf_collision_enabled:
            return {
                "all_valid": True,
                "valid_count": 0,
                "total_count": 0,
                "min_margin": np.inf,
            }

        q_bar = x_bar[:, : self.DoF]
        centers = self._compute_esdf_sphere_centers(q_bar)
        flat_centers = centers.reshape((-1, 3))
        distances, _, valid = self.esdf_map.query(flat_centers)
        distances = distances.reshape((q_bar.shape[0], len(self.esdf_sphere_names)))
        valid = valid.reshape((q_bar.shape[0], len(self.esdf_sphere_names)))
        margins = distances - self.esdf_sphere_radii[None, :] - self.esdf_safety_margin

        valid_margins = margins[valid]
        return {
            "all_valid": bool(np.all(valid)),
            "valid_count": int(np.count_nonzero(valid)),
            "total_count": int(valid.size),
            "min_margin": float(np.min(valid_margins)) if valid_margins.size else np.nan,
        }

    def _set_initial_guess(self, curr_p_map, i, x_bar_initial, u_bar_initial):
        """Set initial guess for state, control, and multipliers.

        Args:
            curr_p_map (casadi.struct_MX): Current parameter map (not used, but kept for consistency).
            i (int): Horizon step index.
            x_bar_initial (ndarray): Initial state trajectory guess, shape (N+1, nx).
            u_bar_initial (ndarray): Initial control trajectory guess, shape (N, nu).
        """
        t1 = time.perf_counter()
        self.ocp_solver.set(i, "x", x_bar_initial[i])
        if i < self.N:
            self.ocp_solver.set(i, "u", u_bar_initial[i])
        if self.lam_bar is not None:
            self.ocp_solver.set(i, "lam", self.lam_bar[i])
        t2 = time.perf_counter()
        self.log["time_ocp_set_params_set_x"] += t2 - t1

    def _set_tracking_params(self, curr_p_map, r_bar_map, i):
        """Set tracking cost function parameters.

        Args:
            curr_p_map (casadi.struct_MX): Current parameter map to update.
            r_bar_map (dict): Dictionary mapping cost function names to reference trajectories.
            i (int): Horizon step index.
        """
        t1 = time.perf_counter()
        p_keys = self.p_struct.keys()

        # List of all possible tracking cost functions (world frame only)
        all_tracking_costs = ["BasePoseSE2", "BaseVel3", "EEPoseSE3", "EEVel6"]

        for name in all_tracking_costs:
            p_name_r = f"r_{name}"  # Reference parameter for the tracking cost (e.g., r_EEPoseSE3)
            p_name_W = f"W_{name}"  # Weight matrix parameter for the tracking cost (e.g., W_EEPoseSE3)

            if p_name_r in p_keys:
                if name in r_bar_map:
                    # Reference provided: set reference and use configured weights
                    curr_p_map[p_name_r] = r_bar_map[name][i]
                    config_key = self._get_config_key_for_cost_name(name)
                    cost_params = self.params["cost_params"].get(config_key, {})
                    weight_key = "P" if i == self.N else "Qk"
                    weights = cost_params.get(
                        weight_key, [1.0] * len(r_bar_map[name][i])
                    )

                    if isinstance(weights, (int, float)):
                        weights = [weights] * len(r_bar_map[name][i])
                    weights = np.array(weights)

                    # Apply masks to zero out weights for masked dimensions
                    if (
                        name in ["BasePoseSE2", "BaseVel3"]
                        and self.base_mask is not None
                    ):
                        if len(weights) == len(self.base_mask):
                            weights = weights * self.base_mask.astype(float)
                        else:
                            raise ValueError(
                                f"Weights length {len(weights)} does not match base mask length {len(self.base_mask)}"
                            )
                    elif name in ["EEPoseSE3", "EEVel6"] and self.ee_mask is not None:
                        if len(weights) == len(self.ee_mask):
                            weights = weights * self.ee_mask.astype(float)
                        else:
                            raise ValueError(
                                f"Weights length {len(weights)} does not match ee mask length {len(self.ee_mask)}"
                            )

                    curr_p_map[p_name_W] = np.diag(weights)
                else:
                    # No reference provided: set weights to zero (minimize control effort only)
                    config_key = self._get_config_key_for_cost_name(name)
                    cost_params = self.params["cost_params"].get(config_key, {})
                    weight_key = "P" if i == self.N else "Qk"
                    # Get dimension from default weights
                    default_weights = cost_params.get(weight_key, [1.0])
                    if isinstance(default_weights, (list, np.ndarray)):
                        dim = len(default_weights)
                    else:
                        # Scalar weight - determine dimension from cost function name
                        if name == "EEPoseSE3":
                            dim = 6
                        elif name == "BasePoseSE2":
                            dim = 3
                        elif name == "EEVel6":
                            dim = 6
                        elif name == "BaseVel3":
                            dim = 3
                        else:
                            raise ValueError(f"Unknown cost function name: {name}")
                    # Set zero reference and zero weights
                    curr_p_map[p_name_r] = np.zeros(dim)
                    curr_p_map[p_name_W] = np.diag(np.zeros(dim))
            else:
                raise RuntimeError(f"Parameter {p_name_r} not found in p_struct keys")

        t2 = time.perf_counter()
        self.log["time_ocp_set_params_tracking"] += t2 - t1

    def _set_ocp_params(self, curr_p_map, i):
        """Set OCP parameters for the current horizon step.

        Args:
            curr_p_map (casadi.struct_MX): Current parameter map.
            i (int): Horizon step index.
        """
        t1 = time.perf_counter()
        self.ocp_solver.set(i, "p", curr_p_map.cat.full().flatten())
        t2 = time.perf_counter()
        self.log["time_ocp_set_params_setp"] += t2 - t1

    def _solve_and_extract(self, xo, t, curr_p_map_bar, x_bar_initial, u_bar_initial):
        """Solve the OCP and extract solution.

        Args:
            xo (ndarray): Current state vector.
            t (float): Current control time.
            curr_p_map_bar (list): List of parameter maps for each horizon step.
            x_bar_initial (ndarray): Initial state trajectory guess, shape (N+1, nx).
            u_bar_initial (ndarray): Initial control trajectory guess, shape (N, nu).
        """
        t1 = time.perf_counter()
        self.ocp_solver.solve_for_x0(xo, fail_on_nonzero_status=False)
        t2 = time.perf_counter()
        self.log["time_ocp_solve"] = t2 - t1

        self.ocp_solver.print_statistics()
        status = (
            self.ocp_solver.get_status()
            if hasattr(self.ocp_solver, "get_status")
            else getattr(self.ocp_solver, "status", 0)
        )
        self.log["solver_status"] = status
        self.log["solver_fallback"] = False
        self.log["solver_status_accepted"] = False
        self.log["status2_esdf_all_valid"] = False
        self.log["status2_esdf_valid_count"] = 0
        self.log["status2_esdf_total_count"] = 0
        self.log["status2_esdf_min_margin"] = np.nan
        if self.ocp.solver_options.nlp_solver_type != "SQP_RTI":
            self.log["step_size"] = np.mean(self.ocp_solver.get_stats("alpha"))
        else:
            self.log["step_size"] = -1
        self.log["sqp_iter"] = self.ocp_solver.get_stats("sqp_iter")
        self.log["qp_iter"] = sum(self.ocp_solver.get_stats("qp_iter"))
        self.log["cost_final"] = self.ocp_solver.get_cost()

        if status != 0:
            x_bar, u_bar, lam_bar = self._extract_solver_solution()

            self.log["iter_snapshot"] = {
                "t": t,
                "xo": xo,
                "p_map_bar": [p.cat.full().flatten() for p in curr_p_map_bar],
                "x_bar_init": x_bar_initial,
                "u_bar_init": u_bar_initial,
                "x_bar": x_bar,
                "u_bar": u_bar,
            }

            if status == 2 and self._accept_status2_solution_if_safe(
                x_bar, u_bar, lam_bar, t
            ):
                return

            if self.params["acados"]["raise_exception_on_failure"]:
                raise Exception(
                    f"acados_ocp_solver returned status {status}"
                )

            self._apply_solver_failure_fallback(xo, t, status)
            return
        else:
            self.log["iter_snapshot"] = None

        x_bar, u_bar, lam_bar = self._extract_solver_solution()
        self._set_solver_solution(x_bar, u_bar, lam_bar, t)

    def _extract_solver_solution(self):
        """Read the current state, control, and multiplier trajectories."""
        x_bar = np.zeros_like(self.x_bar)
        u_bar = np.zeros_like(self.u_bar)
        lam_bar = []
        for i in range(self.N):
            x_bar[i, :] = self.ocp_solver.get(i, "x")
            u_bar[i, :] = self.ocp_solver.get(i, "u")
            if lam_bar is not None:
                try:
                    lam_bar.append(self.ocp_solver.get(i, "lam"))
                except Exception:
                    lam_bar = None
        x_bar[self.N, :] = self.ocp_solver.get(self.N, "x")
        if lam_bar is not None:
            try:
                lam_bar.append(self.ocp_solver.get(self.N, "lam"))
            except Exception:
                lam_bar = None
        return x_bar, u_bar, lam_bar

    def _set_solver_solution(self, x_bar, u_bar, lam_bar, t):
        self.x_bar = np.asarray(x_bar, dtype=float).copy()
        self.u_bar = np.asarray(u_bar, dtype=float).copy()
        self.lam_bar = lam_bar
        self.t_bar = t + np.arange(self.N) * self.dt
        self.v_cmd = self.x_bar[0][self.robot.DoF :].copy()

    def _accept_status2_solution_if_safe(self, x_bar, u_bar, lam_bar, t):
        """Accept max-iteration solver output when the ESDF trajectory is safe."""
        if not np.all(np.isfinite(x_bar)) or not np.all(np.isfinite(u_bar)):
            return False

        esdf_stats = self._evaluate_solution_esdf_margins(x_bar)
        self.log["status2_esdf_all_valid"] = esdf_stats["all_valid"]
        self.log["status2_esdf_valid_count"] = esdf_stats["valid_count"]
        self.log["status2_esdf_total_count"] = esdf_stats["total_count"]
        self.log["status2_esdf_min_margin"] = esdf_stats["min_margin"]

        if not esdf_stats["all_valid"]:
            return False
        if esdf_stats["min_margin"] < self.esdf_accept_status2_min_margin:
            return False

        self.py_logger.warning(
            "acados solver returned status 2 at t=%.3f; accepting current "
            "solution with ESDF min margin %.4f",
            t,
            esdf_stats["min_margin"],
        )
        self.log["solver_status_accepted"] = True
        self._set_solver_solution(x_bar, u_bar, lam_bar, t)
        return True

    def _apply_solver_failure_fallback(self, xo, t, status):
        """Use a safe zero-velocity trajectory when acados returns a bad status."""
        self.py_logger.warning(
            "acados solver returned status %s at t=%.3f; using zero-velocity fallback",
            status,
            t,
        )
        self.log["solver_fallback"] = True

        self.x_bar = np.tile(xo, (self.N + 1, 1))
        self.x_bar[:, self.DoF :] = 0.0
        self.u_bar = np.zeros_like(self.u_bar)

        # Do not warm-start the next solve from failed primal/dual iterates.
        self.t_bar = None
        self.lam_bar = None
        self.v_cmd = np.zeros(self.nx - self.DoF)

    def _update_logging(self, curr_p_map_bar):
        """Update logging and visualization data.

        Args:
            curr_p_map_bar (list): List of parameter maps for each horizon step.
        """
        t1 = time.perf_counter()

        self.ee_bar, self.base_bar = self._getEEBaseTrajectories(self.x_bar)

        for name in self.collision_link_names:
            self.log["_".join([name, "constraint"])] = self.evaluate_constraints(
                self.collisionCsts[name], self.x_bar, self.u_bar, curr_p_map_bar
            )
        if self.esdf_collision_enabled:
            self.log["esdf_constraint"] = self.evaluate_constraints(
                self.esdfCollisionCst, self.x_bar, self.u_bar, curr_p_map_bar
            )

        self.log["ee_pos"] = self.ee_bar.copy()
        self.log["base_pos"] = self.base_bar.copy()
        self.log["ocp_param"] = [p.cat.full().flatten() for p in curr_p_map_bar]
        self.log["x_bar"] = self.x_bar.copy()
        self.log["u_bar"] = self.u_bar.copy()
        t2 = time.perf_counter()
        self.log["time_ocp_overhead"] = t2 - t1

    def _get_log(self):
        """Get log dictionary structure with default keys.

        Returns:
            dict: Log dictionary with default keys initialized to zero or empty.
        """
        log = {
            "cost_final": 0,
            "step_size": 0,
            "sqp_iter": 0,
            "qp_iter": 0,
            "solver_status": 0,
            "solver_fallback": False,
            "solver_status_accepted": False,
            "status2_esdf_all_valid": False,
            "status2_esdf_valid_count": 0,
            "status2_esdf_total_count": 0,
            "status2_esdf_min_margin": np.nan,
            "time_ocp_set_params": 0,
            "time_ocp_solve": 0,
            "time_ocp_set_params_set_x": 0,
            "time_ocp_set_params_tracking": 0,
            "time_ocp_set_params_setp": 0,
            "time_esdf_linearization": 0,
            "state_constraint": 0,
            "control_constraint": 0,
            "esdf_all_valid": False,
            "esdf_valid_count": 0,
            "esdf_total_count": 0,
            "esdf_min_distance": np.nan,
            "esdf_min_clearance": np.nan,
            "esdf_min_margin": np.nan,
            "esdf_constraint": 0,
            "x_bar": 0,
            "u_bar": 0,
            "lam_bar": 0,
            "ee_pos": 0,
            "base_pos": 0,
            "ocp_param": {},
            "iter_snapshot": {},
        }
        for name in self.collision_link_names:
            log["_".join([name, "constraint"])] = 0
            log["_".join([name, "constraint", "gradient"])] = 0

        return log

    def reset(self):
        """Reset controller state and solver."""
        super().reset()
        self.ocp_solver.reset()


if __name__ == "__main__":
    from mm_utils import parsing

    path_to_config = parse_ros_path(
        {"package": "mm_run", "path": "config/3d_collision.yaml"}
    )
    config = parsing.load_config(path_to_config)

    controller = MPC(config["controller"])
