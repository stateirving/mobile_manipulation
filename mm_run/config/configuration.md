# Configuration Reference

Runtime configuration is YAML, normally under `mm_run/config/`. This document
is a module map and field guide, not a frozen schema. The referenced profile
YAML and the consuming Python class remain authoritative.

## Loading and Composition

`mm_utils.parsing.load_config()` recursively merges `include` files, then
applies the current file as the final override. Nested dictionaries merge;
later/scalar/list values replace earlier values. Inclusion depth is limited to
five by default.

```yaml
include:
  - package: mm_run
    path: config/controller/MPC.yaml
  - package: mm_run
    path: config/robot/stretch.yaml
  - package: mm_run
    path: config/shared.yaml
    key: optional_parent_key
```

Package/path mappings resolve through the ROS package index. String paths also
support environment-variable and user expansion. Numeric arrays may use parser
expressions already present in the profiles, such as `2.pi`.

After editing installed runtime YAML, rebuild and source the workspace:

```bash
colcon build --packages-select mm_run
source install/setup.bash
```

## Canonical Profiles

| Profile | Purpose |
|---|---|
| `simple_experiment.yaml` | Minimal synchronous/asynchronous simulation |
| `stretch_esdf_offline_ompl_wbmpc.yaml` | Canonical Stretch OMPL + offline ESDF + WB-MPC |
| `stretch_esdf_online_nvblox_ompl_wbmpc.yaml` | Online simulated-depth nvblox overlay |
| `stretch_esdf_teleop_export.yaml` | Keyboard capture and offline NPZ export overlay |
| `stretch_esdf_sim_real_commissioning.yaml` | Simulated-room ESDF with real Stretch state/output |
| `ur10_esdf_offline_ompl_wbmpc.yaml` | Mobile UR10 offline-ESDF profile |
| `stretch_command_adapter_real.yaml` | Real Stretch feedback/command safety adapter |
| `stretch_wbmpc_shadow_real.yaml` | Real-state WB-MPC runner |
| `stretch_wbmpc_sim_esdf_real_test.yaml` | Hardware-in-the-loop commissioning runner |
| `stretch_*_state_probe_real.yaml` | Read-only base, joint, and full-state validation |

## Experiment Top-Level Modules

```yaml
include: []
planner: {}
controller: {}
simulation: {}
online_nvblox_sim: {}
teleop_esdf_export: {}
logging: {}
```

Robot and scene fragments normally merge into both `controller` and
`simulation`. Sensor and teleop YAML files can also be standalone inputs to
their own nodes rather than full experiment profiles.

## Planner

`mm_plan.TaskManager` expands `planner.task_defaults` into `planner.tasks`.
A task selects defaults with `defaults: base` or `defaults: ee`, then overrides
individual fields.

```yaml
planner:
  task_defaults:
    base:
      planner_type: OMPLBasePlanner
      base_mask: [true, true, true]
      tracking_pos_err_tol: 0.1
      tracking_ori_err_tol: 0.25
      hold_period: 1.0
      end_stop: false
    ee:
      planner_type: OMPLEEPlanner
      ee_mask: [true, true, true, false, false, false]

  tasks:
    - defaults: base
      name: Base Goal
      base_pose: [2.0, 0.0, 0.0]
    - defaults: ee
      name: EE Goal
      ee_pose: [3.0, -0.5, 0.9, 0.0, 0.0, -0.0708]
```

Supported `planner_type` values are:

- `WaypointPlanner`: `base_pose` and/or `ee_pose`;
- `PathPlanner`: `base_path` and/or `ee_path`, plus `dt`;
- `OMPLBasePlanner`: plans an SE(2) base path;
- `OMPLEEPlanner`: plans a Cartesian end-effector path.

Common completion fields are `tracking_pos_err_tol`,
`tracking_ori_err_tol`, `hold_period`, and `end_stop`.

### OMPL Base and EE Options

The current OMPL planners consume planner-specific bounds, path generation,
ESDF validity, and replanning policies:

```yaml
planner:
  task_defaults:
    base:
      bounds_xy: [[-4.0, 4.0], [-4.0, 4.0]]
      bounds_margin: 0.75
      ompl:
        planner: RRTstar
        solve_time: 3.0
        solve_attempts: 5
        simplify: true
        simplify_time: 0.05
        range: 0.1
        goal_tolerance: 0.08
        state_validity_resolution: 0.01
      path:
        dt: 0.1
        linear_speed: 0.25
        angular_speed: 0.8
        interpolation_resolution: 0.05
        yaw_mode: tangent
      esdf:
        enabled: true
        base_radius: 0.2
        d_safe: 0.2
        query_z: [0.15, 0.35]
        unknown_is_valid: false
        allow_straight_line_fallback: false
      replan:
        enabled: true
        min_interval: 1.0
        check_horizon: 2.0
        check_dt: 0.2
        min_clearance: 0.0
        deviation_threshold: 0.4
        force_periodic: false
        keep_previous_on_failure: true
```

EE defaults use `bounds_xyz`, `tool_radius`, and Cartesian path speeds. See
`stretch_esdf_offline_ompl_wbmpc.yaml` for the complete base and EE examples.

## Controller

### Timing and Behavior

```yaml
controller:
  type: MPC
  dt: 0.1
  prediction_horizon: 2.0
  ctrl_rate: 7
  cmd_vel_pub_rate: 20
  cmd_vel_type: interpolation   # integration | interpolation
  soft_cst: true
  cst_tol_schedule_enabled: false
  ee_pose_tracking_enabled: true
```

`cmd_vel_type: interpolation` sends the optimized velocity trajectory. The
real command adapter expects this mode and applies hardware slew limits itself.

### Robot Model

Robot fragments under `config/robot/` populate both `simulation.robot` and
`controller.robot` where appropriate.

```yaml
controller:
  robot:
    mimic: false
    dims: {q: 11, v: 11, x: 22, u: 11}
    joint_names: []
    x0: []
    time_discretization_dt: 0.1
    base_type: nonholonomic     # omnidirectional | fixed | nonholonomic | floating
    nonholonomic_mode: dynamics
    tool_link_name: link_grasp_center
    base_link_name: base_link
    limits:
      input: {lower: [], upper: []}
      state: {lower: [], upper: []}
    urdf:
      package: mm_assets
      path: stretch/stretch_sim.urdf
      includes: []
      args: {}
    collision_model:
      groups: {}
      objects: {}
      self_collision_pairs: false
      static_obstacle_pairs: {}
      pinocchio_self_collision_pairs: false
      pinocchio_static_obstacle_pairs: {}
```

`collision_model.groups` is the current grouping format. Legacy
`collision_link_names` and `collision_pairs` are still accepted by parts of the
collision stack.

### FCL Collision Avoidance

```yaml
controller:
  self_collision_avoidance_enabled: true
  static_obstacles_collision_avoidance_enabled: true
  self_collision_emergency_stop: true
  collision_constraint_type:
    self: SignedDistanceConstraint
    static_obstacles: SignedDistanceConstraint
  collision_constraints_softened: {self: true, static_obstacles: true}
  collision_safety_margin: {self: 0.25, static_obstacles: 0.15}
  xu_soft: {mu: 0.001, zeta: 0.005}
  collision_soft:
    self: {mu: 0.0001, zeta: 0.005}
    static_obstacles: {mu: 0.0001, zeta: 0.005}
```

### ESDF Collision Module

ESDF collision is independent of the legacy static-obstacle flag and changes
the Acados model when enabled.

```yaml
controller:
  esdf_collision:
    enabled: true
    source: offline             # offline | online_nvblox
    mode: constraint            # constraint | squared_hinge_cost
    map_path:
      package: mm_run
      path: results/.../esdf_grid.npz
    spheres: [base_body_collision]
    d_safe: 0.10
    require_all_corners_valid: true
    invalid_distance: -1.0
    accept_status2_min_margin: 0.0
    initialize_map: true
    name: esdf
    soft_cost: {p: 1.0, mu: 1000.0, smoothing: 0.005}
```

For `source: online_nvblox`, add `online_nvblox` with `voxel_size`,
`integrator_type`, query/update policy, unknown-distance policy, optional
`initial_map_path`, and camera intrinsics (`fx/fy/cx/cy/width/height`).

### Costs and Acados

Current cost keys are `BasePose`, `BaseVel`, `EEPose`, `EEVel`, `Effort`,
optional `ArmExtension`, `Regularization`, and `slack`. Pose/velocity entries use
running `Qk` and terminal `P` weights. `Effort` uses `Qqa/Qqb/Qva/Qvb/Qua/Qub`.

```yaml
controller:
  cost_params:
    BasePose: {Qk: [2, 2, 0], P: [40, 40, 0]}
    EEPose: {Qk: [5, 5, 5, 0, 0, 0], P: [50, 50, 50, 0, 0, 0]}
    Effort: {Qqa: [], Qqb: [], Qva: [], Qvb: [], Qua: [], Qub: []}
    ArmExtension:
      enabled: false
      joint_names: []
      upper: []
      base_task_upper: []
      weight: []
      smoothing: 0.001
    Regularization: {eps: 1.0e-6}
    slack: {z: 500, Z: 500000}
  beta: 0.5
  alpha: 0.05
  acados:
    name: MM
    cython: {enabled: true, recompile: false}
    raise_exception_on_failure: false
    use_custom_hess: true
    use_terminal_cost: true
    ocp_solver_options: {}
    slack_enabled: {x: true, x_e: true, u: false, h_0: true, h: true, h_e: true}
```

Regenerate Acados code after changing dimensions, dynamics, costs, constraints,
ESDF enablement/mode, or solver structure. A map-path-only change does not
require regeneration.

## Simulation and Scene

```yaml
simulation:
  timestep: 0.01
  duration: 25.0
  gravity: [0, 0, -9.81]
  gui: true
  robot: {}
  static_obstacles:
    enabled: false
    collision_enabled: true
    urdf: {}
  dynamic_obstacles:
    enabled: false
    obstacles: []
  collision_sphere_markers:
    enabled: false
    alpha: 0.25
    color: [0.0, 0.7, 1.0]
    object_colors: {}

controller:
  scene:
    enabled: false
    collision_link_names: {static_obstacles: []}
    urdf: {}
```

Isaac-specific camera and video definitions live in `config/sensor/cameras.yaml`
and `config/sim/isaac_sim.yaml`.

## Online nvblox Simulation

`online_nvblox_sim` drives rendered-depth integration for
`experiment_online_nvblox.py`. Major submodules are:

- renderer/image geometry: `renderer`, `width`, `height`, `fov_y_deg`,
  `near`, `far`;
- filtering: `ground_filter_min_z`, `ground_filter_use_segmentation`,
  `exclude_robot`, `exclude_collision_sphere_markers`;
- lifecycle: `initial_scan`, `realtime_scan`, `decay`;
- inspection: `diagnostics`, `preview`.

Realtime cameras support `pose_source: camera_link` or synthetic spin poses,
frame conventions, and RPY pose correction. Use
`stretch_esdf_online_nvblox_ompl_wbmpc.yaml` as the reference profile.

## Teleop ESDF Export

`teleop_esdf_export` configures `teleop_export_esdf.py`:

```yaml
teleop_esdf_export:
  output: mm_run/results/nvblox_esdf/stretch_teleop
  bounds: [-4.2, -4.2, 0.0, 4.2, 4.2, 2.0]
  grid_resolution: 0.02
  query_chunk_size: 131072
  linear_speed: 0.25
  angular_speed: 0.6
  status_interval: 2.0
  ground_aware_free_space: {}
  live_reconstruction: {}
```

The ground-aware module combines a ground-filtered obstacle TSDF with a second
observed-space map. The live reconstruction is a downsampled viewer and does
not set final NPZ resolution.

## Real Stretch Runtime Modules

These are node-specific configs rather than general experiment sections:

- `stretch_base_state_probe`, `stretch_state_probe`, and
  `stretch_full_state_probe`: frame, timestamp, joint mapping, synchronization,
  and freshness validation;
- `stretch_command_adapter`: ROS topics/services, state mapping, watchdogs,
  layered driver/adapter limits, following-error thresholds, and execute gates;
- `stretch_wbmpc_shadow`: controller profile, topics, control/publish rates,
  solver deadline, plan timeout, late-result policy, and optional forward
  prediction.

The complete command adapter contract is in
[`real_command_adapter.md`](./real_command_adapter.md); the deployed values and
ROS graph are in the repository-level [`REAL_DEPLOY.md`](../../REAL_DEPLOY.md).

## Logging

```yaml
logging:
  log_dir: experiment_name
  log_level: 20
```

Experiments normally write under `mm_run/results/<log_dir>/<timestamp>/` with
`combined/`, `sim/`, or `control/` subdirectories. Real WB-MPC runner and
adapter JSONL paths are command-line/config fields in their node-specific
profiles.
