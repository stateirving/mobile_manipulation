# Real Stretch command adapter

The command adapter starts in `shadow` mode. In this mode it subscribes to the
validated Stretch state and an 11-dimensional WB-MPC velocity request, but does
not create publishers for `/stretch/cmd_vel` or `/joint_pose_cmd`.

The adapter republishes its validated named 11-position/11-velocity feedback as
`sensor_msgs/msg/JointState` on `/wbmpc/state`. The separate WB-MPC shadow node
uses that state and publishes only `/wbmpc/velocity_command`. Adapter status has
two intentional operating states: `shadow` and `wbmpc` (plus `latched` after a
stop). Base-target versus arm-target behavior remains inside TaskManager/MPC.

The velocity input topic is `/wbmpc/velocity_command`, using a
`std_msgs/msg/String` JSON command envelope. Its `velocity` array has this exact
order:

```text
[base_vx_map, base_vy_map, base_yaw_rate,
 lift_velocity,
 arm_l3_velocity, arm_l2_velocity, arm_l1_velocity, arm_l0_velocity,
 wrist_yaw_velocity, wrist_pitch_velocity, wrist_roll_velocity]
```

This input is a velocity, not the WB-MPC `u_bar` acceleration. The real runner
uses `cmd_vel_type: interpolation` and samples the optimized `v_bar` velocity
trajectory one MPC step ahead. The adapter is the only real-deploy layer that
slews the current safe velocity toward that target using the configured
hardware acceleration limits.

The envelope also carries `generation`, `valid`, `reason`, `state_stamp`,
`plan_origin_monotonic`, and `valid_until_monotonic`. The adapter's independent
50 Hz watchdog changes an expired/invalid envelope into an immediate base-zero
and measured-qpos arm hold without deactivating streaming mode.

## Build and shadow run

```bash
cd ~/repo/mobile_manipulation
pixi run colcon build --packages-select mm_run
source install/setup.bash
ros2 launch mm_run stretch_command_adapter.launch.py
```

Publish a continuous zero command while checking readiness:

```bash
ros2 topic pub -r 10 /wbmpc/velocity_command \
  std_msgs/msg/String \
  "{data: '{\"generation\":0,\"valid\":false,\"reason\":\"manual zero hold\",\"state_stamp\":null,\"plan_origin_monotonic\":null,\"valid_until_monotonic\":null,\"velocity\":[0,0,0,0,0,0,0,0,0,0,0]}'}"

ros2 topic echo /stretch_command_adapter/status
ros2 node info /stretch_command_adapter
```

In shadow mode, `ros2 node info` must not list publishers for either hardware
command topic. Status records contain the projected lateral velocity, arm
projection residual, limited physical-channel velocity and candidate SG3 qpos.

Run the actual OMPL + WB-MPC controller against live feedback, still without
hardware command publishers:

```bash
ros2 launch mm_run stretch_wbmpc_shadow.launch.py \
  adapter_log:=/tmp/stretch_adapter_wbmpc_shadow.jsonl \
  wbmpc_log:=/tmp/stretch_wbmpc_shadow.jsonl
```

The WB-MPC source predicts feedback forward by the source-state age, adaptive
expected controller time, and configured dispatch latency. Prediction calls the
controller robot's own discrete dynamics and the exact state/input bound arrays
used by acados; it does not add a second full-ESDF scan. At `solver_deadline`
the adapter holds even if OMPL/acados is still blocked. With
`accept_late_results: true`, a completed non-fallback result is rebased to its
completion time, starts a new validity window, and releases that hold.

## Explicit hardware enable

Hardware output is intentionally unavailable from the launch file. After a
physical preflight, run the node directly with `--execute`:

```bash
ros2 run mm_run stretch_command_adapter \
  --config "$(ros2 pkg prefix mm_run)/share/mm_run/config/stretch_command_adapter_real.yaml" \
  --execute
```

Enable requires all of the following:

- a fresh, exactly zero 11-dimensional velocity request;
- fresh synchronized map/odom/joint state;
- `mode=navigation`, `homed=true`, `runstopped=false`;
- streaming-position initially inactive;
- zero existing publishers on `/stretch/cmd_vel` and `/joint_pose_cmd`.

After enable, status changes from `shadow` to `wbmpc`. A solver overrun,
fallback, or temporary lack of plan causes a recoverable soft hold. A stale
command transport, stale state, invalid frame/timestamp, lateral or arm
projection violation, following error, mode/runstop change, control-period
violation, or a second hardware command publisher still latches the adapter.
The latched process continuously publishes zero Twist and retries
streaming-position deactivation. Restart it only after investigating the reason.
