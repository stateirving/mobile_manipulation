# Real Stretch command adapter

The command adapter starts in `shadow` mode. In this mode it subscribes to the
validated Stretch state and an 11-dimensional WB-MPC velocity request, but does
not create publishers for `/stretch/cmd_vel` or `/joint_pose_cmd`.

The adapter republishes its validated named 11-position/11-velocity feedback as
`sensor_msgs/msg/JointState` on `/wbmpc/state`. The separate WB-MPC shadow node
uses that state and publishes only `/wbmpc/velocity_command`. Adapter status has
two intentional operating states: `shadow` and `wbmpc` (plus `latched` after a
stop). Base-target versus arm-target behavior remains inside TaskManager/MPC.

The velocity input topic is `/wbmpc/velocity_command`, using
`std_msgs/msg/Float64MultiArray` in this exact order:

```text
[base_vx_map, base_vy_map, base_yaw_rate,
 lift_velocity,
 arm_l3_velocity, arm_l2_velocity, arm_l1_velocity, arm_l0_velocity,
 wrist_yaw_velocity, wrist_pitch_velocity, wrist_roll_velocity]
```

This input is a velocity, not the WB-MPC `u_bar` acceleration. A real runner
using `cmd_vel_type: integration` must first integrate the acceleration into the
same velocity contract used by the simulation runner.

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
  std_msgs/msg/Float64MultiArray \
  '{data: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}'

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

The WB-MPC source publishes zero while state is unavailable, while the first
OMPL plan/solve is running, after a solver fallback, or when a plan becomes
stale. The adapter independently applies its stricter physical command limits.

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

After enable, status changes from `shadow` to `wbmpc`. A later velocity message
may request motion. A stale state or
command, invalid frame/timestamp, lateral or arm projection violation, following
error, mode/runstop change, control-period violation, or a second hardware
command publisher latches the adapter. The
latched process continuously publishes zero Twist, holds the last valid joint
feedback, and retries streaming-position deactivation. Restart the process only
after investigating the stop reason.
