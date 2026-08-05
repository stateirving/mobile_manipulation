# Stretch 实机启动、遥控与 ESDF 采集 Runbook

最后验证：2026-08-05

本文档记录当前已验证的启动顺序。除特别说明外，每个代码块都在一个独立终端中运行，
不要在同一终端覆盖仍需保持运行的进程。

## 1. 固定信息与安全要求

| 项目              | 当前值                                                                         |
| ----------------- | ------------------------------------------------------------------------------ |
| 机器人 SSH        | `hello-robot@192.168.50.173`                                                 |
| 机器人仓库        | `~/repos/bringup_active_mapmaintenance/online_bringup_active_mapmaintenance` |
| PC ROS/Zenoh 环境 | `~/repo/bringup_active_mapmaintenance/perceive_semantix`                     |
| PC 算法仓库       | `~/repo/mobile_manipulation`                                                 |
| rosbag 输出根目录 | `/home/miao/data/real_stretch_esdf_bags`                                     |
| ROS middleware    | `rmw_zenoh_cpp`                                                              |

安全要求：

- 开始运动前清空机器人周围空间，并确保 runstop 随时可用。
- 普通 Stretch launch 和 `launch-teleop` 不能同时运行。
- ESDF 采集期间只允许一个底盘命令源；不要同时运行 Nav2、旧 MPC 或其他 teleop。
- 先启动录制，再移动机器人；结束时先停车，再停止 rosbag。
- 不在仓库中记录 SSH 密码；使用 SSH key 或在交互提示中输入凭据。

## 2. 终端与启动顺序

### R1 — 机器人 Zenoh

```bash
ping 192.168.50.173
ssh hello-robot@192.168.50.173
cd ~/repos/bringup_active_mapmaintenance/online_bringup_active_mapmaintenance
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
pixi run -e zenoh launch
```

### P1 — PC Zenoh router

```bash
cd ~/repo/bringup_active_mapmaintenance/perceive_semantix
pixi shell -e zenoh
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
RUST_LOG=zenoh=info ros2 run rmw_zenoh_cpp rmw_zenohd
```

### R2 — Stretch driver：二选一

普通模式用于只读检查和 homing：

```bash
ssh hello-robot@192.168.50.173
cd ~/repos/bringup_active_mapmaintenance/online_bringup_active_mapmaintenance
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
pixi run -e stretch-ros2-zenoh launch
```

ESDF 遥控采集使用 gamepad 模式：

```bash
ssh hello-robot@192.168.50.173
cd ~/repos/bringup_active_mapmaintenance/online_bringup_active_mapmaintenance
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
pixi run -e stretch-ros2-zenoh launch-teleop
```

从普通模式切换到遥控模式时，必须先在 R2 中按 `Ctrl-C` 完全停止普通 launch，再运行
`launch-teleop`。`launch-teleop` 会启动 gamepad 模式和 `stretch_ps4_control`：它订阅
PC 的 `/joy`，发布 `/gamepad_joy` 给 Stretch driver。

### R3 — SAI/Orbbec SLAM

此进程在普通模式和遥控采集模式下都保持运行：

```bash
ssh hello-robot@192.168.50.173
cd ~/repos/bringup_active_mapmaintenance/online_bringup_active_mapmaintenance
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
pixi run -e ros2-orbbec-slam-zenoh launch
```

### P2 — PC ROS 操作终端

```bash
cd ~/repo/bringup_active_mapmaintenance/perceive_semantix
pixi shell -e zenoh
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
```

普通 driver 运行时，可在 P2 中执行 homing：

```bash
ros2 node list
ros2 service call /home_the_robot std_srvs/srv/Trigger
```

如需随后遥控采集，homing 完成后按前述要求停止普通 driver，再启动 `launch-teleop`。

## 3. PS4 手柄遥控

PS4 手柄连接到 PC。先确认设备存在：

```bash
ls -l /dev/input/js0
```

### P3 — PC joystick publisher

```bash
cd ~/repo/bringup_active_mapmaintenance/perceive_semantix
pixi shell -e zenoh
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
ros2 run joy joy_node --ros-args \
  -r __node:=pc_joy_node \
  -p device_id:=0 \
  -p deadzone:=0.1 \
  -p autorepeat_rate:=20.0
```

已验证的通信链：

```text
PC /dev/input/js0
  -> /joy (Zenoh)
  -> robot stretch_ps4_control
  -> /gamepad_joy
  -> Stretch driver (gamepad mode)
```

### P2 — 遥控链验证

```bash
ros2 topic echo /mode --once
ros2 topic hz /joy
ros2 topic echo /gamepad_joy --once
```

预期 `/mode` 为 `gamepad`。`ros2 topic hz /joy` 会持续运行，看到稳定频率后按
`Ctrl-C` 退出。

## 4. ESDF 录制前检查

在 P2 中确认深度、内参和定位链都已产生真实消息：

```bash
ros2 topic info -v /spectacular_ai/depth_image

timeout 15s ros2 topic echo \
  /spectacular_ai/depth_image \
  --field header \
  --once

timeout 15s ros2 topic echo \
  /spectacular_ai/camera_info \
  --field header \
  --once

timeout 15s ros2 run tf2_ros tf2_echo \
  map camera_color_optical_frame
```

通过条件：

- depth frame 为 `camera_color_optical_frame`；
- depth、CameraInfo 和 `map -> camera_color_optical_frame` 均可获得；
- `/mode` 为 `gamepad`；
- R2 只有一个 Stretch driver launch。

SAI depth 是低频关键帧，不是 30 Hz raw stream。短时间没有新帧时，应先轻微改变视角，
不能仅根据 publisher endpoint 存在判断相机正在持续出图。

## 5. 录制实机 ESDF rosbag

### P4 — rosbag recorder

在新的 PC Zenoh shell 中运行。修改 `ESDF_BAG_NAME`，确保每次采集使用新目录：

```bash
cd ~/repo/bringup_active_mapmaintenance/perceive_semantix
pixi shell -e zenoh
export RMW_IMPLEMENTATION=rmw_zenoh_cpp

ESDF_BAG_ROOT=/home/miao/data/real_stretch_esdf_bags
ESDF_BAG_NAME="$(date +%F_%H-%M-%S)_esdf_scan"
mkdir -p "$ESDF_BAG_ROOT"

ros2 bag record \
  -o "$ESDF_BAG_ROOT/$ESDF_BAG_NAME" \
  /mode \
  /stretch/joint_states \
  /odom \
  /tf \
  /tf_static \
  /spectacular_ai/camera_info \
  /spectacular_ai/depth_image \
  /spectacular_ai/color_image \
  /spectacular_ai/map
```

该 topic 集合与已成功的 `2026-08-05_stage0_motion_01` 一致。若还要保存遥控输入审计，
可在命令末尾追加 `/joy /gamepad_joy`。

看到 recorder 已订阅各 topic 后再开始移动。建议采集动作：

1. 原地缓慢旋转，覆盖四周；
2. 做小幅前后平移，增加视差；
3. 缓慢改变 head pan/tilt，覆盖低处、桌面和障碍边缘；
4. 每个关键视角短暂停留，等待 SAI 发布关键帧；
5. 不靠近人员、玻璃、镜面或未清空区域。

结束时先将手柄回中并确认机器人停止，再在 P4 按 `Ctrl-C`。记录终端打印的实际 bag
路径，不要覆盖已有 bag。

### 录制结果验证

仍在 P4 中运行：

```bash
ros2 bag info "$ESDF_BAG_ROOT/$ESDF_BAG_NAME"
```

至少确认以下 topic count 非零：

- `/spectacular_ai/depth_image`
- `/spectacular_ai/camera_info`
- `/tf` 和 `/tf_static`
- `/stretch/joint_states`
- `/odom`

如果换了终端，重新设置 `ESDF_BAG_ROOT` 和具体的 `ESDF_BAG_NAME`，或者直接给
`ros2 bag info` 传绝对路径。

## 6. 在 PC 离线导出 ESDF

当前阶段推荐 rosbag 录制后离线转换。命令需要 PC NVIDIA GPU：

```bash
cd ~/repo/mobile_manipulation

pixi run python mm_run/scripts/export_real_rosbag_esdf.py \
  /home/miao/data/real_stretch_esdf_bags/REPLACE_WITH_BAG_NAME \
  -o mm_run/results/nvblox_esdf/real_bag/REPLACE_WITH_BAG_NAME \
  --bounds XMIN YMIN ZMIN XMAX YMAX ZMAX \
  --voxel-size 0.05 \
  --grid-resolution 0.05 \
  --ground-min-z 0.08
```

`--bounds` 必须按该 bag 的 `map` 坐标深度端点范围设置，并在每个方向保留至少
0.30–0.50 m 余量；不要直接复用另一袋数据的边界。已验证示例：

```text
2026-08-05_stage0_motion_01: -4.2 -4.2 -0.2 4.2 4.2 2.2
2026-08-05_stage0_motion_02: -4.2 -5.9 -0.2 5.7 4.2 4.0
```

导出后应检查 `|distance| <= surface_band` 的零表面是否落入最外 3 层体素；若是，扩大
对应方向后重新导出。边界还必须完整覆盖 planner 的 `bounds_xy`，否则边界外查询会被
正确判为 invalid，但会无意中缩小可规划区域。

默认启用双地图地面处理，输出包括：

```text
esdf_grid.npz
map.nvblox
observed_space.nvblox
metadata.json
esdf_surface_preview.png
esdf_surface_band.ply
slices/
```

检查终端最后的 `known`、`valid` 和 `start_valid`。`start_valid=False` 时不要把地图交给
OMPL/WB-MPC。

## 7. 可视化导出的 ESDF

交互式 PyBullet 视图：

```bash
cd ~/repo/mobile_manipulation

pixi run python mm_run/scripts/visualize_esdf_npz.py \
  mm_run/results/nvblox_esdf/real_bag/REPLACE_WITH_BAG_NAME/esdf_grid.npz \
  --surface-band 0.08 \
  --color-mode height
```

鼠标旋转、平移和缩放，按 `Q` 关闭。检查 ESDF 正负方向时使用
`--color-mode distance`。

静态预览：

```bash
xdg-open \
  mm_run/results/nvblox_esdf/real_bag/REPLACE_WITH_BAG_NAME/esdf_surface_preview.png
```

## 8. 停止顺序

1. 手柄回中，确认机器人完全停止。
2. P4：`Ctrl-C` 停止 rosbag，并运行 `ros2 bag info`。
3. P3：`Ctrl-C` 停止 `joy_node`。
4. R3：`Ctrl-C` 停止 SAI/Orbbec SLAM。
5. R2：`Ctrl-C` 停止 Stretch driver/teleop。
6. P1、R1：最后停止两端 Zenoh。

异常情况下优先触发 runstop；不要为了保存 rosbag 而延迟停车。

## 9. 当前已知限制

- `depth_scale=0.001 m/unit` 尚未通过已知距离独立冻结。
- 实机转换器已有 map-Z 地面过滤和 observed-space 双地图，但尚无 robot self mask。
- SAI depth 是低频关键帧；完整覆盖需要底盘平移与 head pan/tilt 扫描。
- 当前离线 ESDF 可用于接口、几何和 planner-valid 验证，尚未授权实机 WB-MPC 下发。
