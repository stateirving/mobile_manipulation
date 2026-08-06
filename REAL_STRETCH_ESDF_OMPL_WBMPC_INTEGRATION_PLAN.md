# Stretch 实机接入：ESDF 采集 + Offline OMPL + WB-MPC 计划

更新日期：2026-08-06

## 1. 目标与边界

目标是在不复制或替换 `bringup_active_mapmaintenance` 的前提下，让本仓库运行在工作站上，通过现有 ROS 2 + Zenoh 网络接入 Stretch 实机，并复用本仓库已经在 PyBullet 中验证的以下能力：

1. 从真实深度相机数据采集并导出离线 `esdf_grid.npz`。
2. 使用该离线 ESDF 进行 OMPL 底盘路径规划和末端路径规划。
3. 使用全身 MPC（WB-MPC）跟踪规划结果，并将输出安全地拆分为 Stretch 底盘和机械臂命令。
4. 机器人全局位姿支持 SLAM 和 Vicon 两种来源，但规划、ESDF 和控制层只使用统一的 `map` 坐标系接口。

本计划只描述后续实施步骤。本次不修改已有 Python、YAML、launch、URDF 或 bringup 仓库。

## 2. 当前基线与已确认事实

### 2.1 实机 bringup

依据本仓库的 `REAL_DEPLOY.md` 和本机只读副本 `/home/miao/repo/bringup_active_mapmaintenance`：

- Stretch 本机通过 `online_bringup_active_mapmaintenance` 启动 ROS 2 Humble、`rmw_zenoh_cpp` 和 Stretch 驱动。
- 工作站和机器人各运行 Zenoh router，工作站上的本仓库作为同一 ROS graph 中的远程节点运行。
- Stretch 驱动当前以 `navigation` 模式启动，已确认的重映射为：
  - 状态：`/stretch/joint_states`，类型 `sensor_msgs/msg/JointState`。
  - 底盘命令：`/stretch/cmd_vel`，类型 `geometry_msgs/msg/Twist`。
- 2026-08-05 的首次实机 graph 快照同时出现 `/joint_states` 和
  `/stretch/joint_states`。后续只读核查确认：`/stretch/joint_states` 由
  `stretch_driver` 直接以 reliable/volatile/keep-last-1 发布；`/joint_state_publisher`
  订阅它后在 `/joint_states` 上发布用于 `robot_state_publisher` 的聚合消息。
  实机状态适配器应使用前者，不应把后者当作原始驱动反馈。
- `slam_toolbox` 当前运行同步建图节点，栅格地图 topic 从 `/map` 重映射到了 `/lidar_map`。这不会自动改变 TF 中的 `map` frame 名称。
- Spectacular AI/Orbbec 不随 Stretch bringup 启动，由独立的
  `ros2-orbbec-slam-zenoh` 环境运行。感知代码默认订阅
  `/spectacular_ai/camera_info`、`/spectacular_ai/color_image` 和
  `/spectacular_ai/depth_image`，最终实机配置仍以运行时 graph 快照为准。
- 机器人端已确认存在
  `~/repos/bringup_active_mapmaintenance/online_bringup_active_mapmaintenance/ros2_orbbec_slam`，
  外层 `ros2-orbbec-slam-zenoh` pixi 环境组合 Orbbec/Spectacular AI 与
  `rmw_zenoh_cpp==0.1.2`，SDK 来自同级 `third_party/spectacularAI` 和
  `third_party/OrbbecSDK`。
- 外层 bringup 仓库的 gitlink 与机器人端 `ros2_orbbec_slam` 实际 commit 已确认一致：
  `31515dd3252b18232a59cb685f005d6d0829356a`。工作站副本没有初始化该
  submodule；阶段 0 已另行只读检出相同 commit 核查其 launch/发布源码。
- `sai_orbbec` 已在机器人端完成构建并可正常启动；实机 ESDF 使用
  `base_link <-> camera_color_optical_frame` 标定外参。
- `/spectacular_ai/depth_image` 不是 Orbbec 的 30 Hz raw stream。commit
  `31515dd...` 的 `sai_publisher` 只遍历 Spectacular AI
  `MapperOutput.updatedKeyFrames`，并在更新关键帧存在时发布对齐后的
  `16UC1` depth、RGB 和 `CameraInfo`；机器人静止时该 topic 可能长时间没有新消息。
  depth 通过 `getAlignedDepthFrame(rgbFrame)` 对齐到 RGB；三类消息共用回调时刻的
  ROS `now()`，不是传感器原始采集时间。当前源码未设置
  `CameraInfo.header.frame_id`，适配器需按已冻结的 RGB optical frame 补齐并校验。
- `/home_the_robot` 已确认为 `std_srvs/srv/Trigger`，并有 `/is_homed`
  (`std_msgs/msg/Bool`) 可供状态检查；但本次快照未记录其值和 homing
  完成条件。
- 机械臂侧已确认暴露
  `/stretch_controller/follow_joint_trajectory`
  (`control_msgs/action/FollowJointTrajectory`)；同时存在 streaming-position 激活/停用服务、
  `/joint_pose_cmd` (`Float64MultiArray`) 以及 navigation/position/trajectory 模式切换服务。
  2026-08-06 后续低速测试确认 action 接受 `joint_lift`、`wrist_extension` 和
  `joint_wrist_yaw` 的两点轨迹；单点轨迹会 abort。另一路 streaming-position 已在
  `navigation` 模式下以 10 Hz 与底盘 Twist 并发通过低幅实机验证。它仍不等于完整的
  WB-MPC 命令安全验收：超时、仲裁、全关节跟踪误差和故障注入尚待阶段 3 完成。
- 核查时 `/stretch/cmd_vel` 的 publisher count 为 0，`stretch_driver` 是唯一
  subscriber；trajectory action server 属于 `/stretch_driver`，当时 client count 为 0。
- 核查时驱动状态为 `navigation`、homed=true、runstopped=false、
  streaming-position=false。这些是单次运行时状态，不能硬编码为默认假设。

### 2.2 本仓库现状

- `mm_run/scripts/teleop_export_esdf.py` 只从 PyBullet 相机采集，不订阅 ROS 图像或 TF。
- 导出的 `esdf_grid.npz` 已有稳定读取接口：`mm_control.esdf_map.ESDFMap`。
- `stretch_esdf_offline_ompl_wbmpc.yaml` 已串起离线 ESDF、OMPL base/EE planner 和 WB-MPC。
- `mm_run/scripts/experiment.py` 将规划/控制循环直接绑定到了 `BulletSimulation`，不能直接用于实机。
- `mm_run/nodes/mpc_ros.py` 虽然是 ROS 2 节点，但依赖当前环境中不存在的
  `mobile_manipulation_central` 接口，并不是现有 Stretch bringup 的适配器。
- 当前控制模型状态为 11 维：
  `[base_x, base_y, base_yaw, lift, arm_l3, arm_l2, arm_l1, arm_l0, wrist_yaw, wrist_pitch, wrist_roll]`。
- 控制 URDF 与 bringup URDF 的核心关节名称相近，但轮子、头部、夹爪的可动/固定定义不同；不能只凭同名假设两者运动学和零位完全一致。

### 2.3 当前不能直接连实机的三个阻塞项

1. **状态契约已完成阶段 1 验证**：已完成按名关节映射、限位/时间戳检查、
   `wrist_extension == sum(joint_arm_l*)`、base 三维状态、`/odom.twist`
   frame/符号、完整 11+11 状态同步，以及模型 FK 对 live TF 的多姿态验证。
   详见阶段 1 的 2026-08-06 验证记录。
2. **命令安全适配器尚未实现**：WB-MPC 输出 11 维速度；底盘
   `/stretch/cmd_vel` 与 streaming-position 的协议和 `navigation` 并发能力已经确认，
   但仍需把速度积分为受限位置目标，并实现 receive-time watchdog、唯一 command owner、
   跟踪误差门限和锁存停止后，才能接入 WB-MPC 实机输出。
3. **坐标系尚未闭合**：ESDF、OMPL 和 MPC 必须使用同一个固定坐标系；SLAM 和 Vicon 的原点、时间戳、漂移/跳变语义不同，不能直接替换 topic 名。

### 2.4 阶段 0 首次实机快照（2026-08-05）

快照来自工作站
`bringup_active_mapmaintenance/perceive_semantix` 的 `zenoh` pixi 环境。本次未记录实际采集时刻；
`2026-08-05` 是证据纳入本计划的日期。

已确认：

- 主要节点包括 `/stretch_driver`、`/robot_state_publisher`、`/slam_toolbox`、
  `/sllidar_node`、`/joint_state_publisher` 和 `/laser_filter`。
- ROS graph 警告存在同名节点；列表中 `/laser_filter` 出现两次，需确认是否为重复启动。
- 存在 `/odom` (`nav_msgs/msg/Odometry`)、`/pose`
  (`geometry_msgs/msg/PoseWithCovarianceStamped`)、`/tf`、`/tf_static` 和
  `/lidar_map` (`nav_msgs/msg/OccupancyGrid`)。
- 存在 `/stretch/joint_states` 和 `/joint_states`，两者类型均为
  `sensor_msgs/msg/JointState`；存在 `/joint_limits` (`JointState`)。
- 存在 `/stretch/cmd_vel` (`geometry_msgs/msg/Twist`)。
- 机械臂 trajectory action、streaming-position 服务、模式切换服务、homing/stow/stop
  服务以及 runstop/self-collision-avoidance 服务均在 graph 中。
- 存在 `/mode`、`/is_homed`、`/is_runstopped` 和
  `/is_streaming_position` 状态 topic。
- `/stretch/joint_states` 一帧包含四段 arm、lift、wrist 三轴、头部、夹爪以及
  `wrist_extension`，position/velocity/effort 数组均完整；四段 arm 位置之和与
  `wrist_extension` 一致。
- `/joint_states` 由 `joint_state_publisher` 二次发布，添加了左/右轮关节，
  但不含 `wrist_extension`。
- `/odom` 由 `stretch_driver` 发布，`header.frame_id=odom`、
  `child_frame_id=base_link`；静止时 twist.linear.y=0。
- 约 10 秒的工作站侧测量中，`/stretch/joint_states` 和 `/odom` 均稳定在
  约 30 Hz，`/scan_filtered` 约 8.16 Hz。
- 原始 `/tf` 和 `/tf_static` 能跨 Zenoh 收到；机器人本体树从
  `base_link` 延伸到 arm/wrist/gripper，包含 `link_grasp_center`、
  `camera_color_optical_frame` 和 `gripper_camera_depth_optical_frame` 等静态 frame。
- `/spectacular_ai/sai_publisher` 已在 graph 中，发布 camera info、color/depth image、
  global/local point cloud、map point cloud、pose 和 TF 接口。这些 publisher 的 QoS 均为
  reliable/volatile/keep-last-1。
- `sai_publisher` 参数为 `base_frame=base_link`、
  `camera_frame=camera_color_optical_frame`、`global_frame=map`、
  `use_tf_frames=true`、`publish_tf_instead_of_pose=true`。因此定位输出首选 TF，
  不依赖 `/spectacular_ai/pose` topic。
- `camera_color_optical_frame <-> base_link` 已可持续查询；当次快照中
  `map` frame 尚未出现，因此 `map -> base_link` 和 `map -> camera` 不可用。
- 相机重新枚举并重启 pipeline 后，工作站已收到 frame
  `camera_color_optical_frame` 的 depth 消息；源码确认 encoding 为 `16UC1`。
  45 秒静止测量不足两帧，符合其“更新关键帧”而非 raw stream 的发布语义。
- `map -> base_link` 也只在含 RGB 的更新关键帧回调中作为动态 `/tf` 发布；
  新启动的 volatile TF 订阅者需等待下一关键帧，不能把 endpoint 存在等同于
  任意时刻均可查询 `map` frame。

### 2.5 阶段 0 运动 rosbag 快照（2026-08-05）

- 工作站外部数据目录
  `/home/miao/data/real_stretch_esdf_bags/2026-08-05_stage0_motion_01`
  已录制 53.379 s、380.6 MiB、5373 条消息；大型 bag 不进入 Git。
- 关键消息计数：depth 33、CameraInfo 121、RGB 118、SAI map 121、动态 TF
  1506、静态 TF 1、`/stretch/joint_states` 1158、`/odom` 1157。
- 33 帧 depth 均为 `1280x720`、`16UC1`、little-endian、step 2560、frame
  `camera_color_optical_frame`；每一帧都存在时间戳完全相等的 CameraInfo、RGB 和
  `map -> base_link` TF。
- 实机 CameraInfo 固定为 `1280x720`，`K=[750.025, 0, 636.264, 0, 749.733, 369.197, 0, 0, 1]`；`D=[]`、`distortion_model` 和
  `header.frame_id` 为空。它不同于仓库中基于 Femto Mega USD 的旧配置，实机适配器
  必须采用 bag 中的 K 并显式补齐已冻结的 optical frame。
- depth 非零 raw 值范围 510..5945，中位数 1466，零值占 40.89%；数值强烈符合
  毫米语义，但 `depth_scale=0.001 m/unit` 仍需通过已知距离或 SDK scale API 最终冻结。
- depth 的运动时平均 cadence 为 0.766 Hz，间隔中位数 0.825 s、最大 5.229 s；
  recorder 接收时间比消息 header 晚 125..192 ms（中位数 151 ms）。
- 本次 TF 中只有 SAI `map -> base_link`（125 条），没有 `map -> odom` 或
  `odom -> base_link`，也没有同一 child 的多 parent；因此该 bag 内没有 TF authority
  冲突，但这不是计划中常规 `map -> odom -> base_link` 链，定位适配器必须显式支持。
- 本次主要是底盘运动：里程计路径约 0.525 m、累计 yaw 约 -544°；head pan/tilt
  变化均小于 0.007 rad。该 bag 足以验证接口和首次离线 ESDF，不代表完整空间覆盖。
- 最后一帧 SAI map 含 2424 个有限点，header frame 为 `map`；其 raw 点范围较大，
  仅用于诊断，ESDF 仍以同步 depth + CameraInfo + TF 为输入。
- 工作站 NVIDIA RTX 4060 Laptop GPU、driver 595.84、PyTorch 2.9.1+cu128 已确认；
  `torch.cuda.is_available()` 为 true，可运行当前 CUDA-only nvblox 路径。

### 2.6 首次实机 ESDF 离线导出（2026-08-05）

- 已增加 `mm_run/scripts/export_real_rosbag_esdf.py`，离线读取上述 bag 的
  `/spectacular_ai/depth_image`、CameraInfo、`/tf` 和 `/tf_static`，按每帧 depth
  header 时间插值并组合 `T_map_camera`，再由 nvblox-torch 构建 TSDF/ESDF。
- 首次导出使用 `depth_scale=0.001 m/unit`、depth 0.25..4.0 m、voxel/grid 0.05 m、
  bounds `[-4.2,-4.2,-0.2]..[4.2,4.2,2.2]`，并按 endpoint `map z >= 0.08 m`
  做临时地面过滤。33/33 depth 帧完成融合，共使用 13,554,973 个有效深度像素；
  相机高度约 1.298..1.300 m，动态 TF 插值年龄最大 32.94 ms、P95 31.65 ms。
- 已接入与仿真一致的双地图地面处理：过滤 `map z < 0.08 m` 终点的 obstacle TSDF
  只提供非地面障碍距离；第二张 occupancy map 使用未过滤深度保留相机射线的
  observed-free 证据。导出时只将 occupancy log-odds `< 0` 的体素从 unknown 补为
  valid，并从非地面障碍 site 传播距离（上限 2.0 m）；未观察体素仍为 invalid。
- nvblox 未观测区哨兵距离为 100 m，导出器用运行时常量将其标为 invalid。固定查询网格
  1,399,489 点中，原始 obstacle ESDF 有效 188,500 点；occupancy 确认 1,022,640 个
  observed-free 点并补齐 888,208 点，最终有效 1,076,708 点（76.94%）。
  `|distance| <= 0.08 m` 的零表面带有 35,600 点；自由空间补齐不引入地面零表面。
- 按当前 offline OMPL base 配置（`query_z=[0.15,0.35]`、`base_radius=0.20 m`、
  `d_safe=0.20 m`、5 cm lattice、XY bounds ±4 m）自动验收：两层均 known 的 lattice
  占 70.50%，planner-valid 占 59.59%，共有 14 个连通分量。bag 的精确首帧
  `map -> base_link` 起点约 `(-0.0002, 0.0002)`，两层 clearance 约 0.543/0.450 m，
  `start_valid=true`；目标仍必须检查是否属于同一连通分量。
- 产物目录为
  `mm_run/results/nvblox_esdf/real_bag/2026-08-05_stage0_motion_01/`，包含
  `esdf_grid.npz`、`map.nvblox`、`observed_space.nvblox`、`metadata.json`、四个高度切片、
  `esdf_surface_band.ply` 和 `esdf_surface_preview.png`。NPZ 已可由
  `mm_control.esdf_map.ESDFMap` 加载；metadata 保存双地图融合统计、全部
  `T_map_camera`/`T_map_base` 和 planner-quality 报告。
- 复现命令（在仓库根目录）：

  ```bash
  pixi run python mm_run/scripts/export_real_rosbag_esdf.py \
    /home/miao/data/real_stretch_esdf_bags/2026-08-05_stage0_motion_01 \
    -o mm_run/results/nvblox_esdf/real_bag/2026-08-05_stage0_motion_01 \
    --bounds -4.2 -4.2 -0.2 4.2 4.2 2.2 \
    --voxel-size 0.05 --grid-resolution 0.05 --ground-min-z 0.08
  ```
- 双地图已消除首版“过滤地面后起点上方成为 unknown”的 invalid 问题，但当前结果仍未
  通过完整 planner-ready 门：depth scale 尚未独立标定、没有机器人 self mask，本次平移
  覆盖很小且只有 33 个低频关键帧，并存在 14 个 planner-valid 连通分量。因此图中的空洞
  或断裂不能直接判定为可通行区域，目标必须在同一连通分量内单独验收。

剩余的明确缺口：

- 尚未通过人工低速正/反向运动确认 `/odom.twist` 的实际方向和 yaw 符号。
- 已完成该 bag 的 TF parent 检查；仍需长时检查 SAI `map -> base_link` 的连续性、
  闭环跳变和最终 authority 选择。
- depth/CameraInfo/TF 同步已确认；仍需冻结 depth scale，并补做 head pan/tilt
  扫描以评估关键帧空间覆盖。尚未发现 Vicon 接口。
- action server、driver mode 以及 lift/extension/wrist-yaw 两点轨迹已确认；仍未确认
  wrist pitch/roll 完整保持、goal 抢占/cancel/hold 及与底盘 Twist 的并发行为。
- 未记录 commit/version、Zenoh 时延/丢帧和两台机器的时钟偏差；
  当前频率测量仅是短时快照，仍需长时统计。

## 3. 目标架构

```text
机器人（bringup_active_mapmaintenance）
  深度/CameraInfo ───────────────┐
  /stretch/joint_states ────────┤        Zenoh        工作站（本仓库）
  /odom + /tf + /tf_static ─────┼──────────────────>  实机状态适配器
  SLAM 或 Vicon 位姿 ────────────┘                         │
                                                         ├─> 统一 q/v（map frame）
  实机 ESDF 采集节点 <── 深度 + map<-camera TF             │
          │                                              v
          └─> esdf_grid.npz ─> ESDFMap ─> OMPL ─> WB-MPC
                                                         │
                                        实机命令/安全适配器
                                           ├─> /stretch/cmd_vel
                                           └─> 机械臂受支持的控制接口
```

架构约束：

- `bringup_active_mapmaintenance` 负责硬件驱动、传感器、基础 TF、SLAM 和 Zenoh；本仓库不复制它的 bringup。
- 本仓库负责 ESDF 生成、规划、WB-MPC、状态/命令适配、日志和安全监督。
- 离线运行时只加载冻结的 `esdf_grid.npz`，不在控制回路中更新地图。
- `map` 是 ESDF、OMPL target、机器人 base pose 和 EE pose 的唯一规划坐标系。
- SLAM/Vicon 只在定位适配层分支，下游 `TaskManager`、OMPL 和 MPC 不感知定位来源。

## 4. 统一 ROS 接口契约

下表中的“待确认”项必须在阶段 0 从真实 ROS graph 获取，不能根据源码猜测。

| 用途               | 首选接口                                                                                  |                                                                                      状态 | 适配要求                                                                                      |
| ------------------ | ----------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------: | --------------------------------------------------------------------------------------------- |
| 关节反馈           | `/stretch/joint_states` (`JointState`)                                                |                                                           四段 arm 及速度已确认，约 30 Hz | 按`name` 映射，禁止依赖数组顺序；校验聚合 `wrist_extension` 与四段之和                    |
| 底盘速度反馈       | `/odom` (`Odometry`)                                                                  |                                       `odom`/`base_link` 已确认，符号待验证，约 30 Hz | 转成`map` 或 `base_link` 下定义明确的速度；禁止混用 world/body velocity                   |
| SLAM 位姿          | 当前 SAI 为 TF`map -> base_link`；常规链为 `map -> odom -> base_link`                 | 运动 bag 捕获 125 条 SAI direct TF，无`map -> odom`/`odom -> base_link` 且无多 parent | 定位适配器显式区分 direct/chained TF；采集节点先启动并等待有效 TF，持续检查跳变与 authority   |
| Vicon 位姿         | Vicon rigid-body topic/TF                                                                 |                                                                                    待确认 | 应用`vicon_world -> map` 和 marker -> `base_link` 外参                                    |
| 深度关键帧         | `/spectacular_ai/depth_image` (`Image`)                                               |      运动 bag 33 帧；`1280x720 16UC1`，与 CameraInfo/RGB/TF 精确同 stamp，平均 0.766 Hz | 冻结`0.001 m/unit` 候选 scale；按关键帧使用并执行覆盖率质量门，禁止按 30 Hz raw stream 使用 |
| 相机内参           | `/spectacular_ai/camera_info` (`CameraInfo`)                                          |                                    运动 bag 121 条；实机 K 已冻结候选，D/model/frame 为空 | 使用实机 K；按`camera_color_optical_frame` 补 frame，并将空畸变解释写入 metadata            |
| 原始深度（可选）   | 尚无 ROS topic                                                                            |                                          当前`sai_orbbec` 不发布每个 Orbbec depth frame | 若关键帧密度不满足 ESDF 质量门，再增加独立 raw-depth adapter；避免与 SAI 同时独占 USB         |
| 相机位姿           | TF`map <- camera_color_optical_frame`                                                   |          33 帧 depth 均有精确同 stamp 的`map -> base_link`，且静态/关节 TF 可组合到相机 | bag 离线 tf2 lookup 必须支持 SAI direct`map -> base_link` 并验证完整组合变换                |
| 底盘命令           | `/stretch/cmd_vel` (`Twist`)                                                          |                                                        驱动订阅已确认，核查时无 publisher | 将 MPC 的 map-frame base velocity 转成 body-frame 非完整约束命令                              |
| 机械臂命令         | `/stretch_controller/follow_joint_trajectory`                                           |                  已确认 lift、`wrist_extension`、wrist yaw 两点轨迹；单点 goal 会 abort | trajectory 模式切换曾造成反馈中断，不选作当前 WB-MPC 高频入口；cancel/hold 仍作为故障恢复候选验证 |
| Streaming position | `/activate_streaming_position`、`/deactivate_streaming_position`、`/joint_pose_cmd` | `navigation` 下 10 Hz 与 Twist 并发通过；SG3 为固定 10 元素全量位置向量；小幅 yaw 返回误差约 0.007 rad | 每帧由新鲜反馈/内部目标构造完整向量，base 两项置零；外置 receive-time watchdog、仲裁和跟踪误差门 |
| Homing             | `/home_the_robot` (`Trigger`) + `/is_homed` (`Bool`)                              |                                                                                接口已确认 | 控制节点只检查状态；是否自动调用由 launch 参数决定                                            |
| 急停/停止          | `/is_runstopped`、`/runstop`、`/stop_the_robot` + 零 Twist + 机械臂 cancel/hold     |                                                              接口/type 已确认，行为待确认 | 必须为锁存故障，不能在下一帧自动恢复                                                          |

接口配置全部参数化，默认值只能在完成 graph 快照后写入实机 YAML。

## 5. 定位方案

### 5.1 统一输出

实现一个定位适配器，对上游暴露两种 `source`：`slam` 和 `vicon`，但始终输出：

- `T_map_base(t)`：机器人底盘在 `map` 中的 SE(2) 位姿。
- `v_base(t)`：带明确 frame 语义的底盘线速度和角速度。
- 状态时间戳、数据龄期、来源健康状态和最近一次跳变大小。

运行中默认禁止 SLAM/Vicon 自动热切换。切换来源前先停止机器人、锁存零命令、重新检查 `T_map_base` 连续性，再人工重新使能控制。

### 5.2 SLAM 分支

1. 从真实 graph 确认 `map -> odom` 和 `odom -> base_link` 的发布者、频率和 authority。
2. 确保全链每段只有一个 authority，避免重复 TF。
3. ESDF 采集阶段保存 SLAM pose graph/地图及其初始位姿信息。
4. 离线 ESDF 执行阶段优先运行 SLAM localization 模式，而不是继续无约束建图，避免 loop closure 或重定位导致 `map` 突跳。
5. 监控 `map -> base_link` 的平移/旋转跳变；超过阈值立即停止，不把跳变后的状态送入 MPC 连续控制。

注意：`/lidar_map` 是 OccupancyGrid topic 名，而 `map` 是规划 frame；两者不要因为命名相似而混为一体。

### 5.3 Vicon 分支

1. 从实机 graph 确认底盘 rigid body 的 topic、frame、消息类型和时间戳来源；末端 Vicon 不能替代底盘定位。
2. 标定 Vicon marker frame 到 `base_link` 的刚体外参。
3. 标定 `vicon_world -> map`：使用不少于三个分散的静态底盘姿态做 SE(2) 拟合，并保存残差报告。
4. 将 Vicon pose 转到 `map` 后再进入状态向量；速度优先使用滤波后的时间差分，并记录滤波延迟。
5. 对丢帧、遮挡、时间戳回退和大跳变设置同样的锁存停止策略。

### 5.4 ESDF 与定位来源的兼容规则

- 每个 ESDF 输出必须记录采集时的固定 frame 和定位 source。
- 使用另一种定位 source 回放该 ESDF 时，必须存在经过验证的 `map` 对齐变换；否则重新采图。
- 不允许仅把 Vicon pose 的 frame_id 改成 `map` 来“完成对齐”。

## 6. 分阶段实施计划

### 阶段 0：冻结实机接口和版本

工作位置：机器人和工作站，全部只读检查，不发送运动命令。

任务：

1. 记录三个仓库/子模块 commit、ROS distro、`rmw_zenoh_cpp` 版本、Stretch firmware/driver 版本和所用 pixi 环境。
2. 在按 `REAL_DEPLOY.md` 启动后保存以下快照：
   - `ros2 node list`
   - `ros2 topic list -t` 与关键 topic 的 `ros2 topic info -v`
   - `ros2 service list -t`
   - `ros2 action list -t`
   - TF frame 图、各 transform authority 和频率
   - `/stretch/joint_states` 的完整 `name` 列表
3. 在人工缓慢移动相机时，同步捕获 SAI depth 关键帧、CameraInfo 和
   `map -> camera`；冻结 `16UC1` scale、关键帧 cadence，并据 ESDF 覆盖率决定
   是否需要单独的 raw-depth adapter。
4. 确认 SLAM TF 链；确认 Vicon base rigid body 接口。
5. 在底盘静止时确认 `/odom` 的 twist frame 语义；通过小幅人工遥控确认正方向和 yaw 符号。
6. 确认机械臂控制入口：
   - `navigation` 模式能否与机械臂 action/velocity 控制同时工作；
   - action 接受的 joint names；
   - `joint_arm` 是聚合关节还是四段关节；
   - 最大安全发送率、goal 抢占、cancel 和 hold 行为。
7. 测量 Zenoh 下关键 topic 的频率、丢帧、端到端延迟和工作站/机器人时钟偏差。

交付物：一份带时间戳的 ROS graph/TF/interface 清单。

通过条件：状态、定位、相机、底盘命令和机械臂命令五类契约都已确定。机械臂 streaming
协议及其与 `navigation` 的低幅并发已于 2026-08-06 确认；在阶段 3 watchdog、仲裁、
跟踪误差门和故障注入验收完成前，仍不进入 WB-MPC enabled 实机下发。

### 阶段 1：统一模型与实机状态

2026-08-06 状态：**阶段 1 已完成。** 模型审计决策、关节状态链路和 base 状态适配器已经完成。关节部分
在静止、wrist yaw、lift 和 extension 低速运动中通过；base 部分在显式 odom-local
诊断模式下通过静止、约 5 cm 直线往返和约 4 度 yaw 往返的 frame、方向和速度符号
验证。Spectacular AI 的 `map -> base_link` 被实测确认为 sparse keyframe TF，正式
状态链已改为关键帧全局锚点加 30 Hz `/odom` 传播，并通过 30 秒静止和约 2 cm 低速
往返初测；启动必须等待首个关键帧，关键帧修正跳变和无锚运动距离均 fail closed。
base 与 8 个关节已按控制器顺序组成完整 11 位置/11 速度状态；修正一次 33 ms
错周期配对后，594 个实机完整状态的 base/joint 时间戳差均为零且保持 30 Hz。
模型 FK 对 live TF 已完成 151 个同时间戳样本的多姿态验证；真实聚合伸展约束下，
可达近最坏姿态实测 33.74 mm / 4.60 度，与受约束离线预测 34.04 mm 相符。
原始 39.07 mm 离线样本独立改变四段伸展量，实机不可达。项目决定仍保留 nominal
控制模型并记录该已知偏差，不在状态桥中加入 offset。trajectory-mode 测试出现约
300 ms 空窗，并在一次返回 navigation 后出现反馈流停止，作为后续 receive-time
watchdog 和命令恢复测试的实测依据。重启 driver 后最终验收为 navigation、homed、
runstop 未触发，JointState 121/121 有效且约 30 Hz。

任务：

1. 对比 `mm_assets/stretch/stretch_ctrl.urdf` 与 bringup 的校准 URDF：关节原点、轴、limit、tool frame、camera frame 和 collision sphere parent link。
2. 建立按名称配置的状态映射，而非硬编码消息下标。
3. 定义模型状态构造：
   - base 三维来自统一定位适配器；
   - lift/wrist 来自 `JointState`；
   - 若实机只有聚合 `joint_arm`，定义聚合值到四段模型值的唯一分配规则，并用 FK 与 TF 实测验证；
   - 缺少速度时使用带时间戳的滤波差分，不能填零后当作真实反馈。
4. 写状态一致性检查：关节 limit、时间戳单调性、数据龄期、FK 的 `link_grasp_center` 与实机 TF 的误差。
5. 先运行只读 state bridge，记录数据，不创建任何 command publisher。

通过条件：静止和人工低速运动数据中，模型 FK、底盘位姿、关节方向和速度符号全部通过误差门限。

### 阶段 2：实机 ESDF 采集与导出

推荐先实现“rosbag 录制后离线生成”，再增加实时采集。这样可重复调试深度单位、TF、地面过滤和 nvblox 参数，而不必反复移动机器人。

任务：

1. 在工作站 GPU 上增加 ROS 2 ESDF 采集入口，复用 `OnlineNvbloxESDFMap` 与现有 NPZ 导出格式。
2. 同步 depth + CameraInfo，并按 depth 消息时间戳查询 `T_map_camera`。
3. 保留当前双地图语义：
   - 障碍 TSDF/ESDF 地图；
   - 未过滤深度构建的 observed-space 地图，用于区分已观察自由空间和未知空间。
4. 替换仿真专用 segmentation ground mask：实机先使用经标定的 map-Z/平面过滤，并评估低矮障碍误删；必要时加入平面估计。
5. 加入机器人自身过滤，避免相机看到的机械臂、夹爪或底盘被固化为环境障碍。
6. 采图时使用独占遥控命令源，扫描目标工作区；控制器保持 disabled。
7. 导出：
   - `esdf_grid.npz`
   - 原始 `map.nvblox` 和 `observed_space.nvblox`
   - `metadata.json`
   - 对应 rosbag 路径/哈希和质量报告
8. metadata 至少记录：frame、定位 source、时间范围、相机内参与 depth scale、TF frame、voxel/grid resolution、bounds、ground/self filter 参数、URDF/配置 commit 和 map 对齐变换。
9. 复用现有 planner-valid 检查，额外检查机器人初始位姿、全部任务目标、连通性、未知体素比例和关键 collision sphere 的 clearance。

通过条件：同一 rosbag 重建结果可重复；RViz/PyBullet 可视化与真实场景对齐；起点和测试目标位于同一已观察自由空间连通分量。

### 阶段 3：实机命令与安全适配器

2026-08-06 前置验证：`eoa_wrist_dw3_tool_sg3` 的 streaming-position 命令顺序已从
官方驱动源码核对，并用 10 Hz wrist yaw + 低速底盘往返实测确认可与 `navigation`
共存；期间 JointState 连续约 30 Hz，最大接收间隔 41.2 ms，退出后模式仍为
`navigation`、streaming=false。底盘返回误差约 0.67 mm，wrist yaw 返回误差约
0.0070 rad，说明后续适配器必须使用反馈闭环而不能假设命令已精确执行。
当前阶段 3 尚未完成；下一步是把已确认协议封装为默认 shadow 的命令适配器，并加入
外部 receive-time watchdog、唯一命令所有权和锁存停止。

任务：

1. 将 WB-MPC 输出拆分为：
   - 底盘：world/map velocity 转为 `base_link` 下的 Twist，并强制满足 Stretch 非完整约束；
   - 机械臂：按阶段 0 确认的驱动接口转换，不假设四段伸缩关节能独立驱动。
2. 若驱动仅接受 `FollowJointTrajectory`，评估“短时域位置轨迹 + 可抢占 action”是否满足当前 20 Hz 命令发布需求；若不能，先实现受支持的低层速度接口或降低控制架构频率，不能把高频 action goal 当速度 topic 使用。
3. 为所有输出做双层限幅：本仓库模型 limit 与实际 Stretch driver limit 取更保守值。
4. 实现唯一 command owner/仲裁：WBMPC 运行时禁用 teleop、Nav2/旧 stretch_mpc 等其他 `/stretch/cmd_vel` 发布者；退出后再显式交还控制权。
5. 实现 watchdog 和锁存停止。以下任一事件触发零底盘命令、机械臂 cancel/hold，并要求人工重新使能：
   - JointState、TF、Vicon/SLAM 或相机/控制状态超时；
   - 定位跳变或时间戳回退；
   - ESDF 起点/当前 collision sphere 无效；
   - solver failure/fallback 连续超限；
   - 控制周期 deadline 连续 miss；
   - Zenoh 连接中断；
   - Ctrl-C、节点异常或外部 E-stop。
6. 将 `enabled`、`shadow`、`base_only`、`arm_only`、`whole_body` 设为明确模式；节点启动默认 `shadow`，绝不启动即运动。

通过条件：所有故障注入都能在规定时间内停止；任意时刻只有一个底盘 command publisher；机械臂 cancel/hold 行为经过实机验证。

### 阶段 4：实机 planner/controller runner

任务：

1. 新建实机 runner，复用 `MPC`、`TaskManager`、`ESDFMap` 和 OMPL planner；不要复用 `BulletSimulation` 主循环。
2. 将仿真和实机共有逻辑抽成纯 Python 控制步，I/O 分别由 Bullet adapter 和 ROS adapter 提供，避免两套规划/控制行为漂移。
3. 新建独立实机配置，继承 `stretch_esdf_offline_ompl_wbmpc.yaml`，只覆盖：
   - ROS topic/frame/QoS；
   - `localization.source`；
   - ESDF 绝对路径及 metadata 校验；
   - 实机 rate、速度/加速度上限、watchdog 和启动模式；
   - 真实任务目标。
4. 启动时执行 preflight：homed、state ready、TF ready、定位健康、ESDF metadata 匹配、当前机器人全身状态有效、solver 与配置哈希匹配。
5. 运行时发布计划、MPC horizon、collision spheres、ESDF clearance、定位状态和 safety state，frame 统一为 `map`。
6. 日志同时保存 ROS 时间和单调时钟，记录原始/限幅后命令、数据龄期、solver 状态、最小 ESDF 距离和停止原因。

通过条件：shadow 模式持续运行完整任务时长，无状态错位、TF 查询失败、周期超时或非法命令。

### 阶段 5：逐级实机验证

按以下顺序执行，每一级失败都回到上一层；禁止跳过 shadow 直接 whole-body：

1. **离线回放**：用真实 rosbag 驱动状态和相机适配器，不连接命令 topic。
2. **实机 shadow**：在线读取实机状态并计算 OMPL/WB-MPC，只记录命令。
3. **底盘离地/轮架测试**（若硬件条件允许）：验证坐标方向、停止和限幅。
4. **base-only**：空旷区、低速、短距离；分别用 SLAM 和 Vicon。
5. **arm-only**：底盘锁止，从安全 home 附近做 lift/arm/wrist 小幅动作。
6. **顺序任务**：先 base，停稳后 EE；验证离线 ESDF 与真实 clearance。
7. **whole-body**：低速协调运动，先无障碍，再加入单一已知障碍。
8. **完整任务**：执行实际 OMPL base/EE 任务序列，分别形成 SLAM 和 Vicon 验收日志。
9. **故障测试**：断开定位、延迟 JointState、停止 Zenoh router、制造 solver failure、触发 E-stop，逐项验证锁存停止。

建议的初始验收指标（阶段 0 后按实测能力冻结）：

- JointState/TF 稳定频率不低于控制所需频率，数据龄期不超过两个控制周期。
- WB-MPC 求解时间的 P99 小于控制周期，且不出现连续 fallback。
- shadow 输出始终在实机限幅内，非完整底盘横向速度命令为零。
- 静态姿态下 FK tool pose 与实机 TF/Vicon tool pose 的误差处于标定门限内。
- SLAM/Vicon 各自完成相同短路径；跨定位源使用同一 ESDF 时通过 map 对齐检查。
- 实测最小障碍间距不小于经标定后的安全阈值，并保留额外硬件误差裕量。

## 7. 建议的后续文件边界

实施时预计只在本仓库新增/调整以下职责，不把硬件驱动复制进来：

- `mm_run`：实机 runner、ROS state/command adapter、ESDF capture node、launch。
- `mm_run/config`：实机公共配置，以及 SLAM/Vicon 两个轻量覆盖配置。
- `mm_control`：仅在需要时提取 ESDF 集成/导出复用接口，不引入 ROS 依赖。
- `mm_plan`：保持定位无关；只补充资源/状态契约测试。
- `test`：关节映射、frame 变换、world/body velocity、watchdog、ESDF metadata 和 rosbag 回放测试。
- `docs/results`：graph 快照、标定报告、每级验收记录；大型 rosbag/NPZ 不提交 Git。

建议的配置层次：

```text
stretch_esdf_offline_ompl_wbmpc.yaml       # 当前算法基线
└── stretch_esdf_offline_ompl_wbmpc_real.yaml
    ├── localization_slam.yaml             # 只覆盖定位 source/frame
    └── localization_vicon.yaml            # 只覆盖定位 source/frame/外参
```

## 8. 明确的决策门

以下问题由阶段 0 的实机证据决定，未确认前不进入相应编码或运动测试：

1. 是否冻结已部署的 `ros2_orbbec_slam`/`sai_orbbec` 作为深度数据源；
   其实际 depth/CameraInfo topic、encoding、QoS、optical frame 和时间戳是什么？
2. SLAM 离线运行使用哪一个保存的 pose graph/map，谁发布 `odom -> base_link`？
3. Vicon 底盘 rigid body 的名称、frame 和 marker-to-base 外参是什么？
4. Stretch 在当前 driver 版本中能否同时接收底盘 Twist 和机械臂控制？需要 `navigation`、`trajectory` 或其他模式？
5. 实机 arm feedback/command 是聚合 `joint_arm` 还是四段 joint；四段模型应如何一致映射？
6. 工作站到机器人 Zenoh 的实测延迟是否允许 7 Hz MPC + 20 Hz 命令；控制命令应在哪台机器上运行才能满足 watchdog？
7. ESDF 采集与执行是否使用同一定位 source；若不同，`vicon_world -> map` 的标定由谁维护和版本化？

## 9. 推荐实施顺序与里程碑

| 里程碑 | 内容                                      |         可运动 |
| ------ | ----------------------------------------- | -------------: |
| M0     | graph/TF/driver/Zenoh 契约冻结            |             否 |
| M1     | 状态适配 + SLAM/Vicon 统一输出 + rosbag   |             否 |
| M2     | 实机 ESDF 采集、导出和质量门              | 仅人工遥控采图 |
| M3     | 命令适配、仲裁、watchdog、shadow runner   |             否 |
| M4     | base-only 与 arm-only 低速验收            |   是，分离运动 |
| M5     | 顺序 OMPL + WB-MPC                        |             是 |
| M6     | whole-body + ESDF 避障，SLAM/Vicon 双验收 |             是 |

首个可执行工作包应为 **M0 + M1 的只读状态链路和 rosbag 记录**。它能最早暴露 joint、TF、相机和 Zenoh 契约问题，同时不会向机器人发送任何运动命令。
