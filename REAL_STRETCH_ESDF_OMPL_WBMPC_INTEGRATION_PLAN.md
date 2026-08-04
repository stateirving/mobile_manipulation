# Stretch 实机接入：ESDF 采集 + Offline OMPL + WB-MPC 计划

更新日期：2026-08-04

## 1. 目标与边界

目标是在不复制或替换 `bringup_active_mapmaintenance` 的前提下，让本仓库运行在工作站上，通过现有 ROS 2 + Zenoh 网络接入 Stretch 实机，并复用本仓库已经在 PyBullet 中验证的以下能力：

1. 从真实深度相机数据采集并导出离线 `esdf_grid.npz`。
2. 使用该离线 ESDF 进行 OMPL 底盘路径规划和末端路径规划。
3. 使用全身 MPC（WB-MPC）跟踪规划结果，并将输出安全地拆分为 Stretch 底盘和机械臂命令。
4. 机器人全局位姿支持 SLAM 和 Vicon 两种来源，但规划、ESDF 和控制层只使用统一的 `map` 坐标系接口。

本计划只描述后续实施步骤。本次不修改已有 Python、YAML、launch、URDF 或 bringup 仓库。

## 2. 当前基线与已确认事实

### 2.1 实机 bringup

依据本仓库的 `real deploy.txt` 和本机只读副本 `/home/miao/repo/bringup_active_mapmaintenance`：

- Stretch 本机通过 `online_bringup_active_mapmaintenance` 启动 ROS 2 Humble、`rmw_zenoh_cpp` 和 Stretch 驱动。
- 工作站和机器人各运行 Zenoh router，工作站上的本仓库作为同一 ROS graph 中的远程节点运行。
- Stretch 驱动当前以 `navigation` 模式启动，已确认的重映射为：
  - 状态：`/stretch/joint_states`，类型 `sensor_msgs/msg/JointState`。
  - 底盘命令：`/stretch/cmd_vel`，类型 `geometry_msgs/msg/Twist`。
- `broadcast_odom_tf` 当前为 `False`，所以必须在实机 graph 中确认究竟由哪个节点发布 `odom -> base_link`；若没有该变换，SLAM TF 链并不完整。
- `slam_toolbox` 当前运行同步建图节点，栅格地图 topic 从 `/map` 重映射到了 `/lidar_map`。这不会自动改变 TF 中的 `map` frame 名称。
- Spectacular AI/Orbbec SLAM 的 launch 当前被注释；虽然感知代码默认订阅
  `/spectacular_ai/camera_info`、`/spectacular_ai/color_image` 和
  `/spectacular_ai/depth_image`，但不能据此假设实机启动后一定存在这些 topic。
- `/home_the_robot` 已经用于实际部署流程；控制节点仍需通过 graph 快照确认其服务类型、可用条件和 homing 完成状态。

### 2.2 本仓库现状

- `mm_run/scripts/teleop_export_esdf.py` 只从 PyBullet 相机采集，不订阅 ROS 图像或 TF。
- 导出的 `esdf_grid.npz` 已有稳定读取接口：`mm_control.esdf_map.ESDFMap`。
- `stretch_esdf_offline_ompl_wbmpc.yaml` 已串起离线 ESDF、OMPL base/EE planner 和 WB-MPC。
- `mm_run/scripts/experiment.py` 将规划/控制循环直接绑定到了 `BulletSimulation`，不能直接用于实机。
- `mm_run/nodes/mpc_ros.py` 虽然是 ROS 2 节点，但依赖当前环境中不存在的
  `mobile_manipulation_central` 接口，并不是现有 Stretch bringup 的适配器。
- 当前控制模型状态为 11 维：
  `[base_x, base_y, base_yaw, lift, arm_l3, arm_l2, arm_l1, arm_l0,
  wrist_yaw, wrist_pitch, wrist_roll]`。
- 控制 URDF 与 bringup URDF 的核心关节名称相近，但轮子、头部、夹爪的可动/固定定义不同；不能只凭同名假设两者运动学和零位完全一致。

### 2.3 当前不能直接连实机的三个阻塞项

1. **状态契约不一致**：模型使用四段伸缩关节，实机驱动可能只反馈聚合的 `joint_arm`，需要以实际 `/stretch/joint_states` 为准建立双向映射。
2. **命令契约不一致**：WB-MPC 输出 11 维速度，而 bringup 已确认的直接控制入口只有底盘 `/stretch/cmd_vel`；机械臂的速度/轨迹入口及其与 `navigation` 模式能否并发尚未确认。
3. **坐标系尚未闭合**：ESDF、OMPL 和 MPC 必须使用同一个固定坐标系；SLAM 和 Vicon 的原点、时间戳、漂移/跳变语义不同，不能直接替换 topic 名。

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

| 用途 | 首选接口 | 状态 | 适配要求 |
|---|---|---:|---|
| 关节反馈 | `/stretch/joint_states` (`JointState`) | 已确认 | 按 `name` 映射，禁止依赖数组顺序；记录位置、速度是否完整 |
| 底盘速度反馈 | `/odom` (`Odometry`) | 待实机确认 | 转成 `map` 或 `base_link` 下定义明确的速度；禁止混用 world/body velocity |
| SLAM 位姿 | TF `map -> odom -> base_link` | 部分确认 | 用 tf2 lookup 组合变换，不直接扫描 `/tf` 等待一条 `map -> base_link` 消息 |
| Vicon 位姿 | Vicon rigid-body topic/TF | 待确认 | 应用 `vicon_world -> map` 和 marker -> `base_link` 外参 |
| 深度 | 参数化的 `Image` topic | 待确认 | 支持 `16UC1`/`32FC1`，显式配置米制 scale |
| 相机内参 | 与深度匹配的 `CameraInfo` | 待确认 | 校验分辨率、K、畸变处理和时间戳 |
| 相机位姿 | TF `map <- camera_optical_frame` | 待确认 | 按图像时间戳查询，不使用“最新 TF”代替历史 TF |
| 底盘命令 | `/stretch/cmd_vel` (`Twist`) | 已确认 | 将 MPC 的 map-frame base velocity 转成 body-frame 非完整约束命令 |
| 机械臂命令 | 候选 `/stretch_controller/follow_joint_trajectory` | 待确认 | 确认 driver mode、joint names、最大更新率、抢占/取消语义后再实现 |
| Homing | `/home_the_robot` (`Trigger`) | 使用中 | 控制节点只检查状态；是否自动调用由 launch 参数决定 |
| 急停/停止 | 零 Twist + 机械臂 cancel/hold + 硬件 E-stop | 待确认 | 必须为锁存故障，不能在下一帧自动恢复 |

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
2. 解决当前 `broadcast_odom_tf=False` 带来的 TF 链风险；全链只能有一个 authority，避免重复 TF。
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
2. 在按 `real deploy.txt` 启动后保存以下快照：
   - `ros2 node list`
   - `ros2 topic list -t` 与关键 topic 的 `ros2 topic info -v`
   - `ros2 service list -t`
   - `ros2 action list -t`
   - TF frame 图、各 transform authority 和频率
   - `/stretch/joint_states` 的完整 `name` 列表
3. 确认相机 depth、CameraInfo、光学 frame、encoding、QoS 和稳定帧率。
4. 确认 SLAM TF 链；确认 Vicon base rigid body 接口。
5. 在底盘静止时确认 `/odom` 的 twist frame 语义；通过小幅人工遥控确认正方向和 yaw 符号。
6. 确认机械臂控制入口：
   - `navigation` 模式能否与机械臂 action/velocity 控制同时工作；
   - action 接受的 joint names；
   - `joint_arm` 是聚合关节还是四段关节；
   - 最大安全发送率、goal 抢占、cancel 和 hold 行为。
7. 测量 Zenoh 下关键 topic 的频率、丢帧、端到端延迟和工作站/机器人时钟偏差。

交付物：一份带时间戳的 ROS graph/TF/interface 清单。

通过条件：状态、定位、相机、底盘命令和机械臂命令五类契约都已确定。尤其是机械臂并发控制未确认前，不进入 WB-MPC 实机下发阶段。

### 阶段 1：统一模型与实机状态

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

1. 深度数据最终来自 Spectacular AI、Orbbec 原生驱动还是另一相机节点？
2. SLAM 离线运行使用哪一个保存的 pose graph/map，谁发布 `odom -> base_link`？
3. Vicon 底盘 rigid body 的名称、frame 和 marker-to-base 外参是什么？
4. Stretch 在当前 driver 版本中能否同时接收底盘 Twist 和机械臂控制？需要 `navigation`、`trajectory` 或其他模式？
5. 实机 arm feedback/command 是聚合 `joint_arm` 还是四段 joint；四段模型应如何一致映射？
6. 工作站到机器人 Zenoh 的实测延迟是否允许 7 Hz MPC + 20 Hz 命令；控制命令应在哪台机器上运行才能满足 watchdog？
7. ESDF 采集与执行是否使用同一定位 source；若不同，`vicon_world -> map` 的标定由谁维护和版本化？

## 9. 推荐实施顺序与里程碑

| 里程碑 | 内容 | 可运动 |
|---|---|---:|
| M0 | graph/TF/driver/Zenoh 契约冻结 | 否 |
| M1 | 状态适配 + SLAM/Vicon 统一输出 + rosbag | 否 |
| M2 | 实机 ESDF 采集、导出和质量门 | 仅人工遥控采图 |
| M3 | 命令适配、仲裁、watchdog、shadow runner | 否 |
| M4 | base-only 与 arm-only 低速验收 | 是，分离运动 |
| M5 | 顺序 OMPL + WB-MPC | 是 |
| M6 | whole-body + ESDF 避障，SLAM/Vicon 双验收 | 是 |

首个可执行工作包应为 **M0 + M1 的只读状态链路和 rosbag 记录**。它能最早暴露 joint、TF、相机和 Zenoh 契约问题，同时不会向机器人发送任何运动命令。
