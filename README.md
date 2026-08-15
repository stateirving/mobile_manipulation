# Mobile Manipulation

A ROS 2 research framework for mobile-manipulator planning, whole-body model
predictive control, ESDF collision avoidance, and PyBullet validation. The
repository currently supports Stretch and a mobile UR10 model.

> [!IMPORTANT]
> Use the pixi environment, build into the default `install/` tree, and source
> `install/setup.bash` before resolving package paths or launching ROS nodes.

> [!WARNING]
> Real-robot operation is safety critical. Follow the complete
> [Real Stretch Deployment Runbook](./REAL_DEPLOY.md). The offline/simulated
> ESDF does not detect obstacles in the physical workspace.

## Documentation

- [Real Stretch deployment, ESDF capture, ROS data flow, and WB-MPC execution](./REAL_DEPLOY.md)
- [Configuration reference](./mm_run/config/configuration.md)
- [Real Stretch command-adapter contract](./mm_run/config/real_command_adapter.md)

## Quick Start

Clone submodules, enter the environment, and build:

```bash
cd ~/repo/mobile_manipulation
git submodule update --init --recursive
pixi shell
colcon build
source install/setup.bash
```

In every new terminal:

```bash
cd ~/repo/mobile_manipulation
pixi shell
source install/setup.bash
```

Verify that ROS resolves packages from the default install tree:

```bash
ros2 pkg prefix mm_run
ros2 pkg prefix mm_control
```

Rebuild after changing Python code or YAML under `mm_run/config/`.

## Repository Layout

| Package          | Purpose                                                      |
| ---------------- | ------------------------------------------------------------ |
| `mm_assets`    | Robot, scene, URDF, xacro, and mesh assets                   |
| `mm_control`   | Acados-based MPC and WB-MPC controllers                      |
| `mm_plan`      | Task management and base/EE planners, including OMPL         |
| `mm_run`       | Runtime configurations, launch files, nodes, and experiments |
| `mm_simulator` | PyBullet simulation interface                                |
| `mm_utils`     | Parsing, mathematics, logging, and plotting utilities        |

## Common Workflows

### Simple Synchronous PyBullet Experiment

Generate the Acados controller after changing its model, costs, constraints, or
solver options:

```bash
python3 mm_control/scripts/generate_acados_code.py \
  --config "$(ros2 pkg prefix mm_run)/share/mm_run/config/simple_experiment.yaml"
```

Run the experiment:

```bash
python3 mm_run/scripts/experiment.py \
  --config "$(ros2 pkg prefix mm_run)/share/mm_run/config/simple_experiment.yaml" \
  --GUI
```

### Stretch Offline-ESDF OMPL + WB-MPC

This is the canonical static-ESDF configuration shared by simulation and the
current real-state deployment profiles.

```bash
python3 mm_control/scripts/generate_acados_code.py \
  --config "$(ros2 pkg prefix mm_run)/share/mm_run/config/stretch_esdf_offline_ompl_wbmpc.yaml"

python3 mm_run/scripts/experiment.py \
  --config "$(ros2 pkg prefix mm_run)/share/mm_run/config/stretch_esdf_offline_ompl_wbmpc.yaml" \
  --GUI
```

### Stretch Online-nvblox OMPL + WB-MPC

This profile inherits the offline setup, replaces the static ESDF with online
nvblox queries, and uses OMPL for base and Cartesian end-effector paths.

```bash
python3 mm_control/scripts/generate_acados_code.py \
  --config "$(ros2 pkg prefix mm_run)/share/mm_run/config/stretch_esdf_online_nvblox_ompl_wbmpc.yaml"

python3 mm_run/scripts/experiment_online_nvblox.py \
  --config "$(ros2 pkg prefix mm_run)/share/mm_run/config/stretch_esdf_online_nvblox_ompl_wbmpc.yaml" \
  --GUI
```

### Mobile UR10 Offline-ESDF OMPL + WB-MPC

```bash
python3 mm_control/scripts/generate_acados_code.py \
  --config "$(ros2 pkg prefix mm_run)/share/mm_run/config/ur10_esdf_offline_ompl_wbmpc.yaml"

python3 mm_run/scripts/experiment.py \
  --config "$(ros2 pkg prefix mm_run)/share/mm_run/config/ur10_esdf_offline_ompl_wbmpc.yaml" \
  --GUI
```

### Simulated Stretch ESDF Capture

`nvblox-torch` requires a CUDA GPU for capture/export. Viewing an exported NPZ
does not require CUDA.

```bash
pixi run python mm_run/scripts/teleop_export_esdf.py \
  --config mm_run/config/stretch_esdf_teleop_export.yaml
```

PyBullet keyboard controls:

- `I` / `K`: forward / backward
- `J` / `L`: turn left / right
- `Space`: stop
- `X`: stop mapping and export the final ESDF
- `Q`: close only the secondary reconstruction viewer

The default output is:

```text
mm_run/results/nvblox_esdf/stretch_teleop/<TIMESTAMP>/
├── esdf_grid.npz
├── map.nvblox
├── observed_space.nvblox
└── metadata.json
```

OMPL treats unknown space as invalid. Before exporting, observe the intended
start region from several viewpoints so the base collision samples lie in
known free space.

### Inspect an Exported ESDF

```bash
pixi run python mm_run/scripts/visualize_esdf_npz.py \
  /ABSOLUTE/PATH/TO/esdf_grid.npz \
  --color-mode height
```

Use `--color-mode distance` to inspect negative, zero, and positive signed
distance regions.

### Real Stretch

The real workflow includes robot/Zenoh/SLAM bringup, PS4 teleoperation, rosbag
capture, offline ESDF export, a read-only preflight, shadow WB-MPC validation,
and an explicitly enabled hardware test. All commands and safety gates are in
[REAL_DEPLOY.md](./REAL_DEPLOY.md).

The real-state shadow pipeline can be launched without creating hardware
command publishers:

```bash
ros2 launch mm_run stretch_wbmpc_shadow.launch.py \
  adapter_log:=/tmp/stretch_adapter_wbmpc_shadow.jsonl \
  wbmpc_log:=/tmp/stretch_wbmpc_shadow.jsonl
```

The adapter publishes validated state on `/wbmpc/state`; the runner publishes
an 11-D velocity envelope on `/wbmpc/velocity_command`. This launch never
passes `--execute`.

## Configuration and Generated Controllers

Configurations live under `mm_run/config/` and can include other YAML files.
The main groups are:

- `robot/`: kinematics, dimensions, bounds, and collision geometry
- `controller/`: horizons, rates, costs, and solver settings
- `scene/`: environment assets
- `sim/`: simulator settings
- top-level experiment profiles: planner tasks and composed overrides

Regenerate Acados code after changing model dimensions, dynamics, costs,
constraints, or solver structure. Changing only an ESDF NPZ path does not
require regeneration.

Runtime-generated maps should use an absolute `map_path`. A package/path
mapping resolves under `install/mm_run/share/mm_run`, while capture output is
normally written into the source tree.

## Results

Experiments write timestamped results under `mm_run/results/`, commonly with
`sim/` and `control/` subdirectories. The real deployment runbook documents
its JSONL diagnostics and comparison plotting commands.
