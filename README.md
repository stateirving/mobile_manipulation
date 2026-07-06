# Mobile Manipulation
A ROS-based framework for mobile manipulation research, featuring MPC-based control, robot simulation, and planning utilities.

> [!IMPORTANT]
> Current ROS2/pixi workflow. Build into the default `install/` tree and source
> `install/setup.bash` for all runtime commands.
>
> Make sure submodules are cloned
> `git submodule update --init --recursive`
>
> For installation, simply install the pixi environment:
> `pixi shell`
>
> The following commands were tested:
>
> - Build packages
>   - `colcon build && source install/setup.bash`
>
> - Compile MPC Controller
>   - `python3 mm_control/scripts/generate_acados_code.py --config $(ros2 pkg prefix mm_run)/share/mm_run/config/simple_experiment.yaml`
>
> - Run Controller with PyBullet Simulation (Synchronous)
>   - `python3 mm_run/scripts/experiment.py --config $(ros2 pkg prefix mm_run)/share/mm_run/config/simple_experiment.yaml --GUI`
>
> The two commands above no longer depend on `mobile_manipulation_central` or `ur_description`. Legacy ROS launch files and nodes still do.
>
> - Run Controller and Simulation Asynchronously (ROS Nodes)
>   - `ros2 launch mm_run run_pybullet_sim.launch.py config:=$(ros2 pkg prefix mm_run)/share/mm_run/config/simple_experiment.yaml`

## Package Overview
- **mm_assets**: Robot and scene URDF/mesh files
- **mm_control**: MPC controller implementation using Acados
- **mm_plan**: Planning base classes and simple planners
- **mm_run**: Launch files, configurations, and ROS nodes
- **mm_simulator**: PyBullet simulation interface
- **mm_utils**: Utility functions for math, parsing, logging, etc.

Configuration parameters are documented in [configuration.md](./mm_run/config/configuration.md).

## Current Setup
Use the pixi environment and build the ROS2 packages into the default colcon
install tree:

```bash
cd ~/repo/mobile_manipulation
git submodule update --init --recursive
pixi shell
colcon build && source install/setup.bash
```

In a new terminal, run `pixi shell` and `source install/setup.bash` again before
using `ros2 pkg prefix`, launching nodes, or running experiments.

## Legacy Installation Notes
The notes below are for older ROS Noetic/catkin workflows and some legacy launch
files. They are not required for the synchronous pixi commands above.

### Prerequisites
For the legacy workflow, ensure you have ROS Noetic installed on your system.
Follow the [ROS Noetic installation guide](http://wiki.ros.org/noetic/Installation/Ubuntu) if it's not already set up.

### Installation of `mobile_manipulation_central`
`mobile_manipulation_central` is no longer required for the synchronous commands above. It is still required for some legacy ROS launch files and nodes in this repository.

```bash
cd ~/catkin_ws/src
git clone https://github.com/utiasDSL/mobile_manipulation_central
git checkout mm_dev
cd ~/catkin_ws
catkin build mobile_manipulation_central
source devel/setup.bash
```

### Pinocchio
```bash
sudo apt install libeigen3-dev ros-noetic-eigenpy ros-noetic-hpp-fcl ros-noetic-pinocchio
```

Make sure to source your ROS environment:
```bash
source /opt/ros/noetic/setup.bash
```

### Acados
Follow the instructions on the [Acados website](https://docs.acados.org/installation/). Don't forget to install the Python interface.

### Installing this repo
```bash
cd ~/catkin_ws/src
git clone https://github.com/utiasDSL/mobile_manipulation
cd ~/catkin_ws
catkin build mobile_manipulation
source devel/setup.bash
python3 -m pip install -r requirements.txt
```

## Usage
### Compile MPC Controller
```bash
python3 mm_control/scripts/generate_acados_code.py --config $(ros2 pkg prefix mm_run)/share/mm_run/config/simple_experiment.yaml
```

### Run Controller with PyBullet Simulation (Synchronous)
```bash
python3 mm_run/scripts/experiment.py --config $(ros2 pkg prefix mm_run)/share/mm_run/config/simple_experiment.yaml --GUI
```

### Run Controller and Simulation Asynchronously (ROS Nodes)
```bash
ros2 launch mm_run run_pybullet_sim.launch.py config:=$(ros2 pkg prefix mm_run)/share/mm_run/config/simple_experiment.yaml
```

### Visualize Results
Results are saved to `mm_run/results/[EXPERIMENT_NAME]/[TIMESTAMP]/` with `sim/` and `control/` subfolders.

```bash
roscd mm_utils/scripts
python3 plot_logs.py --folder ../../mm_run/results/[EXPERIMENT_NAME]/[TIMESTAMP]/ --tracking
```

### Isaac Sim (Optional)
If using Isaac Sim, ensure [mm_sim_isaac](https://github.com/TracyDuX/mm_sim_isaac) is installed:
```bash
ros2 launch mm_run isaac_sim.launch config:=$(ros2 pkg prefix mm_run)/share/mm_run/config/3d_collision.yaml isaac-venv:=$ISAACSIM_PYTHON
```

## Configuration
Configuration files are located in `mm_run/config/`. Key configuration options include:

- **Robot**: Robot model parameters (`config/robot/`)
- **Scene**: Environment and obstacle definitions (`config/scene/`)
- **Controller**: MPC parameters (`config/controller/`)
- **Simulation**: Simulation settings (`config/sim/`)

## ESDF MPC Validation
Use the default colcon install tree. Runtime config paths are resolved through
`$(ros2 pkg prefix mm_run)`, so rebuild after editing Python files or YAML under
`mm_run/config/`. The validation commands below assume you are already inside
`pixi shell`.

Build only after code/config changes:

```bash
cd ~/repo/mobile_manipulation
colcon build && source install/setup.bash
```

Verify the default install tree:

```bash
ros2 pkg prefix mm_run
ros2 pkg prefix mm_control
```

Compile the ESDF MPC acados solver:

```bash
python3 mm_control/scripts/generate_acados_code.py --config $(ros2 pkg prefix mm_run)/share/mm_run/config/stretch_esdf_offline.yaml
```

Run the PyBullet validation:

```bash
python3 mm_run/scripts/experiment.py --config $(ros2 pkg prefix mm_run)/share/mm_run/config/stretch_esdf_offline.yaml --GUI
```

### OMPL Base/EE Planner Offline ESDF WB-MPC
This config inherits `stretch_esdf_offline.yaml`, uses OMPL for the base path
and Cartesian EE path, and keeps the existing whole-body MPC controller.

Compile the offline OMPL WB-MPC acados solver:

```bash
python3 mm_control/scripts/generate_acados_code.py --config $(ros2 pkg prefix mm_run)/share/mm_run/config/stretch_esdf_offline_ompl_wbmpc.yaml
```

Run the offline OMPL WB-MPC PyBullet validation:

```bash
python3 mm_run/scripts/experiment.py --config $(ros2 pkg prefix mm_run)/share/mm_run/config/stretch_esdf_offline_ompl_wbmpc.yaml --GUI
```

### Online nvblox ESDF MPC Validation
Compile the online ESDF MPC acados solver:

```bash
python3 mm_control/scripts/generate_acados_code.py --config $(ros2 pkg prefix mm_run)/share/mm_run/config/stretch_esdf_online_nvblox.yaml
```

Run the online nvblox PyBullet validation:

```bash
python3 mm_run/scripts/experiment_online_nvblox.py --config $(ros2 pkg prefix mm_run)/share/mm_run/config/stretch_esdf_online_nvblox.yaml --GUI
```

### OMPL Base Planner Online nvblox WB-MPC
This config inherits `stretch_esdf_online_nvblox.yaml`, uses OMPL for the base
path, and keeps the existing whole-body MPC controller.

Compile the OMPL WB-MPC acados solver:

```bash
python3 mm_control/scripts/generate_acados_code.py --config $(ros2 pkg prefix mm_run)/share/mm_run/config/stretch_esdf_online_nvblox_ompl_wbmpc.yaml
```

Run the OMPL WB-MPC online nvblox validation:

```bash
python3 mm_run/scripts/experiment_online_nvblox.py --config $(ros2 pkg prefix mm_run)/share/mm_run/config/stretch_esdf_online_nvblox_ompl_wbmpc.yaml --GUI
```

### CasADi Local Grid Online nvblox MPC
This experimental config inherits `stretch_esdf_online_nvblox.yaml`
and changes only the ESDF collision backend to a local CasADi interpolant.

Full command sequence:

```bash
cd ~/repo/mobile_manipulation
pixi shell
colcon build && source install/setup.bash
python3 mm_control/scripts/generate_acados_code.py --config $(ros2 pkg prefix mm_run)/share/mm_run/config/stretch_esdf_online_nvblox_casadi_local_grid.yaml
python3 mm_run/scripts/experiment_online_nvblox.py --config $(ros2 pkg prefix mm_run)/share/mm_run/config/stretch_esdf_online_nvblox_casadi_local_grid.yaml --GUI
```

If the package is already built and `install/setup.bash` is already sourced,
only rerun the last command to repeat the same experiment:

```bash
python3 mm_run/scripts/experiment_online_nvblox.py --config $(ros2 pkg prefix mm_run)/share/mm_run/config/stretch_esdf_online_nvblox_casadi_local_grid.yaml --GUI
```

Quick checks:

```bash
python3 -c "from mm_control.local_esdf_grid import LocalESDFGridSampler; print(LocalESDFGridSampler({}).shape)"
```

If you edit the ESDF MPC model structure or solver options such as
`nlp_solver_max_iter`, rebuild `mm_run` and run the matching compile command
again before running the experiment. For the CasADi local grid backend, changing
`voxel_size`, `size_xy`, `z_range`, or `acados.name` also requires regenerating
the solver.
