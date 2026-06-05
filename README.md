# Mobile Manipulation
A ROS-based framework for mobile manipulation research, featuring MPC-based control, robot simulation, and planning utilities.

> [!IMPORTANT] 
> The instructions in this readme are not yet updated for ROS2 and pixi. Use the following commands.
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

## Installation
### Prerequisites
Ensure you have ROS Noetic installed on your system. Follow the [ROS Noetic installation guide](http://wiki.ros.org/noetic/Installation/Ubuntu) if it's not already set up.

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
ros2 launch mm_run run_pybullet_sim.launch config:=$(ros2 pkg prefix mm_run)/share/mm_run/config/simple_experiment.yaml
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
Use the merged ESDF validation config below. It keeps the challenge target, loads
the warehouse scene visually, disables PyBullet collision for the scene, and uses
the nvblox ESDF as the MPC obstacle field.

Build the edited packages into the clean install tree:

```bash
cd ~/repo/mobile_manipulation
pixi run colcon build \
  --packages-select mm_simulator mm_control mm_run \
  --build-base build_clean \
  --install-base install_clean
```

Source the clean install tree:

```bash
source install_clean/setup.bash
```

Compile the ESDF MPC acados solver:

```bash
python3 mm_control/scripts/generate_acados_code.py \
  --config $(ros2 pkg prefix mm_run)/share/mm_run/config/validate_esdf_mpc_challenge.yaml
```

Run the PyBullet validation:

```bash
python3 mm_run/scripts/experiment.py \
  --config $(ros2 pkg prefix mm_run)/share/mm_run/config/validate_esdf_mpc_challenge.yaml \
  --GUI
```

If you edit the ESDF MPC model structure or solver options such as
`nlp_solver_max_iter`, rebuild `mm_run` and run the compile command again before
running the experiment.
