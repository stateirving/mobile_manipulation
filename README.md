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
The verified real-robot startup, teleoperation, rosbag capture, and ESDF export
procedure is documented in [REAL_DEPLOY.md](./REAL_DEPLOY.md).

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

### OMPL Base/EE Planner Offline ESDF WB-MPC
This is the canonical offline ESDF + OMPL base/EE + whole-body MPC config. It
also owns the shared offline ESDF, scene, solver, robot, and simulation settings.

Compile the offline OMPL WB-MPC acados solver:

```bash
python3 mm_control/scripts/generate_acados_code.py --config $(ros2 pkg prefix mm_run)/share/mm_run/config/stretch_esdf_offline_ompl_wbmpc.yaml
```

Run the offline OMPL WB-MPC PyBullet validation:

```bash
python3 mm_run/scripts/experiment.py --config $(ros2 pkg prefix mm_run)/share/mm_run/config/stretch_esdf_offline_ompl_wbmpc.yaml --GUI
```

### OMPL Base/EE Planner Online nvblox WB-MPC
This config inherits the offline OMPL WB-MPC setup, replaces the static ESDF
with an online nvblox map, and uses OMPL for the base and Cartesian EE paths.

Compile the OMPL WB-MPC acados solver:

```bash
python3 mm_control/scripts/generate_acados_code.py --config $(ros2 pkg prefix mm_run)/share/mm_run/config/stretch_esdf_online_nvblox_ompl_wbmpc.yaml
```

Run the OMPL WB-MPC online nvblox validation:

```bash
python3 mm_run/scripts/experiment_online_nvblox.py --config $(ros2 pkg prefix mm_run)/share/mm_run/config/stretch_esdf_online_nvblox_ompl_wbmpc.yaml --GUI
```

If you edit the ESDF MPC model structure or solver options such as
`nlp_solver_max_iter`, rebuild `mm_run` and run the matching compile command
again before running the experiment.

### UR10 OMPL Base/EE Planner Offline ESDF WB-MPC
This config uses the UR10 mounted on a holonomic planar base, OMPL for the base
and Cartesian EE paths, and whole-body MPC with offline ESDF collision avoidance.

Compile the UR10 offline OMPL WB-MPC acados solver:

```bash
python3 mm_control/scripts/generate_acados_code.py --config $(ros2 pkg prefix mm_run)/share/mm_run/config/ur10_esdf_offline_ompl_wbmpc.yaml
```

Run the UR10 offline OMPL WB-MPC PyBullet validation:

```bash
python3 mm_run/scripts/experiment.py --config $(ros2 pkg prefix mm_run)/share/mm_run/config/ur10_esdf_offline_ompl_wbmpc.yaml --GUI
```

### Stretch Keyboard Teleoperation ESDF Capture and Replay

The commands below use the current source-tree configs and assume they are run
from the repository root. `nvblox-torch` requires a CUDA GPU for capture and
export; viewing an exported NPZ does not require CUDA.

Capture an ESDF by teleoperating the simulated Stretch with its onboard depth
camera:

```bash
pixi run python mm_run/scripts/teleop_export_esdf.py \
  --config mm_run/config/stretch_esdf_teleop_export.yaml
```

Keyboard controls in the PyBullet window:

- `I` / `K`: move forward / backward.
- `J` / `L`: turn left / right.
- `Space`: stop.
- `X`: stop mapping, update the final ESDF, and export it.

A second, empty PyBullet window displays a live low-resolution reconstruction.
It does not load or overlay the ground-truth URDF scene:

- Height-colored points: valid samples close to the reconstructed ESDF zero
  surface.

The live reconstruction queries a 10 cm grid every 30 simulation steps. The
viewer runs in a separate process, and only downsampled NumPy point clouds are
sent to it, so it never participates in camera rendering. Close only the viewer
with `Q`; keep focus on the main simulation window for teleoperation. The final
NPZ is still exported at 2 cm resolution.

Ground handling uses two nvblox maps. In PyBullet, segmentation IDs remove only
the `plane.urdf` endpoints from the obstacle TSDF, so low obstacle geometry is
preserved; the world-Z threshold is a fallback when segmentation is unavailable.
A secondary occupancy map integrates the unfiltered depth and retains
negative-log-odds free-space evidence along camera rays. During export,
non-ground obstacle distances are propagated only into those observed-free
voxels. This keeps the ground out of the navigation ESDF without turning the
space above the observed floor into unknown. Unobserved voxels remain invalid.

After export, the script evaluates the map with the offline base planner's
actual `query_z`, `base_radius`, and `d_safe` settings. It reports the known and
planner-valid lattice ratios, labels connected free-space components, and
prints an explicit warning when a start/goal is invalid or a valid goal is not
reachable from the start through observed free space.

The default output is a timestamped directory:

```text
mm_run/results/nvblox_esdf/stretch_teleop/<TIMESTAMP>/
├── esdf_grid.npz
├── map.nvblox
├── observed_space.nvblox
└── metadata.json
```

Before pressing `X`, observe the intended navigation start from several
viewpoints. OMPL treats unknown space as invalid and will reject a start pose
whose base collision query points were not observed.

Convert a real Spectacular-AI ROS 2 bag offline with the same two-map ground
semantics:

```bash
pixi run python mm_run/scripts/export_real_rosbag_esdf.py \
  /ABSOLUTE/PATH/TO/ROSBAG_DIRECTORY \
  -o mm_run/results/nvblox_esdf/real_bag/MAP_NAME \
  --bounds -4.2 -4.2 -0.2 4.2 4.2 2.2 \
  --voxel-size 0.05 --grid-resolution 0.05 --ground-min-z 0.08
```

The primary TSDF receives ground-filtered depth, while `observed_space.nvblox`
receives unfiltered depth and supplies observed-free ray evidence. The final
NPZ keeps genuinely unobserved voxels invalid and records fusion and base
planner quality statistics in `metadata.json`. The converter defaults to the
canonical offline base checks (`query_z=[0.15, 0.35]`, required clearance
0.4 m, XY bounds ±4 m); matching CLI options can override them for another
planner profile.

Reconstruct and inspect the approximate ESDF zero surface in PyBullet:

```bash
pixi run python mm_run/scripts/visualize_esdf_npz.py \
  /ABSOLUTE/PATH/TO/esdf_grid.npz \
  --color-mode height
```

Use signed-distance coloring when checking the two sides of the surface:

```bash
pixi run python mm_run/scripts/visualize_esdf_npz.py \
  /ABSOLUTE/PATH/TO/esdf_grid.npz \
  --color-mode distance
```

In `height` mode, low points are blue and high points are red. In `distance`
mode, negative/zero/positive distances are red/white/blue. Press `Q` in the
PyBullet window to close the viewer.

To replay the existing offline OMPL + WB-MPC pipeline with the captured map,
set `controller.esdf_collision.map_path` in
`mm_run/config/stretch_esdf_offline_ompl_wbmpc.yaml` to an absolute path:

```yaml
controller:
  esdf_collision:
    source: "offline"
    map_path: "/ABSOLUTE/PATH/TO/esdf_grid.npz"
```

Do not use a `{package: "mm_run", path: ...}` mapping for a runtime-generated
file: package paths resolve under `install/mm_run/share/mm_run`, while capture
outputs are written under the source tree by default.

Run the offline replay using the edited source config:

```bash
pixi run python mm_run/scripts/experiment.py \
  --config mm_run/config/stretch_esdf_offline_ompl_wbmpc.yaml \
  --GUI
```

Changing only the NPZ path does not require regenerating the acados solver.
