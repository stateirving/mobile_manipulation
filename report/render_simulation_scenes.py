"""Render Fig. 3 as an overview of the complete configured Stretch task."""

from __future__ import annotations

import copy
import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pybullet as pyb
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path as MplPath

from mm_simulator import simulation
from mm_utils import parsing

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "mm_run/config/stretch_esdf_offline_ompl_wbmpc.yaml"
OUTPUT = ROOT / "report/figures/simulation_scenes.png"

BASE_COLOR = [0.04, 0.35, 0.78, 1.0]
EE_COLOR = [0.91, 0.16, 0.08, 1.0]
START_COLOR = [0.12, 0.12, 0.12, 1.0]


def _sphere(position: np.ndarray, radius: float, color: list[float]) -> None:
    visual = pyb.createVisualShape(pyb.GEOM_SPHERE, radius=radius, rgbaColor=color)
    pyb.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=visual,
        basePosition=np.asarray(position, dtype=float).tolist(),
    )


def _disc(position: np.ndarray, radius: float, color: list[float]) -> None:
    visual = pyb.createVisualShape(
        pyb.GEOM_CYLINDER, radius=radius, length=0.035, rgbaColor=color
    )
    pyb.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=visual,
        basePosition=np.asarray(position, dtype=float).tolist(),
    )


def _project(
    points: dict[str, np.ndarray],
    view: list[float],
    projection: list[float],
    width: int,
    height: int,
) -> dict[str, tuple[float, float]]:
    """Project world points to Matplotlib image coordinates."""
    view_matrix = np.asarray(view).reshape(4, 4, order="F")
    projection_matrix = np.asarray(projection).reshape(4, 4, order="F")
    pixels = {}
    for name, point in points.items():
        homogeneous = np.r_[np.asarray(point, dtype=float), 1.0]
        clip = projection_matrix @ view_matrix @ homogeneous
        ndc = clip[:3] / clip[3]
        pixels[name] = (
            float((ndc[0] + 1.0) * width / 2.0),
            float((1.0 - ndc[1]) * height / 2.0),
        )
    return pixels


def _render(
    config: dict,
    q: np.ndarray,
    markers: dict[str, tuple[np.ndarray, str]],
    camera: dict,
) -> tuple[np.ndarray, dict[str, tuple[float, float]]]:
    sim_config = copy.deepcopy(config["simulation"])
    sim_config["gui"] = False
    sim_config["collision_sphere_markers"]["enabled"] = False
    world = simulation.BulletSimulation(
        config=sim_config,
        timestamp=datetime.datetime.now(),
    )
    world.robot.reset_joint_configuration(q)

    if camera.get("hide_robot", False):
        robot_uid = world.robot.uid
        for shape in pyb.getVisualShapeData(robot_uid):
            pyb.changeVisualShape(
                robot_uid,
                shape[1],
                rgbaColor=[*shape[7][:3], 0.0],
            )

    cutaway_links = set(camera.get("cutaway_links", []))
    if cutaway_links and world.static_obstacles_uid is not None:
        obstacle_uid = world.static_obstacles_uid
        for link_index in range(pyb.getNumJoints(obstacle_uid)):
            link_name = pyb.getJointInfo(obstacle_uid, link_index)[12].decode()
            if link_name not in cutaway_links:
                continue
            visual_data = pyb.getVisualShapeData(obstacle_uid)
            for shape in visual_data:
                if shape[1] == link_index:
                    rgba = [*shape[7][:3], 0.0]
                    pyb.changeVisualShape(obstacle_uid, link_index, rgbaColor=rgba)

    for position, marker_type in markers.values():
        if marker_type == "hidden":
            continue
        if marker_type == "base":
            _disc(position, 0.16, BASE_COLOR)
        elif marker_type == "start":
            _disc(position, 0.13, START_COLOR)
        else:
            _sphere(position, 0.075, EE_COLOR)

    width, height = 1200, 760
    view = pyb.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=camera["target"],
        distance=camera["distance"],
        yaw=camera["yaw"],
        pitch=camera["pitch"],
        roll=0.0,
        upAxisIndex=2,
    )
    projection = pyb.computeProjectionMatrixFOV(
        fov=camera["fov"],
        aspect=width / height,
        nearVal=0.05,
        farVal=20.0,
    )
    rgba = pyb.getCameraImage(
        width,
        height,
        viewMatrix=view,
        projectionMatrix=projection,
        renderer=pyb.ER_TINY_RENDERER,
        shadow=1,
        lightDirection=[-2.0, -3.0, 6.0],
    )[2]
    image = np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)[..., :3]
    pixels = _project(
        {name: position for name, (position, _) in markers.items()},
        view,
        projection,
        width,
        height,
    )
    pyb.disconnect()
    return image, pixels


def _task_targets(config: dict) -> tuple[list[np.ndarray], list[np.ndarray]]:
    base_targets = []
    ee_targets = []
    for task in config["planner"]["tasks"]:
        if "base_pose" in task:
            base_targets.append(np.asarray(task["base_pose"], dtype=float))
        if "ee_pose" in task:
            ee_targets.append(np.asarray(task["ee_pose"], dtype=float))
    if len(base_targets) != 2 or len(ee_targets) != 3:
        raise ValueError(
            "Fig. 3 expects the configured two-base-goal/three-EE-goal task; "
            f"found {len(base_targets)} and {len(ee_targets)} targets"
        )
    return base_targets, ee_targets


def _arrow(axis, start, end, color, connectionstyle="arc3,rad=0.0") -> None:
    axis.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "lw": 2.4,
            "linestyle": "--",
            "mutation_scale": 13,
            "connectionstyle": connectionstyle,
        },
    )


def _path_arrow(axis, pixels, color, linestyle="--") -> None:
    """Draw one continuous routed arrow with a single terminal arrowhead."""
    path = MplPath(
        np.asarray(pixels, dtype=float),
        [MplPath.MOVETO, *([MplPath.LINETO] * (len(pixels) - 1))],
    )
    axis.add_patch(
        FancyArrowPatch(
            path=path,
            arrowstyle="-|>",
            color=color,
            linewidth=2.4,
            linestyle=linestyle,
            mutation_scale=13,
            zorder=4,
        )
    )


def _label(axis, pixel, text, offset, color) -> None:
    axis.annotate(
        text,
        xy=pixel,
        xytext=offset,
        textcoords="offset points",
        color=color,
        fontsize=9.5,
        fontweight="bold",
        ha="center",
        va="center",
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "white",
            "edgecolor": color,
            "alpha": 0.94,
        },
        arrowprops={"arrowstyle": "-", "color": color, "lw": 1.3},
    )


def _target_symbols(axis, pixels, color) -> None:
    coordinates = np.asarray(list(pixels), dtype=float)
    axis.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        s=95,
        marker="o",
        facecolor=color,
        edgecolor="white",
        linewidth=1.4,
        zorder=5,
    )


def main() -> None:
    config = parsing.load_config(str(CONFIG))
    base_targets, ee_targets = _task_targets(config)
    q_initial = np.asarray(config["simulation"]["robot"]["home"], dtype=float)

    base_markers = {
        "start": (np.array([q_initial[0], q_initial[1], 0.055]), "start"),
        "route_1": (np.array([0.65, -1.85, 0.055]), "hidden"),
        "route_2": (np.array([1.35, -2.30, 0.055]), "hidden"),
        "base_1": (np.r_[base_targets[0][:2], 0.055], "base"),
        "base_2": (np.r_[base_targets[1][:2], 0.055], "base"),
    }
    overview = _render(
        config,
        q_initial,
        base_markers,
        {
            "target": [1.35, -0.75, 0.0],
            "distance": 5.0,
            "yaw": 5.0,
            "pitch": -72.0,
            "fov": 47.0,
        },
    )

    q_work = q_initial.copy()
    q_work[:3] = base_targets[-1]
    ee_markers = {
        f"ee_{index + 1}": (target[:3], "ee") for index, target in enumerate(ee_targets)
    }
    ee_markers.update(
        {
            "lower_clear_high": (np.array([2.58, -0.50, 0.90]), "hidden"),
            "lower_clear_low": (np.array([2.58, -0.50, 0.40]), "hidden"),
            "side_clear": (np.array([2.58, 0.50, 0.40]), "hidden"),
        }
    )
    manipulation = _render(
        config,
        q_work,
        ee_markers,
        {
            "target": [2.85, -0.05, 0.48],
            "distance": 2.70,
            "yaw": -105.0,
            "pitch": -38.0,
            "fov": 47.0,
            "hide_robot": True,
        },
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.45), constrained_layout=True)
    for axis, (image, _) in zip(axes, (overview, manipulation)):
        axis.imshow(image)
        axis.set_axis_off()

    axes[0].text(
        0.025,
        0.96,
        "(a) Base navigation: steps 1--2",
        transform=axes[0].transAxes,
        va="top",
        fontsize=10.5,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84},
    )
    axes[1].text(
        0.025,
        0.96,
        "(b) Oblique top view: manipulation steps 3--5",
        transform=axes[1].transAxes,
        va="top",
        fontsize=10.5,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84},
    )

    base_pixels = overview[1]
    route_2 = np.asarray(base_pixels["route_2"])
    base_1 = np.asarray(base_pixels["base_1"])
    arrow_end = base_1 + 24.0 * (route_2 - base_1) / np.linalg.norm(route_2 - base_1)
    _path_arrow(
        axes[0],
        [
            base_pixels["start"],
            base_pixels["route_1"],
            base_pixels["route_2"],
            arrow_end,
        ],
        "#0759b5",
    )
    _arrow(
        axes[0],
        base_pixels["base_1"],
        base_pixels["base_2"],
        "#0759b5",
        connectionstyle="arc3,rad=-0.13",
    )
    _label(axes[0], base_pixels["start"], "Start", (-5, -29), "#222222")
    _label(axes[0], base_pixels["base_1"], "1  Base goal", (-2, 31), "#0759b5")
    _label(axes[0], base_pixels["base_2"], "2  Work area", (10, -31), "#0759b5")

    ee_pixels = manipulation[1]
    _target_symbols(
        axes[1],
        [ee_pixels[f"ee_{index}"] for index in range(1, 4)],
        "#d51f14",
    )
    _path_arrow(
        axes[1],
        [
            ee_pixels["ee_1"],
            ee_pixels["lower_clear_high"],
            ee_pixels["lower_clear_low"],
            ee_pixels["ee_2"],
        ],
        "#b71910",
        linestyle="-",
    )
    _path_arrow(
        axes[1],
        [
            ee_pixels["ee_2"],
            ee_pixels["lower_clear_low"],
            ee_pixels["side_clear"],
            ee_pixels["ee_3"],
        ],
        "#b71910",
        linestyle="-",
    )
    _label(axes[1], ee_pixels["ee_1"], "3  Approach", (39, 10), "#b71910")
    _label(axes[1], ee_pixels["ee_2"], "4  Lower", (42, -8), "#b71910")
    _label(axes[1], ee_pixels["ee_3"], "5  Side transfer", (54, -44), "#b71910")

    fig.text(
        0.5,
        0.015,
        "Configured sequence: base goal  $\\rightarrow$  work area  "
        "$\\rightarrow$  approach  $\\rightarrow$  lower  "
        "$\\rightarrow$  side transfer",
        ha="center",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=240, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
