#!/usr/bin/env python3
"""Convert AWS RoboMaker Small Warehouse models into this repo's scene format."""

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml


# The skipped props are either non-obstacle scenery or DAE assets that make
# PyBullet's URDF importer throw in this environment. Shelves and walls are
# enough for the first nvblox ESDF mapping pass.
DEFAULT_SKIP_MODELS = (
    "Ground",
    "Lamp",
    "Roof",
    "PalletJack",
    "TrashCan",
    "Cluttering",
    "Bucket",
)


def parse_pose(text):
    values = [float(v) for v in text.split()]
    if len(values) != 6:
        raise ValueError(f"Expected 6 pose values, got: {text}")
    xyz = values[:3]
    rpy = values[3:]
    return xyz, rpy


def model_name_from_uri(uri):
    prefix = "model://"
    if not uri.startswith(prefix):
        raise ValueError(f"Only model:// URIs are supported, got: {uri}")
    return uri[len(prefix) :].split("/", 1)[0]


def mesh_path_from_uri(uri):
    prefix = "model://"
    if not uri.startswith(prefix):
        raise ValueError(f"Only model:// URIs are supported, got: {uri}")
    model_name, rel_path = uri[len(prefix) :].split("/", 1)
    return model_name, rel_path


def parse_world(world_path, skip_models):
    root = ET.parse(world_path).getroot()
    world = root.find("world")
    if world is None:
        raise ValueError(f"No <world> element found in {world_path}")

    instances = []
    for model in world.findall("model"):
        include_uri = model.findtext("include/uri")
        pose_text = model.findtext("pose", "0 0 0 0 0 0")
        if include_uri is None:
            continue

        source_model_name = model_name_from_uri(include_uri.strip())
        if any(token in source_model_name for token in skip_models):
            continue

        xyz, rpy = parse_pose(pose_text)
        instances.append(
            {
                "instance_name": model.attrib["name"],
                "source_model_name": source_model_name,
                "xyz": xyz,
                "rpy": rpy,
            }
        )
    return instances


def parse_model_meshes(models_dir, model_name):
    sdf_path = models_dir / model_name / "model.sdf"
    root = ET.parse(sdf_path).getroot()
    model = root.find("model")
    if model is None:
        raise ValueError(f"No <model> element found in {sdf_path}")

    visual_uri = model.findtext(".//visual/geometry/mesh/uri")
    collision_uri = model.findtext(".//collision/geometry/mesh/uri")
    if visual_uri is None:
        raise ValueError(f"No visual mesh URI found in {sdf_path}")
    if collision_uri is None:
        collision_uri = visual_uri

    visual_model, visual_rel = mesh_path_from_uri(visual_uri.strip())
    collision_model, collision_rel = mesh_path_from_uri(collision_uri.strip())
    if visual_model != model_name or collision_model != model_name:
        raise ValueError(f"Unexpected model URI in {sdf_path}")
    return visual_rel, collision_rel


def copy_model_assets(models_dir, dest_models_dir, model_names):
    if dest_models_dir.exists():
        shutil.rmtree(dest_models_dir)
    dest_models_dir.mkdir(parents=True, exist_ok=True)
    for model_name in sorted(model_names):
        src = models_dir / model_name
        dst = dest_models_dir / model_name
        shutil.copytree(
            src,
            dst,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(".DS_Store", "__pycache__"),
        )


def copy_license(aws_root, dest_dir):
    license_path = aws_root / "LICENSE"
    if license_path.exists():
        shutil.copyfile(license_path, dest_dir / "LICENSE")


def sanitize_name(name):
    return name.replace("-", "_").replace(".", "_")


def mesh_filename(model_name, rel_path):
    return (
        "file://$(find mm_assets)/scenes/meshes/aws_small_warehouse/models/"
        f"{model_name}/{rel_path}"
    )


def write_xacro(xacro_path, instances, meshes_by_model):
    xacro_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        '<robot name="aws_small_warehouse" xmlns:xacro="http://www.ros.org/wiki/xacro">',
        "",
        '  <link name="aws_small_warehouse_base_link"/>',
        "",
    ]

    for item in instances:
        instance_name = sanitize_name(item["instance_name"])
        model_name = item["source_model_name"]
        _visual_rel, collision_rel = meshes_by_model[model_name]
        xyz = " ".join(f"{v:.6g}" for v in item["xyz"])
        rpy = " ".join(f"{v:.6g}" for v in item["rpy"])
        collision_mesh = mesh_filename(model_name, collision_rel)
        visual_mesh = collision_mesh

        lines += [
            f'  <link name="{instance_name}_link">',
            "    <collision>",
            '      <origin rpy="0 0 0" xyz="0 0 0"/>',
            "      <geometry>",
            f'        <mesh filename="{collision_mesh}" scale="1 1 1"/>',
            "      </geometry>",
            "    </collision>",
            "    <visual>",
            '      <origin rpy="0 0 0" xyz="0 0 0"/>',
            "      <geometry>",
            f'        <mesh filename="{visual_mesh}" scale="1 1 1"/>',
            "      </geometry>",
            "    </visual>",
            "  </link>",
            f'  <joint name="{instance_name}_joint" type="fixed">',
            f'    <origin xyz="{xyz}" rpy="{rpy}"/>',
            '    <parent link="aws_small_warehouse_base_link"/>',
            f'    <child link="{instance_name}_link"/>',
            "  </joint>",
            "",
        ]

    lines.append("</robot>")
    xacro_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_scene_yaml(scene_yaml_path):
    scene_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    scene = {
        "simulation": {
            "static_obstacles": {
                "enabled": True,
                "urdf": {
                    "package": "mm_assets",
                    "path": "scenes/urdf/aws_small_warehouse.urdf",
                    "includes": [
                        "$(find mm_assets)/scenes/xacro/aws_small_warehouse.urdf.xacro"
                    ],
                },
            }
        },
        "controller": {
            "scene": {
                "enabled": True,
                # The AWS warehouse uses mesh obstacles. The current analytic
                # MPC collision stack supports primitive signed distances, so
                # leave this empty and use nvblox ESDF for collision costs.
                "collision_link_names": {"static_obstacles": []},
                "urdf": {
                    "package": "mm_assets",
                    "path": "scenes/urdf/aws_small_warehouse.urdf",
                    "includes": [
                        "$(find mm_assets)/scenes/xacro/aws_small_warehouse.urdf.xacro"
                    ],
                },
            }
        },
    }
    scene_yaml_path.write_text(yaml.safe_dump(scene, sort_keys=False), encoding="utf-8")


def write_export_yaml(export_yaml_path):
    export_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    export_yaml_path.write_text(
        """simulation:
  robot:
    home: [0, 0, 0, 0.25pi, -0.25pi, 0.5pi, -0.25pi, 0.5pi, 0.417pi]
  static_obstacles:
    enabled: true

include:
  - package: "mm_run"
    path: "config/sim/simulation.yaml"
  - package: "mm_run"
    path: "config/scene/aws_small_warehouse.yaml"

logging:
  log_dir: "aws_small_warehouse_esdf"
  log_level: 20
""",
        encoding="utf-8",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert AWS RoboMaker Small Warehouse to mm_assets/mm_run scene files."
    )
    parser.add_argument(
        "--aws-root",
        default="third_party/aws-robomaker-small-warehouse-world",
        help="Path to the cloned aws-robomaker-small-warehouse-world repository.",
    )
    parser.add_argument(
        "--world",
        default="worlds/no_roof_small_warehouse.world",
        help="World file relative to --aws-root.",
    )
    parser.add_argument(
        "--skip-model-token",
        action="append",
        default=list(DEFAULT_SKIP_MODELS),
        help="Skip AWS models whose names contain this token. Can be repeated.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    aws_root = (repo_root / args.aws_root).resolve()
    world_path = aws_root / args.world
    models_dir = aws_root / "models"

    instances = parse_world(world_path, tuple(args.skip_model_token))
    if not instances:
        raise RuntimeError(f"No model instances found in {world_path}")

    model_names = {item["source_model_name"] for item in instances}
    meshes_by_model = {
        model_name: parse_model_meshes(models_dir, model_name)
        for model_name in sorted(model_names)
    }

    copy_model_assets(
        models_dir,
        repo_root / "mm_assets/scenes/meshes/aws_small_warehouse/models",
        model_names,
    )
    copy_license(aws_root, repo_root / "mm_assets/scenes/meshes/aws_small_warehouse")
    write_xacro(
        repo_root / "mm_assets/scenes/xacro/aws_small_warehouse.urdf.xacro",
        instances,
        meshes_by_model,
    )
    write_scene_yaml(repo_root / "mm_run/config/scene/aws_small_warehouse.yaml")
    write_export_yaml(repo_root / "mm_run/config/aws_small_warehouse_esdf.yaml")

    print(f"Converted {len(instances)} instances from {world_path}")
    print("Wrote mm_assets/scenes/xacro/aws_small_warehouse.urdf.xacro")
    print("Wrote mm_run/config/scene/aws_small_warehouse.yaml")
    print("Wrote mm_run/config/aws_small_warehouse_esdf.yaml")


if __name__ == "__main__":
    main()
