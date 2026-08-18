from glob import glob
from pathlib import Path

from setuptools import find_packages, setup

package_name = "mm_assets"


def recursive_data_files(source_dir):
    data_files = []
    root = Path(source_dir)
    if not root.exists():
        return data_files

    files_by_parent = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        files_by_parent.setdefault(path.parent, []).append(str(path))

    for parent, files in sorted(files_by_parent.items()):
        target = Path("share") / package_name / parent
        data_files.append((str(target), sorted(files)))
    return data_files


setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/config/ur10", glob("config/ur10/*.yaml")),
        ("share/" + package_name + "/thing/meshes", glob("thing/meshes/*.dae")),
        (
            "share/" + package_name + "/thing/meshes/ridgeback",
            glob("thing/meshes/ridgeback/*"),
        ),
        (
            "share/" + package_name + "/thing/meshes/ur10/collision",
            glob("thing/meshes/ur10/collision/*"),
        ),
        (
            "share/" + package_name + "/thing/meshes/ur10/visual",
            glob("thing/meshes/ur10/visual/*"),
        ),
        ("share/" + package_name + "/thing/xacro", glob("thing/xacro/*.xacro")),
        ("share/" + package_name + "/thing/xacro/ur_inc", glob("thing/xacro/ur_inc/*")),
        ("share/" + package_name + "/stretch", glob("stretch/*.urdf")),
        ("share/" + package_name + "/stretch/meshes", glob("stretch/meshes/*")),
        ("share/" + package_name + "/scenes", glob("scenes/*.sh")),
        ("share/" + package_name + "/scenes/xacro", glob("scenes/xacro/*")),
    ]
    + recursive_data_files("scenes/meshes")
    + recursive_data_files("scenes/urdf"),
    install_requires=["setuptools"],
    zip_safe=True,
    author="Benjamin Bogenberger, Xiaochen Miao",
    author_email="benjamin.bogenberger@tum.de, xiaochen.miao@tum.de",
    maintainer="benni",
    maintainer_email="benjamin.bogenberger@tum.de",
    description="Mobile manipulation robot and scene assets (URDF, meshes)",
    license="MIT",
    entry_points={
        "console_scripts": [],
    },
)
