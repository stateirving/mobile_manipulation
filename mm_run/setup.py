from glob import glob
from os.path import isfile
from setuptools import find_packages, setup

package_name = "mm_run"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*")),
        ("share/" + package_name + "/nodes", [p for p in glob("nodes/*") if isfile(p)]),
        ("share/" + package_name + "/rviz", glob("rviz/*")),
        ("share/" + package_name + "/config/controller", glob("config/controller/*")),
        ("share/" + package_name + "/config/robot", glob("config/robot/*")),
        ("share/" + package_name + "/config/scene", glob("config/scene/*")),
        ("share/" + package_name + "/config/sensor", glob("config/sensor/*")),
        ("share/" + package_name + "/config/sim", glob("config/sim/*")),
        ("share/" + package_name + "/config/teleop", glob("config/teleop/*")),
        ("share/" + package_name + "/config/test_experiment", glob("config/test_experiment/*")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/config", glob("config/*.md")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="benni",
    maintainer_email="benjamin.bogenberger@tum.de",
    description="Mobile manipulation run/launch package",
    license="MIT",
    entry_points={
        "console_scripts": [
            "experiment=scripts.experiment:main",
            "sim_ros=nodes.sim_ros:main",
            "isaac_sim_ros=nodes.isaac_sim_ros:main",
            "mpc_ros=nodes.mpc_ros:main",
            "planner_test_ros=nodes.planner_test_ros:main",
        ],
    },
)
