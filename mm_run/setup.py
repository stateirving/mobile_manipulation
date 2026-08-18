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
        (
            "share/" + package_name + "/launch",
            [p for p in glob("launch/*") if isfile(p)],
        ),
        ("share/" + package_name + "/config/controller", glob("config/controller/*")),
        ("share/" + package_name + "/config/robot", glob("config/robot/*")),
        ("share/" + package_name + "/config/scene", glob("config/scene/*")),
        ("share/" + package_name + "/config/sim", glob("config/sim/*")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/config", glob("config/*.md")),
        (
            "share/"
            + package_name
            + "/results/nvblox_esdf/esdf_test_room_full_2cm_dense/2026-06-15_12-32-10",
            glob(
                "results/nvblox_esdf/esdf_test_room_full_2cm_dense/"
                "2026-06-15_12-32-10/esdf_grid.npz"
            ),
        ),
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
            "mpc_ros=nodes.mpc_ros:main",
            "stretch_command_adapter=nodes.stretch_command_adapter:main",
            "stretch_wbmpc_shadow=nodes.stretch_wbmpc_shadow:main",
            "plot_real_base_path=scripts.plot_real_base_path:main",
            "plot_real_command_state=scripts.plot_real_command_state:main",
        ],
    },
)
