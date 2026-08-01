from setuptools import find_packages, setup

package_name = "mm_utils"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(where="src", exclude=["test"]),
    package_dir={"": "src"},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="benni",
    maintainer_email="benjamin.bogenberger@tum.de",
    description="Mobile manipulation utilities package",
    license="MIT",
    entry_points={
        "console_scripts": [],
    },
)
