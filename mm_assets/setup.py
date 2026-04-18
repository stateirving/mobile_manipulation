from glob import glob
from setuptools import find_packages, setup

package_name = 'mm_assets'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/thing/meshes', glob('thing/meshes/*.dae')),
        ('share/' + package_name + '/thing/meshes/ridgeback', glob('thing/meshes/ridgeback/*')),
        ('share/' + package_name + '/thing/meshes/ur10/collision', glob('thing/meshes/ur10/collision/*')),
        ('share/' + package_name + '/thing/meshes/ur10/visual', glob('thing/meshes/ur10/visual/*')),
        ('share/' + package_name + '/thing/xacro', glob('thing/xacro/*')),
        ('share/' + package_name + '/stretch', glob('stretch/*.urdf')),
        ('share/' + package_name + '/stretch/meshes', glob('stretch/meshes/*')),
        ('share/' + package_name + '/scenes', glob('scenes/*.sh')),
        ('share/' + package_name + '/scenes/xacro', glob('scenes/xacro/*')),
        ('share/' + package_name + '/scenes/meshes', glob('scenes/meshes/*.dae')),
        ('share/' + package_name + '/scenes/meshes/chair/model', glob('scenes/meshes/chair/model/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='benni',
    maintainer_email='benjamin.bogenberger@tum.de',
    description='Mobile manipulation robot and scene assets (URDF, meshes)',
    license='MIT',
    entry_points={
        'console_scripts': [
        ],
    },
)
