"""Launch the real Stretch command adapter in non-commanding shadow mode."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config = PathJoinSubstitution(
        [FindPackageShare("mm_run"), "config", "stretch_command_adapter_real.yaml"]
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("config", default_value=default_config),
            DeclareLaunchArgument("log", default_value=""),
            Node(
                package="mm_run",
                executable="stretch_command_adapter",
                name="stretch_command_adapter",
                output="screen",
                arguments=[
                    "--config",
                    LaunchConfiguration("config"),
                    "--log",
                    LaunchConfiguration("log"),
                ],
            ),
        ]
    )
