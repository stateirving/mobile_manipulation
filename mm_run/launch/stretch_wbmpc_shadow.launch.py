"""Launch real-state WB-MPC and the command adapter with hardware output disabled."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_share = FindPackageShare("mm_run")
    default_adapter_config = PathJoinSubstitution(
        [package_share, "config", "stretch_command_adapter_real.yaml"]
    )
    default_wbmpc_config = PathJoinSubstitution(
        [package_share, "config", "stretch_wbmpc_shadow_real.yaml"]
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "adapter_config", default_value=default_adapter_config
            ),
            DeclareLaunchArgument("wbmpc_config", default_value=default_wbmpc_config),
            DeclareLaunchArgument("adapter_log", default_value=""),
            DeclareLaunchArgument("wbmpc_log", default_value=""),
            Node(
                package="mm_run",
                executable="stretch_command_adapter",
                name="stretch_command_adapter",
                output="screen",
                arguments=[
                    "--config",
                    LaunchConfiguration("adapter_config"),
                    "--log",
                    LaunchConfiguration("adapter_log"),
                ],
            ),
            Node(
                package="mm_run",
                executable="stretch_wbmpc_shadow",
                name="stretch_wbmpc_shadow",
                output="screen",
                arguments=[
                    "--config",
                    LaunchConfiguration("wbmpc_config"),
                    "--log",
                    LaunchConfiguration("wbmpc_log"),
                ],
            ),
        ]
    )
