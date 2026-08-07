"""Launch simulated-ESDF WB-MPC against real Stretch state and commands."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _adapter(arguments, condition):
    return Node(
        package="mm_run",
        executable="stretch_command_adapter",
        name="stretch_command_adapter",
        output="screen",
        emulate_tty=True,
        arguments=arguments,
        condition=condition,
    )


def generate_launch_description():
    package_share = FindPackageShare("mm_run")
    adapter_config = LaunchConfiguration("adapter_config")
    wbmpc_config = LaunchConfiguration("wbmpc_config")
    adapter_log = LaunchConfiguration("adapter_log")
    wbmpc_log = LaunchConfiguration("wbmpc_log")
    execute = LaunchConfiguration("execute")

    adapter_base_arguments = [
        "--config",
        adapter_config,
        "--log",
        adapter_log,
    ]
    shadow_adapter = _adapter(
        adapter_base_arguments,
        UnlessCondition(execute),
    )
    execute_adapter = _adapter(
        adapter_base_arguments + ["--execute"],
        IfCondition(execute),
    )
    runner = Node(
        package="mm_run",
        executable="stretch_wbmpc_shadow",
        name="stretch_wbmpc_runner",
        output="screen",
        emulate_tty=True,
        arguments=["--config", wbmpc_config, "--log", wbmpc_log],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("execute", default_value="false"),
            DeclareLaunchArgument("execute_delay", default_value="8.0"),
            DeclareLaunchArgument(
                "adapter_config",
                default_value=PathJoinSubstitution(
                    [package_share, "config", "stretch_command_adapter_real.yaml"]
                ),
            ),
            DeclareLaunchArgument(
                "wbmpc_config",
                default_value=PathJoinSubstitution(
                    [
                        package_share,
                        "config",
                        "stretch_wbmpc_sim_esdf_real_test.yaml",
                    ]
                ),
            ),
            DeclareLaunchArgument("adapter_log", default_value=""),
            DeclareLaunchArgument("wbmpc_log", default_value=""),
            runner,
            shadow_adapter,
            TimerAction(
                period=LaunchConfiguration("execute_delay"),
                actions=[execute_adapter],
            ),
        ]
    )
