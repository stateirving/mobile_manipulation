from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    args = [
        DeclareLaunchArgument("config"),
        DeclareLaunchArgument("ctrl_config", default_value="default"),
        DeclareLaunchArgument("planner_config", default_value="default"),
        DeclareLaunchArgument("logging_sub_folder", default_value="default"),
    ]

    mm_run_share = FindPackageShare("mm_run")
    central_share = FindPackageShare("mobile_manipulation_central")
    assets_share = FindPackageShare("mm_assets")

    pybullet_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([mm_run_share, "launch", "pybullet_sim.launch.py"])
        ),
        launch_arguments={
            k: LaunchConfiguration(k)
            for k in ["config", "ctrl_config", "planner_config", "logging_sub_folder"]
        }.items(),
    )

    controller = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([mm_run_share, "launch", "controller.launch.py"])
        ),
        launch_arguments={
            k: LaunchConfiguration(k)
            for k in ["config", "ctrl_config", "planner_config", "logging_sub_folder"]
        }.items(),
    )

    kinematics_params = LaunchConfiguration("kinematics_params")
    visual_params = LaunchConfiguration("visual_params")
    transform_params = LaunchConfiguration("transform_params")
    file_name = LaunchConfiguration("file_name")

    args += [
        DeclareLaunchArgument(
            "kinematics_params",
            default_value=PathJoinSubstitution(
                [central_share, "config/ur10/kinematics_parameters.yaml"]
            ),
        ),
        DeclareLaunchArgument(
            "visual_params",
            default_value=PathJoinSubstitution(
                [central_share, "config/ur10/visual_parameters.yaml"]
            ),
        ),
        DeclareLaunchArgument(
            "transform_params",
            default_value=PathJoinSubstitution(
                [central_share, "config/urdf_transform.yaml"]
            ),
        ),
        DeclareLaunchArgument(
            "file_name",
            default_value=PathJoinSubstitution(
                [assets_share, "thing/urdf/thing_sim.urdf"]
            ),
        ),
    ]

    robot_description_content = ParameterValue(
        Command(
            [
                "xacro ",
                file_name,
                " kinematics_params:=",
                kinematics_params,
                " visual_params:=",
                visual_params,
                " transform_params:=",
                transform_params,
            ]
        ),
        value_type=str,
    )

    ur10_state_pub = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="ur10_state_publisher",
        namespace="ur10",
        parameters=[{"robot_description": robot_description_content}],
    )

    ridgeback_state_pub = Node(
        package="mobile_manipulation_central",
        executable="ridgeback_state_publisher_node",
        name="ridgeback_state_publisher",
        output="screen",
    )

    return LaunchDescription(
        args
        + [
            pybullet_sim,
            controller,
            ur10_state_pub,
            ridgeback_state_pub,
        ]
    )
