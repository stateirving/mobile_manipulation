import argparse
import datetime
import os
import sys
import time

import numpy as np
import rclpy
from rclpy.utilities import remove_ros_args
from mobile_manipulation_central.simulation_ros_interface import (
    SimulatedMobileManipulatorROSInterface,
    SimulatedViconObjectInterface,
)

from mm_simulator import simulation
from mm_utils import parsing
from mm_utils.logging import DataLogger


def main(argv=None):
    np.set_printoptions(precision=3, suppress=True)
    
    # Use sys.argv if argv is not provided
    if argv is None:
        argv = sys.argv
    
    # Initialize ROS2 (this must be done before removing ROS args)
    rclpy.init(args=argv)
    
    # Remove ROS-specific arguments before parsing script arguments
    argv_without_ros = remove_ros_args(args=argv)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", required=True, help="Path to configuration file."
    )
    parser.add_argument(
        "--video",
        nargs="?",
        default=None,
        const="",
        help="Record video. Optionally specify prefix for video directory.",
    )
    parser.add_argument(
        "--logging_sub_folder",
        type=str,
        help="save data in a sub folder of logging directory",
    )
    parser.add_argument(
        "--GUI",
        action="store_true",
        help="Enable PyBullet GUI. This overwrites the yaml settings",
    )
    args = parser.parse_args(argv_without_ros[1:])

    # load configuration and overwrite with args
    config = parsing.load_config(args.config)

    if args.GUI:
        config["simulation"]["gui"] = True

    if args.logging_sub_folder != "default":
        config["logging"]["log_dir"] = os.path.join(
            config["logging"]["log_dir"], args.logging_sub_folder
        )

    sim_config = config["simulation"]

    # start the simulation
    timestamp = datetime.datetime.now()
    sim = simulation.BulletSimulation(
        config=sim_config, timestamp=timestamp, cli_args=args
    )
    robot = sim.robot

    # initial time, state, input
    t = 0.0

    # Create shared timestamp for logging (format: YYYY-MM-DD_HH-MM-SS)
    session_timestamp = timestamp.strftime("%Y-%m-%d_%H-%M-%S")

    # Create ROS2 node (rclpy.init() already called at start of main())
    node = rclpy.create_node("sim_ros")
    node.declare_parameter("experiment_timestamp", session_timestamp)

    # init logger
    logger = DataLogger(config, name="sim")
    logger.add("sim_timestep", sim.timestep)
    logger.add("duration", sim.duration)

    logger.add("nq", sim_config["robot"]["dims"]["q"])
    logger.add("nv", sim_config["robot"]["dims"]["v"])
    logger.add("nx", sim_config["robot"]["dims"]["x"])
    logger.add("nu", sim_config["robot"]["dims"]["u"])

    # The ROS interface historically assumes a 3-DoF mobile base plus an arm
    # published on the legacy /ur10 topics. Prefer simulation joint names,
    # but tolerate configs that only define them under controller.robot.
    joint_names = sim_config["robot"].get(
        "joint_names", config.get("controller", {}).get("robot", {}).get("joint_names")
    )
    if joint_names is None:
        raise KeyError("Missing robot.joint_names in both simulation and controller config")
    arm_joint_names = joint_names[3:]
    ros_interface = SimulatedMobileManipulatorROSInterface(
        node, arm_joint_names=arm_joint_names
    )
    ros_interface.publish_time(t)

    vicon_tool_interface = SimulatedViconObjectInterface(
        node, sim_config["robot"]["tool_vicon_name"]
    )
    while not ros_interface.ready():
        rclpy.spin_once(node, timeout_sec=0.0)
        q, v = robot.joint_states()
        ros_interface.publish_feedback(t, q, v)
        ros_interface.publish_time(t)
        t += sim.timestep
        time.sleep(sim.timestep)
        if not rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()
            return

    print("Control commands received. Proceed ... ")
    t0 = t
    while rclpy.ok() and t - t0 <= sim.duration:
        q, v = robot.joint_states()
        ros_interface.publish_feedback(t, q, v)
        ros_interface.publish_time(t)

        cmd_vel_world = robot.command_velocity(ros_interface.cmd_vel, bodyframe=True)
        ee_curr_pos, ee_curr_orn = robot.link_pose()
        base_curr_pos, _ = robot.link_pose(-1)
        vicon_tool_interface.publish_pose(t, ee_curr_pos, ee_curr_orn)

        # log
        r_ew_w, Q_we = robot.link_pose()
        v_ew_w, ω_ew_w = robot.link_velocity()
        logger.append("ts", t)
        logger.append("xs", np.hstack((q, v)))
        logger.append("cmd_vels", cmd_vel_world)
        logger.append("r_ew_ws", r_ew_w)
        logger.append("Q_wes", Q_we)
        logger.append("v_ew_ws", v_ew_w)
        logger.append("ω_ew_ws", ω_ew_w)

        logger.append("r_bw_ws", q[:2])
        logger.append("yaw_bw_ws", q[2])
        logger.append("v_bw_ws", v[:2])
        logger.append("ω_bw_ws", v[2])

        t, _ = sim.step(t)
        start_time = time.perf_counter()
        while start_time + sim.timestep > time.perf_counter():
            rclpy.spin_once(node, timeout_sec=0.0)

    logger.save(session_timestamp=session_timestamp)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
