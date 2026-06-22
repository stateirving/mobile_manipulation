import argparse

from mm_control.MPC import MPC
from mm_utils import parsing

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Path to config file.")
    parser.add_argument(
        "-n",
        "--name",
        default="",
        required=False,
        help="Name Identifier for acados ocp. This will overwrite the name in the config file if provided.",
    )

    args = parser.parse_args()

    config = parsing.load_config(args.config)
    if args.name != "":
        config["controller"]["acados"]["name"] = args.name

    config["controller"]["acados"]["cython"]["enabled"] = True
    config["controller"]["acados"]["cython"]["recompile"] = True
    if config["controller"].get("esdf_collision", {}).get("enabled", False):
        config["controller"]["esdf_collision"]["initialize_map"] = False

    ctrl_config = config["controller"]
    ctrl_type = ctrl_config["type"]

    if ctrl_type == "MPC":
        control_class = MPC
    else:
        raise ValueError(f"Unknown controller type: {ctrl_type}")

    controller = control_class(ctrl_config)
