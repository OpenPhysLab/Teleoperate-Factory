#!/usr/bin/env python3
"""Connect/read/disconnect smoke test; never sends a motion command."""

import argparse

from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", default="192.168.1.19")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    mode = getattr(rm_thread_mode_e, "RM_TRIPLE_MODE_E", None)
    arm = RoboticArm(mode) if mode is not None else RoboticArm()
    try:
        handle = arm.rm_create_robot_arm(args.ip, args.port, 3)
        print("CONNECT_OK handle_id={}".format(getattr(handle, "id", handle)))
        print("JOINT_STATE {}".format(arm.rm_get_joint_degree()))
    finally:
        arm.rm_delete_robot_arm()
        print("DISCONNECT_OK")


if __name__ == "__main__":
    main()
