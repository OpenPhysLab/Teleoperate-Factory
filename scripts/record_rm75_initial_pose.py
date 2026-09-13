#!/usr/bin/env python3
"""Read one RM75 joint state and save it as a reusable initial-pose YAML."""

import argparse
from datetime import datetime, timezone
import math
from pathlib import Path

import yaml

from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", default="192.168.1.19")
    parser.add_argument("--arm", choices=("left", "right"), default="right")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--output", default="src/oculus_reader/config/rm75_initial_poses.yaml")
    parser.add_argument("--gripper-width", type=float, default=0.07,
                        help="open gripper width in metres")
    args = parser.parse_args()
    mode = getattr(rm_thread_mode_e, "RM_TRIPLE_MODE_E", None)
    arm = RoboticArm(mode) if mode is not None else RoboticArm()
    try:
        handle = arm.rm_create_robot_arm(args.ip, args.port, 3)
        if handle is None:
            raise RuntimeError("API2 returned an empty handle")
        result = arm.rm_get_joint_degree()
        if isinstance(result, tuple) and len(result) >= 2:
            status, values = result[0], result[1]
            if status != 0:
                raise RuntimeError("joint state returned status {}".format(status))
        else:
            values = result
        degrees = [float(value) for value in list(values)[:7]]
        if len(degrees) != 7 or not all(math.isfinite(value) for value in degrees):
            raise RuntimeError("invalid seven-joint feedback: {}".format(degrees))
    finally:
        arm.rm_delete_robot_arm()
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    arm_payload = {
        "source_ip": args.ip,
        "source_captured_at": datetime.now(timezone.utc).isoformat(),
        "target_joint_state_deg": degrees,
        "target_joint_state": [math.radians(value) for value in degrees],
        "initial_gripper_width": max(0.0, float(args.gripper_width)),
        "gripper_state": "open" if args.gripper_width > 0 else "closed",
    }
    if output.exists():
        with output.open(encoding="utf-8") as stream:
            payload = yaml.safe_load(stream) or {}
    else:
        payload = {}
    payload["initial_gripper_width"] = max(0.0, float(args.gripper_width))
    payload["gripper_state"] = "open" if args.gripper_width > 0 else "closed"
    payload.setdefault("arms", {})[args.arm] = arm_payload
    with output.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(payload, stream, sort_keys=False, allow_unicode=True)
    print("saved initial pose from {} to {}".format(args.ip, output))
    print("joint_state_deg={}".format(degrees))


if __name__ == "__main__":
    main()
