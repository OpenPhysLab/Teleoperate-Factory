#!/usr/bin/env python3
"""Move one RM75 to the saved initial pose after explicit --apply."""

import argparse
import math
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "oculus_reader" / "scripts"))
from rm_api2_control import RealManAPI2Controller


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", default=None)
    parser.add_argument("--arm", choices=("left", "right"), default="left")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--pose", default="src/oculus_reader/config/rm75_initial_poses.yaml")
    parser.add_argument("--apply", action="store_true",
                        help="required; without it only prints the target")
    parser.add_argument("--duration", type=float, default=2.0)
    args = parser.parse_args()
    with Path(args.pose).expanduser().open(encoding="utf-8") as stream:
        pose = yaml.safe_load(stream) or {}
    args.ip = args.ip or ("192.168.1.18" if args.arm == "left" else "192.168.1.19")
    if pose.get("arms", {}).get(args.arm):
        pose = dict(pose, **pose["arms"][args.arm])
    joints = pose.get("target_joint_state")
    if not joints or len(joints) != 7:
        raise SystemExit("pose file must contain seven target_joint_state radians")
    width = float(pose.get("initial_gripper_width", 0.07))
    print("target {} joints(deg): {}".format(
        args.ip, [round(math.degrees(float(value)), 3) for value in joints]))
    print("target gripper width: {:.4f} m".format(width))
    if not args.apply:
        print("dry inspection only; add --apply to move the arm")
        return 0
    controller = RealManAPI2Controller(
        ip=args.ip, port=args.port, target_joint_state=joints,
        init_duration=max(0.1, args.duration))
    try:
        controller.init_pose(width)
    finally:
        controller.close()
    print("initial pose applied to {}".format(args.ip))


if __name__ == "__main__":
    main()
