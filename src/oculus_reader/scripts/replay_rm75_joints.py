#!/usr/bin/env python3
"""Replay recorded RM75 joint trajectories from a dataset manifest.

No Quest or IK is used; recorded ``robot.*.joints_rad`` values are sent
directly to API2. Add --record-video to record live camera views during replay.
Start with ``--dry-run`` and a short frame range.
"""
import argparse
from datetime import datetime
import json
import math
import time
from pathlib import Path

from rm_api2_control import RealManAPI2Controller


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path)
    p.add_argument("--mode", choices=("single", "double"), default="double")
    p.add_argument("--right-ip", default="192.168.1.19")
    p.add_argument("--left-ip", default="192.168.1.18")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--rate", type=float, default=25.0)
    p.add_argument("--speed", type=float, default=1.0,
                   help="time scale; 0.5 is half speed, 2 is double")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int, default=0, help="0 means all frames")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-gripper", action="store_true")
    p.add_argument("--record-video", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--no-video", action="store_true",
                   help="disable default live-camera video recording")
    p.add_argument("--video-dir", type=Path,
                   help="new output directory; default: /media/lyj/Data/GaussianRDT_Videos/replay_video_TIMESTAMP (one MP4 per camera)")
    p.add_argument("--camera-config", type=Path,
                   default=Path(__file__).resolve().parent.parent / "config" / "cameras.yaml")
    args = p.parse_args(argv)
    if not all(math.isfinite(v) and v > 0 for v in (args.rate, args.speed)):
        p.error("--rate and --speed must be positive")
    if args.start < 0 or args.count < 0:
        p.error("--start and --count must be non-negative")
    with args.manifest.open() as manifest:
        rows = [json.loads(line) for line in manifest if line.strip()]
    rows = rows[args.start: args.start + args.count if args.count else None]
    if not rows:
        p.error("manifest contains no selected frames")
    controllers = {}
    video = None
    try:
        if (not args.no_video) or args.record_video or args.video_dir is not None:
            from replay_video import ReplayVideoRecorder
            video_dir = args.video_dir or Path("/media/lyj/Data/GaussianRDT_Videos") / (
                "replay_video_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
            video = ReplayVideoRecorder(video_dir, args.camera_config)
            video.start()
        ips = {"right": args.right_ip}
        if args.mode == "double":
            ips["left"] = args.left_ip
        for hand, ip in ips.items():
            controllers[hand] = RealManAPI2Controller(
                ip=ip, port=args.port, max_joint_step_deg=2.5,
                dry_run=args.dry_run)
        period = 1.0 / (args.rate * args.speed)
        next_tick = time.monotonic()
        sent = 0
        for index, row in enumerate(rows, start=args.start):
            if video is not None:
                video.check()
                video.replay_row = index
            robot = row.get("robot", {})
            for hand, controller in controllers.items():
                state = robot.get(hand, {})
                joints = state.get("joints_rad")
                if joints is None or len(joints) != 7:
                    continue
                controller.command_joints(joints)
                if not args.no_gripper and "gripper" in state:
                    controller.command_gripper(float(state["gripper"]))
                sent += 1
            next_tick += period
            time.sleep(max(0.0, next_tick - time.monotonic()))
        print("replayed {} arm samples from {}".format(sent, args.manifest))
    finally:
        try:
            for controller in controllers.values():
                controller.close()
        finally:
            if video is not None:
                video.close()


if __name__ == "__main__":
    main()
