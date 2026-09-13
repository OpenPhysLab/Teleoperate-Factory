#!/usr/bin/env python3
"""Standalone Meta Quest -> RM75 teleoperation using RealMan API2.

This entry point deliberately imports no rospy, tf, geometry_msgs or ROS
messages.  It reads the same Quest APK stream through :class:`OculusReader`,
solves RM75 IK locally, and sends API2 joint commands directly.
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import time

import numpy as np

from rm_api2_control import RealManAPI2Controller
try:
    from oculus_reader import OculusReader
    _QUEST_IMPORT_ERROR = None
except ImportError as exc:
    OculusReader = None
    _QUEST_IMPORT_ERROR = exc

try:
    import pinocchio as pin
    from robot_ik import (Arm_IK, calc_pose_incre, create_transformation_matrix,
                          matrix_to_xyzrpy, quaternion_from_matrix)
    _IK_IMPORT_ERROR = None
except ImportError as exc:
    # Keep ``--help`` and configuration inspection usable on a clean machine;
    # run() reports the actionable dependency error before connecting hardware.
    pin = None
    _IK_IMPORT_ERROR = exc


STOP = False


def _stop_handler(signum, frame):
    del signum, frame
    global STOP
    STOP = True


def adjustment_matrix(transform):
    """Return Quest pose in the APK tracking frame.

    DAgger's verified ``perm_xyz_pnn`` mapping is applied to *increments*
    later.  The former Piper adjustment rotation must not be applied here,
    otherwise the mapping is rotated twice.
    """
    transform = np.asarray(transform, dtype=float)
    if transform.shape != (4, 4):
        raise ValueError("Quest transform must be 4x4")
    return transform.copy()


def load_config(path):
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load rm75.yaml") from exc
    with open(path, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    pose_file = config.get("initial_pose_file")
    if pose_file:
        pose_path = pose_file if os.path.isabs(pose_file) else os.path.join(
            os.path.dirname(os.path.abspath(path)), pose_file)
        with open(pose_path, "r", encoding="utf-8") as stream:
            pose = yaml.safe_load(stream) or {}
        if pose.get("target_joint_state"):
            config["target_joint_state"] = pose["target_joint_state"]
        for arm_name, arm_pose in (pose.get("arms", {}) or {}).items():
            if arm_pose.get("target_joint_state"):
                config["{}_target_joint_state".format(arm_name)] = arm_pose["target_joint_state"]
        if pose.get("initial_gripper_width") is not None:
            config["initial_gripper_width"] = pose["initial_gripper_width"]
    return config


def param(config, name, default, arm=None):
    if arm and "{}_{}".format(arm, name) in config:
        return config["{}_{}".format(arm, name)]
    return config.get(name, default)


def make_controller(config, args, arm=None):
    ip = (getattr(args, "{}_ip".format(arm), None) if arm else args.ip)
    if not ip and arm:
        ip = param(config, "{}_rm_ip".format(arm), None)
    controller = RealManAPI2Controller(
        ip=ip or param(config, "rm_ip", "192.168.1.18", arm),
        port=args.port,
        lower_deg=param(config, "joint_lower_deg", None, arm),
        upper_deg=param(config, "joint_upper_deg", None, arm),
        target_joint_state=param(config, "target_joint_state", None, arm),
        gripper_max_width=param(config, "gripper_max_width", 0.07, arm),
        gripper_inverted=param(config, "gripper_inverted", False, arm),
        max_joint_step_deg=param(config, "max_joint_step_deg", 2.5, arm),
        init_duration=param(config, "init_pose_duration", 1.5, arm),
        init_rate=param(config, "init_pose_rate", 50.0, arm),
        high_follow=param(config, "rm_high_follow", False, arm),
        trajectory_mode=param(config, "rm_trajectory_mode", 1, arm),
        radio=param(config, "rm_radio", 50, arm),
        log_level=param(config, "rm_log_level", 3, arm),
        dry_run=args.dry_run)
    controller.teleop_mapping = param(config, "teleop_mapping", None, arm)
    controller.use_internal_ik = bool(param(config, "rm_use_internal_ik", False, arm))
    return controller


def solve_and_send(controller, solver, raw_transform, base_pose, zero_pose,
                   trigger, enabled, block_collision):
    # Clutch-style relative control.  Absolute Quest coordinates are not
    # meaningful until a button is pressed; anchoring on the rising edge
    # prevents tracking offsets from driving the arm while the hand is still.
    if not enabled:
        setattr(controller, "_rm_follow_anchor", None)
        return False
    hand_tf = adjustment_matrix(raw_transform)
    if not np.all(np.isfinite(hand_tf)) or hand_tf.shape != (4, 4):
        return False
    anchor = getattr(controller, "_rm_follow_anchor", None)
    if anchor is None:
        try:
            q0 = controller.current_joint_positions(strict=True)
            api_pose = controller.current_tcp_pose(strict=not controller.dry_run)
            if api_pose is not None:
                t0 = create_transformation_matrix(*api_pose)
            else:
                t0 = solver.forward_pose(q0)
            solver.reset_seed(q0)
        except Exception:
            return False
        setattr(controller, "_rm_follow_anchor", (hand_tf.copy(), t0.copy(), q0.copy()))
        return False  # clutch engagement never commands a jump
    h0, t0, _ = anchor
    # DAgger-style decoupled mapping: translate the Quest origin in its fixed
    # tracking frame, then rotate that delta by a per-arm constant map.  Do
    # not use H0^-1 H for translation: that couples hand yaw/pitch to XYZ.
    # First express world-space translation in the hand's initial local
    # frame.  The configured C matrix describes the *initial axis alignment*
    # between that hand frame and the RM flange frame; applying C directly to
    # world deltas was the source of forward/backward -> vertical motion.
    hand_dp = h0[:3, :3].T.dot(hand_tf[:3, 3] - h0[:3, 3])
    target_transform = t0.copy()
    mapping = getattr(controller, "teleop_mapping", None)
    if mapping:
        perm = np.asarray(mapping.get("position_permutation", [0, 1, 2]), dtype=int)
        sign = np.asarray(mapping.get("position_sign", [1, 1, 1]), dtype=float)
        if perm.shape == (3,) and sign.shape == (3,):
            hand_dp = sign * hand_dp[perm]
        scale = np.asarray(mapping.get("position_scale", [1, 1, 1]), dtype=float)
        if scale.shape == (3,):
            hand_dp = hand_dp * scale
        map_rot = np.asarray(mapping.get("position_rotation", np.eye(3)), dtype=float)
        if map_rot.shape == (3, 3) and np.all(np.isfinite(map_rot)):
            hand_dp = map_rot.dot(hand_dp)
    target_transform[:3, 3] = t0[:3, 3] + hand_dp
    if mapping and bool(mapping.get("track_rotation", True)):
        rdelta = h0[:3, :3].T.dot(hand_tf[:3, :3])
        cmap = np.asarray(mapping.get("position_rotation", np.eye(3)), dtype=float)
        if cmap.shape == (3, 3):
            rdelta = cmap.dot(rdelta).dot(cmap.T)
        target_transform[:3, :3] = t0[:3, :3].dot(rdelta)
        if bool(mapping.get("lock_vertical_down", False)):
            # Keep the flange tool axis vertical/down; permit only yaw about
            # that axis so the actuator can rotate without tilting.
            yaw = math.atan2(rdelta[1, 0], rdelta[0, 0])
            cy, sy = math.cos(yaw), math.sin(yaw)
            target_transform[:3, :3] = t0[:3, :3].dot(
                np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]]))
    if not np.all(np.isfinite(target_transform)):
        return False
    if getattr(controller, "use_internal_ik", False):
        # RealMan API2 performs the RM75-specific IK in the controller.
        # This avoids mixing a Piper-derived numerical solver with RM75-6F.
        result = controller.command_pose(matrix_to_xyzrpy(target_transform),
                                         follow=controller.high_follow)
        return result == 0
    quat = quaternion_from_matrix(target_transform)
    target_se3 = pin.SE3(pin.Quaternion(quat[3], quat[0], quat[1], quat[2]),
                         np.asarray(target_transform[:3, 3], dtype=float))
    q, _, collision = solver.ik_fun(target_se3.homogeneous, 0)
    if q is None or (collision and block_collision):
        return False
    # A single bad tracking/IK frame must never be converted into a long
    # limit-clipped march toward a joint limit.  Compare against the last
    # successfully commanded vector before handing it to API2.
    previous_q = getattr(controller, "last_command", None)
    if previous_q is not None:
        prev = np.asarray(previous_q, dtype=float).reshape(-1)
        cand = np.asarray(q, dtype=float).reshape(-1)
        if prev.shape == cand.shape and np.max(np.abs(cand - prev)) > np.deg2rad(15.0):
            return False
    try:
        controller.command_joints(q)
        # Quest reports trigger/grip as 0 at rest and 1 when squeezed.  RM75
        # width is the opposite convention: max width means open.  Keep the
        # hand open while following with no trigger pressed.
        controller.command_gripper((1.0 - float(trigger)) *
                                    controller.gripper_max_width)
        return True
    except Exception:
        return False


def run(args):
    if args.rate <= 0.0:
        raise ValueError("--rate must be greater than zero")
    if _IK_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Pinocchio is required for RM75 IK (CasADi is optional): {}".format(
                _IK_IMPORT_ERROR))
    if _QUEST_IMPORT_ERROR is not None:
        raise RuntimeError(
            "pure-python-adb is required for Quest input: {}".format(
                _QUEST_IMPORT_ERROR))
    config = load_config(args.config)
    reader = OculusReader(ip_address=args.quest_ip)
    controllers = []
    try:
        if args.mode == "single":
            right = make_controller(config, args)
            right_ik = Arm_IK(params=config)
            right.init_pose(param(config, "initial_gripper_width", 0.0))
            controllers.append(right)
            base_right = list(param(config, "base_pose", config.get(
                "right_base_pose", [0.19, 0.0, 0.2, 0, 0, 0])))
            zero_right = list(param(config, "teleop_zero_pose",
                                    [0.0, 0.0, 0.8505, 0, 0, 0]))
        else:
            right = make_controller(config, args, "right")
            controllers.append(right)
            left = make_controller(config, args, "left")
            controllers.append(left)
            right_ik, left_ik = Arm_IK("right", config), Arm_IK("left", config)
            right.init_pose(param(config, "initial_gripper_width", 0.0, "right"))
            left.init_pose(param(config, "initial_gripper_width", 0.0, "left"))
            base_right = list(param(config, "base_pose",
                                    [0.19, 0.0, 0.2, 0, 0, 0], "right"))
            base_left = list(param(config, "base_pose",
                                   [0.19, 0.0, 0.2, 0, 0, 0], "left"))
            zero_right = list(param(config, "teleop_zero_pose",
                                    [0.0, 0.0, 0.8505, 0, 0, 0], "right"))
            zero_left = list(param(config, "teleop_zero_pose",
                                   zero_right, "left"))

        next_tick = time.monotonic()
        previous_right_reset = False
        previous_left_reset = False
        while not STOP:
            transforms, buttons = reader.get_transformations_and_buttons()
            if args.mode == "single":
                if "r" in transforms:
                    right_reset = bool(buttons.get("A", False))
                    if right_reset and not previous_right_reset:
                        right.init_pose(param(config, "initial_gripper_width", 0.0))
                        base_right = matrix_to_xyzrpy(adjustment_matrix(transforms["r"]))
                    solve_and_send(right, right_ik, transforms["r"], base_right,
                                   zero_right, _trigger(buttons, "rightTrig"),
                                   buttons.get("B", False), args.allow_collision is False)
                    previous_right_reset = right_reset
            else:
                if "r" in transforms:
                    right_reset = bool(buttons.get("A", False))
                    if right_reset and not previous_right_reset:
                        right.init_pose(param(config, "initial_gripper_width", 0.0, "right"))
                        base_right = matrix_to_xyzrpy(adjustment_matrix(transforms["r"]))
                    solve_and_send(right, right_ik, transforms["r"], base_right,
                                   zero_right, _trigger(buttons, "rightTrig"),
                                   buttons.get("B", False), args.allow_collision is False)
                    previous_right_reset = right_reset
                if "l" in transforms:
                    left_reset = bool(buttons.get("X", False))
                    if left_reset and not previous_left_reset:
                        left.init_pose(param(config, "initial_gripper_width", 0.0, "left"))
                        base_left = matrix_to_xyzrpy(adjustment_matrix(transforms["l"]))
                    solve_and_send(left, left_ik, transforms["l"], base_left,
                                   zero_left, _trigger(buttons, "leftTrig"),
                                   buttons.get("Y", False), args.allow_collision is False)
                    previous_left_reset = left_reset
            next_tick += 1.0 / args.rate
            time.sleep(max(0.0, next_tick - time.monotonic()))
    finally:
        reader.stop()
        for controller in controllers:
            controller.close()


def _trigger(buttons, key):
    value = buttons.get(key, None)
    if value is None:
        # APK versions differ: some publish index trigger, others only grip.
        hand = "right" if str(key).lower().startswith("right") else "left"
        value = buttons.get(hand + "Trig", buttons.get(hand + "Grip", None))
        if value is None:
            # Older APKs expose only the digital index-trigger flag (RTr/LTr).
            digital_key = "RTr" if hand == "right" else "LTr"
            grip_key = "RG" if hand == "right" else "LG"
            value = 1.0 if bool(buttons.get(digital_key, False) or
                                  buttons.get(grip_key, False)) else 0.0
    else:
        # A few APKs always include an analogue field (stuck at zero) and
        # report the physical grip as a separate digital flag.
        hand = "right" if str(key).lower().startswith("right") else "left"
        if bool(buttons.get("RG" if hand == "right" else "LG", False) or
                buttons.get("RTr" if hand == "right" else "LTr", False)):
            value = 1.0
    def _scalar(v):
        if isinstance(v, (tuple, list, np.ndarray)):
            return float(v[0] if len(v) else 0.0)
        return float(v)
    result = _scalar(value)
    # APKs publish rightTrig=0 together with the physical rightGrip value.
    # Use whichever analog channel is active.
    for alt in (hand + "Grip", hand + "Trig"):
        if alt in buttons:
            result = max(result, _scalar(buttons[alt]))
    return max(0.0, min(1.0, result))


def main():
    default_config = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "config", "rm75.yaml"))
    parser = argparse.ArgumentParser(description="ROS-free Quest to RM75 API2 teleop")
    parser.add_argument("--mode", choices=("single", "double"), default="single")
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--ip", default=None, help="single-arm controller IP")
    parser.add_argument("--right-ip", default=None)
    parser.add_argument("--left-ip", default=None)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--quest-ip", default=None,
                        help="Quest ADB IP; omit for USB")
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-collision", action="store_true")
    args = parser.parse_args()
    signal.signal(signal.SIGINT, _stop_handler)
    signal.signal(signal.SIGTERM, _stop_handler)
    run(args)


if __name__ == "__main__":
    main()
