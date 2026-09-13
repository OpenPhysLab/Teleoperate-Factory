#!/usr/bin/env python3
"""ROS-free RealMan API2 controller.

The teleoperation loop uses radians and metres internally.  API2 expects arm
commands in degrees and gripper positions in the integer range 1..1000; this
module is the only place where that protocol conversion is performed.
"""

from __future__ import annotations

import math
import time


RM75_LOWER_DEG = [-178.0, -130.0, -178.0, -135.0, -178.0, -128.0, -360.0]
RM75_UPPER_DEG = [178.0, 130.0, 178.0, 135.0, 178.0, 128.0, 360.0]


def _numbers(values, size, name):
    values = [float(x) for x in values]
    if len(values) != size or not all(math.isfinite(x) for x in values):
        raise ValueError("{} must contain {} finite values".format(name, size))
    return values


class RealManAPI2Controller:
    """One RM75 connection, usable from a normal Python process."""

    def __init__(self, ip="192.168.1.18", port=8080, arm_dof=7,
                 lower_deg=None, upper_deg=None, target_joint_state=None,
                 gripper_max_width=0.07, gripper_inverted=False,
                 max_joint_step_deg=2.5, init_duration=1.5,
                 init_rate=50.0, high_follow=False, trajectory_mode=1,
                 radio=50, log_level=3, dry_run=False):
        self.arm_dof = int(arm_dof)
        if self.arm_dof != 7:
            raise ValueError("RM75 API2 mode requires arm_dof=7")
        self.lower_deg = _numbers(lower_deg or RM75_LOWER_DEG, self.arm_dof,
                                  "lower_deg")
        self.upper_deg = _numbers(upper_deg or RM75_UPPER_DEG, self.arm_dof,
                                  "upper_deg")
        if any(a >= b for a, b in zip(self.lower_deg, self.upper_deg)):
            raise ValueError("every lower limit must be below its upper limit")
        self.target = _numbers(target_joint_state or [0.0] * self.arm_dof,
                               self.arm_dof, "target_joint_state")
        self.gripper_max_width = max(float(gripper_max_width), 1e-6)
        self.gripper_inverted = bool(gripper_inverted)
        self.max_joint_step_deg = max(0.0, float(max_joint_step_deg))
        self.init_duration = max(0.0, float(init_duration))
        self.init_rate = max(1.0, float(init_rate))
        self.high_follow = bool(high_follow)
        self.trajectory_mode = int(trajectory_mode)
        self.radio = int(radio)
        self.dry_run = bool(dry_run)
        self.label = str(ip)
        # Until the first state read, there is no safe command baseline.  Do
        # not assume all-zero joints: a real arm may be parked elsewhere.
        self.last_command = None
        self.last_state = [0.0] * self.arm_dof
        self._sdk = None
        self._warned_gripper = False
        if not self.dry_run:
            try:
                from Robotic_Arm.rm_robot_interface import RoboticArm
                try:
                    from Robotic_Arm.rm_robot_interface import rm_thread_mode_e
                except ImportError:
                    rm_thread_mode_e = None
            except ImportError as exc:
                raise RuntimeError(
                    "API2 Python SDK is required; install Robotic_Arm or set --dry-run") from exc
            mode = getattr(rm_thread_mode_e, "RM_TRIPLE_MODE_E", None)
            self._sdk = RoboticArm(mode) if mode is not None else RoboticArm()
            try:
                handle = self._sdk.rm_create_robot_arm(str(ip), int(port), int(log_level))
            except Exception as exc:
                raise RuntimeError(
                    "failed to connect to RM75 at {}:{}: {}".format(ip, port, exc)) from exc
            if handle is None or getattr(handle, "id", 0) <= 0:
                raise RuntimeError("API2 returned an empty robot handle")
            print("Connected to RM75 at {}:{}".format(ip, port))

    def _clamp(self, joints):
        joints = _numbers(joints, self.arm_dof, "joint positions")
        return [math.radians(max(lo, min(hi, math.degrees(q))))
                for q, lo, hi in zip(joints, self.lower_deg, self.upper_deg)]

    def _limited(self, joints):
        joints = self._clamp(joints)
        if self.last_command is None:
            self.current_joint_positions()
            self.last_command = list(self.last_state)
        if self.max_joint_step_deg <= 0.0:
            return joints
        step = math.radians(self.max_joint_step_deg)
        return [old + max(-step, min(step, new - old))
                for old, new in zip(self.last_command, joints)]

    def command_joints(self, joints_rad):
        joints_rad = self._limited(joints_rad)
        degrees = [math.degrees(x) for x in joints_rad]
        if self.dry_run:
            self.last_command = list(joints_rad)
            self.last_state = list(joints_rad)
            print("[dry-run {}] joints(deg): {}".format(self.label,
                " ".join("{:.2f}".format(x) for x in degrees)))
            return 0
        try:
            if hasattr(self._sdk, "rm_movej_canfd"):
                result = self._sdk.rm_movej_canfd(
                    degrees, self.high_follow, 0,
                    self.trajectory_mode, self.radio)
            else:
                result = self._sdk.rm_movej_follow(degrees)
            if isinstance(result, int) and result != 0:
                print("WARNING: API2 joint command returned {}".format(result))
            if result == 0:
                self.last_command = list(joints_rad)
            return result
        except Exception as exc:
            print("WARNING: API2 joint command failed: {}".format(exc))
            return -1

    def command_pose(self, pose_xyzrpy, follow=False):
        """Send TCP pose to the RM controller's documented internal IK."""
        pose = _numbers(pose_xyzrpy, 6, "TCP pose xyzrpy")
        if self.dry_run:
            print("[dry-run {}] TCP pose: {}".format(self.label, pose))
            return 0
        method = getattr(self._sdk, "rm_movep_canfd", None)
        if method is None:
            raise RuntimeError("API2 SDK lacks rm_movep_canfd")
        return method(pose, bool(follow), self.trajectory_mode, self.radio)

    def command_gripper(self, width_m):
        # Keep the last commanded physical width available to data logging.
        self.last_gripper_width = float(width_m)
        normalized = max(0.0, min(1.0, float(width_m) / self.gripper_max_width))
        if self.gripper_inverted:
            normalized = 1.0 - normalized
        position = max(1, min(1000, int(round(normalized * 1000.0))))
        if self.dry_run:
            print("[dry-run] gripper: {}/1000".format(position))
            return 0
        for method_name in ("rm_set_gripper_position", "rm_set_gripper"):
            method = getattr(self._sdk, method_name, None)
            if method is None:
                continue
            try:
                result = None
                last_type_error = None
                for call_args in ((position, False, 1), (position, False),
                                  (position,)):
                    try:
                        result = method(*call_args)
                        last_type_error = None
                        break
                    except TypeError as exc:
                        last_type_error = exc
                if last_type_error is not None:
                    raise last_type_error
                if isinstance(result, int) and result != 0:
                    print("WARNING: API2 gripper command returned {}".format(result))
                return result
            except Exception as exc:
                print("WARNING: API2 gripper command failed: {}".format(exc))
                return -1
        if not self._warned_gripper:
            print("WARNING: this API2 SDK has no fixed-position gripper method")
            self._warned_gripper = True
        return -1

    def current_joint_positions(self, strict=False):
        if self.dry_run or self._sdk is None:
            return list(self.last_state)
        method = getattr(self._sdk, "rm_get_joint_degree", None)
        if method is None:
            if strict:
                raise RuntimeError("API2 joint feedback is unavailable")
            return list(self.last_state)
        try:
            result = method()
            if isinstance(result, tuple) and len(result) >= 2:
                if isinstance(result[0], int) and result[0] != 0:
                    raise RuntimeError("API2 joint feedback status {}".format(result[0]))
                values = list(result[1])
            else:
                values = list(result)
            self.last_state = [math.radians(x) for x in
                               _numbers(values[:self.arm_dof], self.arm_dof, "feedback")]
        except Exception as exc:
            if strict:
                raise RuntimeError("cannot read fresh RM75 joints") from exc
            print("WARNING: cannot read RM75 state: {}".format(exc))
        return list(self.last_state)

    def current_tcp_pose(self, strict=False):
        """Controller-reported flange/tool pose [m,m,m,rad,rad,rad]."""
        if self.dry_run or self._sdk is None:
            return None
        try:
            status, state = self._sdk.rm_get_current_arm_state()
            if status != 0:
                raise RuntimeError("API2 arm state status {}".format(status))
            return _numbers(state.get("pose", []), 6, "API2 TCP pose")
        except Exception as exc:
            if strict:
                raise RuntimeError("cannot read API2 TCP pose: {}".format(exc)) from exc
            return None

    def init_pose(self, initial_gripper_width=0.0):
        if self.dry_run:
            # No real feedback in dry-run: simulate arrival at the saved home,
            # not repeated travel from an invented all-zero posture.
            self.last_state = self._clamp(self.target)
            self.last_command = list(self.last_state)
            print("[dry-run {}] HOME joints(deg): {}".format(
                self.label, [round(math.degrees(x), 3) for x in self.last_state]))
            self.command_gripper(initial_gripper_width)
            return
        current = self.current_joint_positions(strict=True)
        self.last_command = list(current)
        target = self._clamp(self.target)
        steps = max(1, int(round(self.init_duration * self.init_rate)))
        if self.max_joint_step_deg > 0.0:
            largest = max(abs(math.degrees(a - b)) for a, b in zip(current, target))
            steps = max(steps, int(math.ceil(largest / self.max_joint_step_deg)))
        for index in range(steps + 1):
            alpha = float(index) / float(steps)
            joints = [(1.0 - alpha) * a + alpha * b
                      for a, b in zip(current, target)]
            self.command_joints(joints)
            if index == steps or self.dry_run:
                self.command_gripper(initial_gripper_width)
            if not self.dry_run:
                time.sleep(1.0 / self.init_rate)

    def close(self):
        if self._sdk is not None:
            try:
                self._sdk.rm_delete_robot_arm()
            except Exception:
                pass
