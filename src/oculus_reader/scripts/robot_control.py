#!/usr/bin/env python3
"""Robot control backends used by the Quest teleoperation nodes.

The Quest/IK part of the application works in SI units (joint angles in
radians and a normalized gripper command in ``[0, 1]``).  RealMan's ROS
driver and Python SDK use degrees for arm commands, while the ROS state topic
uses radians.  Keeping that conversion here makes it possible to switch
between the ROS driver, the official SDK and a simulator without changing
the teleoperation code.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Iterable, List, Optional

import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header


RM75_LIMITS_DEG = [-178.0, -130.0, -178.0, -135.0,
                   -178.0, -128.0, -360.0], \
                  [178.0, 130.0, 178.0, 135.0,
                   178.0, 128.0, 360.0]


def _as_float_list(values: Iterable[float], length: int, name: str) -> List[float]:
    values = [float(v) for v in values]
    if len(values) != length:
        raise ValueError("{} must contain {} values, got {}".format(
            name, length, len(values)))
    if not all(math.isfinite(v) for v in values):
        raise ValueError("{} contains a non-finite value".format(name))
    return values


class RobotController:
    """Common command/state API for RM75 and legacy Piper deployments.

    Parameters are read from the node's private namespace.  The supported
    backends are:

    ``rm_ros``
        Publish the official ``rm_msgs/JointPos`` and ``rm_msgs/Gripper_Set``
        messages.  This is the recommended RM75 setup when ``rm_driver`` is
        already running.
    ``rm_sdk``
        Connect directly through the official ``Robotic_Arm`` Python SDK.
    ``joint_state``
        Publish ``sensor_msgs/JointState`` for simulators or custom bridges.
    ``dry_run``
        Validate and log commands without sending them.
    """

    def __init__(self, name: str = "arm", namespace: str = "",
                 command_topic: Optional[str] = None,
                 gripper_topic: Optional[str] = None,
                 state_topic: Optional[str] = None,
                 rm_ip: Optional[str] = None,
                 rm_port: Optional[int] = None):
        self.name = name
        # A dual-arm node creates two controllers in the same ROS private
        # namespace.  Keep the common parameters as defaults, while allowing
        # e.g. ``~left_joint_lower_deg`` to override them for one arm.
        self._param_prefix = "" if name in ("", "arm") else "{}_".format(name)
        self._lock = threading.RLock()
        self.backend = str(self._param("robot_backend", "rm_ros")).lower()
        self.arm_dof = int(self._param("arm_dof", 7))
        if self.arm_dof < 1:
            raise ValueError("~arm_dof must be positive")

        default_names = ["joint{}".format(i + 1) for i in range(self.arm_dof)]
        self.joint_names = list(self._param("joint_names", default_names))
        if len(self.joint_names) != self.arm_dof:
            raise ValueError("~joint_names must have ~arm_dof entries")

        default_lower, default_upper = RM75_LIMITS_DEG
        if self.arm_dof != 7:
            default_lower = [-180.0] * self.arm_dof
            default_upper = [180.0] * self.arm_dof
        self.lower_deg = _as_float_list(
            self._param("joint_lower_deg", default_lower), self.arm_dof,
            "~joint_lower_deg")
        self.upper_deg = _as_float_list(
            self._param("joint_upper_deg", default_upper), self.arm_dof,
            "~joint_upper_deg")
        if any(lo >= hi for lo, hi in zip(self.lower_deg, self.upper_deg)):
            raise ValueError("each lower joint limit must be smaller than upper")

        self.target_joint_state = _as_float_list(
            self._param("target_joint_state", [0.0] * self.arm_dof),
            self.arm_dof, "~target_joint_state")
        self.gripper_max_width = float(self._param("gripper_max_width", 0.07))
        self.gripper_max_width = max(self.gripper_max_width, 1e-6)
        self.gripper_inverted = bool(self._param("gripper_inverted", False))
        self.gripper_enabled = bool(self._param("gripper_enabled", True))
        self.command_rate_hz = float(self._param("init_pose_rate", 50.0))
        self.init_duration = float(self._param("init_pose_duration", 0.5))
        self.max_joint_step_deg = max(0.0, float(self._param(
            "max_joint_step_deg", 2.5)))
        self.command_rate_hz = max(self.command_rate_hz, 1.0)
        self.init_duration = max(self.init_duration, 0.0)
        self._last_state = [0.0] * self.arm_dof
        self._state_received = False
        self._last_gripper = 0.0
        self._last_command = list(self.target_joint_state)
        self._warned_gripper_sdk = False

        self._command_topic_override = command_topic
        self._gripper_topic_override = gripper_topic
        self._state_topic_override = state_topic
        self._rm_ip_override = rm_ip
        self._rm_port_override = rm_port
        self._joint_pub = None
        self._rm_joint_msg = None
        self._rm_gripper_msg = None
        self._sdk = None
        self._sdk_rm_class = None

        if self.backend == "rm_ros":
            self._init_rm_ros(namespace)
        elif self.backend == "rm_sdk":
            self._init_rm_sdk()
        elif self.backend == "joint_state":
            self._init_joint_state(namespace)
        elif self.backend == "dry_run":
            rospy.logwarn("robot_control is in dry_run mode; no arm commands will be sent")
            self._subscribe_state(self._state_topic_override or
                                  self._param("state_topic", "/joint_states"))
        else:
            raise ValueError("unsupported ~robot_backend={!r}".format(self.backend))

    def _init_rm_ros(self, namespace: str) -> None:
        try:
            from rm_msgs.msg import JointPos, Gripper_Set
        except ImportError as exc:
            raise RuntimeError(
                "robot_backend=rm_ros requires the official rm_robot ROS "
                "workspace (rm_msgs/rm_driver). Set robot_backend=joint_state "
                "for a simulator or install rm_robot first.") from exc

        self._rm_joint_msg = JointPos
        self._rm_gripper_msg = Gripper_Set
        default_joint_topic = "/rm_driver/JointPos"
        default_gripper_topic = "/rm_driver/Gripper_Set"
        self._joint_pub = rospy.Publisher(
            self._command_topic_override or self._param("joint_command_topic", default_joint_topic),
            JointPos, queue_size=1)
        if self.gripper_enabled:
            self._gripper_pub = rospy.Publisher(
                self._gripper_topic_override or self._param("gripper_command_topic", default_gripper_topic),
                Gripper_Set, queue_size=1)
        else:
            self._gripper_pub = None
        self._subscribe_state(self._state_topic_override or self._param("state_topic", "/joint_states"))

    def _init_joint_state(self, namespace: str) -> None:
        topic = self._command_topic_override or self._param("joint_command_topic", "/joint_commands")
        self._joint_pub = rospy.Publisher(topic, JointState, queue_size=1)
        self._gripper_pub = None
        self._subscribe_state(self._state_topic_override or self._param("state_topic", "/joint_states"))

    def _init_rm_sdk(self) -> None:
        try:
            from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e
        except ImportError as exc:
            raise RuntimeError(
                "robot_backend=rm_sdk requires the RealMan API2 Python package "
                "(pip install Robotic_Arm or add RM_API2 to PYTHONPATH)") from exc

        mode = getattr(rm_thread_mode_e, "RM_TRIPLE_MODE_E", None)
        self._sdk = RoboticArm(mode) if mode is not None else RoboticArm()
        ip = str(self._rm_ip_override or self._param("rm_ip", "192.168.1.18"))
        port = int(self._rm_port_override or self._param("rm_port", 8080))
        level = int(self._param("rm_log_level", 3))
        try:
            handle = self._sdk.rm_create_robot_arm(ip, port, level)
        except Exception as exc:
            raise RuntimeError(
                "failed to connect to RealMan arm at {}:{}: {}".format(
                    ip, port, exc)) from exc
        if handle is None:
            raise RuntimeError("RealMan SDK returned an empty robot handle")
        rospy.loginfo("Connected to RealMan arm at %s:%d (handle=%s)",
                      ip, port, getattr(handle, "id", handle))
        # Low-follow is the safe default: high-follow requires a stable
        # <=10 ms command period, which a Quest ROS callback normally cannot
        # guarantee.  The YAML can explicitly enable it for a validated setup.
        self._sdk_follow = bool(self._param("rm_high_follow", False))
        self._sdk_trajectory_mode = int(self._param("rm_trajectory_mode", 1))
        self._sdk_radio = int(self._param("rm_radio", 50))
        # SDK state polling is optional; a state topic can still be used when
        # a separate ROS driver publishes feedback.
        self._subscribe_state(self._state_topic_override or self._param("state_topic", ""))

    def _param(self, name: str, default):
        """Read an arm-specific private parameter, then the shared default."""
        specific = "~{}{}".format(self._param_prefix, name)
        if self._param_prefix and rospy.has_param(specific):
            return rospy.get_param(specific)
        return rospy.get_param("~{}".format(name), default)

    def _subscribe_state(self, topic: str) -> None:
        if topic:
            rospy.Subscriber(topic, JointState, self._state_callback, queue_size=1)

    def _state_callback(self, msg: JointState) -> None:
        with self._lock:
            if msg.name:
                by_name = dict(zip(msg.name, msg.position))
                values = [by_name.get(name, None) for name in self.joint_names]
                if any(v is None for v in values):
                    values = list(msg.position[:self.arm_dof])
            else:
                values = list(msg.position[:self.arm_dof])
            if len(values) == self.arm_dof and all(math.isfinite(float(v)) for v in values):
                self._last_state = [float(v) for v in values]
                self._state_received = True

    def _clamp_rad(self, joints: Iterable[float]) -> List[float]:
        joints = _as_float_list(joints, self.arm_dof, "joint positions")
        return [math.radians(max(lo, min(hi, math.degrees(q))))
                for q, lo, hi in zip(joints, self.lower_deg, self.upper_deg)]

    def _send_arm(self, joints_rad: Iterable[float], publish: bool = True) -> None:
        joints_rad = self._clamp_rad(joints_rad)
        if self.max_joint_step_deg > 0.0 and self._last_command is not None:
            max_step = math.radians(self.max_joint_step_deg)
            joints_rad = [last + max(-max_step, min(max_step, target - last))
                          for last, target in zip(self._last_command, joints_rad)]
        self._last_command = list(joints_rad)
        joints_deg = [math.degrees(v) for v in joints_rad]
        if not publish:
            return
        if self.backend == "dry_run":
            rospy.logdebug("dry-run joint command (deg): %s", joints_deg)
            return
        if self.backend == "rm_ros":
            msg = self._rm_joint_msg()
            msg.joint = joints_deg
            msg.expand = float(self._param("rm_expand", 0.0))
            self._joint_pub.publish(msg)
        elif self.backend == "joint_state":
            msg = JointState(header=Header(stamp=rospy.Time.now()))
            msg.name = list(self.joint_names)
            msg.position = joints_rad
            self._joint_pub.publish(msg)
        elif self.backend == "rm_sdk":
            try:
                if hasattr(self._sdk, "rm_movej_canfd"):
                    ret = self._sdk.rm_movej_canfd(
                        joints_deg, self._sdk_follow, 0,
                        self._sdk_trajectory_mode, self._sdk_radio)
                elif hasattr(self._sdk, "rm_movej_follow"):
                    ret = self._sdk.rm_movej_follow(joints_deg)
                else:
                    raise AttributeError(
                        "installed RealMan SDK has no joint follow method")
                if isinstance(ret, int) and ret != 0:
                    rospy.logwarn_throttle(
                        1.0, "RealMan joint command returned %d", ret)
            except Exception as exc:
                # A transient network/API error must not bring down the Quest
                # subscriber callback.  The next frame can recover naturally.
                rospy.logwarn_throttle(1.0,
                                       "RealMan joint command failed: %s", exc)

    def _send_gripper(self, width_m: float) -> None:
        if not self.gripper_enabled:
            return
        normalized = max(0.0, min(1.0, float(width_m) / self.gripper_max_width))
        if self.gripper_inverted:
            normalized = 1.0 - normalized
        self._last_gripper = normalized
        position = max(1, min(1000, int(round(normalized * 1000.0))))
        if self.backend == "dry_run":
            rospy.logdebug("dry-run gripper command: %d/1000", position)
        elif self.backend == "rm_ros":
            msg = self._rm_gripper_msg()
            msg.position = position
            self._gripper_pub.publish(msg)
        elif self.backend == "joint_state":
            # A custom simulator may expose the gripper as one extra joint.
            if bool(self._param("joint_state_has_gripper", False)):
                msg = JointState(header=Header(stamp=rospy.Time.now()))
                msg.name = list(self.joint_names) + [str(self._param(
                    "gripper_joint_name", "gripper_joint"))]
                msg.position = list(self._last_command) + [width_m]
                self._joint_pub.publish(msg)
        elif self.backend == "rm_sdk":
            # API2 versions use slightly different names for this optional
            # accessory.  Prefer the fixed-position command and fail loudly
            # only once if the installed SDK does not provide it.
            candidates = ("rm_set_gripper_position", "rm_set_gripper")
            for method_name in candidates:
                method = getattr(self._sdk, method_name, None)
                if method is None:
                    continue
                try:
                    try:
                        result = method(position, False, 1)
                    except TypeError:
                        result = method(position)
                except Exception as exc:
                    rospy.logwarn_throttle(
                        1.0, "RealMan gripper command failed: %s", exc)
                    return
                if isinstance(result, int) and result != 0:
                    rospy.logwarn_throttle(
                        1.0, "RealMan gripper command returned %d", result)
                return
            if not self._warned_gripper_sdk:
                rospy.logwarn("RealMan SDK has no fixed-position gripper method; "
                              "use rm_ros or configure the accessory separately")
                self._warned_gripper_sdk = True

    def command(self, joints_rad: Iterable[float], gripper_width_m: Optional[float] = None) -> None:
        """Send one validated arm command and, optionally, a gripper command."""
        with self._lock:
            combined_joint_state = (
                self.backend == "joint_state" and gripper_width_m is not None and
                bool(self._param("joint_state_has_gripper", False)))
            self._send_arm(joints_rad, publish=not combined_joint_state)
            if gripper_width_m is not None:
                self._send_gripper(gripper_width_m)

    def current_joint_positions(self) -> List[float]:
        """Return the latest state in radians, or query the SDK if available."""
        with self._lock:
            if self.backend == "rm_sdk" and hasattr(self._sdk, "rm_get_joint_degree"):
                try:
                    ret = self._sdk.rm_get_joint_degree()
                    # API2 releases have returned both ``(code, values)`` and
                    # the values directly.  Accept either form so state
                    # feedback remains useful across SDK upgrades.
                    if isinstance(ret, tuple) and len(ret) >= 2:
                        if isinstance(ret[0], int) and ret[0] != 0:
                            values = []
                        else:
                            values = list(ret[1])
                    else:
                        values = list(ret)
                    values = values[:self.arm_dof]
                    if len(values) == self.arm_dof:
                        self._last_state = [math.radians(float(v)) for v in values]
                        self._state_received = True
                except Exception as exc:  # SDK errors should not stop teleop
                    rospy.logdebug("RealMan joint state query failed: %s", exc)
            return list(self._last_state)

    def init_pose(self) -> None:
        """Move smoothly to the configured zero/initial joint pose."""
        # Apply the same model limits as runtime commands.  A stale YAML file
        # must not make the initialization interpolation cross a hard stop.
        target = self._clamp_rad(self.target_joint_state)
        wait_timeout = max(0.0, float(self._param("state_wait_timeout", 1.0)))
        deadline = time.monotonic() + wait_timeout
        while (not self._state_received and self.backend not in ("rm_sdk", "dry_run")
               and not rospy.is_shutdown() and time.monotonic() < deadline):
            rospy.sleep(0.02)
        current = self.current_joint_positions()
        if not self._state_received:
            current = list(target)
        self._last_command = list(current)
        steps = max(1, int(round(self.init_duration * self.command_rate_hz)))
        if self.max_joint_step_deg > 0.0:
            largest_delta_deg = max(abs(math.degrees(t - c))
                                    for c, t in zip(current, target))
            steps = max(steps, int(math.ceil(
                largest_delta_deg / self.max_joint_step_deg)))
        rate = rospy.Rate(self.command_rate_hz)
        combined_joint_state = (
            self.backend == "joint_state" and self.gripper_enabled and
            bool(self._param("joint_state_has_gripper", False)))
        initial_gripper_width = float(self._param("initial_gripper_width", 0.0))
        for step in range(steps + 1):
            if rospy.is_shutdown():
                return
            alpha = float(step) / float(steps)
            joints = [(1.0 - alpha) * c + alpha * t for c, t in zip(current, target)]
            self._send_arm(joints, publish=not combined_joint_state)
            if combined_joint_state:
                self._send_gripper(initial_gripper_width)
            rate.sleep()
        if self.gripper_enabled and not combined_joint_state:
            self._send_gripper(initial_gripper_width)

    # Backward-compatible names used by the original Piper teleop scripts.
    def joint_control(self, joints_rad: Iterable[float], gripper_width_m: float = 0.0) -> None:
        self.command(joints_rad, gripper_width_m)

    def joint_control_piper(self, *values: float) -> None:
        if len(values) == self.arm_dof + 1:
            self.command(values[:-1], values[-1])
        elif len(values) == self.arm_dof:
            self.command(values)
        else:
            raise ValueError("expected {} joints plus gripper".format(self.arm_dof))

    def left_joint_control_piper(self, *values: float) -> None:
        self.joint_control_piper(*values)

    def right_joint_control_piper(self, *values: float) -> None:
        self.joint_control_piper(*values)

    def left_init_pose(self) -> None:
        self.init_pose()

    def right_init_pose(self) -> None:
        self.init_pose()

    def shutdown(self) -> None:
        if self.backend == "rm_sdk" and self._sdk is not None:
            try:
                self._sdk.rm_delete_robot_arm()
            except Exception:
                pass


# Keep imports in downstream scripts working while they migrate from Piper.
PIPER = RobotController
