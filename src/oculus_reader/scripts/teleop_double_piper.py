#!/usr/bin/env python3
"""Meta Quest dual-arm teleoperation.

Both arms use the same configurable IK and controller backends as the single
arm node.  For RM ROS deployments the two command topics can be namespaced by
``~left_joint_command_topic`` and ``~right_joint_command_topic``.
"""

import time

import numpy as np
import pinocchio as pin
import rospy
from geometry_msgs.msg import PoseStamped
from tf.transformations import euler_from_quaternion, quaternion_from_euler

from oculus_reader import OculusReader
from robot_control import RobotController
from robot_ik import Arm_IK, calc_pose_incre


class VR:
    def __init__(self):
        self.right_control = RobotController(
            name="right",
            command_topic=rospy.get_param("~right_joint_command_topic", None),
            gripper_topic=rospy.get_param("~right_gripper_command_topic", None),
            state_topic=rospy.get_param("~right_state_topic", None),
            rm_ip=rospy.get_param("~right_rm_ip", None),
            rm_port=rospy.get_param("~rm_port", None))
        self.left_control = RobotController(
            name="left",
            command_topic=rospy.get_param("~left_joint_command_topic", None),
            gripper_topic=rospy.get_param("~left_gripper_command_topic", None),
            state_topic=rospy.get_param("~left_state_topic", None),
            rm_ip=rospy.get_param("~left_rm_ip", None),
            rm_port=rospy.get_param("~rm_port", None))
        rospy.on_shutdown(self.right_control.shutdown)
        rospy.on_shutdown(self.left_control.shutdown)
        # IK parameters can be shared or overridden with ~right_* / ~left_*
        # (for example, different tool offsets or official URDF variants).
        self.right_ik = Arm_IK("right")
        self.left_ik = Arm_IK("left")
        self.right_control.init_pose()
        self.left_control.init_pose()
        self.oculus_reader = OculusReader()
        time.sleep(0.5)

        default_base = [0.19, 0.0, 0.2, 0, 0, 0]
        self.base_right = list(rospy.get_param("~right_base_pose", default_base))
        self.base_left = list(rospy.get_param("~left_base_pose", default_base))
        self.zero_pose = rospy.get_param("~teleop_zero_pose", default_base)
        self.right_zero_pose = rospy.get_param(
            "~right_teleop_zero_pose", self.zero_pose)
        self.left_zero_pose = rospy.get_param(
            "~left_teleop_zero_pose", self.zero_pose)
        self.block_on_collision = bool(rospy.get_param(
            "~block_on_collision", True))
        self.right_block_on_collision = bool(rospy.get_param(
            "~right_block_on_collision", self.block_on_collision))
        self.left_block_on_collision = bool(rospy.get_param(
            "~left_block_on_collision", self.block_on_collision))
        rospy.Subscriber("/right_handle_pose", PoseStamped,
                         self.right_handle_pose_callback, queue_size=1)
        rospy.Subscriber("/left_handle_pose", PoseStamped,
                         self.left_handle_pose_callback, queue_size=1)

    @staticmethod
    def _target(x, y, z, roll, pitch, yaw):
        q = quaternion_from_euler(roll, pitch, yaw)
        return pin.SE3(pin.Quaternion(q[3], q[0], q[1], q[2]),
                       np.array([x, y, z], dtype=float))

    def _get_ik_solution(self, controller, solver, pose, gripper, enabled,
                         block_on_collision):
        sol_q, _, is_collision = solver.ik_fun(pose.homogeneous, 0)
        if sol_q is None:
            return
        if is_collision and block_on_collision:
            rospy.logwarn_throttle(1.0, "IK target rejected because of self-collision")
            return
        if enabled:
            controller.joint_control(sol_q, gripper)
        if is_collision:
            rospy.logwarn_throttle(
                1.0, "Robotic arm self-collision detected; command was not blocked")

    def right_handle_pose_callback(self, msg):
        x, y, z = msg.pose.position.x, msg.pose.position.y, msg.pose.position.z
        roll, pitch, yaw = euler_from_quaternion([
            msg.pose.orientation.x, msg.pose.orientation.y,
            msg.pose.orientation.z, msg.pose.orientation.w])
        _, buttons = self.oculus_reader.get_transformations_and_buttons()
        pose = [x, y, z, roll, pitch, yaw]
        if buttons.get("A", False):
            self.right_control.init_pose()
            self.base_right = pose
        target = calc_pose_incre(self.base_right, pose, self.right_zero_pose)
        width = buttons.get("rightTrig", [0.0])[0] * self.right_control.gripper_max_width
        self._get_ik_solution(self.right_control, self.right_ik,
                              self._target(*target), width,
                              buttons.get("B", False),
                              self.right_block_on_collision)

    def left_handle_pose_callback(self, msg):
        x, y, z = msg.pose.position.x, msg.pose.position.y, msg.pose.position.z
        roll, pitch, yaw = euler_from_quaternion([
            msg.pose.orientation.x, msg.pose.orientation.y,
            msg.pose.orientation.z, msg.pose.orientation.w])
        _, buttons = self.oculus_reader.get_transformations_and_buttons()
        pose = [x, y, z, roll, pitch, yaw]
        if buttons.get("X", False):
            self.left_control.init_pose()
            self.base_left = pose
        target = calc_pose_incre(self.base_left, pose, self.left_zero_pose)
        width = buttons.get("leftTrig", [0.0])[0] * self.left_control.gripper_max_width
        self._get_ik_solution(self.left_control, self.left_ik,
                              self._target(*target), width,
                              buttons.get("Y", False),
                              self.left_block_on_collision)


if __name__ == "__main__":
    rospy.init_node("teleop_double_piper_node")
    VR()
    rospy.spin()
