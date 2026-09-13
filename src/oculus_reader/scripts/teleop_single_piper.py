#!/usr/bin/env python3
import rospy
import tf2_ros
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from geometry_msgs.msg import PoseStamped

import pinocchio as pin
from oculus_reader import OculusReader

import numpy as np

from tools import MATHTOOLS
from robot_control import RobotController
from robot_ik import Arm_IK, calc_pose_incre

class VR:
    def __init__(self):
        self.robot_control = RobotController()
        rospy.on_shutdown(self.robot_control.shutdown)
        self.tools = MATHTOOLS()
        self.inverse_solution = Arm_IK()
        self.robot_control.init_pose()
        
        # 这里可选为 WIFI连接 或 USB连接
        # oculus_reader = OculusReader(ip_address='10.12.11.14')    #  WIFI连接
        self.oculus_reader = OculusReader()                         #  USB连接
        
        # 延时0.5秒，确保 OculusReader 初始化完成   
        import time
        time.sleep(0.5)

        self.base_RR = [0.19, 0.0, 0.2, 0, 0, 0]
        self.zero_pose = rospy.get_param('~teleop_zero_pose', self.base_RR)
        # 订阅回调
        rospy.Subscriber('/right_handle_pose', PoseStamped, self.handle_pose_callback, queue_size=1)
        
    def get_ik_solution(self, x,y,z,roll,pitch,yaw,gripper,b):
        
        q = quaternion_from_euler(roll, pitch, yaw)
        target = pin.SE3(
            pin.Quaternion(q[3], q[0], q[1], q[2]),
            np.array([x, y, z]),
        )
        sol_q, tau_ff, is_collision = self.inverse_solution.ik_fun(target.homogeneous,0)
        # print("result:", sol_q)
        
        if sol_q is None:
            return
        if is_collision and rospy.get_param('~block_on_collision', True):
            rospy.logwarn_throttle(1.0, "IK target rejected because of self-collision")
            return
        if b:
            self.robot_control.joint_control(sol_q, gripper)
        if is_collision :
            print("\33[31m-------------------       Robotic arm self-collision!!!       -----------------------------\033[0m") 

    def handle_pose_callback(self, msg):
        # print(msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)
        self.x = msg.pose.position.x
        self.y = msg.pose.position.y
        self.z = msg.pose.position.z
        (self.roll, self.pitch, self.yaw) = euler_from_quaternion([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
        
        _, buttons = self.oculus_reader.get_transformations_and_buttons()

        RR = [self.x,self.y,self.z,self.roll,self.pitch,self.yaw]
            
        if buttons.get('A', False):
            # 按下A键后，机械臂回到初始点位并且记录 右 坐标原点
            self.robot_control.init_pose()
            self.base_RR = [self.x,self.y,self.z,self.roll,self.pitch,self.yaw]
                    
        RR_ = calc_pose_incre(self.base_RR, RR, self.zero_pose)
        
        trigger = buttons.get('rightTrig', [0.0])
        r_gripper_value = max(0.0, min(1.0, float(trigger[0]))) * self.robot_control.gripper_max_width
        # 按下B键后，开始遥操作
        self.get_ik_solution(RR_[0], RR_[1], RR_[2], RR_[3], RR_[4], RR_[5],
                             r_gripper_value, buttons.get('B', False))
            

if __name__ == '__main__':
    rospy.init_node('teleop_single_piper_node')
    vr = VR()
    rospy.spin()
    
