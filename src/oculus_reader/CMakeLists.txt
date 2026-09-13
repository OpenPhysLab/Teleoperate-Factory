cmake_minimum_required(VERSION 3.0.2)
project(oculus_reader)

find_package(catkin REQUIRED COMPONENTS
  rospy
  geometry_msgs
  sensor_msgs
  std_msgs
  tf
  tf2_ros
)

catkin_package(
  CATKIN_DEPENDS geometry_msgs sensor_msgs std_msgs tf tf2_ros
)

install(DIRECTORY config launch urdf
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

catkin_install_python(PROGRAMS
  scripts/teleop_single_piper.py
  scripts/teleop_double_piper.py
  scripts/robot_control.py
  scripts/robot_ik.py
  scripts/rm_api2_control.py
  scripts/rm_api2_teleop.py
  scripts/synchronized_cameras.py
  scripts/rm75_teleop_dataset.py
  scripts/piper_control.py
  scripts/oculus_reader.py
  scripts/pub_pose.py
  scripts/tools.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
