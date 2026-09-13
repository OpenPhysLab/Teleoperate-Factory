# RM75 迁移说明

当前仓库的 Quest 位姿读取、IK 和机械臂通信已经拆开。RM75 的推荐入口是
纯 Python API2：不启动 ROS、不发布 ROS 话题，直接通过 `OculusReader` 读取
Quest，并调用睿尔曼官方 `Robotic_Arm` SDK。ROS `rm_driver` 后端仍保留，
用于已有 ROS 系统的兼容场景。

## 纯 Python API2（推荐）

安装 Python 依赖：

```bash
./scripts/setup_rm75_api2.sh
source ./scripts/activate_rm75_api2.sh
```

只做 API2 连接和状态读取（不会发送运动命令）：

```bash
python3 scripts/test_rm75_api2_connection.py --ip 192.168.1.19
```

单臂 USB Quest：

```bash
source ./scripts/activate_rm75_api2.sh
python3 src/oculus_reader/scripts/rm_api2_teleop.py \
  --mode single --ip 192.168.1.18
```

双臂：

```bash
source ./scripts/activate_rm75_api2.sh
python3 src/oculus_reader/scripts/rm_api2_teleop.py \
  --mode double --left-ip 192.168.1.18 --right-ip 192.168.1.19
```

Quest 通过 Wi-Fi ADB 时增加 `--quest-ip <Quest_IP>`；USB 连接则省略该参数。
首次运行建议先不连接机械臂，验证 Quest 和 IK：

```bash
source ./scripts/activate_rm75_api2.sh
python3 src/oculus_reader/scripts/rm_api2_teleop.py \
  --mode single --dry-run
```

API2 模式默认关闭高跟随（`rm_high_follow: false`），并限制每帧最大关节
变化为 2.5°。确认控制器和通信周期稳定后，才建议在配置中显式开启高跟随。

## 三台 RealSense 相机

当前设备映射记录在 [`config/cameras.yaml`](src/oculus_reader/config/cameras.yaml)：

- D435：`/dev/video2`–`/dev/video7`
- D405 `235123074600`：`/dev/video8`–`/dev/video13`
- D405 `235123071832`：`/dev/video14`–`/dev/video19`

非 ROS 健康检查命令如下。RealSense 一个物理设备会暴露多个 V4L2 节点，
因此测试脚本按“每个物理相机至少一个节点能出帧”判断，不要求每个节点都能被
OpenCV 独占打开：

```bash
source ./scripts/activate_rm75_api2.sh
python3 ./scripts/test_cameras.py --preferred-only --frames 1
```

### 三相机同步采集（无 ROS）

同步采集器默认通过 `pyrealsense2` 打开三台相机，使用 RealSense 的
`global_time` 硬件时间戳，再在时间窗口内配成一组。当前设备通过 USB 物理
端口自动映射到 `/dev/video4`、`/dev/video10` 和 `/dev/video16`，不依赖
SDK 序列号与 UVC 序列号是否一致。采集目标为 25 Hz，默认同步窗口为 18 ms：

```bash
source ./scripts/activate_rm75_api2.sh
python3 src/oculus_reader/scripts/synchronized_cameras.py \
  --frames 300 --save-dir recordings/session01
```

输出目录包含每组的三张 JPEG 和 `manifest.jsonl`。清单记录 `frame_id`、
每路原始时间戳、帧序号和 `sync_span_ms`，便于后处理时检查同步质量。也可
用 `--duration 60` 按时间采集，或加 `--display` 打开预览；按 `q`/`Esc`
退出。`--sync-window-ms` 可覆盖配置文件中的窗口，窗口越小同步要求越严，
但在 USB 带宽繁忙时可能降低成组帧率。

当前设备在 25 Hz 配置下按时间戳降采样输出，组间隔约 40 ms；三路硬件时间
跨度实测约 10–18 ms。可以用 `--sync-window-ms 15` 进一步收紧，但会降低
成组帧率；必须以实际 `matched synchronized groups` 为准。`--fps 30` 可用于
测试更高采集率，但不保证三路的时间跨度更小。

### Meta Quest 3 遥操作同步数采

`rm75_teleop_dataset.py` 复用现有 `OculusReader`（Quest 2/3 使用同一套
APK 数据协议），并把 Quest 位姿、按键、三路图像和 RM75 反馈写入同一条
`manifest.jsonl`。建议先只测 Quest 与相机，不连接/不运动机械臂：

```bash
source ./scripts/activate_rm75_api2.sh
python3 src/oculus_reader/scripts/rm75_teleop_dataset.py \
  --no-robot --duration 30 --save-dir recordings/quest3_test
```

Quest 通过 USB 时省略 `--quest-ip`；网络 ADB 时增加 `--quest-ip <Quest_IP>`。
当前已验证的 Quest 3S 无线地址为 `172.16.204.100`。以后重新连接可执行：

```bash
./scripts/connect_quest_wifi.sh 172.16.204.100
```

无线 ADB 建立后可以拔掉 USB 线；Quest 重启后通常需要重新通过 USB 执行一次
`tcpip 5555`，再运行该脚本。
确认 Quest 位姿和相机数据正常后，再进行双 RM75 数采（开始前清空机械臂
周围区域，并准备急停）：

```bash
source ./scripts/activate_rm75_api2.sh
python3 src/oculus_reader/scripts/rm75_teleop_dataset.py \
  --mode double --left-ip 192.168.1.18 --right-ip 192.168.1.19 \
  --duration 60 --save-dir recordings/session01
```

数采生命周期默认完全由 Quest 按键控制，三个相机会一直显示预览窗口：

```bash
python3 src/oculus_reader/scripts/rm75_teleop_dataset.py \
  --mode double --left-ip 192.168.1.18 --right-ip 192.168.1.19 \
  --quest-ip 172.16.204.100 \
  --save-dir recordings/session01
```

VR 按键为：右摇杆（如果 APK 上报 `RJ`）或右手 `A` 开始采集；左摇杆
（`LJ`）或左手 `X` 结束并自动保存；左右摇杆同时按下，或 `A+X` 同时
按下，则删除当前会话并结束、不保存；右手 `B`/左手 `Y` 按住才允许对应
机械臂跟随；两个扳机控制对应夹爪。当前 Quest 3 APK 实测没有上报 `RJ/LJ`，
因此建议使用 `A/X` 生命周期映射。
程序启动和结束时都会恢复 `rm75_initial_pose.yaml` 中的初始关节姿态并打开
夹爪。每个同步帧组持续写入一条 JSONL，不需要重复启动来采单条。

如果没有图形桌面，可加 `--no-display` 关闭视频窗口。旧版终端交互仍可用：
加 `--interactive` 后，`s/p/d/q` 分别表示开始、暂停、删除、结束。

如需先验证 IK/按钮映射但不让机械臂运动，使用 `--dry-run`；`--no-robot`
则完全不连接 API2。数采记录中 `robot` 字段包含两臂最新关节反馈，
`robot`/`quest` 各自带有主机单调时钟字段，`quest_transforms` 和
`quest_buttons` 保存 Quest 原始数据。

### 双臂统一初始姿态

当前已分别从 18、19 号机械臂读取关节状态并保存到
[`rm75_initial_poses.yaml`](src/oculus_reader/config/rm75_initial_poses.yaml)，
两臂后续初始化会分别恢复各自姿态，夹爪目标开口为 `0.07 m`。
若以后需要重新记录某一侧：

```bash
source ./scripts/activate_rm75_api2.sh
python3 scripts/record_rm75_initial_pose.py --ip 192.168.1.19
```

只查看目标、不运动：

```bash
python3 scripts/apply_rm75_initial_pose.py --ip 192.168.1.18
```

确认现场安全后执行对齐：

```bash
python3 scripts/apply_rm75_initial_pose.py --ip 192.168.1.18 --apply --duration 3
```

默认清单同时保存 `timestamps_ns`（RealSense 时间戳）、
`host_timestamps_ns`（主机到达时间）和 `timestamp_domains`。当前三台设备
返回的域为 `global_time`。这解决了跨 USB 设备主机调度延迟的问题，但仍不是
外部硬件触发：三台相机的曝光时刻仍可能相差数毫秒到十几毫秒。若要进一步
降低到亚毫秒级，需要支持硬件同步的流配置、同步线/外部触发，以及按序列号
建立稳定设备名。没有 `pyrealsense2` 时可用 `--backend v4l2` 回退到主机
到达时间同步。

## ROS 兼容模式

如果现场已有官方 ROS 驱动，可以继续使用下列入口。单臂通过 `rm_driver`
发布 `rm_msgs/JointPos`（角度，单位为度）和 `rm_msgs/Gripper_Set`（夹爪
位置，1–1000），并从 `/joint_states` 读取弧度反馈。

### 安装官方 RM ROS 包

RM75 的 ROS1 驱动支持 Ubuntu 18.04/20.04、ROS Melodic/Noetic。把官方
`rm_robot` 放到同一个 catkin 工作空间后编译：

```bash
cd ~/questVR_ws/src
git clone https://github.com/RealManRobot/rm_robot.git
cd ..
catkin_make
source devel/setup.bash
```

若使用 SDK 后端，安装瑞尔曼 API2 Python 包（或将 `RM_API2` 加入
`PYTHONPATH`）：

```bash
pip install Robotic_Arm
```

## 单臂 RM75

确认机械臂控制器 IP（默认 `192.168.1.18`），再启动：

```bash
source ~/questVR_ws/devel/setup.bash
roslaunch oculus_reader teleop_single_rm75.launch rm_ip:=192.168.1.18
```

如果驱动已经由其他 launch 启动，可使用：

```bash
roslaunch oculus_reader teleop_single_rm75.launch start_driver:=false
```

主要参数都在 [`config/rm75.yaml`](src/oculus_reader/config/rm75.yaml)，包括
关节限位、初始位姿、末端工具偏置、Quest 零点、夹爪开口、IK 迭代次数和
安全限幅。标准 RM75-B 的末端长度 `d7=144 mm` 已放入轻量 URDF；RM75-6F、
RM75-6FB 或安装了工具时，应把 `urdf_path` 和 `ik_ee_xyz/ik_ee_rpy` 改成
现场确认过的官方模型/工具参数。

## 双臂 RM75

官方 ROS1 驱动当前把控制指令订阅在绝对 `/rm_driver/*` 话题，两个驱动
实例不能仅靠 ROS namespace 隔离。因此双臂 launch 默认使用 API2 SDK，分别
连接 `192.168.1.18` 和 `192.168.1.19`：

```bash
roslaunch oculus_reader teleop_double_rm75.launch \
  left_rm_ip:=192.168.1.18 right_rm_ip:=192.168.1.19
```

如现场使用的是已修改为相对话题的驱动，可将 `robot_backend:=rm_ros`，并
让驱动分别发布到 launch 中配置的 `/left/...`、`/right/...` 话题。

双臂的限位、末端工具和 IK 参数默认继承 `rm75.yaml`，也可以分别加上
`left_`/`right_` 前缀覆盖，例如 `~left_urdf_path`、
`~right_ik_ee_xyz`、`~left_gripper_inverted`。这样左右两台不同型号或
不同夹爪方向的 RM75 不需要复制一套节点代码。

## 后端选择

| `robot_backend` | 用途 |
| --- | --- |
| `rm_ros` | 推荐的单臂 RM75；依赖官方 `rm_msgs`/`rm_driver` |
| `rm_sdk` | API2 Python SDK，适合双臂或不使用 ROS 驱动 |
| `joint_state` | 仿真、自定义桥接，发布 `sensor_msgs/JointState` |
| `dry_run` | 只计算 IK 和打印检查，不向机械臂发命令 |

首次接线建议先使用无驱动的安全检查模式，确认 Quest、URDF 和 IK 都能
启动：

```bash
roslaunch oculus_reader teleop_single_rm75.launch \
  start_driver:=false robot_backend:=dry_run show_rviz:=false
```

Piper 原有入口仍保留：`teleop_single_piper.launch` 和
`teleop_double_piper.launch` 使用旧 Piper 驱动与 `joint_state` 后端。

## 依据的官方资料

- [RM75 本体参数与 D-H 模型](https://develop.realman-robotics.com/robot/robotParameter/RM75OntologyParameters/)
- [RM API2 Python 快速开始](https://develop.realman-robotics.com/robot/apipython/getStarted/)
- [API2 关节跟随/角度透传](https://develop.realman-robotics.com/robot/apipython/classes/movePlan/)
- [RM ROS 话题说明](https://develop.realman-robotics.com/robot/ros/control/)
- [官方 ROS/消息包仓库](https://github.com/RealManRobot/rm_robot)
