"""
DAgger launch file - starts all required nodes for DAgger operation.

Nodes launched:
0. PolicyServer - gRPC 策略推理服务（可选，auto_launch=true 时自动启动）
1. realman_driver_node - Robot arm driver (from realman_teleop)
2. vr_input_node - VR data reception (from realman_teleop)
3. camera_node x2 - RealSense cameras (from realman_teleop)
4. dagger_node - DAgger control node (replaces vr_teleop_node)
5. dagger_control_panel - Web UI control panel (optional, enabled by default)

NOTE: vr_teleop_node is NOT launched (replaced by dagger_node).
NOTE: dagger_node 启动后进入 IDLE，用户点击"开始推理"时 connect() 自动重试等待 PolicyServer（最多 180 秒）。

Usage:
    # 启动（系统 Python 3.12 已有 rclpy + torch，无需 conda activate）
    ros2 launch dagger/launch/dagger.launch.py

    # python 路径解析优先级：
    #   1. dagger_params.yaml 的 python_path 字段（非 "auto" 的绝对路径）
    #   2. sys.executable（ros2 launch 使用的系统 Python，推荐）

    # 参数覆盖：直接修改 config/dagger_params.yaml 或 teleop_params.yaml
    # NOTE: dagger_node 和 control_panel 使用 ExecuteProcess 启动（非 ROS2 包注册节点），
    #       不支持 ros2 launch 命令行参数覆盖（如 ip:=xxx）。请通过配置文件修改参数。
"""
import os
import sys
import yaml
import shlex
from launch import LaunchDescription
from launch.actions import ExecuteProcess, LogInfo, RegisterEventHandler, EmitEvent
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch_ros.actions import Node


def _cleanup_stale_processes():
    """启动前清理残留的旧进程，防止双 publisher 导致 state topic 数据污染。

    已知踩坑：旧 realman_driver_node 残留会以 100Hz 发布 home 位置 state，
    与新 driver 交替被 dagger_node 接收，导致 obs 在 home 和真实位置之间震荡。
    """
    import subprocess
    import signal

    targets = ["realman_driver_node", "dagger_node", "control_panel", "policy_server"]
    killed = []
    for name in targets:
        try:
            result = subprocess.run(
                ["pgrep", "-f", name], capture_output=True, text=True, timeout=5
            )
            if result.stdout.strip():
                pids = result.stdout.strip().split("\n")
                for pid in pids:
                    pid = pid.strip()
                    if pid:
                        try:
                            os.kill(int(pid), signal.SIGTERM)
                            killed.append(f"{name}(PID={pid})")
                        except (ProcessLookupError, PermissionError):
                            pass
        except Exception:
            pass

    if killed:
        import time
        time.sleep(1)  # 等待旧进程退出
        print(f"[DAgger Launch] 已清理残留进程: {', '.join(killed)}")
    else:
        print("[DAgger Launch] 无残留进程")


def generate_launch_description():
    # ===== 启动前清理残留旧进程 =====
    _cleanup_stale_processes()

    # ===== Resolve config file paths =====
    # dagger/ is the parent of launch/
    dagger_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_root = os.path.dirname(dagger_dir)

    # Teleop params (hardware config for driver, cameras, VR input)
    teleop_config = os.path.join(
        project_root, "vr_teleop", "spacemouse_control_arm",
        "ros2_realman_ws", "src", "realman_teleop", "config",
        "teleop_params.yaml",
    )

    # DAgger params (DAgger-specific config)
    dagger_config = os.path.join(dagger_dir, "config", "dagger_params.yaml")

    if not os.path.exists(teleop_config):
        print(f"[DAgger Launch] WARNING: teleop config not found: {teleop_config}")
    if not os.path.exists(dagger_config):
        print(f"[DAgger Launch] WARNING: dagger config not found: {dagger_config}")

    # Read teleop config for dry_run default
    teleop_data = {}
    if os.path.exists(teleop_config):
        with open(teleop_config, "r") as f:
            teleop_data = yaml.safe_load(f) or {}

    # Read dagger config for DAgger-specific params
    dagger_data = {}
    if os.path.exists(dagger_config):
        with open(dagger_config, "r") as f:
            dagger_data = yaml.safe_load(f) or {}

    dry_run = teleop_data.get(
        "realman_driver_node", {}
    ).get("ros__parameters", {}).get("dry_run", False)

    # ===== Python 路径 =====
    # dagger_node / control_panel / PolicyServer 需要同时 import rclpy（ROS2）和 torch（ML）。
    # 解析优先级：
    #   1. dagger_params.yaml 中 python_path 字段（非 "auto" 的绝对路径）
    #   2. sys.executable（ros2 launch 使用的系统 Python，已有 rclpy + torch）
    python_cfg = dagger_data.get("python_path", "auto")
    if python_cfg and python_cfg != "auto" and os.path.exists(python_cfg):
        node_python = python_cfg
    else:
        node_python = sys.executable
    print(f"[DAgger Launch] Using python: {node_python}")

    # ===== PYTHONPATH for vr_utils import =====
    vr_utils_path = os.path.join(
        project_root, "vr_teleop", "spacemouse_control_arm",
        "ros2_realman_ws", "src", "realman_teleop", "realman_teleop",
    )

    # ===== Node definitions =====

    # 1. Robot arm driver
    # sigterm_timeout: 给 driver 足够时间关闭 RM+ 生态协议（默认只有 5s，RM+ 关闭需要 ~2s）
    driver_node = Node(
        package="realman_teleop",
        executable="realman_driver_node",
        name="realman_driver_node",
        parameters=[teleop_config],
        output="log",
        sigterm_timeout="10",
    )

    # 2. VR input node
    vr_input_node = Node(
        package="realman_teleop",
        executable="vr_input_node",
        name="vr_input_node",
        parameters=[teleop_config],
        output="log",
    )

    # 3. Camera nodes
    camera_node_cam0 = Node(
        package="realman_teleop",
        executable="camera_node",
        name="camera_node_cam0",
        parameters=[teleop_config],
        output="log",
    )

    camera_node_cam1 = Node(
        package="realman_teleop",
        executable="camera_node",
        name="camera_node_cam1",
        parameters=[teleop_config],
        output="log",
    )

    # ===== PolicyServer 自动启动配置（提前读取，dagger_node_params 需要引用） =====
    policy_server_config = dagger_data.get("policy_server", {})
    auto_launch = bool(policy_server_config.get("auto_launch", False))
    ps_policy_type = str(policy_server_config.get("policy_type", "act"))
    ps_pretrained_path = str(policy_server_config.get("pretrained_path", ""))
    ps_device = str(policy_server_config.get("device", "cuda"))
    ps_port = int(policy_server_config.get("port", 50051))
    ps_extra_args = list(policy_server_config.get("extra_args", []))
    # PolicyServer 专用 Python 路径（VLA 模型可能需要 conda 环境）
    ps_python_cfg = policy_server_config.get("python_path", "auto")
    if ps_python_cfg and ps_python_cfg != "auto" and os.path.exists(ps_python_cfg):
        ps_python = ps_python_cfg
    else:
        ps_python = node_python

    # PolicyServer 工作目录（RoboCOIN 根目录）
    robocoin_dir = os.path.join(project_root, "RoboCOIN")

    # 4. DAgger node (replaces vr_teleop_node)
    # 使用 ExecuteProcess 启动（dagger_node 未注册到 realman_teleop 包的 entry_points）
    # Merge teleop VR params + dagger-specific params
    vr_teleop_params = teleop_data.get(
        "vr_teleop_node", {}
    ).get("ros__parameters", {})

    dagger_vr_config = dagger_data.get("vr_control", {})
    dagger_inference_config = dagger_data.get("inference", {})
    dagger_safety_config = dagger_data.get("safety", {})

    dagger_node_params = {
        # --- Hardware / VR (from teleop config + dagger overrides) ---
        "ip": vr_teleop_params.get("ip", "192.168.1.18"),
        "port": vr_teleop_params.get("port", 8080),
        "dry_run": dry_run,
        "state_topic": vr_teleop_params.get("state_topic", "/rm/state_joint_state"),
        "control_hz": float(dagger_vr_config.get(
            "control_hz",
            vr_teleop_params.get("control_hz", 50.0),
        )),
        "trigger_threshold": float(dagger_vr_config.get(
            "trigger_threshold",
            vr_teleop_params.get("trigger_threshold", 0.85),
        )),
        "alpha_pos": float(dagger_vr_config.get(
            "alpha_pos",
            vr_teleop_params.get("alpha_pos", 0.8),
        )),
        "alpha_rot": float(dagger_vr_config.get(
            "alpha_rot",
            vr_teleop_params.get("alpha_rot", 0.8),
        )),
        "side": dagger_vr_config.get(
            "side",
            vr_teleop_params.get("side", "right"),
        ),
        "rotation_preset": dagger_vr_config.get(
            "rotation_preset",
            vr_teleop_params.get("rotation_preset", "perm_xyz_pnn"),
        ),
        "action_pose_topic": vr_teleop_params.get("action_pose_topic", "/rm/action_pose"),
        "action_joint_topic": "/rm/action_joint_state",
        "frame_id": vr_teleop_params.get("frame_id", "rm_base"),
        # --- Inference (from dagger_params.yaml) ---
        "enable_policy": bool(dagger_inference_config.get("enable_policy", True)),
        "server_address": str(dagger_data.get("server_address", "127.0.0.1:50051")),
        "task": str(dagger_data.get("task", "")),
        "f_exec": float(dagger_data.get("f_exec", 30.0)),
        "n_action_steps": int(dagger_data.get("n_action_steps", 0)),
        "T_inter": float(dagger_data.get("T_inter", 0.0)),
        "t_inf": float(dagger_data.get("t_inf", 0.0)),
        "chunk_size_threshold": int(dagger_data.get("chunk_size_threshold", 0)),
        "hold_on_empty": bool(dagger_data.get("hold_on_empty", True)),
        # --- Camera names ---
        "camera_names": list(dagger_data.get("camera_configs", {}).keys()) or ["cam0", "cam1"],
        # --- Camera ROS topic names (映射: inference key → ROS2 话题名) ---
        "camera_ros_names": list(dagger_data.get("camera_ros_names", {}).values()) or ["cam0", "cam1"],
        # --- Camera rotations (与 async_inference_client 一致) ---
        "camera_rotation_names": list(dagger_data.get("camera_rotations", {}).keys()),
        "camera_rotation_degrees": [int(v) for v in dagger_data.get("camera_rotations", {}).values()],
        # --- Safety (from dagger_params.yaml) ---
        "max_joint_delta_deg": float(dagger_safety_config.get("max_joint_delta_deg", 5.0)),
        "max_pose_delta_m": float(dagger_safety_config.get("max_pose_delta_m", 0.02)),
        # --- Recording (from dagger_params.yaml) ---
        "enable_recording": bool(dagger_data.get("enable_recording", False)),
        "repo_id": str(dagger_data.get("repo_id", "local/dagger_realman_001")),
        "dataset_root": str(dagger_data.get("dataset_root", "/home/ubuntu/Desktop/Workspace/lerobot_policy_deploy/data/dagger/")),
        "recording_task": str(dagger_data.get("recording_task", "pick and place")),
        "recording_fps": int(dagger_data.get("recording_fps", 30)),
        "use_videos": bool(dagger_data.get("use_videos", True)),
        "image_writer_threads": int(dagger_data.get("image_writer_threads", 4)),
        "max_episodes": int(dagger_data.get("max_episodes", 0)),
        # --- PolicyServer 信息（透传给 status 话题） ---
        "policy_type": ps_policy_type if auto_launch else "",
        "pretrained_path": ps_pretrained_path if auto_launch else "",
        # --- PolicyServer extra_args（用于 OpenPI 双重截取检测） ---
        "policy_server_extra_args": ps_extra_args if auto_launch else [],
    }

    # 构建 dagger_node 的 --ros-args -p 参数列表
    dagger_node_script = os.path.join(dagger_dir, "dagger_node.py")
    dagger_node_cmd = [node_python, dagger_node_script, "--ros-args"]
    for key, value in dagger_node_params.items():
        # ROS2 参数格式：-p key:=value
        # 注意：ROS2 不支持空值参数（-p key:= 会报错），跳过空字符串
        if isinstance(value, str) and value == "":
            continue
        # 列表类型（如 camera_names）需要用 JSON 数组格式
        if isinstance(value, list):
            # 实际上 rclpy 支持: -p camera_names:="[cam0, cam1]"
            list_str = str(value)
            dagger_node_cmd.extend(["-p", f"{key}:={list_str}"])
        elif isinstance(value, bool):
            # ROS2 bool: -p key:=true/false（小写）
            dagger_node_cmd.extend(["-p", f"{key}:={str(value).lower()}"])
        else:
            dagger_node_cmd.extend(["-p", f"{key}:={value}"])

    # PYTHONPATH：vr_utils（VR 控制）+ project_root（import dagger.core.*）+ ROS2
    ros2_python_path = "/opt/ros/jazzy/lib/python3.12/site-packages"
    dagger_pythonpath = f"{vr_utils_path}:{project_root}:{ros2_python_path}"
    # 保留已有 PYTHONPATH（可能包含用户自定义路径）
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    if existing_pythonpath:
        dagger_pythonpath = f"{dagger_pythonpath}:{existing_pythonpath}"
    print(f"[DAgger Launch] PYTHONPATH={dagger_pythonpath}")

    dagger_node_process = ExecuteProcess(
        cmd=["bash", "-c",
             f"export PYTHONPATH={shlex.quote(dagger_pythonpath)} && exec {shlex.join(dagger_node_cmd)}"],
        name="dagger_node",
        output="screen",
    )

    # 5. DAgger Web UI control panel
    dagger_web_ui_config = dagger_data.get("web_ui", {})
    web_ui_enabled = dagger_web_ui_config.get("enable", True)
    web_ui_port = int(dagger_web_ui_config.get("port", 5002))
    camera_names = list(dagger_data.get("camera_configs", {}).keys()) or ["cam0_rgb", "cam1_rgb"]
    camera_ros_names = list(dagger_data.get("camera_ros_names", {}).values()) or list(camera_names)

    actions = [
        # 硬件节点（立即启动）
        driver_node,
        vr_input_node,
        camera_node_cam0,
        camera_node_cam1,
    ]

    # ===== PolicyServer 自动启动（ExecuteProcess） =====
    if auto_launch:
        if not ps_pretrained_path:
            # pretrained_path 为空：跳过 PolicyServer，打印错误日志
            actions.append(LogInfo(
                msg="[DAgger Launch] ERROR: policy_server.auto_launch=true 但 pretrained_path 为空，跳过 PolicyServer 启动"
            ))
        else:
            # 构建 PolicyServer 启动命令
            ps_cmd = [
                ps_python, "-m",
                "lerobot.extensions.unified_deploy.server.policy_server",
                f"--policy_type={ps_policy_type}",
                f"--pretrained_path={ps_pretrained_path}",
                f"--device={ps_device}",
                f"--port={ps_port}",
            ]
            # 追加额外参数
            for arg in ps_extra_args:
                ps_cmd.append(str(arg))

            policy_server_process = ExecuteProcess(
                cmd=ps_cmd,
                cwd=robocoin_dir,
                name="policy_server",
                output="screen",
                additional_env={"PYTHONUNBUFFERED": "1"},
            )
            actions.append(policy_server_process)
            actions.append(LogInfo(
                msg=f"[DAgger Launch] PolicyServer 启动: type={ps_policy_type}, "
                    f"path={ps_pretrained_path}, device={ps_device}, port={ps_port}"
            ))

    # ===== dagger_node 启动 =====
    # dagger_node 启动后进入 IDLE 模式，不会立即连接 PolicyServer。
    # 用户点击"开始推理"时 connect() 会自动重试等待 PolicyServer 就绪（最多 180 秒）。
    actions.append(dagger_node_process)

    # ===== Web UI（ExecuteProcess，非 ROS2 包注册节点） =====
    if web_ui_enabled:
        web_ui_script = os.path.join(dagger_dir, "web_ui", "control_panel.py")
        # control_panel.py 内部用 rclpy 声明参数，通过 --ros-args -p 传递
        control_panel_cmd = [
            node_python, web_ui_script,
            "--ros-args",
            "-p", f"camera_names:={camera_names}",
            "-p", f"camera_ros_names:={camera_ros_names}",
            "-p", f"web_ui_port:={web_ui_port}",
        ]
        control_panel_process = ExecuteProcess(
            cmd=["bash", "-c",
                 f"export PYTHONPATH={shlex.quote(dagger_pythonpath)} && exec {shlex.join(control_panel_cmd)}"],
            name="dagger_control_panel",
            output="screen",
        )
        actions.append(control_panel_process)
        actions.append(LogInfo(
            msg=f"[DAgger Launch] ========== Web UI: http://0.0.0.0:{web_ui_port} =========="
        ))

    # ===== Shutdown event handler =====
    # 当 dagger_node 退出时，触发整个 launch 优雅关闭。
    # ROS2 launch 会向所有子进程发送 SIGTERM，driver_node 收到后关闭 RM+ 生态协议。
    actions.append(RegisterEventHandler(
        OnProcessExit(
            target_action=dagger_node_process,
            on_exit=[
                LogInfo(msg="[DAgger Launch] dagger_node 已退出，正在关闭所有节点（driver_node 将安全关闭 RM+ 协议）..."),
                EmitEvent(event=Shutdown(reason="dagger_node exited")),
            ],
        )
    ))

    return LaunchDescription(actions)
