"""
异步推理客户端

基于 ring buffer 的异步推理 + 固定频率执行框架。
替代 RealmanClient 的请求制推理，实现平滑的机械臂控制。

核心改进:
1. 推理线程完全异步，不阻塞执行
2. 固定频率执行（f_exec Hz），ring buffer 空时 hold last action
3. K_inf 裁剪：推理延迟期间消耗的 action 自动跳过
4. 稳定性约束：T_inter + t_inf < L * T_step

参考: thoughts/异步推理公式推导.md

用法:
    # 使用 YAML 配置
    python dagger/async_inference_client.py --config dagger/config/dagger_params.yaml

    # 命令行参数
    python dagger/async_inference_client.py --server 127.0.0.1:50051 --f-exec 30 --visualize

    # dry_run 模式（测试相机和通信，不控制机械臂）
    python dagger/async_inference_client.py --config dagger/config/dagger_params.yaml --dry-run
"""

import math
import os
import sys
import pickle
import threading
import time

# 在 SSH 环境下禁用 OpenCV GUI（必须在导入 cv2 之前设置）
# 设置为空字符串，让 OpenCV 不尝试初始化任何 GUI 后端
os.environ.pop('DISPLAY', None)
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List

import grpc

# === 路径设置 ===
# 项目根目录
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# RoboCOIN/src/ — 用于引用 lerobot.transport (gRPC 生成代码)
_ROBOCOIN_SRC = os.path.join(_PROJECT_ROOT, "RoboCOIN", "src")
if _ROBOCOIN_SRC not in sys.path:
    sys.path.insert(0, _ROBOCOIN_SRC)

from dagger.core.ring_buffer import ActionRingBuffer

# === 诊断日志（按 session 时间戳分目录）===
from dagger.core.diag_log import create_diag_logger
_ac_diag_log = create_diag_logger("async_client", print_to_stdout=False)

# === 数据类型：必须使用原始模块路径 ===
# pickle 序列化会记录类的模块路径，Server 端只认识 lerobot.extensions... 的路径。
# 如果用 dagger.deps.data_types，Server 反序列化时会报 "No module named 'dagger'"。
from lerobot.extensions.unified_deploy.core.data_types import (
    UnifiedObservation,
    UnifiedAction,
    ActionChunk,
)

# === 工具函数和配置：使用本地副本（不涉及序列化）===
from dagger.deps.utils import (
    rad_to_deg,
    gripper_normalized_to_sdk,
    binarize_gripper_sdk,
    rotate_image,
)
from dagger.deps.configs import RealmanClientConfig
from dagger.core.gripper_controller import GripperAsyncController

# gRPC transport（通过 sys.path 引用 RoboCOIN/src/lerobot/transport/）
import lerobot.transport.services_pb2 as services_pb2
import lerobot.transport.services_pb2_grpc as services_pb2_grpc
from lerobot.transport.utils import send_bytes_in_chunks, grpc_channel_options


@dataclass
class AsyncInferenceClientConfig(RealmanClientConfig):
    """扩展配置，新增异步推理参数"""
    f_exec: float = 30.0       # 固定执行频率 Hz
    T_inter: float = 0.0       # 推理间隔秒数（0=推理完成后立即触发下一次）
    hold_on_empty: bool = True  # ring buffer 空时是否重发最后一个 action
    t_inf: float = 0.0         # 实测推理延迟秒数（0=自动测量）
    chunk_size_threshold: int = 0  # chunk_size 阈值（0=自动从模型获取）

    # === Phase 2B: UDP 状态配置 ===
    enable_udp_state: bool = False   # 启用 UDP 状态接收
    udp_state_port: int = 8089       # UDP 接收端口
    udp_push_cycle: int = 2          # UDP 推送周期 (1=5ms, 2=10ms, 5=25ms)
    udp_recv_timeout: float = 1.0    # UDP 接收超时（秒）

    # === Phase 2B: 录制配置 ===
    enable_recording: bool = False
    repo_id: str = "local/dagger_data"
    dataset_root: str = ""
    use_videos: bool = True
    image_writer_threads: int = 4
    image_writer_processes: int = 0
    recording_task: str = ""         # 录制任务描述（写入 episode 元数据）

    def __post_init__(self):
        """覆盖父类验证：n_action_steps=0 在异步模式下表示使用全部 chunk"""
        if self.frequency < 1:
            raise ValueError(f"frequency 必须 >= 1，当前值: {self.frequency}")
        # n_action_steps=0 合法（表示使用全部 chunk），跳过父类的 >= 1 检查


class AsyncInferenceClient:
    """
    异步推理客户端

    线程模型:
      - 主线程: 调用 run()，内部启动执行定时器和推理线程
      - 执行循环: 固定 f_exec Hz，从 ring_buffer 取 action 执行
      - 推理线程: 异步 gRPC 通信，更新 ring_buffer

    使用方式:
      client = AsyncInferenceClient(config)
      client.connect()
      client.run()       # 阻塞，Ctrl+C 退出
      client.disconnect()
    """

    def __init__(self, config: AsyncInferenceClientConfig):
        self.config = config

        # === 复用 RealmanClient 的机器人控制部分 ===
        self.robot = None
        self.gripper_ctrl: Optional[GripperAsyncController] = None
        self.channel = None
        self.stub = None

        # === 新增：异步推理组件 ===
        self.ring_buffer = ActionRingBuffer(capacity=200)
        self.shutdown_event = threading.Event()
        self.server_metadata: Optional[Dict] = None
        self.server_task: Optional[str] = None
        self.server_action_space: Optional[str] = None
        self.timestep = 0

        # === 状态缓存 ===
        self.current_joints: Optional[np.ndarray] = None   # [7] rad
        self.current_ee_pose: Optional[np.ndarray] = None  # [6] (m, rad)
        self.current_gripper: float = 0.0
        self._last_binarized_gripper: Optional[int] = None

        # === 统计 ===
        self._exec_count = 0
        self._infer_count = 0
        self._hold_count = 0
        self._t_inf_avg = 0.0
        self._model_chunk_size: Optional[int] = None  # 模型返回的 chunk 大小（warmup 时记录）
        self._frame_count = 0
        self._fps = 0.0

        # === 运行时 t_inf EMA（自适应 K_inf 裁剪）===
        self._t_inf_ema = 0.0          # 指数移动平均
        self._t_inf_ema_alpha = 0.3    # EMA 平滑系数（越大越跟踪最新值）
        self._t_inf_ema_initialized = False  # 首次推理时用实际值初始化

        # === Phase 2B: UDP 状态接收 ===
        self._udp_receiver = None
        if self.config.enable_udp_state:
            from dagger.core.udp_state import UDPStateReceiver
            self._udp_receiver = UDPStateReceiver(
                port=self.config.udp_state_port,
                arm_dof=7,
            )

        # === Phase 2B: 图像缓存（录制用）===
        self._cached_images: Dict[str, np.ndarray] = {}

        # === Phase 2B: 录制器 ===
        self.recorder = None
        self._policy_action_dim = 8  # 默认值，warmup 后可能更新

        # === 可视化 ===
        self._vis_window_name = "AsyncInferenceClient (Press 'q' to close)"
        self._current_action: Optional[UnifiedAction] = None  # 用于可视化
        self._current_obs: Optional[UnifiedObservation] = None

    # ===== 连接管理（复用 RealmanClient 逻辑）=====

    def connect(self):
        """
        初始化连接
        1. 创建机器人实例（相机 + 机械臂 + 夹爪）
        2. 连接 gRPC PolicyServer
        3. 握手
        """
        # 1. 机器人
        self._create_robot()

        # 连接相机
        for cam in self.robot.cameras.values():
            cam.connect()

        # 相机预热
        if self.robot.cameras:
            print("[AsyncClient] 相机预热中...")
            for _ in range(10):
                for cam in self.robot.cameras.values():
                    cam.async_read()

        if self.config.dry_run:
            print("[DRY RUN] 仿真模式：相机已连接，机械臂未连接")
            self.current_joints = np.zeros(7, dtype=np.float32)
            self.current_ee_pose = np.zeros(6, dtype=np.float32)
        else:
            # 连接机械臂
            self.robot._connect_arm()
            print("[AsyncClient] 机械臂已连接")

            # 创建夹爪控制器
            effective_deadband = 0 if self.config.gripper_binarize else self.config.gripper_deadband
            self.gripper_ctrl = GripperAsyncController(
                ip=self.config.robot_ip,
                port=self.config.robot_port,
                baud=self.config.rm_plus_baud,
                deadband=effective_deadband,
                dry_run=self.config.dry_run,
            )

            if self.config.gripper_binarize:
                print(f"[AsyncClient] 夹爪二值化模式: threshold={self.config.gripper_binarize_threshold}, hysteresis={self.config.gripper_binarize_hysteresis}")
            else:
                print(f"[AsyncClient] 夹爪连续模式: deadband={self.config.gripper_deadband}")

            # === Phase 2B: 启动 UDP 状态接收 ===
            if self._udp_receiver is not None:
                from dagger.core.udp_state import enable_udp_push
                # 注意: RM+ 已由 GripperAsyncController 开启，无需再调 enable_rm_plus()

                # 1. 配置机械臂开启 UDP 推送
                local_ip = self._get_local_ip()
                success = enable_udp_push(
                    robot_arm=self.robot.arm,
                    target_ip=local_ip,
                    target_port=self.config.udp_state_port,
                    cycle=self.config.udp_push_cycle,
                    arm_ip=self.config.robot_ip,
                    arm_port=self.config.robot_port,
                )
                if not success:
                    print("[AsyncClient] 警告: UDP 推送配置失败，将使用 TCP 模式")
                    self._udp_receiver = None
                else:
                    # 2. 启动接收线程
                    self._udp_receiver.start()
                    # 3. 等待首包（最多 3 秒）
                    deadline = time.monotonic() + 3.0
                    while not self._udp_receiver.is_receiving(timeout=0.5):
                        if time.monotonic() > deadline:
                            print("[AsyncClient] 警告: UDP 状态接收超时，回退到 TCP 模式")
                            self._udp_receiver.stop()
                            self._udp_receiver = None
                            break
                        time.sleep(0.1)
                    if self._udp_receiver is not None:
                        print("[AsyncClient] UDP 状态接收已就绪")

            # 获取初始状态
            self._update_state()

        # 2. gRPC
        print(f"[AsyncClient] 连接 PolicyServer: {self.config.server_address}")
        self.channel = grpc.insecure_channel(
            self.config.server_address,
            options=grpc_channel_options(),
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)

        # 3. 握手
        try:
            self.stub.Ready(services_pb2.Empty())
            print("[AsyncClient] PolicyServer 连接成功")
        except grpc.RpcError as e:
            raise ConnectionError(f"无法连接 PolicyServer: {e}")

        # 初始化可视化
        if self.config.visualize:
            try:
                cv2.namedWindow(self._vis_window_name, cv2.WINDOW_NORMAL)
            except cv2.error as e:
                print(f"[警告] 无法初始化可视化窗口（可能在 SSH 环境下）: {e}")
                print("[警告] 可视化功能已禁用")
                self.config.visualize = False

    def _create_robot(self):
        """创建机器人实例（复用 realman_client.py 逻辑）"""
        from lerobot.robots.realman import RealmanConfig
        from lerobot.robots.utils import make_robot_from_config
        from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
        from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401

        cameras = {}
        for cam_name, cam_cfg in self.config.camera_configs.items():
            cam_type = cam_cfg.get("type", "intelrealsense")
            if cam_type == "intelrealsense":
                cameras[cam_name] = RealSenseCameraConfig(
                    serial_number_or_name=cam_cfg.get("serial_number_or_name", ""),
                    fps=cam_cfg.get("fps", 30),
                    width=cam_cfg.get("width", 640),
                    height=cam_cfg.get("height", 480),
                )
            elif cam_type == "opencv":
                cameras[cam_name] = OpenCVCameraConfig(
                    index_or_path=cam_cfg.get("index_or_path", 0),
                    fps=cam_cfg.get("fps", 30),
                    width=cam_cfg.get("width", 640),
                    height=cam_cfg.get("height", 480),
                )

        robot_config = RealmanConfig(
            ip=self.config.robot_ip,
            port=self.config.robot_port,
            cameras=cameras,
        )
        self.robot = make_robot_from_config(robot_config)

    def disconnect(self):
        """清理资源"""
        self.shutdown_event.set()

        # Phase 2B: 关闭录制器
        if self.recorder is not None:
            self.recorder.close()
            print("[AsyncClient] 录制器已关闭")

        # Phase 2B: 停止 UDP 接收
        if self._udp_receiver is not None:
            self._udp_receiver.stop()
            print("[AsyncClient] UDP 状态接收已停止")

        if self.config.visualize:
            cv2.destroyAllWindows()

        if self.gripper_ctrl:
            self.gripper_ctrl.close()

        if self.robot:
            for cam in self.robot.cameras.values():
                cam.disconnect()
            if not self.config.dry_run:
                self.robot._disconnect_arm()

        if self.channel:
            self.channel.close()

        print("[AsyncClient] 已断开连接")

    # ===== 观测采集（复用 realman_client.py 逻辑）=====

    def _update_state(self):
        """更新当前状态（Phase 2B: UDP 优先，TCP 回退）"""
        if self.config.dry_run:
            return

        if self._udp_receiver is not None and self._udp_receiver.is_receiving():
            # UDP 模式: 关节+EEF 从 UDP，夹爪从 TCP（UDP gripper 在此硬件上始终为 None）
            joints_deg, gripper_raw, eef_pose = self._udp_receiver.read_full_state()

            if joints_deg is not None:
                self.current_joints = np.radians(joints_deg).astype(np.float32)

            # 夹爪: UDP 通常为 None，用 TCP rm_get_gripper_state() 补充
            if gripper_raw is not None:
                self.current_gripper = float(gripper_raw) / 1000.0
            else:
                try:
                    ret_code, grip = self.robot.arm.rm_get_gripper_state()
                    if ret_code == 0:
                        self.current_gripper = float(grip['actpos']) / 1000.0
                except Exception:
                    pass  # 保持上一次的 gripper 值

            if eef_pose is not None:
                self.current_ee_pose = np.array(eef_pose, dtype=np.float32)

            # === DIAG-COMPARE: 每帧 gripper 来源写文件 ===
            if not hasattr(self, '_state_update_count'):
                self._state_update_count = 0
            self._state_update_count += 1
            _grip_source = 'UDP' if gripper_raw is not None else 'TCP'
            _ac_diag_log(
                f"[STATE] #{self._state_update_count}: "
                f"gripper={self.current_gripper:.4f}, "
                f"gripper_raw_udp={gripper_raw}, "
                f"source={_grip_source}"
            )
            if self._state_update_count <= 20 or self._state_update_count % 100 == 0:
                print(f"[DIAG-STATE-AC] #{self._state_update_count}: "
                      f"gripper={self.current_gripper:.4f}, "
                      f"gripper_raw_udp={gripper_raw}, "
                      f"source={_grip_source}")
        else:
            if self._udp_receiver is not None and not self._udp_receiver.is_receiving():
                print("[AsyncClient] 警告: UDP 接收中断，回退到 TCP 模式")
            # TCP 回退: 原始方式
            raw_obs = self.robot.get_observation()

            joint_names = self.robot.config.joint_names
            self.current_joints = np.array(
                [raw_obs[f'{name}_pos'] for name in joint_names[:7]],
                dtype=np.float32
            )

            gripper_key = f'{joint_names[-1]}_pos'
            gripper_raw = raw_obs.get(gripper_key, 0.0)
            self.current_gripper = float(gripper_raw) / 1000.0

            self.current_ee_pose = self.robot._get_ee_state()[:6].astype(np.float32)

    def get_observation(self) -> UnifiedObservation:
        """采集观测数据（Phase 2B: 图像缓存 + UDP 状态）"""
        if not self.config.dry_run:
            self._update_state()

        images = {}
        if self.config.dry_run:
            for cam_name, cam in self.robot.cameras.items():
                img = cam.async_read()
                if isinstance(img, np.ndarray):
                    rot_deg = self.config.camera_rotations.get(cam_name, 0)
                    if rot_deg != 0:
                        img = rotate_image(img, rot_deg)
                    images[cam_name] = img
        else:
            raw_obs = self.robot.get_observation()
            for cam_name in self.robot.cameras.keys():
                img = raw_obs.get(cam_name)
                if isinstance(img, np.ndarray):
                    rot_deg = self.config.camera_rotations.get(cam_name, 0)
                    if rot_deg != 0:
                        img = rotate_image(img, rot_deg)
                    images[cam_name] = img

        # Phase 2B: 缓存图像（录制用）
        self._cached_images = {k: v.copy() for k, v in images.items()}

        obs = UnifiedObservation(
            timestamp=time.time(),
            timestep=self.timestep,
            must_go=self.ring_buffer.is_empty,
            joints=self.current_joints.copy() if self.current_joints is not None else None,
            ee_pose=self.current_ee_pose.copy() if self.current_ee_pose is not None else None,
            gripper=self.current_gripper,
            images=images,
            task=self.config.task,
        )
        self._current_obs = obs

        # === DEBUG: 打印观测 gripper 值（前 20 次 + 每 50 次）===
        if not hasattr(self, '_get_obs_count'):
            self._get_obs_count = 0
        self._get_obs_count += 1
        if self._get_obs_count <= 20 or self._get_obs_count % 50 == 0:
            print(f"[DIAG-OBS-AC] #{self._get_obs_count}: "
                  f"gripper={self.current_gripper:.4f}, "
                  f"must_go={self.ring_buffer.is_empty}, "
                  f"timestep={self.timestep}")

        return obs

    # ===== 动作执行（复用 realman_client.py:404-447）=====

    # === Phase 2B: 辅助方法 ===

    def _get_local_ip(self) -> str:
        """获取本机 IP 地址（用于 UDP 推送目标），通过连接 robot_ip 确定本机出口 IP"""
        import socket as _socket
        try:
            s = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
            s.connect((self.config.robot_ip, 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    def _init_recorder(self):
        """延迟初始化录制器（warmup 完成后，_policy_action_dim 已确定）"""
        if self.recorder is not None or not self.config.enable_recording:
            return
        from dagger.core.data_recorder import DAggerRecorder
        camera_shapes = {}
        for cam_name in self.robot.cameras.keys():
            h = self.config.camera_configs.get(cam_name, {}).get("height", 480)
            w = self.config.camera_configs.get(cam_name, {}).get("width", 640)
            camera_shapes[cam_name] = (h, w, 3)

        dataset_root = self.config.dataset_root if self.config.dataset_root else None
        self.recorder = DAggerRecorder(
            repo_id=self.config.repo_id,
            fps=int(self.config.f_exec),
            root=dataset_root,
            camera_shapes=camera_shapes,
            policy_action_dim=self._policy_action_dim,
            use_videos=self.config.use_videos,
            image_writer_threads=self.config.image_writer_threads,
            image_writer_processes=self.config.image_writer_processes,
        )
        self.recorder.start_episode(task=self.config.recording_task or self.config.task)
        print(f"[AsyncClient] 录制已启动: repo_id={self.config.repo_id}, policy_action_dim={self._policy_action_dim}")

    def _capture_images(self) -> Dict[str, np.ndarray]:
        """采集当前帧图像（用于录制，独立于推理线程的 get_observation）"""
        images = {}
        for cam_name, cam in self.robot.cameras.items():
            img = cam.async_read()
            if isinstance(img, np.ndarray):
                rot_deg = self.config.camera_rotations.get(cam_name, 0)
                if rot_deg != 0:
                    img = rotate_image(img, rot_deg)
                images[cam_name] = img
        return images

    def _get_state_14d(self) -> Optional[np.ndarray]:
        """
        获取 14D 状态向量 [7 joints_rad, 1 gripper_norm, 6 eef_pose]

        始终从 current_* 缓存变量组装（由 _update_state() 维护）。
        """
        if self.current_joints is None or self.current_ee_pose is None:
            return None

        return np.concatenate([
            self.current_joints[:7],
            [np.clip(self.current_gripper, 0.0, 1.0)],
            self.current_ee_pose[:6],
        ]).astype(np.float32)

    def _get_current_policy_action(self) -> np.ndarray:
        """
        获取当前策略动作（用于录制 policy_action 字段）

        从 ring_buffer.last_popped 获取最近一次执行的 action，
        维度 = action_space 维度 + 1 (gripper)。
        """
        action = self.ring_buffer.last_popped
        if action is not None:
            # 拼接 data + gripper
            gripper = np.array([action.gripper], dtype=np.float32)
            return np.concatenate([action.data.astype(np.float32), gripper])
        else:
            # 无动作时返回零向量
            return np.zeros(self._policy_action_dim, dtype=np.float32)

    def execute_action(self, action: UnifiedAction):
        """
        执行动作

        Server 返回的动作统一为 absolute 格式，Client 直接执行：
        - joints: rad → degree → rm_movej_canfd
        - pose: (m, rad) → rm_movep_canfd
        - gripper: [0,1] → [0,1000] → JSON set_gripper_position
        """
        if self.config.dry_run:
            return

        target = action.data

        if action.action_space == "joints":
            joint_deg = rad_to_deg(target)
            if self.robot.arm is not None:
                self.robot.arm.rm_movej_canfd(list(joint_deg), False, 0.0, 0, 30)
        else:
            pose = target.tolist()
            if self.robot.arm is not None:
                ret = self.robot.arm.rm_movep_canfd(pose, False, 0, 0)
                if ret != 0:
                    print(f"[WARN] rm_movep_canfd 返回错误: {ret}")

        gripper_sdk = gripper_normalized_to_sdk(action.gripper)

        if self.config.gripper_binarize:
            gripper_sdk = binarize_gripper_sdk(
                gripper_sdk,
                threshold=self.config.gripper_binarize_threshold,
                hysteresis=self.config.gripper_binarize_hysteresis,
                last_state=self._last_binarized_gripper,
            )
            self._last_binarized_gripper = gripper_sdk

        # === DIAG-COMPARE: 每帧夹爪执行写文件 ===
        if not hasattr(self, '_exec_action_count'):
            self._exec_action_count = 0
        self._exec_action_count += 1
        obs_gripper = self.current_gripper
        _will_send = (self.gripper_ctrl and
                      (self.gripper_ctrl._last_cmd is None or
                       abs(gripper_sdk - self.gripper_ctrl._last_cmd) >= self.gripper_ctrl.deadband))
        _ac_diag_log(
            f"[GRIP-CMD] #{self._exec_action_count}: "
            f"act={action.gripper:.4f}, sdk={gripper_sdk}, obs={obs_gripper:.4f}, "
            f"delta={action.gripper - obs_gripper:.4f}, "
            f"will_send={_will_send}, last_cmd={self.gripper_ctrl._last_cmd if self.gripper_ctrl else 'N/A'}"
        )
        if self._exec_action_count <= 20 or self._exec_action_count % 50 == 0:
            print(f"[DIAG-EXEC-AC] #{self._exec_action_count}: "
                  f"action.gripper={action.gripper:.4f}, "
                  f"gripper_sdk={gripper_sdk}, "
                  f"obs_gripper={obs_gripper:.4f}, "
                  f"delta={action.gripper - obs_gripper:.4f}")

        if self.gripper_ctrl:
            self.gripper_ctrl.send_async(gripper_sdk)

    # ===== 核心：异步推理 + 固定频率执行 =====

    def run(self):
        """
        主循环：启动推理线程 + 固定频率执行循环

        执行循环在主线程运行（方便 Ctrl+C 退出）。
        推理线程在后台异步运行。
        """
        f_exec = self.config.f_exec
        T_step = 1.0 / f_exec

        print(f"\n{'='*60}")
        print(f"AsyncInferenceClient {'[仿真模式]' if self.config.dry_run else '[真机模式]'}")
        print(f"{'='*60}")
        print(f"服务器: {self.config.server_address}")
        print(f"执行频率: {f_exec} Hz (T_step={T_step:.4f}s)")
        print(f"推理间隔: T_inter={self.config.T_inter}s")
        print(f"Hold on empty: {self.config.hold_on_empty}")
        print(f"任务: {self.config.task if self.config.task else '(等待 Server 返回...)'}")
        print(f"{'='*60}")

        if not self.config.dry_run:
            input("按回车开始...")

        # === Phase 2B: 录制器延迟初始化（等 warmup 确定 _policy_action_dim 后再创建）===
        # recorder 在执行循环中首次录制时初始化，见 _init_recorder()

        # 启动推理线程
        inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        inference_thread.start()

        start_time = time.time()
        print("[AsyncClient] 运行中... (Ctrl+C 停止)")

        # === 固定频率执行循环（主线程）===
        _exec_loop_start = time.monotonic()
        _ac_exec_count = 0
        try:
            while not self.shutdown_event.is_set():
                t_start = time.monotonic()

                # 0. 更新状态缓存（每帧刷新，确保录制拿到最新状态）
                _t_state = time.monotonic()
                if not self.config.dry_run:
                    self._update_state()
                _t_state_done = time.monotonic()

                # 1. 从 ring_buffer 取 action
                _t_pop = time.monotonic()
                action = self.ring_buffer.pop_current()
                _t_pop_done = time.monotonic()

                if action is not None:
                    # 正常执行
                    self._current_action = action
                    _t_exec = time.monotonic()
                    self.execute_action(action)
                    _t_exec_done = time.monotonic()
                    self._exec_count += 1
                elif self.config.hold_on_empty:
                    # buffer 空：hold last action
                    last = self.ring_buffer.last_popped
                    if last is not None:
                        _t_exec = time.monotonic()
                        self.execute_action(last)
                        _t_exec_done = time.monotonic()
                        self._hold_count += 1
                    else:
                        _t_exec = _t_exec_done = time.monotonic()
                else:
                    _t_exec = _t_exec_done = time.monotonic()

                self.timestep += 1
                self._frame_count += 1
                _ac_exec_count += 1

                # === DIAG-COMPARE: 每帧写文件日志，用于对比 dagger_node ===
                _grip_obs = self.current_gripper
                _grip_act = action.gripper if action is not None else -1
                _grip_sdk = gripper_normalized_to_sdk(_grip_act) if action is not None else -1
                _state_ms = (_t_state_done - _t_state) * 1000
                _is_hold_diag = (action is not None and hasattr(action, '_is_hold'))
                _buf_sz = self.ring_buffer.size
                _ac_diag_log(
                    f"[EXEC] #{_ac_exec_count}: "
                    f"grip_obs={_grip_obs:.4f}, grip_act={_grip_act:.4f}, grip_sdk={_grip_sdk}, "
                    f"delta={_grip_act - _grip_obs:.4f}, "
                    f"state_update={_state_ms:.2f}ms, "
                    f"buf={_buf_sz}, hold={action is None}"
                )

                # === TIMING: exec loop 各阶段耗时 ===
                if _ac_exec_count <= 10 or _ac_exec_count % 100 == 0:
                    _elapsed_since_start = time.monotonic() - _exec_loop_start
                    _actual_hz = _ac_exec_count / _elapsed_since_start if _elapsed_since_start > 0 else 0
                    _is_hold = action is None
                    print(f"[TIMING-EXEC-AC] #{_ac_exec_count}: "
                          f"state_update={(_t_state_done-_t_state)*1000:.2f}ms, "
                          f"pop={(_t_pop_done-_t_pop)*1000:.2f}ms, "
                          f"execute={(_t_exec_done-_t_exec)*1000:.2f}ms, "
                          f"total_work={(_t_exec_done-t_start)*1000:.2f}ms, "
                          f"{'HOLD' if _is_hold else 'EXEC'}, "
                          f"buf={self.ring_buffer.size}, "
                          f"actual_hz={_actual_hz:.1f}")

                # === Phase 2B: 录制当前帧 ===
                if self.config.enable_recording:
                    # 延迟初始化：等 warmup 确定 _policy_action_dim 后再创建
                    if self.recorder is None and self._model_chunk_size is not None:
                        self._init_recorder()
                    if self.recorder is not None and self.recorder._episode_started:
                        state_14d = self._get_state_14d()
                        if state_14d is not None:
                            policy_action = self._get_current_policy_action()
                            # 采集当前帧图像（独立于推理线程）
                            rec_images = self._capture_images()
                            # control_source: 0=policy, 1=human (Phase 2B 暂时全部标记为 policy)
                            self.recorder.add_frame(
                                state_14d=state_14d,
                                images=rec_images,
                                policy_action=policy_action,
                                control_source=0,
                            )

                # FPS 统计
                elapsed_total = time.time() - start_time
                self._fps = self._frame_count / elapsed_total if elapsed_total > 0 else 0

                # 可视化
                if self.config.visualize:
                    self._visualize()

                # 定期打印统计
                if self._frame_count % 100 == 0:
                    print(f"Frame {self._frame_count}, FPS={self._fps:.1f}, "
                          f"Exec={self._exec_count}, Hold={self._hold_count}, "
                          f"Infer={self._infer_count}, Buf={self.ring_buffer.size}")

                # 最大步数限制
                if self.config.max_steps > 0 and self._frame_count >= self.config.max_steps:
                    break

                # 帧率控制
                elapsed = time.monotonic() - t_start
                sleep_time = T_step - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print(f"\n[AsyncClient] 用户中断，共执行 {self._frame_count} 帧")

        # === Phase 2B: 结束录制 ===
        if self.recorder is not None:
            self.recorder.finish_episode()
            print(f"[AsyncClient] Episode 录制完成，共 {self.recorder.num_episodes} 个 episode")

        # 统计
        self._print_stats()
        self.shutdown_event.set()

    def _measure_inference_time(self, n_warmup: int = 5) -> float:
        """
        Warmup: 多次推理测量稳定 t_inf

        如果 config.t_inf > 0，直接使用配置值（仍执行 1 次推理获取 chunk_size）。
        否则执行 n_warmup 次推理，丢弃第 1 次（GPU 冷启动），取后 n-1 次平均值。

        参数:
            n_warmup: 总推理次数（默认 5，至少 2 次才能丢弃冷启动）

        返回: t_inf 秒数
        """
        n_warmup = max(n_warmup, 2)

        if self.config.t_inf > 0:
            print(f"[Warmup] 使用配置的 t_inf={self.config.t_inf:.3f}s")
            # 仍需执行 1 次推理获取 chunk_size
            self._warmup_single(tag="chunk_size探测")
            return self.config.t_inf

        print(f"[Warmup] 开始测量推理延迟（{n_warmup} 次，丢弃第 1 次冷启动）...")

        # 等待 gRPC 服务就绪
        max_ready_retries = 10
        for i in range(max_ready_retries):
            if self.shutdown_event.is_set():
                return 0.0
            try:
                self.stub.Ready(services_pb2.Empty())
                print(f"[Warmup] gRPC 服务就绪")
                break
            except grpc.RpcError as e:
                print(f"[Warmup] 等待 gRPC 就绪 ({i+1}/{max_ready_retries}): {e}")
                time.sleep(1.0)
        else:
            print("[Warmup] ERROR: gRPC 服务未就绪，无法测量 t_inf")
            return 0.0

        # 多次推理测量
        t_inf_list = []
        last_actions = None

        for i in range(n_warmup):
            if self.shutdown_event.is_set():
                return 0.0

            result = self._warmup_single(tag=f"第{i+1}/{n_warmup}次")
            if result is None:
                continue

            t_inf_i, actions = result
            t_inf_list.append(t_inf_i)
            last_actions = actions

        if not t_inf_list:
            print("[Warmup] ERROR: 所有 warmup 推理均失败")
            return 0.0

        # 丢弃第 1 次（冷启动），取后面的平均值
        if len(t_inf_list) >= 2:
            t_inf_cold = t_inf_list[0]
            t_inf_stable = t_inf_list[1:]
            t_inf = sum(t_inf_stable) / len(t_inf_stable)
            print(f"[Warmup] 冷启动 t_inf={t_inf_cold:.3f}s（已丢弃）")
            print(f"[Warmup] 稳定测量值: {[f'{t:.3f}s' for t in t_inf_stable]}")
            print(f"[Warmup] 平均 t_inf={t_inf:.3f}s（基于 {len(t_inf_stable)} 次稳定测量）")
        else:
            t_inf = t_inf_list[0]
            print(f"[Warmup] 仅 1 次成功测量，t_inf={t_inf:.3f}s（可能包含冷启动开销）")

        # 将最后一次 warmup 结果写入 buffer（避免浪费）
        if last_actions is not None:
            if self.config.n_action_steps > 0:
                last_actions = last_actions[:self.config.n_action_steps]
            K_inf = int(t_inf * self.config.f_exec)
            if K_inf < len(last_actions):
                valid_actions = last_actions[K_inf:]
            else:
                valid_actions = last_actions[-1:]
            self.ring_buffer.update(valid_actions)

        return t_inf

    def _warmup_single(self, tag: str = "") -> tuple:
        """
        执行单次 warmup 推理

        参数:
            tag: 日志标签

        返回:
            (t_inf, actions) 或 None（失败时）
        """
        try:
            obs = self.get_observation()
            obs_bytes = pickle.dumps(obs)

            t_start = time.monotonic()
            obs_iterator = send_bytes_in_chunks(
                obs_bytes,
                services_pb2.Observation,
                log_prefix=f"[Warmup:{tag}] Observation",
                silent=True,
            )
            self.stub.SendObservations(obs_iterator)
            response = self.stub.GetActions(services_pb2.Empty())
            t_inf = time.monotonic() - t_start

            actions = self._parse_response(response)
            if not actions:
                print(f"[Warmup] {tag} 返回空 actions")
                return None

            # 记录/验证 chunk_size
            chunk_size = len(actions)
            if self._model_chunk_size is None:
                self._model_chunk_size = chunk_size
                # Phase 2B: 从首次推理结果推断 policy_action_dim
                if actions:
                    a0 = actions[0]
                    self._policy_action_dim = len(a0.data) + 1  # data + gripper
            elif chunk_size != self._model_chunk_size:
                print(f"[Warmup] WARNING: chunk_size 不一致: "
                      f"之前={self._model_chunk_size}, 本次={chunk_size}")

            print(f"[Warmup] {tag}: t_inf={t_inf:.3f}s, chunk_size={chunk_size}")
            return (t_inf, actions)

        except Exception as e:
            print(f"[Warmup] {tag} 失败: {e}")
            return None

    def _derive_parameters(self, t_inf: float) -> dict:
        """
        根据实测参数推导运行时参数

        参数:
            t_inf: 实测推理延迟（秒）

        返回:
            dict 包含:
                L: 模型原始 chunk_size
                L_eff: 有效 chunk 长度（考虑 n_action_steps 截取）
                K_inf: 推理延迟消耗的 action 步数
                L_valid: 每次推理的有效 action 数
                T_inter_max: 最大允许推理间隔
                T_inter: 实际使用的推理间隔
                chunk_size_threshold: buffer 触发阈值
                feasible: 是否满足稳定性约束
        """
        f_exec = self.config.f_exec
        T_step = 1.0 / f_exec

        # L: 模型 chunk_size
        L = self._model_chunk_size
        if not L or L == 0:
            print("[Parameters] ERROR: _model_chunk_size 未设置或为 0，无法推导参数")
            print("[Parameters] 降级为被动等待模式（buffer <= 2 时触发推理）")
            return {
                "L": 0, "L_eff": 0, "K_inf": 0, "L_valid": 0,
                "T_inter_max": 0.0, "T_inter": 0.0,
                "chunk_size_threshold": 2,
                "feasible": False,
            }

        # L_eff: 有效 chunk 长度
        n_action_steps = self.config.n_action_steps
        if n_action_steps > 0:
            L_eff = min(n_action_steps, L)
        else:
            L_eff = L

        # K_inf: 推理延迟消耗的步数
        K_inf = int(math.ceil(t_inf * f_exec))

        # L_valid: 每次推理的有效 action 数
        L_valid = L_eff - K_inf

        # T_inter_max: 最大允许推理间隔（稳定性约束）
        # 约束: T_inter + t_inf < L_eff * T_step
        # => T_inter < L_eff * T_step - t_inf
        T_inter_max = L_eff * T_step - t_inf

        # T_inter: 实际使用的推理间隔
        if self.config.T_inter > 0:
            # 用户指定了 T_inter，尽量限制在 T_inter_max 内
            T_inter = min(self.config.T_inter, T_inter_max) if T_inter_max > 0 else self.config.T_inter
        else:
            # 自动推导：T_inter_max 的 80%（20% 安全余量）
            T_inter = T_inter_max * 0.8 if T_inter_max > 0 else 0.0

        # chunk_size_threshold: buffer 安全网阈值
        # 当 buffer 剩余 <= chunk_size_threshold 时紧急触发推理
        if self.config.chunk_size_threshold > 0:
            chunk_size_threshold = self.config.chunk_size_threshold
        else:
            chunk_size_threshold = max(2, K_inf)

        # feasible: 稳定性检查（T_inter 已自动推导，只需检查 L_valid 和 T_inter_max）
        feasible = L_valid > 0 and T_inter_max > 0

        # 打印完整参数推导日志
        print(f"\n{'='*60}")
        print(f"[Parameters] === Phase 1.5B 参数推导 ===")
        print(f"{'='*60}")
        print(f"  模型 chunk_size (L):       {self._model_chunk_size}")
        print(f"  有效 chunk_size (L_eff):   {L_eff}")
        print(f"  n_action_steps:            {n_action_steps} {'(全部)' if n_action_steps == 0 else ''}")
        print(f"  执行频率 (f_exec):         {f_exec} Hz")
        print(f"  执行步长 (T_step):         {T_step:.4f}s")
        print(f"  推理延迟 (t_inf):          {t_inf:.3f}s")
        print(f"  延迟消耗步数 (K_inf):      {K_inf}")
        print(f"  有效 action 数 (L_valid):  {L_valid}")
        print(f"  最大推理间隔 (T_inter_max):{T_inter_max:.3f}s")
        print(f"  实际推理间隔 (T_inter):    {T_inter:.3f}s {'(自动推导)' if self.config.T_inter == 0 else '(用户配置)'}")
        print(f"  Buffer 安全网阈值:         {chunk_size_threshold}")
        print(f"  稳定性约束:                {'PASS' if feasible else 'FAIL'}")
        if not feasible:
            if L_valid <= 0:
                print(f"  WARNING: L_valid={L_valid} <= 0，推理延迟超过 chunk 长度！")
                print(f"           需要降低 f_exec 或增大 chunk_size")
            if T_inter_max <= 0:
                print(f"  WARNING: T_inter_max={T_inter_max:.3f}s <= 0，无可用推理间隔")
                print(f"           需要降低 f_exec 或增大 chunk_size")
        print(f"{'='*60}\n")

        return {
            "L": L,
            "L_eff": L_eff,
            "K_inf": K_inf,
            "L_valid": L_valid,
            "T_inter_max": T_inter_max,
            "T_inter": T_inter,
            "chunk_size_threshold": chunk_size_threshold,
            "feasible": feasible,
        }

    def _inference_loop(self):
        """
        异步推理线程 (Phase 1.5B: 主动定时 + 安全网混合模式)

        两种运行模式:
        - 可行模式 (feasible=True): 定时 T_inter 触发推理，buffer <= threshold 时安全网提前触发
        - 降级模式 (feasible=False): 被动等待 buffer <= threshold 再触发（兜底）

        循环:
        1. 采集观测快照
        2. gRPC 发送 → 接收 ActionChunk
        3. 计算 K_inf，裁剪过时动作
        4. 更新 ring_buffer
        5. 定时等待 T_inter，期间安全网轮询 buffer
        """
        print("[AsyncClient] 推理线程启动")

        # === 启动阶段: 测量 + 推导 ===
        t_inf_measured = self._measure_inference_time()
        t_inf = self.config.t_inf if self.config.t_inf > 0 else t_inf_measured
        params = self._derive_parameters(t_inf)

        T_inter = params["T_inter"]
        threshold = params["chunk_size_threshold"]
        feasible = params["feasible"]

        if not feasible:
            print("[WARN] 系统不可行，降级为被动等待模式 (threshold=2)")
            threshold = 2
            T_inter = 0  # 推理完成立即触发下一次

        # === 主循环: 主动定时 + 安全网 ===
        while not self.shutdown_event.is_set():
            t_start = time.monotonic()

            # === 降级模式: 被动等待 buffer 消耗到阈值以下再推理 ===
            if not feasible:
                while not self.shutdown_event.is_set():
                    if self.ring_buffer.size <= threshold:
                        break
                    time.sleep(0.005)  # 5ms 轮询
                if self.shutdown_event.is_set():
                    break
                t_start = time.monotonic()  # 重新计时（排除等待时间）

            try:
                # 1. 采集观测
                _t_obs = time.monotonic()
                obs = self.get_observation()
                _t_obs_done = time.monotonic()

                # === DIAG-COMPARE: 推理线程观测快照 ===
                _ac_diag_log(
                    f"[INFER-OBS] #{self._infer_count}: "
                    f"gripper={obs.gripper:.4f}, "
                    f"obs_age=0.0ms, "  # async_client 直接读，无延迟
                    f"obs_time={(_t_obs_done-_t_obs)*1000:.2f}ms, "
                    f"buf={self.ring_buffer.size}"
                )
                _t_pickle = time.monotonic()
                obs_bytes = pickle.dumps(obs)
                _t_pickle_done = time.monotonic()

                # 2. gRPC 推理
                _t_grpc = time.monotonic()
                response = self._grpc_infer(obs_bytes)
                _t_grpc_done = time.monotonic()
                if response is None:
                    time.sleep(0.1)
                    continue

                # 3. 解析 ActionChunk
                _t_parse = time.monotonic()
                actions = self._parse_response(response)
                _t_parse_done = time.monotonic()
                if not actions:
                    time.sleep(0.1)
                    continue

                # 截取 n_action_steps
                if self.config.n_action_steps > 0:
                    actions = actions[:self.config.n_action_steps]

                # 4. K_inf 裁剪（基于运行时 EMA 自适应）
                t_inf_actual = time.monotonic() - t_start

                # 更新 EMA
                if not self._t_inf_ema_initialized:
                    self._t_inf_ema = t_inf_actual
                    self._t_inf_ema_initialized = True
                else:
                    self._t_inf_ema = (self._t_inf_ema_alpha * t_inf_actual
                                       + (1 - self._t_inf_ema_alpha) * self._t_inf_ema)

                # 用 EMA 和实际值中的较大者来裁剪，+2 覆盖残余误差
                t_inf_for_clip = max(t_inf_actual, self._t_inf_ema)
                K_inf_actual = int(t_inf_for_clip * self.config.f_exec) + 10
                if K_inf_actual < len(actions):
                    valid_actions = actions[K_inf_actual:]
                else:
                    valid_actions = actions[-1:]  # 至少保留最后一个

                # 5. 更新 ring_buffer
                _t_buf = time.monotonic()
                buf_size = self.ring_buffer.update(valid_actions)
                _t_buf_done = time.monotonic()

                # 统计
                self._infer_count += 1
                self._t_inf_avg = (
                    self._t_inf_avg * (self._infer_count - 1) + t_inf_actual
                ) / self._infer_count

                # === TIMING: 推理线程各阶段耗时 ===
                if self._infer_count <= 5 or self._infer_count % 10 == 0:
                    print(f"[TIMING-INFER-AC] #{self._infer_count}: "
                          f"obs={(_t_obs_done-_t_obs)*1000:.2f}ms, "
                          f"pickle={(_t_pickle_done-_t_pickle)*1000:.2f}ms, "
                          f"grpc={(_t_grpc_done-_t_grpc)*1000:.2f}ms, "
                          f"parse={(_t_parse_done-_t_parse)*1000:.2f}ms, "
                          f"buf_update={(_t_buf_done-_t_buf)*1000:.2f}ms, "
                          f"t_inf_total={t_inf_actual*1000:.1f}ms, "
                          f"K_inf={K_inf_actual}, valid={len(valid_actions)}, "
                          f"buf={buf_size}, ema={self._t_inf_ema*1000:.1f}ms")

                # === DEBUG: 打印 action chunk gripper 信息 ===
                if self._infer_count < 10:
                    _grippers_all = [a.gripper for a in actions]
                    _grippers_valid = [a.gripper for a in valid_actions]
                    _obs_grip = self.current_gripper
                    print(f"[DIAG-ACT-AC] infer#{self._infer_count}: "
                          f"K_inf={K_inf_actual}, chunk={len(actions)}, valid={len(valid_actions)}\n"
                          f"  obs_gripper={_obs_grip:.4f}\n"
                          f"  gripper_all:   min={min(_grippers_all):.4f}, max={max(_grippers_all):.4f}, "
                          f"first5={[f'{g:.3f}' for g in _grippers_all[:5]]}\n"
                          f"  gripper_valid: min={min(_grippers_valid):.4f}, max={max(_grippers_valid):.4f}, "
                          f"first5={[f'{g:.3f}' for g in _grippers_valid[:5]]}")
                elif self._infer_count % 10 == 0:
                    _grippers_valid = [a.gripper for a in valid_actions]
                    _obs_grip = self.current_gripper
                    print(f"[DIAG-ACT-GRIP-AC] infer#{self._infer_count}: "
                          f"obs_grip={_obs_grip:.4f}, "
                          f"K_inf={K_inf_actual}, valid={len(valid_actions)}, "
                          f"grip_valid: min={min(_grippers_valid):.4f}, max={max(_grippers_valid):.4f}, "
                          f"first3={[f'{g:.3f}' for g in _grippers_valid[:3]]}")

            except Exception as e:
                print(f"[Inference] Error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.5)
                continue

            # 6. 定时等待 + 安全网检查
            elapsed = time.monotonic() - t_start
            remaining = T_inter - elapsed
            if remaining > 0:
                T_step = 1.0 / self.config.f_exec
                poll_interval = min(T_step * 0.3, 0.01)  # 最大 10ms
                while remaining > 0 and not self.shutdown_event.is_set():
                    if self.ring_buffer.size <= threshold:
                        break  # 安全网触发，立即开始下次推理
                    time.sleep(min(remaining, poll_interval))
                    remaining = T_inter - (time.monotonic() - t_start)

    def _grpc_infer(self, obs_bytes: bytes):
        """gRPC 推理通信"""
        try:
            obs_iterator = send_bytes_in_chunks(
                obs_bytes,
                services_pb2.Observation,
                log_prefix="[AsyncClient] Observation",
                silent=True,
            )
            self.stub.SendObservations(obs_iterator)

            response = self.stub.GetActions(services_pb2.Empty())
            return response
        except grpc.RpcError as e:
            if not self.shutdown_event.is_set():
                print(f"[gRPC] Error: {e}")
            return None

    def _parse_response(self, response) -> List[UnifiedAction]:
        """解析 gRPC 响应为 UnifiedAction 列表"""
        if len(response.data) == 0:
            return []

        action_chunk = pickle.loads(response.data)

        # 提取元数据（首次接收时）
        if self.server_metadata is None and isinstance(action_chunk, ActionChunk):
            if action_chunk.metadata:
                self.server_metadata = action_chunk.metadata
                self.server_task = action_chunk.metadata.get("task", "")
                self.server_action_space = action_chunk.metadata.get("action_space", "")
                print(f"\n[AsyncClient] 已从 Server 获取配置:")
                print(f"  任务: {self.server_task}")
                print(f"  动作空间: {self.server_action_space}")
                print(f"  动作类型: {action_chunk.metadata.get('action_type', '')}\n")

        # 处理两种格式
        if isinstance(action_chunk, ActionChunk):
            actions = action_chunk.actions
        elif isinstance(action_chunk, list):
            actions = action_chunk
        else:
            actions = self._convert_timed_actions(action_chunk)

        return actions

    def _convert_timed_actions(self, timed_actions) -> List[UnifiedAction]:
        """转换 TimedAction 格式为 UnifiedAction（兼容旧协议）"""
        actions = []
        for ta in timed_actions:
            action_data = ta.get_action()
            if hasattr(action_data, 'numpy'):
                action_data = action_data.numpy()

            if len(action_data) >= 8:
                data = action_data[:7]
                gripper = action_data[7]
                action_space = "joints"
            else:
                data = action_data[:6]
                gripper = action_data[6] if len(action_data) > 6 else 0.0
                action_space = "pose"

            actions.append(UnifiedAction(
                timestamp=ta.get_timestamp(),
                timestep=ta.get_timestep(),
                action_space=action_space,
                action_type="absolute",
                data=np.array(data, dtype=np.float32),
                gripper=float(gripper),
            ))
        return actions

    # ===== 可视化 =====

    def _visualize(self):
        """可视化观测和动作信息"""
        if not self.config.visualize:
            return

        try:
            obs = self._current_obs
            action = self._current_action
            if obs is None:
                return

            COLOR_TITLE = (0, 255, 255)
            COLOR_JOINT = (0, 255, 0)
            COLOR_POSE = (255, 200, 100)
            COLOR_GRIPPER = (255, 0, 255)
            COLOR_STATUS = (200, 200, 200)

            cam_images = []
            for cam_name, img in obs.images.items():
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.putText(img_bgr, cam_name, (10, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cam_images.append(img_bgr)

            if not cam_images:
                return

            img_canvas = np.hstack(cam_images)

            if self.config.vis_scale != 1.0:
                new_w = int(img_canvas.shape[1] * self.config.vis_scale)
                new_h = int(img_canvas.shape[0] * self.config.vis_scale)
                img_canvas = cv2.resize(img_canvas, (new_w, new_h))

            panel_height = 200
            panel_width = img_canvas.shape[1]
            info_panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            line_height = 20
            y_offset = 20

            # [Current State]
            cv2.putText(info_panel, "[Current State]", (10, y_offset), font, 0.5, COLOR_TITLE, 1)
            y_offset += line_height

            if obs.joints is not None:
                joints_deg = np.degrees(obs.joints)
                joint_str = "  Joints(deg): " + " ".join([f"J{i+1}={joints_deg[i]:+7.2f}" for i in range(min(7, len(joints_deg)))])
                cv2.putText(info_panel, joint_str, (10, y_offset), font, font_scale, COLOR_JOINT, 1)
                y_offset += line_height

            if obs.ee_pose is not None:
                pose_str = f"  EE Pose: x={obs.ee_pose[0]:.4f}m y={obs.ee_pose[1]:.4f}m z={obs.ee_pose[2]:.4f}m  rx={obs.ee_pose[3]:.3f} ry={obs.ee_pose[4]:.3f} rz={obs.ee_pose[5]:.3f}"
                cv2.putText(info_panel, pose_str, (10, y_offset), font, font_scale, COLOR_POSE, 1)
                y_offset += line_height

            gripper_sdk = gripper_normalized_to_sdk(obs.gripper)
            gripper_str = f"  Gripper: {obs.gripper:.3f} (SDK: {gripper_sdk})"
            cv2.putText(info_panel, gripper_str, (10, y_offset), font, font_scale, COLOR_GRIPPER, 1)
            y_offset += line_height

            cv2.line(info_panel, (10, y_offset + 5), (panel_width - 10, y_offset + 5), (80, 80, 80), 1)
            y_offset += 15

            # [Target]
            action_space_str = f"({action.action_space})" if action else ""
            cv2.putText(info_panel, f"[Target] {action_space_str}", (10, y_offset), font, 0.5, COLOR_TITLE, 1)
            y_offset += line_height

            if action is not None:
                if action.action_space == "joints":
                    target_deg = np.degrees(action.data)
                    target_str = "  Target(deg): " + " ".join([f"J{i+1}={target_deg[i]:+7.2f}" for i in range(min(7, len(target_deg)))])
                    cv2.putText(info_panel, target_str, (10, y_offset), font, font_scale, COLOR_JOINT, 1)
                else:
                    target_str = f"  Target Pose: x={action.data[0]:.4f}m y={action.data[1]:.4f}m z={action.data[2]:.4f}m  rx={action.data[3]:.3f} ry={action.data[4]:.3f} rz={action.data[5]:.3f}"
                    cv2.putText(info_panel, target_str, (10, y_offset), font, font_scale, COLOR_POSE, 1)
                y_offset += line_height

                target_gripper_sdk = gripper_normalized_to_sdk(action.gripper)
                if self.config.gripper_binarize and self._last_binarized_gripper is not None:
                    target_gripper_str = f"  Target Gripper: {action.gripper:.3f} -> SDK: {target_gripper_sdk} -> Bin: {self._last_binarized_gripper}"
                else:
                    target_gripper_str = f"  Target Gripper: {action.gripper:.3f} (SDK: {target_gripper_sdk})"
                cv2.putText(info_panel, target_gripper_str, (10, y_offset), font, font_scale, COLOR_GRIPPER, 1)
            else:
                cv2.putText(info_panel, "  (No action yet)", (10, y_offset), font, font_scale, COLOR_STATUS, 1)
            y_offset += line_height

            cv2.line(info_panel, (10, y_offset + 5), (panel_width - 10, y_offset + 5), (80, 80, 80), 1)
            y_offset += 15

            # 状态栏
            status_str = (f"Frame: {self._frame_count} | FPS: {self._fps:.1f} | "
                          f"Buf: {self.ring_buffer.size} | Exec: {self._exec_count} | "
                          f"Hold: {self._hold_count} | Infer: {self._infer_count}")
            cv2.putText(info_panel, status_str, (10, y_offset), font, font_scale, COLOR_STATUS, 1)

            canvas = np.vstack([img_canvas, info_panel])
            cv2.imshow(self._vis_window_name, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.shutdown_event.set()

        except cv2.error as e:
            print(f"[警告] 可视化错误: {e}")
            print("[警告] 禁用可视化功能")
            self.config.visualize = False
        except Exception as e:
            print(f"[警告] 可视化异常: {e}")
            import traceback
            traceback.print_exc()

    def _print_stats(self):
        """打印运行统计"""
        print(f"\n{'='*60}")
        print(f"[AsyncClient] === Statistics ===")
        print(f"  Total steps: {self._frame_count}")
        print(f"  Actions executed: {self._exec_count}")
        print(f"  Hold repeats: {self._hold_count}")
        print(f"  Inferences: {self._infer_count}")
        print(f"  Avg inference time: {self._t_inf_avg:.3f}s")
        if self._t_inf_avg > 0:
            K_inf_avg = self._t_inf_avg * self.config.f_exec
            print(f"  Avg K_inf: {K_inf_avg:.1f} actions")
        if self._t_inf_ema_initialized:
            print(f"  EMA t_inf: {self._t_inf_ema:.3f}s (K_inf≈{int(self._t_inf_ema * self.config.f_exec) + 2})")
        print(f"  Avg FPS: {self._fps:.1f}")

        # Phase 2B: UDP 遥测
        if self._udp_receiver is not None:
            print(f"  --- UDP Telemetry ---")
            print(f"  UDP recv_count: {self._udp_receiver.recv_count}")
            print(f"  UDP error_count: {self._udp_receiver._error_count}")
            print(f"  UDP is_receiving: {self._udp_receiver.is_receiving(timeout=self.config.udp_recv_timeout)}")

        # Phase 2B: 录制统计
        if self.recorder is not None:
            print(f"  --- Recording ---")
            print(f"  Episodes saved: {self.recorder.num_episodes}")

        print(f"{'='*60}")


# ===== 入口 =====
def main():
    """命令行入口"""
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Async Inference Client (DAgger Phase 1)")
    parser.add_argument("--config", type=str, default=None, help="YAML 配置文件路径")
    parser.add_argument("--server", type=str, default=None, help="PolicyServer 地址 (host:port)")
    parser.add_argument("--robot-ip", type=str, default=None, help="机械臂 IP 地址")
    parser.add_argument("--robot-port", type=int, default=None, help="机械臂端口")
    parser.add_argument("--f-exec", type=float, default=None, help="固定执行频率 Hz")
    parser.add_argument("--T-inter", type=float, default=None, help="推理间隔秒数")
    parser.add_argument("--n-action-steps", type=int, default=None, help="每次截取的动作步数 (0=全部)")
    parser.add_argument("--task", type=str, default=None, help="任务指令")
    parser.add_argument("--camera-configs", type=str, default=None, help="相机配置 (YAML 格式)")
    parser.add_argument("--camera-rotations", type=str, default=None, help="相机旋转配置 (YAML 格式)")
    parser.add_argument("--dry-run", action="store_true", help="仿真模式")
    parser.add_argument("--visualize", action="store_true", help="显示可视化窗口")
    parser.add_argument("--vis-scale", type=float, default=None, help="可视化缩放比例")
    parser.add_argument("--max-steps", type=int, default=None, help="最大执行步数 (0=无限)")
    parser.add_argument("--no-hold", action="store_true", help="禁用 hold_on_empty")
    parser.add_argument("--gripper-binarize", action="store_true", help="启用夹爪二值化")
    parser.add_argument("--gripper-deadband", type=int, default=None, help="夹爪死区阈值")
    parser.add_argument("--enable-recording", action="store_true", help="启用录制")
    parser.add_argument("--repo-id", type=str, default=None, help="录制数据集 repo_id")
    parser.add_argument("--dataset-root", type=str, default=None, help="录制数据集根目录")
    args = parser.parse_args()

    # 从 YAML 加载基础配置
    yaml_config = {}
    if args.config:
        with open(args.config) as f:
            yaml_config = yaml.safe_load(f) or {}

    # 构建配置：YAML 为基础，命令行参数覆盖
    def get_val(cli_val, yaml_key, default):
        if cli_val is not None:
            return cli_val
        return yaml_config.get(yaml_key, default)

    # 相机配置
    camera_configs = yaml_config.get("camera_configs", {})
    if args.camera_configs:
        camera_configs = yaml.safe_load(args.camera_configs) or {}

    camera_rotations = yaml_config.get("camera_rotations", {})
    if args.camera_rotations:
        camera_rotations = yaml.safe_load(args.camera_rotations) or {}

    config = AsyncInferenceClientConfig(
        robot_ip=get_val(args.robot_ip, "robot_ip", "192.168.1.18"),
        robot_port=get_val(args.robot_port, "robot_port", 8080),
        server_address=get_val(args.server, "server_address", "127.0.0.1:50051"),
        frequency=int(get_val(args.f_exec, "f_exec", 30.0)),
        n_action_steps=get_val(args.n_action_steps, "n_action_steps", 0),
        camera_configs=camera_configs,
        camera_rotations=camera_rotations,
        task=get_val(args.task, "task", ""),
        dry_run=args.dry_run or yaml_config.get("dry_run", False),
        visualize=args.visualize or yaml_config.get("visualize", False),
        vis_scale=get_val(args.vis_scale, "vis_scale", 1.0),
        max_steps=get_val(args.max_steps, "max_steps", 0),
        gripper_deadband=get_val(args.gripper_deadband, "gripper_deadband", 20),
        rm_plus_baud=yaml_config.get("rm_plus_baud", 115200),
        gripper_binarize=args.gripper_binarize or yaml_config.get("gripper_binarize", False),
        gripper_binarize_threshold=yaml_config.get("gripper_binarize_threshold", 500),
        gripper_binarize_hysteresis=yaml_config.get("gripper_binarize_hysteresis", 50),
        # 异步推理参数
        f_exec=get_val(args.f_exec, "f_exec", 30.0),
        T_inter=get_val(args.T_inter, "T_inter", 0.0),
        hold_on_empty=not args.no_hold and yaml_config.get("hold_on_empty", True),
        t_inf=float(yaml_config.get("t_inf", 0.0)),
        chunk_size_threshold=int(yaml_config.get("chunk_size_threshold", 0)),
        # Phase 2B: UDP 状态配置
        enable_udp_state=yaml_config.get("enable_udp_state", False),
        udp_state_port=int(yaml_config.get("udp_state_port", 8089)),
        udp_push_cycle=int(yaml_config.get("udp_push_cycle", 2)),
        udp_recv_timeout=float(yaml_config.get("udp_recv_timeout", 1.0)),
        # Phase 2B: 录制配置
        enable_recording=args.enable_recording or yaml_config.get("enable_recording", False),
        repo_id=get_val(args.repo_id, "repo_id", "local/dagger_data"),
        dataset_root=get_val(args.dataset_root, "dataset_root", ""),
        use_videos=yaml_config.get("use_videos", True),
        image_writer_threads=int(yaml_config.get("image_writer_threads", 4)),
        image_writer_processes=int(yaml_config.get("image_writer_processes", 0)),
        recording_task=yaml_config.get("recording_task", ""),
    )

    client = AsyncInferenceClient(config)
    try:
        client.connect()
        client.run()
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()
