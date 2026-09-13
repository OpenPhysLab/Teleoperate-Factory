"""
DAgger ROS2 Node - Phase 3C: VR Human Control + Policy Inference + Recording

Unified DAgger node that replaces vr_teleop_node, adding:
- State machine: IDLE / HUMAN / POLICY modes
- VR teleoperation (HUMAN mode) via existing realman_driver_node
- Policy inference (POLICY mode) via InferenceBridge + gRPC PolicyServer
- Recording integration (Phase 3C) via DAggerRecorder

NOTE: Threading / lock safety
    This node uses rclpy single-threaded executor (the default).
    All ROS2 callbacks (subscriptions, timers, services) are dispatched
    sequentially on the same thread, so lock ordering between _vr_lock,
    _mode_lock, _state_lock, and _image_lock is safe.
    _on_image callback 只缓存原始 ROS2 Image msg 引用（<0.1ms），
    imgmsg_to_cv2 延迟到推理线程的 _build_observation() 中按需执行（通过 _image_converter 注入）。
    obs_feed_worker 只传递 state + raw msg 引用（<1ms），不持有 GIL 做图像转换。
    Recording sync callback (_on_recording_sync) only enqueues messages (<0.1ms);
    heavy work (imgmsg_to_cv2 + add_frame) runs in a dedicated _recording_worker thread.
    Observation feeding 由独立的 _obs_feed_worker 线程以 control_hz 频率持续驱动，
    不区分 HUMAN/POLICY 模式，生命周期绑定到 inference_bridge。
    Lock ordering: vr_lock -> mode_lock, vr_lock -> state_lock.
    _recording_worker accesses _mode_lock and _recorder_lock independently (no nesting).

State machine transitions:
    IDLE --[enable_control]--> POLICY (if policy enabled) or HUMAN
    POLICY --[trigger press]--> HUMAN (VR takeover)
    HUMAN --[trigger release]--> POLICY (resume inference)
    HUMAN/POLICY --[disable_control]--> IDLE

Data flow:
    /vr/{side}/pose, /vr/{side}/trigger, /vr/{side}/grip
    /camera/{cam}/color/image_raw
            |
        dagger_node
            |-- subscribes VR topics + camera images + robot state
            |-- HUMAN mode: calculates target Cartesian pose -> /rm/action_pose
            |-- POLICY mode: pops from InferenceBridge -> /rm/action_joint_state
            |-- feeds observations to InferenceBridge (background gRPC thread)
            |
        realman_driver_node (handles all robot communication)

Usage:
    ros2 run <package> dagger_node
    # or via launch file: dagger/launch/dagger.launch.py
"""
import math
import os
import signal
import sys
import time
import threading
from enum import IntEnum
from typing import Optional, Dict

import numpy as np
from scipy.spatial.transform import Rotation as R

import queue

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32, String
from sensor_msgs.msg import JointState, Image
from std_srvs.srv import Trigger
from cv_bridge import CvBridge
import message_filters
import json

# === Import vr_utils directly (pure numpy, no ROS2 deps) ===
_VR_UTILS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "vr_teleop", "spacemouse_control_arm", "ros2_realman_ws",
    "src", "realman_teleop", "realman_teleop",
)
if _VR_UTILS_DIR not in sys.path:
    sys.path.insert(0, _VR_UTILS_DIR)

from vr_utils import (
    VRCoordinateTransform,
    PoseFilter,
    VRControlState,
)

# === Import InferenceBridge ===
from dagger.core.inference_bridge import InferenceBridge, InferenceBridgeConfig

# === 诊断日志（按 session 时间戳分目录）===
from dagger.core.diag_log import create_diag_logger
_diag_log = create_diag_logger("dagger_node", print_to_stdout=True)

# === Import DAggerRecorder ===
from dagger.core.data_recorder import DAggerRecorder


# =============================================================================
# Control Mode
# =============================================================================

class ControlMode(IntEnum):
    IDLE = 0     # No control output
    HUMAN = 1    # VR teleoperation
    POLICY = 2   # Policy inference (Phase 3B placeholder)


# =============================================================================
# DAgger Node
# =============================================================================

class DAggerNode(Node):
    """
    DAgger ROS2 node with VR human control + policy inference (Phase 3B).

    Subscribes to VR topics, camera images, and robot state.
    HUMAN mode: computes target Cartesian pose -> /rm/action_pose
    POLICY mode: pops actions from InferenceBridge -> /rm/action_joint_state
    """

    def __init__(self):
        super().__init__("dagger_node")

        # ===== Parameters =====
        self.declare_parameter("ip", "192.168.1.18")
        self.declare_parameter("port", 8080)
        self.declare_parameter("dry_run", False)
        self.declare_parameter("state_topic", "/rm/state_joint_state")
        self.declare_parameter("control_hz", 50.0)
        self.declare_parameter("trigger_threshold", 0.85)
        self.declare_parameter("alpha_pos", 0.3)
        self.declare_parameter("alpha_rot", 0.3)
        self.declare_parameter("side", "right")
        self.declare_parameter("rotation_preset", "current")
        self.declare_parameter("action_pose_topic", "/rm/action_pose")
        self.declare_parameter("action_joint_topic", "/rm/action_joint_state")
        self.declare_parameter("frame_id", "rm_base")
        # Phase 3B parameters
        self.declare_parameter("enable_policy", True)
        self.declare_parameter("server_address", "127.0.0.1:50051")
        self.declare_parameter("task", "")
        self.declare_parameter("f_exec", 30.0)
        self.declare_parameter("n_action_steps", 0)
        self.declare_parameter("T_inter", 0.0)
        self.declare_parameter("t_inf", 0.0)
        self.declare_parameter("chunk_size_threshold", 0)
        self.declare_parameter("hold_on_empty", True)
        self.declare_parameter("camera_names", ["cam0", "cam1"])
        self.declare_parameter("camera_ros_names", ["cam0", "cam1"])
        self.declare_parameter("camera_rotation_names", [""])  # camera names with rotation（占位，空字符串让 ROS2 推断为 STRING_ARRAY）
        self.declare_parameter("camera_rotation_degrees", [0])  # corresponding rotation degrees（占位，让 ROS2 推断为 INTEGER_ARRAY）
        self.declare_parameter("max_joint_delta_deg", 5.0)
        self.declare_parameter("max_pose_delta_m", 0.02)
        # Phase 3C recording parameters
        self.declare_parameter("enable_recording", False)
        self.declare_parameter("repo_id", "local/dagger_realman_001")
        self.declare_parameter("dataset_root", "/home/ubuntu/Desktop/Workspace/lerobot_policy_deploy/data/dagger/")
        self.declare_parameter("recording_task", "pick and place")
        self.declare_parameter("recording_fps", 30)
        self.declare_parameter("use_videos", True)
        self.declare_parameter("image_writer_threads", 4)
        self.declare_parameter("max_episodes", 0)  # 0 = unlimited
        # PolicyServer 信息（用于 status 透传，不影响推理逻辑）
        self.declare_parameter("policy_type", "")
        self.declare_parameter("pretrained_path", "")
        self.declare_parameter("policy_server_extra_args", [""])

        # Read parameters
        self.ip = self.get_parameter("ip").get_parameter_value().string_value
        self.port = self.get_parameter("port").get_parameter_value().integer_value
        self.dry_run = self.get_parameter("dry_run").get_parameter_value().bool_value
        self.state_topic = self.get_parameter("state_topic").get_parameter_value().string_value
        self.control_hz = self.get_parameter("control_hz").get_parameter_value().double_value
        self.trigger_threshold = self.get_parameter("trigger_threshold").get_parameter_value().double_value
        self.alpha_pos = self.get_parameter("alpha_pos").get_parameter_value().double_value
        self.alpha_rot = self.get_parameter("alpha_rot").get_parameter_value().double_value
        self.side = self.get_parameter("side").get_parameter_value().string_value
        self.rotation_preset = self.get_parameter("rotation_preset").get_parameter_value().string_value
        self.action_pose_topic = self.get_parameter("action_pose_topic").get_parameter_value().string_value
        self.action_joint_topic = self.get_parameter("action_joint_topic").get_parameter_value().string_value
        self.frame_id = self.get_parameter("frame_id").get_parameter_value().string_value
        # Phase 3B
        self.enable_policy = self.get_parameter("enable_policy").get_parameter_value().bool_value
        self.server_address = self.get_parameter("server_address").get_parameter_value().string_value
        self.task = self.get_parameter("task").get_parameter_value().string_value
        self.f_exec = self.get_parameter("f_exec").get_parameter_value().double_value
        self.n_action_steps = self.get_parameter("n_action_steps").get_parameter_value().integer_value
        self.T_inter = self.get_parameter("T_inter").get_parameter_value().double_value
        self.t_inf = self.get_parameter("t_inf").get_parameter_value().double_value
        self.chunk_size_threshold = self.get_parameter("chunk_size_threshold").get_parameter_value().integer_value
        self.hold_on_empty = self.get_parameter("hold_on_empty").get_parameter_value().bool_value
        self.camera_names = self.get_parameter("camera_names").get_parameter_value().string_array_value
        self.camera_ros_names = self.get_parameter("camera_ros_names").get_parameter_value().string_array_value
        # 构建 camera_rotations dict（inference_key → rotation_deg）
        rot_names = self.get_parameter("camera_rotation_names").get_parameter_value().string_array_value
        rot_degrees = self.get_parameter("camera_rotation_degrees").get_parameter_value().integer_array_value
        self._camera_rotations: Dict[str, int] = {}
        # 过滤掉占位默认值（空字符串）
        rot_names = [n for n in rot_names if n]
        rot_degrees = rot_degrees[:len(rot_names)]
        if len(rot_names) == len(rot_degrees):
            self._camera_rotations = dict(zip(rot_names, rot_degrees))
        # 构建 ROS2 话题名 → 推理 key 的映射
        if len(self.camera_ros_names) == len(self.camera_names):
            self._ros_to_inference_key = dict(zip(self.camera_ros_names, self.camera_names))
        else:
            self.get_logger().warn(
                f"camera_ros_names({self.camera_ros_names}) 与 camera_names({self.camera_names}) 长度不一致，使用 camera_names 作为话题名"
            )
            self._ros_to_inference_key = dict(zip(self.camera_names, self.camera_names))
        self.max_joint_delta_deg = self.get_parameter("max_joint_delta_deg").get_parameter_value().double_value
        self.max_pose_delta_m = self.get_parameter("max_pose_delta_m").get_parameter_value().double_value
        # Phase 3C
        self.enable_recording = self.get_parameter("enable_recording").get_parameter_value().bool_value
        self.repo_id = self.get_parameter("repo_id").get_parameter_value().string_value
        self.dataset_root = self.get_parameter("dataset_root").get_parameter_value().string_value
        self.recording_task = self.get_parameter("recording_task").get_parameter_value().string_value
        self.recording_fps = self.get_parameter("recording_fps").get_parameter_value().integer_value
        self.use_videos = self.get_parameter("use_videos").get_parameter_value().bool_value
        self.image_writer_threads = self.get_parameter("image_writer_threads").get_parameter_value().integer_value
        self.max_episodes = self.get_parameter("max_episodes").get_parameter_value().integer_value
        # PolicyServer 信息
        self.policy_type = self.get_parameter("policy_type").get_parameter_value().string_value
        self.pretrained_path = self.get_parameter("pretrained_path").get_parameter_value().string_value
        try:
            self._policy_server_extra_args = [
                a for a in self.get_parameter("policy_server_extra_args").get_parameter_value().string_array_value
                if a  # 过滤掉占位空字符串
            ]
        except Exception:
            self._policy_server_extra_args = []

        # ===== State machine =====
        self._mode = ControlMode.IDLE
        self._mode_lock = threading.Lock()

        # ===== VR control components =====
        self.coord_transform = VRCoordinateTransform(
            side=self.side,
            rotation_preset=self.rotation_preset,
        )
        self.pose_filter = PoseFilter(self.alpha_pos, self.alpha_rot)
        self.vr_state = VRControlState()
        self._vr_lock = threading.Lock()

        # Current VR input
        self._current_pose: Optional[PoseStamped] = None
        self._current_trigger: float = 0.0
        self._current_grip: float = 1.0  # Initial: open
        self._vr_last_update: float = 0.0

        # Gripper toggle state
        self._gripper_closed = False  # False=open(1.0), True=closed(0.0)
        self._last_grip_value = 0.0   # For edge detection

        # ===== Robot state cache (from /rm/state_joint_state) =====
        self._cached_state: Optional[JointState] = None
        self._cached_eef_pose: Optional[np.ndarray] = None
        self._state_lock = threading.Lock()

        # ===== Camera image cache =====
        self._cv_bridge = CvBridge()
        self._cached_image_msgs: Dict[str, Image] = {}  # 缓存原始 ROS2 Image msg，延迟转换
        self._image_lock = threading.Lock()

        # ===== Safety: last executed joint positions =====
        self._last_executed_joints: Optional[np.ndarray] = None

        # ===== InferenceBridge (Phase 3B) =====
        self._inference_bridge: Optional[InferenceBridge] = None
        if self.enable_policy:
            bridge_config = InferenceBridgeConfig(
                server_address=self.server_address,
                task=self.task,
                f_exec=self.f_exec,
                n_action_steps=self.n_action_steps,
                T_inter=self.T_inter,
                t_inf=self.t_inf,
                chunk_size_threshold=self.chunk_size_threshold,
                hold_on_empty=self.hold_on_empty,
            )
            self._inference_bridge = InferenceBridge(bridge_config)
            self._inference_bridge.set_image_converter(self._convert_raw_image_msgs)

            # === n_action_steps 双重截取防护 (OpenPI) ===
            self._check_openpi_double_truncation()

        # ===== Recording (Phase 3C) =====
        self._recorder: Optional[DAggerRecorder] = None
        self._recorder_initialized = False
        self._recorder_init_failed = False
        self._recorder_init_in_progress = False  # 防止多次触发后台初始化
        self._recorder_lock = threading.Lock()
        self._episode_active = False
        self._episode_paused = False
        self._episode_saving = False  # 异步保存进行中标志
        self._last_policy_action: Optional[np.ndarray] = None
        # 消息同步录制：频率限制
        self._last_recording_ts: Optional[float] = None
        self._min_frame_interval = 0.5 / self.recording_fps if self.recording_fps > 0 else 0.0
        self._recording_frame_count = 0

        # ===== 录制诊断计数器 =====
        self._sync_triggered = 0
        self._sync_throttled = 0
        self._sync_queue_full = 0
        self._sync_recorded = 0
        self._sync_stats_last_time = 0.0

        # ===== 录制异步化：队列 + 后台线程 =====
        self._recording_queue = queue.Queue(maxsize=5)
        self._recording_stop = threading.Event()
        # Fix C: 旧帧丢弃计数
        self._sync_stale_dropped = 0
        # Fix B: worker 每帧处理结果计数器
        self._worker_processed = 0
        self._worker_skipped = 0
        self._worker_errored = 0
        self._worker_stale = 0
        self._worker_stats_last_time = 0.0

        # ===== 丢帧诊断变量 =====
        self._last_sync_ts: float = 0.0  # 上次 recording sync 回调时间
        self._last_camera_ts: Dict[str, float] = {}  # 每个 camera 上次帧时间


        # ===== Subscribe VR topics =====
        pose_topic = f"/vr/{self.side}/pose"
        trigger_topic = f"/vr/{self.side}/trigger"
        grip_topic = f"/vr/{self.side}/grip"

        self.pose_sub = self.create_subscription(PoseStamped, pose_topic, self._on_vr_pose, 10)
        self.trigger_sub = self.create_subscription(Float32, trigger_topic, self._on_vr_trigger, 10)
        self.grip_sub = self.create_subscription(Float32, grip_topic, self._on_vr_grip, 10)

        # ===== Subscribe robot state =====
        self.state_sub = self.create_subscription(
            JointState, self.state_topic, self._on_arm_state, 10,
        )
        self.get_logger().info(f"Subscribing to state: {self.state_topic}")

        # ===== Subscribe camera images =====
        # 用 ROS2 话题名订阅，回调中用推理 key 存储
        self._image_subs = []
        for ros_name, inference_key in self._ros_to_inference_key.items():
            topic = f"/camera/{ros_name}/color/image_raw"
            sub = self.create_subscription(
                Image, topic,
                lambda msg, name=inference_key: self._on_image(msg, name),
                10,
            )
            self._image_subs.append(sub)
            self.get_logger().info(f"Subscribing to camera: {topic} → key={inference_key}")

        # ===== 消息同步录制订阅器（仅在 enable_recording=True 时创建） =====
        self._recording_sync = None
        if self.enable_recording:
            # 独立的 state subscriber（不与控制循环的 _on_arm_state 共享）
            self._sync_state_sub = message_filters.Subscriber(
                self, JointState, self.state_topic, qos_profile=10,
            )
            # 独立的 camera image subscribers（用 ROS2 话题名）
            self._sync_image_subs = []
            for ros_name in self._ros_to_inference_key.keys():
                topic = f"/camera/{ros_name}/color/image_raw"
                img_sub = message_filters.Subscriber(
                    self, Image, topic, qos_profile=10,
                )
                self._sync_image_subs.append(img_sub)

            # ApproximateTimeSynchronizer: state + 所有 camera images
            sync_inputs = [self._sync_state_sub] + self._sync_image_subs
            self._recording_sync = message_filters.ApproximateTimeSynchronizer(
                sync_inputs,
                queue_size=30,
                slop=0.05,
                allow_headerless=False,
            )
            self._recording_sync.registerCallback(self._on_recording_sync)
            self.get_logger().info(
                f"Recording sync subscriber created: state + {len(self.camera_names)} cameras, "
                f"queue_size=30, slop=0.05s, recording_fps={self.recording_fps}"
            )

        # ===== Publishers =====
        self.action_pose_pub = self.create_publisher(JointState, self.action_pose_topic, 10)
        self.action_joint_pub = self.create_publisher(JointState, self.action_joint_topic, 10)
        self.status_pub = self.create_publisher(String, "/dagger/status", 10)

        # ===== Services =====
        self.enable_control_srv = self.create_service(
            Trigger, "/dagger/enable_control", self._handle_enable_control,
        )
        self.disable_control_srv = self.create_service(
            Trigger, "/dagger/disable_control", self._handle_disable_control,
        )
        self.set_human_srv = self.create_service(
            Trigger, "/dagger/set_human", self._handle_set_human,
        )
        self.set_policy_srv = self.create_service(
            Trigger, "/dagger/set_policy", self._handle_set_policy,
        )
        self.set_idle_srv = self.create_service(
            Trigger, "/dagger/set_idle", self._handle_set_idle,
        )
        # Phase 3C: recording services
        self.start_episode_srv = self.create_service(
            Trigger, "/dagger/start_episode", self._handle_start_episode,
        )
        self.stop_episode_srv = self.create_service(
            Trigger, "/dagger/stop_episode", self._handle_stop_episode,
        )
        self.pause_episode_srv = self.create_service(
            Trigger, "/dagger/pause_episode", self._handle_pause_episode,
        )
        self.discard_episode_srv = self.create_service(
            Trigger, "/dagger/discard_episode", self._handle_discard_episode,
        )
        self.new_episode_srv = self.create_service(
            Trigger, "/dagger/new_episode", self._handle_new_episode,
        )
        # Phase 4: 统一会话控制
        self.start_session_srv = self.create_service(
            Trigger, "/dagger/start_session", self._handle_start_session,
        )
        self.stop_session_srv = self.create_service(
            Trigger, "/dagger/stop_session", self._handle_stop_session,
        )
        self.pause_session_srv = self.create_service(
            Trigger, "/dagger/pause_session", self._handle_pause_session,
        )
        self.resume_session_srv = self.create_service(
            Trigger, "/dagger/resume_session", self._handle_resume_session,
        )

        # ===== Driver follow control (service clients) =====
        self.driver_enable_follow_client = self.create_client(Trigger, "/driver/enable_follow")
        self.driver_disable_follow_client = self.create_client(Trigger, "/driver/disable_follow")

        # ===== Control timer =====
        period = 1.0 / self.control_hz if self.control_hz > 0 else 0.02
        self.timer = self.create_timer(period, self._execution_tick)


        # ===== Obs feed worker thread（独立线程持续喂 observation 给 bridge）=====
        self._obs_feed_thread: Optional[threading.Thread] = None
        self._obs_feed_stop = threading.Event()
        # obs 诊断计数器（在 __init__ 中初始化，避免 hasattr 竞态）
        self._obs_feed_count = 0
        self._obs_skip_state_count = 0
        self._obs_skip_image_count = 0

        # ===== Policy execution thread (async_client style while+sleep) =====
        self._policy_exec_thread: Optional[threading.Thread] = None
        # Per-thread stop Event: each exec loop thread gets its own Event to avoid
        # race conditions where _start_policy_exec_loop().clear() erases the stop
        # signal for a still-running old thread.  _policy_exec_stop always points
        # to the *current* (most recently started) thread's Event.
        self._policy_exec_stop = threading.Event()
        self._policy_exec_stop_time = 0.0  # monotonic timestamp when stop flag was set
        self._session_paused = False  # True when session is paused (POLICY mode but not executing)
        self._clear_buffer_on_exec_start = False  # Flag: clear buffer before first action execution

        # ===== Motion Gating: 防止 warmup/录制未就绪时下发 action =====
        self._motion_gate = threading.Event()  # 初始 unset，action 不下发
        self._recording_ready = False  # 录制链路就绪标志

        # ===== Stats =====
        self.stats_timer = self.create_timer(0.5, self._publish_status)
        self.log_timer = self.create_timer(10.0, self._log_stats)
        self._cmd_count = 0
        self._exec_count = 0
        self._safety_reject_count = 0
        # 控制频率监控：滑动窗口记录每次 action 发布的时间戳
        self._cmd_timestamps: list[float] = []
        self._cmd_ts_lock = threading.Lock()
        self._CMD_WINDOW_SEC = 2.0  # 2秒滑动窗口

        # ===== PolicyServer probe (IDLE 模式下后台线程检测 Server 是否就绪) =====
        self._server_reachable = False
        self._server_probe_stop = threading.Event()
        self._server_probe_thread = None
        if self.enable_policy:
            self._server_probe_thread = threading.Thread(
                target=self._probe_server_loop, daemon=True,
            )
            self._server_probe_thread.start()

        # ===== Recorder 预初始化（启动时后台线程，避免第一次 start_session 等待 4+ 秒） =====
        self._recording_worker_thread: Optional[threading.Thread] = None
        if self.enable_recording:
            self._recorder_init_in_progress = True
            threading.Thread(
                target=self._preinit_recorder,
                daemon=True,
                name="RecorderPreInit"
            ).start()
            # 启动录制后台线程
            self._recording_worker_thread = threading.Thread(
                target=self._recording_worker,
                daemon=True,
                name="RecordingWorker",
            )
            self._recording_worker_thread.start()

        self.get_logger().info(f"DAgger node started (Phase 3C: VR + Policy + Recording)")
        self.get_logger().info(f"  Robot IP: {self.ip}, Control Hz: {self.control_hz}")
        self.get_logger().info(f"  VR side: {self.side}, trigger threshold: {self.trigger_threshold}")
        self.get_logger().info(f"  Pose topic: {self.action_pose_topic}")
        self.get_logger().info(f"  Joint topic: {self.action_joint_topic}")
        self.get_logger().info(f"  Policy enabled: {self.enable_policy}")
        if self.enable_policy:
            self.get_logger().info(f"  Server: {self.server_address}")
            self.get_logger().info(f"  Cameras: {self.camera_names}")
        self.get_logger().info(f"  Recording enabled: {self.enable_recording}")
        if self.enable_recording:
            self.get_logger().info(f"  Repo: {self.repo_id}, root: {self.dataset_root}")
            max_ep_str = str(self.max_episodes) if self.max_episodes > 0 else "unlimited"
            self.get_logger().info(f"  Recording FPS: {self.recording_fps}, max_episodes: {max_ep_str}")
        self.get_logger().info(f"  Safety: max_joint_delta={self.max_joint_delta_deg}deg, max_pose_delta={self.max_pose_delta_m}m (reserved, not enforced)")
        self.get_logger().info(f"  Mode: IDLE (call /dagger/enable_control to start)")

    # =========================================================================
    # OpenPI n_action_steps 双重截取防护
    # =========================================================================

    def _check_openpi_double_truncation(self):
        """检测 OpenPI + n_action_steps > 0 的双重截取风险。

        PolicyServer 端的 n_action_steps 截取是"盲截取"（不知道下游 f_exec 和 t_inf），
        会与 InferenceBridge 的自适应 K_inf 裁剪冲突，导致有效动作数大幅减少甚至系统不可行。
        """
        if self.policy_type != "openpi":
            return

        # 解析 extra_args 中的 --n_action_steps
        n_steps_extra = 0
        for i, arg in enumerate(self._policy_server_extra_args):
            if arg.startswith("--n_action_steps="):
                try:
                    n_steps_extra = int(arg.split("=", 1)[1])
                except ValueError:
                    pass
            elif arg == "--n_action_steps" and i + 1 < len(self._policy_server_extra_args):
                try:
                    n_steps_extra = int(self._policy_server_extra_args[i + 1])
                except ValueError:
                    pass

        if n_steps_extra > 0:
            self.get_logger().warn(
                f"[OpenPI 双重截取风险] extra_args 中 --n_action_steps={n_steps_extra} > 0。"
                f"PolicyServer 端会先截取 ActionChunk 到 {n_steps_extra} 步，"
                f"InferenceBridge 再做 K_inf 裁剪，可能导致有效动作数为负、系统不可行。"
                f"建议设置 --n_action_steps=0，让 InferenceBridge 的自适应 K_inf 裁剪处理所有截取逻辑。"
            )

        if self.n_action_steps > 0:
            self.get_logger().warn(
                f"[OpenPI 双重截取风险] 顶层 n_action_steps={self.n_action_steps} > 0。"
                f"OpenPI 模型通常输出 action_horizon=50 的长 chunk，"
                f"建议设置 n_action_steps=0，让 InferenceBridge 的自适应 K_inf 裁剪处理所有截取逻辑。"
            )

    # =========================================================================
    # Robot state callback
    # =========================================================================

    def _on_arm_state(self, msg: JointState):
        """
        Receive robot state from realman_driver_node.

        State format (14-dim):
            position[0:7]  = joint angles (rad)
            position[7]    = gripper openness (0-1)
            position[8:14] = EEF pose [x, y, z, rx, ry, rz] (m/rad)
        """
        # === DIAG-COMPARE: 每帧 ROS2 state 到达时间 + gripper 值 ===
        if not hasattr(self, '_on_state_count'):
            self._on_state_count = 0
            self._on_state_last_gripper = None
            self._on_state_mono = time.monotonic()
        self._on_state_count += 1
        gripper_val = msg.position[7] if len(msg.position) > 7 else None
        _now = time.monotonic()
        _dt_ms = (_now - self._on_state_mono) * 1000
        self._on_state_mono = _now
        _g = f"{gripper_val:.4f}" if gripper_val is not None else "N/A"
        _diag_log(f"[ROS-STATE] #{self._on_state_count}: gripper={_g}, dt={_dt_ms:.1f}ms")
        if gripper_val is not None:
            self._on_state_last_gripper = gripper_val

        with self._state_lock:
            self._cached_state = msg
            if len(msg.position) >= 14:
                self._cached_eef_pose = np.array(msg.position[8:14], dtype=np.float64)
            elif self._cached_eef_pose is None:
                self.get_logger().warn(
                    "State topic missing EEF pose (need 14-dim), VR control may be inaccurate",
                    throttle_duration_sec=10.0,
                )

    def _get_current_arm_state(self) -> Optional[tuple]:
        """
        Get current arm state from cache.

        Returns: (joint_rad, eef_pose, gripper) or None
        """
        with self._state_lock:
            if self._cached_state is None:
                return None
            if len(self._cached_state.position) < 7:
                return None
            joint_rad = np.array(self._cached_state.position[:7])
            gripper = self._cached_state.position[7] if len(self._cached_state.position) > 7 else 0.0
            eef_pose = self._cached_eef_pose
            return joint_rad, eef_pose, gripper

    def _get_state_14d(self) -> Optional[np.ndarray]:
        """Get current 14-dim state vector for inference observation."""
        with self._state_lock:
            if self._cached_state is None:
                return None
            pos = self._cached_state.position
            if len(pos) < 14:
                return None
            state_14d = np.array(pos[:14], dtype=np.float64)
            return state_14d

    # =========================================================================
    # Camera image callback
    # =========================================================================

    def _on_image(self, msg: Image, cam_name: str):
        """
        Receive camera image, cache raw ROS2 Image msg（延迟转换）。

        不在回调中做 imgmsg_to_cv2，避免阻塞 executor（每相机 3-5ms）。
        转换延迟到 _get_cached_images() 被调用时按需执行。

        IMPORTANT: 不在此处旋转图像！camera_node 已经根据 teleop_params.yaml 的 rotate_deg
        参数旋转过了（cam0: 180°）。
        """
        # [REC-DIAG] camera 帧间隔监控
        now_mono = time.monotonic()
        last_ts = self._last_camera_ts.get(cam_name, 0.0)
        if last_ts > 0:
            camera_gap_ms = (now_mono - last_ts) * 1000.0
            if camera_gap_ms > 100.0:
                self.get_logger().warn(
                    f"[REC-DIAG] camera_gap={camera_gap_ms:.1f}ms, camera={cam_name}"
                )
        self._last_camera_ts[cam_name] = now_mono

        with self._image_lock:
            self._cached_image_msgs[cam_name] = msg

    def _get_cached_images(self) -> Dict[str, np.ndarray]:
        """
        从缓存的原始 ROS2 Image msg 按需转换为 numpy array。

        IMPORTANT: 返回的图像顺序必须与 camera_names 参数一致，
        确保与 async_inference_client 的顺序相同（cam0_rgb, cam1_rgb）。
        ROS2 callback 的到达顺序是不确定的，不能依赖 dict 的插入顺序。

        imgmsg_to_cv2 在此处执行（而非 _on_image 回调中），
        避免阻塞 ROS2 executor。此方法在 exec loop 独立线程中调用。
        """
        with self._image_lock:
            # 取出原始 msg 引用（不 copy msg，只持有引用）
            msgs_snapshot = {}
            for cam_name in self.camera_names:
                if cam_name in self._cached_image_msgs:
                    msgs_snapshot[cam_name] = self._cached_image_msgs[cam_name]

        # 在锁外做 imgmsg_to_cv2（CPU 密集操作，不阻塞其他线程访问 _image_lock）
        ordered_images = {}
        for cam_name, msg in msgs_snapshot.items():
            try:
                img = self._cv_bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
                ordered_images[cam_name] = img
            except Exception as e:
                self.get_logger().warn(
                    f"Failed to convert cached image from {cam_name}: {e}",
                    throttle_duration_sec=5.0,
                )

        # 诊断：打印图像顺序（仅前 3 次）
        if not hasattr(self, '_image_order_logged'):
            self._image_order_logged = 0
        if self._image_order_logged < 3:
            self._image_order_logged += 1
            _diag_log(f"[IMAGE-ORDER] camera_names={self.camera_names}, "
                     f"ordered_keys={list(ordered_images.keys())}, "
                     f"cached_keys={list(msgs_snapshot.keys())}")

        return ordered_images

    def _get_raw_image_msgs(self) -> Dict[str, 'Image']:
        """从 _cached_image_msgs 取 raw ROS2 Image msg 引用（<0.1ms）。

        轻量模式下 obs_feed_worker 调用此方法，不做 imgmsg_to_cv2 转换。
        按 camera_names 顺序返回，确保与推理 key 一致。
        """
        with self._image_lock:
            raw_msgs = {}
            for cam_name in self.camera_names:
                if cam_name in self._cached_image_msgs:
                    raw_msgs[cam_name] = self._cached_image_msgs[cam_name]
            return raw_msgs

    def _convert_raw_image_msgs(self, raw_msgs: Dict) -> Dict[str, np.ndarray]:
        """将 raw ROS2 Image msg 转换为 numpy RGB array。

        由 InferenceBridge 在推理线程中调用（通过 set_image_converter 注入）。
        封装 cv_bridge.imgmsg_to_cv2 调用逻辑。

        Args:
            raw_msgs: {cam_name: ROS2 Image msg}

        Returns:
            {cam_name: numpy_rgb array}
        """
        images = {}
        for cam_name, msg in raw_msgs.items():
            try:
                img = self._cv_bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
                images[cam_name] = img
            except Exception as e:
                self.get_logger().warn(
                    f"[_convert_raw_image_msgs] {cam_name} 转换失败: {e}",
                    throttle_duration_sec=5.0,
                )
        return images

    # =========================================================================
    # VR input callbacks
    # =========================================================================

    def _on_vr_pose(self, msg: PoseStamped):
        """Receive VR controller pose."""
        with self._vr_lock:
            self._current_pose = msg
            self._vr_last_update = time.time()
            self.vr_state.current_vr_pos = np.array([
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
            ])
            self.vr_state.current_vr_quat = np.array([
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ])
            self.vr_state.last_update_time = time.time()

    def _on_vr_trigger(self, msg: Float32):
        """
        Receive trigger value with edge detection and state machine transitions.

        State machine:
        - POLICY mode + trigger press -> transition to HUMAN (VR takeover)
        - HUMAN mode + trigger press -> activate VR control (anchor point)
        - HUMAN mode + trigger release -> transition to POLICY (if enabled)
        """
        with self._vr_lock:
            old_trigger = self._current_trigger
            self._current_trigger = msg.data
            self.vr_state.current_trigger = msg.data

            was_active = old_trigger >= self.trigger_threshold
            is_active = msg.data >= self.trigger_threshold

            with self._mode_lock:
                mode = self._mode

            # 诊断：记录 trigger 边缘事件（按下/松开时记录，非持续按住）
            if was_active != is_active:
                _diag_log(
                    f"[VR-TRIGGER] {'PRESS' if is_active else 'RELEASE'}: "
                    f"value={msg.data:.3f}, threshold={self.trigger_threshold}, "
                    f"mode={mode.name}"
                )

            if mode == ControlMode.IDLE:
                if was_active != is_active:
                    _diag_log(f"[VR-TRIGGER] IGNORED: mode=IDLE, trigger={'PRESS' if is_active else 'RELEASE'}")
                return

            if not was_active and is_active:
                # Trigger pressed
                _t_trigger = time.monotonic()
                if mode == ControlMode.POLICY:
                    self._transition_to_human()
                    _diag_log(
                        f"[TRIGGER-DIAG] POLICY->HUMAN total={( time.monotonic()-_t_trigger)*1000:.1f}ms"
                    )
                elif mode == ControlMode.HUMAN:
                    self._activate_control()
                    _diag_log(
                        f"[TRIGGER-DIAG] HUMAN re-activate total={( time.monotonic()-_t_trigger)*1000:.1f}ms"
                    )
            elif was_active and not is_active:
                # Trigger released
                if mode == ControlMode.HUMAN and self.enable_policy:
                    self._transition_to_policy()
                elif mode == ControlMode.HUMAN:
                    self._deactivate_control()

    def _on_vr_grip(self, msg: Float32):
        """
        Receive grip value with toggle control.

        Grip button (side grip):
        - Press once -> gripper closes (0.0)
        - Press again -> gripper opens (1.0)
        - Edge detection prevents repeated triggers
        """
        with self._vr_lock:
            current_grip = msg.data

            # Edge detection: from not pressed (<0.5) to pressed (>=0.5)
            if self._last_grip_value < 0.5 and current_grip >= 0.5:
                self._gripper_closed = not self._gripper_closed
                status = "closed" if self._gripper_closed else "open"
                self.get_logger().info(f"Gripper toggle: {status}")

            self._last_grip_value = current_grip
            self._current_grip = 0.0 if self._gripper_closed else 1.0
            self.vr_state.current_grip = self._current_grip

    # =========================================================================
    # VR control activation/deactivation
    # =========================================================================

    def _activate_control(self):
        """Activate VR control (trigger pressed)."""
        _t0 = time.monotonic()
        self.vr_state.is_active = True

        # Record VR initial pose
        self.vr_state.init_vr_pos = self.vr_state.current_vr_pos.copy()
        self.vr_state.init_vr_quat = self.vr_state.current_vr_quat.copy()
        _t1 = time.monotonic()

        # Get current arm state — 检查时效性，过期则等待刷新
        arm_state = self._get_current_arm_state()
        _state_stale = False
        if arm_state is not None:
            # 检查 _cached_state 的时效性（_on_arm_state 每帧更新 _on_state_mono）
            with self._state_lock:
                state_age_ms = (time.monotonic() - getattr(self, '_on_state_mono', 0.0)) * 1000
            if state_age_ms > 50.0:
                _state_stale = True
                # 等待最多 100ms 获取新的 arm state
                _wait_deadline = time.monotonic() + 0.1
                _initial_count = getattr(self, '_on_state_count', 0)
                while time.monotonic() < _wait_deadline:
                    if getattr(self, '_on_state_count', 0) > _initial_count:
                        arm_state = self._get_current_arm_state()
                        break
                    time.sleep(0.005)
                else:
                    # 超时，使用当前缓存值
                    self.get_logger().warn(
                        f"[_activate_control] arm state 过期（{state_age_ms:.0f}ms），等待刷新超时，使用缓存值"
                    )

        _t2 = time.monotonic()
        # 用于 filter 预热的初始位姿
        _init_pos = None
        _init_quat = None
        if arm_state is not None:
            joint_rad, eef_pose, gripper = arm_state
            if eef_pose is not None:
                self.vr_state.init_arm_pos = np.array(eef_pose[:3])
                # Euler to quaternion (intrinsic xyz, matching Realman SDK RPY convention)
                euler_rad = eef_pose[3:6]
                rot = R.from_euler('xyz', euler_rad)
                self.vr_state.init_arm_quat = rot.as_quat()

                # 保存用于 filter 预热
                _init_pos = self.vr_state.init_arm_pos
                _init_quat = self.vr_state.init_arm_quat

                self.vr_state.is_initialized = True
                self.get_logger().info(
                    f"VR control activated, init pos: {self.vr_state.init_arm_pos}"
                )
            else:
                self.get_logger().warn("Cannot get EEF pose")
                self.vr_state.is_initialized = False
        else:
            self.get_logger().warn("Cannot get arm state, VR control may be inaccurate")
            self.vr_state.is_initialized = False

        # Reset filter（用当前 arm pose 预填充，避免冷启动跳变）
        self.pose_filter.reset(initial_pos=_init_pos, initial_quat=_init_quat)
        _t3 = time.monotonic()

        _diag_log(
            f"[ACTIVATE-DIAG] vr_copy={(_t1-_t0)*1000:.1f}ms, "
            f"arm_state={(_t2-_t1)*1000:.1f}ms (stale={_state_stale}), "
            f"filter_reset={(_t3-_t2)*1000:.1f}ms, "
            f"total={(_t3-_t0)*1000:.1f}ms"
        )

    def _deactivate_control(self):
        """Deactivate VR control (trigger released, staying in HUMAN mode)."""
        self.vr_state.is_active = False
        self.vr_state.is_initialized = False
        self.get_logger().info("VR control deactivated")

    def _transition_to_human(self):
        """
        Transition from POLICY to HUMAN mode (VR takeover).

        推理线程持续运行（shadow mode），不暂停、不清空 buffer。
        shadow tick 会在 HUMAN execution tick 中消费 buffer 更新 _last_policy_action。

        关键：必须同步停止 exec loop（join），确保 exec loop 完全停止后
        才切换模式并开始 VR pose 发布。否则 exec loop 和 _tick_human
        会同时向 driver 发送 joint action 和 pose action，导致机械臂行为混乱。
        """
        _t0 = time.monotonic()

        # 同步停止 exec loop（与 experiment 分支一致）
        # 必须 join，确保 exec loop 线程完全退出后再切换模式
        self._stop_policy_exec_loop()
        _t1 = time.monotonic()

        # ---- 用当前机械臂夹爪状态初始化 VR gripper，避免模式切换时夹爪跳变 ----
        # 滞后区间 [0.3, 0.7]：只在明确远离阈值时才改变状态，中间区域保持当前状态不变。
        # 避免 policy 输出 gripper≈0.5 时每次切换都翻转映射。
        _prev_grip = self._current_grip
        _prev_closed = self._gripper_closed
        current_gripper = getattr(self, '_on_state_last_gripper', None)
        if current_gripper is not None:
            if current_gripper <= 0.3:
                # 明确关闭
                self._gripper_closed = True
                self._current_grip = 0.0
            elif current_gripper >= 0.7:
                # 明确打开
                self._gripper_closed = False
                self._current_grip = 1.0
            else:
                # 滞后区间 (0.3, 0.7)：保持当前 VR 侧状态不变
                pass
            self.vr_state.current_grip = self._current_grip
            _diag_log(
                f"[GRIP-SYNC] POLICY->HUMAN: arm_gripper={current_gripper:.4f}, "
                f"gripper_closed={self._gripper_closed} (was {_prev_closed}), "
                f"current_grip={self._current_grip:.1f} (was {_prev_grip:.1f}), "
                f"zone={'CLOSED' if current_gripper <= 0.3 else 'OPEN' if current_gripper >= 0.7 else 'HYSTERESIS'}"
            )
        else:
            _diag_log(f"[GRIP-SYNC] POLICY->HUMAN: no arm gripper state, keeping current_grip={_prev_grip:.1f}")

        with self._mode_lock:
            self._mode = ControlMode.HUMAN
        _t2 = time.monotonic()

        # 注意：不调用 pause() / clear_buffer()，推理线程持续运行
        # shadow tick (_tick_shadow) 会在 HUMAN tick 后消费 buffer

        # Activate VR control (set anchor point)
        self._activate_control()

        # 立即发一帧 movep，消除等待下一个 timer tick 的空窗期
        # （_activate_control 已设置 is_initialized=True 和锚点）
        if self.vr_state.is_initialized:
            try:
                # 调用方 _on_vr_trigger() 已持有 _vr_lock，不可再次获取（Lock 不可重入）
                target_pose = self._calculate_target_pose()
                if target_pose is not None:
                    self._publish_pose_action(target_pose)
                    _diag_log("[TRANSITION-IMMEDIATE] 立即发送首帧 movep，消除 timer tick 等待")
            except Exception as _e:
                _diag_log(f"[TRANSITION-IMMEDIATE] 首帧 movep 失败: {_e}")

        _t3 = time.monotonic()

        _diag_log(
            f"[TRANSITION-DIAG] POLICY->HUMAN: "
            f"stop_flag={(_t1-_t0)*1000:.1f}ms, "
            f"mode_lock={(_t2-_t1)*1000:.1f}ms, "
            f"activate={(_t3-_t2)*1000:.1f}ms, "
            f"total={(_t3-_t0)*1000:.1f}ms"
        )
        self._publish_status()

    def _transition_to_policy(self):
        """
        Transition from HUMAN to POLICY mode (resume inference).

        Deactivates VR control, clears buffer（丢弃 HUMAN 期间积累的过时 action）。
        不调用 resume()，因为推理线程在 HUMAN 模式下未暂停。
        """
        _t0 = time.monotonic()

        self.vr_state.is_active = False
        self.vr_state.is_initialized = False

        # 清空 buffer：HUMAN 期间的 action 基于专家操作时的状态，不适合直接执行
        # 注意：用 clear()（保留 last_popped 做 hold 过渡）而非 reset()
        # 推理线程在 HUMAN 模式下持续运行（shadow mode），不调用 resume()
        _buf_before = 0
        _has_last_popped = False
        if self._inference_bridge is not None:
            _buf_before = self._inference_bridge.ring_buffer.size
            _has_last_popped = self._inference_bridge.ring_buffer.last_popped is not None
            self._inference_bridge.clear_buffer()
        _t1 = time.monotonic()

        self._last_policy_action = None

        # Reset safety tracking for smooth transition
        self._last_executed_joints = None

        with self._mode_lock:
            self._mode = ControlMode.POLICY
        _t2 = time.monotonic()

        # 记录切换时间戳，供 exec loop 计算首个 action 延迟
        self._transition_to_policy_mono = _t0

        self._start_policy_exec_loop()
        _t3 = time.monotonic()

        _diag_log(
            f"[TRANSITION-DIAG] HUMAN->POLICY: "
            f"deactivate+clear={(_t1-_t0)*1000:.1f}ms, "
            f"mode_lock={(_t2-_t1)*1000:.1f}ms, "
            f"start_exec={(_t3-_t2)*1000:.1f}ms, "
            f"total={(_t3-_t0)*1000:.1f}ms, "
            f"buf_cleared={_buf_before}, has_last_popped={_has_last_popped}"
        )
        self.get_logger().info("Transition: HUMAN -> POLICY (buffer cleared)")
        self._publish_status()

    # =========================================================================
    # Target pose calculation (matches vr_teleop_node exactly)
    # =========================================================================

    def _calculate_target_pose(self) -> Optional[list]:
        """
        Calculate target Cartesian pose from VR input.

        Returns:
            [x, y, z, qw, qx, qy, qz] pose list, or None
        """
        # 1. Position delta
        position_diff = self.vr_state.current_vr_pos - self.vr_state.init_vr_pos

        # 2. Coordinate transform
        position_diff_transformed = self.coord_transform.transform_position(position_diff)

        # 3. Target position
        target_pos = self.vr_state.init_arm_pos + position_diff_transformed

        # 4. Target rotation
        target_quat = self.coord_transform.transform_rotation(
            self.vr_state.current_vr_quat,
            self.vr_state.init_vr_quat,
            self.vr_state.init_arm_quat,
        )

        # 5. Filter
        filtered_pos, filtered_quat = self.pose_filter.filter_pose(target_pos, target_quat)

        # 6. Build target pose [x, y, z, qw, qx, qy, qz]
        # scipy quaternion: [x, y, z, w] -> Realman SDK: [qw, qx, qy, qz]
        qx, qy, qz, qw = filtered_quat
        target_pose = [
            float(filtered_pos[0]),
            float(filtered_pos[1]),
            float(filtered_pos[2]),
            float(qw),
            float(qx),
            float(qy),
            float(qz),
        ]

        return target_pose

    def _sync_arm_state(self):
        """Sync arm state as anchor point (when trigger is not pressed)."""
        arm_state = self._get_current_arm_state()
        if arm_state is not None:
            joint_rad, eef_pose, gripper = arm_state
            if eef_pose is not None:
                self.vr_state.init_arm_pos = np.array(eef_pose[:3])
                euler_rad = eef_pose[3:6]
                rot = R.from_euler('xyz', euler_rad)
                self.vr_state.init_arm_quat = rot.as_quat()

        # Sync VR initial pose
        self.vr_state.init_vr_pos = self.vr_state.current_vr_pos.copy()
        self.vr_state.init_vr_quat = self.vr_state.current_vr_quat.copy()

    # =========================================================================
    # Main execution tick
    # =========================================================================

    def _execution_tick(self):
        """
        Main control loop, called at control_hz.

        Dispatches to the appropriate handler based on current mode.
        Observation feeding 由独立的 _obs_feed_worker 线程处理，不在此回调中。
        Recording is driven by _on_recording_sync (message-synchronized), not here.
        """
        with self._mode_lock:
            mode = self._mode

        if mode == ControlMode.IDLE:
            return

        # Observation feeding 已移至独立的 _obs_feed_worker 线程，不再在 timer 回调中调用。

        if mode == ControlMode.HUMAN:
            self._tick_human()
            # Shadow tick: 消费 buffer 中的 policy action 用于录制
            self._tick_shadow()
        elif mode == ControlMode.POLICY:
            # Action execution 由 _policy_exec_loop（独立线程）驱动
            # Observation feeding 由 _obs_feed_worker（独立线程）驱动
            pass

    def _tick_human(self):
        """HUMAN mode tick: VR teleoperation.

        """
        if not hasattr(self, '_tick_human_count'):
            self._tick_human_count = 0
        self._tick_human_count += 1

        with self._vr_lock:
            # Check VR data timeout
            _vr_age = time.time() - self._vr_last_update
            if _vr_age > 0.5:
                _diag_log(f"[TICK-HUMAN] #{self._tick_human_count}: SKIP vr_timeout ({_vr_age:.3f}s)")
                self.get_logger().warn(
                    "VR data timeout (>0.5s since last update), skipping HUMAN tick",
                    throttle_duration_sec=5.0,
                )
                return

            # If trigger not pressed, sync arm state as anchor
            if not self.vr_state.is_active:
                _diag_log(f"[TICK-HUMAN] #{self._tick_human_count}: SKIP is_active=False")
                self._sync_arm_state()
                return

            # If not initialized, skip
            if not self.vr_state.is_initialized:
                _diag_log(f"[TICK-HUMAN] #{self._tick_human_count}: SKIP is_initialized=False")
                return

            # Calculate target Cartesian pose
            target_pose = self._calculate_target_pose()
            if target_pose is None:
                _diag_log(f"[TICK-HUMAN] #{self._tick_human_count}: SKIP target_pose=None")
                return

        # Publish action
        _diag_log(
            f"[TICK-HUMAN] #{self._tick_human_count}: PUB pose "
            f"grip={self._current_grip:.2f} "
            f"pos=[{target_pose[0]:.4f},{target_pose[1]:.4f},{target_pose[2]:.4f}]"
        )
        self._publish_pose_action(target_pose)

    def _tick_shadow(self):
        """
        Shadow tick: HUMAN 模式下消费 ring buffer 中的 policy action，
        仅更新 _last_policy_action 用于录制，不发送给机器人。
        """
        if self._inference_bridge is None:
            return

        # warmup 未完成时不操作
        if not self._inference_bridge.warmup_done:
            return

        action = self._inference_bridge.get_next_action()

        if action is None:
            if self.hold_on_empty:
                action = self._inference_bridge.last_action
            if action is None:
                return

        # 更新 policy action 用于录制（与 _tick_policy 相同的拼接逻辑）
        self._last_policy_action = np.concatenate([
            np.array(action.data, dtype=np.float32),
            np.array([action.gripper], dtype=np.float32),
        ])

    def _tick_policy(self):
        """
        POLICY mode tick: pop action from InferenceBridge and execute.

        Action flow:
        1. Pop next action from ring buffer
        2. If empty and hold_on_empty, reuse last action
        3. Safety check (max joint delta)
        4. Publish joint action to /rm/action_joint_state
        """
        if self._inference_bridge is None:
            return

        # Wait for warmup before executing
        if not self._inference_bridge.warmup_done:
            return

        action = self._inference_bridge.get_next_action()

        if action is None:
            if self.hold_on_empty:
                action = self._inference_bridge.last_action
            if action is None:
                return

        # Track policy action for recording (raw data + gripper)
        self._last_policy_action = np.concatenate([
            np.array(action.data, dtype=np.float32),
            np.array([action.gripper], dtype=np.float32),
        ])

        # Parse action based on action_space
        if action.action_space == "joints":
            joint_rad = np.array(action.data[:7], dtype=np.float64)
            gripper = float(action.gripper)
        elif action.action_space == "pose":
            # Pose actions: [x, y, z, rx, ry, rz] absolute pose from server
            # 直接透传给 driver（与 async_inference_client 一致），不做坐标转换
            pose_data = action.data
            gripper = float(action.gripper)
            if len(pose_data) >= 6:
                pose_6d = [float(v) for v in pose_data[:6]]
                self._publish_pose_action_with_gripper(pose_6d, gripper)
            return
        else:
            self.get_logger().warn(
                f"Unknown action_space: {action.action_space}",
                throttle_duration_sec=5.0,
            )
            return

        # 直接执行 joint action（与 async_inference_client 一致，无 safety check）
        self._last_executed_joints = joint_rad.copy()
        self._publish_joint_action(joint_rad.tolist(), gripper)

    # =========================================================================
    # Policy execution loop (async_client style while+sleep)
    # =========================================================================

    def _start_policy_exec_loop(self):
        """Start the policy execution thread (async_client style while+sleep)."""
        # join 前置：确保旧 exec 线程已退出（从 _transition_to_human 延迟到这里）
        if self._policy_exec_thread is not None and self._policy_exec_thread.is_alive():
            _t_join_start = time.monotonic()
            self._policy_exec_thread.join(timeout=2.0)
            _t_join_done = time.monotonic()
            _diag_log(
                f"[EXEC-DIAG] Old exec thread joined: {(_t_join_done-_t_join_start)*1000:.1f}ms, "
                f"still_alive={self._policy_exec_thread.is_alive()}"
            )
        # Create a FRESH Event for the new thread — prevents race condition where
        # clear() on a shared Event erases the stop signal for a still-running old thread.
        # The old thread keeps its own Event reference and will still see its stop flag.
        stop_event = threading.Event()
        self._policy_exec_stop = stop_event
        self._policy_exec_stop_time = 0.0  # Reset diagnostic timestamp for new exec loop
        self._policy_exec_thread = threading.Thread(
            target=self._policy_exec_loop, args=(stop_event,), daemon=True,
            name="policy_exec_loop",
        )
        self._policy_exec_thread.start()
        self.get_logger().info(
            f"Policy exec loop started (f_exec={self.f_exec}Hz, T_step={1.0/self.f_exec:.4f}s)"
        )

    def _stop_policy_exec_loop(self):
        """Stop the policy execution thread."""
        self._policy_exec_stop.set()
        self._policy_exec_stop_time = time.monotonic()  # 记录 stop 时间供诊断
        if self._policy_exec_thread is not None and self._policy_exec_thread.is_alive():
            self._policy_exec_thread.join(timeout=2.0)
            self._policy_exec_thread = None

    # =========================================================================
    # Obs feed worker（独立线程持续喂 observation 给 inference bridge）
    # =========================================================================

    def _start_obs_feed_worker(self):
        """启动 obs feed 线程。幂等：如果已在运行则跳过。"""
        if self._obs_feed_thread is not None and self._obs_feed_thread.is_alive():
            return
        self._obs_feed_stop.clear()
        self._obs_feed_thread = threading.Thread(
            target=self._obs_feed_loop, daemon=True,
            name="obs_feed_worker",
        )
        self._obs_feed_thread.start()
        self.get_logger().info(
            f"Obs feed worker started (control_hz={self.control_hz}Hz)"
        )

    def _stop_obs_feed_worker(self):
        """停止 obs feed 线程。"""
        self._obs_feed_stop.set()
        if self._obs_feed_thread is not None and self._obs_feed_thread.is_alive():
            self._obs_feed_thread.join(timeout=2.0)
        self._obs_feed_thread = None
        self.get_logger().info("Obs feed worker stopped")

    def _obs_feed_loop(self):
        """
        Obs feed 线程主函数：以 control_hz 频率持续将 state + image 推送给 inference bridge。

        与 _recording_worker 一致，外层 try/except 捕获异常只 log 不退出。
        使用 Event.wait() 替代 time.sleep()，在等待期间充分释放 GIL，
        减少与 exec loop / executor 线程的 GIL 竞争。
        """
        period = 1.0 / self.control_hz if self.control_hz > 0 else 0.02
        _consecutive_skips = 0
        _total_feeds = 0
        _total_skips = 0
        while not self._obs_feed_stop.is_set():
            try:
                ok = self._update_inference_observation()
                _total_feeds += 1
                if ok:
                    if _consecutive_skips >= 10:
                        # 从连续跳过中恢复——记录恢复信息
                        with self._mode_lock:
                            _mode = self._mode.name
                        _diag_log(
                            f"[OBS-FEED-RESUME] 恢复 obs 喂入: "
                            f"连续跳过={_consecutive_skips}, mode={_mode}"
                        )
                    _consecutive_skips = 0
                else:
                    _consecutive_skips += 1
                    _total_skips += 1
                    # 连续跳过 50 次（~1s @50Hz）时打印警告
                    if _consecutive_skips == 50 or (_consecutive_skips > 0 and _consecutive_skips % 500 == 0):
                        with self._mode_lock:
                            _mode = self._mode.name
                        _diag_log(
                            f"[OBS-FEED-WARN] 连续 obs 跳过={_consecutive_skips}, "
                            f"mode={_mode}, total_skips={_total_skips}/{_total_feeds}"
                        )
            except Exception as e:
                self.get_logger().error(
                    f"[ObsFeedWorker] error: {e}",
                    throttle_duration_sec=5.0,
                )
            self._obs_feed_stop.wait(timeout=period)

    def _policy_exec_loop(self, stop_event: threading.Event):
        """
        Policy execution loop — mirrors async_inference_client.run() exactly.

        Runs in a dedicated thread. Fixed-frequency while+sleep loop:
        1. Pop action from ring buffer
        2. Execute (publish via ROS2)
        3. Sleep to maintain f_exec Hz

        Observation feeding 由独立的 _obs_feed_worker 线程以 control_hz 频率处理，
        不在此循环中调用，以避免 obs 构建耗时拖慢执行频率。

        Args:
            stop_event: Per-thread Event — only THIS thread checks this Event.
                        Prevents race conditions with shared Events across threads.
        """
        T_step = 1.0 / self.f_exec if self.f_exec > 0 else 0.1

        # Wait for warmup
        while not stop_event.is_set():
            if self._inference_bridge is not None and self._inference_bridge.warmup_done:
                break
            time.sleep(0.05)

        if stop_event.is_set():
            return

        self.get_logger().info("[PolicyExecLoop] Warmup done, starting execution")
        _exec_loop_start = time.monotonic()
        self._exec_count = 0  # 重置计数器，避免 actual_hz 因历史累积而虚高

        # 如果标记了需要清空 buffer（resume 后），完全重置再开始执行
        if self._clear_buffer_on_exec_start:
            self._clear_buffer_on_exec_start = False
            if self._inference_bridge is not None:
                self._inference_bridge.reset_buffer()
                self.get_logger().info("[PolicyExecLoop] Buffer reset (resume after pause)")

        # 无条件完全清空 buffer（含 last_popped）+ 刷新 observation：
        # 确保第一帧执行的 action 是基于当前真实状态的推理结果，
        # 而不是 warmup 残留、旧 session 残留、或基于旧 observation 的过时 action。
        if self._inference_bridge is not None:
            self._update_inference_observation()
            self._inference_bridge.reset_buffer()
            self.get_logger().info("[PolicyExecLoop] Buffer reset + observation refreshed before first execution")

        # Fix 10: 等待 inference 线程填充第一个 chunk，避免首帧 HOLD
        if self._inference_bridge is not None:
            _prefill_start = time.monotonic()
            _prefill_timeout = 0.1  # 最多等 100ms
            while not stop_event.is_set():
                if self._inference_bridge.ring_buffer.size > 0:
                    _diag_log(f"[EXEC-PREFILL] Buffer 预填充完成, "
                              f"buf={self._inference_bridge.ring_buffer.size}, "
                              f"wait={(time.monotonic()-_prefill_start)*1000:.1f}ms")
                    break
                if time.monotonic() - _prefill_start > _prefill_timeout:
                    _diag_log(f"[EXEC-PREFILL] 超时 {_prefill_timeout*1000:.0f}ms, buf=0, 降级为 HOLD")
                    break
                time.sleep(0.005)

        while not stop_event.is_set():
            t_start = time.monotonic()

            # Check mode — stop if no longer POLICY
            _t_mode = time.monotonic()
            with self._mode_lock:
                if self._mode != ControlMode.POLICY:
                    break
            _t_mode_done = time.monotonic()

            # Pop action from ring buffer
            _t_pop = time.monotonic()
            action = self._inference_bridge.get_next_action()
            _t_pop_done = time.monotonic()

            if action is None:
                if self.hold_on_empty:
                    action = self._inference_bridge.last_action
                if action is None:
                    # 无 last_action（首帧或 reset 后）：发送当前关节位置保持 CANFD 流
                    state_14d = self._get_state_14d()
                    if state_14d is not None:
                        joint_rad = state_14d[:7].tolist()
                        gripper = float(np.clip(state_14d[7], 0.0, 1.0))
                        self._publish_joint_action(joint_rad, gripper)
                        if self._exec_count <= 5 or self._exec_count % 100 == 0:
                            _diag_log(f"[TIMING-EXEC] #{self._exec_count}: HOLD-KEEPALIVE, "
                                  f"buf_size={self._inference_bridge.ring_buffer.size}")
                    else:
                        if self._exec_count <= 5 or self._exec_count % 100 == 0:
                            _diag_log(f"[TIMING-EXEC] #{self._exec_count}: HOLD (no action, no state), "
                                  f"buf_size={self._inference_bridge.ring_buffer.size}")
                    elapsed = time.monotonic() - t_start
                    wait_time = T_step - elapsed
                    if wait_time > 0:
                        self._inference_bridge.ring_buffer.wait_for_action(timeout=wait_time)
                    continue

            # Motion Gating: warmup 阶段有 action 但不下发，机械臂保持 Home
            if not self._motion_gate.is_set():
                elapsed = time.monotonic() - t_start
                wait_time = T_step - elapsed
                if wait_time > 0:
                    self._motion_gate.wait(timeout=wait_time)
                continue

            # 首个 action 延迟诊断：从 HUMAN→POLICY 切换到首帧执行的总耗时
            if self._exec_count == 0:
                _switch_mono = getattr(self, '_transition_to_policy_mono', 0.0)
                if _switch_mono > 0:
                    _first_action_delay = (time.monotonic() - _switch_mono) * 1000
                    _diag_log(
                        f"[FIRST-ACTION] 模式切换到首帧执行延迟={_first_action_delay:.1f}ms, "
                        f"action_source={'hold' if action is self._inference_bridge.last_action else 'fresh'}"
                    )

            # Track policy action for recording
            self._last_policy_action = np.concatenate([
                np.array(action.data, dtype=np.float32),
                np.array([action.gripper], dtype=np.float32),
            ])

            # === DIAG-COMPARE: 每帧写文件日志，与 async_client 对齐 ===
            state_14d = self._get_state_14d()
            obs_gripper = state_14d[7] if state_14d is not None else -1
            _buf_sz = self._inference_bridge.ring_buffer.size
            _is_hold = (action is self._inference_bridge.ring_buffer.last_popped and
                        self._inference_bridge.ring_buffer.is_empty)
            _diag_log(
                f"[EXEC] #{self._exec_count}: "
                f"grip_obs={obs_gripper:.4f}, grip_act={action.gripper:.4f}, "
                f"grip_sdk={int(np.clip(action.gripper, 0.0, 1.0) * 1000)}, "
                f"delta={action.gripper - obs_gripper:.4f}, "
                f"buf={_buf_sz}, hold={_is_hold}, "
                f"action_space={action.action_space}"
            )

            # Execute action
            _t_exec = time.monotonic()
            if action.action_space == "pose":
                pose_data = action.data
                gripper = float(action.gripper)
                if len(pose_data) >= 6:
                    pose_6d = [float(v) for v in pose_data[:6]]
                    self._publish_pose_action_with_gripper(pose_6d, gripper)
            elif action.action_space == "joints":
                joint_rad = np.array(action.data[:7], dtype=np.float64)
                gripper = float(action.gripper)
                # 直接执行（与 async_inference_client 一致，无 safety check）
                self._last_executed_joints = joint_rad.copy()
                self._publish_joint_action(joint_rad.tolist(), gripper)
            else:
                self.get_logger().warn(
                    f"Unknown action_space: {action.action_space}",
                    throttle_duration_sec=5.0,
                )
            _t_exec_done = time.monotonic()

            self._exec_count += 1

            # === TIMING: exec loop 各阶段耗时 ===
            if self._exec_count <= 10 or self._exec_count % 100 == 0:
                _elapsed_since_start = time.monotonic() - _exec_loop_start
                _actual_hz = self._exec_count / _elapsed_since_start if _elapsed_since_start > 0 else 0
                _diag_log(f"[TIMING-EXEC] #{self._exec_count}: "
                      f"mode_lock={(_t_mode_done-_t_mode)*1000:.2f}ms, "
                      f"pop={(_t_pop_done-_t_pop)*1000:.2f}ms, "
                      f"execute={(_t_exec_done-_t_exec)*1000:.2f}ms, "
                      f"total_work={(_t_exec_done-t_start)*1000:.2f}ms, "
                      f"buf={self._inference_bridge.ring_buffer.size}, "
                      f"actual_hz={_actual_hz:.1f}")

            # Frame rate control (identical to async_client)
            # 用 stop_event.wait 替代 time.sleep，使 _stop_policy_exec_loop() 能立即唤醒线程退出，
            # 避免 trigger 回调被 join() 阻塞最多 33ms
            elapsed = time.monotonic() - t_start
            sleep_time = T_step - elapsed
            if sleep_time > 0:
                stop_event.wait(timeout=sleep_time)
                if stop_event.is_set():
                    break

        _t_exit = time.monotonic()
        _stop_time = getattr(self, '_policy_exec_stop_time', 0.0)
        _exit_delay = (_t_exit - _stop_time) * 1000 if _stop_time > 0 else -1
        _diag_log(
            f"[EXEC-DIAG] PolicyExecLoop stopped, exit_delay={_exit_delay:.1f}ms "
            f"(from stop flag to thread exit)"
        )

    def _update_inference_observation(self):
        """Feed current state + images to InferenceBridge for background inference.

        返回 True 表示成功喂入 obs，False 表示跳过（用于 obs_feed_loop 统计）。
        """
        if self._inference_bridge is None:
            return False

        _t0 = time.monotonic()
        state_14d = self._get_state_14d()
        if state_14d is None:
            # 诊断：记录跳过原因——state 为 None（driver 尚未发布 or 帧长度 < 14）
            self._obs_skip_state_count += 1
            if self._obs_skip_state_count <= 5 or self._obs_skip_state_count % 100 == 0:
                _diag_log(f"[OBS-SKIP] state_14d=None (skip#{self._obs_skip_state_count})")
            return False
        _t1 = time.monotonic()

        raw_msgs = self._get_raw_image_msgs()
        if not raw_msgs:
            # 诊断：记录跳过原因——图像缓存为空（camera 尚未到达）
            self._obs_skip_image_count += 1
            if self._obs_skip_image_count <= 5 or self._obs_skip_image_count % 100 == 0:
                with self._image_lock:
                    cached_cams = list(self._cached_image_msgs.keys())
                _diag_log(
                    f"[OBS-SKIP] raw_msgs empty (skip#{self._obs_skip_image_count}), "
                    f"cached_cams={cached_cams}, expected={self.camera_names}"
                )
            return False
        _t2 = time.monotonic()

        self._inference_bridge.update_observation(state_14d, raw_image_msgs=raw_msgs)
        _t3 = time.monotonic()

        self._obs_feed_count += 1
        # 降低正常路径的日志频率：前 10 帧 + 每 200 帧
        if self._obs_feed_count <= 10 or self._obs_feed_count % 200 == 0:
            _diag_log(
                f"[OBS-FEED] #{self._obs_feed_count}: "
                f"gripper={state_14d[7]:.4f}, "
                f"state={(_t1-_t0)*1000:.2f}ms, "
                f"raw_msg={(_t2-_t1)*1000:.2f}ms, "
                f"bridge={(_t3-_t2)*1000:.2f}ms, "
                f"total={(_t3-_t0)*1000:.2f}ms"
            )
        return True

    def _safety_check_joints(self, target_joints: np.ndarray) -> tuple[bool, np.ndarray]:
        """
        检查目标关节位置是否在安全范围内（与上次执行位置的 delta 不超过限制）。

        如果超出限制，将 delta clip 到安全范围内，返回 clipped 后的关节位置。
        这样可以让机械臂逐步追赶目标轨迹，而不是完全丢弃 action。

        Returns:
            (is_within_limit, joints_to_execute)
            - is_within_limit: True 表示在限制内，False 表示被 clip
            - joints_to_execute: 可执行的关节位置（原始或 clipped）
        """
        if self._last_executed_joints is None:
            # 模式切换后第一个 action：使用当前机械臂状态作为基准
            arm_state = self._get_current_arm_state()
            if arm_state is not None:
                self._last_executed_joints = arm_state[0].copy()
            else:
                return True, target_joints  # 无基准，直接执行

        delta_rad = target_joints - self._last_executed_joints
        delta_deg = np.degrees(np.abs(delta_rad))
        max_delta = np.max(delta_deg)

        if max_delta > self.max_joint_delta_deg:
            self._safety_reject_count += 1
            worst_joint = int(np.argmax(delta_deg))

            # Clip delta 到安全范围内
            max_rad = np.radians(self.max_joint_delta_deg)
            clipped_delta = np.clip(delta_rad, -max_rad, max_rad)
            clipped_joints = self._last_executed_joints + clipped_delta

            # 详细调试信息（仅前 10 次 clip 时打印）
            if self._safety_reject_count <= 10:
                self.get_logger().warn(
                    f"Safety clip #{self._safety_reject_count}: joint {worst_joint} delta={max_delta:.1f}deg → clipped to {self.max_joint_delta_deg}deg\n"
                    f"  target (deg): {np.degrees(target_joints).tolist()}\n"
                    f"  last_exec (deg): {np.degrees(self._last_executed_joints).tolist()}\n"
                    f"  clipped (deg): {np.degrees(clipped_joints).tolist()}",
                )
            elif self._safety_reject_count % 100 == 0:
                self.get_logger().warn(
                    f"Safety clip count: {self._safety_reject_count} (joint {worst_joint} delta={max_delta:.1f}deg)",
                )

            return False, clipped_joints

        return True, target_joints

    def _publish_pose_action_with_gripper(self, pose: list, gripper: float):
        """
        Publish Cartesian pose action to /rm/action_pose.

        根据 pose 长度自动设置 msg.name：
        - 6D [x,y,z,rx,ry,rz] → 欧拉角格式
        - 7D [x,y,z,qw,qx,qy,qz] → 四元数格式
        """
        if not hasattr(self, '_pub_pose_count'):
            self._pub_pose_count = 0
        self._pub_pose_count += 1
        _diag_log(f"[PUB-POSE] #{self._pub_pose_count}: gripper={gripper:.4f}, pose_dim={len(pose)}")

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        if len(pose) == 6:
            msg.name = ["x", "y", "z", "rx", "ry", "rz", "gripper"]
        else:
            msg.name = ["x", "y", "z", "qw", "qx", "qy", "qz", "gripper"]

        msg.position = list(pose) + [float(np.clip(gripper, 0.0, 1.0))]

        self.action_pose_pub.publish(msg)
        self._cmd_count += 1
        self._record_cmd_timestamp()

    # =========================================================================
    # Action publishing
    # =========================================================================

    def _publish_pose_action(self, target_pose: list):
        """
        Publish Cartesian pose action to /rm/action_pose (VR mode).

        Args:
            target_pose: [x, y, z, qw, qx, qy, qz]
        """
        gripper_open = float(np.clip(self._current_grip, 0.0, 1.0))
        self._publish_pose_action_with_gripper(target_pose, gripper_open)

    def _publish_joint_action(self, joint_rad: list, gripper: float):
        """
        Publish joint action to /rm/action_joint_state.

        Args:
            joint_rad: 7-element list of joint angles in radians
            gripper: gripper openness 0-1
        """
        # === DIAG-COMPARE: 每帧发布 gripper 写文件 ===
        if not hasattr(self, '_publish_count'):
            self._publish_count = 0
        self._publish_count += 1
        _diag_log(f"[PUB-ACTION] #{self._publish_count}: gripper={gripper:.4f}")

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.name = [
            "joint_1", "joint_2", "joint_3", "joint_4",
            "joint_5", "joint_6", "joint_7", "gripper",
        ]
        msg.position = list(joint_rad) + [float(gripper)]

        self.action_joint_pub.publish(msg)
        self._cmd_count += 1
        self._record_cmd_timestamp()

    def _record_cmd_timestamp(self):
        """记录 action 发布时间戳，用于计算实际控制频率。"""
        now = time.monotonic()
        with self._cmd_ts_lock:
            self._cmd_timestamps.append(now)
            cutoff = now - self._CMD_WINDOW_SEC
            self._cmd_timestamps = [t for t in self._cmd_timestamps if t > cutoff]

    def _get_control_hz(self) -> float:
        """计算 2 秒滑动窗口内的实际控制频率。"""
        now = time.monotonic()
        with self._cmd_ts_lock:
            cutoff = now - self._CMD_WINDOW_SEC
            valid = [t for t in self._cmd_timestamps if t > cutoff]
            if len(valid) < 2:
                return 0.0
            return len(valid) / self._CMD_WINDOW_SEC

    # =========================================================================
    # Service handlers
    # =========================================================================

    def _call_driver_service(self, client, service_name: str) -> bool:
        """Call a driver service (non-blocking)."""
        if not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(f"Service {service_name} not available")
            return False

        try:
            future = client.call_async(Trigger.Request())
            start = time.time()
            while not future.done() and time.time() - start < 1.0:
                time.sleep(0.05)

            if future.done():
                result = future.result()
                if result.success:
                    self.get_logger().info(f"{service_name}: {result.message}")
                    return True
                else:
                    self.get_logger().warn(f"{service_name}: {result.message}")
                    return False
            else:
                self.get_logger().warn(f"{service_name} call timed out")
                return False
        except Exception as e:
            self.get_logger().error(f"{service_name} call failed: {e}")
            return False

    def _handle_enable_control(self, request, response):
        """
        Enable DAgger control.

        Default mode: POLICY (if policy enabled), otherwise HUMAN.
        Also enables driver follow for real robot and starts inference.
        """
        with self._mode_lock:
            if self._mode != ControlMode.IDLE:
                response.success = False
                response.message = f"Already in {self._mode.name} mode"
                return response

            # Enable driver follow on real robot
            if not self.dry_run:
                self.get_logger().info("Enabling driver follow...")
                self._call_driver_service(
                    self.driver_enable_follow_client,
                    "/driver/enable_follow",
                )

            # Start inference bridge if policy enabled
            if self.enable_policy and self._inference_bridge is not None:
                try:
                    self._inference_bridge.connect()
                    self._inference_bridge.start_inference_loop()
                    self._mode = ControlMode.POLICY
                    self._start_obs_feed_worker()
                    self._start_policy_exec_loop()
                    mode_name = "POLICY"
                except Exception as e:
                    self.get_logger().error(f"Failed to connect to PolicyServer: {e}")
                    self.get_logger().info("Falling back to HUMAN mode")
                    self._mode = ControlMode.HUMAN
                    mode_name = "HUMAN (policy connection failed)"
            else:
                self._mode = ControlMode.HUMAN
                mode_name = "HUMAN"

            response.success = True
            response.message = f"DAgger control enabled ({mode_name})"
            self.get_logger().info(f"DAgger control enabled ({mode_name})")
            self._publish_status()

        return response

    def _handle_disable_control(self, request, response):
        """
        Disable DAgger control (return to IDLE).

        Stops inference and disables driver follow for real robot.
        """
        with self._mode_lock:
            if self._mode == ControlMode.IDLE:
                response.success = False
                response.message = "Already in IDLE mode"
                return response

            self._mode = ControlMode.IDLE
            self.vr_state.is_active = False
            self.vr_state.is_initialized = False
            self._last_executed_joints = None

            # Stop policy exec loop
            self._stop_policy_exec_loop()

            # Stop obs feed worker + inference bridge
            self._stop_obs_feed_worker()
            if self._inference_bridge is not None:
                self._inference_bridge.stop_inference_loop()

            # Disable driver follow on real robot
            if not self.dry_run:
                self.get_logger().info("Disabling driver follow...")
                self._call_driver_service(
                    self.driver_disable_follow_client,
                    "/driver/disable_follow",
                )

            response.success = True
            response.message = "DAgger control disabled (IDLE)"
            self.get_logger().info("DAgger control disabled (IDLE)")
            self._publish_status()

        return response

    def _handle_set_human(self, request, response):
        """Switch to HUMAN (VR) mode."""
        with self._mode_lock:
            old_mode = self._mode
            if old_mode == ControlMode.IDLE:
                response.success = False
                response.message = "Control not enabled. Call /dagger/enable_control first."
                return response

            # Pause inference if switching from POLICY
            if old_mode == ControlMode.POLICY and self._inference_bridge is not None:
                self._stop_policy_exec_loop()
                self._inference_bridge.pause()
                self._inference_bridge.clear_buffer()

            self._mode = ControlMode.HUMAN
            self._last_policy_action = None  # 清零，避免录制过时的 policy 输出

        # vr_state 写入移到 mode_lock 外，避免与 _on_vr_trigger（vr_lock→mode_lock）形成 ABBA 死锁
        with self._vr_lock:
            self.vr_state.is_active = False
            self.vr_state.is_initialized = False
            self.pose_filter.reset()

        response.success = True
        response.message = f"Switched from {old_mode.name} to HUMAN mode"
        self.get_logger().info(f"Mode: {old_mode.name} -> HUMAN")
        self._publish_status()

        return response

    def _handle_set_policy(self, request, response):
        """Switch to POLICY mode."""
        with self._mode_lock:
            old_mode = self._mode
            if old_mode == ControlMode.IDLE:
                response.success = False
                response.message = "Control not enabled. Call /dagger/enable_control first."
                return response

            if not self.enable_policy or self._inference_bridge is None:
                response.success = False
                response.message = "Policy not enabled or InferenceBridge not initialized"
                return response

            self.vr_state.is_active = False
            self.vr_state.is_initialized = False
            self._last_executed_joints = None
            self._last_policy_action = None  # 清零，避免切换后 hold 旧 action

            # Resume inference
            self._inference_bridge.clear_buffer()
            self._inference_bridge.resume()

            self._mode = ControlMode.POLICY
            self._start_policy_exec_loop()
            response.success = True
            response.message = f"Switched from {old_mode.name} to POLICY mode"
            self.get_logger().info(f"Mode: {old_mode.name} -> POLICY")
            self._publish_status()

        return response

    def _handle_set_idle(self, request, response):
        """Switch to IDLE mode (same as disable_control but without driver follow change)."""
        with self._mode_lock:
            old_mode = self._mode
            self._mode = ControlMode.IDLE
            self.vr_state.is_active = False
            self.vr_state.is_initialized = False
            self._last_executed_joints = None

            # Stop policy exec loop
            self._stop_policy_exec_loop()

            # Stop obs feed worker + inference
            self._stop_obs_feed_worker()
            if self._inference_bridge is not None:
                self._inference_bridge.stop_inference_loop()

            response.success = True
            response.message = f"Switched from {old_mode.name} to IDLE"
            self.get_logger().info(f"Mode: {old_mode.name} -> IDLE")
            self._publish_status()

        return response

    # =========================================================================
    # Recording (Phase 3C)
    # =========================================================================

    def _on_recording_sync(self, state_msg: JointState, *image_msgs):
        """
        消息同步录制回调：由 ApproximateTimeSynchronizer 驱动。

        异步化设计：回调只做前置检查 + 入队（<1ms），
        imgmsg_to_cv2 和 add_frame 在后台 _recording_worker 线程中执行。

        Args:
            state_msg: 同步的 JointState 消息
            *image_msgs: 同步的 Image 消息（顺序与 self.camera_names 一致）
        """
        # 前置检查：未录制或已暂停时直接 return
        if not self._episode_active:
            _diag_log("[REC-SYNC] skip: episode_active=False", throttle_key="rec_sync_not_active", throttle_sec=2.0)
            return
        if self._episode_paused:
            _diag_log("[REC-SYNC] skip: episode_paused=True", throttle_key="rec_sync_paused", throttle_sec=2.0)
            return
        if self._recorder is None and self._recorder_initialized:
            _diag_log("[REC-SYNC] skip: recorder=None but initialized=True (init failed)", throttle_key="rec_sync_init_fail", throttle_sec=5.0)
            return  # 初始化失败过，不再重试

        # Warmup 检查：policy 模式下必须等 warmup 完成后才开始录制
        # 避免在模型加载期间用错误的 action_dim 初始化 recorder
        if self.enable_policy and self._inference_bridge is not None:
            if not self._inference_bridge.warmup_done:
                _diag_log("[REC-SYNC] skip: warmup_done=False", throttle_key="rec_sync_warmup", throttle_sec=2.0)
                return

        # Motion Gating: 录制链路未就绪时跳过（防止 warmup 阶段录入无效帧）
        if not self._recording_ready:
            _diag_log("[REC-SYNC] skip: recording_ready=False", throttle_key="rec_sync_not_ready", throttle_sec=2.0)
            return

        self._sync_triggered += 1

        # [REC-DIAG] sync 回调间隔监控
        now_mono = time.monotonic()
        if self._last_sync_ts > 0:
            sync_gap_ms = (now_mono - self._last_sync_ts) * 1000.0
            expect_ms = 1000.0 / self.recording_fps if self.recording_fps > 0 else 33.3
            if sync_gap_ms > expect_ms * 1.5:
                self.get_logger().warn(
                    f"[REC-DIAG] sync_gap={sync_gap_ms:.1f}ms (expect ≤{expect_ms * 1.5:.1f}ms), "
                    f"frame=#{self._sync_triggered}"
                )
        self._last_sync_ts = now_mono

        # 频率限制：跳过间隔不足的帧
        now = time.time()
        if self._min_frame_interval > 0 and self._last_recording_ts is not None:
            if now - self._last_recording_ts < self._min_frame_interval:
                self._sync_throttled += 1
                self._log_recording_stats()
                return

        # 入队：将消息引用 + 时间戳打包，不做 imgmsg_to_cv2
        enqueue_ts = time.monotonic()
        try:
            self._recording_queue.put_nowait((state_msg, image_msgs, now, enqueue_ts))
        except queue.Full:
            self._sync_queue_full += 1
            self.get_logger().warn(
                "[RecordingSync] 录制队列已满，丢弃当前帧",
                throttle_duration_sec=2.0,
            )
            self._log_recording_stats()
            return

        self._sync_recorded += 1
        self._last_recording_ts = now
        self._log_recording_stats()

    def _log_recording_stats(self):
        """每 10 秒通过 _diag_log 汇总录制链路诊断计数器。不打印终端。"""
        now = time.monotonic()
        if self._sync_stats_last_time == 0.0:
            self._sync_stats_last_time = now
            return
        elapsed = now - self._sync_stats_last_time
        if elapsed < 10.0:
            return
        fps = self._sync_recorded / elapsed if elapsed > 0 else 0.0
        _diag_log(
            f"[REC-STATS] {elapsed:.1f}s: "
            f"triggered={self._sync_triggered} "
            f"throttled={self._sync_throttled} "
            f"queue_full={self._sync_queue_full} "
            f"recorded={self._sync_recorded} "
            f"fps={fps:.1f}"
        )
        # 重置计数器
        self._sync_triggered = 0
        self._sync_throttled = 0
        self._sync_queue_full = 0
        self._sync_recorded = 0
        self._sync_stats_last_time = now

    def _recording_worker(self):
        """
        录制后台线程：从队列取出消息，执行 imgmsg_to_cv2 + add_frame。

        在 _recording_stop 被 set 时退出循环。
        """
        self.get_logger().info("[RecordingWorker] 后台录制线程启动")
        while not self._recording_stop.is_set():
            try:
                item = self._recording_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            state_msg, image_msgs, now, enqueue_ts = item

            # [REC-DIAG] 队列深度和消费延迟监控
            dequeue_ts = time.monotonic()
            consume_delay_ms = (dequeue_ts - enqueue_ts) * 1000.0
            queue_depth = self._recording_queue.qsize()
            if queue_depth > 5 or consume_delay_ms > 100.0:
                self.get_logger().warn(
                    f"[REC-DIAG] queue_depth={queue_depth}, consume_delay={consume_delay_ms:.1f}ms, "
                    f"frame=#{self._recording_frame_count}"
                )

            # Fix C: 丢弃超过 2 秒的旧帧（init 期间积压的消息引用可能已失效）
            age_sec = dequeue_ts - enqueue_ts
            if age_sec > 2.0:
                self._worker_stale += 1
                self._sync_stale_dropped += 1
                _diag_log(f"[REC-WORKER] stale frame dropped: age={age_sec:.1f}s, "
                          f"total_stale={self._worker_stale}",
                          throttle_key="rec_worker_stale", throttle_sec=2.0)
                continue

            # Fix B: 调用 inner 并跟踪处理结果
            frame_count_before = self._recording_frame_count
            try:
                self._on_recording_sync_inner(state_msg, now, *image_msgs)
                if self._recording_frame_count > frame_count_before:
                    self._worker_processed += 1
                else:
                    self._worker_skipped += 1
            except Exception as e:
                self._worker_errored += 1
                _diag_log(f"[REC-WORKER] error: {e}",
                          throttle_key="rec_worker_error", throttle_sec=2.0)
                self.get_logger().error(
                    f"Recording worker error: {e}",
                    throttle_duration_sec=5.0,
                )

            # Fix B: 每 5 秒汇总一次 worker 处理结果
            now_mono = time.monotonic()
            if now_mono - self._worker_stats_last_time >= 5.0:
                self._worker_stats_last_time = now_mono
                _diag_log(f"[REC-WORKER] 5s stats: processed={self._worker_processed}, "
                          f"skipped={self._worker_skipped}, errored={self._worker_errored}, "
                          f"stale_dropped={self._worker_stale}, "
                          f"frame_count={self._recording_frame_count}")
                # 重置周期计数器
                self._worker_processed = 0
                self._worker_skipped = 0
                self._worker_errored = 0
                self._worker_stale = 0

        self.get_logger().info("[RecordingWorker] 后台录制线程退出")

    def _on_recording_sync_inner(self, state_msg: JointState, now: float, *image_msgs):
        """录制回调内部逻辑，由 _on_recording_sync 调用（带 try-except 保护）。"""
        # 从 sync 消息中直接提取 14D state
        if len(state_msg.position) < 14:
            self.get_logger().warn(
                f"Recording sync: state message has {len(state_msg.position)} dims (< 14), skipping frame",
                throttle_duration_sec=5.0,
            )
            _diag_log(f"[REC-INNER] skip: state_dims={len(state_msg.position)} < 14",
                      throttle_key="rec_inner_state_short", throttle_sec=2.0)
            return
        state_14d = np.array(state_msg.position[:14], dtype=np.float64)

        # 从 sync 消息中直接转换 images
        images_rgb = {}
        for i, cam_name in enumerate(self.camera_names):
            try:
                img = self._cv_bridge.imgmsg_to_cv2(image_msgs[i], desired_encoding="rgb8")
                images_rgb[cam_name] = img
            except Exception as e:
                self.get_logger().warn(
                    f"Recording sync: failed to convert image from {cam_name}: {e}",
                    throttle_duration_sec=5.0,
                )
                _diag_log(f"[REC-INNER] skip: imgmsg_to_cv2 failed for {cam_name}: {e}",
                          throttle_key="rec_inner_img_fail", throttle_sec=2.0)
                return

        # Lazy-init recorder（首次 sync 回调时初始化，此时已有 image shape）
        # 为避免阻塞 executor，在后台线程中初始化
        if not self._recorder_initialized and not self._recorder_init_in_progress and not self._recorder_init_failed:
            self._recorder_init_in_progress = True
            _diag_log(f"[REC-INNER] lazy-init triggered: initialized={self._recorder_initialized}, "
                      f"in_progress=True, failed={self._recorder_init_failed}")

            def _background_init():
                """后台线程：初始化 DAggerRecorder + start_episode"""
                self._init_recorder_from_sync(state_14d, images_rgb)
                if self._recorder is not None:
                    # recorder 刚初始化，且 episode 已 active（start_episode 时 recorder 还不存在）
                    # 需要补调 start_episode
                    with self._recorder_lock:
                        self._recorder.start_episode(task=self.recording_task)
                    _diag_log("[REC-INNER] lazy-init done: recorder ready, start_episode called")
                else:
                    _diag_log("[REC-INNER] lazy-init done: recorder is STILL None (init failed)")
                # 无论成功失败，都清除 in_progress 标志
                self._recorder_init_in_progress = False

            init_thread = threading.Thread(target=_background_init, daemon=True, name="RecorderInit")
            init_thread.start()
            # 立即返回，不阻塞 executor
            return

        # 如果初始化正在进行中，跳过本帧录制
        if self._recorder_init_in_progress:
            _diag_log("[REC-INNER] skip: init_in_progress=True", throttle_key="rec_inner_init_progress", throttle_sec=2.0)
            return

        # 读取当前 mode 确定 control_source: 0=policy, 1=human
        with self._mode_lock:
            mode = self._mode
        control_source = 1 if mode == ControlMode.HUMAN else 0

        # Policy action: 使用 _last_policy_action（可用时）或零填充
        if self._last_policy_action is not None:
            policy_action = self._last_policy_action
        else:
            dim = self._inference_bridge.policy_action_dim if self._inference_bridge is not None else 8
            policy_action = np.zeros(dim, dtype=np.float32)

        # 录制帧
        with self._recorder_lock:
            if self._recorder is not None and self._episode_active and not self._episode_paused:
                # Fix G: 防御性对齐 policy_action 维度，避免 shape mismatch 导致 add_frame 抛异常
                expected_dim = self._recorder._policy_action_dim
                actual_dim = policy_action.shape[0]
                if actual_dim != expected_dim:
                    _diag_log(f"[REC-INNER] policy_action dim mismatch: actual={actual_dim} expected={expected_dim}, 对齐中",
                              throttle_key="rec_dim_mismatch", throttle_sec=5.0)
                    aligned = np.zeros(expected_dim, dtype=np.float32)
                    copy_len = min(actual_dim, expected_dim)
                    aligned[:copy_len] = policy_action[:copy_len]
                    policy_action = aligned
                self._recorder.add_frame(state_14d, images_rgb, policy_action, control_source)
                self._recording_frame_count += 1  # 只在实际录制时增加
                if self._recording_frame_count % 30 == 1:  # 每 30 帧打印一次
                    self.get_logger().info(f"[RecordingSync] Frame {self._recording_frame_count} recorded")
            else:
                # 诊断：为什么没有录制
                reasons = []
                if self._recorder is None:
                    reasons.append("recorder=None")
                if not self._episode_active:
                    reasons.append("episode_not_active")
                if self._episode_paused:
                    reasons.append("episode_paused")
                if reasons:
                    _diag_log(f"[REC-INNER] frame NOT recorded: {', '.join(reasons)}",
                              throttle_key="rec_inner_not_recorded", throttle_sec=2.0)
                    self.get_logger().warn(
                        f"[RecordingSync] Frame NOT recorded: {', '.join(reasons)}",
                        throttle_duration_sec=2.0
                    )

        # Fix H: 不在 worker 线程更新 _last_recording_ts，避免与回调线程竞争
        # throttle 控制只在 _on_recording_sync（ROS2 回调线程）中进行

    def _wait_preinit_and_start_episode(self):
        """等待预初始化完成后补调 start_episode。"""
        deadline = time.monotonic() + 15.0
        while self._recorder_init_in_progress:
            if self._mode == ControlMode.IDLE:
                self.get_logger().info("[RecorderWaitPreInit] 会话已停止，中止等待")
                return
            if time.monotonic() >= deadline:
                self.get_logger().error("[RecorderWaitPreInit] 等待预初始化超时（15s）")
                return
            time.sleep(0.05)

        with self._recorder_lock:
            if self._recorder is not None and self._episode_active:
                self._recorder.start_episode(task=self.recording_task)
                self.get_logger().info("[RecorderWaitPreInit] 预初始化完成，episode 已开始")
            else:
                self.get_logger().warn("[RecorderWaitPreInit] 预初始化完成但 recorder 为 None 或 episode 未激活")

    def _preinit_recorder(self):
        """
        Node 启动时预初始化 DAggerRecorder（后台线程）。

        在 __init__ 结束后立即启动，等待第一帧图像到达后创建 recorder。
        这样第一次 start_session 时 recorder 已经就绪，不需要等待 4+ 秒。

        注意：此时 policy 还没连接，policy_action_dim 使用默认值 8。
        """
        try:
            # Step 1: 等待第一帧图像到达（确定 camera_shapes）
            self.get_logger().info("[RecorderPreInit] 等待相机图像...")
            deadline = time.monotonic() + 30.0
            while True:
                with self._image_lock:
                    if len(self._cached_image_msgs) >= len(self.camera_names):
                        camera_shapes = {k: (m.height, m.width, 3) for k, m in self._cached_image_msgs.items()}
                        break
                if time.monotonic() >= deadline:
                    self.get_logger().error("[RecorderPreInit] 等待相机图像超时（30s），预初始化失败")
                    self._recorder_init_in_progress = False
                    return
                time.sleep(0.1)

            # Step 2: policy_action_dim — 等待 inference bridge warmup 确定实际维度
            policy_action_dim = 8  # 默认值
            if self.enable_policy and self._inference_bridge is not None:
                warmup_deadline = time.monotonic() + 60.0
                self.get_logger().info("[RecorderPreInit] 等待 inference bridge warmup 确定 policy_action_dim...")
                while time.monotonic() < warmup_deadline:
                    if self._mode == ControlMode.IDLE:
                        self.get_logger().info("[RecorderPreInit] 会话已停止，中止等待 warmup")
                        self._recorder_init_in_progress = False
                        return
                    bridge_dim = self._inference_bridge.policy_action_dim
                    # warmup 完成后 bridge 会从默认值 8 更新为实际值
                    if self._inference_bridge.warmup_done:
                        policy_action_dim = bridge_dim
                        self.get_logger().info(
                            f"[RecorderPreInit] warmup 完成，policy_action_dim={policy_action_dim}")
                        break
                    time.sleep(0.5)
                else:
                    self.get_logger().warn(
                        f"[RecorderPreInit] 等待 warmup 超时（60s），使用默认 policy_action_dim={policy_action_dim}"
                    )

            # Step 3: 初始化 DAggerRecorder
            self.get_logger().info(
                f"[RecorderPreInit] 开始初始化: cameras={list(camera_shapes.keys())}, "
                f"policy_action_dim={policy_action_dim}"
            )
            from dagger.core.data_recorder import DAggerRecorder
            self._recorder = DAggerRecorder(
                repo_id=self.repo_id,
                fps=self.recording_fps,
                root=self.dataset_root,
                camera_shapes=camera_shapes,
                policy_action_dim=policy_action_dim,
                use_videos=self.use_videos,
                image_writer_threads=self.image_writer_threads,
            )
            self._recorder_initialized = True
            self.get_logger().info(
                f"[RecorderPreInit] 初始化完成: repo={self.repo_id}, "
                f"cameras={list(camera_shapes.keys())}"
            )
        except Exception as e:
            self.get_logger().error(f"[RecorderPreInit] 初始化失败: {e}")
            # 不设置 _recorder_init_failed，允许后续 start_session 重试
        finally:
            self._recorder_init_in_progress = False

    def _init_recorder_eager(self):
        """
        在 start_session 时主动初始化 DAggerRecorder（后台线程）。

        与 _init_recorder_from_sync 的区别：
        - 不依赖 sync 回调，在 start_session 时立即触发
        - 等待 warmup 完成后才初始化（确保 policy_action_dim 正确）
        - 等待第一帧图像到达后才能确定 camera_shapes
        """
        try:
            # Step 1: 等待 warmup 完成（确保 policy_action_dim 正确）
            if self.enable_policy and self._inference_bridge is not None:
                self.get_logger().info("[RecorderEagerInit] 等待 warmup 完成...")
                deadline = time.monotonic() + 60.0
                while not self._inference_bridge.warmup_done:
                    if self._mode == ControlMode.IDLE:
                        self.get_logger().info("[RecorderEagerInit] 会话已停止，中止初始化")
                        self._recorder_init_in_progress = False
                        return
                    if time.monotonic() >= deadline:
                        self.get_logger().warn("[RecorderEagerInit] Warmup 超时（60s），使用默认 action_dim=8")
                        break
                    time.sleep(0.1)

            # Step 2: 等待第一帧图像到达（确定 camera_shapes）
            self.get_logger().info("[RecorderEagerInit] 等待相机图像...")
            deadline = time.monotonic() + 10.0
            while True:
                with self._image_lock:
                    if len(self._cached_image_msgs) >= len(self.camera_names):
                        # 所有相机都有图像了
                        camera_shapes = {k: (m.height, m.width, 3) for k, m in self._cached_image_msgs.items()}
                        break
                if self._mode == ControlMode.IDLE:
                    self.get_logger().info("[RecorderEagerInit] 会话已停止，中止初始化")
                    self._recorder_init_in_progress = False
                    return
                if time.monotonic() >= deadline:
                    self.get_logger().error("[RecorderEagerInit] 等待相机图像超时（10s），初始化失败")
                    self._recorder_init_failed = True
                    self._recorder_init_in_progress = False
                    return
                time.sleep(0.05)

            # Step 3: 确定 policy_action_dim
            policy_action_dim = 8  # default
            if self._inference_bridge is not None and self._inference_bridge.warmup_done:
                policy_action_dim = self._inference_bridge.policy_action_dim

            # Step 4: 初始化 DAggerRecorder
            self.get_logger().info(
                f"[RecorderEagerInit] 开始初始化: cameras={list(camera_shapes.keys())}, "
                f"policy_action_dim={policy_action_dim}"
            )
            from dagger.core.data_recorder import DAggerRecorder
            self._recorder = DAggerRecorder(
                repo_id=self.repo_id,
                fps=self.recording_fps,
                root=self.dataset_root,
                camera_shapes=camera_shapes,
                policy_action_dim=policy_action_dim,
                use_videos=self.use_videos,
                image_writer_threads=self.image_writer_threads,
            )
            self._recorder_initialized = True

            # Step 5: 如果 episode 已 active，补调 start_episode
            with self._recorder_lock:
                if self._episode_active:
                    self._recorder.start_episode(task=self.recording_task)
                    self.get_logger().info(
                        f"[RecorderEagerInit] 初始化完成，episode 已开始: repo={self.repo_id}"
                    )
                else:
                    self.get_logger().info(
                        f"[RecorderEagerInit] 初始化完成: repo={self.repo_id}"
                    )

        except Exception as e:
            self._recorder_init_failed = True
            self.get_logger().error(f"[RecorderEagerInit] 初始化失败: {e}")
        finally:
            self._recorder_init_in_progress = False

    def _init_recorder_from_sync(self, state_14d: np.ndarray, images_rgb: dict):
        """
        从 sync 回调中 lazy-init DAggerRecorder。

        与原 _init_recorder 逻辑相同，但使用 sync 消息中的 image 来确定 camera shapes。
        失败后设置标志位，避免每 tick 重试导致错误刷屏。
        """
        if self._recorder_initialized:
            return
        if self._recorder_init_failed:
            return
        if not self.enable_recording:
            return

        # Determine policy_action_dim
        policy_action_dim = 8  # default
        if self._inference_bridge is not None and self._inference_bridge.warmup_done:
            policy_action_dim = self._inference_bridge.policy_action_dim

        # Build camera_shapes from sync images
        camera_shapes = {}
        for cam_key, img in images_rgb.items():
            camera_shapes[cam_key] = img.shape  # (H, W, C)

        try:
            self._recorder = DAggerRecorder(
                repo_id=self.repo_id,
                fps=self.recording_fps,
                root=self.dataset_root,
                camera_shapes=camera_shapes,
                policy_action_dim=policy_action_dim,
                use_videos=self.use_videos,
                image_writer_threads=self.image_writer_threads,
            )
            self._recorder_initialized = True
            self.get_logger().info(
                f"DAggerRecorder initialized (from sync): repo={self.repo_id}, "
                f"cameras={list(camera_shapes.keys())}, "
                f"policy_action_dim={policy_action_dim}"
            )
        except Exception as e:
            self._recorder_init_failed = True
            self.get_logger().error(
                f"DAggerRecorder 初始化失败（不再重试）: {e}"
            )

    def _handle_new_episode(self, request, response):
        """在当前推理会话中开始新 episode（不重启推理）。"""
        # 必须在非 IDLE 模式（推理会话运行中）
        with self._mode_lock:
            mode = self._mode
        if mode == ControlMode.IDLE:
            response.success = False
            response.message = "推理会话未运行，请先调用 /dagger/start_session"
            return response

        if not self.enable_recording:
            response.success = False
            response.message = "Recording not enabled (enable_recording=false)"
            return response

        if self._recording_sync is None:
            response.success = False
            response.message = "Recording sync subscriber not created"
            return response

        with self._recorder_lock:
            if self._episode_saving:
                response.success = False
                response.message = "上一个 episode 正在保存中，请稍后再试"
                return response

            if self._episode_active:
                response.success = False
                response.message = "当前 episode 仍在录制，请先调用 /dagger/stop_episode"
                return response

            if self._recorder is not None:
                if self.max_episodes > 0 and self._recorder.num_episodes >= self.max_episodes:
                    response.success = False
                    response.message = (
                        f"已达最大 episode 数 ({self._recorder.num_episodes}/{self.max_episodes})"
                    )
                    return response
                self._recorder.start_episode(task=self.recording_task)

            self._episode_active = True
            self._episode_paused = self._session_paused  # If session is paused, start episode paused too
            self._last_policy_action = None
            self._last_recording_ts = None
            self._recording_frame_count = 0

        # 重置 gripper 状态（在 recorder_lock 外，用 vr_lock 保护）
        with self._vr_lock:
            self._gripper_closed = False
            self._current_grip = 1.0
            self._last_grip_value = 0.0
            self.vr_state.current_grip = 1.0
        self.get_logger().info("[NewEpisode] Gripper state reset: open")

        with self._recorder_lock:
            ep_num = (self._recorder.num_episodes + 1) if self._recorder is not None else 1

        response.success = True
        response.message = f"新 Episode {ep_num} 已开始 (task: {self.recording_task})"
        self.get_logger().info(f"New episode {ep_num} started in current session")
        return response

    def _handle_start_episode(self, request, response):
        """Start a new recording episode."""
        if not self.enable_recording:
            response.success = False
            response.message = "Recording not enabled (enable_recording=false)"
            return response

        if self._recording_sync is None:
            response.success = False
            response.message = "Recording sync subscriber not created"
            return response

        with self._recorder_lock:
            if self._episode_saving:
                response.success = False
                response.message = "上一个 episode 正在保存中，请稍后再试"
                return response

            if self._episode_active:
                response.success = False
                response.message = "Episode already active. Call /dagger/stop_episode first."
                return response

            # 如果 recorder 已初始化，检查 max_episodes 限制
            if self._recorder is not None:
                if self.max_episodes > 0 and self._recorder.num_episodes >= self.max_episodes:
                    response.success = False
                    response.message = (
                        f"Max episodes reached ({self._recorder.num_episodes}/{self.max_episodes})"
                    )
                    return response
                self._recorder.start_episode(task=self.recording_task)

            # 标记 episode 开始（recorder 可能在 sync 回调中 lazy-init）
            self._episode_active = True
            self._episode_paused = False
            self._last_policy_action = None
            self._last_recording_ts = None
            self._recording_frame_count = 0

            ep_num = (self._recorder.num_episodes + 1) if self._recorder is not None else 1

        response.success = True
        response.message = f"Episode {ep_num} started (task: {self.recording_task})"
        self.get_logger().info(f"Recording episode {ep_num} started")
        return response

    def _handle_stop_episode(self, request, response):
        """Stop the current recording episode and save (async)."""
        with self._recorder_lock:
            if self._episode_saving:
                response.success = False
                response.message = "上一个 episode 正在保存中，请稍后再试"
                return response

            if not self._episode_active or self._recorder is None:
                response.success = False
                response.message = "No active episode to stop"
                return response

            self._episode_active = False
            self._episode_paused = False
            self._episode_saving = True
            frame_count = self._recording_frame_count

        # 后台线程执行 finish_episode（可能耗时较长），立即返回
        threading.Thread(
            target=self._save_episode_background,
            args=(frame_count,),
            daemon=True,
            name="EpisodeSave",
        ).start()

        response.success = True
        response.message = f"Episode 保存已开始（{frame_count} 帧），后台执行中"
        self.get_logger().info(f"Recording episode save started ({frame_count} frames)")
        return response

    def _save_episode_background(self, frame_count: int):
        """后台线程：执行 finish_episode 保存。完成后清除 _episode_saving 标志。"""
        _t0 = time.monotonic()
        try:
            with self._recorder_lock:
                if self._recorder is not None:
                    self._recorder.finish_episode()
                    num_eps = self._recorder.num_episodes
                else:
                    num_eps = -1
            _elapsed_ms = (time.monotonic() - _t0) * 1000
            _diag_log(f"[EPISODE-SAVE] 完成: {num_eps} episodes, {frame_count} frames, "
                      f"耗时={_elapsed_ms:.0f}ms")
            self.get_logger().info(
                f"[EpisodeSave] Episode 保存完成（{num_eps} total, {_elapsed_ms:.0f}ms）"
            )
        except Exception as e:
            self.get_logger().error(f"[EpisodeSave] 保存失败: {e}")
        finally:
            self._episode_saving = False
            self._publish_status()

    def _handle_pause_episode(self, request, response):
        """Pause the current recording episode (stop recording frames, keep buffer)."""
        with self._recorder_lock:
            if not self._episode_active:
                response.success = False
                response.message = "No active episode to pause"
                return response

            if self._episode_paused:
                response.success = False
                response.message = "Episode already paused"
                return response

            self._episode_paused = True
            frame_count = self._recording_frame_count

        response.success = True
        response.message = f"Episode paused ({frame_count} frames recorded)"
        self.get_logger().info(f"Recording episode paused ({frame_count} frames)")
        self._publish_status()
        return response

    def _handle_discard_episode(self, request, response):
        """Discard the current recording episode (clear buffer, don't save)."""
        with self._recorder_lock:
            if not self._episode_active:
                response.success = False
                response.message = "No active episode to discard"
                return response

            self._episode_active = False
            self._episode_paused = False

            if self._recorder is not None:
                self._recorder.discard_episode()

            frame_count = self._recording_frame_count

        response.success = True
        response.message = f"Episode discarded ({frame_count} frames dropped)"
        self.get_logger().info(f"Recording episode discarded ({frame_count} frames)")
        self._publish_status()
        return response

    # =========================================================================
    # 统一会话控制 (start_session / stop_session)
    # =========================================================================

    def _handle_start_session(self, request, response):
        """
        开始推理会话：enable_control(POLICY) + start_episode。

        流程：
        1. 启用 driver follow
        2. 如果 enable_policy：后台线程连接 PolicyServer + warmup，完成后切换到 POLICY
           如果 !enable_policy：直接进入 HUMAN 模式（纯 VR 遥操）
        3. 如果录制已启用，标记 episode 开始（等 warmup 后初始化 recorder）

        注意：connect() 和 warmup 在后台线程执行，不阻塞 ROS2 executor。
        """
        with self._mode_lock:
            if self._mode != ControlMode.IDLE:
                response.success = False
                response.message = f"会话已在运行中（当前模式: {self._mode.name}）"
                return response

            # Step 1: 启用 driver follow
            if not self.dry_run:
                self.get_logger().info("[start_session] 启用 driver follow...")
                self._call_driver_service(
                    self.driver_enable_follow_client,
                    "/driver/enable_follow",
                )

            # Step 2: 连接 PolicyServer
            if self.enable_policy and self._inference_bridge is not None:
                # 先切换到 POLICY 模式，让 _execution_tick 开始喂观测
                self._mode = ControlMode.POLICY
                self._session_paused = False  # 确保新会话不继承前一个会话的暂停状态
                # 清除上一次会话的 stop 标志（否则新会话的 exec loop 会立即退出）
                # 创建新的 Event，避免影响可能仍在退出的旧线程
                self._policy_exec_stop = threading.Event()
                self._policy_exec_stop_time = 0.0
                # 后台线程执行 connect + start_inference_loop（不阻塞 executor）
                threading.Thread(
                    target=self._connect_and_start_inference,
                    daemon=True,
                ).start()
            else:
                # 策略推理未启用：直接进入 HUMAN 模式（纯 VR 遥操）
                self._mode = ControlMode.HUMAN
                # HUMAN-only 模式无需 Motion Gating，直接解锁
                self._recording_ready = True
                self._motion_gate.set()
                self.get_logger().info("[start_session] 策略推理未启用，进入纯 VR 模式（motion gate 已解锁）")

        # Step 3: 自动开始录制（在 mode_lock 外操作，避免死锁）
        recording_msg = ""
        if self.enable_recording:
            with self._recorder_lock:
                if not self._episode_active:
                    if self._recorder is not None:
                        if self.max_episodes > 0 and self._recorder.num_episodes >= self.max_episodes:
                            recording_msg = f"（录制跳过：已达最大 episode 数 {self.max_episodes}）"
                            _diag_log(f"[START-SESSION] 录制跳过：已达 max_episodes={self.max_episodes}")
                        else:
                            self._recorder.start_episode(task=self.recording_task)
                            self._episode_active = True
                            self._episode_paused = False
                            self._last_policy_action = None
                            self._last_recording_ts = None
                            self._recording_frame_count = 0
                            recording_msg = "，录制已开始"
                            _diag_log("[START-SESSION] recorder 已就绪，start_episode 已调用")
                    else:
                        # recorder 尚未初始化
                        self._recorder_init_failed = False
                        self._episode_active = True
                        self._episode_paused = False
                        self._last_policy_action = None
                        self._last_recording_ts = None
                        self._recording_frame_count = 0

                        if self._recorder_init_in_progress:
                            # 预初始化正在进行中，等待它完成（最多 10 秒）
                            recording_msg = "，录制已开始（等待预初始化完成）"
                            _diag_log(f"[START-SESSION] recorder=None, init_in_progress=True → 等待预初始化")
                            # 释放 recorder_lock 后在后台等待
                            threading.Thread(
                                target=self._wait_preinit_and_start_episode,
                                daemon=True,
                                name="RecorderWaitPreInit"
                            ).start()
                        else:
                            # 预初始化失败或未启动，立即启动后台初始化
                            self._recorder_initialized = False
                            self._recorder_init_in_progress = True
                            _diag_log(f"[START-SESSION] recorder=None, init_in_progress=False → 启动 eager init")
                            threading.Thread(
                                target=self._init_recorder_eager,
                                daemon=True,
                                name="RecorderEagerInit"
                            ).start()
                            recording_msg = "，录制已开始（后台初始化中）"
                else:
                    _diag_log(f"[START-SESSION] episode_active 已为 True，跳过录制初始化")

        mode_msg = "POLICY 模式，连接中" if self.enable_policy else "HUMAN 模式（纯 VR）"
        response.success = True
        response.message = f"会话已开始（{mode_msg}）{recording_msg}"
        self.get_logger().info(f"[start_session] 会话已开始（{mode_msg}）{recording_msg}")
        self._publish_status()
        return response

    def _connect_and_start_inference(self):
        """后台线程：连接 PolicyServer + 启动推理循环。不阻塞 ROS2 executor。"""
        try:
            self.get_logger().info("[start_session] 后台连接 PolicyServer...")
            self._inference_bridge.connect()
            self._inference_bridge.start_inference_loop()
            self.get_logger().info("[start_session] PolicyServer 已连接，推理循环已启动")

            # 关键：必须在等待 warmup 之前就启动 obs_feed_worker！
            # warmup 的 _measure_inference_time() 需要 obs 才能执行推理，
            # 而 obs 只有在 obs_feed_worker 运行后才会被持续喂入 bridge。
            # 如果不提前启动，warmup 会因为等不到 obs 而超时，导致 feasible=False
            # 进入 degraded mode，推理周期从 ~130ms 退化到 ~1000ms。
            self._start_obs_feed_worker()
            self.get_logger().info("[start_session] Obs feed worker 已启动（供 warmup 使用）")

            # Wait for warmup to complete before clearing buffer.
            # Warmup validates the pipeline but its actions are based on stale
            # (stationary) observations — executing them would cause a jump.
            # Poll with short intervals so we can detect stop requests.
            self.get_logger().info("[start_session] 等待 warmup 完成...")
            deadline = time.monotonic() + 60.0
            while not self._inference_bridge.warmup_done:
                if self._policy_exec_stop.is_set() or self._mode != ControlMode.POLICY:
                    self.get_logger().info("[start_session] 等待 warmup 期间收到停止请求，中止启动")
                    return
                if time.monotonic() >= deadline:
                    self.get_logger().warn("[start_session] Warmup 超时（60s），继续启动")
                    break
                time.sleep(0.1)
            # Double-check stop wasn't requested during final iteration
            if self._policy_exec_stop.is_set() or self._mode != ControlMode.POLICY:
                self.get_logger().info("[start_session] 启动已取消")
                return
            self._inference_bridge.reset_buffer()
            self.get_logger().info("[start_session] Buffer 已重置（丢弃 warmup 残留 action + last_popped）")

            # 关键：用当前真实状态刷新 observation，防止推理线程用旧 obs 产生跳变 action。
            # warmup 期间 observation 没有被更新（IDLE→POLICY 后 exec loop 还没启动），
            # 如果用户在 stop_session 后回了 home，_obs_state_14d 还是上个 session 的旧值。
            self._update_inference_observation()
            self.get_logger().info("[start_session] Observation 已刷新为当前状态")

            # 再次完全清空 buffer（含 last_popped）：防止 session 间 hold 旧 action 导致跳变
            self._inference_bridge.reset_buffer()

            # warmup 竞态修复：obs 刷新 + buffer 清理完成，恢复推理线程。
            # 此时推理线程使用的是当前真实 observation，不会产生基于旧 obs 的跳变 action。
            self._inference_bridge.resume()
            self.get_logger().info("[start_session] 推理线程已恢复（obs 已刷新，buffer 已清空）")

            # === Motion Gating 解锁序列 ===
            # Step A: 允许录制 sync 回调开始入队
            self._recording_ready = True
            _diag_log("[MOTION-GATE] recording_ready=True, 等待首帧 sync 回调...")

            # Step B: 等待至少一帧 sync 回调触发（确保录制链路已开始工作）
            if self.enable_recording and self._episode_active:
                _gate_deadline = time.monotonic() + 2.0  # 最多等 2 秒
                _initial_frame_count = self._recording_frame_count
                while self._recording_frame_count == _initial_frame_count:
                    if self._policy_exec_stop.is_set() or self._mode != ControlMode.POLICY:
                        self.get_logger().info("[MOTION-GATE] 等待首帧期间收到停止请求")
                        return
                    if time.monotonic() >= _gate_deadline:
                        self.get_logger().warn("[MOTION-GATE] 等待首帧 sync 超时（2s），继续解锁")
                        break
                    time.sleep(0.02)

            # Step C: 完全清空 buffer + last_popped（丢弃等待期间积累的 action + 旧 session 残留）
            self._inference_bridge.reset_buffer()

            # Step C.5: Publisher 预热 — 消化首次 publish 的 DDS 初始化开销（5-22ms）
            # 在 motion gate 解锁前发送预热消息，避免正式 action 下发时的延迟
            # 注意：使用当前真实 arm state 作为预热内容，避免发送全零关节角度导致机械臂突然移动
            _t_warmup_pub = time.monotonic()
            state_14d = self._get_state_14d()
            dummy_msg = JointState()
            dummy_msg.header.stamp = self.get_clock().now().to_msg()
            dummy_msg.header.frame_id = self.frame_id
            dummy_msg.name = ["joint_1", "joint_2", "joint_3", "joint_4",
                              "joint_5", "joint_6", "joint_7", "gripper"]
            if state_14d is not None:
                # 用当前关节角度 + 夹爪值预热（发送当前位姿 = 机械臂不动）
                dummy_msg.position = list(state_14d[:7]) + [float(state_14d[7])]
            else:
                # 无 state 时跳过预热，避免发送危险指令
                _diag_log("[MOTION-GATE] Publisher 预热跳过: 无 arm state")
                dummy_msg = None
            if dummy_msg is not None:
                self.action_joint_pub.publish(dummy_msg)
                # 注意：不预热 action_pose_pub，因为 dummy_msg 是 joint 格式，
                # 发到 pose topic 会被 driver 当作非法 pose 执行，导致控制器进入错误状态
                _t_warmup_pub_done = time.monotonic()
                _diag_log(f"[MOTION-GATE] Publisher 预热完成: {(_t_warmup_pub_done-_t_warmup_pub)*1000:.2f}ms")

            # Step D: 解锁 action 下发
            self._motion_gate.set()
            _diag_log("[MOTION-GATE] motion_gate SET, action 下发已解锁")

            self._start_obs_feed_worker()
            self._start_policy_exec_loop()
        except Exception as e:
            self.get_logger().error(f"[start_session] PolicyServer 连接失败: {e}")
            # 回滚到 IDLE（仅当 stop_session 还没先回滚时）
            self._recording_ready = False
            self._motion_gate.clear()
            need_rollback = False
            with self._mode_lock:
                if self._mode == ControlMode.POLICY:
                    self._mode = ControlMode.IDLE
                    need_rollback = True
            self._stop_policy_exec_loop()
            self._stop_obs_feed_worker()
            if need_rollback and not self.dry_run:
                self._call_driver_service(
                    self.driver_disable_follow_client,
                    "/driver/disable_follow",
                )
            self.get_logger().error("[start_session] 已回滚到 IDLE 模式")

    def _handle_stop_session(self, request, response):
        """
        停止推理会话：暂停录制（等待前端确认保存/丢弃）+ 切换到 IDLE。

        流程：
        1. 如果有活跃录制，暂停录制（不保存也不丢弃，等前端确认）
        2. 切换到 IDLE（立即生效，不阻塞 executor）
        3. 后台停止推理 + 禁用 driver follow
        """
        with self._mode_lock:
            if self._mode == ControlMode.IDLE:
                response.success = False
                response.message = "当前已是 IDLE 模式，无需停止"
                return response

            old_mode = self._mode

            # Step 0: 立即停止 action 执行（最高优先级）
            # 执行线程在下一个 sleep 周期内（最多 ~33ms）停止发送 action
            self._policy_exec_stop.set()

            # Step 1: 暂停录制（如果有活跃 episode）
            recording_msg = ""
            with self._recorder_lock:
                if self._episode_active and not self._episode_paused:
                    self._episode_paused = True
                    frame_count = self._recording_frame_count
                    recording_msg = f"，录制已暂停（{frame_count} 帧，等待确认保存/丢弃）"
                elif self._episode_active and self._episode_paused:
                    frame_count = self._recording_frame_count
                    recording_msg = f"，录制已暂停（{frame_count} 帧）"

            # Step 2: 立即切换到 IDLE（让 _execution_tick 停止执行动作）
            self._mode = ControlMode.IDLE
            self._session_paused = False
            self.vr_state.is_active = False
            self.vr_state.is_initialized = False
            self._last_executed_joints = None

            # Motion Gating 重置：下次 start_session 需要重新走解锁流程
            self._recording_ready = False
            self._motion_gate.clear()

        # Stop policy exec loop (outside mode_lock to avoid deadlock)
        self._stop_policy_exec_loop()

        # Step 3: 在 mode_lock 外停止 obs feed worker + 推理 + 禁用 driver follow（避免阻塞 executor）
        self._stop_obs_feed_worker()
        if self._inference_bridge is not None:
            self._inference_bridge.stop_inference_loop()
            # 完全重置 buffer + last_popped，防止下次 start_session 时 hold 旧 action 导致跳变
            self._inference_bridge.reset_buffer()

        if not self.dry_run:
            self.get_logger().info("[stop_session] 禁用 driver follow...")
            self._call_driver_service(
                self.driver_disable_follow_client,
                "/driver/disable_follow",
            )

        response.success = True
        response.message = f"推理会话已停止（{old_mode.name} → IDLE）{recording_msg}"
        self.get_logger().info(f"[stop_session] 会话已停止（{old_mode.name} → IDLE）{recording_msg}")
        self._publish_status()
        return response

    def _handle_pause_session(self, request, response):
        """
        暂停推理会话：停止 action 执行 + 暂停推理，但保持 PolicyServer 连接。
        允许用户在暂停期间保存/丢弃录制、回 Home、开始新 Episode。
        """
        with self._mode_lock:
            if self._mode != ControlMode.POLICY:
                response.success = False
                response.message = f"当前模式 {self._mode.name} 不支持暂停（仅 POLICY 模式）"
                return response

            if self._session_paused:
                response.success = True
                response.message = "会话已处于暂停状态"
                return response

            # 立即停止执行线程
            self._policy_exec_stop.set()
            self._session_paused = True

        # 在 mode_lock 外做清理
        self._stop_policy_exec_loop()

        # 暂停推理循环（保持 gRPC 连接）
        if self._inference_bridge is not None:
            self._inference_bridge.pause()

        # 暂停录制（如果有活跃 episode）
        recording_msg = ""
        with self._recorder_lock:
            if self._episode_active and not self._episode_paused:
                self._episode_paused = True
                recording_msg = f"，录制已暂停（{self._recording_frame_count} 帧）"

        response.success = True
        response.message = f"推理已暂停（连接保持）{recording_msg}"
        self.get_logger().info(f"[pause_session] {response.message}")
        self._publish_status()
        return response

    def _handle_resume_session(self, request, response):
        """
        恢复推理会话：清空 buffer + 恢复推理 + 重启执行线程。
        """
        with self._mode_lock:
            if self._mode != ControlMode.POLICY:
                response.success = False
                response.message = f"当前模式 {self._mode.name} 不支持恢复（仅 POLICY 模式）"
                return response

            if not self._session_paused:
                response.success = False
                response.message = "会话未暂停，无需恢复"
                return response

            self._session_paused = False

        # 恢复推理循环（保持 gRPC 连接）
        if self._inference_bridge is not None:
            # 先 resume 推理线程
            self._inference_bridge.resume()

        # 恢复录制（如果暂停前有活跃 episode）
        recording_msg = ""
        with self._recorder_lock:
            if self._episode_active and self._episode_paused:
                self._episode_paused = False
                recording_msg = "，录制已恢复"

        # 标记：执行线程启动后需要清空 buffer（丢弃暂停期间的旧 action）
        self._clear_buffer_on_exec_start = True

        # 重启执行线程
        self._start_policy_exec_loop()

        response.success = True
        response.message = f"推理已恢复{recording_msg}"
        self.get_logger().info(f"[resume_session] {response.message}")
        self._publish_status()
        return response

    # =========================================================================
    # Status publishing
    # =========================================================================

    def _probe_server_loop(self):
        """Background thread: periodically probe PolicyServer reachability.

        Runs every 2 seconds. Skips probe when inference_bridge is already
        connected (session active). Updates _server_reachable for status publishing.
        Runs in a daemon thread to avoid blocking the ROS2 executor.
        """
        while not self._server_probe_stop.is_set():
            if self._inference_bridge is not None and self._inference_bridge.is_connected:
                # Already connected via session, no need to probe
                pass
            else:
                self._server_reachable = InferenceBridge.probe_server(
                    self.server_address, timeout_sec=1.0,
                )
            self._server_probe_stop.wait(2.0)

    def _publish_status(self):
        """Publish DAgger status for monitoring."""
        msg = String()
        with self._mode_lock:
            mode = self._mode
        status = {
            "mode": mode.name,
            "session_active": mode != ControlMode.IDLE,
            "session_paused": self._session_paused,
            "vr_active": self.vr_state.is_active,
            "cmd_count": self._cmd_count,
            "safety_rejects": self._safety_reject_count,
            # 实际控制频率监控
            "control": {
                "actual_hz": round(self._get_control_hz(), 1),
                "target_hz": self.f_exec if mode == ControlMode.POLICY else (
                    self.control_hz if mode == ControlMode.HUMAN else 0.0
                ),
                "source": mode.name,
            },
        }
        if self._inference_bridge is not None:
            if self._inference_bridge.is_connected:
                # 会话已连接：使用实际连接状态
                status["inference"] = {
                    "warmup_done": self._inference_bridge.warmup_done,
                    "paused": self._inference_bridge.is_paused,
                    "buffer_size": self._inference_bridge.buffer_size,
                    "infer_count": self._inference_bridge._infer_count,
                }
                status["server"] = {
                    "connected": True,
                    "ready": self._inference_bridge.warmup_done,
                    "policy_type": self.policy_type,
                    "pretrained_path": self.pretrained_path,
                }
            elif mode != ControlMode.IDLE:
                # 会话已开始但后台连接中：显示 connecting 状态（非"纯 VR 模式"）
                status["inference"] = {
                    "warmup_done": False,
                    "paused": False,
                    "buffer_size": 0,
                    "infer_count": 0,
                }
                status["server"] = {
                    "connected": False,
                    "ready": False,
                    "policy_type": self.policy_type,
                    "pretrained_path": self.pretrained_path,
                }
            else:
                # IDLE 模式：使用 probe 结果，让 UI 在连接前就能看到 Server 状态
                status["server"] = {
                    "connected": self._server_reachable,
                    "ready": self._server_reachable,
                    "policy_type": self.policy_type,
                    "pretrained_path": self.pretrained_path,
                }
        # 录制状态（即使 recorder 未初始化也要发布，让 UI 知道 episode 状态）
        if self.enable_recording:
            if self._recorder is not None:
                # Recorder 已初始化：使用 recorder 的实际帧数
                status["recording"] = {
                    "episode_active": self._episode_active,
                    "episode_paused": self._episode_paused,
                    "num_episodes": self._recorder.num_episodes,
                    "frame_count": self._recording_frame_count,
                    "recorder_ready": True,
                }
                _diag_log(f"[STATUS] recorder=ready, frame_count={self._recording_frame_count}, "
                          f"episode_active={self._episode_active}",
                          throttle_key="status_rec_ready", throttle_sec=5.0)
            else:
                # Recorder 未初始化（warmup 中或 lazy-init 未触发）：使用实际计数
                status["recording"] = {
                    "episode_active": self._episode_active,
                    "episode_paused": self._episode_paused,
                    "num_episodes": 0,
                    "frame_count": self._recording_frame_count,
                    "recorder_ready": False,
                    "recorder_init_in_progress": self._recorder_init_in_progress,
                }
                _diag_log(f"[STATUS] recorder=None, frame_count={self._recording_frame_count}, "
                          f"episode_active={self._episode_active}, "
                          f"init_in_progress={self._recorder_init_in_progress}, "
                          f"initialized={self._recorder_initialized}, "
                          f"init_failed={self._recorder_init_failed}",
                          throttle_key="status_rec_none", throttle_sec=5.0)
        msg.data = json.dumps(status)
        self.status_pub.publish(msg)

    def _log_stats(self):
        """Periodic stats logging."""
        with self._mode_lock:
            mode = self._mode
        active = "active" if self.vr_state.is_active else "standby"

        parts = [f"mode={mode.name}/{active}", f"cmds={self._cmd_count}"]

        if self._safety_reject_count > 0:
            parts.append(f"safety_rejects={self._safety_reject_count}")

        if self._inference_bridge is not None:
            bridge = self._inference_bridge
            parts.append(f"buf={bridge.buffer_size}")
            parts.append(f"infers={bridge._infer_count}")
            if bridge._t_inf_avg > 0:
                parts.append(f"t_inf_avg={bridge._t_inf_avg:.3f}s")

        if self._recorder is not None:
            rec_status = "REC" if self._episode_active else "idle"
            parts.append(f"rec={rec_status}")
            parts.append(f"episodes={self._recorder.num_episodes}")
            if self._episode_active:
                parts.append(f"frames={self._recorder._frame_count}")

        self.get_logger().info(f"[DAgger] {' | '.join(parts)}")

    # =========================================================================
    # Cleanup
    # =========================================================================

    def destroy_node(self):
        """Clean up resources."""
        self.get_logger().info("Shutting down DAgger node...")
        # Stop server probe thread
        self._server_probe_stop.set()
        if self._server_probe_thread is not None:
            self._server_probe_thread.join(timeout=3.0)
        # 停止录制后台线程（在 recorder 清理之前，确保队列中的帧处理完毕）
        self._recording_stop.set()
        if self._recording_worker_thread is not None and self._recording_worker_thread.is_alive():
            self._recording_worker_thread.join(timeout=3.0)
        # Stop active recording episode
        with self._recorder_lock:
            if self._recorder is not None:
                if self._episode_active:
                    self.get_logger().info("Finishing active recording episode before shutdown...")
                    self._episode_active = False
                    self._episode_paused = False
                    self._recorder.finish_episode()
                self._recorder.close()
                self.get_logger().info("DAggerRecorder closed")
        self._stop_obs_feed_worker()
        if self._inference_bridge is not None:
            self._inference_bridge.disconnect()
        super().destroy_node()


# =============================================================================
# Entry point
# =============================================================================

def main(args=None):
    rclpy.init(args=args)
    node = DAggerNode()
    _cleanup_done = False

    def _cleanup():
        nonlocal _cleanup_done
        if _cleanup_done:
            return
        _cleanup_done = True
        node.destroy_node()
        rclpy.shutdown()

    def _sigterm_handler(signum, frame):
        """SIGTERM handler: ensure clean shutdown of RM+ and recording."""
        node.get_logger().warn("收到 SIGTERM 信号，正在安全关闭...")
        _cleanup()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
