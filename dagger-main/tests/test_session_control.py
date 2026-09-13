"""
统一会话控制（start_session / stop_session）集成测试

测试覆盖:
- 7.1: start_session 正常流程（IDLE → POLICY + 录制）
- 7.2: start_session 失败回滚（PolicyServer 连接失败 → 保持 IDLE）
- 7.3: start_session 重复调用（已在运行中 → 拒绝）
- 7.4: stop_session 正常流程（POLICY → IDLE + 暂停录制）
- 7.5: stop_session 在 IDLE 时调用（拒绝）
- 7.6: stop_session 在 HUMAN 模式下调用（VR 接管中停止）
- 7.7: start_session 策略未启用时拒绝
- 7.8: start_session 录制未启用时仅启动推理
- 7.9: start_session recorder 尚未 lazy-init 时标记 episode
- 7.10: stop_session 录制已暂停时不重复暂停
- 7.11: _publish_status 包含 server 字段
- 7.12: _publish_status 包含 recording 字段

注意: dagger_node.py 依赖 ROS2 (rclpy, sensor_msgs 等)，
需要在模块加载前通过 sys.modules 注入 mock 来绕过。
"""
import os
import sys
import time
import threading
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass, field

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# ============================================================================
# 在 import dagger_node 之前，mock 掉所有 ROS2 / 外部依赖模块
# ============================================================================

_ROS2_MODULES = [
    "rclpy", "rclpy.node", "rclpy.executors", "rclpy.callback_groups",
    "rclpy.qos", "rclpy.parameter",
    "geometry_msgs", "geometry_msgs.msg",
    "std_msgs", "std_msgs.msg",
    "sensor_msgs", "sensor_msgs.msg",
    "std_srvs", "std_srvs.srv",
    "cv_bridge",
    "message_filters",
]

_mocks = {}
for mod_name in _ROS2_MODULES:
    if mod_name not in sys.modules:
        _mocks[mod_name] = MagicMock()
        sys.modules[mod_name] = _mocks[mod_name]


class _FakeNode:
    """Minimal stand-in for rclpy.node.Node."""
    def __init__(self, *args, **kwargs):
        pass
    def get_logger(self):
        return MagicMock()
    def declare_parameter(self, *args, **kwargs):
        return MagicMock()
    def get_parameter(self, *args, **kwargs):
        m = MagicMock()
        m.get_parameter_value.return_value = MagicMock()
        return m
    def create_subscription(self, *args, **kwargs):
        return MagicMock()
    def create_publisher(self, *args, **kwargs):
        return MagicMock()
    def create_timer(self, *args, **kwargs):
        return MagicMock()
    def create_service(self, *args, **kwargs):
        return MagicMock()


sys.modules["rclpy.node"].Node = _FakeNode

sys.modules["sensor_msgs.msg"].JointState = MagicMock
sys.modules["sensor_msgs.msg"].Image = MagicMock
sys.modules["geometry_msgs.msg"].PoseStamped = MagicMock
sys.modules["std_msgs.msg"].Float32 = MagicMock
sys.modules["std_msgs.msg"].String = MagicMock
sys.modules["std_srvs.srv"].Trigger = MagicMock

sys.modules["cv_bridge"].CvBridge = MagicMock

sys.modules["message_filters"].ApproximateTimeSynchronizer = MagicMock
sys.modules["message_filters"].Subscriber = MagicMock

_vr_utils_mock = MagicMock()
sys.modules["vr_utils"] = _vr_utils_mock

from dagger.dagger_node import DAggerNode, ControlMode  # noqa: E402


# ============================================================================
# Helpers
# ============================================================================

@dataclass
class FakeVRState:
    """模拟 VRControlState"""
    is_active: bool = False
    is_initialized: bool = False
    current_trigger: float = 0.0
    current_vr_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    current_vr_quat: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0, 1.0]))
    init_vr_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    init_vr_quat: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0, 1.0]))
    init_arm_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    init_arm_quat: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0, 1.0]))
    current_grip: float = 0.0


def _make_mock_node(
    mode=ControlMode.IDLE,
    enable_policy=True,
    enable_recording=True,
    dry_run=True,
    has_bridge=True,
    has_recorder=True,
    episode_active=False,
    episode_paused=False,
    max_episodes=0,
):
    """创建 mock DAggerNode 实例，绕过 __init__。"""
    node = object.__new__(DAggerNode)

    node._mode = mode
    node._mode_lock = threading.Lock()
    node._recorder_lock = threading.Lock()

    # 配置参数
    node.dry_run = dry_run
    node.enable_policy = enable_policy
    node.enable_recording = enable_recording
    node.recording_task = "pick and place"
    node.max_episodes = max_episodes
    node.policy_type = "act"
    node.pretrained_path = "outputs/test_model/checkpoints/last/pretrained_model"
    node.f_exec = 20.0
    node.control_hz = 30.0
    node._server_reachable = False

    # VR 状态
    node.vr_state = FakeVRState()

    # 推理桥
    if has_bridge:
        node._inference_bridge = MagicMock()
        node._inference_bridge.is_connected = True
        node._inference_bridge.warmup_done = True
        node._inference_bridge.is_paused = False
        node._inference_bridge.buffer_size = 5
        node._inference_bridge._infer_count = 10
    else:
        node._inference_bridge = None

    # 录制器
    if has_recorder:
        node._recorder = MagicMock()
        node._recorder.num_episodes = 0
        node._recorder._frame_count = 0
    else:
        node._recorder = None
    node._recorder_initialized = has_recorder

    # 录制状态
    node._episode_active = episode_active
    node._episode_paused = episode_paused
    node._last_policy_action = None
    node._last_recording_ts = None
    node._recording_frame_count = 0

    # 控制状态
    node._cmd_count = 0
    node._safety_reject_count = 0
    node._last_executed_joints = None
    node._cmd_timestamps = []
    node._cmd_ts_lock = threading.Lock()
    node._CMD_WINDOW_SEC = 2.0
    node._session_paused = False
    node._recorder_init_in_progress = False
    node._recorder_init_failed = False

    # Driver service 客户端
    node.driver_enable_follow_client = MagicMock()
    node.driver_disable_follow_client = MagicMock()

    # Policy exec loop (independent thread, async_client style)
    node._policy_exec_thread = None
    node._policy_exec_stop = threading.Event()
    node._start_policy_exec_loop = MagicMock()
    node._stop_policy_exec_loop = MagicMock()

    # Status publisher
    node.status_pub = MagicMock()

    # Logger
    node.get_logger = MagicMock(return_value=MagicMock())

    # _call_driver_service mock
    node._call_driver_service = MagicMock()

    return node


def _make_request_response():
    """创建 mock Trigger Request/Response"""
    request = MagicMock()
    response = MagicMock()
    response.success = False
    response.message = ""
    return request, response


# ============================================================================
# 7.1: start_session 正常流程
# ============================================================================

class TestStartSessionNormal:
    """start_session: IDLE → POLICY + 录制"""

    def test_mode_changes_to_policy(self):
        """start_session 应将模式从 IDLE 切换到 POLICY"""
        node = _make_mock_node()
        req, resp = _make_request_response()

        result = node._handle_start_session(req, resp)

        assert result.success is True
        assert node._mode == ControlMode.POLICY

    def test_inference_bridge_connected(self):
        """start_session 应调用 inference_bridge.connect() 和 start_inference_loop()"""
        node = _make_mock_node()
        req, resp = _make_request_response()

        node._handle_start_session(req, resp)

        node._inference_bridge.connect.assert_called_once()
        node._inference_bridge.start_inference_loop.assert_called_once()

    def test_recording_started(self):
        """start_session 应自动开始录制"""
        node = _make_mock_node()
        req, resp = _make_request_response()

        node._handle_start_session(req, resp)

        assert node._episode_active is True
        assert node._episode_paused is False
        node._recorder.start_episode.assert_called_once_with(task="pick and place")

    def test_recording_frame_count_reset(self):
        """start_session 应重置录制帧计数"""
        node = _make_mock_node()
        node._recording_frame_count = 42
        req, resp = _make_request_response()

        node._handle_start_session(req, resp)

        assert node._recording_frame_count == 0

    def test_response_message_contains_recording(self):
        """响应消息应包含录制信息"""
        node = _make_mock_node()
        req, resp = _make_request_response()

        result = node._handle_start_session(req, resp)

        assert "POLICY" in result.message
        assert "录制已开始" in result.message

    def test_status_published(self):
        """start_session 应发布状态更新"""
        node = _make_mock_node()
        # 需要 mock _publish_status 因为它访问更多属性
        node._publish_status = MagicMock()
        req, resp = _make_request_response()

        node._handle_start_session(req, resp)

        node._publish_status.assert_called_once()

    def test_driver_follow_enabled_when_not_dry_run(self):
        """非 dry_run 模式下应启用 driver follow"""
        node = _make_mock_node(dry_run=False)
        req, resp = _make_request_response()

        node._handle_start_session(req, resp)

        node._call_driver_service.assert_called_once()

    def test_driver_follow_skipped_in_dry_run(self):
        """dry_run 模式下不调用 driver follow"""
        node = _make_mock_node(dry_run=True)
        req, resp = _make_request_response()

        node._handle_start_session(req, resp)

        node._call_driver_service.assert_not_called()


# ============================================================================
# 7.2: start_session 失败回滚
# ============================================================================

class TestStartSessionFailure:
    """start_session: PolicyServer 连接失败 → 后台回滚到 IDLE"""

    def test_connect_failure_rollback_to_idle(self):
        """PolicyServer 连接失败时，后台线程应回滚到 IDLE"""
        node = _make_mock_node()
        node._inference_bridge.connect.side_effect = RuntimeError("gRPC 连接超时")
        req, resp = _make_request_response()

        # start_session 立即返回 success（连接在后台线程）
        result = node._handle_start_session(req, resp)
        assert result.success is True

        # 等后台线程完成回滚
        import time
        deadline = time.monotonic() + 2.0
        while node._mode != ControlMode.IDLE and time.monotonic() < deadline:
            time.sleep(0.05)
        assert node._mode == ControlMode.IDLE

    def test_connect_failure_rollback_driver(self):
        """连接失败时后台线程应回滚 driver follow（非 dry_run）"""
        node = _make_mock_node(dry_run=False)
        node._inference_bridge.connect.side_effect = RuntimeError("gRPC 连接超时")
        req, resp = _make_request_response()

        node._handle_start_session(req, resp)

        # 等后台线程完成回滚
        import time
        deadline = time.monotonic() + 2.0
        while node._mode != ControlMode.IDLE and time.monotonic() < deadline:
            time.sleep(0.05)

        # 应调用两次：一次 enable（start_session），一次 disable（后台回滚）
        assert node._call_driver_service.call_count == 2

    def test_connect_failure_recording_rollback(self):
        """连接失败时，录制标记在 start_session 中已设置（后台回滚不影响）"""
        node = _make_mock_node()
        node._inference_bridge.connect.side_effect = RuntimeError("gRPC 连接超时")
        req, resp = _make_request_response()

        node._handle_start_session(req, resp)

        # 录制标记在 start_session 返回时已设置（enable_recording=True）
        # 后台回滚只改 mode，不清理录制状态（由 stop_session 处理）
        assert node._episode_active is True


# ============================================================================
# 7.3: start_session 重复调用
# ============================================================================

class TestStartSessionDuplicate:
    """start_session: 已在运行中 → 拒绝"""

    def test_reject_when_policy(self):
        """POLICY 模式下重复调用应拒绝"""
        node = _make_mock_node(mode=ControlMode.POLICY)
        req, resp = _make_request_response()

        result = node._handle_start_session(req, resp)

        assert result.success is False
        assert "已在运行中" in result.message

    def test_reject_when_human(self):
        """HUMAN 模式下调用应拒绝"""
        node = _make_mock_node(mode=ControlMode.HUMAN)
        req, resp = _make_request_response()

        result = node._handle_start_session(req, resp)

        assert result.success is False
        assert "已在运行中" in result.message


# ============================================================================
# 7.4: stop_session 正常流程
# ============================================================================

class TestStopSessionNormal:
    """stop_session: POLICY → IDLE + 暂停录制"""

    def test_mode_changes_to_idle(self):
        """stop_session 应将模式切换到 IDLE"""
        node = _make_mock_node(mode=ControlMode.POLICY, episode_active=True)
        req, resp = _make_request_response()

        result = node._handle_stop_session(req, resp)

        assert result.success is True
        assert node._mode == ControlMode.IDLE

    def test_recording_paused(self):
        """stop_session 应暂停录制（不保存也不丢弃）"""
        node = _make_mock_node(mode=ControlMode.POLICY, episode_active=True)
        req, resp = _make_request_response()

        node._handle_stop_session(req, resp)

        assert node._episode_active is True  # 仍然 active，等待确认
        assert node._episode_paused is True

    def test_inference_stopped(self):
        """stop_session 应停止推理循环"""
        node = _make_mock_node(mode=ControlMode.POLICY)
        req, resp = _make_request_response()

        node._handle_stop_session(req, resp)

        node._inference_bridge.stop_inference_loop.assert_called_once()

    def test_vr_state_reset(self):
        """stop_session 应重置 VR 状态"""
        node = _make_mock_node(mode=ControlMode.POLICY)
        node.vr_state.is_active = True
        node.vr_state.is_initialized = True
        req, resp = _make_request_response()

        node._handle_stop_session(req, resp)

        assert node.vr_state.is_active is False
        assert node.vr_state.is_initialized is False

    def test_last_executed_joints_cleared(self):
        """stop_session 应清除 _last_executed_joints"""
        node = _make_mock_node(mode=ControlMode.POLICY)
        node._last_executed_joints = np.zeros(7)
        req, resp = _make_request_response()

        node._handle_stop_session(req, resp)

        assert node._last_executed_joints is None

    def test_response_message_contains_recording_info(self):
        """响应消息应包含录制暂停信息"""
        node = _make_mock_node(mode=ControlMode.POLICY, episode_active=True)
        node._recording_frame_count = 150
        req, resp = _make_request_response()

        result = node._handle_stop_session(req, resp)

        assert "录制已暂停" in result.message
        assert "150" in result.message

    def test_driver_follow_disabled_when_not_dry_run(self):
        """非 dry_run 模式下应禁用 driver follow"""
        node = _make_mock_node(mode=ControlMode.POLICY, dry_run=False)
        req, resp = _make_request_response()

        node._handle_stop_session(req, resp)

        node._call_driver_service.assert_called_once()

    def test_status_published(self):
        """stop_session 应发布状态更新"""
        node = _make_mock_node(mode=ControlMode.POLICY)
        node._publish_status = MagicMock()
        req, resp = _make_request_response()

        node._handle_stop_session(req, resp)

        node._publish_status.assert_called_once()


# ============================================================================
# 7.5: stop_session 在 IDLE 时调用
# ============================================================================

class TestStopSessionIdle:
    """stop_session: IDLE 模式下调用 → 拒绝"""

    def test_reject_when_idle(self):
        """IDLE 模式下调用 stop_session 应拒绝"""
        node = _make_mock_node(mode=ControlMode.IDLE)
        req, resp = _make_request_response()

        result = node._handle_stop_session(req, resp)

        assert result.success is False
        assert "IDLE" in result.message


# ============================================================================
# 7.6: stop_session 在 HUMAN 模式下调用
# ============================================================================

class TestStopSessionHuman:
    """stop_session: HUMAN 模式下停止"""

    def test_human_to_idle(self):
        """HUMAN 模式下 stop_session 应切换到 IDLE"""
        node = _make_mock_node(mode=ControlMode.HUMAN, episode_active=True)
        req, resp = _make_request_response()

        result = node._handle_stop_session(req, resp)

        assert result.success is True
        assert node._mode == ControlMode.IDLE
        assert "HUMAN → IDLE" in result.message


# ============================================================================
# 7.7: start_session 策略未启用 → 纯 VR 模式
# ============================================================================

class TestStartSessionNoPolicy:
    """start_session: 策略未启用时进入纯 VR (HUMAN) 模式"""

    def test_human_mode_when_policy_disabled(self):
        """enable_policy=False 时应进入 HUMAN 模式"""
        node = _make_mock_node(enable_policy=False)
        req, resp = _make_request_response()

        result = node._handle_start_session(req, resp)

        assert result.success is True
        assert node._mode == ControlMode.HUMAN

    def test_human_mode_when_no_bridge(self):
        """inference_bridge=None 时应进入 HUMAN 模式"""
        node = _make_mock_node(has_bridge=False)
        req, resp = _make_request_response()

        result = node._handle_start_session(req, resp)

        assert result.success is True
        assert node._mode == ControlMode.HUMAN


# ============================================================================
# 7.8: start_session 录制未启用
# ============================================================================

class TestStartSessionNoRecording:
    """start_session: 录制未启用时仅启动推理"""

    def test_no_recording_when_disabled(self):
        """enable_recording=False 时不应启动录制"""
        node = _make_mock_node(enable_recording=False)
        req, resp = _make_request_response()

        result = node._handle_start_session(req, resp)

        assert result.success is True
        assert node._mode == ControlMode.POLICY
        assert node._episode_active is False
        node._recorder.start_episode.assert_not_called()

    def test_message_no_recording_info(self):
        """录制未启用时响应消息不包含录制信息"""
        node = _make_mock_node(enable_recording=False)
        req, resp = _make_request_response()

        result = node._handle_start_session(req, resp)

        assert "录制" not in result.message


# ============================================================================
# 7.9: start_session recorder 尚未 lazy-init
# ============================================================================

class TestStartSessionLazyRecorder:
    """start_session: recorder 尚未初始化时标记 episode"""

    def test_episode_active_without_recorder(self):
        """recorder=None 时应标记 episode_active=True"""
        node = _make_mock_node(has_recorder=False)
        req, resp = _make_request_response()

        result = node._handle_start_session(req, resp)

        assert result.success is True
        assert node._episode_active is True
        assert "warmup" in result.message or "等待" in result.message or "初始化" in result.message

    def test_frame_count_reset_without_recorder(self):
        """recorder=None 时也应重置帧计数"""
        node = _make_mock_node(has_recorder=False)
        node._recording_frame_count = 10
        req, resp = _make_request_response()

        node._handle_start_session(req, resp)

        assert node._recording_frame_count == 0


# ============================================================================
# 7.10: stop_session 录制已暂停
# ============================================================================

class TestStopSessionAlreadyPaused:
    """stop_session: 录制已暂停时不重复暂停"""

    def test_already_paused_stays_paused(self):
        """录制已暂停时 stop_session 不应改变暂停状态"""
        node = _make_mock_node(
            mode=ControlMode.POLICY,
            episode_active=True,
            episode_paused=True,
        )
        node._recording_frame_count = 50
        req, resp = _make_request_response()

        result = node._handle_stop_session(req, resp)

        assert result.success is True
        assert node._episode_paused is True
        assert "50" in result.message


# ============================================================================
# 7.11: _publish_status 包含 server 字段
# ============================================================================

class TestPublishStatusServer:
    """_publish_status 应包含 server 字段"""

    def test_status_has_server_field(self):
        """status JSON 应包含 server.connected/ready/policy_type/pretrained_path"""
        import json
        node = _make_mock_node(mode=ControlMode.POLICY)

        node._publish_status()

        # 获取 publish 调用的参数
        call_args = node.status_pub.publish.call_args
        msg = call_args[0][0]
        data = json.loads(msg.data)

        assert "server" in data
        assert data["server"]["connected"] is True
        assert data["server"]["ready"] is True
        assert data["server"]["policy_type"] == "act"
        assert "test_model" in data["server"]["pretrained_path"]

    def test_status_has_inference_field(self):
        """status JSON 应包含 inference 字段"""
        import json
        node = _make_mock_node(mode=ControlMode.POLICY)

        node._publish_status()

        call_args = node.status_pub.publish.call_args
        msg = call_args[0][0]
        data = json.loads(msg.data)

        assert "inference" in data
        assert data["inference"]["warmup_done"] is True
        assert data["inference"]["buffer_size"] == 5


# ============================================================================
# 7.12: _publish_status 包含 recording 字段
# ============================================================================

class TestPublishStatusRecording:
    """_publish_status 应包含 recording 字段"""

    def test_status_has_recording_field(self):
        """status JSON 应包含 recording.episode_active/paused/num_episodes/frame_count"""
        import json
        node = _make_mock_node(
            mode=ControlMode.POLICY,
            episode_active=True,
            episode_paused=False,
        )
        node._recording_frame_count = 42

        node._publish_status()

        call_args = node.status_pub.publish.call_args
        msg = call_args[0][0]
        data = json.loads(msg.data)

        assert "recording" in data
        assert data["recording"]["episode_active"] is True
        assert data["recording"]["episode_paused"] is False

    def test_status_no_recording_without_recorder(self):
        """recorder=None 且 enable_recording=False 时 status 不包含 recording 字段"""
        import json
        node = _make_mock_node(has_recorder=False, enable_recording=False)

        node._publish_status()

        call_args = node.status_pub.publish.call_args
        msg = call_args[0][0]
        data = json.loads(msg.data)

        assert "recording" not in data


# ============================================================================
# 7.13: max_episodes 限制
# ============================================================================

class TestStartSessionMaxEpisodes:
    """start_session: 达到 max_episodes 时跳过录制"""

    def test_skip_recording_at_max(self):
        """达到 max_episodes 时应跳过录制"""
        node = _make_mock_node(max_episodes=5)
        node._recorder.num_episodes = 5
        req, resp = _make_request_response()

        result = node._handle_start_session(req, resp)

        assert result.success is True
        assert node._mode == ControlMode.POLICY
        assert node._episode_active is False
        node._recorder.start_episode.assert_not_called()
        assert "最大" in result.message or "跳过" in result.message

    def test_allow_recording_below_max(self):
        """未达 max_episodes 时应正常录制"""
        node = _make_mock_node(max_episodes=5)
        node._recorder.num_episodes = 3
        req, resp = _make_request_response()

        result = node._handle_start_session(req, resp)

        assert result.success is True
        assert node._episode_active is True
        node._recorder.start_episode.assert_called_once()


# ============================================================================
# 8.1: _tick_shadow 单元测试
# ============================================================================

class TestTickShadow:
    """_tick_shadow: HUMAN 模式下消费 buffer 更新 _last_policy_action"""

    def test_warmup_not_done_noop(self):
        """warmup 未完成时 _tick_shadow 不操作"""
        node = _make_mock_node(mode=ControlMode.HUMAN)
        node.hold_on_empty = True
        node._inference_bridge.warmup_done = False

        node._tick_shadow()

        assert node._last_policy_action is None
        node._inference_bridge.get_next_action.assert_not_called()

    def test_buffer_has_action(self):
        """buffer 有 action 时应更新 _last_policy_action"""
        node = _make_mock_node(mode=ControlMode.HUMAN)
        node.hold_on_empty = True
        mock_action = MagicMock()
        mock_action.data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        mock_action.gripper = 0.8
        node._inference_bridge.get_next_action.return_value = mock_action

        node._tick_shadow()

        assert node._last_policy_action is not None
        assert len(node._last_policy_action) == 8
        np.testing.assert_allclose(node._last_policy_action[:7], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        np.testing.assert_allclose(node._last_policy_action[7], 0.8)

    def test_buffer_empty_hold_last(self):
        """buffer 为空且 hold_on_empty=True 时应 hold last action"""
        node = _make_mock_node(mode=ControlMode.HUMAN)
        node.hold_on_empty = True
        node._inference_bridge.get_next_action.return_value = None
        mock_last = MagicMock()
        mock_last.data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        mock_last.gripper = 0.5
        node._inference_bridge.last_action = mock_last

        node._tick_shadow()

        assert node._last_policy_action is not None
        assert len(node._last_policy_action) == 8

    def test_buffer_empty_no_last_action(self):
        """buffer 为空且无 last_action 时不更新"""
        node = _make_mock_node(mode=ControlMode.HUMAN)
        node.hold_on_empty = True
        node._inference_bridge.get_next_action.return_value = None
        node._inference_bridge.last_action = None

        node._tick_shadow()

        assert node._last_policy_action is None

    def test_no_bridge_noop(self):
        """inference_bridge=None 时 _tick_shadow 不操作"""
        node = _make_mock_node(mode=ControlMode.HUMAN, has_bridge=False)
        node.hold_on_empty = True

        node._tick_shadow()

        assert node._last_policy_action is None

    def test_hold_on_empty_false_no_fallback(self):
        """hold_on_empty=False 且 buffer 为空时不 fallback"""
        node = _make_mock_node(mode=ControlMode.HUMAN)
        node.hold_on_empty = False
        node._inference_bridge.get_next_action.return_value = None

        node._tick_shadow()

        assert node._last_policy_action is None


# ============================================================================
# 8.2: 模式转换测试（shadow mode 改造后）
# ============================================================================

class TestModeTransitionShadow:
    """验证 POLICY→HUMAN 不 pause、HUMAN→POLICY 不 resume 但 clear_buffer"""

    def test_to_human_no_pause(self):
        """POLICY→HUMAN 不应调用 pause()"""
        node = _make_mock_node(mode=ControlMode.POLICY)
        node._publish_status = MagicMock()
        # _activate_control 需要 mock
        node._activate_control = MagicMock()

        node._transition_to_human()

        assert node._mode == ControlMode.HUMAN
        node._inference_bridge.pause.assert_not_called()

    def test_to_human_no_clear_buffer(self):
        """POLICY→HUMAN 不应调用 clear_buffer()"""
        node = _make_mock_node(mode=ControlMode.POLICY)
        node._publish_status = MagicMock()
        node._activate_control = MagicMock()

        node._transition_to_human()

        node._inference_bridge.clear_buffer.assert_not_called()

    def test_to_human_no_reset_last_policy_action(self):
        """POLICY→HUMAN 不应重置 _last_policy_action"""
        node = _make_mock_node(mode=ControlMode.POLICY)
        node._publish_status = MagicMock()
        node._activate_control = MagicMock()
        node._last_policy_action = np.array([1.0, 2.0, 3.0])

        node._transition_to_human()

        assert node._last_policy_action is not None
        np.testing.assert_allclose(node._last_policy_action, [1.0, 2.0, 3.0])

    def test_to_policy_no_resume(self):
        """HUMAN→POLICY 不应调用 resume()"""
        node = _make_mock_node(mode=ControlMode.HUMAN)
        node._publish_status = MagicMock()

        node._transition_to_policy()

        assert node._mode == ControlMode.POLICY
        node._inference_bridge.resume.assert_not_called()

    def test_to_policy_clears_buffer(self):
        """HUMAN→POLICY 应调用 clear_buffer()"""
        node = _make_mock_node(mode=ControlMode.HUMAN)
        node._publish_status = MagicMock()

        node._transition_to_policy()

        node._inference_bridge.clear_buffer.assert_called_once()

    def test_to_policy_resets_last_executed_joints(self):
        """HUMAN→POLICY 应重置 _last_executed_joints"""
        node = _make_mock_node(mode=ControlMode.HUMAN)
        node._publish_status = MagicMock()
        node._last_executed_joints = np.zeros(7)

        node._transition_to_policy()

        assert node._last_executed_joints is None

    def test_to_policy_deactivates_vr(self):
        """HUMAN→POLICY 应停用 VR"""
        node = _make_mock_node(mode=ControlMode.HUMAN)
        node._publish_status = MagicMock()
        node.vr_state.is_active = True
        node.vr_state.is_initialized = True

        node._transition_to_policy()

        assert node.vr_state.is_active is False
        assert node.vr_state.is_initialized is False


# ============================================================================
# 8.3: _handle_new_episode 测试
# ============================================================================

class TestHandleNewEpisode:
    """_handle_new_episode: 多 episode 工作流"""

    def test_idle_rejected(self):
        """IDLE 模式下应拒绝"""
        node = _make_mock_node(mode=ControlMode.IDLE)
        node._recording_sync = MagicMock()
        req, resp = _make_request_response()

        result = node._handle_new_episode(req, resp)

        assert result.success is False
        assert "未运行" in result.message

    def test_episode_active_rejected(self):
        """当前 episode 仍在录制时应拒绝"""
        node = _make_mock_node(mode=ControlMode.POLICY, episode_active=True)
        node._recording_sync = MagicMock()
        req, resp = _make_request_response()

        result = node._handle_new_episode(req, resp)

        assert result.success is False
        assert "仍在录制" in result.message

    def test_max_episodes_rejected(self):
        """达到 max_episodes 时应拒绝"""
        node = _make_mock_node(mode=ControlMode.POLICY, max_episodes=3)
        node._recording_sync = MagicMock()
        node._recorder.num_episodes = 3
        req, resp = _make_request_response()

        result = node._handle_new_episode(req, resp)

        assert result.success is False
        assert "最大" in result.message

    def test_normal_success(self):
        """正常情况下应成功开始新 episode"""
        node = _make_mock_node(mode=ControlMode.POLICY, episode_active=False)
        node._recording_sync = MagicMock()
        node._recorder.num_episodes = 2
        req, resp = _make_request_response()

        result = node._handle_new_episode(req, resp)

        assert result.success is True
        assert node._episode_active is True
        assert node._episode_paused is False
        assert node._recording_frame_count == 0
        assert node._last_policy_action is None
        node._recorder.start_episode.assert_called_once_with(task="pick and place")

    def test_recording_disabled_rejected(self):
        """enable_recording=False 时应拒绝"""
        node = _make_mock_node(mode=ControlMode.POLICY, enable_recording=False)
        node._recording_sync = MagicMock()
        req, resp = _make_request_response()

        result = node._handle_new_episode(req, resp)

        assert result.success is False

    def test_no_recording_sync_rejected(self):
        """_recording_sync=None 时应拒绝"""
        node = _make_mock_node(mode=ControlMode.POLICY)
        node._recording_sync = None
        req, resp = _make_request_response()

        result = node._handle_new_episode(req, resp)

        assert result.success is False

    def test_human_mode_allowed(self):
        """HUMAN 模式下也应允许新 episode"""
        node = _make_mock_node(mode=ControlMode.HUMAN, episode_active=False)
        node._recording_sync = MagicMock()
        node._recorder.num_episodes = 0
        req, resp = _make_request_response()

        result = node._handle_new_episode(req, resp)

        assert result.success is True
        assert node._episode_active is True

    def test_response_contains_episode_number(self):
        """响应消息应包含 episode 编号"""
        node = _make_mock_node(mode=ControlMode.POLICY, episode_active=False)
        node._recording_sync = MagicMock()
        node._recorder.num_episodes = 4
        req, resp = _make_request_response()

        result = node._handle_new_episode(req, resp)

        assert "5" in result.message

    def test_no_recorder_lazy_init(self):
        """recorder=None 时应标记 episode_active 等待 lazy init"""
        node = _make_mock_node(mode=ControlMode.POLICY, has_recorder=False, episode_active=False)
        node._recording_sync = MagicMock()
        req, resp = _make_request_response()

        result = node._handle_new_episode(req, resp)

        assert result.success is True
        assert node._episode_active is True
        assert node._recording_frame_count == 0
