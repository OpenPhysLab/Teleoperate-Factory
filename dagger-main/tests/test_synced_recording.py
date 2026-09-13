"""
消息同步录制单元测试

测试覆盖:
- 5.1: _on_recording_sync 调用 add_frame 且 state/image 来自 sync 消息
- 5.2: sync 回调频率 > recording_fps 时频率限制生效
- 5.3: _execution_tick() 不再调用任何录制逻辑

注意: dagger_node.py 依赖 ROS2 (rclpy, sensor_msgs 等)，
需要在模块加载前通过 sys.modules 注入 mock 来绕过。
"""
import os
import sys
import time
import threading
from unittest.mock import MagicMock, patch

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

# Mock Node base class — must be a real class (not MagicMock) so that
# DAggerNode instances created via object.__new__ behave normally.
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

# Mock message types used in type hints
sys.modules["sensor_msgs.msg"].JointState = MagicMock
sys.modules["sensor_msgs.msg"].Image = MagicMock
sys.modules["geometry_msgs.msg"].PoseStamped = MagicMock
sys.modules["std_msgs.msg"].Float32 = MagicMock
sys.modules["std_msgs.msg"].String = MagicMock
sys.modules["std_srvs.srv"].Trigger = MagicMock

# Mock CvBridge
sys.modules["cv_bridge"].CvBridge = MagicMock

# Mock message_filters.ApproximateTimeSynchronizer
sys.modules["message_filters"].ApproximateTimeSynchronizer = MagicMock
sys.modules["message_filters"].Subscriber = MagicMock

# Mock vr_utils (loaded via sys.path manipulation in dagger_node.py)
_vr_utils_mock = MagicMock()
sys.modules["vr_utils"] = _vr_utils_mock

# Now import the module under test
from dagger.dagger_node import DAggerNode, ControlMode  # noqa: E402


# ============================================================================
# Helpers
# ============================================================================

def _make_mock_node():
    """
    创建一个 mock DAggerNode 实例，绕过 __init__。
    直接设置测试所需的属性。
    """
    node = object.__new__(DAggerNode)

    node._mode = ControlMode.POLICY
    node._mode_lock = threading.Lock()
    node._recorder = MagicMock()
    node._recorder_initialized = True
    node._recorder_lock = threading.Lock()
    node._recorder_init_in_progress = False
    node._recorder_init_failed = False
    node._episode_active = True
    node._episode_paused = False
    node._last_policy_action = None
    node._last_recording_ts = None
    node._min_frame_interval = 1.0 / 30.0  # 30 FPS
    node._recording_frame_count = 0
    node._cv_bridge = MagicMock()
    node._inference_bridge = None
    node.camera_names = ["cam0", "cam1"]
    node.recording_task = "pick and place"
    node.enable_recording = True
    node.enable_policy = False
    node.get_logger = MagicMock(return_value=MagicMock())

    return node


def _make_state_msg(position_14d):
    """创建模拟的 JointState 消息"""
    msg = MagicMock()
    msg.position = list(position_14d)
    return msg


def _make_image_msg():
    """创建模拟的 Image 消息"""
    return MagicMock()


# ============================================================================
# 5.1: _on_recording_sync 调用 add_frame 且数据来自 sync 消息
# ============================================================================

class TestOnRecordingSync:
    """5.1: 验证 _on_recording_sync 调用 add_frame 且数据来自 sync 消息"""

    def test_sync_callback_calls_add_frame(self):
        """sync 回调应调用 recorder.add_frame"""
        node = _make_mock_node()

        state_14d = np.random.rand(14).tolist()
        state_msg = _make_state_msg(state_14d)
        img0 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img_msg0 = _make_image_msg()
        img_msg1 = _make_image_msg()

        node._cv_bridge.imgmsg_to_cv2.side_effect = [img0, img1]

        node._on_recording_sync(state_msg, img_msg0, img_msg1)

        node._recorder.add_frame.assert_called_once()
        call_args = node._recorder.add_frame.call_args
        recorded_state = call_args[0][0]
        recorded_images = call_args[0][1]

        np.testing.assert_array_almost_equal(recorded_state, state_14d)
        assert "cam0" in recorded_images
        assert "cam1" in recorded_images
        np.testing.assert_array_equal(recorded_images["cam0"], img0)
        np.testing.assert_array_equal(recorded_images["cam1"], img1)

    def test_sync_callback_extracts_state_from_message(self):
        """state 应从 sync 消息的 position[:14] 提取"""
        node = _make_mock_node()

        expected_state = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.5, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3]
        state_msg = _make_state_msg(expected_state)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        node._cv_bridge.imgmsg_to_cv2.return_value = img

        node._on_recording_sync(state_msg, _make_image_msg(), _make_image_msg())

        recorded_state = node._recorder.add_frame.call_args[0][0]
        np.testing.assert_array_almost_equal(recorded_state, expected_state)

    def test_sync_callback_converts_images_from_messages(self):
        """images 应通过 cv_bridge 从 sync 消息转换"""
        node = _make_mock_node()

        state_msg = _make_state_msg(np.zeros(14))
        img_msg0 = _make_image_msg()
        img_msg1 = _make_image_msg()

        img0 = np.ones((480, 640, 3), dtype=np.uint8) * 100
        img1 = np.ones((480, 640, 3), dtype=np.uint8) * 200
        node._cv_bridge.imgmsg_to_cv2.side_effect = [img0, img1]

        node._on_recording_sync(state_msg, img_msg0, img_msg1)

        calls = node._cv_bridge.imgmsg_to_cv2.call_args_list
        assert calls[0][0][0] is img_msg0
        assert calls[1][0][0] is img_msg1
        assert calls[0][1]["desired_encoding"] == "rgb8"

    def test_sync_callback_policy_action_from_shared_variable(self):
        """policy_action 应从 self._last_policy_action 读取"""
        node = _make_mock_node()
        node._last_policy_action = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.5], dtype=np.float32)

        state_msg = _make_state_msg(np.zeros(14))
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        node._cv_bridge.imgmsg_to_cv2.return_value = img

        node._on_recording_sync(state_msg, _make_image_msg(), _make_image_msg())

        recorded_policy_action = node._recorder.add_frame.call_args[0][2]
        np.testing.assert_array_equal(recorded_policy_action, node._last_policy_action)

    def test_sync_callback_zero_policy_action_when_none(self):
        """_last_policy_action 为 None 时应零填充"""
        node = _make_mock_node()
        node._last_policy_action = None

        state_msg = _make_state_msg(np.zeros(14))
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        node._cv_bridge.imgmsg_to_cv2.return_value = img

        node._on_recording_sync(state_msg, _make_image_msg(), _make_image_msg())

        recorded_policy_action = node._recorder.add_frame.call_args[0][2]
        np.testing.assert_array_equal(recorded_policy_action, np.zeros(8, dtype=np.float32))

    def test_sync_callback_control_source_policy(self):
        """POLICY 模式下 control_source 应为 0"""
        node = _make_mock_node()
        node._mode = ControlMode.POLICY

        state_msg = _make_state_msg(np.zeros(14))
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        node._cv_bridge.imgmsg_to_cv2.return_value = img

        node._on_recording_sync(state_msg, _make_image_msg(), _make_image_msg())

        control_source = node._recorder.add_frame.call_args[0][3]
        assert control_source == 0

    def test_sync_callback_control_source_human(self):
        """HUMAN 模式下 control_source 应为 1"""
        node = _make_mock_node()
        node._mode = ControlMode.HUMAN

        state_msg = _make_state_msg(np.zeros(14))
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        node._cv_bridge.imgmsg_to_cv2.return_value = img

        node._on_recording_sync(state_msg, _make_image_msg(), _make_image_msg())

        control_source = node._recorder.add_frame.call_args[0][3]
        assert control_source == 1

    def test_sync_callback_skips_when_not_recording(self):
        """episode 未 active 时不应调用 add_frame"""
        node = _make_mock_node()
        node._episode_active = False

        state_msg = _make_state_msg(np.zeros(14))
        node._on_recording_sync(state_msg, _make_image_msg(), _make_image_msg())

        node._recorder.add_frame.assert_not_called()

    def test_sync_callback_skips_when_recorder_failed(self):
        """recorder 为 None 且已初始化过（失败）时不应调用 add_frame"""
        node = _make_mock_node()
        node._recorder = None
        node._recorder_initialized = True

        state_msg = _make_state_msg(np.zeros(14))
        node._on_recording_sync(state_msg, _make_image_msg(), _make_image_msg())
        # 不应抛异常

    def test_sync_callback_skips_short_state(self):
        """state 不足 14 维时应跳过"""
        node = _make_mock_node()

        state_msg = _make_state_msg(np.zeros(7))
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        node._cv_bridge.imgmsg_to_cv2.return_value = img

        node._on_recording_sync(state_msg, _make_image_msg(), _make_image_msg())

        node._recorder.add_frame.assert_not_called()


# ============================================================================
# 5.2: 频率限制
# ============================================================================

class TestFrequencyLimiting:
    """5.2: 验证频率限制生效"""

    def test_first_frame_always_recorded(self):
        """第一帧（_last_recording_ts=None）应始终录制"""
        node = _make_mock_node()
        node._last_recording_ts = None

        state_msg = _make_state_msg(np.zeros(14))
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        node._cv_bridge.imgmsg_to_cv2.return_value = img

        node._on_recording_sync(state_msg, _make_image_msg(), _make_image_msg())

        node._recorder.add_frame.assert_called_once()

    def test_frame_skipped_within_interval(self):
        """间隔不足 min_frame_interval 时应跳过"""
        node = _make_mock_node()
        node._min_frame_interval = 1.0 / 30.0  # ~33ms
        node._last_recording_ts = time.time()  # 刚刚录制过

        state_msg = _make_state_msg(np.zeros(14))
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        node._cv_bridge.imgmsg_to_cv2.return_value = img

        node._on_recording_sync(state_msg, _make_image_msg(), _make_image_msg())

        node._recorder.add_frame.assert_not_called()

    def test_frame_recorded_after_interval(self):
        """间隔超过 min_frame_interval 时应录制"""
        node = _make_mock_node()
        node._min_frame_interval = 1.0 / 30.0  # ~33ms
        node._last_recording_ts = time.time() - 0.05  # 50ms 前

        state_msg = _make_state_msg(np.zeros(14))
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        node._cv_bridge.imgmsg_to_cv2.return_value = img

        node._on_recording_sync(state_msg, _make_image_msg(), _make_image_msg())

        node._recorder.add_frame.assert_called_once()

    def test_recording_frame_count_increments(self):
        """每次成功录制后 _recording_frame_count 应递增"""
        node = _make_mock_node()
        assert node._recording_frame_count == 0

        state_msg = _make_state_msg(np.zeros(14))
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        node._cv_bridge.imgmsg_to_cv2.return_value = img

        node._on_recording_sync(state_msg, _make_image_msg(), _make_image_msg())
        assert node._recording_frame_count == 1

        # 等待足够间隔后再录一帧
        node._last_recording_ts = time.time() - 0.05
        node._cv_bridge.imgmsg_to_cv2.return_value = img
        node._on_recording_sync(state_msg, _make_image_msg(), _make_image_msg())
        assert node._recording_frame_count == 2

    def test_last_recording_ts_updated(self):
        """录制后 _last_recording_ts 应更新"""
        node = _make_mock_node()
        assert node._last_recording_ts is None

        state_msg = _make_state_msg(np.zeros(14))
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        node._cv_bridge.imgmsg_to_cv2.return_value = img

        before = time.time()
        node._on_recording_sync(state_msg, _make_image_msg(), _make_image_msg())
        after = time.time()

        assert node._last_recording_ts is not None
        assert before <= node._last_recording_ts <= after


# ============================================================================
# 5.3: _execution_tick() 不再调用任何录制逻辑
# ============================================================================

class TestExecutionTickNoRecording:
    """5.3: 验证 _execution_tick() 不再调用任何录制逻辑"""

    def _make_tick_node(self):
        """创建用于测试 _execution_tick 的 mock node"""
        node = object.__new__(DAggerNode)

        node._mode = ControlMode.POLICY
        node._mode_lock = threading.Lock()
        node._inference_bridge = MagicMock()
        node._inference_bridge.warmup_done = True
        node._inference_bridge.get_next_action.return_value = None
        node._inference_bridge.last_action = None
        node.hold_on_empty = True
        node.enable_recording = True
        node.enable_policy = True
        node._recorder = MagicMock()
        node._recorder_initialized = True
        node._recorder_lock = threading.Lock()
        node._episode_active = True
        node._last_policy_action = None

        # mock 控制循环依赖的方法
        node._update_inference_observation = MagicMock()
        node._tick_policy = MagicMock()
        node._tick_human = MagicMock()
        node._tick_shadow = MagicMock()

        node.get_logger = MagicMock(return_value=MagicMock())

        return node

    def test_execution_tick_does_not_have_record_frame(self):
        """_record_frame 方法应已删除"""
        node = self._make_tick_node()
        # DAggerNode 类上不应有 _record_frame 方法
        assert not hasattr(DAggerNode, '_record_frame')

    def test_execution_tick_does_not_call_add_frame(self):
        """_execution_tick 不应直接或间接调用 recorder.add_frame"""
        node = self._make_tick_node()

        node._execution_tick()

        node._recorder.add_frame.assert_not_called()

    def test_execution_tick_does_not_init_recorder(self):
        """_execution_tick 不应调用 recorder 初始化逻辑"""
        node = self._make_tick_node()
        node._recorder_initialized = False
        node._recorder = None

        node._execution_tick()

        assert node._recorder is None

    def test_execution_tick_still_updates_observation(self):
        """_execution_tick 在 HUMAN 模式下应调用 _update_inference_observation"""
        node = self._make_tick_node()
        node._mode = ControlMode.HUMAN

        node._execution_tick()

        node._update_inference_observation.assert_called_once()

    def test_execution_tick_still_calls_tick_policy(self):
        """_execution_tick 在 POLICY 模式下不再调用 _tick_policy（由独立线程驱动），也不调用 _update_inference_observation"""
        node = self._make_tick_node()
        node._mode = ControlMode.POLICY

        node._execution_tick()

        # Policy execution is now driven by _policy_exec_loop thread, not timer
        node._tick_policy.assert_not_called()
        # POLICY mode observation feeding is also in _policy_exec_loop, not _execution_tick
        node._update_inference_observation.assert_not_called()

    def test_execution_tick_still_calls_tick_human(self):
        """_execution_tick 在 HUMAN 模式下仍应调用 _tick_human"""
        node = self._make_tick_node()
        node._mode = ControlMode.HUMAN

        node._execution_tick()

        node._tick_human.assert_called_once()

    def test_execution_tick_idle_returns_early(self):
        """_execution_tick 在 IDLE 模式下应直接返回"""
        node = self._make_tick_node()
        node._mode = ControlMode.IDLE

        node._execution_tick()

        node._update_inference_observation.assert_not_called()
        node._tick_policy.assert_not_called()
        node._tick_human.assert_not_called()
