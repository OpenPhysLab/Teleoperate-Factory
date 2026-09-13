"""
DAggerRecorder 单元测试

测试覆盖:
- build_dagger_features() 生成正确的 features dict
- _postprocess_episode_actions() 正确性
- 占位策略: add_frame 时 action = state
- finish_episode 后 action[t] = state[t+1]
- 空 episode 保护
- DAgger 附加字段 (policy_action, control_source)
- ROBOT_STATE_NAMES 长度和内容
"""
import os
import shutil
import sys
import tempfile
from typing import Dict, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dagger.core.data_recorder import (
    ROBOT_STATE_NAMES,
    DAggerRecorder,
    build_dagger_features,
)


class TestRobotStateNames:
    """测试 ROBOT_STATE_NAMES 常量"""

    def test_length(self):
        assert len(ROBOT_STATE_NAMES) == 14

    def test_joint_names(self):
        for i in range(7):
            assert ROBOT_STATE_NAMES[i] == f"joint_{i+1}_rad"

    def test_gripper_name(self):
        assert ROBOT_STATE_NAMES[7] == "gripper_open"

    def test_eef_names(self):
        expected = [
            "eef_pos_x_m", "eef_pos_y_m", "eef_pos_z_m",
            "eef_rot_euler_x_rad", "eef_rot_euler_y_rad", "eef_rot_euler_z_rad",
        ]
        assert ROBOT_STATE_NAMES[8:14] == expected


class TestBuildDaggerFeatures:
    """测试 build_dagger_features()"""

    def test_basic_features(self):
        features = build_dagger_features(
            camera_shapes={},
            policy_action_dim=7,
        )
        # observation.state
        assert "observation.state" in features
        assert features["observation.state"]["shape"] == (14,)
        assert features["observation.state"]["dtype"] == "float32"
        assert features["observation.state"]["names"] == ROBOT_STATE_NAMES

        # action
        assert "action" in features
        assert features["action"]["shape"] == (14,)
        assert features["action"]["dtype"] == "float32"
        assert features["action"]["names"] == ROBOT_STATE_NAMES

        # policy_action
        assert "policy_action" in features
        assert features["policy_action"]["shape"] == (7,)
        assert features["policy_action"]["dtype"] == "float32"

        # control_source
        assert "control_source" in features
        assert features["control_source"]["shape"] == (1,)
        assert features["control_source"]["dtype"] == "int32"

    def test_with_cameras(self):
        features = build_dagger_features(
            camera_shapes={
                "cam0_rgb": (480, 640, 3),
                "cam1_rgb": (480, 640, 3),
            },
            policy_action_dim=7,
        )
        assert "observation.images.cam0_rgb" in features
        assert "observation.images.cam1_rgb" in features

        cam_feat = features["observation.images.cam0_rgb"]
        assert cam_feat["dtype"] == "video"
        # shape is (C, H, W) in features
        assert cam_feat["shape"] == (3, 480, 640)
        assert cam_feat["names"] == ["channels", "height", "width"]

    def test_different_policy_action_dim(self):
        features = build_dagger_features(
            camera_shapes={},
            policy_action_dim=14,
        )
        assert features["policy_action"]["shape"] == (14,)


class TestPostprocessEpisodeActions:
    """测试 _postprocess_episode_actions() 逻辑（使用 mock dataset）"""

    def _make_recorder_with_mock_dataset(self):
        """创建一个带 mock dataset 的 recorder（绕过 LeRobotDataset.create）"""
        recorder = object.__new__(DAggerRecorder)
        recorder._policy_action_dim = 7
        recorder._episode_started = False
        recorder._current_task = ""
        recorder._frame_count = 0
        recorder.dataset = MagicMock()
        return recorder

    def test_3_frames(self):
        """3 帧 episode: action[0]=state[1], action[1]=state[2], action[2]=state[2]"""
        recorder = self._make_recorder_with_mock_dataset()

        states = [
            np.array([1.0] * 14, dtype=np.float32),
            np.array([2.0] * 14, dtype=np.float32),
            np.array([3.0] * 14, dtype=np.float32),
        ]
        actions = [
            np.array([1.0] * 14, dtype=np.float32),  # placeholder = state[0]
            np.array([2.0] * 14, dtype=np.float32),  # placeholder = state[1]
            np.array([3.0] * 14, dtype=np.float32),  # placeholder = state[2]
        ]

        recorder.dataset.episode_buffer = {
            "size": 3,
            "observation.state": states,
            "action": actions,
        }

        recorder._postprocess_episode_actions()

        # action[0] = state[1]
        np.testing.assert_array_equal(actions[0], states[1])
        # action[1] = state[2]
        np.testing.assert_array_equal(actions[1], states[2])
        # action[2] = state[2] (last frame holds)
        np.testing.assert_array_equal(actions[2], np.array([3.0] * 14, dtype=np.float32))

    def test_1_frame_skips(self):
        """1 帧 episode: 跳过后处理"""
        recorder = self._make_recorder_with_mock_dataset()

        original_action = np.array([1.0] * 14, dtype=np.float32)
        actions = [original_action.copy()]

        recorder.dataset.episode_buffer = {
            "size": 1,
            "observation.state": [np.array([1.0] * 14, dtype=np.float32)],
            "action": actions,
        }

        recorder._postprocess_episode_actions()

        # Action should be unchanged
        np.testing.assert_array_equal(actions[0], original_action)

    def test_0_frames_skips(self):
        """0 帧 episode: 跳过后处理"""
        recorder = self._make_recorder_with_mock_dataset()
        recorder.dataset.episode_buffer = {"size": 0}
        # Should not raise
        recorder._postprocess_episode_actions()

    def test_none_buffer(self):
        """buffer 为 None: 跳过后处理"""
        recorder = self._make_recorder_with_mock_dataset()
        recorder.dataset.episode_buffer = None
        # Should not raise
        recorder._postprocess_episode_actions()

    def test_actions_are_copies(self):
        """后处理后 action 是 state 的副本，不是引用"""
        recorder = self._make_recorder_with_mock_dataset()

        states = [
            np.array([1.0] * 14, dtype=np.float32),
            np.array([2.0] * 14, dtype=np.float32),
        ]
        actions = [
            np.array([1.0] * 14, dtype=np.float32),
            np.array([2.0] * 14, dtype=np.float32),
        ]

        recorder.dataset.episode_buffer = {
            "size": 2,
            "observation.state": states,
            "action": actions,
        }

        recorder._postprocess_episode_actions()

        # Modify state[1] after postprocess
        states[1][0] = 999.0
        # action[0] should NOT be affected (it's a copy)
        assert actions[0][0] != 999.0


class TestDAggerRecorderWorkflow:
    """测试 DAggerRecorder 完整工作流（使用 mock dataset）"""

    def _make_recorder_with_mock_dataset(self):
        """创建一个带 mock dataset 的 recorder"""
        recorder = object.__new__(DAggerRecorder)
        recorder._policy_action_dim = 7
        recorder._episode_started = False
        recorder._current_task = ""
        recorder._frame_count = 0

        mock_dataset = MagicMock()
        mock_dataset.num_episodes = 0

        # Track add_frame calls
        frames_added = []
        def mock_add_frame(frame, task=""):
            frames_added.append((frame.copy(), task))
        mock_dataset.add_frame = mock_add_frame
        mock_dataset._frames_added = frames_added

        recorder.dataset = mock_dataset
        return recorder

    def test_add_frame_before_start_is_noop(self):
        """未 start_episode 时 add_frame 无效"""
        recorder = self._make_recorder_with_mock_dataset()
        state = np.zeros(14, dtype=np.float32)
        policy_action = np.zeros(7, dtype=np.float32)
        recorder.add_frame(state, {}, policy_action, control_source=0)
        assert recorder._frame_count == 0
        assert len(recorder.dataset._frames_added) == 0

    def test_start_add_finish_workflow(self):
        """正常工作流: start -> add_frame x N -> finish"""
        recorder = self._make_recorder_with_mock_dataset()

        # Setup mock episode_buffer for postprocess
        states = []
        actions = []

        original_add_frame = recorder.dataset.add_frame
        def tracking_add_frame(frame, task=""):
            states.append(frame["observation.state"].copy())
            actions.append(frame["action"].copy())
            original_add_frame(frame, task=task)

        recorder.dataset.add_frame = tracking_add_frame
        recorder.dataset.episode_buffer = {
            "size": 0,
            "observation.state": states,
            "action": actions,
        }

        recorder.start_episode(task="test task")
        assert recorder._episode_started is True

        # Add 3 frames with different states
        for i in range(3):
            state = np.full(14, float(i + 1), dtype=np.float32)
            policy_action = np.full(7, float(i) * 0.1, dtype=np.float32)
            recorder.add_frame(state, {}, policy_action, control_source=i % 2)

        # Update size in buffer
        recorder.dataset.episode_buffer["size"] = 3

        assert recorder._frame_count == 3

        # Verify placeholder: action = state at add_frame time
        for i in range(3):
            expected_state = np.full(14, float(i + 1), dtype=np.float32)
            np.testing.assert_array_equal(actions[i], expected_state)

        recorder.finish_episode()

        # After postprocess: action[0]=state[1], action[1]=state[2], action[2]=state[2]
        np.testing.assert_array_equal(actions[0], states[1])
        np.testing.assert_array_equal(actions[1], states[2])
        np.testing.assert_array_equal(actions[2], states[2])

        # save_episode should have been called
        recorder.dataset.save_episode.assert_called_once()
        assert recorder._episode_started is False
        assert recorder._frame_count == 0

    def test_empty_episode_skips_save(self):
        """空 episode 不调用 save_episode"""
        recorder = self._make_recorder_with_mock_dataset()
        recorder.start_episode(task="empty")
        recorder.finish_episode()
        recorder.dataset.save_episode.assert_not_called()
        assert recorder._episode_started is False

    def test_finish_without_start_is_noop(self):
        """未 start 时 finish 无效"""
        recorder = self._make_recorder_with_mock_dataset()
        recorder.finish_episode()
        recorder.dataset.save_episode.assert_not_called()

    def test_frame_data_types(self):
        """验证 add_frame 写入的数据类型"""
        recorder = self._make_recorder_with_mock_dataset()
        recorder.start_episode()

        state = np.ones(14, dtype=np.float64)  # intentionally float64
        policy_action = np.ones(7, dtype=np.float64)
        images = {"cam0_rgb": np.zeros((480, 640, 3), dtype=np.uint8)}

        recorder.add_frame(state, images, policy_action, control_source=1)

        frame, task = recorder.dataset._frames_added[0]
        assert frame["observation.state"].dtype == np.float32
        assert frame["action"].dtype == np.float32
        assert frame["policy_action"].dtype == np.float32
        assert frame["control_source"].dtype == np.int32
        assert frame["control_source"][0] == 1
        assert "observation.images.cam0_rgb" in frame

    def test_task_passed_to_add_frame(self):
        """task 字符串正确传递"""
        recorder = self._make_recorder_with_mock_dataset()
        recorder.start_episode(task="pick up banana")

        state = np.zeros(14, dtype=np.float32)
        policy_action = np.zeros(7, dtype=np.float32)
        recorder.add_frame(state, {}, policy_action)

        _, task = recorder.dataset._frames_added[0]
        assert task == "pick up banana"

    def test_close_finishes_active_episode(self):
        """close() 自动结束活跃的 episode"""
        recorder = self._make_recorder_with_mock_dataset()
        recorder.start_episode()

        state = np.zeros(14, dtype=np.float32)
        policy_action = np.zeros(7, dtype=np.float32)
        recorder.add_frame(state, {}, policy_action)
        recorder._frame_count = 1

        # Setup buffer for postprocess
        recorder.dataset.episode_buffer = {
            "size": 1,
            "observation.state": [state],
            "action": [state.copy()],
        }

        recorder.close()
        recorder.dataset.save_episode.assert_called_once()
        recorder.dataset.stop_image_writer.assert_called_once()

    def test_close_without_active_episode(self):
        """close() 无活跃 episode 时只停止 image writer"""
        recorder = self._make_recorder_with_mock_dataset()
        recorder.close()
        recorder.dataset.save_episode.assert_not_called()
        recorder.dataset.stop_image_writer.assert_called_once()

    def test_num_episodes_property(self):
        """num_episodes 属性代理到 dataset"""
        recorder = self._make_recorder_with_mock_dataset()
        recorder.dataset.num_episodes = 5
        assert recorder.num_episodes == 5
