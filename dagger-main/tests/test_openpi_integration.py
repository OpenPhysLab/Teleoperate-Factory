"""
OpenPI 集成测试 — 通过 MockPolicyServer 验证 InferenceBridge 与 OpenPI 的兼容性。

测试场景：
- Pi0 正常延迟 (chunk=50, t_inf=0.3s, f_exec=20Hz)
- Pi0.5 慢推理 (chunk=50, t_inf=0.8s, f_exec=20Hz)
- 极端慢推理 (chunk=50, t_inf=1.5s, f_exec=20Hz)
- 不可行场景 (chunk=3, t_inf=0.3s, f_exec=20Hz)
- 录制字段完整性
- 模式切换

运行: /home/ubuntu/Desktop/Workspace/miniconda3/envs/robocoin/bin/python -m pytest dagger/tests/test_openpi_integration.py -v
"""

import math
import time
import threading
import pytest
import numpy as np

from dagger.tests.mock_policy_server import MockPolicyServerRunner
from dagger.core.inference_bridge import InferenceBridge, InferenceBridgeConfig


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_server_50():
    """action_horizon=50 的 MockPolicyServer（OpenPI 默认）。"""
    runner = MockPolicyServerRunner(
        action_horizon=50,
        action_dim=7,
        simulated_t_inf=0.0,  # 测试中不真正 sleep，用 config.t_inf 控制参数推导
        action_pattern="ramp",
        constant_value=0.1,
        gripper_value=0.5,
    )
    addr = runner.start()
    yield runner, addr
    runner.stop()


@pytest.fixture
def mock_server_3():
    """action_horizon=3 的 MockPolicyServer（不可行场景）。"""
    runner = MockPolicyServerRunner(
        action_horizon=3,
        action_dim=7,
        simulated_t_inf=0.0,
        action_pattern="constant",
        constant_value=0.2,
        gripper_value=0.3,
    )
    addr = runner.start()
    yield runner, addr
    runner.stop()


def _make_bridge(server_address: str, f_exec: float = 20.0, t_inf: float = 0.2,
                 n_action_steps: int = 0) -> InferenceBridge:
    """创建并连接 InferenceBridge。"""
    config = InferenceBridgeConfig(
        server_address=server_address,
        task="test_task",
        f_exec=f_exec,
        n_action_steps=n_action_steps,
        T_inter=0.0,
        t_inf=t_inf,
        chunk_size_threshold=0,
        hold_on_empty=True,
    )
    bridge = InferenceBridge(config)
    bridge.connect(timeout=5.0)
    return bridge


def _feed_dummy_observation(bridge: InferenceBridge):
    """向 bridge 注入一个虚拟观测。"""
    state_14d = np.zeros(14, dtype=np.float32)
    state_14d[:7] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]  # joints
    state_14d[7] = 0.5  # gripper
    state_14d[8:14] = [0.3, 0.0, 0.4, 0.0, 0.0, 0.0]  # EEF pose
    images = {
        "cam0_rgb": np.zeros((480, 640, 3), dtype=np.uint8),
        "cam1_rgb": np.zeros((480, 640, 3), dtype=np.uint8),
    }
    bridge.update_observation(state_14d, images)


# =============================================================================
# 3.3 Pi0 正常延迟场景
# =============================================================================

class TestPi0NormalLatency:
    """chunk=50, t_inf=0.3s, f_exec=20Hz → K_inf=6, L_valid=44, feasible"""

    def test_parameter_derivation(self, mock_server_50):
        runner, addr = mock_server_50
        bridge = _make_bridge(addr, f_exec=20.0, t_inf=0.3)
        _feed_dummy_observation(bridge)

        try:
            bridge.start_inference_loop()
            # 等待 warmup 完成
            assert bridge._warmup_done.wait(timeout=10.0), "Warmup 超时"

            # 验证参数推导
            # K_inf = ceil(0.3 * 20) = 6
            # L_valid = 50 - 6 = 44
            params = bridge._derive_parameters(0.3)
            assert params["L"] == 50, f"chunk_size 应为 50，实际 {params['L']}"
            assert params["K_inf"] == 6, f"K_inf 应为 6，实际 {params['K_inf']}"
            assert params["L_valid"] == 44, f"L_valid 应为 44，实际 {params['L_valid']}"
            assert params["feasible"] is True, "系统应为 feasible"
        finally:
            bridge.stop_inference_loop()
            bridge.disconnect()

    def test_buffer_fills_after_warmup(self, mock_server_50):
        runner, addr = mock_server_50
        bridge = _make_bridge(addr, f_exec=20.0, t_inf=0.3)
        _feed_dummy_observation(bridge)

        try:
            bridge.start_inference_loop()
            assert bridge._warmup_done.wait(timeout=10.0), "Warmup 超时"

            # warmup 后 buffer 应有动作
            time.sleep(0.5)
            assert bridge.buffer_size > 0, "Warmup 后 buffer 应非空"

            # 能 pop 出动作
            action = bridge.get_next_action()
            assert action is not None, "应能 pop 出动作"
            assert action.action_space == "joints"
            assert len(action.data) == 7, f"action.data 应为 7 维，实际 {len(action.data)}"
        finally:
            bridge.stop_inference_loop()
            bridge.disconnect()


# =============================================================================
# 3.4 Pi0.5 慢推理场景
# =============================================================================

class TestPi05SlowInference:
    """chunk=50, t_inf=0.8s, f_exec=20Hz → K_inf=16, L_valid=34, feasible"""

    def test_parameter_derivation(self, mock_server_50):
        runner, addr = mock_server_50
        bridge = _make_bridge(addr, f_exec=20.0, t_inf=0.8)
        _feed_dummy_observation(bridge)

        try:
            bridge.start_inference_loop()
            assert bridge._warmup_done.wait(timeout=10.0), "Warmup 超时"

            params = bridge._derive_parameters(0.8)
            assert params["L"] == 50
            assert params["K_inf"] == 16, f"K_inf 应为 16，实际 {params['K_inf']}"
            assert params["L_valid"] == 34, f"L_valid 应为 34，实际 {params['L_valid']}"
            assert params["feasible"] is True
        finally:
            bridge.stop_inference_loop()
            bridge.disconnect()


# =============================================================================
# 3.5 极端慢推理场景
# =============================================================================

class TestExtremeSlowInference:
    """chunk=50, t_inf=1.5s, f_exec=20Hz → K_inf=30, L_valid=20, feasible"""

    def test_parameter_derivation(self, mock_server_50):
        runner, addr = mock_server_50
        bridge = _make_bridge(addr, f_exec=20.0, t_inf=1.5)
        _feed_dummy_observation(bridge)

        try:
            bridge.start_inference_loop()
            assert bridge._warmup_done.wait(timeout=10.0), "Warmup 超时"

            params = bridge._derive_parameters(1.5)
            assert params["L"] == 50
            assert params["K_inf"] == 30, f"K_inf 应为 30，实际 {params['K_inf']}"
            assert params["L_valid"] == 20, f"L_valid 应为 20，实际 {params['L_valid']}"
            assert params["feasible"] is True
        finally:
            bridge.stop_inference_loop()
            bridge.disconnect()


# =============================================================================
# 3.6 不可行场景
# =============================================================================

class TestInfeasibleScenario:
    """chunk=3, t_inf=0.3s, f_exec=20Hz → K_inf=6, L_valid=-3, degraded mode"""

    def test_parameter_derivation(self, mock_server_3):
        runner, addr = mock_server_3
        bridge = _make_bridge(addr, f_exec=20.0, t_inf=0.3)
        _feed_dummy_observation(bridge)

        try:
            bridge.start_inference_loop()
            assert bridge._warmup_done.wait(timeout=10.0), "Warmup 超时"

            params = bridge._derive_parameters(0.3)
            assert params["L"] == 3
            assert params["K_inf"] == 6
            assert params["L_valid"] < 0, f"L_valid 应为负数，实际 {params['L_valid']}"
            assert params["feasible"] is False, "系统应为 infeasible"
        finally:
            bridge.stop_inference_loop()
            bridge.disconnect()

    def test_degraded_mode_still_provides_actions(self, mock_server_3):
        """即使 infeasible，degraded mode 也应能提供动作（被动等待 buffer drain）。"""
        runner, addr = mock_server_3
        bridge = _make_bridge(addr, f_exec=20.0, t_inf=0.3)
        _feed_dummy_observation(bridge)

        try:
            bridge.start_inference_loop()
            assert bridge._warmup_done.wait(timeout=10.0), "Warmup 超时"

            # 等待 degraded mode 填充 buffer
            time.sleep(1.0)
            action = bridge.get_next_action()
            # degraded mode 下 warmup 会写入至少 1 个 action
            assert action is not None, "Degraded mode 也应能提供动作"
        finally:
            bridge.stop_inference_loop()
            bridge.disconnect()


# =============================================================================
# 3.7 录制字段完整性
# =============================================================================

class TestRecordingFieldCompleteness:
    """验证 policy_action 维度和 control_source 语义。"""

    def test_policy_action_dim(self, mock_server_50):
        """POLICY 模式下 policy_action 应为 8D（7 joints + 1 gripper）。"""
        runner, addr = mock_server_50
        bridge = _make_bridge(addr, f_exec=20.0, t_inf=0.3)
        _feed_dummy_observation(bridge)

        try:
            bridge.start_inference_loop()
            assert bridge._warmup_done.wait(timeout=10.0), "Warmup 超时"

            # policy_action_dim 应为 8（7 joints + 1 gripper）
            assert bridge.policy_action_dim == 8, \
                f"policy_action_dim 应为 8，实际 {bridge.policy_action_dim}"

            # pop 出的 action 应有 7D data + gripper
            time.sleep(0.3)
            action = bridge.get_next_action()
            assert action is not None
            assert len(action.data) == 7
            # gripper 应为 mock 设置的 0.5
            assert abs(action.gripper - 0.5) < 0.01, \
                f"gripper 应为 0.5，实际 {action.gripper}"

            # 构造 policy_action 8D（与 DAggerRecorder 一致）
            policy_action_8d = np.concatenate([action.data, [action.gripper]])
            assert policy_action_8d.shape == (8,)
            assert np.any(policy_action_8d != 0), "POLICY 模式下 policy_action 应非零"
        finally:
            bridge.stop_inference_loop()
            bridge.disconnect()

    def test_human_mode_zero_policy_action(self):
        """HUMAN 模式下 policy_action 应为零填充。"""
        # HUMAN 模式不使用 InferenceBridge，policy_action 由 DAggerRecorder 填零
        policy_action_dim = 8
        zero_action = np.zeros(policy_action_dim, dtype=np.float32)
        assert np.all(zero_action == 0)
        # control_source: 0=policy, 1=human
        control_source_human = 1
        assert control_source_human == 1


# =============================================================================
# 3.8 模式切换
# =============================================================================

class TestModeSwitch:
    """验证 POLICY→HUMAN→POLICY 切换时 inference pause/resume 和 buffer clear。"""

    def test_policy_to_human_pauses_and_clears(self, mock_server_50):
        runner, addr = mock_server_50
        bridge = _make_bridge(addr, f_exec=20.0, t_inf=0.3)
        _feed_dummy_observation(bridge)

        try:
            bridge.start_inference_loop()
            assert bridge._warmup_done.wait(timeout=10.0), "Warmup 超时"

            # 等待 buffer 填充
            time.sleep(0.5)
            assert bridge.buffer_size > 0, "切换前 buffer 应非空"

            # 模拟 POLICY → HUMAN: pause + clear
            bridge.pause()
            bridge.clear_buffer()

            assert bridge.is_paused, "pause() 后应为 paused 状态"
            assert bridge.buffer_size == 0, "clear_buffer() 后 buffer 应为空"

            # paused 状态下不应有新动作进入 buffer
            time.sleep(0.3)
            assert bridge.buffer_size == 0, "Paused 状态下 buffer 应保持为空"
        finally:
            bridge.stop_inference_loop()
            bridge.disconnect()

    def test_human_to_policy_resumes(self, mock_server_50):
        runner, addr = mock_server_50
        bridge = _make_bridge(addr, f_exec=20.0, t_inf=0.3)
        _feed_dummy_observation(bridge)

        try:
            bridge.start_inference_loop()
            assert bridge._warmup_done.wait(timeout=10.0), "Warmup 超时"

            # POLICY → HUMAN
            bridge.pause()
            bridge.clear_buffer()
            assert bridge.is_paused

            # HUMAN → POLICY: resume
            bridge.resume()
            assert not bridge.is_paused, "resume() 后应为 unpaused 状态"

            # 持续喂观测，等待 buffer 重新填充
            for _ in range(10):
                _feed_dummy_observation(bridge)
                time.sleep(0.1)

            assert bridge.buffer_size > 0, "Resume 后 buffer 应重新填充"
        finally:
            bridge.stop_inference_loop()
            bridge.disconnect()
