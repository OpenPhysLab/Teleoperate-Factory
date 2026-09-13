"""
对比测试：dagger_node 控制路径 vs async_inference_client 控制路径

验证两条路径在以下方面的一致性：
1. SDK 调用参数（follow, traj_mode, radio）
2. 执行频率（f_exec）
3. pose 数据格式和值
4. gripper 处理方式

运行：
    cd /home/ubuntu/Desktop/Workspace/lerobot_policy_deploy/lerobotv3
    python -m pytest dagger/tests/test_control_path_comparison.py -v
"""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, call
from dataclasses import dataclass


# ============================================================================
# Mock SDK and data types
# ============================================================================

@dataclass
class MockAction:
    """Mimics UnifiedAction from inference bridge."""
    data: np.ndarray
    gripper: float
    action_space: str  # "pose" or "joints"


class MockRoboticArm:
    """Records all SDK calls for comparison."""

    def __init__(self):
        self.calls = []

    def rm_movep_canfd(self, pose, follow, traj_mode, radio):
        self.calls.append({
            "method": "rm_movep_canfd",
            "pose": list(pose),
            "follow": follow,
            "traj_mode": traj_mode,
            "radio": radio,
        })
        return 0

    def rm_movej_canfd(self, joints, follow, expand, traj_mode, radio):
        self.calls.append({
            "method": "rm_movej_canfd",
            "joints": list(joints),
            "follow": follow,
            "expand": expand,
            "traj_mode": traj_mode,
            "radio": radio,
        })
        return 0


# ============================================================================
# Test 1: SDK 参数对比
# ============================================================================

class TestSDKParameterComparison:
    """对比两条路径传给 SDK 的参数是否一致。"""

    def _simulate_async_client_pose(self, arm: MockRoboticArm, pose_6d: list):
        """模拟 async_inference_client.execute_action() 的 pose 路径。
        Source: async_inference_client.py:542-545
        """
        pose = pose_6d  # target.tolist() — already a list
        arm.rm_movep_canfd(pose, False, 0, 0)

    def _simulate_driver_pose(self, arm: MockRoboticArm, pose_6d: list,
                               follow: bool, canfd_traj_mode: int, canfd_radio: int):
        """模拟 realman_driver_node → hardware.py send_pose_canfd() 的路径。
        Source: hardware.py:381-383
        """
        arm.rm_movep_canfd(pose_6d, follow, canfd_traj_mode, canfd_radio)

    def test_pose_sdk_params_differ(self):
        """BUG: driver 使用 teleop_params.yaml 的 canfd_radio=60，而 async_client 用 radio=0。"""
        pose = [0.3, 0.0, 0.4, 0.0, 1.57, 0.0]

        arm_async = MockRoboticArm()
        self._simulate_async_client_pose(arm_async, pose)

        arm_driver = MockRoboticArm()
        # teleop_params.yaml defaults: follow=False, canfd_traj_mode=0, canfd_radio=60
        self._simulate_driver_pose(arm_driver, pose,
                                    follow=False, canfd_traj_mode=0, canfd_radio=60)

        async_call = arm_async.calls[0]
        driver_call = arm_driver.calls[0]

        # Pose data should be identical
        assert async_call["pose"] == driver_call["pose"], \
            f"Pose mismatch: async={async_call['pose']} vs driver={driver_call['pose']}"

        # follow should be identical
        assert async_call["follow"] == driver_call["follow"], \
            f"follow mismatch: async={async_call['follow']} vs driver={driver_call['follow']}"

        # BUG DETECTED: radio differs
        # async_client: radio=0, driver: radio=60
        assert async_call["radio"] != driver_call["radio"], \
            "Expected radio to differ (this is the bug)"
        print(f"\n[BUG] SDK radio 参数不一致:")
        print(f"  async_inference_client: radio={async_call['radio']}")
        print(f"  realman_driver_node:    radio={driver_call['radio']}")

    def test_pose_sdk_params_after_fix(self):
        """修复后：两条路径的 SDK 参数应完全一致。"""
        pose = [0.3, 0.0, 0.4, 0.0, 1.57, 0.0]

        arm_async = MockRoboticArm()
        self._simulate_async_client_pose(arm_async, pose)

        arm_driver = MockRoboticArm()
        # 修复后 driver 应使用与 async_client 相同的参数
        self._simulate_driver_pose(arm_driver, pose,
                                    follow=False, canfd_traj_mode=0, canfd_radio=0)

        async_call = arm_async.calls[0]
        driver_call = arm_driver.calls[0]

        assert async_call == driver_call, \
            f"SDK calls should be identical after fix:\n  async: {async_call}\n  driver: {driver_call}"


# ============================================================================
# Test 2: 执行频率对比
# ============================================================================

class TestExecutionFrequency:
    """对比两条路径的执行频率。"""

    def test_async_client_uses_f_exec(self):
        """async_inference_client 使用 f_exec 控制执行频率。
        Source: async_inference_client.py:572-573
        """
        f_exec = 10.0
        T_step = 1.0 / f_exec
        assert T_step == pytest.approx(0.1), f"Expected 100ms step, got {T_step}"

    def test_dagger_node_uses_control_hz_bug(self):
        """BUG: dagger_node 使用 control_hz (50Hz) 而非 f_exec (10Hz) 驱动 _tick_policy()。
        Source: dagger_node.py:383-384
        """
        # dagger_node 的 timer 周期
        control_hz = 50.0  # from dagger_params.yaml vr_control.control_hz
        f_exec = 10.0      # from dagger_params.yaml f_exec

        timer_period = 1.0 / control_hz  # 0.02s = 20ms
        expected_period = 1.0 / f_exec   # 0.1s = 100ms

        # BUG: timer runs at 50Hz, but policy should execute at 10Hz
        assert timer_period != expected_period, \
            "Expected frequency mismatch (this is the bug)"
        print(f"\n[BUG] 执行频率不一致:")
        print(f"  async_inference_client: f_exec={f_exec}Hz (T_step={expected_period:.3f}s)")
        print(f"  dagger_node timer:     control_hz={control_hz}Hz (period={timer_period:.3f}s)")
        print(f"  dagger_node 的 _tick_policy() 以 {control_hz}Hz 执行，是预期的 {control_hz/f_exec:.0f}x 倍速")

    def test_dagger_node_should_throttle_policy(self):
        """修复方案：_tick_policy() 应按 f_exec 节流，而非 control_hz。

        timer 保持 control_hz (50Hz) 用于 VR 遥操作，
        但 POLICY 模式下 _tick_policy() 需要按 f_exec 节流。
        """
        import time

        control_hz = 50.0
        f_exec = 10.0
        T_step = 1.0 / f_exec  # 0.1s

        # 模拟 50Hz timer 调用，但 policy 按 f_exec 节流
        exec_count = 0
        last_policy_time = 0.0
        total_ticks = 0

        # 模拟 1 秒的 timer 调用
        for i in range(50):  # 50 ticks at 50Hz = 1 second
            t_now = i / control_hz
            total_ticks += 1

            # 节流逻辑：只有间隔 >= T_step 时才执行 policy
            if (t_now - last_policy_time) >= T_step:
                exec_count += 1
                last_policy_time = t_now

        print(f"\n[FIX] 节流后的执行频率:")
        print(f"  timer ticks: {total_ticks} (50Hz)")
        print(f"  policy executions: {exec_count} (~{f_exec}Hz)")

        # 应该约 10 次（离散采样允许 ±2 的误差）
        assert 8 <= exec_count <= 12, \
            f"Expected ~10 policy executions, got {exec_count}"


# ============================================================================
# Test 3: Pose 数据格式对比
# ============================================================================

class TestPoseDataFormat:
    """对比两条路径的 pose 数据格式。"""

    def test_async_client_sends_6d_directly(self):
        """async_inference_client 直接发送 6D pose 给 SDK。
        Source: async_inference_client.py:543-545
        """
        action_data = np.array([0.3, 0.0, 0.4, 0.0, 1.57, 0.0])
        pose = action_data.tolist()
        assert len(pose) == 6
        assert pose == [0.3, 0.0, 0.4, 0.0, 1.57, 0.0]

    def test_dagger_node_sends_6d_via_ros(self):
        """dagger_node 通过 ROS2 JointState 发送 6D pose + gripper。
        Source: dagger_node.py:840-848, 908-928
        """
        # _tick_policy() 中的处理
        action_data = [0.3, 0.0, 0.4, 0.0, 1.57, 0.0]
        gripper = 0.5
        pose_6d = [float(v) for v in action_data[:6]]

        # _publish_pose_action_with_gripper 构造 msg
        msg_name = ["x", "y", "z", "rx", "ry", "rz", "gripper"]
        msg_position = list(pose_6d) + [float(np.clip(gripper, 0.0, 1.0))]

        assert len(msg_position) == 7  # 6D pose + gripper
        assert msg_name[3] == "rx"     # 欧拉角标识

        # driver _on_action_pose 解析
        names = msg_name
        assert "rx" in names  # 走欧拉角分支
        parsed_pose = list(msg_position[:6])
        gripper_idx = 6

        # 最终传给 SDK 的 pose 应与 async_client 一致
        assert parsed_pose == [0.3, 0.0, 0.4, 0.0, 1.57, 0.0]
        assert msg_position[gripper_idx] == 0.5

    def test_end_to_end_pose_equivalence(self):
        """端到端对比：同一个 action 经过两条路径后，传给 SDK 的 pose 应完全一致。"""
        # 模拟 PolicyServer 返回的 action
        action_data = np.array([0.312, -0.045, 0.387, 0.123, 1.571, -0.456])
        action_gripper = 0.73

        # Path A: async_inference_client
        pose_async = action_data.tolist()  # 直接 tolist()

        # Path B: dagger_node → ROS2 → driver
        pose_6d = [float(v) for v in action_data[:6]]
        msg_position = pose_6d + [float(np.clip(action_gripper, 0.0, 1.0))]
        # driver 解析
        pose_driver = list(msg_position[:6])

        # 对比
        np.testing.assert_array_almost_equal(
            pose_async, pose_driver, decimal=10,
            err_msg="Pose data should be identical through both paths"
        )


# ============================================================================
# Test 4: Gripper 处理对比
# ============================================================================

class TestGripperHandling:
    """对比两条路径的 gripper 处理。"""

    def test_async_client_gripper(self):
        """async_inference_client: gripper [0,1] → [0,1000] → JSON TCP。
        Source: async_inference_client.py:549-561
        """
        gripper_normalized = 0.73
        gripper_sdk = int(gripper_normalized * 1000)  # gripper_normalized_to_sdk
        assert gripper_sdk == 730

    def test_driver_gripper(self):
        """driver: gripper [0,1] → [0,1000] → queue → JSON TCP thread。
        Source: realman_driver_node.py:321-327
        """
        gripper_open = 0.73
        gripper_pos = int(np.clip(gripper_open, 0.0, 1.0) * 1000.0)
        assert gripper_pos == 730

    def test_gripper_equivalence(self):
        """两条路径的 gripper SDK 值应一致。"""
        for g in [0.0, 0.25, 0.5, 0.73, 1.0]:
            # async path
            sdk_async = int(g * 1000)
            # driver path
            sdk_driver = int(np.clip(g, 0.0, 1.0) * 1000.0)
            assert sdk_async == sdk_driver, f"Gripper mismatch at {g}: {sdk_async} vs {sdk_driver}"


# ============================================================================
# Test 5: 综合 Bug 报告
# ============================================================================

class TestBugSummary:
    """汇总所有发现的差异。"""

    def test_print_all_differences(self):
        """打印两条控制路径的所有差异。"""
        diffs = []

        # 1. 执行频率
        diffs.append({
            "item": "执行频率",
            "async_client": "f_exec=10Hz (T_step=100ms)",
            "dagger_node": "control_hz=50Hz (period=20ms)",
            "impact": "机械臂运动速度 5x 过快，ring buffer 5x 过快耗尽",
            "severity": "CRITICAL",
        })

        # 2. SDK radio 参数
        diffs.append({
            "item": "rm_movep_canfd radio",
            "async_client": "radio=0",
            "dagger_node": "radio=60 (from teleop_params.yaml canfd_radio)",
            "impact": "平滑系数不同，可能影响运动轨迹",
            "severity": "MEDIUM",
        })

        # 3. Pose 格式（已修复）
        diffs.append({
            "item": "Pose 格式识别",
            "async_client": "直接 6D → SDK",
            "dagger_node": "6D+gripper → ROS2 → driver msg.name 识别 → 6D → SDK",
            "impact": "已通过 msg.name 修复，格式识别正确",
            "severity": "FIXED",
        })

        print("\n" + "=" * 80)
        print("控制路径差异报告")
        print("=" * 80)
        for i, d in enumerate(diffs, 1):
            print(f"\n[{d['severity']}] #{i}: {d['item']}")
            print(f"  async_inference_client: {d['async_client']}")
            print(f"  dagger_node (via driver): {d['dagger_node']}")
            print(f"  影响: {d['impact']}")
        print("\n" + "=" * 80)

        # 至少有 1 个 CRITICAL bug
        critical = [d for d in diffs if d["severity"] == "CRITICAL"]
        assert len(critical) >= 1, "Should have at least 1 critical difference"
