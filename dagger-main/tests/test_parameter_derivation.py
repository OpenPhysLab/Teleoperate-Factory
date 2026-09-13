"""
Phase 1.5B 参数推导 + Warmup 单元测试

测试 _derive_parameters()、n_action_steps 配置行为、以及多次 warmup 测量。
"""

import math
import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# 路径设置
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ROBOCOIN_SRC = os.path.join(_PROJECT_ROOT, "RoboCOIN", "src")
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _ROBOCOIN_SRC not in sys.path:
    sys.path.insert(0, _ROBOCOIN_SRC)

from dagger.async_inference_client import AsyncInferenceClientConfig, AsyncInferenceClient


def make_client(f_exec=30.0, n_action_steps=0, T_inter=0.0, t_inf=0.0,
                chunk_size_threshold=0, model_chunk_size=None):
    """创建一个用于测试的 AsyncInferenceClient（跳过 robot/gRPC 初始化）"""
    config = AsyncInferenceClientConfig(
        f_exec=f_exec,
        n_action_steps=n_action_steps,
        T_inter=T_inter,
        t_inf=t_inf,
        chunk_size_threshold=chunk_size_threshold,
        # 测试不需要真实机器人连接
        dry_run=True,
    )
    client = AsyncInferenceClient(config)
    if model_chunk_size is not None:
        client._model_chunk_size = model_chunk_size
    return client


class TestDeriveParameters(unittest.TestCase):
    """测试 _derive_parameters()"""

    def test_standard_case(self):
        """标准场景: f_exec=30, chunk=50, t_inf=0.2s, n_action_steps=0"""
        client = make_client(f_exec=30.0, n_action_steps=0, T_inter=0.0,
                             model_chunk_size=50)
        result = client._derive_parameters(t_inf=0.2)

        self.assertEqual(result["L"], 50)
        self.assertEqual(result["L_eff"], 50)  # n_action_steps=0 → 使用全部
        self.assertEqual(result["K_inf"], math.ceil(0.2 * 30))  # = 6
        self.assertEqual(result["L_valid"], 50 - 6)  # = 44
        self.assertAlmostEqual(result["T_inter_max"], 50 / 30.0 - 0.2, places=5)
        self.assertTrue(result["feasible"])

    def test_full_chunk_case(self):
        """全 chunk 使用: n_action_steps=0, chunk=20, t_inf=0.1s"""
        client = make_client(f_exec=30.0, n_action_steps=0, T_inter=0.0,
                             model_chunk_size=20)
        result = client._derive_parameters(t_inf=0.1)

        self.assertEqual(result["L_eff"], 20)
        K_inf = math.ceil(0.1 * 30)  # = 3
        self.assertEqual(result["K_inf"], K_inf)
        self.assertEqual(result["L_valid"], 20 - K_inf)
        self.assertTrue(result["feasible"])

    def test_n_action_steps_truncation(self):
        """n_action_steps < chunk_size: L_eff 被截取"""
        client = make_client(f_exec=30.0, n_action_steps=10, T_inter=0.0,
                             model_chunk_size=50)
        result = client._derive_parameters(t_inf=0.1)

        self.assertEqual(result["L"], 50)
        self.assertEqual(result["L_eff"], 10)  # min(10, 50)
        K_inf = math.ceil(0.1 * 30)  # = 3
        self.assertEqual(result["L_valid"], 10 - K_inf)  # = 7
        self.assertTrue(result["feasible"])

    def test_n_action_steps_exceeds_chunk(self):
        """n_action_steps > chunk_size: L_eff = chunk_size"""
        client = make_client(f_exec=30.0, n_action_steps=100, T_inter=0.0,
                             model_chunk_size=50)
        result = client._derive_parameters(t_inf=0.1)

        self.assertEqual(result["L_eff"], 50)  # min(100, 50) = 50

    def test_boundary_case(self):
        """边界: t_inf 恰好消耗所有 action（L_valid=0 → infeasible）"""
        # chunk=10, f_exec=30, t_inf = 10/30 = 0.333... → K_inf = ceil(10) = 10
        client = make_client(f_exec=30.0, n_action_steps=0, T_inter=0.0,
                             model_chunk_size=10)
        t_inf = 10.0 / 30.0  # exactly uses all steps
        result = client._derive_parameters(t_inf=t_inf)

        self.assertEqual(result["K_inf"], 10)
        self.assertEqual(result["L_valid"], 0)
        self.assertFalse(result["feasible"])

    def test_infeasible_high_latency(self):
        """不可行: 推理延迟超过 chunk 长度"""
        client = make_client(f_exec=30.0, n_action_steps=0, T_inter=0.0,
                             model_chunk_size=10)
        result = client._derive_parameters(t_inf=1.0)  # K_inf = ceil(30) = 30 >> 10

        self.assertFalse(result["feasible"])
        self.assertLess(result["L_valid"], 0)

    def test_T_inter_clamped_to_max(self):
        """T_inter 过大时被 clamp 到 T_inter_max"""
        client = make_client(f_exec=30.0, n_action_steps=0, T_inter=5.0,
                             model_chunk_size=50)
        result = client._derive_parameters(t_inf=0.1)

        # T_inter_max = 50/30 - 0.1 ≈ 1.567, T_inter=5.0 被 clamp
        self.assertTrue(result["feasible"])  # 物理上可行
        self.assertAlmostEqual(result["T_inter"], result["T_inter_max"], places=5)

    def test_model_chunk_size_unset(self):
        """_model_chunk_size 未设置: 降级"""
        client = make_client(f_exec=30.0, model_chunk_size=None)
        result = client._derive_parameters(t_inf=0.1)

        self.assertFalse(result["feasible"])
        self.assertEqual(result["L"], 0)
        self.assertEqual(result["chunk_size_threshold"], 2)

    def test_model_chunk_size_zero(self):
        """_model_chunk_size=0: 降级"""
        client = make_client(f_exec=30.0, model_chunk_size=0)
        result = client._derive_parameters(t_inf=0.1)

        self.assertFalse(result["feasible"])
        self.assertEqual(result["L"], 0)

    def test_chunk_size_threshold_override(self):
        """配置 chunk_size_threshold 作为 buffer 安全网阈值（不再覆盖 L）"""
        client = make_client(f_exec=30.0, n_action_steps=0, T_inter=0.0,
                             chunk_size_threshold=20, model_chunk_size=50)
        result = client._derive_parameters(t_inf=0.1)

        self.assertEqual(result["L"], 50)  # L = model_chunk_size, not overridden
        self.assertEqual(result["L_eff"], 50)
        self.assertEqual(result["chunk_size_threshold"], 20)  # 配置值用于 buffer 阈值

    def test_t_inf_zero(self):
        """t_inf=0: K_inf=0, 全部 action 有效"""
        client = make_client(f_exec=30.0, n_action_steps=0, T_inter=0.0,
                             model_chunk_size=50)
        result = client._derive_parameters(t_inf=0.0)

        self.assertEqual(result["K_inf"], 0)
        self.assertEqual(result["L_valid"], 50)
        self.assertTrue(result["feasible"])


class TestNActionStepsConfig(unittest.TestCase):
    """测试 n_action_steps=0 配置不被 or 100 覆盖"""

    def test_n_action_steps_zero_stays_zero(self):
        """n_action_steps=0 在 YAML 中应保持为 0，不被默认为 100"""
        config = AsyncInferenceClientConfig(
            n_action_steps=0,
            dry_run=True,
        )
        self.assertEqual(config.n_action_steps, 0)

    def test_n_action_steps_positive_preserved(self):
        """n_action_steps=10 应保持为 10"""
        config = AsyncInferenceClientConfig(
            n_action_steps=10,
            dry_run=True,
        )
        self.assertEqual(config.n_action_steps, 10)

    def test_config_t_inf_default(self):
        """t_inf 默认为 0.0"""
        config = AsyncInferenceClientConfig(dry_run=True)
        self.assertEqual(config.t_inf, 0.0)

    def test_config_chunk_size_threshold_default(self):
        """chunk_size_threshold 默认为 0"""
        config = AsyncInferenceClientConfig(dry_run=True)
        self.assertEqual(config.chunk_size_threshold, 0)


class TestWarmupMultiMeasure(unittest.TestCase):
    """测试多次 warmup 测量逻辑"""

    def _make_warmup_client(self, t_inf_config=0.0):
        """创建用于 warmup 测试的 client（mock gRPC）"""
        client = make_client(f_exec=30.0, n_action_steps=0, t_inf=t_inf_config)
        # Mock gRPC stub
        client.stub = MagicMock()
        client.stub.Ready.return_value = None
        return client

    def _mock_warmup_single(self, client, t_inf_sequence):
        """
        Mock _warmup_single 返回指定的 t_inf 序列。
        每次调用返回 (t_inf, fake_actions)。
        """
        from lerobot.extensions.unified_deploy.core.data_types import UnifiedAction
        import numpy as np

        fake_actions = [
            UnifiedAction(
                timestamp=0.0, timestep=i, action_space="joints",
                action_type="absolute", data=np.zeros(7, dtype=np.float32),
                gripper=0.0,
            )
            for i in range(50)
        ]

        call_count = [0]
        def side_effect(tag=""):
            idx = call_count[0]
            call_count[0] += 1
            if idx < len(t_inf_sequence):
                t = t_inf_sequence[idx]
                # 设置 chunk_size（模拟真实行为）
                client._model_chunk_size = len(fake_actions)
                return (t, fake_actions)
            return None

        return side_effect

    def test_multi_warmup_discards_cold_start(self):
        """多次 warmup 应丢弃第 1 次冷启动，取后面的平均值"""
        client = self._make_warmup_client(t_inf_config=0.0)

        # 模拟: 第1次 3.4s（冷启动），后4次 ~0.1s
        t_inf_seq = [3.4, 0.10, 0.12, 0.09, 0.11]
        client._warmup_single = MagicMock(side_effect=self._mock_warmup_single(client, t_inf_seq))

        t_inf = client._measure_inference_time(n_warmup=5)

        # 应丢弃 3.4s，取 [0.10, 0.12, 0.09, 0.11] 的平均
        expected = (0.10 + 0.12 + 0.09 + 0.11) / 4
        self.assertAlmostEqual(t_inf, expected, places=5)
        self.assertEqual(client._warmup_single.call_count, 5)

    def test_multi_warmup_minimum_two(self):
        """n_warmup < 2 时自动提升到 2"""
        client = self._make_warmup_client(t_inf_config=0.0)

        t_inf_seq = [0.5, 0.1]
        client._warmup_single = MagicMock(side_effect=self._mock_warmup_single(client, t_inf_seq))

        t_inf = client._measure_inference_time(n_warmup=1)  # 会被提升到 2

        # 丢弃 0.5s，取 [0.1] 的平均
        self.assertAlmostEqual(t_inf, 0.1, places=5)
        self.assertEqual(client._warmup_single.call_count, 2)

    def test_config_t_inf_skips_measurement(self):
        """config.t_inf > 0 时直接使用配置值，但仍执行 1 次推理获取 chunk_size"""
        client = self._make_warmup_client(t_inf_config=0.5)

        t_inf_seq = [0.1]  # chunk_size 探测
        client._warmup_single = MagicMock(side_effect=self._mock_warmup_single(client, t_inf_seq))

        t_inf = client._measure_inference_time(n_warmup=5)

        self.assertAlmostEqual(t_inf, 0.5, places=5)  # 使用配置值
        self.assertEqual(client._warmup_single.call_count, 1)  # 只调用 1 次获取 chunk_size
        self.assertEqual(client._model_chunk_size, 50)

    def test_all_warmup_fail_returns_zero(self):
        """所有 warmup 推理失败时返回 0.0"""
        client = self._make_warmup_client(t_inf_config=0.0)

        # 所有调用返回 None（失败）
        client._warmup_single = MagicMock(return_value=None)

        t_inf = client._measure_inference_time(n_warmup=3)

        self.assertEqual(t_inf, 0.0)

    def test_partial_warmup_failure(self):
        """部分 warmup 失败时，用成功的结果计算"""
        client = self._make_warmup_client(t_inf_config=0.0)

        # 第1次成功(冷启动)，第2次失败，第3次成功，第4次成功，第5次失败
        call_count = [0]
        from lerobot.extensions.unified_deploy.core.data_types import UnifiedAction
        import numpy as np
        fake_actions = [
            UnifiedAction(
                timestamp=0.0, timestep=i, action_space="joints",
                action_type="absolute", data=np.zeros(7, dtype=np.float32),
                gripper=0.0,
            )
            for i in range(50)
        ]
        results = [
            (2.0, fake_actions),  # 冷启动
            None,                  # 失败
            (0.12, fake_actions),  # 成功
            (0.10, fake_actions),  # 成功
            None,                  # 失败
        ]
        def side_effect(tag=""):
            idx = call_count[0]
            call_count[0] += 1
            if idx < len(results) and results[idx] is not None:
                client._model_chunk_size = 50
            return results[idx] if idx < len(results) else None

        client._warmup_single = MagicMock(side_effect=side_effect)

        t_inf = client._measure_inference_time(n_warmup=5)

        # 成功的: [2.0, 0.12, 0.10]，丢弃第1个(2.0)，平均 [0.12, 0.10]
        expected = (0.12 + 0.10) / 2
        self.assertAlmostEqual(t_inf, expected, places=5)

    def test_only_cold_start_succeeds(self):
        """只有冷启动那次成功时，只能用那个值"""
        client = self._make_warmup_client(t_inf_config=0.0)

        results_data = [3.0]  # 只有 1 次成功
        client._warmup_single = MagicMock(
            side_effect=self._mock_warmup_single(client, results_data)
        )

        t_inf = client._measure_inference_time(n_warmup=3)

        # 只有 1 次成功，无法丢弃，直接使用
        self.assertAlmostEqual(t_inf, 3.0, places=5)


if __name__ == "__main__":
    unittest.main()
