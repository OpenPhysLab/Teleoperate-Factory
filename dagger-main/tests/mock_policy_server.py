"""
MockPolicyServer — 模拟 OpenPI 输出特征的 gRPC PolicyServer。

用于集成测试，无需 GPU 或 OpenPI 依赖。支持配置：
- action_horizon: ActionChunk 中的动作数量（OpenPI 默认 50）
- action_dim: 每个动作的关节维度（不含 gripper，默认 7）
- simulated_t_inf: 模拟推理延迟（秒）
- action_pattern: 动作值模式（"constant" 或 "ramp"）
"""

import io
import pickle
import time
import threading
from concurrent import futures
from typing import Optional

import numpy as np
import grpc

# === Path setup ===
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ROBOCOIN_SRC = os.path.join(_PROJECT_ROOT, "RoboCOIN", "src")
if _ROBOCOIN_SRC not in sys.path:
    sys.path.insert(0, _ROBOCOIN_SRC)

import lerobot.transport.services_pb2 as services_pb2
import lerobot.transport.services_pb2_grpc as services_pb2_grpc
from lerobot.extensions.unified_deploy.core.data_types import (
    UnifiedAction,
    ActionChunk,
)


class MockPolicyServer(services_pb2_grpc.AsyncInferenceServicer):
    """模拟 PolicyServer 的 gRPC 服务，返回可配置的 ActionChunk。"""

    def __init__(
        self,
        action_horizon: int = 50,
        action_dim: int = 7,
        simulated_t_inf: float = 0.3,
        action_pattern: str = "constant",
        constant_value: float = 0.1,
        gripper_value: float = 0.5,
    ):
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.simulated_t_inf = simulated_t_inf
        self.action_pattern = action_pattern
        self.constant_value = constant_value
        self.gripper_value = gripper_value

        # 统计
        self.infer_count = 0
        self.last_observation_bytes: Optional[bytes] = None

    def Ready(self, request, context):
        """健康检查，始终返回成功。"""
        return services_pb2.Empty()

    def SendObservations(self, request_iterator, context):
        """接收流式观测数据，拼接 chunks 并存储。"""
        buf = io.BytesIO()
        for chunk in request_iterator:
            buf.write(chunk.data)
        self.last_observation_bytes = buf.getvalue()
        return services_pb2.Empty()

    def GetActions(self, request, context):
        """模拟推理延迟后返回 ActionChunk。"""
        # 模拟推理延迟
        if self.simulated_t_inf > 0:
            time.sleep(self.simulated_t_inf)

        # 生成动作
        actions = []
        for i in range(self.action_horizon):
            if self.action_pattern == "ramp":
                # 线性递增：每步 +0.01
                data = np.full(self.action_dim, self.constant_value + i * 0.01, dtype=np.float32)
            else:
                # constant
                data = np.full(self.action_dim, self.constant_value, dtype=np.float32)

            actions.append(UnifiedAction(
                timestamp=time.time(),
                timestep=i,
                action_space="joints",
                action_type="absolute",
                data=data,
                gripper=self.gripper_value,
            ))

        chunk = ActionChunk(
            actions=actions,
            metadata={
                "task": "mock_task",
                "action_space": "joints",
                "action_type": "absolute",
            },
        )

        self.infer_count += 1
        response_bytes = pickle.dumps(chunk)
        return services_pb2.Actions(data=response_bytes)

    def SendPolicyInstructions(self, request, context):
        """接收策略配置（Mock 忽略）。"""
        return services_pb2.Empty()


class MockPolicyServerRunner:
    """管理 MockPolicyServer 的生命周期（启动/停止）。"""

    def __init__(self, port: int = 0, **server_kwargs):
        """
        Args:
            port: gRPC 端口，0 表示自动分配。
            **server_kwargs: 传给 MockPolicyServer 的参数。
        """
        self.servicer = MockPolicyServer(**server_kwargs)
        self._server: Optional[grpc.Server] = None
        self._port: int = port
        self.address: str = ""

    def start(self) -> str:
        """启动 gRPC 服务器，返回地址（如 "127.0.0.1:50123"）。"""
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        services_pb2_grpc.add_AsyncInferenceServicer_to_server(
            self.servicer, self._server
        )

        if self._port == 0:
            self._port = self._server.add_insecure_port("127.0.0.1:0")
        else:
            self._server.add_insecure_port(f"127.0.0.1:{self._port}")

        self.address = f"127.0.0.1:{self._port}"
        self._server.start()
        return self.address

    def stop(self, grace: float = 1.0):
        """停止 gRPC 服务器。"""
        if self._server:
            self._server.stop(grace)
            self._server = None
