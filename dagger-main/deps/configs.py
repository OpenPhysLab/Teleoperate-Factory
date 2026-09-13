"""
Client 配置类（复制自 RoboCOIN/src/lerobot/extensions/unified_deploy/client/configs.py）

定义 RealmanClient 的配置参数。
注意：imports 已调整为引用 dagger/deps/ 内的 constants.py
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

from dagger.deps.constants import (
    DEFAULT_FPS,
    DEFAULT_GRPC_PORT,
    REALMAN_RM_PLUS_BAUD,
)


@dataclass
class RealmanClientConfig:
    """
    Realman Client 配置
    """
    # 机器人配置
    robot_ip: str = "192.168.1.18"
    robot_port: int = 8080

    # 服务器配置
    server_address: str = f"127.0.0.1:{DEFAULT_GRPC_PORT}"

    # 控制配置
    frequency: int = 10
    n_action_steps: int = 10

    # 相机配置
    camera_rotations: Dict[str, int] = field(default_factory=dict)
    camera_configs: Dict[str, dict] = field(default_factory=dict)

    # 任务配置
    task: str = ""

    # 运行配置
    dry_run: bool = False
    visualize: bool = False
    vis_scale: float = 1.0
    max_steps: int = 2000

    # 夹爪配置
    gripper_deadband: int = 20
    rm_plus_baud: int = REALMAN_RM_PLUS_BAUD

    # 夹爪二值化配置
    gripper_binarize: bool = False
    gripper_binarize_threshold: int = 500
    gripper_binarize_hysteresis: int = 50

    # 动作队列配置
    chunk_size_threshold: float = 0.5

    def __post_init__(self):
        """验证配置"""
        if self.frequency < 1:
            raise ValueError(f"frequency 必须 >= 1，当前值: {self.frequency}")
        if self.n_action_steps < 1:
            raise ValueError(f"n_action_steps 必须 >= 1，当前值: {self.n_action_steps}")
        if self.chunk_size_threshold < 0 or self.chunk_size_threshold > 1:
            raise ValueError(f"chunk_size_threshold 必须在 [0, 1] 范围内，当前值: {self.chunk_size_threshold}")

    @property
    def frame_interval(self) -> float:
        """帧间隔（秒）"""
        return 1.0 / self.frequency
