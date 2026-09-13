"""
统一数据类型定义（复制自 RoboCOIN/src/lerobot/extensions/unified_deploy/core/data_types.py）

定义 Client 和 Server 之间通信的标准数据格式。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np


@dataclass
class UnifiedObservation:
    """
    统一观测格式（Client → Server）

    Client 采集机器人状态和图像后，封装为此格式发送给 Server。
    Server 根据此观测进行策略推理。

    Attributes:
        timestamp: Unix 时间戳（秒），用于计算延迟
        timestep: 帧序号，从 0 开始递增
        must_go: 是否必须处理（当动作队列为空时设为 True）
        joints: 关节角度 [n_joints] (rad)，Realman 为 7 维
        ee_pose: 末端位姿 [x, y, z, rx, ry, rz] (m, rad)
        gripper: 夹爪状态，归一化到 [0, 1]，0=全开，1=全闭
        images: 图像字典 {相机名: [H, W, C] uint8}
        task: 任务指令文本（VLA 模型需要）
    """
    timestamp: float
    timestep: int
    must_go: bool = False

    # 状态数据（始终同时提供 joints 和 ee_pose）
    joints: Optional[np.ndarray] = None      # [7] rad
    ee_pose: Optional[np.ndarray] = None     # [6] (m, rad)
    gripper: float = 0.0                     # [0, 1]

    # 图像数据
    images: Dict[str, np.ndarray] = field(default_factory=dict)  # {cam_name: [H,W,C] uint8}

    # 任务指令
    task: str = ""

    def to_dict(self) -> dict:
        """转换为可序列化的字典"""
        return {
            "timestamp": self.timestamp,
            "timestep": self.timestep,
            "must_go": self.must_go,
            "joints": self.joints.tolist() if self.joints is not None else None,
            "ee_pose": self.ee_pose.tolist() if self.ee_pose is not None else None,
            "gripper": self.gripper,
            "images": {k: v.tobytes() for k, v in self.images.items()},
            "image_shapes": {k: v.shape for k, v in self.images.items()},
            "task": self.task,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UnifiedObservation":
        """从字典恢复"""
        images = {}
        if "images" in data and "image_shapes" in data:
            for k, v in data["images"].items():
                shape = data["image_shapes"][k]
                images[k] = np.frombuffer(v, dtype=np.uint8).reshape(shape)

        return cls(
            timestamp=data["timestamp"],
            timestep=data["timestep"],
            must_go=data.get("must_go", False),
            joints=np.array(data["joints"]) if data.get("joints") else None,
            ee_pose=np.array(data["ee_pose"]) if data.get("ee_pose") else None,
            gripper=data.get("gripper", 0.0),
            images=images,
            task=data.get("task", ""),
        )


@dataclass
class UnifiedAction:
    """
    统一动作格式（Server → Client）

    Server 推理后，将动作封装为此格式返回给 Client。
    Client 根据 action_space 和 action_type 执行相应的控制。

    Attributes:
        timestamp: 动作时间戳（秒）
        timestep: 对应的帧序号
        action_space: 动作空间 "joints" | "pose"
        action_type: 动作类型 "absolute" | "delta"
        data: 动作数据
            - joints: [7] rad (关节角度)
            - pose: [6] (x,y,z,rx,ry,rz) (m, rad)
        gripper: 目标夹爪状态 [0, 1]
    """
    timestamp: float
    timestep: int
    action_space: str      # "joints" | "pose"
    action_type: str       # "absolute" | "delta"
    data: np.ndarray       # [7] for joints, [6] for pose
    gripper: float = 0.0   # [0, 1]

    def to_dict(self) -> dict:
        """转换为可序列化的字典"""
        return {
            "timestamp": self.timestamp,
            "timestep": self.timestep,
            "action_space": self.action_space,
            "action_type": self.action_type,
            "data": self.data.tolist(),
            "gripper": self.gripper,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UnifiedAction":
        """从字典恢复"""
        return cls(
            timestamp=data["timestamp"],
            timestep=data["timestep"],
            action_space=data["action_space"],
            action_type=data["action_type"],
            data=np.array(data["data"]),
            gripper=data.get("gripper", 0.0),
        )


@dataclass
class ActionChunk:
    """
    动作块（包含多个连续动作）

    Server 一次推理返回多个动作，封装为 ActionChunk。

    Attributes:
        actions: 动作列表
        metadata: 元数据（可选），包含：
            - task: Server 实际使用的任务指令（VLA 模型）
            - action_space: 动作空间 (joints/pose)
            - action_type: 动作类型 (absolute/delta)
    """
    actions: List[UnifiedAction]
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> dict:
        return {
            "actions": [a.to_dict() for a in self.actions],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ActionChunk":
        return cls(
            actions=[UnifiedAction.from_dict(a) for a in data["actions"]],
            metadata=data.get("metadata"),
        )


@dataclass
class PolicyConfig:
    """
    策略配置（Client 发送给 Server，用于初始化策略）
    """
    policy_type: str
    pretrained_path: str
    action_space: str = "joints"
    action_type: str = "absolute"
    device: str = "cuda"
    n_action_steps: int = 0
    openpi_config_name: str = ""
    default_prompt: str = ""

    def to_dict(self) -> dict:
        return {
            "policy_type": self.policy_type,
            "pretrained_path": self.pretrained_path,
            "action_space": self.action_space,
            "action_type": self.action_type,
            "device": self.device,
            "n_action_steps": self.n_action_steps,
            "openpi_config_name": self.openpi_config_name,
            "default_prompt": self.default_prompt,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PolicyConfig":
        return cls(**data)
