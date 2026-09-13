"""
常量定义（复制自 RoboCOIN/src/lerobot/extensions/unified_deploy/core/constants.py）

定义支持的策略类型、动作空间、动作类型等常量。
"""

# 支持的策略类型
SUPPORTED_POLICIES = [
    "act",           # Action Chunking Transformer
    "diffusion",     # Diffusion Policy
    "pi0",           # PI0 VLA
    "pi0fast",       # PI0 Fast VLA
    "smolvla",       # SmolVLA
    "vqbet",         # VQ-BeT
    "tdmpc",         # TD-MPC
]

# 支持的动作空间
SUPPORTED_ACTION_SPACES = [
    "joints",        # 关节空间控制 (rm_movej_canfd)
    "pose",          # 笛卡尔空间控制 (rm_movep_canfd)
]

# 支持的动作类型
SUPPORTED_ACTION_TYPES = [
    "absolute",      # 绝对位置/位姿
    "delta",         # 增量位置/位姿
]

# 默认配置
DEFAULT_FPS = 30
DEFAULT_INFERENCE_LATENCY = 1 / DEFAULT_FPS
DEFAULT_OBS_QUEUE_TIMEOUT = 2.0

# Realman 机械臂相关
REALMAN_JOINT_COUNT = 7          # 7 自由度
REALMAN_GRIPPER_MIN = 0          # 夹爪 SDK 最小值
REALMAN_GRIPPER_MAX = 1000       # 夹爪 SDK 最大值
REALMAN_RM_PLUS_BAUD = 115200    # RM+ 生态默认波特率

# gRPC 相关
DEFAULT_GRPC_HOST = "0.0.0.0"
DEFAULT_GRPC_PORT = 50051
DEFAULT_CHUNK_SIZE = 1024 * 1024  # 1MB，用于大数据分块传输
