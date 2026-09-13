"""
DAgger 数据录制器（纯 Python，无 ROS2 依赖）

录制 LeRobot v2.1 格式数据，主轨迹格式与 VR 遥操完全一致:
- observation.state (14D): 机械臂完整状态 [7 joints_rad, 1 gripper_01, 6 eef_pose]
- action (14D): action[t] = state[t+1]（后处理生成）
- observation.images.*: RGB 图像

DAgger 附加字段（VR 数据无此字段，训练时可忽略）:
- policy_action (N,): policy 推理原始输出
- control_source (1,): 0=policy, 1=human

采用与 VR 录制一致的"占位 + 后处理"策略:
  add_frame(t) 时 action[t] = state[t] (临时占位)
  finish_episode() 前调用 _postprocess_episode_actions():
    action[t] = state[t+1], 最后一帧保持不变
"""
import copy
import logging
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ROBOT_STATE_NAMES 与 VR 录制 (dataset_recorder_node.py:122) 一致
ROBOT_STATE_NAMES = [
    "joint_1_rad", "joint_2_rad", "joint_3_rad", "joint_4_rad",
    "joint_5_rad", "joint_6_rad", "joint_7_rad",
    "gripper_open",
    "eef_pos_x_m", "eef_pos_y_m", "eef_pos_z_m",
    "eef_rot_euler_x_rad", "eef_rot_euler_y_rad", "eef_rot_euler_z_rad",
]


def build_dagger_features(
    camera_shapes: Dict[str, Tuple[int, int, int]],
    policy_action_dim: int,
) -> dict:
    """
    构建 DAgger 录制的 features dict。

    是 VR 录制 features 的超集:
    - observation.state (14,) + action (14,) + images = 与 VR 一致
    - policy_action (policy_action_dim,) + control_source (1,) = DAgger 附加

    参数:
        camera_shapes: {cam_name: (H, W, C)} 相机配置
        policy_action_dim: policy 输出的 action 维度
    """
    features = {
        # 主轨迹字段（与 VR 一致）
        "observation.state": {
            "dtype": "float32",
            "shape": (14,),
            "names": ROBOT_STATE_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (14,),
            "names": ROBOT_STATE_NAMES,
        },
        # DAgger 附加字段
        "policy_action": {
            "dtype": "float32",
            "shape": (policy_action_dim,),
        },
        "control_source": {
            "dtype": "int32",
            "shape": (1,),
        },
    }
    # 图像字段
    for cam_name, shape in camera_shapes.items():
        height, width, channels = shape
        features[f"observation.images.{cam_name}"] = {
            "dtype": "video",
            "shape": (channels, height, width),
            "names": ["channels", "height", "width"],
        }
    return features


class DAggerRecorder:
    def __init__(
        self,
        repo_id: str,
        fps: int,
        root: str,
        camera_shapes: Dict[str, Tuple[int, int, int]],
        policy_action_dim: int,
        use_videos: bool = True,
        image_writer_threads: int = 4,
        image_writer_processes: int = 0,
    ):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self._policy_action_dim = policy_action_dim
        self._episode_started = False
        self._current_task = ""
        self._frame_count = 0

        # 构建 features
        features = build_dagger_features(camera_shapes, policy_action_dim)

        # 创建或加载 LeRobotDataset
        # root + repo_id 拼接为完整路径，使同一 root 下可存多个 dataset
        from pathlib import Path
        dataset_path = Path(root) / repo_id if root else None

        # 如果目录已存在，加载已有数据集（支持追加 episode）
        # 如果目录不存在，创建新数据集
        if dataset_path is not None and dataset_path.exists():
            logger.info(f"加载已有数据集: {dataset_path}")
            try:
                self.dataset = LeRobotDataset(
                    repo_id=repo_id,
                    root=root,
                )
                logger.info(f"数据集已加载，当前 episodes: {self.dataset.num_episodes}")
            except Exception as e:
                logger.error(f"加载数据集失败: {e}，将创建新数据集")
                # 加载失败，删除损坏的目录并重新创建
                import shutil
                shutil.rmtree(dataset_path)
                self.dataset = LeRobotDataset.create(
                    repo_id=repo_id,
                    fps=fps,
                    root=dataset_path,
                    features=features,
                    use_videos=use_videos,
                    image_writer_threads=image_writer_threads,
                    image_writer_processes=image_writer_processes,
                )
        else:
            logger.info(f"创建新数据集: {dataset_path}")
            self.dataset = LeRobotDataset.create(
                repo_id=repo_id,
                fps=fps,
                root=dataset_path,
                features=features,
                use_videos=use_videos,
                image_writer_threads=image_writer_threads,
                image_writer_processes=image_writer_processes,
            )

    def start_episode(self, task: str = ""):
        """开始新 episode"""
        self._current_task = task
        self._episode_started = True
        self._frame_count = 0

    def add_frame(
        self,
        state_14d: np.ndarray,
        images: Dict[str, np.ndarray],
        policy_action: np.ndarray,
        control_source: int = 0,
    ):
        """
        添加一帧数据。

        与 VR 录制一致的占位策略:
        - action 暂时用 state[t] 占位
        - finish_episode() 前通过 _postprocess_episode_actions() 修正为 state[t+1]

        参数:
            state_14d: (14,) 当前状态 [7 joints_rad, 1 gripper_01, 6 eef_pose]
            images: {cam_name: np.ndarray (H, W, C)} RGB 图像
            policy_action: (N,) policy 原始输出
            control_source: 0=policy, 1=human
        """
        if not self._episode_started:
            return

        frame = {
            "observation.state": state_14d.astype(np.float32),
            "action": state_14d.astype(np.float32),  # 临时占位，后处理替换
            "policy_action": policy_action.astype(np.float32),
            "control_source": np.array([control_source], dtype=np.int32),
        }
        for cam_name, img in images.items():
            frame[f"observation.images.{cam_name}"] = img

        self.dataset.add_frame(frame, task=self._current_task)
        self._frame_count += 1

    def _postprocess_episode_actions(self):
        """
        后处理 episode buffer 中的 action 字段。
        与 VR 录制 (dataset_recorder_node.py:630-663) 完全一致:
          action[t] = state[t+1]
          action[T-1] = state[T-1] (最后一帧保持不变)
        """
        buffer = self.dataset.episode_buffer
        if buffer is None:
            return

        size = buffer.get("size", 0)
        if size <= 1:
            logger.warning("episode 帧数不足，跳过 action 后处理")
            return

        states = buffer.get("observation.state")
        actions = buffer.get("action")

        if states is None or actions is None:
            logger.error("episode buffer 中缺少 state 或 action 字段")
            return

        # action[t] = state[t+1]，最后一帧保持不变
        for t in range(size - 1):
            actions[t] = states[t + 1].copy()

        logger.info("action 后处理完成: %d 帧", size)

    def discard_episode(self):
        """丢弃当前 episode: 清空 episode buffer，不保存"""
        if not self._episode_started:
            return

        # 重置 episode buffer（LeRobotDataset 内部状态）
        if hasattr(self.dataset, 'episode_buffer') and self.dataset.episode_buffer is not None:
            self.dataset.episode_buffer = self.dataset.create_episode_buffer()

        logger.info("Episode 已丢弃: %d 帧被清空", self._frame_count)
        self._episode_started = False
        self._frame_count = 0

    def finish_episode(self):
        """结束 episode: 后处理 action，然后保存"""
        if not self._episode_started:
            return

        # 空 episode 保护
        if self._frame_count == 0:
            logger.warning("空 episode（没有 add_frame），跳过保存")
            self._episode_started = False
            return

        # 后处理: action[t] = state[t+1]
        self._postprocess_episode_actions()

        # 保存（异常时保留 episode 状态，允许重试）
        try:
            self.dataset.save_episode()
        except Exception as e:
            logger.error("save_episode() 失败: %s", e)
            logger.error("episode 状态已保留，可重试 finish_episode()")
            return

        logger.info(
            "Episode 保存完成: %d 帧, 共 %d episodes",
            self._frame_count, self.dataset.num_episodes,
        )

        self._episode_started = False
        self._frame_count = 0

    @property
    def num_episodes(self) -> int:
        return self.dataset.num_episodes

    def close(self):
        """清理资源，包括 LeRobotDataset 的 PNG 图像缓存目录"""
        if self._episode_started:
            self.finish_episode()
        self.dataset.stop_image_writer()
        self._cleanup_image_cache()

    def _cleanup_image_cache(self):
        """清理 LeRobotDataset 的 temp/ PNG 缓存目录。

        LeRobotDataset 录制时先将每帧 PNG 写入 ./temp/{repo_id}/ 下，
        save_episode() 成功后会编码为 MP4 并删除对应 episode 的 PNG。
        但进程异常退出时这些 PNG 不会被清理，会持续占用磁盘空间。
        """
        import shutil
        try:
            cache_root = getattr(self.dataset, 'image_cache_root', None)
            if cache_root is not None and cache_root.exists():
                shutil.rmtree(cache_root)
                logger.info("已清理图像缓存目录: %s", cache_root)
        except Exception as e:
            logger.warning("清理图像缓存目录失败: %s", e)
