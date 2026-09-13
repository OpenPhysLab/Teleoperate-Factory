"""
通用工具函数（复制自 RoboCOIN/src/lerobot/extensions/unified_deploy/core/utils.py）

提供单位转换、图像处理等工具函数。
"""

import numpy as np
import cv2
from typing import Optional


def rad_to_deg(rad: np.ndarray) -> np.ndarray:
    """弧度转角度"""
    return np.degrees(rad)


def deg_to_rad(deg: np.ndarray) -> np.ndarray:
    """角度转弧度"""
    return np.radians(deg)


def gripper_normalized_to_sdk(normalized: float) -> int:
    """夹爪归一化值 [0,1] 转 SDK 位置 [0,1000]"""
    return int(max(0.0, min(1.0, float(normalized))) * 1000)


def binarize_gripper_sdk(
    gripper_sdk: int,
    threshold: int = 500,
    hysteresis: int = 50,
    last_state: Optional[int] = None
) -> int:
    """夹爪二值化处理（带滞后机制）"""
    upper_bound = threshold + hysteresis
    lower_bound = threshold - hysteresis

    if gripper_sdk > upper_bound:
        return 1000
    elif gripper_sdk < lower_bound:
        return 0
    else:
        if last_state is not None:
            return last_state
        else:
            return 1000 if gripper_sdk >= threshold else 0


def gripper_sdk_to_normalized(sdk_value: int) -> float:
    """夹爪 SDK 位置 [0,1000] 转归一化值 [0,1]"""
    return max(0.0, min(1.0, float(sdk_value) / 1000.0))


def rotate_image(img: np.ndarray, rotate_deg: int) -> np.ndarray:
    """旋转图像，支持 0, 90, 180, 270 度"""
    if rotate_deg == 0 or img is None:
        return img
    elif rotate_deg == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif rotate_deg == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif rotate_deg == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return img


def resize_image(img: np.ndarray, target_size: tuple) -> np.ndarray:
    """调整图像大小，target_size = (H, W)"""
    if img is None:
        return img
    return cv2.resize(img, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)


def image_to_tensor_format(img: np.ndarray) -> np.ndarray:
    """HWC uint8 [0,255] → CHW float32 [0,1]"""
    if img is None:
        return img
    img = img.transpose(2, 0, 1)
    img = img.astype(np.float32) / 255.0
    return img


def tensor_to_image_format(tensor: np.ndarray) -> np.ndarray:
    """CHW float32 [0,1] → HWC uint8 [0,255]"""
    if tensor is None:
        return tensor
    img = tensor.transpose(1, 2, 0)
    img = (img * 255).astype(np.uint8)
    return img


def compute_delta(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """计算增量 = target - current"""
    return target - current


def apply_delta(current: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """应用增量 = current + delta"""
    return current + delta


def cumsum_delta(deltas: np.ndarray, initial: np.ndarray) -> np.ndarray:
    """累加增量序列，得到绝对位置序列"""
    cumsum = np.cumsum(deltas, axis=0)
    return initial + cumsum


def clip_joints(joints: np.ndarray, joint_limits: Optional[np.ndarray] = None) -> np.ndarray:
    """裁剪关节角度到安全范围"""
    if joint_limits is None:
        return joints
    return np.clip(joints, joint_limits[:, 0], joint_limits[:, 1])
