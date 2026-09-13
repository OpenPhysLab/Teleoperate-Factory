#!/usr/bin/env python3
"""
verify_dagger_dataset_for_openpi.py — 验证 DAgger 录制的 LeRobot v2.1 数据集是否兼容 OpenPI 训练。

检查项:
1. observation.state: shape=(14,) float32
2. action: shape=(14,) float32, action[t]==state[t+1]（非末帧）
3. 图像 key: observation.images.cam0_rgb / cam1_rgb 存在，分辨率一致
4. DAgger 附加字段: policy_action, control_source 存在
5. Episode 统计: 帧数、policy/human 比例
6. norm_stats 重算提醒

用法:
    python dagger/scripts/verify_dagger_dataset_for_openpi.py --dataset_path /path/to/dataset

    # 跳过 action[t]==state[t+1] 校验（未经后处理的数据集）
    python dagger/scripts/verify_dagger_dataset_for_openpi.py --dataset_path /path/to/dataset --skip_action_state_check
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_dataset_info(dataset_path: Path) -> dict:
    """加载 meta/info.json。"""
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        print(f"[FAIL] meta/info.json 不存在: {info_path}")
        sys.exit(1)
    with open(info_path) as f:
        return json.load(f)


def load_episodes(dataset_path: Path) -> List[dict]:
    """加载 meta/episodes.jsonl。"""
    episodes_path = dataset_path / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        print(f"[FAIL] meta/episodes.jsonl 不存在: {episodes_path}")
        sys.exit(1)
    episodes = []
    with open(episodes_path) as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes


def load_parquet_data(dataset_path: Path, episode_idx: int) -> dict:
    """加载单个 episode 的 parquet 数据。"""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("[FAIL] pyarrow 未安装，无法读取 parquet 文件")
        sys.exit(1)

    # LeRobot v2.1 格式: data/chunk-000/episode_000000.parquet
    chunk_idx = episode_idx // 1000
    parquet_path = dataset_path / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{episode_idx:06d}.parquet"
    if not parquet_path.exists():
        return {}

    table = pq.read_table(parquet_path)
    return table.to_pydict()


def check_features(info: dict) -> Tuple[bool, dict]:
    """检查 features 定义。"""
    features = info.get("features", {})
    results = {}
    all_pass = True

    # 1. observation.state
    state_feat = features.get("observation.state", {})
    state_shape = tuple(state_feat.get("shape", []))
    state_dtype = state_feat.get("dtype", "")
    if state_shape == (14,) and state_dtype == "float32":
        results["observation.state"] = f"PASS shape={state_shape} dtype={state_dtype}"
    else:
        results["observation.state"] = f"FAIL shape={state_shape} dtype={state_dtype} (expected (14,) float32)"
        all_pass = False

    # 2. action
    action_feat = features.get("action", {})
    action_shape = tuple(action_feat.get("shape", []))
    action_dtype = action_feat.get("dtype", "")
    if action_shape == (14,) and action_dtype == "float32":
        results["action"] = f"PASS shape={action_shape} dtype={action_dtype}"
    else:
        results["action"] = f"FAIL shape={action_shape} dtype={action_dtype} (expected (14,) float32)"
        all_pass = False

    # 3. 图像 keys
    expected_cams = ["observation.images.cam0_rgb", "observation.images.cam1_rgb"]
    for cam_key in expected_cams:
        cam_feat = features.get(cam_key, {})
        if cam_feat:
            cam_shape = tuple(cam_feat.get("shape", []))
            results[cam_key] = f"PASS shape={cam_shape}"
            if len(cam_shape) == 3 and cam_shape != (480, 640, 3):
                results[cam_key] += f" WARNING: 非标准分辨率 (expected 480x640x3)"
        else:
            results[cam_key] = "FAIL 不存在"
            all_pass = False

    # 4. DAgger 附加字段
    policy_action_feat = features.get("policy_action", {})
    if policy_action_feat:
        pa_shape = tuple(policy_action_feat.get("shape", []))
        results["policy_action"] = f"PASS shape={pa_shape}"
    else:
        results["policy_action"] = "FAIL 不存在"
        all_pass = False

    control_source_feat = features.get("control_source", {})
    if control_source_feat:
        cs_shape = tuple(control_source_feat.get("shape", []))
        results["control_source"] = f"PASS shape={cs_shape}"
    else:
        results["control_source"] = "FAIL 不存在"
        all_pass = False

    return all_pass, results


def check_action_state_relationship(dataset_path: Path, episodes: List[dict],
                                     max_episodes: int = 5) -> Tuple[bool, List[str]]:
    """验证 action[t] == state[t+1]（非末帧）。"""
    messages = []
    all_pass = True
    n_check = min(len(episodes), max_episodes)

    for ep_idx in range(n_check):
        data = load_parquet_data(dataset_path, ep_idx)
        if not data:
            messages.append(f"  Episode {ep_idx}: 无法加载 parquet")
            all_pass = False
            continue

        states = data.get("observation.state", [])
        actions = data.get("action", [])

        if not states or not actions:
            messages.append(f"  Episode {ep_idx}: state 或 action 为空")
            all_pass = False
            continue

        n_frames = len(states)
        mismatches = 0
        max_diff = 0.0

        for t in range(n_frames - 1):
            state_t = np.array(states[t], dtype=np.float32)
            action_t = np.array(actions[t], dtype=np.float32)
            state_t1 = np.array(states[t + 1], dtype=np.float32)

            diff = np.max(np.abs(action_t - state_t1))
            if diff > 1e-5:
                mismatches += 1
                max_diff = max(max_diff, diff)

        if mismatches == 0:
            messages.append(f"  Episode {ep_idx}: PASS ({n_frames} frames)")
        else:
            messages.append(f"  Episode {ep_idx}: FAIL {mismatches}/{n_frames-1} mismatches, max_diff={max_diff:.6f}")
            all_pass = False

    return all_pass, messages


def check_image_consistency(dataset_path: Path, info: dict, episodes: List[dict],
                            max_episodes: int = 3) -> Tuple[bool, List[str]]:
    """检查图像分辨率一致性。"""
    messages = []
    all_pass = True
    features = info.get("features", {})

    cam_keys = [k for k in features if k.startswith("observation.images.")]
    if not cam_keys:
        messages.append("  WARNING: 无图像 key")
        return True, messages

    n_check = min(len(episodes), max_episodes)

    for cam_key in cam_keys:
        expected_shape = tuple(features[cam_key].get("shape", []))
        # 检查视频或图像文件是否存在
        cam_name = cam_key.replace("observation.images.", "")

        # 检查 videos/ 目录
        video_dir = dataset_path / "videos"
        if video_dir.exists():
            video_files = list(video_dir.rglob(f"*{cam_name}*"))
            if video_files:
                messages.append(f"  {cam_key}: 视频模式, expected_shape={expected_shape}, {len(video_files)} files")
                continue

        # 检查 images/ 目录
        image_dir = dataset_path / "images"
        if image_dir.exists():
            image_files = list((image_dir / cam_name).glob("*.png"))[:5]
            if image_files:
                from PIL import Image
                shapes = set()
                for img_path in image_files:
                    img = Image.open(img_path)
                    shapes.add((img.height, img.width, 3))
                if len(shapes) == 1:
                    actual_shape = shapes.pop()
                    if actual_shape == tuple(expected_shape):
                        messages.append(f"  {cam_key}: PASS shape={actual_shape}")
                    else:
                        messages.append(f"  {cam_key}: WARNING shape={actual_shape} != expected {expected_shape}")
                else:
                    messages.append(f"  {cam_key}: FAIL 分辨率不一致: {shapes}")
                    all_pass = False
                continue

        messages.append(f"  {cam_key}: expected_shape={expected_shape} (无法验证实际文件)")

    return all_pass, messages


def check_dagger_fields(dataset_path: Path, episodes: List[dict],
                        max_episodes: int = 5) -> Tuple[bool, List[str]]:
    """检查 DAgger 附加字段并统计 policy/human 比例。"""
    messages = []
    all_pass = True
    n_check = min(len(episodes), max_episodes)

    total_policy = 0
    total_human = 0

    for ep_idx in range(n_check):
        data = load_parquet_data(dataset_path, ep_idx)
        if not data:
            continue

        policy_actions = data.get("policy_action", [])
        control_sources = data.get("control_source", [])

        if not policy_actions:
            messages.append(f"  Episode {ep_idx}: FAIL policy_action 为空")
            all_pass = False
            continue

        if not control_sources:
            messages.append(f"  Episode {ep_idx}: FAIL control_source 为空")
            all_pass = False
            continue

        n_frames = len(control_sources)
        n_policy = sum(1 for cs in control_sources if np.array(cs).flatten()[0] == 0)
        n_human = n_frames - n_policy
        total_policy += n_policy
        total_human += n_human

        # 检查 POLICY 帧的 policy_action 非零
        policy_nonzero = 0
        for i, cs in enumerate(control_sources):
            if np.array(cs).flatten()[0] == 0:  # policy
                pa = np.array(policy_actions[i], dtype=np.float32)
                if np.any(pa != 0):
                    policy_nonzero += 1

        messages.append(
            f"  Episode {ep_idx}: {n_frames} frames, "
            f"policy={n_policy} ({100*n_policy/n_frames:.0f}%), "
            f"human={n_human} ({100*n_human/n_frames:.0f}%), "
            f"policy_action_nonzero={policy_nonzero}/{n_policy}"
        )

    if total_policy + total_human > 0:
        total = total_policy + total_human
        messages.append(
            f"  总计: policy={total_policy} ({100*total_policy/total:.0f}%), "
            f"human={total_human} ({100*total_human/total:.0f}%)"
        )

    return all_pass, messages


def episode_statistics(episodes: List[dict], dataset_path: Path) -> List[str]:
    """Episode 统计报告。"""
    messages = []
    n_episodes = len(episodes)
    messages.append(f"  Episode 数量: {n_episodes}")

    if n_episodes == 0:
        return messages

    frame_counts = []
    for ep_idx in range(n_episodes):
        data = load_parquet_data(dataset_path, ep_idx)
        if data:
            states = data.get("observation.state", [])
            frame_counts.append(len(states))
        else:
            frame_counts.append(0)

    if frame_counts:
        messages.append(
            f"  帧数: min={min(frame_counts)}, "
            f"mean={np.mean(frame_counts):.1f}, "
            f"max={max(frame_counts)}, "
            f"total={sum(frame_counts)}"
        )

    return messages


def main():
    parser = argparse.ArgumentParser(description="验证 DAgger 数据集兼容 OpenPI 训练")
    parser.add_argument("--dataset_path", type=str, required=True, help="LeRobot v2.1 数据集路径")
    parser.add_argument("--skip_action_state_check", action="store_true",
                        help="跳过 action[t]==state[t+1] 校验")
    parser.add_argument("--max_episodes", type=int, default=5,
                        help="抽样检查的最大 episode 数")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"[FAIL] 数据集路径不存在: {dataset_path}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"DAgger Dataset Verification for OpenPI")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_path}")

    # 加载元数据
    info = load_dataset_info(dataset_path)
    episodes = load_episodes(dataset_path)
    print(f"LeRobot version: {info.get('codebase_version', 'unknown')}")
    print(f"FPS: {info.get('fps', 'unknown')}")

    all_pass = True

    # 1. Features 检查
    print(f"\n--- Features 检查 ---")
    feat_pass, feat_results = check_features(info)
    for key, msg in feat_results.items():
        print(f"  {key}: {msg}")
    all_pass &= feat_pass

    # 2. action[t] == state[t+1] 检查
    if not args.skip_action_state_check:
        print(f"\n--- action[t]==state[t+1] 检查 (前 {args.max_episodes} episodes) ---")
        as_pass, as_msgs = check_action_state_relationship(
            dataset_path, episodes, args.max_episodes
        )
        for msg in as_msgs:
            print(msg)
        all_pass &= as_pass
    else:
        print(f"\n--- action[t]==state[t+1] 检查: SKIPPED ---")

    # 3. 图像一致性检查
    print(f"\n--- 图像一致性检查 ---")
    img_pass, img_msgs = check_image_consistency(
        dataset_path, info, episodes, min(args.max_episodes, 3)
    )
    for msg in img_msgs:
        print(msg)
    all_pass &= img_pass

    # 4. DAgger 附加字段检查
    print(f"\n--- DAgger 附加字段检查 (前 {args.max_episodes} episodes) ---")
    dag_pass, dag_msgs = check_dagger_fields(
        dataset_path, episodes, args.max_episodes
    )
    for msg in dag_msgs:
        print(msg)
    all_pass &= dag_pass

    # 5. Episode 统计
    print(f"\n--- Episode 统计 ---")
    ep_msgs = episode_statistics(episodes, dataset_path)
    for msg in ep_msgs:
        print(msg)

    # 6. 总结
    print(f"\n{'='*60}")
    if all_pass:
        print("[PASS] 所有检查通过，数据集兼容 OpenPI 训练")
    else:
        print("[FAIL] 部分检查未通过，请修复后重新验证")

    # 7. norm_stats 提醒
    print(f"\n[提醒] 训练前请重算 norm_stats:")
    print(f"  python compute_norm_stats.py --dataset_path={dataset_path}")
    print(f"{'='*60}\n")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
