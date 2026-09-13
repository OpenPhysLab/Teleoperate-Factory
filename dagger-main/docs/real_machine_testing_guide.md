# DAgger 系统真机测试文档

> 覆盖所有开发阶段的真机测试方法，按顺序执行。
> 最后更新：2026-02-11 (danger-policy-bug-ui-0211: 修复 POLICY 模式 5x 速度 bug)

---

## 目录

1. [环境准备](#1-环境准备)
2. [Phase 1: 异步推理框架](#2-phase-1-异步推理框架)
3. [Phase 2A: UDP 状态接收 + 录制器](#3-phase-2a-udp-状态接收--录制器)
4. [Phase 2B: 混合状态 + 录制集成](#4-phase-2b-混合状态--录制集成)
5. [Phase 3: DAgger Node (ROS2 + VR 接管)](#5-phase-3-dagger-node-ros2--vr-接管)
6. [fix-dagger-recording-stutter: 消息同步录制](#6-fix-dagger-recording-stutter-消息同步录制)
7. [DAgger 完整工作流程（UI + VR + Shadow Mode + 多 Episode）](#7-dagger-完整工作流程ui--vr--shadow-mode--多-episode)
8. [Web UI 控制面板测试用例](#8-web-ui-控制面板测试用例)
9. [通用故障排查](#9-通用故障排查)

---

## 1. 环境准备

### 1.1 硬件清单

| 设备 | 要求 | 备注 |
|------|------|------|
| Realman RM65-B | 已上电，IP `192.168.1.18:8080` | 网线直连或同一子网 |
| Intel RealSense x2 | 序列号 `230422272673`, `420122071413` | USB3 连接 |
| GPU | CUDA 可用 | Server 端推理 |
| VR 手柄 | Quest3 或兼容设备 | Phase 3+ 需要 |

### 1.2 软件环境

**终端初始化（每个新终端都需要执行）**:

```bash
# 1. 激活 conda 环境
conda activate robocoin

# 2. Source ROS2 Jazzy
source /opt/ros/jazzy/setup.bash

# 3. 构建 ROS2 工作空间（首次或代码变更后）
cd ~/Desktop/Workspace/lerobot_policy_deploy/lerobotv3/vr_teleop/spacemouse_control_arm/ros2_realman_ws
colcon build --symlink-install

# 4. Source 工作空间
source ~/Desktop/Workspace/lerobot_policy_deploy/lerobotv3/vr_teleop/spacemouse_control_arm/ros2_realman_ws/install/setup.bash

# 5. 回到项目根目录
cd ~/Desktop/Workspace/lerobot_policy_deploy/lerobotv3
```

**验证基础依赖**:

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import grpc; print(f'gRPC: {grpc.__version__}')"
ros2 topic list  # 应无报错（Phase 3+ 需要）
```

> **提示**: 后续所有"新终端"均需执行步骤 1、2、4。步骤 3 仅在 `realman_teleop` 包代码变更后需要重新执行。
> 可将步骤 1/2/4 写入 `~/.bashrc` 或创建 alias 简化操作。

### 1.3 工作目录

```bash
cd ~/Desktop/Workspace/lerobot_policy_deploy/lerobotv3
```

所有后续命令均在此目录下执行（除非特别说明）。

### 1.4 单元测试（每次真机测试前先跑）

```bash
PYTHONPATH=$(pwd) /home/ubuntu/Desktop/Workspace/miniconda3/envs/robocoin/bin/python \
    -m pytest dagger/tests/ -v --rootdir=dagger
```

**判定**: 全部 PASS 才继续真机测试。

---

## 2. Phase 1: 异步推理框架

> 验证 gRPC 通信、Ring Buffer、异步推理、K_inf 裁剪

### 测试 1.1: dry_run 通信测试

**目的**: 验证 Client ↔ Server gRPC 通信正常

**终端 1 — PolicyServer (debug 模式)**:

```bash
conda activate robocoin
cd ~/Desktop/Workspace/lerobot_policy_deploy/lerobotv3/RoboCOIN

python -m lerobot.extensions.unified_deploy.server.policy_server \
    --debug \
    --debug_dataset_path=/home/ding/project/ours_method/spacemouse_control_arm/data/local/realman_teleop_vr_0129_02 \
    --debug_episode=0 \
    --debug_action_space=joints \
    --debug_action_type=absolute \
    --port=50051
```

等待输出 `PolicyServer 已启动，监听 0.0.0.0:50051`。

**终端 2 — AsyncInferenceClient (dry_run)**:

```bash
conda activate robocoin
cd ~/Desktop/Workspace/lerobot_policy_deploy/lerobotv3

python dagger/async_inference_client.py \
    --config dagger/config/dagger_params.yaml \
    --dry-run \
    --f-exec 10 \
    --max-steps 200
```

**判定标准**:

- [ ] Server 无报错
- [ ] Client 输出 `[AsyncClient] 已从 Server 获取配置`
- [ ] Client `Infer > 0`
- [ ] Client `Exec + Hold > 0`

---

### 测试 1.2: 真机 10Hz 执行

**目的**: 验证机械臂跟随数据集轨迹运动

**终端 1**: 保持 PolicyServer 运行（同 1.1）

**终端 2**:

```bash
python dagger/async_inference_client.py \
    --config dagger/config/dagger_params.yaml \
    --f-exec 10 \
    --max-steps 300
```

> **注意**: 去掉了 `--dry-run`，机械臂会真实运动！确保周围无障碍物。

**判定标准**:

- [ ] 机械臂正常运动
- [ ] 无异常抖动或突变
- [ ] `Ctrl+C` 正常退出

---

### 测试 1.3: 真机 30Hz 执行

**目的**: 验证高频率下的平滑度

**终端 2**:

```bash
python dagger/async_inference_client.py \
    --config dagger/config/dagger_params.yaml \
    --f-exec 30 \
    --max-steps 600
```

**判定标准**:

- [ ] FPS 接近 30
- [ ] 运动比 10Hz 更平滑
- [ ] `K_inf` 值合理（< chunk 的一半）

---

### 测试 1.4: 真实 ACT Policy 推理

**目的**: 验证真实 policy 推理下异步框架的功能正确性

**终端 1 — PolicyServer (真实模型)**:

```bash
conda activate robocoin
cd ~/Desktop/Workspace/lerobot_policy_deploy/lerobotv3/RoboCOIN

python -m lerobot.extensions.unified_deploy.server.policy_server \
    --policy_type=act \
    --pretrained_path=<模型路径> \
    --device=cuda \
    --port=50051
```

**终端 2**:

```bash
python dagger/async_inference_client.py \
    --config dagger/config/dagger_params.yaml \
    --server 127.0.0.1:50051 \
    --f-exec 30 \
    --task "pick up the yellow banana and put it in the white plate" \
    --max-steps 1500
```

**判定标准**:

- [ ] Warmup 完成，参数自动推导正确
- [ ] FPS 稳定接近 30
- [ ] Hold 比例 < 20%
- [ ] 机械臂运动方向和幅度正确
- [ ] 夹爪动作正确

---

## 3. Phase 2A: UDP 状态接收 + 录制器

> 验证 UDP 状态推送、DAggerRecorder 基础功能

### 测试 2A.1: UDP 状态接收

**目的**: 验证 UDP 状态推送正常工作

> **注意**: `target_ip` 应为运行脚本的本机 IP（即接收 UDP 数据的机器），不是机械臂 IP。可通过 `ip addr show` 或 `hostname -I` 查看。示例中 `192.168.1.100` 需替换为你的实际 IP。

**单终端**:

```bash
conda activate robocoin
cd ~/Desktop/Workspace/lerobot_policy_deploy/lerobotv3

python -c "
from dagger.core.udp_state import UDPStateReceiver, enable_udp_push
from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e
import time

# 创建机械臂连接
arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
arm.rm_create_robot_arm('192.168.1.18', 8080)

# 启用 UDP 推送
# 启用 UDP 推送（arm_ip/arm_port 供 TCP fallback 使用）
enable_udp_push(arm, target_ip='192.168.1.100', target_port=8089, cycle=2,
                arm_ip='192.168.1.18', arm_port=8080)
time.sleep(0.5)

# 启动接收
receiver = UDPStateReceiver(port=8089)
receiver.start()
time.sleep(1.0)

# 读取状态
for i in range(10):
    state = receiver.read_state_14d()
    if state is not None:
        print(f'[{i}] joints={state[:7].round(3)}, gripper={state[7]:.3f}, eef={state[8:14].round(3)}')
    else:
        print(f'[{i}] No data')
    time.sleep(0.1)

print(f'recv_count={receiver.recv_count}, is_receiving={receiver.is_receiving()}')
receiver.stop()
"
```

**判定标准**:

- [ ] 10 次读取中至少 8 次有数据（非 None）
- [ ] joints 值合理（弧度范围）
- [ ] eef 值合理（米/弧度范围）
- [ ] `recv_count > 0`
- [ ] `is_receiving() = True`

---

### 测试 2A.2: RM+ 夹爪状态

**目的**: 验证 RM+ 夹爪状态是否可用

```bash
python -c "
from dagger.core.udp_state import UDPStateReceiver, enable_udp_push, enable_rm_plus
from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e
import time

enable_rm_plus('192.168.1.18', 8080, baud=115200)
time.sleep(0.5)

arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
arm.rm_create_robot_arm('192.168.1.18', 8080)
enable_udp_push(arm, target_ip='192.168.1.100', target_port=8089, cycle=2,
                arm_ip='192.168.1.18', arm_port=8080)
time.sleep(0.5)

receiver = UDPStateReceiver(port=8089)
receiver.start()
time.sleep(1.0)

state = receiver.read_state_14d()
if state is not None:
    print(f'gripper={state[7]:.3f}')
    print('NOTE: 如果 gripper=0.000 且 RM+ 报 offline，需要 TCP fallback (Phase 2B)')
else:
    print('No data')

receiver.stop()
"
```

**判定标准**:

- [ ] 脚本无报错
- [ ] 记录 gripper 值（可能为 0 如果 RM+ offline，这是已知问题）

---

## 4. Phase 2B: 混合状态 + 录制集成

> 验证 UDP+TCP 混合状态、录制集成到 AsyncInferenceClient

### 测试 2B.1: 录制功能（AsyncInferenceClient）

**目的**: 验证 AsyncInferenceClient 的录制功能

**终端 1**: 启动 PolicyServer（debug 或真实模型）

**终端 2**:

```bash
python dagger/async_inference_client.py \
    --config dagger/config/dagger_params.yaml \
    --f-exec 10 \
    --max-steps 100 \
    --enable-recording \
    --repo-id "local/test_recording_phase2b" \
    --dataset-root "/tmp/dagger_test/"
```

**判定标准**:

- [ ] 运行完成无报错
- [ ] `/tmp/dagger_test/local/test_recording_phase2b/` 目录已创建
- [ ] 包含 `data/` 和 `meta/` 子目录
- [ ] 帧数接近 100

**验证录制数据**:

```bash
python -c "
import json, os
root = '/tmp/dagger_test/local/test_recording_phase2b'
info = json.load(open(os.path.join(root, 'meta', 'info.json')))
print(f'total_frames: {info.get(\"total_frames\", \"N/A\")}')
print(f'fps: {info.get(\"fps\", \"N/A\")}')
print(f'features: {list(info.get(\"features\", {}).keys())[:10]}...')
"
```

---

## 5. Phase 3: DAgger Node (ROS2 + VR 接管)

> 验证 ROS2 DAgger 节点、VR 接管、模式切换

> **启动架构说明**（fix-launch-executable-registration）：
> - 硬件节点（driver, vr_input, camera x2）通过 `Node()` 启动（已注册到 `realman_teleop` 包）
> - `dagger_node` 和 `control_panel` 通过 `ExecuteProcess` 启动（独立 Python 脚本，未注册到 ROS2 包）
> - PolicyServer 通过 `ExecuteProcess` 启动（gRPC 服务，非 ROS2 节点）
> - 启动命令为 `ros2 launch dagger/launch/dagger.launch.py`（必须通过 ros2 launch，因为 launch_ros 依赖系统 Python 3.12）
> - `ExecuteProcess` 节点（dagger_node / control_panel / PolicyServer）使用系统 Python 启动（默认 `sys.executable`，也可在 `dagger_params.yaml` 的 `python_path` 字段手动指定绝对路径）
> - 参数覆盖通过修改 `dagger/config/dagger_params.yaml` 和 `teleop_params.yaml`，不支持命令行 `:=` 覆盖
> - dagger_node 启动后进入 IDLE 模式，Web UI 可实时监控 PolicyServer 状态（probe 检测），用户确认 Server 就绪后点击"开始推理"

> **danger-policy-bug-ui-0211 修复说明**（2026-02-11）：
> 修复了 POLICY 模式下机械臂运动速度异常（5x 快于预期）的 3 个 bug：
> 1. **执行频率错误**：`_tick_policy()` 原本在 `control_hz`（50Hz）定时器中执行，实际应以 `f_exec`（10Hz）执行。修复：改为独立 `_policy_exec_loop()` 线程，以 `f_exec` 频率运行。
> 2. **SDK 参数不一致**：`teleop_params.yaml` 中 `canfd_radio: 60` 改为 `0`（与 async_inference_client 一致）。
> 3. **Pose 格式误判**：driver 的 `_on_action_pose()` 原本仅靠 `len(position)` 判断格式，7 值（6D 欧拉+夹爪）被误判为四元数。修复：通过 `msg.name` 字段显式识别（`"rx"` → 欧拉，`"qw"` → 四元数）。
>
> **关键**：bug 3 的修复在 `realman_driver_node.py`（ROS2 包源码），必须 `colcon build` 才能同步到 `install/` 目录。如果跳过 build，installed 的旧代码仍会误判 pose 格式。

### 5.0 完整启动指令（重要，每次测试前必执行）

以下是从零开始启动 DAgger 系统的完整步骤。**每次测试前都应按顺序执行**，特别是 `colcon build` 步骤（确保 driver 源码与 installed 一致）。

#### 步骤 1: 构建 ROS2 工作空间

> **为什么必须 build？** `realman_driver_node` 等硬件节点通过 `Node()` 启动，ROS2 从 `install/` 目录加载代码（非源码目录）。如果源码有修改但未 build，运行的仍是旧代码。当前 build 未使用 `--symlink-install`，所以每次源码变更后都需要重新 build。

```bash
cd ~/Desktop/Workspace/lerobot_policy_deploy/lerobotv3/vr_teleop/spacemouse_control_arm/ros2_realman_ws
colcon build
```

验证 build 成功（source 和 installed 应无差异）：

```bash
diff src/realman_teleop/realman_teleop/realman_driver_node.py \
     install/realman_teleop/lib/python3.12/site-packages/realman_teleop/realman_driver_node.py
# 无输出 = 一致
```

#### 步骤 2: 初始化终端环境

```bash
# 激活 conda（如果需要 torch 等 ML 依赖）
conda activate robocoin

# Source ROS2 Jazzy
source /opt/ros/jazzy/setup.bash

# Source 工作空间（必须在 colcon build 之后）
source ~/Desktop/Workspace/lerobot_policy_deploy/lerobotv3/vr_teleop/spacemouse_control_arm/ros2_realman_ws/install/setup.bash

# 回到项目根目录
cd ~/Desktop/Workspace/lerobot_policy_deploy/lerobotv3
```

#### 步骤 3: 修改配置（按测试阶段）

编辑 `dagger/config/dagger_params.yaml`，按下方"配置说明"中的阶段 A/B/C 修改。

#### 步骤 4: 启动 DAgger 系统

```bash
ros2 launch dagger/launch/dagger.launch.py
```

这一条命令会自动启动所有节点：
- `realman_driver_node` — 机械臂驱动（从 installed ROS2 包加载）
- `vr_input_node` — VR 数据接收
- `camera_node` x2 — RealSense 相机
- `dagger_node` — DAgger 控制节点（ExecuteProcess，从源码直接运行）
- `dagger_control_panel` — Web UI 控制面板（ExecuteProcess）
- `policy_server` — PolicyServer gRPC 服务（`auto_launch: true` 时自动启动）

#### 步骤 5: 打开 Web UI

浏览器访问 `http://<机器人IP>:5002`，确认 Server 状态卡片显示绿色"在线"后，点击"开始推理"。

#### 快速参考（一键复制）

```bash
# === 完整启动流程（从终端打开到系统运行） ===

# 1. Build ROS2 包（确保 driver 源码同步）
cd ~/Desktop/Workspace/lerobot_policy_deploy/lerobotv3/vr_teleop/spacemouse_control_arm/ros2_realman_ws && colcon build

# 2. 初始化环境
conda activate robocoin
source /opt/ros/jazzy/setup.bash
source ~/Desktop/Workspace/lerobot_policy_deploy/lerobotv3/vr_teleop/spacemouse_control_arm/ros2_realman_ws/install/setup.bash
cd ~/Desktop/Workspace/lerobot_policy_deploy/lerobotv3

# 3. 启动（确保 dagger_params.yaml 已按需修改）
ros2 launch dagger/launch/dagger.launch.py
```

> **注意事项**：
> - `colcon build` 只需在 `realman_teleop` 包源码变更后执行。如果只修改了 `dagger_node.py`、`control_panel.py` 或 `dagger_params.yaml`，无需重新 build（这些通过 ExecuteProcess 直接从源码运行）。
> - `source install/setup.bash` 必须在 `colcon build` 之后执行，否则加载的是旧的 build 产物。
> - 如果遇到 `executable 'xxx' not found`，说明 build 失败或未 source 工作空间。
> - `teleop_params.yaml` 中 `canfd_radio` 必须为 `0`（已修复），如果被意外改回非零值会导致运动异常。

---

### 5.0.1 配置说明（重要，测试前必读）

Phase 3 分三个阶段递进测试，每个阶段需要修改 `dagger/config/dagger_params.yaml` 中的不同配置项。

#### 阶段 A: 纯 VR 遥操（无推理、无录制）

验证 ROS2 launch + VR 手柄控制机械臂的基础功能。

```yaml
# dagger/config/dagger_params.yaml 关键配置
inference:
  enable_policy: false          # 不启用策略推理

enable_recording: false          # 不启用录制

policy_server:
  auto_launch: false             # 不自动启动 PolicyServer
```

#### 阶段 B: Policy 推理 + 录制（无 VR 介入）

验证策略推理控制机械臂 + 自动录制数据。

```yaml
# dagger/config/dagger_params.yaml 关键配置
inference:
  enable_policy: true            # ← 启用策略推理

enable_recording: true           # ← 启用录制
repo_id: "local/dagger_realman_001"
dataset_root: "/home/ubuntu/Desktop/Workspace/lerobot_policy_deploy/data/dagger/"
recording_task: "pick and place"

policy_server:
  auto_launch: true              # ← launch 自动启动 PolicyServer
  policy_type: "act"
  pretrained_path: "/path/to/your/pretrained_model"   # ← 改为你的模型路径
  device: "cuda"
  port: 50051
```

> **模型路径示例**: `/home/ubuntu/Desktop/Workspace/lerobot_policy_deploy/lerobotv3/RoboCOIN/outputs/pretrained_model_delta_pose_0207_act-150episodes/pretrained_model`

#### 阶段 C: Policy + VR 介入（完整 DAgger）

配置与阶段 B 完全相同，区别在于测试时会用 VR trigger 介入。无需额外修改配置。

---

### 测试 3.0: 纯 VR 遥操（阶段 A）

**目的**: 验证 ROS2 launch 正常启动，VR 手柄能控制机械臂

**配置**: 按上方"阶段 A"修改 `dagger_params.yaml`

**启动**: 按 [5.0 完整启动指令](#50-完整启动指令重要每次测试前必执行) 执行（build → 环境初始化 → launch）

**操作步骤**:

1. 等待所有节点启动（日志无报错）
2. 打开 Web UI: `http://<机器人IP>:5002`
3. 点击"开始会话"（纯 VR 模式下按钮显示为"开始会话"）
4. 按下 VR trigger，用手柄移动机械臂
5. 松开 trigger，机械臂停止跟随
6. `Ctrl+C` 退出

**判定标准**:

- [ ] 6 个节点全部启动无报错（driver, vr_input, camera x2, dagger_node, control_panel）
- [ ] Web UI 可访问，显示"纯 VR 模式"
- [ ] 按下 trigger 后机械臂跟随 VR 手柄运动
- [ ] 松开 trigger 后机械臂停止
- [ ] `Ctrl+C` 退出时 RM+ 协议正常关闭（日志显示 `RM+ 生态协议已关闭`）

---

### 测试 3.1: Policy 推理 + 自动录制（阶段 B）

**目的**: 验证策略推理控制机械臂 + 自动录制数据

**配置**: 按上方"阶段 B"修改 `dagger_params.yaml`，确保：
- `inference.enable_policy: true`
- `enable_recording: true`
- `policy_server.auto_launch: true`
- `policy_server.pretrained_path` 指向你的模型路径

> **注意**: 如果 `auto_launch: true`，launch 会自动启动 PolicyServer，无需手动开终端。仅当 `auto_launch: false` 时才需要手动启动 PolicyServer。

**启动**: 按 [5.0 完整启动指令](#50-完整启动指令重要每次测试前必执行) 执行（build → 环境初始化 → launch）

**操作步骤**:

1. 等待所有节点启动 + PolicyServer warmup 完成
2. 打开 Web UI: `http://<机器人IP>:5002`
3. 确认 Server 状态卡片显示绿色"在线"
4. 点击"开始推理"
   - 系统自动：启用 driver follow → 连接 PolicyServer → 进入 POLICY 模式 → 开始录制
5. 观察机械臂执行策略动作 10-15 秒
6. 观察 Web UI：
   - 模式徽章：绿色 "POLICY"
   - 推理详情：Buffer 和推理次数递增
   - 录制详情：帧数持续递增
7. 点击"停止录制" → 弹出确认框 → 点击"保存"
   - 推理不中断，机械臂继续运动
8. （可选）点击"开始新 Episode" → 录制第 2 个 episode → 停止录制 → 保存
9. 点击"停止推理" → 回到 IDLE 模式
10. `Ctrl+C` 退出

**判定标准**:

- [ ] 所有节点启动无报错（driver, vr_input, camera x2, dagger_node, control_panel, PolicyServer）
- [ ] Server 状态卡片显示绿色"在线"
- [ ] 点击"开始推理"后模式变为 POLICY，机械臂开始运动
- [ ] 录制帧数持续递增
- [ ] "停止录制"后推理不中断（机械臂继续运动）
- [ ] "停止推理"后回到 IDLE，机械臂停止

**验证录制数据**:

```bash
# 检查数据目录
ls -la ~/Desktop/Workspace/lerobot_policy_deploy/data/dagger/local/dagger_realman_001/

# 验证数据完整性
python -c "
import json, os
root = os.path.expanduser('~/Desktop/Workspace/lerobot_policy_deploy/data/dagger/local/dagger_realman_001')
info = json.load(open(os.path.join(root, 'meta', 'info.json')))
print(f'total_frames: {info.get(\"total_frames\", \"N/A\")}')
print(f'fps: {info.get(\"fps\", \"N/A\")}')
print(f'features: {list(info.get(\"features\", {}).keys())[:10]}...')
print(f'episodes: {info.get(\"total_episodes\", \"N/A\")}')
"

# 检查 parquet 数据
python -c "
import pandas as pd, os, glob
root = os.path.expanduser('~/Desktop/Workspace/lerobot_policy_deploy/data/dagger/local/dagger_realman_001')
files = sorted(glob.glob(os.path.join(root, 'data', '**', '*.parquet'), recursive=True))
for f in files:
    df = pd.read_parquet(f)
    print(f'{os.path.basename(f)}: {len(df)} frames, columns={list(df.columns)[:8]}...')
"
```

---

### 测试 3.2: Policy + VR 介入示教 + 录制（阶段 C，完整 DAgger）

**目的**: 验证 VR trigger 介入时模式正确切换，shadow mode 正常工作，录制同时保存专家动作和策略动作

**配置**: 与阶段 B 相同（`enable_policy: true`, `enable_recording: true`），无需额外修改

**前提**: 测试 3.1 通过

**启动**: 同测试 3.1

**操作步骤**:

1. 打开 Web UI，确认 Server 在线
2. 点击"开始推理"（自动开始录制）
3. 等待 5 秒（POLICY 模式，策略控制机械臂）
4. **按下 VR trigger**（力度 > 0.85）
   - Web UI 模式徽章变为黄色 "HUMAN"
   - 机械臂切换为 VR 手柄控制
   - 后台 shadow mode 持续运行（推理线程不暂停）
5. 用 VR 手柄示教正确动作 5-10 秒
6. **松开 VR trigger**
   - Web UI 模式徽章变回绿色 "POLICY"
   - 策略从当前位置（专家停下的位置）继续推理
   - 无跳变或回退
7. 观察策略继续运行 5 秒
8. （可选）重复步骤 4-7 多次介入
9. 点击"停止录制" → 保存
10. 点击"停止推理" → 回到 IDLE
11. `Ctrl+C` 退出

**判定标准**:

- [ ] 按下 trigger 后日志显示 `Mode: POLICY -> HUMAN`
- [ ] HUMAN 模式下机械臂跟随 VR 手柄运动
- [ ] HUMAN 模式下日志显示 shadow tick 消费 buffer
- [ ] 松开 trigger 后日志显示 `Mode: HUMAN -> POLICY`
- [ ] 策略从专家停下的位置继续推理（无跳变）
- [ ] 录制全程帧数持续递增（HUMAN 模式下也在录制）
- [ ] 模式切换无异常抖动

**验证 DAgger 录制数据（control_source + policy_action）**:

```bash
python -c "
import pandas as pd, numpy as np, os, glob

root = os.path.expanduser('~/Desktop/Workspace/lerobot_policy_deploy/data/dagger/local/dagger_realman_001')
files = sorted(glob.glob(os.path.join(root, 'data', '**', '*.parquet'), recursive=True))

if not files:
    print('ERROR: 未找到 parquet 文件')
    exit()

df = pd.read_parquet(files[-1])  # 最新 episode
print(f'=== {os.path.basename(files[-1])} ===')
print(f'总帧数: {len(df)}')

# 1. 检查 control_source
if 'control_source' in df.columns:
    cs = df['control_source'].values
    policy_count = (cs == 0).sum()
    human_count = (cs == 1).sum()
    transitions = sum(1 for i in range(1, len(cs)) if cs[i] != cs[i-1])
    print(f'POLICY 帧 (control_source=0): {policy_count} ({policy_count/len(cs)*100:.1f}%)')
    print(f'HUMAN 帧 (control_source=1): {human_count} ({human_count/len(cs)*100:.1f}%)')
    print(f'模式切换次数: {transitions}')
    if policy_count > 0 and human_count > 0:
        print('PASS: 两种模式都有记录')
    else:
        print('FAIL: 缺少某种模式的帧')
else:
    print('FAIL: control_source 列不存在')

# 2. 检查 policy_action（shadow mode 验证）
pa_cols = [c for c in df.columns if 'policy_action' in c]
if pa_cols:
    print(f'policy_action 列数: {len(pa_cols)}')
    if 'control_source' in df.columns:
        human_mask = df['control_source'].values == 1
        if human_mask.any():
            pa_data = df[pa_cols].values[human_mask]
            nonzero_pct = (np.abs(pa_data).sum(axis=1) > 1e-6).mean() * 100
            print(f'HUMAN 帧中非零 policy_action 比例: {nonzero_pct:.1f}%')
            if nonzero_pct > 50:
                print('PASS: shadow mode 正常工作')
            else:
                print('WARN: shadow mode 可能未正常工作')
else:
    print('FAIL: policy_action 列不存在')
"
```

---

### 测试 3.3: 安全检查

**目的**: 验证关节安全限制生效

**操作**: 在 POLICY 模式下观察日志

**判定标准**:

- [ ] 无 `Safety check REJECTED` 日志（正常运行时不应触发）
- [ ] 如果手动制造大幅度跳变（如重启 Server），应看到安全拒绝日志

---

## 6. fix-dagger-recording-stutter: 消息同步录制

> 验证 ApproximateTimeSynchronizer 驱动的录制，消除帧重复和时间戳错位

### 测试 6.1: 同步录制基础功能

**目的**: 验证 sync 回调正常触发，录制帧无重复

**终端 1**: 启动 PolicyServer

**终端 2**: 启动 DAgger Launch

```bash
ros2 launch dagger/launch/dagger.launch.py
```

等待 dagger_node 进入 POLICY 模式后：

**终端 3 — 启动录制**:

```bash
ros2 service call /dagger/start_episode std_srvs/srv/Trigger
```

等待 10-15 秒，然后停止录制：

```bash
ros2 service call /dagger/stop_episode std_srvs/srv/Trigger
```

**判定标准**:

- [ ] `start_episode` 返回 `success: True`
- [ ] dagger_node 日志显示 `DAggerRecorder initialized (from sync)`
- [ ] dagger_node 日志显示录制帧数递增
- [ ] `stop_episode` 返回 `success: True`
- [ ] 日志显示 episode 保存成功

---

### 测试 6.2: 录制帧率验证

**目的**: 验证录制帧率接近 30Hz，无重复帧

**操作**: 完成测试 6.1 后，检查录制数据

```bash
python -c "
import json, os
import numpy as np

# 修改为实际录制路径
root = os.path.expanduser('~/Desktop/Workspace/lerobot_policy_deploy/data/dagger/local/dagger_realman_001')

# 读取 info
info_path = os.path.join(root, 'meta', 'info.json')
if os.path.exists(info_path):
    info = json.load(open(info_path))
    total = info.get('total_frames', 0)
    fps = info.get('fps', 0)
    print(f'total_frames: {total}')
    print(f'fps: {fps}')
    duration = total / fps if fps > 0 else 0
    print(f'duration: {duration:.1f}s')
    print(f'effective_fps: {total / duration:.1f}' if duration > 0 else 'N/A')
else:
    print(f'info.json not found at {info_path}')
    print('Available files:')
    for f in os.listdir(root) if os.path.isdir(root) else []:
        print(f'  {f}')
"
```

**判定标准**:

- [ ] `total_frames` 合理（10s 录制约 300 帧）
- [ ] `effective_fps` 接近 30（±5）
- [ ] 无明显帧丢失（total_frames / duration ≈ 30）

---

### 测试 6.3: 重复帧检测

**目的**: 验证连续帧的 state 不重复（核心改进点）

```bash
python -c "
import numpy as np
import os

root = os.path.expanduser('~/Desktop/Workspace/lerobot_policy_deploy/data/dagger/local/dagger_realman_001')

# 尝试读取 parquet 数据
try:
    import pandas as pd
    parquet_files = []
    data_dir = os.path.join(root, 'data')
    if os.path.isdir(data_dir):
        for f in sorted(os.listdir(data_dir)):
            if f.endswith('.parquet'):
                parquet_files.append(os.path.join(data_dir, f))

    if parquet_files:
        df = pd.read_parquet(parquet_files[0])
        print(f'Loaded {len(df)} frames from {parquet_files[0]}')

        # 检查 state 列（observation.state）
        state_cols = [c for c in df.columns if 'state' in c.lower() and 'observation' in c.lower()]
        if state_cols:
            print(f'State columns: {state_cols[:5]}')

        # 检查连续帧是否重复
        # 提取所有 state 相关列
        state_data = df[[c for c in df.columns if c.startswith('observation.state')]].values
        if len(state_data) > 1:
            diffs = np.abs(np.diff(state_data, axis=0))
            zero_diffs = np.all(diffs < 1e-8, axis=1)
            dup_count = np.sum(zero_diffs)
            dup_pct = dup_count / len(diffs) * 100
            print(f'Total frames: {len(state_data)}')
            print(f'Duplicate consecutive frames: {dup_count} ({dup_pct:.1f}%)')
            print(f'PASS: duplicate rate < 5%' if dup_pct < 5 else f'FAIL: duplicate rate {dup_pct:.1f}% >= 5%')
        else:
            print('Not enough frames to check duplicates')
    else:
        print(f'No parquet files found in {data_dir}')
except Exception as e:
    print(f'Error: {e}')
"
```

**判定标准**:

- [ ] 重复帧比例 < 5%（修复前约 40%，修复后应接近 0%）
- [ ] 这是本次 fix 的核心验证指标

---

### 测试 6.4: 控制循环不受影响

**目的**: 验证录制不影响 50Hz 控制循环

**操作**: 在测试 6.1 录制期间，观察 dagger_node 日志

**判定标准**:

- [ ] 控制循环 FPS 仍接近 50Hz（或 `control_hz` 配置值）
- [ ] 机械臂运动平滑，无因录制导致的卡顿
- [ ] 推理 observation 更新正常（从缓存读取，不受 sync 回调影响）

---

### 测试 6.5: VR 接管 + 录制

**目的**: 验证 VR 接管期间录制正确记录 control_source

**操作步骤**:

1. 启动 DAgger Launch + PolicyServer
2. 启动录制: `ros2 service call /dagger/start_episode std_srvs/srv/Trigger`
3. 等待 5 秒（POLICY 模式录制）
4. 按下 VR trigger（切换到 HUMAN 模式）
5. 用 VR 手柄操作 5 秒
6. 松开 VR trigger（切换回 POLICY 模式）
7. 等待 5 秒
8. 停止录制: `ros2 service call /dagger/stop_episode std_srvs/srv/Trigger`

**判定标准**:

- [ ] 录制成功完成
- [ ] 数据中包含 `control_source` 字段
- [ ] POLICY 段 `control_source=0`，HUMAN 段 `control_source=1`

**验证 control_source**:

```bash
python -c "
import pandas as pd
import os

root = os.path.expanduser('~/Desktop/Workspace/lerobot_policy_deploy/data/dagger/local/dagger_realman_001')
data_dir = os.path.join(root, 'data')
parquet_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.parquet')])

if parquet_files:
    df = pd.read_parquet(parquet_files[-1])  # 最新 episode
    if 'control_source' in df.columns:
        cs = df['control_source'].values
        policy_count = (cs == 0).sum()
        human_count = (cs == 1).sum()
        print(f'Total frames: {len(cs)}')
        print(f'POLICY frames (control_source=0): {policy_count}')
        print(f'HUMAN frames (control_source=1): {human_count}')
        print(f'Transitions: {sum(1 for i in range(1, len(cs)) if cs[i] != cs[i-1])}')
    else:
        print('control_source column not found')
        print(f'Available columns: {list(df.columns)}')
"
```

---

### 测试 6.6: 相机离线恢复

**目的**: 验证相机断开时 sync 回调停止，重连后恢复

**操作步骤**:

1. 启动 DAgger Launch + 录制
2. 拔掉一个相机 USB
3. 观察日志（sync 回调应停止触发）
4. 重新插入相机
5. 观察日志（sync 回调应恢复）

**判定标准**:

- [ ] 相机断开后录制帧数停止增长
- [ ] 无崩溃或异常
- [ ] 相机重连后录制恢复（可能需要重启 camera_node）

---

## 7. DAgger 完整工作流程（UI + VR + Shadow Mode + 多 Episode）

> 本节描述 DAgger 系统的完整真机使用流程，包括 Web UI 操作、VR 手柄键位、Shadow Mode 原理、多 Episode 录制。
> 这是真机测试的核心章节，建议在执行后续测试用例前通读。

### 7.0 设计目标

DAgger 系统的核心工作流程：

1. 操作员在 Web UI 上点击"开始推理"，系统自动进入 POLICY 模式并开始录制
2. 策略（policy）控制机械臂执行任务
3. 人类专家观察策略表现，随时可通过 VR 手柄 trigger 介入示教
4. 介入期间，策略推理在后台继续运行（shadow mode），录制同时保存专家动作和策略动作
5. 专家松开 trigger 后，策略从当前位置继续推理（无缝衔接）
6. 任务完成后，操作员在 Web UI 上停止录制，选择保存或丢弃
7. 可重复录制任意多个 episode，无需重启推理

---

### 7.1 Web UI 按钮功能说明

```
┌─────────────────────────────────────────────────────────────┐
│                    DAgger 控制面板                            │
│                  [IDLE] / [POLICY] / [HUMAN]                 │
├──────────────┬──────────────────────┬───────────────────────┤
│  左列         │  中列                │  右列                  │
│              │                      │                        │
│ Server 状态   │  RGB 相机预览         │  系统日志              │
│ ● 在线/离线   │  ┌────┐  ┌────┐     │  [INFO] ...            │
│ 策略: act     │  │cam0│  │cam1│     │  [WARN] ...            │
│ 模型: ...     │  └────┘  └────┘     │  [ERROR] ...           │
│              │                      │  [清空]                 │
│ 节点状态      │                      │                        │
│ 模式: IDLE    │                      │                        │
│ 帧数: 0       │                      │                        │
│              │                      │                        │
│ 推理控制      │                      │                        │
│ [开始推理]    │                      │                        │
│ [停止推理]    │                      │                        │
│              │                      │                        │
│ 录制控制      │                      │                        │
│ [停止录制]    │                      │                        │
│ [开始新Episode]│                     │                        │
│              │                      │                        │
│ [回到Home位姿] │                     │                        │
├──────────────┴──────────────────────┴───────────────────────┤
│                        状态栏                                │
└─────────────────────────────────────────────────────────────┘
```

#### Web UI 按钮一览

| 按钮 | 功能 | 何时可用 | 调用的后端服务 |
|------|------|---------|--------------|
| **开始推理** | 连接 PolicyServer → 进入 POLICY 模式 → 自动开始录制 | IDLE 模式且 Server 在线 | `/dagger/start_session` |
| **停止推理** | 如有活跃录制则先暂停并弹确认框，确认后停止推理回到 IDLE | POLICY 或 HUMAN 模式 | `/dagger/stop_session` |
| **停止录制** | 暂停当前 episode 录制，弹出保存/丢弃确认框（不停止推理） | 录制中（episode_active 且未暂停） | `/dagger/pause_episode` |
| **开始新 Episode** | 在当前推理会话中开始新一轮录制（不重启推理） | 推理运行中且无活跃录制 | `/dagger/new_episode` |
| **回到 Home 位姿** | 机械臂回到预设初始位姿 | 始终可用 | `/dagger/go_home` |
| **清空**（日志区） | 清空右侧系统日志 | 始终可用 | 前端操作 |

#### 确认对话框（弹窗）

点击"停止录制"或"停止推理"（有活跃录制时）后弹出：

| 按钮 | 功能 | 调用的后端服务 |
|------|------|--------------|
| **保存** | 保存当前 episode 数据到磁盘 | `/dagger/stop_episode` |
| **丢弃** | 丢弃当前 episode 数据 | `/dagger/discard_episode` |

> 如果是从"停止推理"触发的弹窗，保存/丢弃完成后会自动继续执行停止推理（`stop_session`）。

---

### 7.2 VR 手柄键位说明

| 操作 | 键位 | 效果 | 对应模式切换 |
|------|------|------|-------------|
| **介入示教** | 按下右手 trigger（力度 > 0.85） | 机械臂切换为 VR 手柄控制，策略推理在后台继续（shadow mode） | POLICY → HUMAN |
| **释放控制** | 松开右手 trigger（力度 < 0.85） | 机械臂切换回策略控制，从当前位置继续推理 | HUMAN → POLICY |

> VR 接管是纯物理操作，无需在 Web UI 上点击任何按钮。Web UI 会自动反映模式变化。

---

### 7.3 Shadow Mode 工作原理

Shadow Mode 是 DAgger 数据收集的核心机制：

```
POLICY 模式:
  推理线程 → buffer → 执行循环 → 机械臂（策略控制）
  录制: action=策略动作, policy_action=策略动作, control_source=0

VR 介入（HUMAN 模式）:
  推理线程 → buffer → shadow tick → _last_policy_action（仅记录，不控制）
  VR 手柄 → 执行循环 → 机械臂（专家控制）
  录制: action=专家动作, policy_action=策略动作(shadow), control_source=1

VR 释放（回到 POLICY 模式）:
  清空 buffer（丢弃 HUMAN 期间的过时 action）
  推理线程基于当前观测生成新 chunk → buffer → 执行循环 → 机械臂
```

**关键设计**：
- HUMAN 模式下推理线程**不暂停**，持续运行并产生 action
- `_tick_shadow()` 在每个执行周期消费 buffer 中的 policy action，更新 `_last_policy_action`
- 录制时每帧同时保存：专家动作（`action`）+ 策略动作（`policy_action`）+ 控制源标记（`control_source`）
- 回到 POLICY 时只清空 buffer，推理线程立即基于最新观测生成新 chunk，实现无缝衔接

---

### 7.4 录制数据字段说明

| 字段 | 维度 | 说明 |
|------|------|------|
| `observation.state` | 14D | 机械臂状态：7 关节 + 1 夹爪 + 6 EEF 位姿 |
| `observation.images.cam0_rgb` | H×W×3 | 相机 0 RGB 图像 |
| `observation.images.cam1_rgb` | H×W×3 | 相机 1 RGB 图像 |
| `action` | 8D | 实际执行的动作（POLICY 模式=策略动作，HUMAN 模式=专家动作） |
| `policy_action` | 8D | 策略推理的动作（POLICY 和 HUMAN 模式下都有，HUMAN 模式下来自 shadow tick） |
| `control_source` | 1D | 0=策略控制（POLICY），1=专家控制（HUMAN） |

> `policy_action` 在 HUMAN 模式下记录的是策略"本来会执行"的动作（送入 SDK 之前的绝对值 action），用于 DAgger 训练时对比专家和策略的差异。

---

### 7.5 完整操作流程（单 Episode）

```
步骤 1: 启动系统
  $ ros2 launch dagger/launch/dagger.launch.py
  等待所有节点启动，Web UI 可访问

步骤 2: 打开 Web UI
  浏览器访问 http://<机器人IP>:5002
  确认 Server 状态卡片显示绿色"在线"

步骤 3: 开始推理
  点击 [开始推理] 按钮
  → 系统连接 PolicyServer，进入 POLICY 模式
  → 自动开始录制（录制状态显示"录制中"）
  → 机械臂开始执行策略动作

步骤 4: 观察策略表现
  观察机械臂是否正确执行任务
  如果策略表现良好，跳到步骤 7

步骤 5: VR 介入示教（可选，可多次）
  按下 VR trigger → 进入 HUMAN 模式
  → Web UI 模式徽章变为黄色 "HUMAN"
  → 用 VR 手柄引导机械臂完成正确动作
  → 后台 shadow mode 持续记录策略动作

步骤 6: 释放 VR 控制
  松开 VR trigger → 回到 POLICY 模式
  → Web UI 模式徽章变为绿色 "POLICY"
  → 策略从当前位置继续推理（无缝衔接）
  → 可重复步骤 5-6 多次

步骤 7: 停止录制
  点击 [停止录制] 按钮
  → 弹出确认对话框
  → 点击 [保存] 保存数据，或 [丢弃] 丢弃数据
  → 推理仍在运行（机械臂继续执行策略）
```

### 7.6 完整操作流程（多 Episode）

```
步骤 1-7: 同上（完成第 1 个 episode 的录制和保存/丢弃）

步骤 8: 开始新 Episode
  点击 [开始新 Episode] 按钮
  → 开始录制新的 episode（推理不中断）
  → 录制状态显示"录制中 | Episode: 2"

步骤 9: 重复步骤 4-7
  观察、VR 介入、停止录制、保存/丢弃

步骤 10: 重复步骤 8-9（录制任意多个 episode）

步骤 11: 结束会话
  方式 A: 无活跃录制时，直接点击 [停止推理]
  方式 B: 有活跃录制时，点击 [停止推理]
    → 自动暂停录制并弹出确认框
    → 保存/丢弃后自动停止推理
  → 机械臂停止运动，回到 IDLE 模式

步骤 12: 回到 Home（可选）
  点击 [回到 Home 位姿] 按钮
  → 机械臂回到初始位姿
```

> **与 VR 遥操录制的对比**：工作流程与 VR 遥操 ROS2 UI 一致（开始录制 → 操作 → 停止录制 → 保存/丢弃 → 新 Episode），区别在于 DAgger 系统中"操作"阶段是策略+专家混合控制，且额外记录 `policy_action` 和 `control_source` 字段。

---

## 8. Web UI 控制面板测试用例

> 验证 Web UI 控制面板的启动、Server 状态监控、统一推理会话控制、录制确认流程、Shadow Mode、多 Episode 工作流
>
> **UI 重构说明**（dagger-ui-polishment + dagger-ui-fix-0211）：
> - 移除了独立的 IDLE/HUMAN/POLICY 模式切换按钮，改为"开始推理"/"停止推理"统一会话控制
> - 新增独立的"停止录制"和"开始新 Episode"按钮，支持多 Episode 工作流
> - 录制随 start_session 自动开始，停止录制不停止推理
> - 新增 Server 状态卡片（连接/就绪/策略类型/模型路径）
> - 去掉 depth 相机预留，RGB 预览区域更大
> - VR trigger 接管仍通过物理操作触发，Web UI 自动反映状态变化
> - HUMAN 模式下推理线程持续运行（shadow mode），录制同时保存策略动作和专家动作

### 8.0 Web UI 节点 Hz 含义说明（重要）

Web UI 左侧"节点状态"区域显示的 Hz 值是**各 ROS2 话题的消息发布频率**，由 `TopicMonitor` 统计 2 秒滑动窗口内的消息数量计算得出。

**这些 Hz 值不代表控制/执行频率**，它们反映的是数据流速率：

| 节点 | 监听话题 | Hz 含义 | 典型值 |
|------|---------|---------|--------|
| DAgger | `/dagger/status` | dagger_node 状态发布频率（`stats_timer` 每 0.5s 发一次） | ~2.0 Hz |
| Driver | `/rm/state_joint_state` | 机械臂状态发布频率（由 `teleop_params.yaml` 的 `state_hz` 决定） | ~100 Hz |
| VR Input | `/vr/right/pose` | VR 手柄位姿数据流频率 | ~72 Hz（Quest3） |
| Cam0/Cam1 | `/camera/<name>/color/image_raw` | 相机图像发布频率 | ~30 Hz |

**关键区分**：
- **Driver Hz (~100)** 是状态发布频率，不是动作执行频率。无论 POLICY 还是 HUMAN 模式，driver 都以 `state_hz` 发布状态。
- **DAgger Hz (~2)** 是状态心跳频率，不是控制频率。实际控制频率由以下决定：
  - POLICY 模式：`f_exec`（如 10 Hz），由 `_policy_exec_loop()` 独立线程控制
  - HUMAN 模式：`control_hz`（如 50 Hz），由 ROS2 定时器回调控制
- 模式切换（POLICY ↔ HUMAN）**不会改变** Web UI 中任何节点的 Hz 显示值。

> **常见误解**：看到 Driver Hz 显示 ~100 并不意味着机械臂在以 100Hz 执行动作。Driver 以 100Hz 发布状态，但接收动作的频率取决于 dagger_node 发送动作的频率（POLICY 模式 = `f_exec`，HUMAN 模式 = `control_hz`）。

### 前提

- `dagger_params.yaml` 中 `policy_server.auto_launch: true`（或手动启动 PolicyServer）
- `dagger_params.yaml` 中 `web_ui.enable: true`（默认）
- Web UI 端口默认 `5002`（由 `web_ui.port` 配置）

---

### 测试 8.1: Web UI 启动

**目的**: 验证 Web UI 控制面板节点正常启动

**操作**:

1. 启动 DAgger Launch:

```bash
conda activate robocoin
cd ~/Desktop/Workspace/lerobot_policy_deploy/lerobotv3

ros2 launch dagger/launch/dagger.launch.py
```

2. 观察终端日志

**判定标准**:

- [ ] 日志显示 `Web UI 已启动: http://0.0.0.0:5002`
- [ ] 浏览器访问 `http://<机器人IP>:5002` 能打开控制面板页面
- [ ] 页面标题显示 "DAgger 控制面板"
- [ ] 页面底部显示实际访问地址
- [ ] 页面布局为三列：左（Server 状态 + 节点 + 推理控制 + Home）、中（RGB 相机）、右（日志）

---

### 测试 8.2: Server 状态卡片

**目的**: 验证 Server 状态卡片正确反映 PolicyServer 连接状态

**操作**: 打开 Web UI 页面，观察左侧"Server 状态"区域

**判定标准**:

- [ ] PolicyServer 未启动时：红色圆点 + "离线"
- [ ] PolicyServer 启动但 warmup 未完成时：黄色圆点 + "连接中（Warmup）"
- [ ] PolicyServer 就绪后：绿色圆点 + "在线"
- [ ] 策略类型正确显示（如 "策略: act"）
- [ ] 模型路径正确显示（长路径自动截断，hover 显示完整路径）

---

### 测试 8.3: 节点状态监控

**目的**: 验证 Web UI 正确显示各节点在线状态和频率

**操作**: 观察左侧"节点状态"区域

**判定标准**:

- [ ] DAgger 节点显示"在线"，Hz > 0（来自 /dagger/status 话题）
- [ ] Driver 节点显示"在线"，Hz 接近 state_hz（来自 /rm/state_joint_state）
- [ ] VR Input 节点显示"在线"，Hz > 0（来自 /vr/right/pose）
- [ ] 相机节点显示"在线"，Hz 接近 30（来自 /camera/camX/color/image_raw）
- [ ] 拔掉 VR 手柄后，VR Input 节点变为"离线"

---

### 测试 8.4: 开始推理（自动录制）

**目的**: 验证"开始推理"按钮启动完整会话（POLICY 模式 + 自动开始录制）

**操作**:

1. 确保 Server 状态卡片显示"在线"（绿色）
2. 点击"开始推理"按钮

**判定标准**:

- [ ] 点击后底部状态栏显示"正在启动推理会话..."
- [ ] 顶部模式徽章变为绿色 "POLICY"
- [ ] 推理控制区状态变为"运行中"（绿色）
- [ ] 推理详情显示 Buffer 和推理次数递增
- [ ] 录制详情显示"录制中 | Episodes: X | 帧: Y"，帧数持续递增（录制自动开始）
- [ ] "开始推理"按钮变为禁用，"停止推理"按钮变为可用
- [ ] "停止录制"按钮变为可用，"开始新 Episode"按钮禁用（录制中）
- [ ] 机械臂开始跟随策略推理运动

---

### 测试 8.5: 停止录制（保存，不停止推理）

**目的**: 验证"停止录制"按钮仅暂停录制，不停止推理，推理继续运行

**操作**:

1. 确保处于推理运行状态（测试 8.4 完成后）
2. 等待 5-10 秒
3. 点击"停止录制"按钮
4. 在弹出的确认对话框中点击"保存"

**判定标准**:

- [ ] 点击后弹出确认对话框
- [ ] 点击"保存"后，对话框关闭，底部状态栏显示保存成功消息
- [ ] 顶部模式徽章**仍为**绿色 "POLICY"（推理未停止）
- [ ] 机械臂**继续**跟随策略推理运动
- [ ] "停止录制"按钮变为禁用
- [ ] "开始新 Episode"按钮变为可用（可以开始新一轮录制）
- [ ] 录制数据目录已创建（检查 `dataset_root` 路径）

---

### 测试 8.6: 停止录制（丢弃）

**目的**: 验证丢弃录制功能

**操作**:

1. 点击"开始推理"（自动开始录制）
2. 等待 3-5 秒
3. 点击"停止录制"
4. 在确认对话框中点击"丢弃"

**判定标准**:

- [ ] 点击"丢弃"后，对话框关闭
- [ ] 底部状态栏显示丢弃成功消息
- [ ] 丢弃的 episode 数据未保存
- [ ] 推理**继续运行**（模式仍为 POLICY）
- [ ] "开始新 Episode"按钮变为可用

---

### 测试 8.7: 多 Episode 工作流

**目的**: 验证在同一推理会话中录制多个 episode（和 VR 遥操 UI 逻辑一致）

**操作**:

1. 点击"开始推理"（自动开始 Episode 1 录制）
2. 等待 5 秒
3. 点击"停止录制" → 保存
4. 确认"开始新 Episode"按钮可用
5. 点击"开始新 Episode"（开始 Episode 2 录制）
6. 等待 5 秒
7. 点击"停止录制" → 保存
8. 点击"开始新 Episode"（开始 Episode 3 录制）
9. 等待 3 秒
10. 点击"停止录制" → 丢弃
11. 点击"停止推理"（无活跃录制，直接停止）

**判定标准**:

- [ ] 每次"开始新 Episode"后录制状态显示新的 Episode 编号
- [ ] 每次"停止录制"后推理不中断，机械臂继续运动
- [ ] 保存的 2 个 episode 数据完整（检查 `dataset_root` 路径下的 episode 目录）
- [ ] 丢弃的 Episode 3 数据未保存
- [ ] 最终"停止推理"时无弹窗（因为无活跃录制），直接回到 IDLE
- [ ] 全程推理未重启（推理次数持续递增，无 warmup 重新执行）

---

### 测试 8.8: VR 接管 + Shadow Mode

**目的**: 验证 VR 接管期间 shadow mode 正常工作，录制同时保存专家动作和策略动作

**操作**:

1. 点击"开始推理"（自动开始录制）
2. 等待 5 秒（POLICY 模式，策略控制）
3. 按下 VR trigger（进入 HUMAN 模式）
4. 用 VR 手柄操作 5 秒（专家示教）
5. 松开 VR trigger（回到 POLICY 模式）
6. 等待 5 秒（策略从当前位置继续）
7. 点击"停止录制" → 保存

**判定标准**:

- [ ] VR trigger 按下后，Web UI 模式徽章变为黄色 "HUMAN"
- [ ] HUMAN 模式下机械臂跟随 VR 手柄运动
- [ ] HUMAN 模式下日志显示 `Shadow tick: popped action from buffer`（shadow mode 消费 buffer）
- [ ] VR trigger 松开后，模式恢复绿色 "POLICY"
- [ ] 策略从专家停下的位置继续推理（无跳变或回退）
- [ ] 录制全程帧数持续递增（HUMAN 模式下也在录制）

**验证录制数据**:

```bash
python -c "
import pandas as pd, numpy as np, os

root = os.path.expanduser('~/Desktop/Workspace/lerobot_policy_deploy/data/dagger/local/dagger_realman_001')
data_dir = os.path.join(root, 'data')
parquet_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.parquet')])

if parquet_files:
    df = pd.read_parquet(parquet_files[-1])
    print(f'Total frames: {len(df)}')

    # 检查 control_source
    if 'control_source' in df.columns:
        cs = df['control_source'].values
        policy_count = (cs == 0).sum()
        human_count = (cs == 1).sum()
        transitions = sum(1 for i in range(1, len(cs)) if cs[i] != cs[i-1])
        print(f'POLICY frames (control_source=0): {policy_count}')
        print(f'HUMAN frames (control_source=1): {human_count}')
        print(f'Mode transitions: {transitions}')
        print(f'PASS: both modes present' if policy_count > 0 and human_count > 0 else 'FAIL: missing mode')
    else:
        print('FAIL: control_source column not found')

    # 检查 policy_action
    pa_cols = [c for c in df.columns if 'policy_action' in c]
    if pa_cols:
        print(f'policy_action columns: {len(pa_cols)}')
        # 检查 HUMAN 帧的 policy_action 是否非零（shadow mode 产生的）
        if 'control_source' in df.columns:
            human_mask = df['control_source'].values == 1
            if human_mask.any():
                pa_data = df[pa_cols].values[human_mask]
                nonzero_pct = (np.abs(pa_data).sum(axis=1) > 1e-6).mean() * 100
                print(f'HUMAN frames with non-zero policy_action: {nonzero_pct:.1f}%')
                print(f'PASS: shadow mode active' if nonzero_pct > 50 else 'WARN: shadow mode may not be working')
    else:
        print('FAIL: policy_action columns not found')
"
```

---

### 测试 8.9: VR 接管后策略无缝衔接

**目的**: 验证 VR 释放后策略从当前位置继续推理，无跳变

**操作**:

1. 点击"开始推理"
2. 等待策略运行 5 秒
3. 按下 VR trigger，用 VR 手柄将机械臂移动到一个明显不同的位置
4. 松开 VR trigger
5. 观察机械臂行为

**判定标准**:

- [ ] 松开 trigger 后机械臂**不会**跳回 VR 介入前的位置
- [ ] 策略从当前位置（专家停下的位置）继续推理
- [ ] 无明显抖动或撞击感（buffer 已清空，新 chunk 基于最新观测）
- [ ] 日志显示 `Transition: HUMAN -> POLICY (buffer cleared)`

---

### 测试 8.10: 停止推理（有活跃录制时）

**目的**: 验证"停止推理"在有活跃录制时先弹确认框，确认后再停止推理

**操作**:

1. 点击"开始推理"（自动开始录制）
2. 等待 5 秒
3. 点击"停止推理"（此时有活跃录制）
4. 在弹出的确认对话框中点击"保存"

**判定标准**:

- [ ] 点击"停止推理"后弹出确认对话框（不是直接停止）
- [ ] 点击"保存"后，先保存录制数据，然后自动停止推理
- [ ] 顶部模式徽章变为灰色 "IDLE"
- [ ] 机械臂停止运动
- [ ] 录制数据已保存

---

### 测试 8.11: 相机预览

**目的**: 验证 Web UI 实时显示 RGB 相机画面（无 depth）

**操作**: 观察中间"相机预览"区域

**判定标准**:

- [ ] 两个 RGB 相机画面正常显示（320x240 缩略图）
- [ ] 画面实时更新（约 30 FPS）
- [ ] 相机标签正确显示（cam0, cam1）
- [ ] 用手遮挡相机时画面实时变化

---

### 测试 8.12: 系统日志

**目的**: 验证 Web UI 实时显示系统日志

**操作**: 观察右侧"系统日志"区域

**判定标准**:

- [ ] 日志实时滚动更新
- [ ] 日志包含时间戳、节点名、消息内容
- [ ] INFO/WARN/ERROR 级别日志颜色区分（蓝/黄/红）
- [ ] 点击"清空"按钮后日志清除

---

### 测试 8.13: 推理控制区状态显示

**目的**: 验证推理控制区正确反映推理和录制状态

**操作**: 观察左侧"推理控制"区域在不同阶段的变化

**判定标准**:

- [ ] 启动前：状态"待机"，信息"点击'开始推理'启动会话"
- [ ] Warmup 完成后：状态"运行中"（绿色），显示 Buffer 和推理次数
- [ ] VR 接管时：状态"运行中（VR 接管）"
- [ ] 录制中：显示"录制中 | Episodes: X | 帧: Y"
- [ ] 录制暂停后：显示"无活跃录制"，"开始新 Episode"按钮可用

---

### 测试 8.14: 机械臂 Home

**目的**: 验证通过 Web UI 控制机械臂回到 Home 位姿

**操作**:

1. 确保处于待机状态（未运行推理）
2. 点击"回到 Home 位姿"按钮

**判定标准**:

- [ ] 按钮点击后显示"机械臂移动中，请稍候..."
- [ ] 机械臂移动到预设 Home 位置
- [ ] 完成后提示"已到达 home 位姿"
- [ ] 按钮在移动期间禁用，完成后恢复

---

### 测试 8.15: 完整 DAgger 工作流（端到端）

**目的**: 端到端验证完整 DAgger 工作流程：开始推理 → 策略运行 → VR 介入 → 释放 → 停止录制 → 新 Episode → 停止推理

**操作**:

1. 点击"回到 Home 位姿"（确保初始位置一致）
2. 点击"开始推理"（自动开始 Episode 1 录制）
3. 观察策略运行 5 秒
4. 按下 VR trigger，用 VR 手柄示教 5 秒
5. 松开 VR trigger，观察策略继续 5 秒
6. 再次按下 VR trigger 示教 3 秒，松开
7. 点击"停止录制" → 保存
8. 点击"回到 Home 位姿"
9. 点击"开始新 Episode"（开始 Episode 2 录制）
10. 观察策略运行 10 秒（不介入）
11. 点击"停止推理" → 保存（自动停止推理）

**判定标准**:

- [ ] Episode 1 包含多次 POLICY↔HUMAN 切换，control_source 有 0 和 1
- [ ] Episode 1 的 HUMAN 帧有非零 policy_action（shadow mode）
- [ ] Episode 2 全部为 POLICY 帧（control_source 全为 0）
- [ ] 两个 episode 数据完整保存
- [ ] 全程无崩溃、无异常抖动
- [ ] VR 释放后策略无缝衔接（无跳变）

---

### 测试 8.16: Server 离线时的错误处理

**目的**: 验证 PolicyServer 未就绪时点击"开始推理"的错误提示

**操作**:

1. 不启动 PolicyServer（或手动关闭）
2. 确认 Server 状态卡片显示"离线"（红色）
3. 点击"开始推理"

**判定标准**:

- [ ] 底部状态栏显示错误消息（如"PolicyServer 未连接"）
- [ ] "开始推理"按钮恢复可用（未卡在禁用状态）
- [ ] 模式保持 IDLE，不会进入 POLICY

---

## 9. 通用故障排查

| 现象 | 可能原因 | 解决方案 |
|------|---------|---------|
| `No module named 'rclpy'` | 未 source ROS2 环境 | `source /opt/ros/jazzy/setup.bash` |
| `No module named 'dagger'` | pickle 序列化路径错误 | 确认 data_types 从 `lerobot.extensions...` 导入 |
| `Infer=0` 始终为 0 | gRPC 通信失败 | 检查 Server 是否启动，端口是否正确 |
| FPS 远低于 f_exec | 观测采集太慢 | 检查相机连接，降低 f_exec |
| 机械臂抖动 | 动作不连续 | 启用 hold_on_empty，降低 f_exec |
| UDP `recv_count=0` | UDP 推送未启用 | 确认 `enable_udp_push()` 已调用 |
| RM+ gripper offline | RM+ 硬件问题 | 已知问题，使用 TCP fallback |
| sync 回调不触发 | 相机或 state topic 未发布 | `ros2 topic hz /rm/state_joint_state` 检查 |
| 录制帧数为 0 | recorder 未初始化 | 检查日志中是否有 `DAggerRecorder initialized` |
| `terminate called` 退出时 | Realman SDK C 库线程清理 | 不影响功能，可忽略 |
| Web UI 页面无法访问 | 端口被占用或防火墙 | `ss -tlnp \| grep 5002` 检查端口，`sudo ufw allow 5002` 开放端口 |
| Web UI 相机画面黑屏 | 相机节点未启动或话题名不匹配 | `ros2 topic list \| grep camera` 检查话题 |
| Web UI "开始推理"无响应 | dagger_node Service 未注册 | `ros2 service list \| grep dagger` 检查 `/dagger/start_session` 是否存在 |
| Web UI "开始推理"报错 | PolicyServer 未就绪 | 检查 Server 状态卡片是否显示"在线"（绿色） |
| Web UI 日志不更新 | /rosout 订阅 QoS 不匹配 | 检查 dagger_control_panel 节点是否正常运行 |
| Server 状态卡片始终"离线" | PolicyServer 未启动或 gRPC 端口不通 | 检查 `policy_server.auto_launch` 配置，或手动启动 PolicyServer |
| `executable 'dagger_node' not found` | 使用了 `ros2 run` 而非 `ros2 launch` | 改用 `ros2 launch dagger/launch/dagger.launch.py` |
| `No module named 'vr_utils'` | PYTHONPATH 未包含 vr_utils 路径 | launch 文件已自动设置，确认通过 `ros2 launch dagger/launch/dagger.launch.py` 启动 |
| `No module named 'lark'` | 缺少 lark 依赖（ros2 launch 需要） | `pip install lark` |

### 紧急停止

| 方式 | 操作 |
|------|------|
| 优雅退出 | `Ctrl+C` |
| ROS2 节点停止 | `ros2 lifecycle set /dagger_node shutdown` |
| 急停 | 按机械臂急停按钮 |

---

## 附录: 测试结果记录模板

```
日期: ____
测试人: ____
代码版本/commit: ____

Phase 1:
  1.1 dry_run:     [ ] PASS / [ ] FAIL  备注: ____
  1.2 真机 10Hz:   [ ] PASS / [ ] FAIL  备注: ____
  1.3 真机 30Hz:   [ ] PASS / [ ] FAIL  备注: ____
  1.4 ACT Policy:  [ ] PASS / [ ] FAIL  备注: ____

Phase 2A:
  2A.1 UDP 状态:   [ ] PASS / [ ] FAIL  备注: ____
  2A.2 RM+ 夹爪:   [ ] PASS / [ ] FAIL  备注: ____

Phase 2B:
  2B.1 录制功能:   [ ] PASS / [ ] FAIL  备注: ____

Phase 3:
  3.0 纯VR遥操(阶段A):       [ ] PASS / [ ] FAIL  备注: ____
  3.1 Policy推理+录制(阶段B): [ ] PASS / [ ] FAIL  备注: ____ episodes_saved=____
  3.2 Policy+VR介入(阶段C):   [ ] PASS / [ ] FAIL  备注: ____ policy_frames=____ human_frames=____ shadow_active=____
  3.3 安全检查:               [ ] PASS / [ ] FAIL  备注: ____

fix-recording-stutter:
  6.1 同步录制:    [ ] PASS / [ ] FAIL  备注: ____
  6.2 帧率验证:    [ ] PASS / [ ] FAIL  effective_fps=____
  6.3 重复帧检测:  [ ] PASS / [ ] FAIL  duplicate_rate=____%
  6.4 控制循环:    [ ] PASS / [ ] FAIL  备注: ____
  6.5 VR+录制:     [ ] PASS / [ ] FAIL  备注: ____
  6.6 相机离线:    [ ] PASS / [ ] FAIL  备注: ____

Web UI + DAgger 工作流 (Section 8):
  8.1  启动:            [ ] PASS / [ ] FAIL  端口=____
  8.2  Server状态:      [ ] PASS / [ ] FAIL  备注: ____
  8.3  节点监控:        [ ] PASS / [ ] FAIL  备注: ____
  8.4  开始推理(自动录制): [ ] PASS / [ ] FAIL  备注: ____
  8.5  停止录制(保存):  [ ] PASS / [ ] FAIL  备注: ____
  8.6  停止录制(丢弃):  [ ] PASS / [ ] FAIL  备注: ____
  8.7  多Episode:       [ ] PASS / [ ] FAIL  episodes_saved=____
  8.8  VR+Shadow Mode:  [ ] PASS / [ ] FAIL  shadow_active=____
  8.9  策略无缝衔接:    [ ] PASS / [ ] FAIL  备注: ____
  8.10 停止推理(有录制): [ ] PASS / [ ] FAIL  备注: ____
  8.11 相机预览:        [ ] PASS / [ ] FAIL  备注: ____
  8.12 系统日志:        [ ] PASS / [ ] FAIL  备注: ____
  8.13 推理控制区:      [ ] PASS / [ ] FAIL  备注: ____
  8.14 Home:            [ ] PASS / [ ] FAIL  备注: ____
  8.15 端到端工作流:    [ ] PASS / [ ] FAIL  备注: ____
  8.16 Server离线:      [ ] PASS / [ ] FAIL  备注: ____
```
