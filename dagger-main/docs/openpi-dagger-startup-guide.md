# OpenPI + DAgger 启动指南

## 系统架构

OpenPI-DAgger 是一个 3 进程系统，通过 2 个终端启动：

```
┌─────────────────────────────────────────────────────────────────────┐
│ Terminal 1: OpenPI WebSocket Server                                 │
│ (conda: openpi, port 8000)                                         │
│ serve_realman_policy.py → 加载 Pi0/Pi0.5 模型 → WebSocket 服务     │
│ ⚠️ 模型加载需要 ~2 分钟，必须最先启动                                │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ WebSocket (localhost:8000)
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Terminal 2: ros2 launch dagger.launch.py                            │
│ (系统 Python 3.12, 无需 conda)                                      │
│                                                                     │
│  ┌─ PolicyServer gRPC Bridge (auto_launch, port 50051) ──────────┐ │
│  │  policy_server.py --policy_type=openpi                         │ │
│  │  --openpi_remote_host=localhost:8000                            │ │
│  │  接收 gRPC 请求 → 转发到 OpenPI WebSocket → 返回 ActionChunk   │ │
│  └────────────────────────────┬───────────────────────────────────┘ │
│                               │ gRPC (localhost:50051)              │
│                               ▼                                     │
│  ┌─ DAgger Node ─────────────────────────────────────────────────┐ │
│  │  IDLE → start_session → POLICY (推理) / HUMAN (VR 介入)       │ │
│  │  InferenceBridge → gRPC 推理 → RingBuffer → 执行              │ │
│  │  录制: state + images + policy_action + control_source         │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  ┌─ ROS2 Nodes ─────────────────────────────────────────────────┐  │
│  │  realman_driver_node (机械臂驱动)                              │  │
│  │  vr_input_node (VR 手柄)                                      │  │
│  │  camera_node x2 (RealSense 相机)                              │  │
│  │  dagger_control_panel (Web UI, port 5002)                     │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

数据流：
```
VR 手柄 → /vr/right/* → dagger_node (HUMAN 模式: 直接遥操)
相机 → /camera/cam*/image_raw → dagger_node → InferenceBridge → gRPC → PolicyServer → OpenPI
机械臂状态 → /rm/state_joint_state → dagger_node
dagger_node → /rm/action_joint_state (POLICY) 或 /rm/action_pose (HUMAN) → driver_node → 机械臂
```

---

## 前置条件

### 环境要求

| 组件 | 环境 | 说明 |
|------|------|------|
| OpenPI Server | conda `openpi` + openpi venv | JAX + OpenPI 框架 |
| PolicyServer + DAgger | 系统 Python 3.12 | rclpy + torch + grpc |
| ROS2 节点 | 系统 Python 3.12 | realman_teleop 包 |

### 硬件要求
- Realman 机械臂（IP: 192.168.1.18）
- VR 手柄（Quest 3 + ALVR）
- 2x Intel RealSense 相机（cam0: 腕部, cam1: 基座）
- NVIDIA GPU（CUDA，用于 OpenPI 推理）

### 文件路径约定
```
项目根目录: /home/ubuntu/Desktop/Workspace/lerobot_policy_deploy/lerobotv3
├── openpi/                          # OpenPI 框架（含 .venv）
├── realman_openpi_training/         # Realman OpenPI 配置和训练脚本
│   ├── serve_realman_policy.py      # OpenPI WebSocket 服务启动脚本
│   └── realman_config.py            # Realman 数据集配置
├── RoboCOIN/                        # RoboCOIN 统一部署框架
│   └── src/lerobot/extensions/unified_deploy/
│       └── server/policy_server.py  # gRPC PolicyServer
├── dagger/                          # DAgger 系统
│   ├── dagger_node.py               # DAgger ROS2 主节点
│   ├── core/inference_bridge.py     # gRPC 推理桥接
│   ├── launch/dagger.launch.py      # ROS2 launch 文件
│   ├── config/dagger_params.yaml    # 运行时配置
│   └── scripts/verify_dagger_dataset_for_openpi.py  # 数据集验证
└── vr_teleop/                       # VR 遥操作 ROS2 包
```

---

## 第一步：配置 dagger_params.yaml

启动前必须修改 `dagger/config/dagger_params.yaml`：

```yaml
# ===== PolicyServer 配置 =====
server_address: "127.0.0.1:50051"
task: "pick up the yellow banana and put it in the white plate"

# ===== PolicyServer 自动启动 =====
policy_server:
  auto_launch: true
  policy_type: "openpi"                    # ← 改为 openpi
  pretrained_path: "/path/to/checkpoint/30000"  # ← 改为你的 checkpoint 路径
  device: "cuda"
  port: 50051
  extra_args:                              # ← OpenPI 远程模式参数
    - "--openpi_remote_host=localhost:8000"
    - "--openpi_config_name=realman_pi0"   # 或 realman_pi05

# ===== 推理配置（OpenPI 推荐值） =====
f_exec: 20.0           # 执行频率 20Hz
n_action_steps: 0      # ⚠️ 必须为 0！由 InferenceBridge 自适应裁剪
T_inter: 0.0           # 最大吞吐
t_inf: 0.0             # 0=warmup 时自动测量

# ===== 录制配置 =====
enable_recording: true
repo_id: "local/dagger_openpi_0215"
dataset_root: "/home/ubuntu/Desktop/Workspace/lerobot_policy_deploy/lerobotv3/dagger/data"
recording_task: "pick up the yellow banana and put it in the white plate"
```

> **关键参数说明**：
> - `n_action_steps: 0` — OpenPI 模型输出 action_horizon=50 的长 chunk。设为 0 表示不在 PolicyServer 端截取，由 InferenceBridge 的 K_inf 自适应裁剪处理。如果同时在两端截取会导致"双重截取"，有效动作数大幅减少。
> - `t_inf: 0.0` — 设为 0 让 warmup 阶段自动测量实际推理延迟（Pi0 通常 ~0.3s，Pi0.5 ~0.8s）。
> - `pretrained_path` — 这里填的是 RoboCOIN PolicyServer 的 checkpoint 路径（用于 gRPC 桥接的元数据），OpenPI 模型的实际 checkpoint 在 Terminal 1 的 `--checkpoint` 参数中指定。

---

## 第二步：启动 Terminal 1 — OpenPI WebSocket Server

**⚠️ 必须最先启动！模型加载需要 ~2 分钟。**

```bash
# 激活 OpenPI 环境
cd /home/ubuntu/Desktop/Workspace/lerobot_policy_deploy/lerobotv3/openpi
source .venv/bin/activate

# 启动 OpenPI 推理服务
python ../realman_openpi_training/serve_realman_policy.py \
    --config=pi0_realman_inference \
    --checkpoint=/path/to/your/checkpoint/30000 \
    --prompt="pick up the yellow banana and put it in the white plate" \
    --port=8000
```

**等待看到以下日志后再启动 Terminal 2：**
```
INFO:root:OpenPI 推理服务已启动，监听端口 8000
INFO:root:RoboCOIN 可通过 --openpi_remote_host=localhost:8000 连接
```

### 参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `--config` | OpenPI 配置名 | `pi0_realman_inference` 或 `pi05_realman_inference` |
| `--checkpoint` | checkpoint 目录（含 `model.safetensors` + `assets/`） | `/path/to/checkpoint/30000` |
| `--prompt` | 默认任务指令（可被 dagger_node 的 `task` 参数覆盖） | `"pick up the yellow banana"` |
| `--port` | WebSocket 端口（与 `--openpi_remote_host` 一致） | `8000` |
| `--record` | 调试用：记录每次推理的输入输出 | 不加此参数 |

### checkpoint 目录结构
```
checkpoint/30000/
├── model.safetensors          # 模型权重
├── metadata.pt                # 训练元数据
└── assets/
    └── <asset_id>/
        └── norm_stats.json    # 归一化统计（自动检测）
```

---

## 第三步：启动 Terminal 2 — ROS2 Launch

确认 Terminal 1 的 OpenPI 服务已就绪后：

```bash
# 系统 Python 3.12 已有 rclpy + torch，无需 conda activate
cd /home/ubuntu/Desktop/Workspace/lerobot_policy_deploy/lerobotv3

# 一键启动所有节点
ros2 launch dagger/launch/dagger.launch.py
```

launch 会按顺序启动：
1. `realman_driver_node` — 机械臂驱动（UDP 状态推送 + TCP 控制）
2. `vr_input_node` — VR 手柄数据接收
3. `camera_node_cam0` + `camera_node_cam1` — RealSense 相机
4. `policy_server` — gRPC 桥接（自动连接 Terminal 1 的 OpenPI WebSocket）
5. `dagger_node` — DAgger 主控节点（启动后进入 IDLE 模式）
6. `dagger_control_panel` — Web UI（http://0.0.0.0:5002）

**启动后 dagger_node 处于 IDLE 模式**，后台每 2 秒探测 PolicyServer 就绪状态。

---

## 第四步：开始推理会话

### 方式 A：通过 Web UI（推荐）

打开浏览器访问 `http://localhost:5002`，点击"开始会话"按钮。

### 方式 B：通过 ROS2 命令行

```bash
# 开始会话（连接 PolicyServer + 启动推理 + 开始录制）
ros2 service call /dagger/start_session std_srvs/srv/Trigger
```

**会话启动流程：**
1. 启用 driver follow（机械臂进入跟随模式）
2. 后台线程连接 PolicyServer（gRPC，最多等待 180 秒）
3. PolicyServer 连接 OpenPI WebSocket（如果尚未连接）
4. Warmup：运行 5 次推理测量 t_inf，自动推导 K_inf / T_inter / L_valid
5. 启动推理循环 + 执行循环
6. 切换到 POLICY 模式，机械臂开始执行策略动作
7. 录制自动开始（lazy-init DAggerRecorder）

**Warmup 日志示例：**
```
[InferenceBridge] Warmup 1/5: t_inf=0.312s
[InferenceBridge] Warmup 2/5: t_inf=0.298s
...
[InferenceBridge] Warmup done: t_inf=0.305s, K_inf=7, L=50, L_valid=43, T_inter=1.36s
```

---

## 第五步：VR 专家介入（DAgger 核心）

在 POLICY 模式运行时，按下 VR 手柄 trigger 即可切换到 HUMAN 模式：

| 操作 | 效果 |
|------|------|
| 按下 trigger（>0.85） | POLICY → HUMAN：暂停推理，清空 buffer，VR 直接控制机械臂 |
| 松开 trigger（<0.85） | HUMAN → POLICY：恢复推理，重新填充 buffer，策略接管控制 |

**录制行为：**
- POLICY 模式：`control_source=0`，`policy_action` = 模型输出的动作
- HUMAN 模式：`control_source=1`，`policy_action` = 零填充（或 shadow tick 的策略动作）

切换是即时的，无需手动操作。

---

## 第六步：停止会话 & 保存数据

### 停止会话

```bash
# 停止会话（暂停录制，等待确认保存/丢弃）
ros2 service call /dagger/stop_session std_srvs/srv/Trigger
```

### 保存 Episode

```bash
# 保存当前 episode（执行 action[t]=state[t+1] 后处理 + 写入磁盘）
ros2 service call /dagger/stop_episode std_srvs/srv/Trigger
```

### 丢弃 Episode（如果数据质量不好）

```bash
# 丢弃当前 episode
ros2 service call /dagger/discard_episode std_srvs/srv/Trigger
```

### 多 Episode 工作流

```bash
# Episode 1
ros2 service call /dagger/start_session std_srvs/srv/Trigger
# ... 运行一段时间 ...
ros2 service call /dagger/stop_session std_srvs/srv/Trigger
ros2 service call /dagger/stop_episode std_srvs/srv/Trigger    # 保存

# Episode 2
ros2 service call /dagger/start_session std_srvs/srv/Trigger
# ... 运行一段时间 ...
ros2 service call /dagger/stop_session std_srvs/srv/Trigger
ros2 service call /dagger/stop_episode std_srvs/srv/Trigger    # 保存

# 完成后 Ctrl+C 关闭 Terminal 2
```

---

## 第七步：验证录制数据集

```bash
cd /home/ubuntu/Desktop/Workspace/lerobot_policy_deploy/lerobotv3

python dagger/scripts/verify_dagger_dataset_for_openpi.py \
    --dataset_path=dagger/data/local/dagger_openpi_0215 \
    --max_episodes=5
```

验证项：
- observation.state: shape=(14,), float32
- action: shape=(14,), float32
- action[t] == state[t+1]（非终端帧）
- 图像 key: cam0_rgb, cam1_rgb 存在
- DAgger 字段: policy_action, control_source 存在
- Episode 统计: 帧数、policy/human 比例

---

## 故障排查

### 连接问题

| 症状 | 原因 | 解决 |
|------|------|------|
| `start_session` 超时（180s） | OpenPI 服务未启动或未就绪 | 检查 Terminal 1 日志，确认 "监听端口 8000" 已打印 |
| `PolicyServer connect failed` | gRPC 桥接未启动 | 检查 `dagger_params.yaml` 的 `auto_launch: true` |
| `openpi_remote_host unreachable` | WebSocket 端口不匹配 | 确认 `--port` 和 `--openpi_remote_host` 端口一致 |

### 推理问题

| 症状 | 原因 | 解决 |
|------|------|------|
| Warmup 后 L_valid 很小或 <0 | 推理延迟太高 / chunk_size 太小 | 降低 f_exec 或检查 GPU 负载 |
| 机械臂抖动 | f_exec 太高或安全限制太松 | 降低 f_exec，检查 `max_joint_delta_deg` |
| 机械臂不动 | ring buffer 持续为空 | 检查推理线程日志，确认 gRPC 通信正常 |
| 双重截取警告 | n_action_steps 在两端都设置了 | 设 `n_action_steps: 0`（YAML）或去掉 PolicyServer 的 `--n_action_steps` |

### 录制问题

| 症状 | 原因 | 解决 |
|------|------|------|
| 帧数为 0 | warmup 未完成就停止了 | 等待 warmup 日志后再操作 |
| 图像缺失 | 相机节点未启动 | 检查 RealSense USB 连接 |
| gripper 值异常（始终 1000） | 读取了 Modbus 夹爪而非 RM+ | 确认 UDP RM+ 状态正常（参考 BUG-018/019） |

### 关键日志关键词

```bash
# 在 Terminal 2 中搜索这些关键词判断状态：
# 正常启动：
"OpenPI 推理服务已启动"          # Terminal 1: OpenPI 就绪
"PolicyServer 启动"              # Terminal 2: gRPC 桥接启动
"会话已开始（POLICY 模式）"       # dagger_node: 会话启动
"Warmup done"                    # InferenceBridge: warmup 完成
"Frame 1 recorded"               # 录制开始

# 异常：
"connect failed"                 # 连接失败
"degraded mode"                  # 推理不可行，降级运行
"safety reject"                  # 安全检查拒绝动作
"double truncation"              # 双重截取警告
```

---

## 附录：Pi0 vs Pi0.5 参数对比

| 参数 | Pi0 | Pi0.5 |
|------|-----|-------|
| `--config` | `pi0_realman_inference` | `pi05_realman_inference` |
| `--openpi_config_name` | `realman_pi0` | `realman_pi05` |
| 典型 t_inf | ~0.3s | ~0.8s |
| action_horizon | 50 | 50 |
| 典型 K_inf (f_exec=20) | 6 | 16 |
| 典型 L_valid | 44 | 34 |
| 推荐 f_exec | 20-30 Hz | 15-20 Hz |

---

## 快速启动清单

```
□ 1. 修改 dagger/config/dagger_params.yaml
     - policy_type: "openpi"
     - pretrained_path: 你的 checkpoint 路径
     - extra_args: openpi_remote_host + config_name
     - n_action_steps: 0

□ 2. Terminal 1: 启动 OpenPI WebSocket Server
     cd openpi && source .venv/bin/activate
     python ../realman_openpi_training/serve_realman_policy.py ...
     → 等待 "监听端口 8000"

□ 3. Terminal 2: 启动 ROS2 Launch
     ros2 launch dagger/launch/dagger.launch.py
     → 等待所有节点启动

□ 4. 开始会话
     Web UI (http://localhost:5002) 或
     ros2 service call /dagger/start_session ...
     → 等待 "Warmup done"

□ 5. 运行 DAgger
     策略自动执行，VR trigger 随时介入
     → 观察 Web UI 状态

□ 6. 停止 & 保存
     ros2 service call /dagger/stop_session ...
     ros2 service call /dagger/stop_episode ...

□ 7. 验证数据集
     python dagger/scripts/verify_dagger_dataset_for_openpi.py ...
```
