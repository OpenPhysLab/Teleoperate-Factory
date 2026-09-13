# DAgger 异步推理部署运行指南

> 本文档是 DAgger 系统部署的第一个章节，介绍如何基于 `dagger` 框架和 `unified_deploy` Server 进行异步策略推理。
> 最后更新：2026-02-11

---

## 目录

1. [系统架构总览](#1-系统架构总览)
2. [环境准备](#2-环境准备)
3. [运行方式一：AsyncInferenceClient（纯策略推理）](#3-运行方式一asyncinferenceclient纯策略推理)
4. [运行方式二：DAgger Node（ROS2 + VR 接管）](#4-运行方式二dagger-nodeROS2--VR-接管)
5. [Web UI 控制面板](#5-web-ui-控制面板)
6. [Server 端启动详解](#6-server-端启动详解)
7. [关键配置参数](#7-关键配置参数)
8. [运行时日志解读](#8-运行时日志解读)
9. [故障排查](#9-故障排查)

---

## 1. 系统架构总览

整个异步推理系统由两部分组成：**Server（策略推理）** 和 **Client（机器人控制）**。

```
┌─────────────────────────────────────────────────────────────────┐
│                        Server 端 (GPU 机器)                      │
│                                                                   │
│  PolicyServer (gRPC)                                             │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Adapter (ACT / Diffusion / VLA / OpenPI / Octo)        │    │
│  │  ┌──────────┐  ┌──────────────┐  ┌────────────────┐    │    │
│  │  │ 模型加载  │  │ 观测预处理    │  │ Action Chunk   │    │    │
│  │  │ .load()  │  │ prepare_obs() │  │ 生成 + 后处理  │    │    │
│  │  └──────────┘  └──────────────┘  └────────────────┘    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                          ↕ gRPC (pickle 序列化)                  │
└─────────────────────────────────────────────────────────────────┘
                           ↕ 网络 (默认 127.0.0.1:50051)
┌─────────────────────────────────────────────────────────────────┐
│                        Client 端 (机器人侧)                      │
│                                                                   │
│  方式 A: AsyncInferenceClient          方式 B: DAgger Node       │
│  ┌────────────────────────┐    ┌──────────────────────────┐     │
│  │ 推理线程 → Ring Buffer  │    │ InferenceBridge          │     │
│  │ 执行循环 → 机械臂控制   │    │ + VR 遥操作 + 模式切换   │     │
│  │ (独立进程，直连机械臂)  │    │ (ROS2 节点，话题通信)    │     │
│  └────────────────────────┘    └──────────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

**两种 Client 模式**：

| 模式 | 文件 | 适用场景 | 特点 |
|------|------|---------|------|
| **AsyncInferenceClient** | `dagger/async_inference_client.py` | 纯策略推理部署 | 独立进程，直连机械臂，无需 ROS2 |
| **DAgger Node** | `dagger/dagger_node.py` | DAgger 在线学习 | ROS2 节点，支持 VR 接管 + 录制 |

两种模式共享同一个 **PolicyServer**。

---

## 2. 环境准备

### 2.1 硬件要求

- **机械臂**: Realman RM65-B，已上电，网络可达（默认 `192.168.1.18:8080`）
- **相机**: Intel RealSense x2（序列号见 `dagger_params.yaml`）
- **GPU**: CUDA 可用（Server 端推理）
- **VR 手柄**: （仅 DAgger Node 模式需要）

### 2.2 软件环境

```bash
# 激活 conda 环境
conda activate robocoin

# 验证环境
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import grpc; print(f'gRPC: {grpc.__version__}')"
```

### 2.3 目录结构

```
lerobotv3/
├── RoboCOIN/src/lerobot/extensions/unified_deploy/
│   └── server/
│       ├── policy_server.py          ← Server 入口
│       └── adapters/                 ← 策略适配器 (ACT, Diffusion, VLA...)
├── dagger/
│   ├── async_inference_client.py     ← Client 方式 A
│   ├── dagger_node.py                ← Client 方式 B (ROS2)
│   ├── core/
│   │   ├── inference_bridge.py       ← gRPC 推理桥接层
│   │   ├── ring_buffer.py            ← Action 环形缓冲区
│   │   ├── udp_state.py              ← UDP 状态接收
│   │   ├── data_recorder.py          ← DAgger 录制器
│   │   └── gripper_controller.py     ← 异步夹爪控制
│   ├── config/
│   │   └── dagger_params.yaml        ← 运行时配置
│   └── launch/
│       └── dagger.launch.py          ← ROS2 Launch 文件
```

---

## 3. 运行方式一：AsyncInferenceClient（纯策略推理）

> 适用于：不需要 VR 接管，只需要策略推理控制机械臂的场景。

> 遇到ModuleNotFoundError: No module named 'dagger'错误，可使用：
```bash
PYTHONPATH=/home/ubuntu/Desktop/Workspace/lerobot_policy_deploy/lerobotv3:$PYTHONPATH
```

### 3.1 启动流程（两个终端）

#### 终端 1 — 启动 PolicyServer

```bash
conda activate robocoin
cd ~/Desktop/Workspace/lerobot_policy_deploy/lerobotv3/RoboCOIN

python -m lerobot.extensions.unified_deploy.server.policy_server \
    --policy_type=act \
    --pretrained_path=<模型路径> \
    --device=cuda \
    --port=50051
```

等待输出 `PolicyServer 已启动，监听 0.0.0.0:50051` 后继续。

> **支持的 policy_type**: `act`, `diffusion`, `pi0`, `pi0fast`, `smolvla`, `openpi`, `octo`, `debug`

#### 终端 2 — 启动 AsyncInferenceClient

```bash
conda activate robocoin
cd ~/Desktop/Workspace/lerobot_policy_deploy/lerobotv3

python dagger/async_inference_client.py \
    --config dagger/config/dagger_params.yaml \
    --server 127.0.0.1:50051 \
    --task "pick up the yellow banana and put it in the white plate"
```

### 3.2 常用命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--config` | 无 | YAML 配置文件路径 |
| `--server` | YAML 中的值 | PolicyServer 地址 (host:port) |
| `--robot-ip` | YAML 中的值 | 机械臂 IP |
| `--f-exec` | YAML 中的值 | 执行频率 Hz（建议 30） |
| `--T-inter` | YAML 中的值 | 推理间隔秒数（0=自动推导） |
| `--n-action-steps` | YAML 中的值 | ActionChunk 截取步数（0=全部） |
| `--task` | YAML 中的值 | 任务指令（VLA 模型需要） |
| `--dry-run` | false | 仿真模式（不控制机械臂） |
| `--visualize` | false | 显示可视化窗口 |
| `--max-steps` | YAML 中的值 | 最大执行步数（0=无限） |
| `--no-hold` | false | 禁用 hold_on_empty |

### 3.3 推荐配置

**首次测试（安全）**：

```bash
python dagger/async_inference_client.py \
    --config dagger/config/dagger_params.yaml \
    --dry-run \
    --f-exec 10 \
    --max-steps 200
```

**正式部署（30Hz）**：

```bash
python dagger/async_inference_client.py \
    --config dagger/config/dagger_params.yaml \
    --f-exec 30 \
    --task "pick up the yellow banana and put it in the white plate"
```

**高频部署（50Hz，需 chunk_size >= 50）**：

```bash
python dagger/async_inference_client.py \
    --config dagger/config/dagger_params.yaml \
    --f-exec 50 \
    --n-action-steps 0 \
    --task "pick up the yellow banana and put it in the white plate"
```

> **重要**：50Hz 时不要设 `--n-action-steps 20`，有效 action 太少会导致严重卡顿。详见 [async-inference-system-explained.md](async-inference-system-explained.md#7-真机测试分析-2026-02-09)。

### 3.4 工作流程

```
启动后自动执行:

1. 连接机械臂 (TCP)
2. 初始化相机
3. 连接 PolicyServer (gRPC)
4. Warmup: 5 次预热推理
   - 丢弃第 1 次（GPU 冷启动）
   - 取后 4 次平均值作为 t_inf
   - 获取模型 chunk_size
5. 参数自动推导 (T_inter, K_inf, chunk_size_threshold)
6. 启动推理线程 (后台)
7. 启动执行循环 (主线程, f_exec Hz)
   ┌─────────────────────────────────────────┐
   │  推理线程:                               │
   │    采集观测 → gRPC 推理 → K_inf 裁剪     │
   │    → 写入 Ring Buffer → 等待 T_inter     │
   │                                          │
   │  执行循环:                               │
   │    从 Ring Buffer pop → 执行 action      │
   │    → buffer 空时 hold last action        │
   └─────────────────────────────────────────┘
8. Ctrl+C 退出，释放资源
```

---

## 4. 运行方式二：DAgger Node（ROS2 + VR 接管）

> 适用于：DAgger 在线学习，需要 VR 手柄实时接管和数据录制的场景。

### 4.1 启动流程

#### 方式 A — 自动启动 PolicyServer（推荐）

只需一个终端，Launch 文件自动启动 PolicyServer + 所有 ROS2 节点：

```bash
conda activate robocoin
cd ~/Desktop/Workspace/lerobot_policy_deploy/lerobotv3

ros2 launch dagger/launch/dagger.launch.py
```

前提：在 `dagger/config/dagger_params.yaml` 中配置 `policy_server` 段：

```yaml
policy_server:
  auto_launch: true
  policy_type: "act"
  pretrained_path: "outputs/act_realman_eef/checkpoints/last/pretrained_model"
  device: "cuda"
  port: 50051
```

Launch 文件会先启动 PolicyServer，延迟 5 秒后再启动 dagger_node，确保 gRPC 通道就绪。

#### 方式 B — 手动启动 PolicyServer

适用于 Server 和 Client 在不同机器上的场景：

**终端 1 — 启动 PolicyServer**

```bash
conda activate robocoin
cd ~/Desktop/Workspace/lerobot_policy_deploy/lerobotv3/RoboCOIN

python -m lerobot.extensions.unified_deploy.server.policy_server \
    --policy_type=act \
    --pretrained_path=<模型路径> \
    --device=cuda \
    --port=50051
```

**终端 2 — 启动 DAgger Launch**

```bash
conda activate robocoin
cd ~/Desktop/Workspace/lerobot_policy_deploy/lerobotv3

# 设置 auto_launch=false（或在 yaml 中配置）
ros2 launch dagger/launch/dagger.launch.py
```

**Launch 参数覆盖**：

```bash
# 指定机械臂 IP
ros2 launch dagger/launch/dagger.launch.py ip:=192.168.1.18

# 仿真模式
ros2 launch dagger/launch/dagger.launch.py dry_run:=true

# 指定 VR 数据源
ros2 launch dagger/launch/dagger.launch.py vr_data_source:=quest3
```

### 4.2 Launch 启动的节点

| 节点 | 功能 |
|------|------|
| `PolicyServer` | 策略推理服务（auto_launch=true 时） |
| `realman_driver_node` | 机械臂驱动（状态发布 + 动作执行） |
| `vr_input_node` | VR 手柄输入（位姿 + 按键） |
| `camera_node` x N | 相机图像发布 |
| `dagger_node` | DAgger 核心节点（推理 + VR 接管 + 录制） |
| `control_panel` | Web UI 控制面板（端口 5000） |

### 4.3 统一会话控制

DAgger Node 使用 **统一会话控制** 管理推理和录制的生命周期：

```
                    ┌──────────┐
                    │   IDLE   │  ← 启动后默认
                    └────┬─────┘
                         │ start_session
                         ▼
                    ┌──────────┐
              ┌────►│  POLICY  │◄────┐
              │     └────┬─────┘     │
              │          │           │
              │  trigger │  trigger  │
              │  release │  press    │
              │          ▼           │
              │     ┌──────────┐    │
              └─────┤  HUMAN   ├────┘
                    └────┬─────┘
                         │ stop_session
                         ▼
                    ┌──────────┐
                    │   IDLE   │  ← 录制暂停，等待确认
                    └──────────┘
```

**会话操作**：

| 操作 | ROS2 Service | Web UI | 效果 |
|------|-------------|--------|------|
| **开始会话** | `/dagger/start_session` | "开始推理"按钮 | 连接 PolicyServer → POLICY 模式 → 自动开始录制 |
| **停止会话** | `/dagger/stop_session` | "停止推理"按钮 | 暂停录制 → 停止推理 → IDLE → 弹窗确认保存/丢弃 |
| **VR 接管** | 自动（trigger 按下） | — | POLICY → HUMAN，暂停推理，VR 直接控制 |
| **VR 释放** | 自动（trigger 松开） | — | HUMAN → POLICY，恢复推理 |

**通过命令行操作**：

```bash
# 开始推理会话
ros2 service call /dagger/start_session std_srvs/srv/Trigger

# 停止推理会话
ros2 service call /dagger/stop_session std_srvs/srv/Trigger
```

> **注意**：旧的 `/dagger/start_episode` 和 `/dagger/stop_episode` service 仍然可用，但推荐使用统一会话控制。

### 4.4 录制功能

录制在 **start_session** 时自动开始，**stop_session** 时暂停等待确认：

1. **开始推理** → 自动创建新 episode，开始录制状态 + 图像 + 策略动作
2. **VR 接管** → 录制继续（记录 HUMAN 控制源）
3. **VR 释放** → 录制继续（记录 POLICY 控制源）
4. **停止推理** → 录制暂停，Web UI 弹出确认弹窗
   - **保存**：调用 `/dagger/save_episode`，数据写入磁盘
   - **丢弃**：调用 `/dagger/discard_episode`，丢弃本次数据

录制配置在 `dagger_params.yaml` 中：

```yaml
enable_recording: true
repo_id: "local/dagger_realman_001"
dataset_root: "/path/to/data/dagger/"
recording_fps: 30
max_episodes: 0          # 0=无限制
recording_task: "pick and place"
```

---

## 5. Web UI 控制面板

DAgger 系统内置 Web UI 控制面板，通过浏览器即可监控和操作。

### 5.1 访问方式

Launch 启动后，Web UI 自动运行在 `http://<机器人侧IP>:5000`。

```bash
# 本机访问
http://localhost:5000

# 远程访问（替换为实际 IP）
http://192.168.1.100:5000
```

### 5.2 页面布局

```
┌─────────────────────────────────────────────────────────────┐
│                    DAgger 控制面板                            │
│                  [IDLE] / [POLICY] / [HUMAN]                 │
├──────────────┬──────────────────────┬───────────────────────┤
│  左列         │  中列                │  右列                  │
│              │                      │                        │
│ Server 状态   │  RGB 相机预览         │  系统日志              │
│ ● 在线/离线   │  ┌────┐  ┌────┐     │  [INFO] ...            │
│ 策略: act     │  │cam1│  │cam2│     │  [WARN] ...            │
│ 模型: ...     │  └────┘  └────┘     │  [ERROR] ...           │
│              │                      │                        │
│ 节点状态      │                      │                        │
│ 模式: IDLE    │                      │                        │
│ 帧数: 0       │                      │                        │
│              │                      │                        │
│ 推理控制      │                      │                        │
│ [开始推理]    │                      │                        │
│ [停止推理]    │                      │                        │
│              │                      │                        │
│ [Home]       │                      │                        │
├──────────────┴──────────────────────┴───────────────────────┤
│                        状态栏                                │
└─────────────────────────────────────────────────────────────┘
```

### 5.3 核心功能

| 功能 | 说明 |
|------|------|
| **Server 状态卡片** | 显示 PolicyServer 连接状态（绿/黄/红圆点）、策略类型、模型路径 |
| **节点状态** | 当前控制模式、推理帧数、buffer 大小、warmup 状态 |
| **开始推理** | 调用 `start_session`：连接 Server → POLICY → 自动录制 |
| **停止推理** | 调用 `stop_session`：暂停录制 → IDLE → 弹窗确认保存/丢弃 |
| **RGB 相机预览** | 实时显示各相机 RGB 画面（MJPEG 流） |
| **系统日志** | 实时滚动显示 dagger_node 日志 |
| **Home** | 调用 `/dagger/go_home`，机械臂回到初始位姿 |

### 5.4 Server 状态指示

| 圆点颜色 | 状态 | 含义 |
|----------|------|------|
| 🟢 绿色 | 在线 | PolicyServer 已连接且 warmup 完成，可以开始推理 |
| 🟡 黄色 | 连接中 | PolicyServer 已连接但 warmup 未完成 |
| 🔴 红色 | 离线 | PolicyServer 未连接或连接断开 |

### 5.5 停止推理确认弹窗

当有活跃录制时，点击"停止推理"后弹出确认弹窗：

- **保存数据**：将本次 episode 数据写入磁盘（调用 `/dagger/save_episode`）
- **丢弃数据**：丢弃本次 episode 数据（调用 `/dagger/discard_episode`）

无活跃录制时直接回到 IDLE，不弹窗。

---

## 6. Server 端启动详解

### 6.1 完整参数

```bash
python -m lerobot.extensions.unified_deploy.server.policy_server \
    --host 0.0.0.0 \
    --port 50051 \
    --policy_type act \
    --pretrained_path <模型路径> \
    --device cuda \
    --fps 30 \
    --n_action_steps 0
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | `0.0.0.0` | gRPC 监听地址 |
| `--port` | `50051` | gRPC 监听端口 |
| `--policy_type` | `act` | 策略类型 |
| `--pretrained_path` | `""` | 模型路径（**必需**） |
| `--device` | `cuda` | 推理设备 |
| `--fps` | `30` | 目标帧率 |
| `--n_action_steps` | `0` | 返回的动作步数（0=模型默认） |
| `--debug` | false | Debug 模式（从数据集读取 action） |

### 6.2 支持的策略类型

| policy_type | 适配器 | 模型框架 |
|-------------|--------|---------|
| `act` | ACTAdapter | LeRobot ACT |
| `diffusion` | DiffusionAdapter | LeRobot Diffusion Policy |
| `pi0` / `pi0fast` / `smolvla` | VLAAdapter | LeRobot VLA |
| `openpi` | OpenPIAdapter | OpenPI π₀/π₀.5 |
| `octo` | OctoAdapter | Octo (WebSocket) |
| `debug` | DebugAdapter | 从数据集回放 |

### 6.3 Server 自动配置读取

Server 启动时从模型目录下的 `train_config.json` 自动读取：

- `action_space`: `"joints"` 或 `"pose"`
- `action_type`: `"absolute"` 或 `"delta"`
- `delta_mask`: delta 维度掩码
- `delta_mode`: `"relative"` 或 `"frame_diff"`

**无需手动指定**，Server 日志会打印读取到的配置。

### 6.4 Debug 模式（无需 GPU）

用于测试通信链路，从数据集回放 action：

```bash
python -m lerobot.extensions.unified_deploy.server.policy_server \
    --debug \
    --debug_dataset_path=/path/to/dataset \
    --debug_episode=0 \
    --debug_action_space=joints \
    --debug_action_type=absolute \
    --port=50051
```

---

## 7. 关键配置参数

### 7.1 dagger_params.yaml 核心参数

```yaml
# === 连接 ===
robot_ip: "192.168.1.18"
robot_port: 8080
server_address: "127.0.0.1:50051"

# === PolicyServer 自动启动 ===
policy_server:
  auto_launch: true       # true=Launch 自动启动 PolicyServer
  policy_type: "act"      # 策略类型（act/diffusion/pi0/...）
  pretrained_path: ""     # 模型路径（必需，为空则跳过启动）
  device: "cuda"          # 推理设备
  port: 50051             # gRPC 端口

# === 执行频率 ===
f_exec: 30.0          # 建议 30Hz，首次测试用 10Hz

# === 推理控制（建议全部设 0，自动推导）===
n_action_steps: 0     # 0=使用模型全部 chunk_size
T_inter: 0.0          # 0=自动推导（T_inter_max * 0.8）
t_inf: 0.0            # 0=warmup 自动测量
chunk_size_threshold: 0  # 0=自动推导

# === 安全 ===
hold_on_empty: true    # buffer 空时重发最后 action（推荐 true）
```

### 7.2 参数自动推导

当 `T_inter=0, t_inf=0, chunk_size_threshold=0` 时，系统在 warmup 后自动推导：

```
输入:
  L = 模型 chunk_size (warmup 获取，如 50)
  f_exec = 30 Hz
  t_inf = warmup 测量值 (如 0.105s)

推导:
  T_step = 1/30 = 0.033s
  K_inf = ceil(0.105 * 30) = 4
  L_valid = 50 - 4 = 46
  T_inter_max = 50/30 - 0.105 = 1.562s
  T_inter = 1.562 * 0.8 = 1.249s (20% 安全余量)
  chunk_size_threshold = max(2, 4) = 4

含义:
  每次推理产生 46 个有效 action
  46 个 action 在 30Hz 下可供 1.53s
  推理间隔 1.25s，留有 0.28s 余量
```

### 7.3 稳定性约束

```
必须满足: L_valid > 0 且 T_inter_max > 0
即: chunk_size > K_inf 且 chunk_size / f_exec > t_inf
```

不满足时系统自动降级为被动等待模式（等 buffer 消耗到阈值再推理）。

---

## 8. 运行时日志解读

### 8.1 Warmup 阶段

```
[AsyncClient] Warmup 1/5: t_inf=3.412s (cold start, discarded)
[AsyncClient] Warmup 2/5: t_inf=0.102s, chunk_size=50
[AsyncClient] Warmup 3/5: t_inf=0.098s, chunk_size=50
[AsyncClient] Warmup 4/5: t_inf=0.115s, chunk_size=50
[AsyncClient] Warmup 5/5: t_inf=0.105s, chunk_size=50
[AsyncClient] t_inf(stable avg) = 0.105s, chunk_size = 50
```

- 第 1 次是 GPU 冷启动，自动丢弃
- 后 4 次取平均值

### 8.2 参数推导

```
[AsyncClient] === Parameter Derivation ===
  L=50, L_eff=50, K_inf=4, L_valid=46
  T_inter=1.249s (auto), T_inter_max=1.562s
  chunk_size_threshold=4, feasible=True
```

### 8.3 运行时周期日志

```
Frame 100, FPS=29.8, Exec=95, Hold=5, Infer=3, Buf=7
```

| 字段 | 含义 | 健康范围 |
|------|------|---------|
| `FPS` | 实际执行频率 | 接近 f_exec |
| `Exec` | 正常执行的 action 数 | 持续增长 |
| `Hold` | buffer 空时重发次数 | Hold/(Exec+Hold) < 10% |
| `Infer` | 完成的推理次数 | 持续增长 |
| `Buf` | 当前 buffer 剩余 | > 0 为佳 |

### 8.4 推理日志

```
[Inference] #3: t_inf=0.150s, K_inf=7, chunk=50, valid=43, buf=43
```

| 字段 | 含义 |
|------|------|
| `t_inf` | 本次推理耗时 |
| `K_inf` | 裁剪掉的过时 action 数（含 +2 安全余量） |
| `chunk` | Server 返回的总 action 数 |
| `valid` | 裁剪后有效 action 数 |
| `buf` | 更新后 buffer 大小 |

---

## 9. 故障排查

| 现象 | 可能原因 | 解决方案 |
|------|---------|---------|
| `No module named 'dagger'` | pickle 序列化路径错误 | 确认 data_types 从 `lerobot.extensions.unified_deploy.core.data_types` 导入 |
| `Infer=0` 始终为 0 | gRPC 通信失败 | 检查 Server 是否启动，端口是否正确 |
| FPS 远低于 f_exec | 观测采集太慢 | 检查相机连接，降低 f_exec |
| 机械臂抖动/撞击感 | K_inf 裁剪不足 | 已修复（+2 安全余量），确认使用最新代码 |
| Hold 比例 > 50% | 推理太慢或 chunk 太小 | 降低 f_exec，或设 `n_action_steps=0` 使用全部 chunk |
| Warmup 降级 | GPU 冷启动导致 t_inf 偏大 | 已修复（多次测量），确认使用最新代码 |
| `K_inf > chunk` | 推理太慢 | 降低 f_exec 或优化 Server 推理速度 |
| Server CUDA OOM | GPU 显存不足 | 关闭其他 GPU 进程，或用 `--device cpu` |
| `terminate called` 退出时 | Realman SDK C 库线程清理 | 不影响功能，可忽略 |
| Web UI "开始推理"失败 | PolicyServer 未启动或连接超时 | 检查 Server 状态卡片是否显示绿色，确认 `server_address` 配置正确 |
| Web UI Server 状态红色 | PolicyServer 未启动 | 确认 `auto_launch: true` 且 `pretrained_path` 非空，或手动启动 Server |
| start_session 报"已在运行中" | 重复点击"开始推理" | 先点"停止推理"回到 IDLE 再重新开始 |
| 停止后弹窗不出现 | 录制未启用或无活跃 episode | 检查 `enable_recording: true`，确认会话期间有录制数据 |

---

## 附录：相关文档

| 文档 | 内容 |
|------|------|
| [async-inference-system-explained.md](async-inference-system-explained.md) | 异步推理系统原理深度解析（K_inf 裁剪、参数推导公式、真机测试数据） |
| [phase1_testing_guide.md](phase1_testing_guide.md) | Phase 1 测试指南（分步测试用例、对比测试表格） |
| [common-pitfalls.md](common-pitfalls.md) | 常见易犯错误与踩坑记录 |
| [real_machine_testing_guide.md](real_machine_testing_guide.md) | 真机测试指南（Web UI 测试、集成测试用例） |
