# DAgger 系统架构文档

> 本文档是 DAgger 系统的完整架构参考，涵盖系统设计、模块职责、数据流、线程模型、状态机、录制系统和 Web UI。适合需要理解系统全貌或进行二次开发的工程师阅读。

## 📋 目录

- [1. 系统概述](#1-系统概述)
- [2. 目录结构](#2-目录结构)
- [3. 系统架构总览](#3-系统架构总览)
- [4. 核心模块详解](#4-核心模块详解)
  - [4.1 DAggerNode (dagger_node.py)](#41-daggernodedagger_nodepy)
  - [4.2 InferenceBridge (core/inference_bridge.py)](#42-inferencebridgecoreinference_bridgepy)
  - [4.3 ActionRingBuffer (core/ring_buffer.py)](#43-actionringbuffercorering_bufferpy)
  - [4.4 UDPStateReceiver (core/udp_state.py)](#44-udpstatereceivercoredp_statepy)
  - [4.5 DAggerRecorder (core/data_recorder.py)](#45-daggerrecordercoredata_recorderpy)
  - [4.6 GripperAsyncController (core/gripper_controller.py)](#46-gripperasynccontrollercoreripper_controllerpy)
  - [4.7 AsyncInferenceClient (async_inference_client.py)](#47-asyncinferenceclientasync_inference_clientpy)
- [5. 状态机](#5-状态机)
- [6. 线程模型与锁策略](#6-线程模型与锁策略)
- [7. 数据流](#7-数据流)
  - [7.1 观测数据流](#71-观测数据流)
  - [7.2 动作执行流](#72-动作执行流)
  - [7.3 录制数据流](#73-录制数据流)
- [8. 会话生命周期](#8-会话生命周期)
- [9. 录制系统](#9-录制系统)
- [10. Web UI 控制面板](#10-web-ui-控制面板)
- [11. 配置参数参考](#11-配置参数参考)
- [12. 关键设计决策](#12-关键设计决策)

---

## 1. 系统概述

DAgger (Dataset Aggregation) 系统是一个基于 ROS2 的机器人策略部署与数据采集框架，核心功能：

- **策略推理执行**: 通过 gRPC 连接 PolicyServer，异步推理并执行动作
- **VR 人类接管**: 操作员通过 VR 手柄实时遥操机械臂，随时接管/释放控制权
- **DAgger 数据录制**: 同步录制机器人状态、相机图像、策略输出和控制来源，生成 LeRobot v2.1 格式数据集
- **Web UI 监控**: 浏览器实时查看系统状态、相机画面、切换模式、管理录制

系统支持两种运行模式：

| 模式 | 入口 | 特点 |
|------|------|------|
| **DAgger Node** (推荐) | `dagger_node.py` + ROS2 launch | 完整功能：VR 接管 + 策略推理 + 录制 + Web UI |
| **AsyncInferenceClient** | `async_inference_client.py` | 纯策略推理，无 VR，无 ROS2 依赖 |


---

## 2. 目录结构

```
dagger/
├── dagger_node.py              # 🎯 核心 ROS2 节点（状态机 + VR + 策略 + 录制）
├── async_inference_client.py   # 纯策略推理客户端（无 ROS2 依赖）
├── __init__.py
│
├── core/                       # 核心模块（无 ROS2 依赖）
│   ├── inference_bridge.py     # gRPC 推理桥接层（观测→推理→ring buffer）
│   ├── ring_buffer.py          # 线程安全动作环形缓冲区
│   ├── udp_state.py            # UDP 机械臂状态接收器（200Hz）
│   ├── data_recorder.py        # DAgger 数据录制器（LeRobot v2.1 格式）
│   ├── gripper_controller.py   # 异步夹爪控制器（RM+ 生态）
│   └── __init__.py
│
├── config/
│   └── dagger_params.yaml      # 运行时配置文件
│
├── launch/
│   └── dagger.launch.py        # ROS2 launch 文件（driver + VR + cameras + dagger_node）
│
├── web_ui/
│   └── control_panel.py        # Flask + ROS2 Web UI 控制面板
│
├── deps/                       # 从 unified_deploy 复制的依赖
│   ├── configs.py              # 配置类
│   ├── constants.py            # 常量定义
│   ├── data_types.py           # 数据类型（UnifiedObservation, ActionChunk 等）
│   ├── gripper_controller.py   # 夹爪控制器（原始版本）
│   └── utils.py                # 工具函数
│
├── docs/                       # 文档
│   ├── architecture.md         # 📖 本文档
│   ├── deployment-guide.md     # 部署指南
│   ├── quick-start-guide.md    # 快速上手
│   ├── common-pitfalls.md      # 踩坑记录（BUG-001~020）
│   ├── async-inference-system-explained.md  # 异步推理系统详解
│   ├── phase1_testing_guide.md # Phase 1 测试指南
│   └── real_machine_testing_guide.md        # 真机测试指南
│
├── data/                       # 录制数据输出目录
├── logs/                       # 诊断日志
└── tests/                      # 单元测试
```

---

## 3. 系统架构总览

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DAgger 系统架构                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    gRPC (50051)    ┌──────────────────────┐      │
│  │ PolicyServer │◄──────────────────►│   InferenceBridge    │      │
│  │  (GPU 推理)   │                    │  ┌────────────────┐  │      │
│  └──────────────┘                    │  │ ActionRingBuffer│  │      │
│                                      │  └───────┬────────┘  │      │
│                                      └──────────┼───────────┘      │
│                                                  │                  │
│  ┌──────────┐   VR Topics    ┌───────────────────┼──────────┐      │
│  │ VR 手柄   │──────────────►│                   │          │      │
│  │ (Quest3)  │  pose/trigger │     DAggerNode    │          │      │
│  └──────────┘               │   ┌────────────┐  │          │      │
│                              │   │ 状态机      │  │          │      │
│  ┌──────────┐  camera topics │   │ IDLE       │  │          │      │
│  │ RealSense│──────────────►│   │ HUMAN ◄──► │  │          │      │
│  │ x2       │  image_raw    │   │ POLICY     │  │          │      │
│  └──────────┘               │   └────────────┘  │          │      │
│                              │         │          │          │      │
│  ┌──────────┐  joint_state  │   ┌─────┴──────┐  │          │      │
│  │ Realman  │◄─────────────►│   │ Recording  │  │          │      │
│  │ Driver   │  action_pose/ │   │ Sync       │  │          │      │
│  │ Node     │  action_joint │   └────────────┘  │          │      │
│  └──────────┘               └───────────────────────────────┘      │
│                                        │                            │
│                              ┌─────────┴──────────┐                │
│                              │  DAggerRecorder     │                │
│                              │  (LeRobot v2.1)     │                │
│                              └────────────────────┘                │
│                                                                     │
│  ┌──────────────────────────────────────────────────┐              │
│  │  Web UI (Flask, port 5002)                        │              │
│  │  订阅 /dagger/status + cameras + rosout           │              │
│  │  代理 ROS2 Service 调用                            │              │
│  └──────────────────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
```

**外部依赖节点**（由 launch 文件启动，非 DAgger 代码）：

| 节点 | 职责 |
|------|------|
| `realman_driver_node` | 机械臂通信：执行 action_pose / action_joint_state，发布 state_joint_state |
| `vr_input_node` | VR 数据采集：发布 /vr/{side}/pose, trigger, grip |
| `camera_node` x N | RealSense 相机：发布 /camera/{name}/color/image_raw |
| `PolicyServer` | GPU 策略推理服务（gRPC，可选自动启动） |


---

## 4. 核心模块详解

### 4.1 DAggerNode (`dagger_node.py`)

> 系统核心，2430 行。ROS2 节点，整合状态机、VR 遥操、策略推理和录制。

**职责**：
- 管理 IDLE / HUMAN / POLICY 三态切换
- HUMAN 模式：接收 VR 手柄数据，计算目标笛卡尔位姿，发布 `/rm/action_pose`
- POLICY 模式：从 InferenceBridge 取 action，安全检查后发布 `/rm/action_joint_state`
- 喂观测给 InferenceBridge（状态 + 图像）
- 驱动录制同步回调（ApproximateTimeSynchronizer）
- 发布 `/dagger/status` JSON 供 Web UI 消费
- 提供 ROS2 Service 接口（模式切换、录制控制、会话管理）

**关键内部组件**：

| 组件 | 说明 |
|------|------|
| `VRControlState` | VR 手柄状态（位姿、trigger、grip、初始化标志） |
| `VRCoordinateTransform` | VR→机械臂坐标变换（rotation_preset 配置） |
| `PoseFilter` | 位姿低通滤波器（alpha_pos / alpha_rot） |
| `InferenceBridge` | gRPC 推理桥接（见 4.2） |
| `DAggerRecorder` | 数据录制器（见 4.5） |

**ROS2 订阅**：

| 话题 | 类型 | 用途 |
|------|------|------|
| `/vr/{side}/pose` | PoseStamped | VR 手柄位姿 |
| `/vr/{side}/trigger` | Float32 | VR 扳机值（0~1） |
| `/vr/{side}/grip` | Float32 | VR 握持值（夹爪控制） |
| `/rm/state_joint_state` | JointState | 14D 机械臂状态 |
| `/camera/{name}/color/image_raw` | Image | RGB 相机图像 |

**ROS2 发布**：

| 话题 | 类型 | 用途 |
|------|------|------|
| `/rm/action_pose` | PoseStamped | HUMAN 模式笛卡尔目标位姿 |
| `/rm/action_joint_state` | JointState | POLICY 模式关节角度 + 夹爪 |
| `/dagger/status` | String (JSON) | 系统状态（2Hz 定时 + 事件触发） |

**ROS2 Service**：

| Service | 功能 |
|---------|------|
| `/dagger/enable_control` | 启用控制（进入 POLICY 或 HUMAN） |
| `/dagger/disable_control` | 禁用控制（回到 IDLE） |
| `/dagger/set_idle` | 切换到 IDLE |
| `/dagger/set_human` | 切换到 HUMAN |
| `/dagger/set_policy` | 切换到 POLICY |
| `/dagger/start_episode` | 开始录制 episode |
| `/dagger/stop_episode` | 停止并保存 episode |
| `/dagger/pause_episode` | 暂停录制 |
| `/dagger/discard_episode` | 丢弃当前 episode |
| `/dagger/new_episode` | 在当前会话中开始新 episode |
| `/dagger/start_session` | 一键启动会话（enable_control + start_episode） |
| `/dagger/stop_session` | 一键停止会话（暂停录制 + IDLE） |
| `/dagger/pause_session` | 暂停会话（停止执行，保持连接） |
| `/dagger/resume_session` | 恢复会话 |

---

### 4.2 InferenceBridge (`core/inference_bridge.py`)

> 661 行。gRPC 推理桥接层，从 AsyncInferenceClient 提取，不管理机器人连接。

**职责**：
- 管理 gRPC 连接（connect / disconnect / probe）
- 构建 `UnifiedObservation`（14D state + images → pickle 序列化）
- 发送推理请求，解析 `ActionChunk` 响应
- 管理 ActionRingBuffer（推理线程写入，执行线程读取）
- Warmup 机制（首次推理验证 pipeline + 测量 t_inf）
- 参数自动推导（T_inter、chunk_size_threshold、K_inf）

**线程模型**：

```
主线程 (ROS2 executor)          推理线程 (_inference_loop)
    │                                │
    ├─ update_observation()          │
    │   写入 _obs_state_14d          │
    │   写入 _obs_images             │
    │   设置 _obs_updated ──────────►│ 等待 _obs_updated
    │                                │ 读取观测快照
    │                                │ gRPC 推理请求
    │                                │ ring_buffer.update()
    │                                │
    ├─ get_next_action()             │
    │   ring_buffer.pop_current()    │
    │                                │
    ├─ pause() / resume()            │
    │   _pause_event 控制 ──────────►│ 等待 _pause_event
```

**关键配置**（`InferenceBridgeConfig`）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `server_address` | `127.0.0.1:50051` | PolicyServer 地址 |
| `task` | `""` | 任务描述（传给 policy） |
| `f_exec` | 30.0 | 执行频率 Hz |
| `n_action_steps` | 0 (auto) | 每次推理使用的 action 步数 |
| `T_inter` | 0.0 (auto) | 推理间隔秒数 |
| `t_inf` | 0.0 (auto) | 实测推理延迟 |
| `chunk_size_threshold` | 0 (auto) | buffer 安全阈值 |
| `hold_on_empty` | true | buffer 空时重发最后一个 action |

**参数自动推导**（当值为 0 时）：
```
chunk_size = 模型返回的 action chunk 大小
K_inf = chunk_size（使用全部 action steps）
T_inter = K_inf / f_exec（消耗完一个 chunk 的时间）
t_inf = warmup 实测（EMA 平滑）
chunk_size_threshold = max(1, K_inf - ceil(t_inf * f_exec) - 2)
```

---

### 4.3 ActionRingBuffer (`core/ring_buffer.py`)

> 70 行。线程安全的动作环形缓冲区。

**设计**：
- `update(actions)`: 推理线程调用，**替换**整个缓冲区（新推理基于更新的观测，旧 action 过时）
- `pop_current()`: 执行线程调用，FIFO 取出一个 action，返回 None 表示空
- `last_popped`: 最近一次 pop 的 action（用于 hold position 和录制）
- `clear()`: 清空缓冲区，不清空 last_popped
- 容量默认 200，基于 `deque(maxlen=N)` 实现

---

### 4.4 UDPStateReceiver (`core/udp_state.py`)

> UDP 机械臂状态接收器，最高 200Hz。

**特点**：
- 独立 UDP 端口（默认 8089），与控制指令完全分离
- 后台线程接收，`read_full_state()` 仅读缓存（零延迟）
- 数据格式：7 关节角度(deg) + 夹爪位置 + 6 TCP 位姿

**启用流程**：
1. `enable_rm_plus(ip, port, baud)` — 通过 JSON TCP 启用 RM+ 生态
2. `enable_udp_push(ip, port, cycle)` — 通过 SDK API 开启 UDP 推送
3. `start()` — 启动后台接收线程

⚠️ **重要**：RM+ 必须在 UDP push 之前启用，否则 `rm_plus_state` 字段为 `"offline"`。

---

### 4.5 DAggerRecorder (`core/data_recorder.py`)

> DAgger 数据录制器，生成 LeRobot v2.1 格式数据集。

**数据格式**（VR 录制的超集）：

| 字段 | 维度 | 说明 |
|------|------|------|
| `observation.state` | (14,) | 7 关节角(rad) + 夹爪(0~1) + 6 TCP 位姿 |
| `action` | (14,) | action[t] = state[t+1]（后处理生成） |
| `observation.images.*` | (C,H,W) | RGB 相机图像 |
| `policy_action` | (N,) | 策略推理原始输出（DAgger 附加） |
| `control_source` | (1,) | 0=policy, 1=human（DAgger 附加） |

**录制策略**：占位 + 后处理
1. `add_frame(t)` 时 `action[t] = state[t]`（临时占位）
2. `finish_episode()` 前调用 `_postprocess_episode_actions()`：`action[t] = state[t+1]`，最后一帧保持不变

**生命周期**：
```
DAggerRecorder() → start_episode() → add_frame() x N → finish_episode() → ... → close()
                                                        discard_episode()
```

---

### 4.6 GripperAsyncController (`core/gripper_controller.py`)

> 异步夹爪控制器，管理 RM+ 生态夹爪的生命周期。

**双子系统**（Realman 夹爪）：

| 子系统 | 控制接口 | 读取接口 | 延迟 |
|--------|----------|----------|------|
| 标准 Modbus | `rm_set_gripper_position()` | `rm_get_gripper_state()` | ~1000ms |
| RM+ 生态 | JSON TCP `set_gripper_position` | UDP `rm_plus_state.pos[0]` | ~5ms |

⚠️ **BUG-018/019 教训**：控制和读取必须用同一子系统。DAgger 使用 RM+ JSON 控制 + UDP RM+ 读取。

---

### 4.7 AsyncInferenceClient (`async_inference_client.py`)

> 纯策略推理客户端，不依赖 ROS2。适合快速测试策略效果。

**与 DAggerNode 的区别**：

| 特性 | AsyncInferenceClient | DAggerNode |
|------|---------------------|------------|
| ROS2 依赖 | ❌ | ✅ |
| VR 接管 | ❌ | ✅ |
| 录制 | ✅（内置） | ✅（ApproximateTimeSynchronizer） |
| 机器人连接 | 直接管理 | 通过 driver_node |
| 相机 | 直接管理 | 通过 camera_node |
| Web UI | ❌ | ✅ |
| 适用场景 | 快速策略验证 | 完整 DAgger 工作流 |


---

## 5. 状态机

DAggerNode 使用三态状态机管理控制模式：

```
                    enable_control (policy=true)
            ┌──────────────────────────────────────┐
            │                                      ▼
         ┌──────┐                           ┌──────────┐
         │ IDLE │                           │  POLICY  │
         └──┬───┘                           └──┬───┬───┘
            │                                  │   │
            │  enable_control (policy=false)    │   │ VR trigger
            │                                  │   │ press
            ▼                                  │   ▼
         ┌──────┐    VR trigger release     ┌──┴──────┐
         │      │◄──────────────────────────│  HUMAN  │
         │ IDLE │    disable_control        │         │
         │      │◄──────────────────────────│         │
         └──────┘                           └─────────┘
```

### 状态说明

| 状态 | 控制输出 | 推理 | VR | 录制 |
|------|----------|------|-----|------|
| **IDLE** | 无 | 停止 | 忽略 | 不可用 |
| **HUMAN** | `/rm/action_pose`（笛卡尔） | 后台运行（shadow tick 消费 buffer） | 活跃 | 记录 control_source=1 |
| **POLICY** | `/rm/action_joint_state`（关节） | 活跃 | 监听 trigger | 记录 control_source=0 |

### VR 接管机制

POLICY 模式下，VR trigger 按下触发切换到 HUMAN：

1. **trigger press** (`> threshold 0.85`):
   - 停止 policy exec loop
   - 暂停推理（`inference_bridge.pause()`）
   - 清空 ring buffer
   - 切换到 HUMAN 模式
   - 重置 VR 状态（`is_active=False, is_initialized=False`）

2. **trigger release** (`< threshold`):
   - 清空 ring buffer（丢弃过时 action）
   - 恢复推理（`inference_bridge.resume()`）
   - 切换到 POLICY 模式
   - 重启 policy exec loop
   - 清除 `_last_executed_joints`（安全检查基准重置）

### Shadow Tick

HUMAN 模式下，`_tick_shadow()` 持续消费 ring buffer 中的 policy action：
- 仅更新 `_last_policy_action` 用于录制
- 不发送给机器人
- 确保录制数据中 `policy_action` 字段始终有值

---

## 6. 线程模型与锁策略

### 6.1 线程清单

| 线程 | 名称 | 生命周期 | 职责 |
|------|------|----------|------|
| ROS2 Executor | 主线程 | 节点全生命周期 | 所有 ROS2 回调（订阅、定时器、Service） |
| 推理线程 | `inference_loop` | connect → disconnect | gRPC 通信 + ring buffer 更新 |
| 执行线程 | `policy_exec_loop` | POLICY 模式期间 | 固定频率从 ring buffer 取 action 并发布 |
| Server 探测 | `server_probe` | 节点全生命周期 | 每 2s 探测 PolicyServer 可达性 |
| Recorder 初始化 | `RecorderPreInit` / `RecorderEagerInit` | 一次性 | 后台初始化 DAggerRecorder |

### 6.2 锁清单

| 锁 | 保护对象 | 持有者 |
|----|----------|--------|
| `_vr_lock` | VR 状态（pose, trigger, grip, is_active） | VR 回调、_tick_human |
| `_mode_lock` | `_mode` 状态机 | Service handler、tick、exec loop |
| `_state_lock` | 机械臂缓存状态（joints, eef） | state 回调、_get_state_14d |
| `_image_lock` | 相机图像缓存 | image 回调、_get_cached_images |
| `_recorder_lock` | DAggerRecorder 实例 | recording sync、episode 操作 |
| `_cmd_ts_lock` | 控制频率时间戳 | action 发布、_get_control_hz |
| `_obs_lock` (bridge) | 观测快照 | update_observation、inference_loop |
| `_lock` (ring_buffer) | 缓冲区 deque | update、pop_current、clear |

### 6.3 锁获取顺序

```
vr_lock → mode_lock     ✅ 安全
vr_lock → state_lock    ✅ 安全
mode_lock → recorder_lock ✅ 安全（stop_session 中）
```

⚠️ **重要**：当前假设 ROS2 单线程 executor。如果切换到 `MultiThreadedExecutor`，必须审计锁顺序以防死锁。

### 6.4 执行线程详解

Policy exec loop 是一个独立线程，模仿 `async_inference_client.run()` 的 while+sleep 模式：

```python
while not stop_event:
    t_start = monotonic()
    
    # 1. 检查模式（如果不再是 POLICY，退出）
    with mode_lock:
        if mode != POLICY: break
    
    # 2. 更新观测（喂给推理线程）
    update_inference_observation()
    
    # 3. 从 ring buffer 取 action
    action = ring_buffer.pop_current()
    if action is None and hold_on_empty:
        action = last_action
    
    # 4. 执行 action（发布 ROS2 消息）
    publish_joint_action(action)
    
    # 5. 频率控制
    sleep(T_step - elapsed)
```

**设计原因**：观测更新和 action 执行在同一线程中，确保时序一致。之前的设计将观测更新放在 50Hz timer 中，导致推理基于过时观测。


---

## 7. 数据流

### 7.1 观测数据流

```
RealSense Camera x2                    Realman Driver Node
  /camera/{name}/color/image_raw          /rm/state_joint_state (14D)
         │                                        │
         ▼                                        ▼
  ┌─ DAggerNode ──────────────────────────────────────────────┐
  │                                                            │
  │  _on_color_image()          _on_arm_state()               │
  │    _cached_images[cam] = img   _current_joints = [7]      │
  │    (with _image_lock)          _current_eef = [6]         │
  │                                _current_gripper = float    │
  │                                (with _state_lock)          │
  │         │                              │                   │
  │         └──────────┬───────────────────┘                   │
  │                    ▼                                       │
  │  _update_inference_observation()                           │
  │    state_14d = _get_state_14d()                            │
  │    images = _get_cached_images()                           │
  │    bridge.update_observation(state_14d, images)            │
  │                    │                                       │
  └────────────────────┼───────────────────────────────────────┘
                       ▼
  ┌─ InferenceBridge ──────────────────────────────────────────┐
  │  _obs_state_14d = state_14d     (with _obs_lock)          │
  │  _obs_images = images                                      │
  │  _obs_updated.set()  ──────►  inference_loop 被唤醒        │
  │                                                            │
  │  inference_loop:                                           │
  │    obs = UnifiedObservation(state_14d, images)             │
  │    bytes = pickle.dumps(obs)                               │
  │    response = stub.Infer(bytes)  ──► PolicyServer (gRPC)   │
  │    action_chunk = pickle.loads(response)                   │
  │    ring_buffer.update(action_chunk.actions)                │
  └────────────────────────────────────────────────────────────┘
```

**14D 状态格式**：
```
index:  [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]      [8]   [9]   [10]  [11]  [12]  [13]
field:  j1   j2   j3   j4   j5   j6   j7   gripper   ex    ey    ez    erx   ery   erz
unit:   rad  rad  rad  rad  rad  rad  rad  0~1       m     m     m     rad   rad   rad
```

### 7.2 动作执行流

#### HUMAN 模式（VR 遥操）

```
VR Controller
  /vr/{side}/pose ──► _on_vr_pose()
  /vr/{side}/trigger ──► _on_vr_trigger()
  /vr/{side}/grip ──► _on_vr_grip()
         │
         ▼
  _tick_human() (50Hz timer)
    1. 检查 VR 数据超时（0.5s）
    2. trigger > threshold → 激活 VR 控制
       - 首次激活: _sync_arm_state() 锚定当前位姿
       - 计算增量: delta = vr_pose - vr_anchor
       - 坐标变换: VRCoordinateTransform.apply()
       - 低通滤波: PoseFilter.filter()
       - 目标位姿: target = arm_anchor + filtered_delta
    3. trigger < threshold → 停用 VR
       - _sync_arm_state() 重新锚定
    4. 发布 /rm/action_pose (PoseStamped)
```

#### POLICY 模式（策略推理）

```
  policy_exec_loop (独立线程, f_exec Hz)
    1. _update_inference_observation()  → 喂观测
    2. action = ring_buffer.pop_current()
       - 有 action: 使用新 action
       - 无 action + hold_on_empty: 重发 last_popped
       - 无 action + !hold: 跳过
    3. _safety_check_joints(action)
       - 逐关节检查 delta < max_joint_delta_deg
       - 超限: 裁剪到安全范围，计数 safety_rejects
    4. _publish_joint_action(joints_rad, gripper)
       → /rm/action_joint_state (JointState)
```

### 7.3 录制数据流

```
  ApproximateTimeSynchronizer
    订阅: /rm/state_joint_state + /camera/*/color/image_raw
    时间戳对齐后触发 _on_recording_sync()
         │
         ▼
  _on_recording_sync()
    1. 前置检查: episode_active? paused? warmup_done?
    2. 频率限制: min_frame_interval (recording_fps)
    3. 从 sync 消息直接提取:
       - state_14d (14D) ← state_msg.position[:14]
       - images_rgb {} ← CvBridge.imgmsg_to_cv2(image_msgs)
    4. control_source = 1 (HUMAN) 或 0 (POLICY)
    5. policy_action = _last_policy_action 或 zeros
    6. recorder.add_frame(state_14d, images_rgb, policy_action, control_source)
```

**为什么用 ApproximateTimeSynchronizer 而不是缓存？**
- 确保 state 和 images 时间戳对齐（<33ms 容差）
- 避免 state 和 image 来自不同时刻的不一致
- ROS2 message_filters 原生支持，无需手动管理同步逻辑

---

## 8. 会话生命周期

### 8.1 start_session 流程

```
用户点击 "开始会话" (Web UI)
    │
    ▼
/dagger/start_session (Service)
    │
    ├─ Step 1: 启用 driver follow（真机模式）
    │   └─ /driver/enable_follow
    │
    ├─ Step 2: 连接 PolicyServer（后台线程）
    │   ├─ enable_policy=true:
    │   │   ├─ 立即切换到 POLICY 模式
    │   │   └─ 后台线程: connect() → start_inference_loop()
    │   │       → 等待 warmup → clear_buffer() → start_policy_exec_loop()
    │   └─ enable_policy=false:
    │       └─ 直接进入 HUMAN 模式（纯 VR 遥操）
    │
    └─ Step 3: 自动开始录制
        ├─ recorder 已初始化: start_episode()
        ├─ recorder 预初始化中: 后台等待完成后 start_episode()
        └─ recorder 未初始化: 启动 _init_recorder_eager() 后台初始化
```

### 8.2 stop_session 流程

```
用户点击 "停止会话" (Web UI)
    │
    ▼
/dagger/stop_session (Service)
    │
    ├─ Step 0: 立即设置 _policy_exec_stop（最高优先级）
    │   └─ 执行线程在 ~33ms 内停止发送 action
    │
    ├─ Step 1: 暂停录制（不保存不丢弃，等前端确认）
    │   └─ episode_paused = True
    │
    ├─ Step 2: 切换到 IDLE
    │   └─ 重置 VR 状态、清除 last_executed_joints
    │
    └─ Step 3: 后台清理
        ├─ inference_bridge.stop_inference_loop()
        └─ /driver/disable_follow（真机模式）
```

### 8.3 pause_session / resume_session

**暂停**：停止 action 执行 + 暂停推理，但保持 gRPC 连接。允许用户在暂停期间：
- 保存/丢弃录制
- 回 Home 位
- 开始新 Episode

**恢复**：清空 buffer（丢弃暂停期间的旧 action）→ 恢复推理 → 重启执行线程。

### 8.4 Recorder 初始化策略

DAggerRecorder 初始化需要 camera_shapes 和 policy_action_dim，这两个信息在节点启动时不一定可用。系统提供三种初始化路径：

| 策略 | 触发时机 | 等待条件 | 优先级 |
|------|----------|----------|--------|
| **预初始化** (`_preinit_recorder`) | 节点启动后立即 | 等待第一帧图像 | 最高（减少首次 start_session 延迟） |
| **主动初始化** (`_init_recorder_eager`) | start_session 时 | 等待 warmup + 图像 | 中（确保 action_dim 正确） |
| **懒初始化** (`_init_recorder_from_sync`) | 首次 sync 回调 | 无（sync 消息自带图像） | 最低（兜底） |


---

## 9. 录制系统

### 9.1 录制架构

```
  ┌─ message_filters.ApproximateTimeSynchronizer ─────────────┐
  │  订阅:                                                     │
  │    /rm/state_joint_state (JointState, 14D)                │
  │    /camera/cam0/color/image_raw (Image)                   │
  │    /camera/cam1/color/image_raw (Image)                   │
  │  slop: 0.033s (33ms 容差)                                 │
  │  queue_size: 10                                            │
  └──────────────────────┬────────────────────────────────────┘
                         │ 时间戳对齐后触发
                         ▼
  _on_recording_sync(state_msg, *image_msgs)
    │
    ├─ 前置检查: episode_active, !paused, warmup_done
    ├─ 频率限制: recording_fps (默认 30Hz)
    ├─ 提取 state_14d + images_rgb
    ├─ 确定 control_source (HUMAN=1 / POLICY=0)
    ├─ 获取 policy_action (_last_policy_action 或 zeros)
    │
    └─► DAggerRecorder.add_frame(state, images, policy_action, control_source)
              │
              ├─ observation.state = state_14d
              ├─ action = state_14d (占位，后处理替换)
              ├─ observation.images.* = images
              ├─ policy_action = policy_action
              └─ control_source = control_source
```

### 9.2 Episode 生命周期

```
start_episode(task)
    │
    ├─ 录制帧... (add_frame x N)
    │   ├─ POLICY 模式: control_source=0, policy_action=实际推理输出
    │   └─ HUMAN 模式: control_source=1, policy_action=shadow tick 消费的值
    │
    ├─ [可选] pause_episode → resume
    │
    └─ finish_episode()
    │   ├─ _postprocess_episode_actions(): action[t] = state[t+1]
    │   └─ 保存到 LeRobot v2.1 数据集
    │
    └─ discard_episode()
        └─ 丢弃所有帧，不保存
```

### 9.3 数据集输出格式

```
{dataset_root}/{repo_id}/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet    # 状态 + action + policy_action + control_source
│       ├── episode_000001.parquet
│       └── ...
├── videos/
│   └── chunk-000/
│       ├── observation.images.cam0_rgb/
│       │   ├── episode_000000.mp4
│       │   └── ...
│       └── observation.images.cam1_rgb/
│           ├── episode_000000.mp4
│           └── ...
├── meta/
│   ├── info.json                     # 数据集元信息
│   ├── episodes.jsonl                # Episode 列表
│   ├── stats.json                    # 统计信息
│   └── tasks.jsonl                   # 任务描述
└── README.md
```

---

## 10. Web UI 控制面板

### 10.1 架构

```
  ┌─ DAggerControlPanelNode (ROS2) ──────────────────────────┐
  │  订阅:                                                    │
  │    /dagger/status (JSON String)  → 缓存 DAgger 状态      │
  │    /rm/state_joint_state         → Driver 心跳监控        │
  │    /vr/right/pose                → VR Input 心跳监控      │
  │    /camera/*/color/image_raw     → 相机预览 (30fps 限流)  │
  │    /rosout                       → 日志收集               │
  │                                                           │
  │  TopicMonitor x N: 统计各话题消息频率 (2s 滑动窗口)       │
  └──────────────────────────────────────────────────────────┘
                         │
                         │ get_full_status()
                         ▼
  ┌─ Flask (port 5002, daemon thread) ───────────────────────┐
  │                                                           │
  │  GET  /              → 单页 HTML (内嵌 CSS + JS)         │
  │  GET  /api/status    → JSON 完整状态                      │
  │  POST /api/set_idle          → 代理 /dagger/set_idle     │
  │  POST /api/set_human         → 代理 /dagger/set_human    │
  │  POST /api/set_policy        → 代理 /dagger/set_policy   │
  │  POST /api/start_session     → 代理 /dagger/start_session│
  │  POST /api/stop_session      → 代理 /dagger/stop_session │
  │  POST /api/pause_session     → 代理 /dagger/pause_session│
  │  POST /api/resume_session    → 代理 /dagger/resume_session│
  │  POST /api/start_episode     → 代理 /dagger/start_episode│
  │  POST /api/stop_episode      → 代理 /dagger/stop_episode │
  │  POST /api/pause_episode     → 代理 /dagger/pause_episode│
  │  POST /api/discard_episode   → 代理 /dagger/discard_episode│
  │  POST /api/new_episode       → 代理 /dagger/new_episode  │
  │  POST /api/move_to_home      → 代理 /driver/move_to_home │
  │                                                           │
  └──────────────────────────────────────────────────────────┘
                         │
                         │ 30fps 轮询 /api/status
                         ▼
  ┌─ 浏览器前端 ─────────────────────────────────────────────┐
  │  ┌─────────────┐ ┌──────────────┐ ┌──────────────────┐  │
  │  │ 节点状态     │ │ 模式控制      │ │ 会话控制          │  │
  │  │ DAgger ●    │ │ [IDLE]       │ │ [开始会话]        │  │
  │  │ Driver ●    │ │ [HUMAN]      │ │ [停止会话]        │  │
  │  │ VR Input ○  │ │ [POLICY]     │ │ [暂停] [恢复]     │  │
  │  │ Cam0 ●     │ │              │ │ [回 Home]         │  │
  │  │ Cam1 ●     │ │              │ │                   │  │
  │  └─────────────┘ └──────────────┘ └──────────────────┘  │
  │  ┌──────────────────────────────────────────────────────┐│
  │  │ 推理状态: warmup ✅ | buffer: 12 | infers: 47       ││
  │  │ Server: connected ✅ | policy: act                   ││
  │  │ 录制: Episode 3 | 帧: 245 | ● REC                   ││
  │  │ 控制频率: 20.1 Hz (目标 20.0)                         ││
  │  └──────────────────────────────────────────────────────┘│
  │  ┌─────────────────┐ ┌────────────────────────────────┐ │
  │  │ 相机预览         │ │ 日志                            │ │
  │  │ [cam0] [cam1]   │ │ 12:34:56 [dagger] Mode: POLICY │ │
  │  │                 │ │ 12:34:57 [driver] Follow ON     │ │
  │  └─────────────────┘ └────────────────────────────────┘ │
  └──────────────────────────────────────────────────────────┘
```

### 10.2 状态 JSON 结构

`/dagger/status` 发布的 JSON 包含以下字段：

```json
{
  "mode": "POLICY",
  "session_active": true,
  "session_paused": false,
  "vr_active": false,
  "cmd_count": 1234,
  "safety_rejects": 2,
  "control": {
    "actual_hz": 20.1,
    "target_hz": 20.0,
    "source": "POLICY"
  },
  "inference": {
    "warmup_done": true,
    "paused": false,
    "buffer_size": 12,
    "infer_count": 47
  },
  "server": {
    "connected": true,
    "ready": true,
    "policy_type": "act",
    "pretrained_path": "/path/to/model"
  },
  "recording": {
    "episode_active": true,
    "episode_paused": false,
    "num_episodes": 3,
    "frame_count": 245,
    "recorder_ready": true
  }
}
```

---

## 11. 配置参数参考

### 11.1 机器人与硬件

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `robot_ip` | `192.168.1.18` | Realman 机械臂 IP |
| `robot_port` | `8080` | 机械臂端口 |
| `rm_plus_baud` | `115200` | RM+ 生态波特率 |
| `dry_run` | `false` | 干跑模式（不连接真机） |

### 11.2 策略推理

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enable_policy` | `true` | 启用策略推理 |
| `server_address` | `127.0.0.1:50051` | PolicyServer gRPC 地址 |
| `policy_type` | `act` | 策略类型（act/diffusion/pi0/pi0fast/smolvla/openpi/octo） |
| `pretrained_path` | - | 预训练模型路径 |
| `task` | - | 任务描述文本 |
| `f_exec` | `20.0` | 执行频率 Hz |
| `n_action_steps` | `0` (auto) | 每次推理使用的 action 步数 |
| `T_inter` | `0.0` (auto) | 推理间隔秒数 |
| `t_inf` | `0.2` | 实测推理延迟秒数 |
| `chunk_size_threshold` | `0` (auto) | buffer 安全阈值 |
| `hold_on_empty` | `true` | buffer 空时重发最后一个 action |

### 11.3 VR 控制

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `control_hz` | `30.0` | VR 控制频率 Hz |
| `side` | `right` | VR 手柄侧（left/right） |
| `trigger_threshold` | `0.85` | 扳机激活阈值 |
| `rotation_preset` | `perm_xyz_pnn` | VR→机械臂旋转映射 |
| `alpha_pos` | `0.8` | 位置低通滤波系数 |
| `alpha_rot` | `0.8` | 旋转低通滤波系数 |

### 11.4 相机

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `camera_names` | `[cam0_rgb, cam1_rgb]` | 推理用相机 key |
| `camera_ros_names` | `[cam0, cam1]` | ROS2 话题名 |
| `camera_rotation_names` | `[cam0_rgb]` | 需要旋转的相机 |
| `camera_rotation_degrees` | `[180]` | 旋转角度 |

### 11.5 录制

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enable_recording` | `true` | 启用录制 |
| `repo_id` | `local/dagger_realman_001` | 数据集 ID |
| `dataset_root` | `.../dagger/data` | 数据集根目录 |
| `recording_task` | `pick and place` | 录制任务描述 |
| `recording_fps` | `30` | 录制帧率 |
| `use_videos` | `true` | 使用视频编码（vs 原始图像） |
| `image_writer_threads` | `4` | 视频编码线程数 |
| `max_episodes` | `0` (无限) | 最大 episode 数 |

### 11.6 安全

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_joint_delta_deg` | `5.0` | 帧间最大关节角度变化（度） |
| `max_pose_delta_m` | `0.02` | 帧间最大笛卡尔位移（米，预留未强制） |

### 11.7 Web UI

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `web_ui.enable` | `true` | 启用 Web UI |
| `web_ui.port` | `5002` | Web UI 端口 |


---

## 12. 关键设计决策

### 12.1 为什么 DAggerNode 不直接连接机器人？

DAggerNode 通过 ROS2 话题与 `realman_driver_node` 通信，而不是直接调用 SDK：

- **职责分离**：driver_node 封装所有硬件通信（TCP/UDP/Modbus），DAggerNode 只关注控制逻辑
- **安全性**：driver_node 内置关节限位、碰撞检测等安全机制
- **复用性**：同一个 driver_node 可以被 VR 遥操、策略推理、手动控制等多种上层节点使用
- **调试便利**：可以用 `ros2 topic echo` 直接观察 action 输出，无需修改代码

### 12.2 为什么 InferenceBridge 从 AsyncInferenceClient 提取？

Phase 1~2 的 `AsyncInferenceClient` 是一个"全包"客户端（机器人连接 + 相机 + 推理 + 录制）。Phase 3 引入 ROS2 后：

- 机器人连接由 driver_node 管理
- 相机由 camera_node 管理
- 录制由 ApproximateTimeSynchronizer 驱动

因此提取出纯推理层 `InferenceBridge`，只负责：观测 → gRPC → ring buffer。这使得 DAggerNode 可以灵活组合 VR 控制和策略推理，而不受硬件管理代码的耦合。

### 12.3 为什么用 ApproximateTimeSynchronizer 做录制同步？

替代方案是在执行循环中直接缓存 state + images 然后录制。选择 ApproximateTimeSynchronizer 的原因：

- **时间戳对齐**：确保 state 和 images 来自同一时刻（<33ms 容差），避免数据不一致
- **解耦执行和录制**：执行循环专注于 action 发布，录制由独立回调驱动
- **帧率独立**：执行频率（20Hz）和录制频率（30Hz）可以不同
- **ROS2 原生**：message_filters 是 ROS2 标准库，无需手动管理同步逻辑

### 12.4 为什么 action[t] = state[t+1]？

这是 LeRobot v2.1 的标准约定，也是 VR 遥操录制使用的格式：

- `action[t]` 表示"在时刻 t 应该执行的动作，使机器人到达时刻 t+1 的状态"
- 训练时，模型学习的是 `observation[t] → action[t]`，即"看到当前状态，预测下一步应该到达的状态"
- 录制时先用 `state[t]` 占位，episode 结束后后处理替换为 `state[t+1]`

### 12.5 为什么 HUMAN 模式下保留 Shadow Tick？

HUMAN 模式下，推理线程仍在后台运行（只是 pause 了），ring buffer 中可能残留 action。Shadow tick 的作用：

- **录制完整性**：确保 `policy_action` 字段在 HUMAN 帧中也有值（虽然不用于控制）
- **平滑切换**：HUMAN → POLICY 切换时，推理线程已经在运行，只需 resume + 清空 buffer
- **分析价值**：事后可以对比 HUMAN 帧中"策略本来会做什么" vs "人类实际做了什么"

### 12.6 为什么 Policy Exec Loop 是独立线程而非 ROS2 Timer？

早期设计使用 ROS2 Timer（50Hz）驱动 action 执行。改为独立线程的原因：

- **观测-执行一致性**：观测更新和 action 执行在同一线程中，确保推理基于最新观测
- **频率精度**：ROS2 Timer 受 executor 调度影响，实际频率可能波动；独立线程用 `time.sleep()` 控制更精确
- **生命周期管理**：线程可以在 POLICY → HUMAN 切换时干净地停止和重启，Timer 需要额外的 enable/disable 逻辑

### 12.7 Recorder 三级初始化策略的权衡

| 策略 | 优点 | 缺点 |
|------|------|------|
| 预初始化 | 首次 start_session 零延迟 | policy_action_dim 可能不准确（默认 8） |
| 主动初始化 | action_dim 准确（等 warmup） | start_session 有 1~3s 延迟 |
| 懒初始化 | 最简单，兜底保障 | 首帧可能丢失 |

实际运行中，预初始化在节点启动后立即开始，大多数情况下在用户点击"开始会话"前已完成。如果 warmup 后发现 action_dim 不匹配，会重新初始化。

### 12.8 pickle 序列化与跨进程兼容

gRPC 通信使用 `pickle.dumps()` 序列化 `UnifiedObservation` 和 `ActionChunk`。关键约束：

- `pickle.dumps()` 记录类的完整模块路径（如 `lerobot.extensions.unified_deploy.core.data_types.UnifiedObservation`）
- 接收端必须能 `import` 同一模块路径，否则 `pickle.loads()` 报 `No module named 'xxx'`
- 因此 `InferenceBridge` 从原始模块导入数据类型，而不是从 `dagger/deps/` 的本地副本导入

---

## 📎 相关文档

| 文档 | 说明 |
|------|------|
| [快速上手指南](quick-start-guide.md) | 环境配置、首次启动、命令速查 |
| [部署指南](deployment-guide.md) | 完整部署流程、配置详解、故障排除 |
| [异步推理系统详解](async-inference-system-explained.md) | T_inter / K_inf / chunk_size 参数推导公式 |
| [踩坑记录](common-pitfalls.md) | BUG-001~020 + PITFALL 系列，含根因分析和修复方案 |
| [Phase 1 测试指南](phase1_testing_guide.md) | Phase 1 异步推理框架测试 |
| [真机测试指南](real_machine_testing_guide.md) | 真机部署测试流程 |

---

> 📝 本文档最后更新: 2025-02  
> 📝 对应代码版本: Phase 3C (dagger_node.py 2430 lines, inference_bridge.py 661 lines)
