# 异步推理系统原理解析

> 维护文件：每次有新的问题解答都会追加到本文档中。
> 最后更新：2026-02-09 (v2 — 修正 K_inf 裁剪语义、warmup 多次测量、安全余量)

---

## 目录

1. [系统架构概览](#1-系统架构概览)
2. [核心流程详解](#2-核心流程详解)
3. [参数自动推导公式](#3-参数自动推导公式)
4. [Warmup 机制](#4-warmup-机制)
5. [K_inf 裁剪原理](#5-kinf-裁剪原理)
6. [两种运行模式](#6-两种运行模式)
7. [真机测试分析 (2026-02-09)](#7-真机测试分析-2026-02-09)
8. [常见问题 FAQ](#8-常见问题-faq)

---

## 1. 系统架构概览

```
┌─────────────────────────────────────────────────────────┐
│                    AsyncInferenceClient                  │
│                                                          │
│  ┌──────────────────┐       ┌──────────────────────┐    │
│  │   推理线程        │       │   执行循环 (主线程)    │    │
│  │  _inference_loop  │       │   run() 内的 while    │    │
│  │                   │       │                       │    │
│  │  1. 采集观测      │       │  固定 f_exec Hz       │    │
│  │  2. gRPC 推理     │  ───► │  从 ring_buffer pop   │    │
│  │  3. K_inf 裁剪    │       │  执行 action          │    │
│  │  4. 写入 buffer   │       │  buffer 空 → hold     │    │
│  │  5. 等待 T_inter  │       │                       │    │
│  └──────────────────┘       └──────────────────────┘    │
│              │                        │                   │
│              ▼                        ▼                   │
│       ┌─────────────────────────────────┐                │
│       │     ActionRingBuffer            │                │
│       │  update() ← 推理线程写入        │                │
│       │  pop_current() ← 执行线程读取   │                │
│       │  clear + extend (替换语义)      │                │
│       └─────────────────────────────────┘                │
└─────────────────────────────────────────────────────────┘
```

**两个线程完全异步**：
- **执行循环**（主线程）：以固定 `f_exec` Hz 频率运行，每帧从 `ring_buffer` 取一个 action 发给机械臂。buffer 空时重复最后一个 action（hold）。
- **推理线程**（后台）：异步调用 gRPC PolicyServer 获取 ActionChunk，裁剪后写入 `ring_buffer`。

**关键设计**：`ring_buffer.update()` 是**替换语义**——每次推理完成后，`clear()` 旧 buffer 再 `extend()` 新 actions。这意味着旧的未执行完的 actions 会被丢弃，因为新推理基于更新的观测，旧 actions 已过时。

---

## 2. 核心流程详解

### 2.1 ActionChunk 的时间语义

**关键事实**：模型在观测时刻 T 采集观测，返回的 ActionChunk 中 `action[0]` 对应的就是时刻 T 应该执行的动作。

```python
# Server 端（所有 adapter 统一行为）
for i in range(n_actions):
    actions.append(UnifiedAction(
        timestep=observation.timestep + i,  # action[0] = 观测时刻
        ...
    ))
```

这意味着：
- `action[0]` = 观测时刻 T 的动作
- `action[1]` = T+1 步的动作
- `action[K_inf]` = T+K_inf 步的动作（推理完成时刻对应的动作）

### 2.2 推理线程一次完整循环（含 T_inter 等待）

```
上次buffer更新          观测采集(T)        推理完成         下次循环开始
      │                    │                  │                  │
      │◄── T_inter 等待 ──►│◄── t_inf 推理 ──►│                  │
      │                    │                  │                  │
      │  执行线程消耗旧chunk │  执行线程消耗旧chunk│                  │
      │  K_inter 个action  │  K_inf 个action  │                  │
      │                    │                  │                  │
                           │                  │
                    模型基于此刻观测      action[0..K_inf-1] 已过时
                    返回 action[0..L-1]  从 action[K_inf] 开始执行
```

**为什么只裁剪 K_inf 而不是 K_inter + K_inf？**

因为 T_inter 等待期间消耗的是**旧 chunk** 的 action（从旧 buffer pop）。新 chunk 的时间起点是观测时刻 T，不是上次 buffer 更新时刻。所以新 chunk 只需要裁剪推理延迟期间过去的 K_inf 个 action。

### 2.3 详细步骤

1. **等待 T_inter**（或安全网触发）：期间执行线程消耗旧 buffer 中的 action
2. **采集观测** `get_observation()`：读取相机图像 + 关节角 + 末端位姿 + 夹爪
3. **gRPC 推理** `_grpc_infer()`：序列化观测 → 发送到 PolicyServer → 等待返回 ActionChunk
4. **n_action_steps 截取**：如果 `n_action_steps > 0`，只取前 N 个 action
5. **K_inf 裁剪**：计算从采集观测到收到结果经过的实际时间 `t_inf_actual`，换算成步数 `K_inf_actual = int(t_inf_actual * f_exec) + 2`（+2 安全余量），跳过前 K_inf 个 action
6. **写入 buffer** `ring_buffer.update(valid_actions)`：替换整个 buffer
7. **回到步骤 1**

### 2.4 执行循环一次完整循环

```python
while not shutdown:
    action = ring_buffer.pop_current()  # 取一个 action
    if action:
        execute_action(action)          # 正常执行
    elif hold_on_empty:
        execute_action(last_popped)     # hold: 重复上一个 action
    sleep(T_step - elapsed)             # 帧率控制
```

---

## 3. 参数自动推导公式

### 3.1 输入参数

| 参数 | 来源 | 说明 |
|------|------|------|
| `L` | warmup 时从模型获取 | 模型原始 chunk_size（如 50） |
| `f_exec` | YAML 配置 | 执行频率 Hz（如 30） |
| `t_inf` | warmup 多次测量 或 YAML 配置 | 推理延迟（秒） |
| `n_action_steps` | YAML 配置 | 截取步数（0=全部） |

### 3.2 推导过程

```
T_step = 1 / f_exec                          # 每步时间间隔

L_eff = min(n_action_steps, L)  if n_action_steps > 0
      = L                       if n_action_steps == 0

K_inf = ceil(t_inf * f_exec)                  # 推理延迟消耗的步数

L_valid = L_eff - K_inf                       # 每次推理的有效 action 数

T_inter_max = L_eff * T_step - t_inf          # 最大允许推理间隔
            = (L_eff / f_exec) - t_inf

T_inter = T_inter_max * 0.8                   # 自动推导（20% 安全余量）
        = min(config.T_inter, T_inter_max)    # 用户指定时 clamp

chunk_size_threshold = max(2, K_inf)          # buffer 安全网阈值
```

### 3.3 稳定性约束

系统可行的条件：
```
L_valid > 0  且  T_inter_max > 0
```

即：**每次推理产生的有效 action 数必须 > 0，且推理间隔有正的可用空间**。

如果不满足（例如 t_inf 太大），系统降级为被动等待模式。

### 3.4 直觉理解

想象一个水龙头（推理线程）往水桶（buffer）里加水，水桶底部有个恒速漏洞（执行线程以 f_exec 消耗）：

- `L_eff` = 每次加水量
- `K_inf` = 加水过程中漏掉的水
- `L_valid` = 实际净增水量
- `T_inter` = 两次加水之间的等待时间
- 稳定性 = 水桶不能漏干（否则 hold/卡顿）

**T_inter_max 的含义**：如果等待时间超过 T_inter_max，buffer 会在下次推理完成前被消耗完，导致 hold。

---

## 4. Warmup 机制

### 4.1 什么是 Warmup？

Warmup 是推理线程启动时执行的**多次预热推理**，目的有两个：

1. **测量 t_inf**：多次推理取稳定平均值（丢弃第 1 次冷启动）
2. **获取 chunk_size**：记录模型返回的 ActionChunk 长度（`_model_chunk_size`）

### 4.2 Warmup 流程（v2 — 多次测量）

```python
def _measure_inference_time(self, n_warmup=5):
    # 如果 config.t_inf > 0，直接使用配置值（仍执行 1 次获取 chunk_size）
    if config.t_inf > 0:
        _warmup_single()  # 只为获取 chunk_size
        return config.t_inf

    # 执行 n_warmup 次推理
    t_inf_list = []
    for i in range(n_warmup):
        result = _warmup_single()
        if result:
            t_inf_list.append(result.t_inf)

    # 丢弃第 1 次（冷启动），取后面的平均值
    t_inf_cold = t_inf_list[0]        # 如 3.4s（丢弃）
    t_inf_stable = t_inf_list[1:]     # 如 [0.10, 0.12, 0.09, 0.11]
    t_inf = mean(t_inf_stable)        # 如 0.105s
```

### 4.3 Warmup 使用真实数据

Warmup 用的是**真实的相机图像和关节状态**（`get_observation()`），不是虚假数据。只是此时执行循环还没开始，机械臂不会执行返回的 action。最后一次 warmup 的结果会写入 buffer，供执行循环启动后立即使用。

### 4.4 config.t_inf = 0 时的行为

当 YAML 中 `t_inf: 0.0` 时：

1. `_measure_inference_time()` 执行 5 次推理
2. 丢弃第 1 次（GPU 冷启动，可能 3+ 秒）
3. 取后 4 次平均值（如 0.105s）
4. 用这个稳定值去推导参数

### 4.5 旧版 Warmup 的问题（已修复）

旧版只执行**一次**推理来测量 t_inf。如果遇到 GPU 冷启动（首次推理 3.4s），这个值会被直接用于参数推导，导致 K_inf=102 > chunk_size=50，系统判定 FAIL 并降级。

**修复方案**：多次测量（默认 5 次），丢弃第 1 次冷启动，取后 4 次平均值。即使部分推理失败，也能用成功的结果计算。

---

## 5. K_inf 裁剪原理

### 5.1 为什么需要 K_inf 裁剪？

模型在观测时刻 T 采集观测，返回的 ActionChunk 是：
```
actions = [a_0, a_1, a_2, ..., a_{L-1}]
```

其中 `a_0` 对应观测时刻 T，`a_1` 对应 T+1 步，以此类推。

从采集观测到收到结果，经过了 `t_inf` 时间。在这段时间里，执行线程已经消耗了 `K_inf` 步（消耗的是旧 buffer 中的 action）。当新 chunk 到达时，时间已经走到了 T+K_inf，所以 `a_0` 到 `a_{K_inf-1}` 对应的时间点已经过去了，执行它们会导致动作滞后（"回到过去"）。

### 5.2 裁剪逻辑（含 +2 安全余量）

```python
# async_inference_client.py 行 760-766
t_inf_actual = time.monotonic() - t_start      # 实际推理耗时
K_inf_actual = int(t_inf_actual * f_exec) + 2  # +2 安全余量

if K_inf_actual < len(actions):
    valid_actions = actions[K_inf_actual:]       # 跳过已过时的
else:
    valid_actions = actions[-1:]                 # 至少保留最后一个
```

### 5.3 为什么加 +2 安全余量？

`t_inf_actual` 的测量存在误差：
- `time.monotonic()` 精度限制
- gRPC 网络抖动
- 线程调度延迟（推理线程计算完 K_inf 到执行线程实际 pop 之间有延迟）
- `ring_buffer.update()` 的 lock 竞争

如果 `t_inf_actual` 偏小，裁剪不够，新 chunk 的第一个 action 实际上是"过去"的动作，执行它等于让机械臂"回退"一小步，造成撞击感/跳变。

+2 安全余量让机械臂执行更"未来"的动作，代价是每次推理少用 2 个 action（对 chunk_size=50 来说可忽略），但能显著减少 chunk 切换时的不连续感。

### 5.4 完整裁剪流程

```
模型返回:  [a_0, a_1, ..., a_49]           # L=50 个 action
                    ↓
n_action_steps截取: [a_0, a_1, ..., a_19]   # 如果 n_action_steps=20
                    ↓
K_inf裁剪(+2):     [a_5, a_6, ..., a_19]    # 如果 K_inf=3+2=5，跳过前5个
                    ↓
写入buffer:        ring_buffer.update([a_5, ..., a_19])  # 15个有效action
                    ↓
执行线程:          每帧 pop 一个执行，15帧后 buffer 空
```

### 5.5 关于 K_inter

T_inter 等待期间消耗的 action 来自**旧 chunk 的 buffer**，不需要从新 chunk 裁剪。新 chunk 的时间起点是观测时刻 T（不是上次 buffer 更新时刻），所以只需裁剪推理延迟 K_inf。

```
上次buffer更新          观测时刻T           推理完成
      │                    │                  │
      │◄── T_inter ──────►│◄── t_inf ──────►│
      │                    │                  │
      │ 消耗旧chunk的action │ 消耗旧chunk的action│
      │ (K_inter 个)       │ (K_inf 个)       │
      │                    │                  │
                           │                  │
                    新chunk起点=T        裁剪 action[0..K_inf-1]
                    action[0]=T时刻动作  从 action[K_inf] 开始用
```

---

## 6. 两种运行模式

### 6.1 可行模式 (feasible=True)

```
推理完成 → 写入buffer → 等待T_inter（期间轮询安全网）→ 采集观测 → 推理 → ...
                              │
                              ├─ 正常：等到T_inter时间到
                              └─ 安全网：buffer <= threshold 时提前触发
```

### 6.2 降级模式 (feasible=False)

```
等待 buffer <= 2 → 采集观测 → 推理 → 写入buffer → 立即开始等待
     ↑                                                  │
     └──────────────────────────────────────────────────┘
```

降级模式下 T_inter=0，推理完成后立即开始等待 buffer 消耗到阈值以下再触发下一次。这本质上就是 Phase 1（1.5B 之前）的行为。

---

## 7. 真机测试分析 (2026-02-09)

### 7.1 测试数据汇总

| # | f_exec | n_action_steps | L_eff | t_inf(warmup) | T_inter | K_inf(avg) | Hold% | 卡顿 |
|---|--------|---------------|-------|---------------|---------|------------|-------|------|
| 1 | 30 | 0 | 50 | **3.394s** | 0(降级) | 3.1 | 2.0% | 有 |
| 2 | 30 | 20 | 20 | 0.023s | 0.515s | 3.6 | 0.09% | 有 |
| 3 | 30 | 0 | 50 | 0.028s | 1.638s | 3.1 | 2.0% | 有 |
| 4 | 50 | 20 | 20 | 0.023s | 0.302s | 7.8 | **18.2%** | **最卡** |
| 5 | 50 | 20 | 20 | 0.023s | 0.302s | 7.3 | **14.4%** | **很卡** |
| 6 | 50 | 50 | 50 | 0.020s | 0.784s | 7.3 | 0% | 较好 |

### 7.2 卡顿原因分析

#### 原因 1：K_inf 裁剪不足导致"回到过去"

旧代码没有安全余量：`K_inf_actual = int(t_inf_actual * f_exec)`。如果 `t_inf_actual` 测量偏小（线程调度延迟、lock 竞争等），裁剪不够，新 chunk 的第一个 action 实际上是"过去"的动作。执行它等于让机械臂回退一小步，造成撞击感。

**已修复**：加 +2 安全余量。

#### 原因 2：Hold 导致的轨迹断层

Hold 就是 buffer 空了，执行线程重复发送上一个 action。hold 期间没有新的 action 进来，机械臂的运动轨迹出现"断层"——先是按旧 chunk 的尾部运动，然后 hold 一段时间，然后突然跳到新 chunk 的动作。

#### 原因 3：`ring_buffer.update()` 的替换语义

`update()` 是 `clear() + extend()`。每次推理完成时，buffer 中还没执行完的旧 actions 全部被丢弃。旧 chunk 的最后执行的 action 和新 chunk 的第一个 action 之间没有连续性保证（模型每次基于不同观测推理），这个跳变是视觉上的"微卡"。

#### 原因 4：n_action_steps=20 + f_exec=50 时最卡的原因

测试 4/5：`L_eff=20, f_exec=50, t_inf≈0.15s`

```
每次推理有效 action: 20 - ceil(0.15*50) = 20 - 8 = 12 个
12 个 action 在 50Hz 下只够用: 12/50 = 0.24s
T_inter = 0.302s
```

0.24s 的 action 要撑 0.302s 的间隔，差 3 帧 hold。K_inf=8 意味着 20 个 action 中有 8 个被裁掉，只剩 12 个，太少了。

#### 原因 5：Warmup t_inf 不准导致参数推导错误

测试 1：旧版 warmup 只测一次，测到 t_inf=3.394s（GPU 冷启动），K_inf=102 > L_eff=50，系统判定 FAIL 降级。

**已修复**：多次测量，丢弃冷启动。

### 7.3 卡顿根因总结

| 根因 | 影响 | 涉及测试 | 状态 |
|------|------|---------|------|
| **K_inf 裁剪不足（无安全余量）** | chunk 切换时撞击感 | 所有测试 | **已修复 (+2)** |
| **warmup 只测一次（冷启动不准）** | 参数推导错误，降级 | 测试 1 | **已修复（多次测量）** |
| **buffer 替换语义导致 chunk 间跳变** | 即使 Hold=0 也有微卡 | 所有测试 | 后续改善 |
| **n_action_steps 太小 + f_exec 太高** | 有效 action 不够 | 测试 4, 5 | 调参解决 |
| **T_inter 接近 T_inter_max** | 安全余量不足 | 测试 3 | 调参解决 |

### 7.4 推荐配置

基于实际数据（t_inf 稳定约 0.1-0.15s, chunk_size=50）：

**最佳配置（30Hz）**：
```yaml
f_exec: 30.0
n_action_steps: 0    # 使用全部 50 个 action
T_inter: 0.0         # 自动推导
t_inf: 0.0           # 多次 warmup 自动测量（已修复冷启动问题）
```

**最佳配置（50Hz）**：
```yaml
f_exec: 50.0
n_action_steps: 0    # 使用全部 50 个 action（不要设 20！）
T_inter: 0.0         # 自动推导
t_inf: 0.0           # 多次 warmup 自动测量
```

**关键建议**：
1. **不要设 `n_action_steps=20`**，除非你有明确理由。20 个 action 在高频下太少
2. **f_exec=50 时必须 n_action_steps=0 或 >=40**，否则有效 action 不够
3. 如果仍想手动指定 t_inf，可以设 `t_inf: 0.15`（你观察到的稳定值）

---

## 8. 常见问题 FAQ

### Q: config.t_inf=0 会导致什么？

A: 系统会在启动时执行 5 次 warmup 推理，丢弃第 1 次（GPU 冷启动），取后 4 次平均值作为 t_inf。这比旧版只测一次准确得多。也可以手动指定 `t_inf: 0.15` 跳过测量。

### Q: T_inter 等待期间消耗的 action 需要从新 chunk 裁剪吗？

A: **不需要**。T_inter 期间消耗的是旧 chunk 的 action（从旧 buffer pop）。新 chunk 的时间起点是观测时刻 T，只需裁剪推理延迟 K_inf。详见 [5.5 关于 K_inter](#55-关于-kinter)。

### Q: 为什么 K_inf 裁剪加了 +2 安全余量？

A: `t_inf_actual` 的测量存在误差（线程调度、lock 竞争、网络抖动等）。如果测量偏小，裁剪不够，新 chunk 的第一个 action 是"过去"的动作，执行它会让机械臂回退一小步，造成撞击感。+2 让机械臂执行更"未来"的动作，代价是每次少用 2 个 action（对 chunk_size=50 可忽略）。

### Q: 为什么 Hold=0 还是感觉卡？

A: 因为 `ring_buffer.update()` 是替换语义。每次新推理完成时，旧 buffer 被清空，新 actions 写入。旧 chunk 的最后一个 action 和新 chunk 的第一个 action 之间可能存在不连续性（跳变），这就是视觉上的"微卡"。后续可以通过 action blending（新旧 chunk 平滑过渡）来改善。

### Q: n_action_steps 设小了为什么更卡？

A: 因为 `L_eff = min(n_action_steps, L)`。设小了意味着每次推理只用前 N 个 action，有效 action 更少。在高频执行下，少量 action 很快被消耗完，buffer 空了就 hold。

公式：`可用时间 = (L_eff - K_inf) / f_exec`。如果可用时间 < T_inter，必然出现 hold。

### Q: 降级模式和正常模式有什么区别？

A:
- **正常模式**：按 T_inter 定时触发推理，buffer 低于阈值时安全网提前触发
- **降级模式**：不定时，等 buffer 消耗到 ≤2 才触发推理（Phase 1 的行为）

降级模式是兜底方案，当参数推导判定系统不可行时自动启用。

---

## 附录：代码关键位置

| 功能 | 文件 | 行号 |
|------|------|------|
| 配置定义 | `async_inference_client.py` | 80-94 |
| warmup 多次测量 | `async_inference_client.py` | 461-570 |
| 单次 warmup 推理 | `async_inference_client.py` | 572-610 |
| 参数推导 | `async_inference_client.py` | 612-715 |
| 推理线程主循环 | `async_inference_client.py` | 717-823 |
| K_inf 裁剪 (+2 安全余量) | `async_inference_client.py` | 760-766 |
| 执行循环 | `async_inference_client.py` | 408-452 |
| ring_buffer | `core/ring_buffer.py` | 16-69 |
| YAML 配置 | `config/dagger_params.yaml` | 全文 |
| 测试 | `tests/test_parameter_derivation.py` | 全文 |
