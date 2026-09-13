# Phase 1 异步推理框架 — 测试指南

## 环境要求

- **Conda 环境**: `robocoin`
- **机械臂**: Realman RM65-B, IP `192.168.1.18`, 已上电
- **相机**: Intel RealSense (序列号 `230422272673`), 已连接
- **两个终端窗口**（SSH 或本地）

---

## 测试 1: dry_run 通信测试（不控制机械臂）

> 目的：验证 Client ↔ Server gRPC 通信正常，ring buffer 工作正常

### 终端 1 — 启动 PolicyServer (debug 模式)

```bash
conda activate robocoin
cd ~/project/ours_method/RoboCOIN

python -m lerobot.extensions.unified_deploy.server.policy_server \
    --debug \
    --debug_dataset_path=/home/ding/project/ours_method/spacemouse_control_arm/data/local/realman_teleop_vr_0129_02 \
    --debug_episode=0 \
    --debug_action_space=joints \
    --debug_action_type=absolute \
    --port=50051
```

等待输出 `PolicyServer 已启动，监听 0.0.0.0:50051` 后继续。

### 终端 2 — 启动 AsyncInferenceClient (dry_run)

```bash
conda activate robocoin
cd ~/project/ours_method

python dagger/async_inference_client.py \
    --config dagger/config/dagger_params.yaml \
    --dry-run \
    --f-exec 10 \
    --max-steps 200
```

### 预期结果

- Server 端：不再报 `No module named 'dagger'`，能正常接收观测
- Client 端：
  - `[AsyncClient] 已从 Server 获取配置` 出现
  - `Infer` 计数开始增长（>0）
  - `Exec` 和/或 `Hold` 计数开始增长
  - 最终统计中 `Avg inference time` 有合理值

### 判定标准

- [ ] Server 无报错
- [ ] Client `Infer > 0`
- [ ] Client `Exec + Hold > 0`

---

## 测试 2: 真机低频率执行（10Hz）

> 目的：验证机械臂能跟随数据集轨迹运动

### 终端 1 — PolicyServer（同测试 1，保持运行）

### 终端 2 — 启动 AsyncInferenceClient（真机）

```bash
conda activate robocoin
cd ~/project/ours_method

python dagger/async_inference_client.py \
    --config dagger/config/dagger_params.yaml \
    --f-exec 10 \
    --max-steps 300
```

> **注意**：去掉了 `--dry-run`，机械臂会真实运动！确保机械臂周围无障碍物。

### 预期结果

- 机械臂跟随数据集中的轨迹运动
- 运动基本平滑（10Hz 下可能有轻微顿挫）
- 按 `Ctrl+C` 能正常退出

### 判定标准

- [ ] 机械臂正常运动
- [ ] 无异常抖动或突变
- [ ] 正常退出，无报错

---

## 测试 3: 提高执行频率（30Hz）

> 目的：验证高频率下的平滑度和 K_inf 裁剪效果

### 终端 2

```bash
python dagger/async_inference_client.py \
    --config dagger/config/dagger_params.yaml \
    --f-exec 30 \
    --max-steps 600
```

### 预期结果

- 运动比 10Hz 更平滑
- 日志中 FPS 接近 30
- `[Inference]` 日志显示合理的 `K_inf` 值（通常 1-5）

### 关键日志解读

```
[Inference] #3: t_inf=0.150s, K_inf=4, chunk=50, valid=46, buf=46
```

| 字段 | 含义 | 正常范围 |
| :--- | :--- | :--- |
| `t_inf` | 本次推理耗时 | 0.05-0.5s |
| `K_inf` | 推理期间消耗的动作数（被跳过） | 1-15 |
| `chunk` | Server 返回的总动作数 | 10-100 |
| `valid` | 裁剪后有效动作数 | chunk - K_inf |
| `buf` | 更新后 buffer 大小 | = valid |

### 判定标准

- [ ] FPS 接近 30
- [ ] 运动比 10Hz 更平滑
- [ ] K_inf 值合理（< chunk 的一半）

---

## 测试 4: Hold vs No-Hold 对比

> 目的：验证 buffer 空时的行为差异

### 4a: Hold 模式（默认）

```bash
python dagger/async_inference_client.py \
    --config dagger/config/dagger_params.yaml \
    --f-exec 30 \
    --max-steps 300
```

### 4b: No-Hold 模式

```bash
python dagger/async_inference_client.py \
    --config dagger/config/dagger_params.yaml \
    --f-exec 30 \
    --max-steps 300 \
    --no-hold
```

### 预期结果

- **Hold 模式**：buffer 空时机械臂保持最后位置，运动连续
- **No-Hold 模式**：buffer 空时机械臂停止，推理间隙有明显停顿

### 判定标准

- [ ] Hold 模式下 `Hold` 计数 > 0，运动连续
- [ ] No-Hold 模式下 `Hold` 计数 = 0，有停顿感

---

## 测试 5: 长时间运行稳定性

> 目的：验证无内存泄漏、无死锁、无累积延迟

```bash
python dagger/async_inference_client.py \
    --config dagger/config/dagger_params.yaml \
    --f-exec 30 \
    --max-steps 0
```

> `--max-steps 0` 表示无限运行，手动 `Ctrl+C` 停止。建议运行 5 分钟。

### 观察要点

- FPS 是否稳定（不随时间下降）
- 内存占用是否稳定（可用 `htop` 监控）
- 无 Python 异常或 gRPC 错误

### 判定标准

- [ ] 运行 5 分钟无崩溃
- [ ] FPS 保持稳定
- [ ] 无内存泄漏

---

## 紧急停止

| 方式 | 操作 |
| :--- | :--- |
| 优雅退出 | 可视化窗口按 `q`（本地模式） |
| 终端中断 | `Ctrl+C` |
| 急停 | 按机械臂急停按钮 |

---

## 统计日志解读

### 周期性日志

```
Frame 100, FPS=29.8, Exec=95, Hold=5, Infer=3, Buf=7
```

| 字段 | 含义 |
| :--- | :--- |
| `Frame` | 总执行帧数 |
| `FPS` | 实际执行频率，应接近 f_exec |
| `Exec` | 从 buffer 取出并执行的动作数 |
| `Hold` | buffer 空时重发最后动作的次数 |
| `Infer` | 完成的推理次数 |
| `Buf` | 当前 buffer 中剩余动作数 |

### 最终统计

```
[AsyncClient] === Statistics ===
  Total steps: 600
  Actions executed: 580
  Hold repeats: 20
  Inferences: 15
  Avg inference time: 0.120s
  Avg K_inf: 3.6 actions
  Avg FPS: 29.8
```

### 健康指标

- `Avg FPS` 应接近 `f_exec`
- `Hold repeats / Total steps` < 10% 为正常
- `Avg K_inf` < `chunk / 2` 为正常（否则推理太慢）

---

## 故障排查

| 现象 | 可能原因 | 解决方案 |
| :--- | :--- | :--- |
| `No module named 'dagger'` | pickle 序列化路径错误 | 确认 data_types 从 `lerobot.extensions...` 导入 |
| `Infer=0` 始终为 0 | gRPC 通信失败 | 检查 Server 是否启动，端口是否正确 |
| FPS 远低于 f_exec | 观测采集太慢 | 检查相机连接，降低 f_exec |
| 机械臂抖动 | 动作不连续 | 启用 hold_on_empty，降低 f_exec |
| `K_inf > chunk` | 推理太慢 | 降低 f_exec 或优化 Server 推理速度 |

---

## 测试 6: 真实 ACT Policy 异步推理（chunk_size=50）

> 目的：验证真实 policy 推理下异步框架的功能正确性，对比原始 RealmanClient

### 前提

- 已有预训练 ACT 模型 checkpoint（chunk_size=50）
- 以下命令中 `<PRETRAINED_PATH>` 替换为你的模型路径，例如：
  `outputs/realman_act/checkpoints/050000/pretrained_model`
- 以下命令中 `<TASK>` 替换为你的任务指令，例如：
  `"pick up the yellow banana and put it in the white plate"`

### 6.1 启动 PolicyServer（真实 ACT 模型）

**终端 1**：

```bash
conda activate robocoin
cd ~/project/ours_method/RoboCOIN

python -m lerobot.extensions.unified_deploy.server.policy_server \
    --policy_type=act \
    --pretrained_path=<PRETRAINED_PATH> \
    --device=cuda \
    --port=50051
```

等待输出：
```
PolicyServer 已启动，监听 0.0.0.0:50051
```

确认 Server 日志中显示的 `action_space` 和 `action_type`（从 `train_config.json` 自动读取）。

---

### 6.2 基准测试：原始 RealmanClient（对照组）

> 先用原始 Client 跑一次，作为对比基准。

**终端 2**：

```bash
conda activate robocoin
cd ~/project/ours_method/RoboCOIN

python -m lerobot.extensions.unified_deploy.client.realman_client \
    --robot_ip="192.168.1.18" \
    --server_address=127.0.0.1:50051 \
    --frequency=10 \
    --n_action_steps=50 \
    --camera_configs="{ cam0_rgb: {type: intelrealsense, serial_number_or_name: '230422272673', fps: 30, width: 640, height: 480}, cam1_rgb: {type: intelrealsense, serial_number_or_name: '420122071413', fps: 30, width: 640, height: 480} }" \
    --camera_rotations="{ cam0_rgb: 180 }" \
    --task=<TASK> \
    --max_steps=500 \
    --chunk_size_threshold=0
```

**记录**：
- [ ] 机械臂运动是否正常完成任务
- [ ] 运动平滑度（主观评分 1-5）
- [ ] 终端输出的 FPS 值
- [ ] 是否有明显卡顿或停顿

按 `Ctrl+C` 退出后，**重启 PolicyServer**（Server 端 DebugAdapter 会重置帧指针，真实 policy 不需要重启，但为了公平对比建议重启）。

---

### 6.3 AsyncInferenceClient 10Hz 测试

**终端 2**：

```bash
conda activate robocoin
cd ~/project/ours_method

python dagger/async_inference_client.py \
    --config dagger/config/dagger_params.yaml \
    --server 127.0.0.1:50051 \
    --f-exec 10 \
    --task "pick and place" \
    --max-steps 500
```

**观察要点**：
- chunk=50 时，10Hz 下每个 chunk 可供 5 秒，推理线程有充足时间
- `Hold` 应该很少（接近 0）
- `K_inf` 应该很小（0-1）

**记录**：
- [ ] 机械臂运动是否正常
- [ ] 运动平滑度（主观评分 1-5）
- [ ] `Avg inference time` 值（真实 ACT 推理耗时）
- [ ] `Hold repeats / Total steps` 比例
- [ ] 与基准测试 6.2 对比：运动轨迹是否一致

---

### 6.4 AsyncInferenceClient 30Hz 测试

```bash
python dagger/async_inference_client.py \
    --config dagger/config/dagger_params.yaml \
    --server 127.0.0.1:50051 \
    --f-exec 30 \
    --task <TASK> \
    --max-steps 1500
```

**观察要点**：
- chunk=50 时，30Hz 下每个 chunk 可供 ~1.67 秒
- 真实 ACT 推理耗时通常 0.05-0.3s，K_inf 预计 2-9
- 有效动作 ~41-48 个，供 ~1.4-1.6 秒，推理线程来得及补充
- `Hold` 比例应 < 20%

**记录**：
- [ ] FPS 是否稳定接近 30
- [ ] 运动是否比 10Hz 更平滑
- [ ] `Avg K_inf` 值
- [ ] `Hold repeats / Total steps` 比例

---

### 6.5 高频率 50Hz 极限测试（可选）

> 测试框架在高频率下的表现极限

```bash
python dagger/async_inference_client.py \
    --config dagger/config/dagger_params.yaml \
    --server 127.0.0.1:50051 \
    --f-exec 50 \
    --task <TASK> \
    --max-steps 2500
```

**观察要点**：
- chunk=50 时，50Hz 下每个 chunk 仅供 1 秒
- 如果 t_inf > 0.2s，K_inf > 10，有效动作 < 40，可能出现较多 Hold
- 这是压力测试，Hold 比例 < 30% 即可接受

---

### 6.6 对比总结表

测试完成后填写：

| 测试项 | FPS | Exec | Hold | Hold% | Avg t_inf | Avg K_inf | 平滑度(1-5) | 备注 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 6.2 原始 Client 10Hz | | | N/A | N/A | | N/A | | 基准 |
| 6.3 Async 10Hz | | | | | | | | |
| 6.4 Async 30Hz | | | | | | | | |
| 6.5 Async 50Hz | | | | | | | | 可选 |

### 预期结论

- **10Hz**：Async Client 与原始 Client 运动轨迹基本一致，Hold ≈ 0
- **30Hz**：运动更平滑，Hold < 20%，K_inf < chunk/2
- **50Hz**：压力测试，Hold 可能较高但不崩溃

---

### 6.7 功能正确性验证清单

- [ ] Server 正确识别 policy 类型和 action_space（从 train_config.json 自动读取）
- [ ] Client 正确接收 Server 配置（`[AsyncClient] 已从 Server 获取配置`）
- [ ] chunk=50 的 ActionChunk 被正确接收和解析
- [ ] K_inf 裁剪正确（`valid = chunk - K_inf`，valid > 0）
- [ ] 机械臂运动方向和幅度正确（与原始 Client 一致）
- [ ] 夹爪动作正确（开合时机与原始 Client 一致）
- [ ] 长时间运行无崩溃（至少 500 步）
- [ ] `Ctrl+C` 能正常退出，资源正确释放

---

### 6.8 常见问题

| 现象 | 可能原因 | 解决方案 |
| :--- | :--- | :--- |
| Server 报 CUDA OOM | GPU 显存不足 | 关闭其他 GPU 进程，或用 `--device cpu` |
| `chunk=1` 频繁出现 | Server 返回单步动作 | 检查 `--n_action_steps` 是否设置正确，ACT 默认应返回 50 |
| 机械臂运动方向反了 | action_space/action_type 不匹配 | 检查 Server 日志中的 action_space 和 action_type |
| Hold 比例 > 50% | 推理太慢或 chunk 太小 | 降低 f_exec，或检查 GPU 利用率 |
| `terminate called` 退出时 | Realman SDK C 库线程清理 | 不影响功能，可忽略 |
