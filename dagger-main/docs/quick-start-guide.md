# DAgger 系统快速启动指南

> 从 build 到运行的完整流程，包括终端模式（async_client + PolicyServer）和 Web UI 模式（dagger_node）
>
> **最后更新**: 2026-02-12

---

## 目录

1. [环境准备](#环境准备)
2. [构建 ROS2 工作空间](#构建-ros2-工作空间)
3. [配置文件说明](#配置文件说明)
4. [模式 1: 终端模式（async_client）](#模式-1-终端模式asyncclient)
5. [模式 2: Web UI 模式（dagger_node）](#模式-2-web-ui-模式daggernode)
6. [常见问题排查](#常见问题排查)
7. [参数调优建议](#参数调优建议)

---

## 环境准备

### 必需组件

- **Conda 环境**: `robocoin`（包含 ROS2 + PyTorch + LeRobot）
- **ROS2**: Jazzy（系统 Python 3.12）
- **硬件**: Realman RM-75 机械臂 + 2 个 RealSense 相机
- **网络**: 机械臂 IP `192.168.1.18`，本机 IP `192.168.1.x`

### 激活环境

```bash
conda activate robocoin
```

---

## 构建 ROS2 工作空间

### 首次构建或清理缓存后

```bash
cd /home/ubuntu/Desktop/Workspace/lerobot_policy_deploy/lerobotv3/vr_teleop/spacemouse_control_arm/ros2_realman_ws

# Source ROS2
source /opt/ros/jazzy/setup.bash

# 构建
colcon build

# Source 工作空间
source install/setup.bash
```

> **注意**：`--symlink-install` 在当前 setuptools 版本下不可用（报 `--editable not recognized`），
> 所以每次修改 `.py` 源码后都需要重新 `colcon build`。

**何时需要重新 build**：
- ✅ 修改 Python 源码（`.py` 文件）
- ✅ 修改 `setup.py` 的 `entry_points`
- ✅ 修改 `package.xml` 依赖
- ❌ 修改 yaml 配置文件（**不需要 build**，直接重新 launch）

### 清理缓存（可选）

如果怀疑运行了旧版本代码，清理所有缓存：

```bash
# 清理 ROS2 build/install/log
rm -rf build/ install/ log/

# 清理 Python __pycache__
find /home/ubuntu/Desktop/Workspace/lerobot_policy_deploy/lerobotv3/dagger -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
```

清理后必须重新 `colcon build`。

---

## 配置文件说明

### 主要配置文件

| 文件 | 路径 | 用途 |
|------|------|------|
| `dagger_params.yaml` | `dagger/config/` | DAgger 系统配置（推理、录制、VR、相机） |
| `teleop_params.yaml` | `vr_teleop/.../config/` | 硬件配置（driver、相机、VR input） |

### 关键参数速查

#### `dagger_params.yaml`

```yaml
# 机器人
robot_ip: "192.168.1.18"
dry_run: false  # true=仿真，false=真机

# PolicyServer 自动启动
policy_server:
  auto_launch: true  # false=手动启动 PolicyServer
  policy_type: "act"
  pretrained_path: "/path/to/model"
  device: "cuda"

# 执行频率
f_exec: 30.0  # Hz，建议 10-30
hold_on_empty: true  # buffer 空时重发最后一个 action

# 录制
enable_recording: true
repo_id: "local/dagger_realman_0212-test"
dataset_root: "/path/to/data"

# 夹爪（仅 async_client 使用）
gripper_deadband: 20  # SDK 值变化 <20 时不发送指令
```

#### `teleop_params.yaml`

```yaml
realman_driver_node:
  ros__parameters:
    ip: "192.168.1.18"
    dry_run: false
    gripper_deadband: 20  # 必须与 dagger_params.yaml 一致！
```

**重要**：`gripper_deadband` 必须在两个文件中都设为 20，否则 dagger_node 会出现夹爪振荡（详见 `common-pitfalls.md` BUG-017）。

---

## 模式 1: 终端模式（async_client）

**特点**：直接调用 SDK，无 ROS2 中间层，适合快速测试和调试。

### 步骤 1: 启动 PolicyServer

**方式 A：独立启动（推荐调试）**

```bash
cd /home/ubuntu/Desktop/Workspace/lerobot_policy_deploy/lerobotv3/RoboCOIN

python -m lerobot.extensions.unified_deploy.server.policy_server \
  --policy-type act \
  --pretrained-path /path/to/pretrained_model \
  --device cuda \
  --port 50051
```

等待日志显示 `PolicyServer 已启动，监听 0.0.0.0:50051`。

**方式 B：通过 dagger launch 自动启动**

在 `dagger_params.yaml` 中设置 `policy_server.auto_launch: true`，然后 `ros2 launch dagger/launch/dagger.launch.py` 会自动启动 PolicyServer（但会同时启动所有 ROS2 节点，不适合纯终端模式）。

### 步骤 2: 启动 async_client

**新终端**（保持 PolicyServer 运行）：

```bash
cd /home/ubuntu/Desktop/Workspace/lerobot_policy_deploy/lerobotv3

python dagger/async_inference_client.py \
  --config dagger/config/dagger_params.yaml \
  --f-exec 30 \
  --max-steps 1000 \
  --enable-recording \
  --repo-id "local/test_async_$(date +%m%d_%H%M)" \
  --dataset-root "/home/ubuntu/Desktop/Workspace/lerobot_policy_deploy/lerobotv3/dagger/data"
```

**参数说明**：
- `--f-exec 30`: 执行频率 30Hz（可选 10/20/30）
- `--max-steps 1000`: 最多执行 1000 步后自动停止（0=无限）
- `--enable-recording`: 启用录制
- `--repo-id`: 数据集 ID（建议加时间戳避免覆盖）
- `--dataset-root`: 数据集保存目录

### 步骤 3: 运行

1. 等待 warmup 完成（~2s）
2. 看到 `按回车开始...` 后按回车
3. 机械臂开始执行 policy 动作
4. `Ctrl+C` 停止，自动保存录制数据

### 日志关键信息

```
[AsyncClient] 推理线程启动
[Warmup] chunk_size探测: t_inf=0.365s, chunk_size=50
[Parameters] === Phase 1.5B 参数推导 ===
  执行频率 (f_exec):         30.0 Hz
  推理延迟 (t_inf):          0.200s
  实际推理间隔 (T_inter):    1.173s (自动推导)
  稳定性约束:                PASS

[TIMING-INFER-AC] #1: t_inf_total=145.5ms, K_inf=29, valid=21, buf=21
[TIMING-EXEC-AC] #100: total_work=0.17ms, EXEC, buf=9, actual_hz=19.8
```

**正常指标**：
- `feasible: PASS` — 系统可行
- `K_inf` 从 29 逐渐降到 26 左右（EMA 自适应）
- `actual_hz` 稳定在 19-21Hz（30Hz 执行频率下的实际吞吐）
- `Hold=0` — 无 buffer 空等待

---

## 模式 2: Web UI 模式（dagger_node）

**特点**：ROS2 架构，支持 VR 遥操作 + Policy 推理切换，适合数据采集和在线学习。

### 步骤 1: 启动 DAgger 系统

```bash
cd /home/ubuntu/Desktop/Workspace/lerobot_policy_deploy/lerobotv3

# 确保已 source ROS2 工作空间
source /home/ubuntu/Desktop/Workspace/lerobot_policy_deploy/lerobotv3/vr_teleop/spacemouse_control_arm/ros2_realman_ws/install/setup.bash

# 启动（会自动启动 PolicyServer + driver + VR input + cameras + dagger_node + Web UI）
ros2 launch dagger/launch/dagger.launch.py
```

**启动的组件**：
1. `policy_server` — gRPC 策略推理服务（如果 `auto_launch: true`）
2. `realman_driver_node` — 机械臂驱动（CANFD + UDP 状态）
3. `vr_input_node` — VR 手柄输入（ADB/UDP）
4. `camera_node` x2 — RealSense 相机
5. `dagger_node` — DAgger 控制节点（VR + Policy + Recording）
6. `dagger_control_panel` — Web UI 控制面板

### 步骤 2: 打开 Web UI

浏览器访问 `http://localhost:5002`（或 `http://<本机IP>:5002`）

**Web UI 界面**：
- **节点状态**：显示所有 ROS2 节点在线状态
- **Server 状态**：PolicyServer 连接状态、模型信息
- **模式控制**：IDLE / HUMAN / POLICY 切换按钮
- **录制控制**：开始/停止 episode、丢弃 episode
- **推理详情**：实时显示推理频率、buffer 大小、K_inf
- **录制详情**：当前 episode 帧数、已保存 episode 数
- **日志**：实时滚动日志

### 步骤 3: 验证系统就绪

等待以下状态全部就绪：

1. **节点状态** — 所有节点显示绿色"在线"
2. **Server 状态** — 显示"已连接"，模型信息正确
3. **当前模式** — 显示"IDLE"（灰色）

如果 PolicyServer 显示"离线"，检查：
- `dagger_params.yaml` 中 `policy_server.auto_launch` 是否为 `true`
- PolicyServer 日志是否有错误（终端输出）
- 模型路径是否正确

### 步骤 4: 开始推理

点击 **"开始推理"** 按钮：

1. 系统自动切换到 **POLICY** 模式（绿色）
2. driver 启用跟随（机械臂开始响应指令）
3. 推理循环启动（后台推理线程开始工作）
4. 录制自动开始（如果 `enable_recording: true`）

**观察指标**（推理详情区域）：
- **推理频率**: ~0.8-1.0 次/秒（取决于 T_inter）
- **Buffer 大小**: 6-25（正常波动）
- **平均推理时间**: 25-30ms（dagger_node 的 obs 是预缓存的，比 async_client 快）

### 步骤 5: VR 接管（可选）

**触发方式**：按下 VR 手柄的 **trigger**（食指扳机）

**状态变化**：
- 模式自动切换：POLICY → **HUMAN**（黄色）
- 推理暂停，buffer 清空
- VR 控制激活，机械臂跟随手柄运动

**恢复 Policy**：松开 trigger

**状态变化**：
- 模式自动切换：HUMAN → **POLICY**（绿色）
- 推理恢复，buffer 重新填充
- 机械臂继续执行 policy 动作

**录制行为**：
- HUMAN 模式：`control_source=1`（人类控制）
- POLICY 模式：`control_source=0`（策略控制）
- 两种模式的数据都会录制到同一个 episode

### 步骤 6: 停止推理

点击 **"暂停推理"** 按钮：

1. 模式切换：POLICY → **IDLE**（灰色）
2. 推理循环停止
3. driver 禁用跟随（机械臂停止响应）
4. 录制暂停（当前 episode 保持打开状态）

**注意**：暂停后 episode 仍在录制中（未保存），可以：
- 点击"开始推理"继续录制到同一 episode
- 点击"停止 Episode"保存当前 episode
- 点击"丢弃 Episode"放弃当前 episode

### 步骤 7: 保存/丢弃 Episode

**保存 Episode**：

点击 **"停止 Episode"** 按钮：
- 当前 episode 保存到磁盘（parquet + 视频）
- 录制详情显示"已保存 episode 数"增加
- 下次"开始推理"会创建新 episode

**丢弃 Episode**：

点击 **"丢弃 Episode"** 按钮：
- 当前 episode 数据被丢弃（不保存）
- 录制帧数清零
- 下次"开始推理"会创建新 episode

### 步骤 8: 关闭系统

终端按 `Ctrl+C`：
- 所有节点优雅关闭
- 当前 episode 自动保存（如果未手动保存）
- PolicyServer 停止
- Web UI 关闭

---

## 常见问题排查

### 1. launch 报错 "package 'realman_teleop' not found"

**原因**：ROS2 工作空间未构建或未 source。

**解决**：

```bash
cd /home/ubuntu/Desktop/Workspace/lerobot_policy_deploy/lerobotv3/vr_teleop/spacemouse_control_arm/ros2_realman_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

### 2. PolicyServer 显示"离线"

**原因 A**：`auto_launch: false` 且未手动启动 PolicyServer。

**解决**：改 `dagger_params.yaml` 中 `policy_server.auto_launch: true`，或手动启动 PolicyServer。

**原因 B**：模型路径错误。

**解决**：检查 `policy_server.pretrained_path` 是否存在，路径必须是绝对路径。

**原因 C**：端口被占用。

**解决**：

```bash
# 检查端口占用
lsof -i :50051

# 杀死占用进程
kill -9 <PID>
```

### 3. 夹爪持续振荡（反复开合）

**原因**：`gripper_deadband` 配置不一致。

**解决**：确保两个文件中 `gripper_deadband` 都是 20：
- `dagger/config/dagger_params.yaml`: `gripper_deadband: 20`
- `vr_teleop/.../config/teleop_params.yaml`: `gripper_deadband: 20`

修改后重新 launch（不需要 build）。

**验证**：

```bash
ros2 param get /realman_driver_node gripper_deadband
# 应返回: Integer value is: 20
```

详见 `common-pitfalls.md` BUG-017。

### 4. 机械臂运动失控（关节剧烈跳变）

**原因**：PolicyServer 加载了旧版代码（`.pth` 文件指向错误）。

**排查**：

```bash
# 检查 .pth 指向
cat /home/ubuntu/.local/lib/python3.12/site-packages/__editable__.robocoin-0.1.0.1.pth

# 应该指向 lerobotv3/RoboCOIN/src，如果不是则修改
echo "/home/ubuntu/Desktop/Workspace/lerobot_policy_deploy/lerobotv3/RoboCOIN/src" > \
  /home/ubuntu/.local/lib/python3.12/site-packages/__editable__.robocoin-0.1.0.1.pth
```

重启 PolicyServer 后生效。详见 `common-pitfalls.md` BUG-015。

### 5. 前几步机械臂有明显跳变

**原因**：DAggerRecorder lazy init 阻塞 executor（已修复）。

**验证**：检查日志中 `obs_age` 是否在前几次推理中异常大（>1s）。如果是，说明修复未生效，检查代码版本。

详见 `common-pitfalls.md` BUG-016。

### 6. Web UI 完全空白，无任何显示

**原因**：前端 JS 语法错误（如 `const` 重复声明）。

**排查**：打开浏览器 DevTools Console，检查是否有 `SyntaxError`。

详见 `common-pitfalls.md` BUG-012。

### 7. 磁盘空间被占满

**原因**：HuggingFace `datasets` 库的 Arrow 缓存累积（永不自动清理）。

**解决**：

```bash
# 安全删除缓存（不影响已保存的数据集）
rm -rf ~/.cache/huggingface/datasets/

# 检查缓存大小
du -sh ~/.cache/huggingface/datasets/
```

详见 `common-pitfalls.md` PITFALL-016。

---

## 参数调优建议

### 执行频率（f_exec）

| 频率 | 适用场景 | 优点 | 缺点 |
|------|---------|------|------|
| 10Hz | 首次测试、慢速任务 | 稳定，buffer 不易空 | 响应慢，轨迹不平滑 |
| 20Hz | 一般任务 | 平衡性能和稳定性 | - |
| 30Hz | 快速任务、精细操作 | 响应快，轨迹平滑 | buffer 易空，需要更高推理频率 |

**建议**：首次测试从 10Hz 开始，逐步提高到 30Hz。

### 推理间隔（T_inter）

- `T_inter: 0.0` — 推理完成后立即触发下一次（最大吞吐）
- `T_inter: 0.5` — 每 0.5s 推理一次（降低 GPU 负载）
- `T_inter: 1.0` — 每 1s 推理一次（极低负载）

**建议**：保持 `T_inter: 0.0`，让系统自动推导最优间隔（Phase 1.5B 自适应）。

### Hold on Empty

- `hold_on_empty: true` — buffer 空时重发最后一个 action（平滑，推荐）
- `hold_on_empty: false` — buffer 空时机械臂停止（安全，但会卡顿）

**建议**：保持 `true`，除非调试 buffer 空问题。

### 录制帧率（recording_fps）

- 必须与 `f_exec` 一致（或略低）
- 过高会导致重复帧，过低会丢失数据
- 建议：`recording_fps = f_exec`

### 夹爪 Deadband

- `gripper_deadband: 20` — 推荐值，过滤 ±0.02 的微小波动
- `gripper_deadband: 10` — 过小，会导致振荡
- `gripper_deadband: 50` — 过大，夹爪响应迟钝

**关键**：`dagger_params.yaml` 和 `teleop_params.yaml` 中必须一致。

---

## 附录：命令速查表

### 构建 ROS2 工作空间

```bash
cd /home/ubuntu/Desktop/Workspace/lerobot_policy_deploy/lerobotv3/vr_teleop/spacemouse_control_arm/ros2_realman_ws
source /opt/ros/jazzy/setup.bash
colcon build && source install/setup.bash
```

### 启动 PolicyServer（独立）

```bash
cd /home/ubuntu/Desktop/Workspace/lerobot_policy_deploy/lerobotv3/RoboCOIN
python -m lerobot.extensions.unified_deploy.server.policy_server \
  --policy-type act \
  --pretrained-path /path/to/model \
  --device cuda \
  --port 50051
```

### 启动 async_client

```bash
cd /home/ubuntu/Desktop/Workspace/lerobot_policy_deploy/lerobotv3
python dagger/async_inference_client.py \
  --config dagger/config/dagger_params.yaml \
  --f-exec 30 \
  --enable-recording \
  --repo-id "local/test_$(date +%m%d_%H%M)"
```

### 启动 dagger_node（Web UI）

```bash
cd /home/ubuntu/Desktop/Workspace/lerobot_policy_deploy/lerobotv3
ros2 launch dagger/launch/dagger.launch.py
```

### 检查 ROS2 参数

```bash
# 检查 gripper_deadband
ros2 param get /realman_driver_node gripper_deadband

# 检查所有 driver 参数
ros2 param list /realman_driver_node
```

### 清理缓存

```bash
# ROS2 build 缓存（清理后必须重新 colcon build）
cd /home/ubuntu/Desktop/Workspace/lerobot_policy_deploy/lerobotv3/vr_teleop/spacemouse_control_arm/ros2_realman_ws
rm -rf build/ install/ log/

# Python __pycache__
find /home/ubuntu/Desktop/Workspace/lerobot_policy_deploy/lerobotv3/dagger -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

# HuggingFace datasets 缓存
rm -rf ~/.cache/huggingface/datasets/
```

---

## 相关文档

- `async-inference-system-explained.md` — 异步推理系统架构详解
- `common-pitfalls.md` — 常见 bug 和踩坑记录
- `deployment-guide.md` — 部署指南
- `phase1_testing_guide.md` — Phase 1 测试指南
- `real_machine_testing_guide.md` — 真机测试指南

---

**文档版本**: v1.0
**最后更新**: 2026-02-12
**作者**: Claude Opus 4.6
