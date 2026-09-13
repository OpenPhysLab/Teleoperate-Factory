# 常见易犯错误与踩坑记录

> 记录 DAgger 异步推理系统开发和调试过程中遇到的 bug、陷阱和经验教训。
> 最后更新：2026-02-12 (BUG-019: rm_get_gripper_state() 读错夹爪子系统——BUG-018 修复方向错误)

---

## 目录

1. [BUG-001: Warmup t_inf 与运行时 t_inf 差 10 倍](#bug-001)
2. [BUG-002: Warmup 单次测量遇到 GPU 冷启动](#bug-002)
3. [BUG-003: pickle 序列化的模块路径陷阱](#bug-003)
4. [BUG-004: n_action_steps 设太小导致高频下严重卡顿](#bug-004)
5. [PITFALL-005: Debug Server replay 速度与执行频率不匹配](#pitfall-005)
6. [BUG-006: 录制数据 state 呈方波](#bug-006)
7. [PITFALL-007: EEF rotation 数据精度低（阶梯状，非 bug）](#pitfall-007)
8. [PITFALL-008: rerun 中看不到 policy_action 字段](#pitfall-008)
9. [PITFALL-009: visualize_dataset.sh 路径配置](#pitfall-009)
10. [BUG-010: ROS2 空字符串参数导致 dagger_node 启动崩溃](#bug-010)
11. [BUG-011: DAggerRecorder 目录已存在导致初始化错误刷屏](#bug-011)
12. [BUG-012: WebUI 前端 const srv 重复声明导致整个 UI 不更新](#bug-012)
13. [BUG-013: POLICY 模式机械臂运动速度异常（5x 快于预期）](#bug-013)
14. [BUG-014: Safety check 导致 delta_joints 模型 action chunk 被全部拒绝](#bug-014)
15. [BUG-015: .pth 路径错误 + delta_mask/delta_mode 缺失导致机械臂失控](#bug-015)
16. [PITFALL-016: HuggingFace datasets Arrow 缓存累积撑满磁盘](#pitfall-016)
17. [BUG-016: DAggerRecorder lazy init 阻塞 SingleThreadedExecutor 导致观测冻结 ~1.7s](#bug-016)
18. [BUG-017: gripper_deadband 配置不生效——dagger_config vs teleop_config 的两条路径](#bug-017)
19. [BUG-018: ⚠️ 错误修复——用 TCP rm_get_gripper_state() 替代 _last_gripper_cmd（已撤回）](#bug-018)
20. [BUG-019: rm_get_gripper_state() 读的是 Modbus 夹爪，不反映 RM+ 夹爪位置](#bug-019)
21. [BUG-020: camera_node 和 dagger_node 双重旋转导致图像方向错误](#bug-020)
22. [PITFALL-021: 录制帧率节流过激导致 ~30% 丢帧](#pitfall-021)

---

<a id="bug-001"></a>
## BUG-001: Warmup t_inf 与运行时 t_inf 差 10 倍

**严重程度**: 高
**发现日期**: 2026-02-09
**状态**: 已修复（EMA 自适应）

### 现象

Warmup 测量 t_inf=0.011s，但运行时实际 t_inf 稳定在 0.10~0.17s。参数推导用了 0.011s，导致 K_inf=1，而运行时实际需要裁剪 5~9 个 action。chunk 切换时机械臂出现明显的撞击感/跳变。

### 真机数据

```
Warmup:  t_inf = 0.011s  (5次测量，丢弃冷启动后平均)
运行时:  t_inf = 0.145s  (Avg, 19次推理)
         t_inf = 0.093~0.171s (范围)
倍数:    0.145 / 0.011 = 13x
```

### 根因

Warmup 时执行循环**尚未启动**，系统处于"安静"状态：
- 只有推理线程在工作，无线程竞争
- 相机未持续采集，无 I/O 竞争
- gRPC 无并发请求，无序列化排队
- OS 线程调度器无需在多个活跃线程间切换

运行时两个线程同时跑，相机持续采集，gRPC 有序列化/反序列化开销，OS 线程调度引入延迟。

### 影响链

```
warmup t_inf 偏小
  → 参数推导 K_inf 偏小 (1 而不是 5~9)
  → T_inter_max 偏大 → T_inter 偏大
  → 运行时 K_inf 裁剪不足（+2 安全余量不够）
  → 新 chunk 的第一个 action 是"过去"的动作
  → 机械臂"回退"一小步 → 撞击感/跳变
```

### 临时解决方案（已废弃）

手动加大安全余量：`K_inf_actual = int(t_inf_actual * f_exec) + 12`。
30Hz 需要 +10，50Hz 需要 +15。本质上是在运行时补偿 warmup 的测量误差。

### 正式修复

引入运行时 EMA（指数移动平均）自适应 K_inf：
- 前几次推理用 warmup 值 + 固定安全余量（冷启动保护）
- 之后用运行时 t_inf 的 EMA 值 + 小安全余量
- 系统自动适应，无需手动调参

### 经验教训

1. **Warmup 环境 ≠ 运行时环境**。任何在"安静"状态下的测量都不能直接用于"繁忙"状态的参数推导。
2. **如果需要手动加 +10 以上的安全余量才能消除问题，说明上游估计有系统性偏差**，应该修复估计本身而不是加大余量。
3. **K_inf 裁剪不足的体感是"撞击"而不是"停顿"**。Hold 是停顿（buffer 空），K_inf 不足是跳变（执行过去的动作）。后者体感更差。

---

<a id="bug-002"></a>
## BUG-002: Warmup 单次测量遇到 GPU 冷启动

**严重程度**: 高
**发现日期**: 2026-02-09
**状态**: 已修复（多次测量取平均）

### 现象

首次运行时 warmup 测到 t_inf=3.394s（GPU 冷启动），导致 K_inf=102 > chunk_size=50，系统判定 FAIL 降级为被动等待模式。

### 根因

旧版 warmup 只执行一次推理。GPU 首次推理需要加载模型权重、编译 kernel、分配显存，耗时远超正常值。

### 修复

多次测量（默认 5 次），丢弃第 1 次冷启动，取后 4 次平均值。

### 经验教训

任何涉及 GPU 的性能测量，**第一次结果必须丢弃**。

---

<a id="bug-003"></a>
## BUG-003: pickle 序列化的模块路径陷阱

**严重程度**: 高
**发现日期**: 2026-02-07
**状态**: 已修复

### 现象

跨进程通信时 `pickle.loads()` 报 `No module named 'dagger.deps.data_types'`。

### 根因

`pickle.dumps(obj)` 会记录类的完整模块路径（如 `dagger.deps.data_types.UnifiedObservation`）。接收端必须能 import 同一模块路径，否则反序列化失败。

### 修复

涉及跨进程序列化的数据类必须从原始模块导入（如 `lerobot.extensions.unified_deploy.core.data_types`），不能用本地副本。

### 经验教训

**pickle 序列化 = 绑定模块路径**。如果你复制了一个类到新位置，pickle 序列化的对象在另一端无法反序列化。

---

<a id="bug-004"></a>
## BUG-004: n_action_steps 设太小导致高频下严重卡顿

**严重程度**: 中
**发现日期**: 2026-02-09
**状态**: 调参解决

### 现象

f_exec=50, n_action_steps=20 时 Hold 率高达 18.2%，机械臂严重卡顿。

### 根因

```
L_eff = 20
K_inf ≈ 8 (实际运行时)
有效 action = 20 - 8 = 12
12 个 action 在 50Hz 下只够 0.24s
T_inter = 0.302s
→ 0.24s 的 action 撑不住 0.302s 的间隔 → 大量 hold
```

### 修复

n_action_steps=0（使用全部 chunk）或 n_action_steps >= 40。

### 经验教训

**有效 action 数 = L_eff - K_inf**。高频执行时 K_inf 更大（同样的 t_inf 乘以更大的 f_exec），有效 action 更少。n_action_steps 必须留足余量。

公式速查：`最小 n_action_steps > K_inf + T_inter * f_exec`

---

<a id="pitfall-005"></a>
## PITFALL-005: Debug Server replay 速度与执行频率不匹配

**严重程度**: 低（非 bug，属预期行为）
**发现日期**: 2026-02-10
**状态**: 已知局限，不影响真实 policy

### 现象

使用 debug server（`DebugAdapter`）replay 一个原本十几秒的 episode 时：
- 机械臂几秒就跑完整个轨迹，速度远快于遥操时的实际速度
- **10Hz 执行反而比 30Hz 更快**，与直觉相反
- 抓取动作还没伸到最低就抬起来了，行为明显不对

### 真机数据

```
测试 1.2 (10Hz):  Exec=200, Hold=0,   Infer=486  → 推理线程疯狂循环
测试 1.3 (30Hz):  Exec=600, Hold=256, Infer=335  → 推理线程较慢
```

### 根因

**debug server 的 replay 速度由推理请求频率决定，而不是由 client 的执行频率决定。**

`DebugAdapter.predict()` 的核心逻辑（`debug_adapter.py:211-252`）：

```python
chunk_size = min(10, len(self.states) - self.frame_idx)
# ... 构建 actions ...
self.frame_idx += chunk_size  # 每次请求都推进 dataset 指针
```

每次 gRPC 推理请求都会消耗 dataset 中的 1~10 帧，**不管 client 是否真正执行了这些帧**。

### 影响链

```
debug server 返回 chunk_size 在 1~10 之间交替
  → AsyncInferenceClient 参数推导判定系统不可行（L_valid ≈ 0）
  → 降级为被动等待模式（threshold=2）
  → 推理线程：buffer ≤ 2 就立即触发新推理（几乎无间隔）
  → 每次推理消耗 debug server 的 dataset 帧（frame_idx += chunk_size）
  → 486 次推理 × ~5 帧/次 ≈ 2430 帧被消耗
  → 原始 episode 只有 ~300-400 帧 → 被循环 replay 6-7 遍
  → 每遍只用几秒 → 机械臂运动速度远快于原始遥操
```

### 为什么 10Hz 比 30Hz 还快？

| 指标 | 10Hz | 30Hz |
|------|------|------|
| 推理次数 | 486 | 335 |
| 推理间隔 | ~0.04s（降级模式） | ~0.06s（降级模式） |
| dataset 帧消耗速度 | ~120 帧/s | ~80 帧/s |

10Hz 时执行循环占用更少 CPU，推理线程的 gRPC 延迟更低（~0.04s vs ~0.06s），所以推理线程循环更快，dataset 帧被消耗得更快。

### 为什么真实 policy 不受影响？

测试 1.4（真实 ACT policy，chunk_size=50）表现正常：
- 系统可行（PASS），`T_inter=1.322s`
- 推理线程每 1.3 秒才请求一次（Infer=19）
- 真实 policy 根据当前观测生成 action，不存在"消耗 dataset 帧"的问题

### 结论

**不是代码 bug。** 这是 debug server 的设计局限——它是为功能验证（连通性、数据格式、执行链路）设计的，不是为速度匹配设计的。

如需 debug server 也能正确模拟速度，可考虑：
1. 让 debug server 返回固定的大 `chunk_size`（如 50），匹配真实 policy 行为
2. 在 debug server 中加入时间同步逻辑（按 wall-clock 时间释放帧）

### 经验教训

1. **debug server 的 chunk_size 必须与真实 policy 一致**，否则 AsyncInferenceClient 的参数推导、降级逻辑、K_inf 裁剪都会走完全不同的路径，测试结果没有参考价值。
2. **降级模式下推理线程几乎无间隔循环**，对 debug server 这种"每次请求消耗状态"的 adapter 来说，会导致 dataset 被快速耗尽。
3. **测试异步推理系统时，优先使用真实 policy**（测试 1.4），debug server 仅用于验证连通性。

---

<a id="bug-006"></a>
## BUG-006: 录制数据 state 呈方波（已修复）

**严重程度**: 高
**发现日期**: 2026-02-10
**状态**: 已修复

### 现象

录制的 `observation.state` 所有维度呈阶梯/方波，每 ~33 帧跳变一次，中间值完全不变。97% 的帧间差异为零。

### 真机数据

```
DAgger 录制: 511/527 帧零差异 (97%), 16 次跳变，间隔 ~33 帧
VR 遥操对比: 8/694 帧零差异 (1.2%), 686 次连续变化
跳变间隔 = T_inter × f_exec ≈ 1.1s × 30Hz = 33 帧
```

### 根因

执行循环（30Hz 主线程）中缺少 `_update_state()` 调用。`current_joints` / `current_ee_pose` / `current_gripper` 只在推理线程的 `get_observation()` 中更新（每 ~1.1s = T_inter 一次），录制每帧读到的是过时缓存。

### 诊断方法

```python
import pandas as pd, numpy as np
df = pd.read_parquet('data/chunk-000/episode_000000.parquet')
states = np.array([df.iloc[i]['observation.state'] for i in range(len(df))])
diffs = np.diff(states[:, 0])  # joint_1
zero_pct = np.sum(np.abs(diffs) < 1e-6) / len(diffs) * 100
print(f'零差异帧: {zero_pct:.1f}%')  # 正常应 <5%，方波时 >90%
```

### 修复

`async_inference_client.py` 执行循环开头加入 `_update_state()` 调用：

```python
# 0. 更新状态缓存（每帧刷新，确保录制拿到最新状态）
if not self.config.dry_run:
    self._update_state()
```

性能开销：~5ms/帧（UDP 路径），30Hz 下占 15% 预算，可接受。

### 经验教训

1. **录制数据的更新频率取决于状态缓存的刷新频率**，而不是录制循环的频率。如果缓存只在推理线程中更新，录制频率再高也只能拿到推理频率的数据。
2. **方波的跳变间隔直接暴露了根因**：33 帧 = T_inter × f_exec，说明状态只在推理时更新。

---

<a id="pitfall-007"></a>
## PITFALL-007: EEF rotation 数据精度低（阶梯状，非 bug）

**严重程度**: 无（硬件限制）
**发现日期**: 2026-02-10
**状态**: 已知局限，非 bug

### 现象

在 rerun 中查看录制数据，`observation.state` 的后 3 维（erx/ery/erz，索引 11-13）呈明显阶梯状，大量连续帧值相同，偶尔跳变 0.001。而前 3 维位置（ex/ey/ez，索引 8-10）则平滑连续。

### 这不是 bug

是 Realman RM65-B SDK/UDP 的硬件精度限制：

| 维度 | 精度 | 说明 |
|------|------|------|
| ex/ey/ez (位置) | 0.000001 m (1μm) | 6 位小数，平滑 |
| erx/ery/erz (姿态) | **0.001 rad (~0.057°)** | 3 位小数，阶梯状 |

### 验证

VR 遥操数据（`realman_teleop_vr_0125_01`）的 rotation 精度也是 0.001 rad，完全一致。两套系统读取的是同一个 SDK 接口，精度相同。

### 量化分析

```python
import pandas as pd, numpy as np
df = pd.read_parquet('data/chunk-000/episode_000000.parquet')
states = np.array([df.iloc[i]['observation.state'] for i in range(len(df))])

# erx unique values 间距恒为 0.001
erx = states[:, 11]
unique_erx = np.unique(erx)
print(f'unique count: {len(unique_erx)}')
print(f'min diff: {np.min(np.diff(unique_erx)):.6f}')  # → 0.001000

# 对比 ex (位置) 精度
ex = states[:, 8]
unique_ex = np.unique(ex)
print(f'ex min diff: {np.min(np.diff(unique_ex)):.6f}')  # → 0.000001
```

### 注意

erz 可能出现 ±π 环绕跳变（如 `3.152 → -3.128`，delta ≈ 6.28），这是角度表示的正常现象，不是数据错误。训练时如果使用 rotation 作为特征，需要考虑角度 wrap-around 处理（参见 `angle_diff()` 函数）。

---

<a id="pitfall-008"></a>
## PITFALL-008: rerun 中看不到 policy_action 字段

**严重程度**: 低
**发现日期**: 2026-02-10
**状态**: 已修复

### 现象

用 `visualize_dataset.py` 生成 .rrd 文件后，rerun 中只有 state/action/images，没有 policy_action 和 control_source。

### 根因

原版 `visualize_dataset.py`（LeRobot 上游）只硬编码了 `action`、`observation.state`、`next.done`、`next.reward`、`next.success` 的渲染，DAgger 新增的 `policy_action` 和 `control_source` 字段未被处理。

### 修复

已在 `RoboCOIN/src/lerobot/scripts/visualize_dataset.py` 中添加：

```python
if "policy_action" in batch:
    for dim_idx, val in enumerate(batch["policy_action"][i]):
        rr.log(f"policy_action/{dim_idx}", rr.Scalar(val.item()))

if "control_source" in batch:
    rr.log("control_source", rr.Scalar(batch["control_source"][i].item()))
```

### rerun 中的位置

左侧面板 → `policy_action/0` ~ `policy_action/7`（8D 策略原始输出）、`control_source`（0=policy, 1=human）。

---

<a id="pitfall-009"></a>
## PITFALL-009: visualize_dataset.sh 路径配置

**严重程度**: 低
**发现日期**: 2026-02-10

### 用法

修改 `visualize_dataset.sh` 中的 `--repo-id` 为数据集的**绝对路径**即可（脚本会自动将绝对路径解析为 root + repo_id）：

```bash
python visualize_dataset.py \
  --repo-id /absolute/path/to/dataset_dir \
  --episode-index 0 \
  --tolerance-s 0.04 \
  --save 1 \
  --output-dir ./output
```

然后用 `rerun output/dataset_dir_episode_0.rrd` 查看。

---

<a id="bug-010"></a>
## BUG-010: ROS2 空字符串参数导致 dagger_node 启动崩溃

**严重程度**: 高
**发现日期**: 2026-02-11
**状态**: 已修复

### 现象

`ros2 launch dagger/launch/dagger.launch.py` 启动后，dagger_node 在 2~3 秒内崩溃（exit code 1），Web UI 中 DAgger 显示离线。其他节点（driver、VR input、cameras）正常运行。

### 错误信息

```
[ERROR] [rcl]: Failed to parse global arguments
rclpy._rclpy_pybind11.RCLError: failed to initialize rcl: Couldn't parse parameter override rule: '-p policy_type:='. Error: error not set, at ./src/rcl/arguments.c:352
```

### 根因

`dagger.launch.py` 中构建 `--ros-args -p key:=value` 参数列表时，当 `policy_server.auto_launch=false`（默认），`policy_type` 和 `pretrained_path` 被设为空字符串 `""`：

```python
"policy_type": ps_policy_type if auto_launch else "",
"pretrained_path": ps_pretrained_path if auto_launch else "",
```

参数构建循环将其生成为 `-p policy_type:=`（冒号等号后无值），ROS2 的 `rcl` 参数解析器无法处理空值，导致 `rclpy.init()` 失败。

### 修复

在参数构建循环中跳过空字符串值：

```python
for key, value in dagger_node_params.items():
    # ROS2 不支持空值参数（-p key:= 会报错），跳过空字符串
    if isinstance(value, str) and value == "":
        continue
    # ... 正常处理其他类型 ...
```

### 排查方法

1. launch.log 中只有 `process has died [exit code 1]`，无 stderr（因为 `ExecuteProcess` 的 stderr 输出到 screen 而非 log 文件）
2. 手动复现：从 launch.log 中复制完整 cmd，在终端直接执行，即可看到完整 traceback
3. 关键线索：cmd 末尾的 `-p policy_type:= -p pretrained_path:=`（空值）

### 经验教训

1. **ROS2 参数不支持空字符串值**。`-p key:=` 会导致 `rclpy.init()` 失败，必须跳过或提供默认值。
2. **`ExecuteProcess` 的 stderr 不写入 launch.log**（只写入 screen），排查崩溃时需要手动复现命令才能看到完整错误。
3. **launch.log 中的 cmd 字段是排查利器**——直接复制到终端执行即可复现问题。

---

<a id="bug-011"></a>
## BUG-011: DAggerRecorder 目录已存在导致初始化错误刷屏

**严重程度**: 中
**发现日期**: 2026-02-11
**状态**: 已修复

### 现象

启动 DAgger 会话后，日志中每 ~33ms（30Hz 录制频率）刷屏报错：

```
[ERROR] Failed to init DAggerRecorder: [Errno 17] File exists: '.../data/local/dagger_realman_0211-test'
```

错误持续不断，直到进程被杀死。录制功能完全不可用。

### 根因

两个问题叠加：

1. **`LeRobotDataset.create()` 不允许目录已存在**：`LeRobotDatasetMetadata.create()` 内部调用 `obj.root.mkdir(parents=True, exist_ok=False)`（`lerobot_dataset.py:329`），当数据集目录已存在时抛出 `FileExistsError`。

2. **`_init_recorder_from_sync()` 无限重试**：该方法在每个录制 tick（~30Hz）被调用，失败后只打印 error 日志，不设置任何失败标志，下一个 tick 继续尝试 → 无限循环报错。

触发场景：上一次会话异常退出（如 Ctrl+C 或崩溃），数据集目录残留未清理，下次启动时 repo_id 相同。

### 影响

- 日志被错误信息淹没（~30 条/秒），掩盖其他重要日志
- 录制功能完全失效（recorder 永远无法初始化）
- 不影响推理和 VR 控制（录制是独立模块）

### 修复（两处）

**1. `data_recorder.py` — DAggerRecorder 构造函数：自动清理残留目录**

```python
# 如果目录已存在（上次异常退出残留），清理后重建
if dataset_path is not None and dataset_path.exists():
    logger.warning(f"数据集目录已存在，清理旧数据: {dataset_path}")
    shutil.rmtree(dataset_path)
```

**2. `dagger_node.py` — `_init_recorder_from_sync()`：失败后不再重试**

```python
self._recorder_init_failed = False  # 构造函数初始化

def _init_recorder_from_sync(self, ...):
    if self._recorder_init_failed:
        return  # 已失败，不再重试
    try:
        self._recorder = DAggerRecorder(...)
        self._recorder_initialized = True
    except Exception as e:
        self._recorder_init_failed = True  # 标记失败，阻止后续重试
        self.get_logger().error(f"DAggerRecorder 初始化失败（不再重试）: {e}")
```

新会话 `_handle_start_session()` 时会重置 `_recorder_init_failed = False`，允许新会话重新尝试。

### 数据集录制逻辑变化

修复前：
- 如果目录已存在 → `FileExistsError` → 每 tick 重试 → 永远失败

修复后：
- 如果目录已存在 → 自动清理旧目录 → 重新创建 → 正常录制
- 如果因其他原因失败 → 标记 `_recorder_init_failed` → 不再重试 → 日志只报一次错
- 新会话开始时重置失败标志 → 允许重新初始化

### 经验教训

1. **lazy-init 的重试必须有上限或失败标志**。在高频回调（30Hz）中无限重试同一个必然失败的操作，会产生灾难性的日志刷屏。
2. **`LeRobotDataset.create()` 的 `exist_ok=False` 是有意设计**（防止覆盖数据），但 DAgger 场景下异常退出是常态，需要在上层处理残留目录。
3. **异常退出后的状态清理是必须考虑的**。机器人系统经常被 Ctrl+C 或意外断电终止，所有持久化状态（文件、目录、锁文件）都需要有恢复策略。

---

<a id="bug-012"></a>
## BUG-012: WebUI 前端 const srv 重复声明导致整个 UI 不更新

**严重程度**: 高
**发现日期**: 2026-02-11
**状态**: 已修复

### 现象

打开 WebUI 控制面板后，节点状态区域为空（看不到任何节点），模式徽章不更新，Server 状态不显示，推理/录制详情全部空白。后端 API `/api/status` 返回数据完全正常。

### 根因

`control_panel.py` 内嵌的 JS `updateStatus()` 函数中，`const srv` 在同一作用域内被声明了两次：

```javascript
// 第 339 行 — Server 状态卡片
const srv = dagger.server || {};
// ... 中间 ~55 行代码 ...
// 第 394 行 — 推理详情
const srv = dagger.server || {};  // SyntaxError!
```

JavaScript 的 `const` 不允许在同一块作用域内重复声明。浏览器解析 `updateStatus()` 时直接抛出 `SyntaxError`，导致整个函数**从未执行**。由于 `updateStatus()` 负责所有 UI 元素的更新（模式徽章、Server 状态、节点列表、推理详情、录制详情、日志），所以整个页面处于初始空白状态。

### 影响

- 所有 UI 元素不更新（节点、Server、模式、推理、录制、日志）
- 后端完全正常，仅前端渲染失效
- 浏览器 Console 中可以看到 `SyntaxError: Identifier 'srv' has already been declared`

### 修复

删除第 394 行重复的 `const srv = dagger.server || {};`，复用第 339 行已声明的 `srv` 变量。

### 经验教训

1. **内嵌 JS 模板没有 lint 检查**。Python 文件中的 JS 字符串不会被 ESLint/IDE 检查，`const` 重复声明这种低级错误无法被自动发现。
2. **`SyntaxError` 是静默的**。浏览器不会在页面上显示 JS 语法错误，只在 Console 中报告。如果不打开 DevTools，完全看不到错误原因。
3. **前端调试第一步：打开浏览器 Console**。遇到 UI 不更新时，先检查 Console 有无 JS 错误。

---

<a id="bug-013"></a>
## BUG-013: POLICY 模式机械臂运动速度异常（5x 快于预期）

**严重程度**: 高
**发现日期**: 2026-02-11
**状态**: 已修复（3 个独立 bug 叠加）

### 现象

`dagger_node` 在 POLICY 模式下执行 pose 动作时，机械臂运动速度远快于预期（约 5 倍），且运动轨迹不正确。同一 PolicyServer + 同一配置下，`async_inference_client` 运动完全正常。

### 根因（3 个独立 bug 叠加）

#### Bug 13a: 执行频率错误（50Hz 而非 10Hz）

`_tick_policy()` 原本在 `_execution_tick()` 定时器回调中执行，该定时器以 `control_hz`（默认 50Hz）运行。虽然有 `_policy_T_step` 节流逻辑试图限制到 `f_exec`（10Hz），但定时器回调本身就以 50Hz 触发，节流精度受限于定时器分辨率。

**实际效果**：policy 动作以 ~50Hz 执行，而非配置的 10Hz，导致 action chunk 被 5 倍速消耗。

**修复**：将 policy 执行从定时器回调中剥离，改为独立的 `_policy_exec_loop()` 线程，使用 `while + sleep` 精确控制 `f_exec` 频率（与 `async_inference_client` 架构一致）。

```python
def _policy_exec_loop(self):
    """独立线程：以 f_exec 频率执行 policy 动作（async_client 风格）"""
    T_step = 1.0 / self.f_exec
    while not self._policy_exec_stop.is_set():
        t_start = time.monotonic()
        action = self._ring_buffer.pop_current()
        self._tick_policy()
        elapsed = time.monotonic() - t_start
        sleep_time = T_step - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
```

**架构变化**：
- `_execution_tick()`（50Hz 定时器）在 POLICY 模式下只负责更新推理观测（`_update_inference_observation()`），不再执行动作
- `_policy_exec_loop()`（独立线程）以 `f_exec` 频率执行动作，线程生命周期与 POLICY 模式绑定
- 进入 POLICY 模式时 `_start_policy_exec_loop()`，离开时 `_stop_policy_exec_loop()`
- 共 10 个模式切换点需要正确调用 start/stop

#### Bug 13b: SDK 参数不一致（canfd_radio）

`teleop_params.yaml` 中 `canfd_radio: 60`，而 `async_inference_client` 使用的 SDK 调用中 `canfd_radio=0`。

`canfd_radio` 是 Realman SDK 的平滑系数参数：
- `0` = 完全透传（动作直接执行）
- `60` = 曲线拟合平滑（SDK 内部插值）

在 `canfd_traj_mode=0`（透传模式）下，`canfd_radio` 理论上被忽略，但实测发现非零值仍会影响运动行为。

**修复**：`teleop_params.yaml` 中 `canfd_radio` 改为 `0`。

#### Bug 13c: Pose 格式误判（7 值欧拉被当作四元数）

`realman_driver_node.py` 的 `_on_action_pose()` 回调原本仅通过 `len(msg.position)` 判断 pose 格式：

```python
# 旧代码（仅长度判断）
if len(msg.position) == 7:
    # 误判为四元数格式 [x, y, z, qw, qx, qy, qz]
    pose = list(msg.position[:7])
```

dagger_node 发送的是 7 值消息 `[x, y, z, rx, ry, rz, gripper]`（6D 欧拉角 + 夹爪），被 driver 误判为四元数格式 `[x, y, z, qw, qx, qy, qz]`。

**后果**：`rx` 被当作 `qw`，`ry` 被当作 `qx`，`rz` 被当作 `qy`，`gripper` 被当作 `qz`。发送给 SDK 的旋转值完全错误，导致机械臂运动方向和幅度异常。

**修复**：通过 `msg.name` 字段显式识别格式（优先），回退到长度判断（向后兼容）：

```python
# 新代码（msg.name 显式识别）
names = list(msg.name) if msg.name else []
if "rx" in names:
    # 欧拉角格式：[x, y, z, rx, ry, rz, (gripper)]
    pose = list(msg.position[:6])
    gripper_idx = 6
elif "qw" in names:
    # 四元数格式：[x, y, z, qw, qx, qy, qz, (gripper)]
    pose = list(msg.position[:7])
    gripper_idx = 7
```

**关键**：此修复在 `realman_driver_node.py`（ROS2 包源码），必须 `colcon build` 才能同步到 `install/` 目录。如果跳过 build，运行的仍是旧代码。

### 为什么 async_inference_client 不受影响？

| 维度 | async_inference_client | dagger_node（修复前） |
|------|----------------------|---------------------|
| 执行频率 | 独立 while+sleep 线程，精确 f_exec | 50Hz 定时器回调，节流不精确 |
| canfd_radio | 直接调用 SDK，radio=0 | 通过 driver 转发，radio=60 |
| Pose 格式 | 直接调用 SDK（无格式判断） | 通过 ROS2 JointState 消息，driver 需判断格式 |

三个 bug 都是 dagger_node 特有的（通过 ROS2 driver 间接控制），async_inference_client 直接调用 SDK 所以不受影响。

### 诊断方法

1. **频率诊断**：创建 `test_exec_frequency_diag.py`，独立测试 while+sleep 循环的频率精度。结果：9.99Hz（0.12% 误差），确认 sleep 方案可行。

2. **对比测试**：同一 PolicyServer + 同一配置，分别用 async_inference_client 和 dagger_node 执行，对比运动速度和轨迹。

3. **driver 日志**：在 `_on_action_pose()` 中打印 `msg.name` 和解析结果，确认格式判断是否正确。

### 经验教训

1. **定时器回调不适合精确频率控制**。ROS2 定时器的实际触发频率受 executor 调度影响，在高负载下可能偏差较大。需要精确频率控制时，应使用独立线程 + `time.sleep()`。

2. **通过中间层（ROS2 driver）转发动作时，必须确保参数完全一致**。直接调用 SDK 和通过 driver 转发是两条不同的代码路径，参数（如 `canfd_radio`）可能不一致。

3. **消息格式判断不能仅靠长度**。7 值消息可以是 `[x,y,z,rx,ry,rz,gripper]`（欧拉+夹爪）也可以是 `[x,y,z,qw,qx,qy,qz]`（四元数），长度相同但含义完全不同。必须通过显式字段（如 `msg.name`）区分。

4. **多个 bug 叠加时现象难以定位**。单独看每个 bug 的影响（5x 频率、平滑参数、旋转错误）都不足以解释全部现象，但叠加后产生了"速度异常 + 轨迹错误"的复合效果。排查时需要逐一隔离变量。

5. **`colcon build` 是 ROS2 包更新的必要步骤**。修改 `realman_teleop` 包源码后，如果不 build，`install/` 目录中的旧代码仍会被加载。这与 `dagger_node.py`（通过 ExecuteProcess 直接运行源码）的行为不同，容易混淆。

---

<a id="bug-014"></a>
## BUG-014: Safety check 导致 delta_joints 模型 action chunk 被全部拒绝

**严重程度**: 高
**发现日期**: 2026-02-11
**状态**: 已修复（移除多余的 safety check）

### 现象

使用 `delta_joints` 模型时，机械臂移动一半就停止不动。日志中显示连续的 safety reject 警告，delta 值从 5.4° 逐渐增长到 78.7°+。

```
Safety reject #1:  joint 3 delta=5.4deg
Safety reject #2:  joint 3 delta=11.2deg
Safety reject #3:  joint 3 delta=18.5deg
...
Safety reject #10: joint 3 delta=78.7deg
```

关键观察：`last_executed_joints` 在所有 reject 中保持不变（始终是初始位置），而 `target_joints` 不断增长。

### 根因

问题是 **`_safety_check_joints()` 函数的设计不适用于 `delta_mode: "relative"` 模型**。

#### delta_mode: "relative" 模型的输出特性

模型输出的 50 个 action 是一个**累进轨迹**：

```
action[0] = observation_state + delta[0]
action[1] = observation_state + delta[0] + delta[1]
action[2] = observation_state + delta[0] + delta[1] + delta[2]
...
action[49] = observation_state + sum(delta[0:50])
```

每个 action 是相对于**观测时刻状态**的累进偏移，不是独立的目标位置。

#### safety check 的错误逻辑

原始 safety check 逻辑：

```python
def _safety_check_joints(self, target_joints):
    delta = abs(target_joints - self._last_executed_joints)
    if max(delta) > max_joint_delta_deg:
        return False  # 拒绝执行，不更新 _last_executed_joints
    return True
```

问题链：

```
1. 初始状态：_last_executed_joints = P0（观测时刻位置）

2. action[0] 到达：target = P0 + 5.4°
   → delta = 5.4° > 5° 限制 → 拒绝
   → _last_executed_joints 仍为 P0（未更新！）

3. action[1] 到达：target = P0 + 11.2°
   → delta = |P0 + 11.2° - P0| = 11.2° > 5° → 拒绝
   → _last_executed_joints 仍为 P0

4. action[2] 到达：target = P0 + 18.5°
   → delta = 18.5° > 5° → 拒绝
   ...

5. 所有 50 个 action 都被拒绝
   → 机械臂完全停止
   → 下一个 chunk 的 action 基于新观测，但机械臂没移动，delta 更大
   → 恶性循环
```

#### 为什么"移动一半停了"？

action chunk 的前几个 action（delta < 5°）可能通过了检查并执行，之后的 action 全部被拒绝。所以看到的现象是"移动一小段后停止"。

### 为什么 async_inference_client 不受影响？

`async_inference_client.py` **没有 safety check 逻辑**，直接把每个 action 发送给机械臂 SDK 执行。

```bash
$ grep -E "safety|max_joint_delta|delta_deg" async_inference_client.py
# 无匹配结果
```

模型在正确的执行频率（`f_exec`）下输出的 action delta 是经过训练验证的，不需要额外的安全限制。

### 修复

**移除 `_safety_check_joints()` 调用**，让 action 直接执行（与 async_inference_client 一致）：

```python
# 修复前（有 safety check）
if not self._safety_check_joints(joint_rad):
    return  # 拒绝执行
self._last_executed_joints = joint_rad.copy()
self._publish_joint_action(joint_rad.tolist(), gripper)

# 修复后（无 safety check，直接执行）
self._last_executed_joints = joint_rad.copy()
self._publish_joint_action(joint_rad.tolist(), gripper)
```

`_safety_check_joints()` 函数保留但不再被调用，可作为未来可选功能。

### 替代方案（如需保留安全限制）

如果确实需要安全限制，正确的做法是 **clip 而不是 reject**：

```python
def _safety_check_joints(self, target_joints) -> tuple[bool, np.ndarray]:
    delta = target_joints - self._last_executed_joints
    max_delta = np.max(np.abs(np.degrees(delta)))

    if max_delta > self.max_joint_delta_deg:
        # Clip delta 到安全范围，而不是完全拒绝
        max_rad = np.radians(self.max_joint_delta_deg)
        clipped_delta = np.clip(delta, -max_rad, max_rad)
        clipped_joints = self._last_executed_joints + clipped_delta
        return False, clipped_joints  # 返回 clipped 后的位置

    return True, target_joints
```

这样机械臂会以安全速度（max 5°/帧）逐步追赶目标轨迹，而不是完全停止。

### 经验教训

1. **不要添加上游系统没有的"安全"逻辑**。`async_inference_client` 正常工作了很久，`dagger_node` 应该复制其行为，而不是自作主张添加限制。

2. **累进轨迹 ≠ 独立目标**。`delta_mode: "relative"` 模型输出的 action chunk 是时间序列轨迹，每个 action 的"delta"是相对于观测时刻的，不是相对于上一个 action 的。

3. **"拒绝并跳过"的安全策略在轨迹执行中是灾难性的**。一旦第一个 action 被拒绝，后续所有 action 的 delta 只会更大（因为机械臂没有移动），导致连锁拒绝。正确的策略是 "clip 并执行"。

4. **调试技巧：对比正常工作的系统**。当 dagger_node 行为异常时，首先检查它与 async_inference_client 的代码差异，找出新增的逻辑。

---

<a id="bug-015"></a>
## BUG-015: .pth 路径错误 + delta_mask/delta_mode 缺失导致机械臂失控

**严重程度**: 致命
**发现日期**: 2026-02-12
**状态**: 已修复

### 现象

POLICY 模式下机械臂运动完全失控：
- 关节角度剧烈跳变（delta 从 1° 增长到 160°）
- gripper 值爆炸（0.8 → 12.7 → 23.16 → 31+，正常范围 0~1）
- 机械臂快速甩动，必须紧急停止

### 根因（两层问题叠加）

#### 第一层：.pth 文件指向旧代码

系统 Python 的 `lerobot` 包通过 editable install 的 `.pth` 文件加载：

```
/home/ubuntu/.local/lib/python3.12/site-packages/__editable__.robocoin-0.1.0.1.pth
```

该文件指向旧版目录 `/home/ubuntu/Desktop/Workspace/lerobot_policy_deploy/RoboCOIN/src`，而非修复后的 `lerobotv3/RoboCOIN/src`。即使修改了 `lerobotv3` 下的代码，PolicyServer 实际加载的仍是旧版。

#### 第二层：旧版 convert_delta_to_absolute() 有两个致命 bug

**Bug 15a: 缺少 delta_mask 处理**

模型的 `train_config.json` 指定：
```json
{
  "delta_mask": [true, true, true, true, true, true, true, false]
}
```

含义：前 7 维（关节）是 delta，第 8 维（gripper）是 absolute。

旧代码将**所有维度**当作 delta 累加：

```python
# 旧代码（错误）
for i in range(1, len(delta_actions)):
    absolute_actions[i] = absolute_actions[i-1] + delta_actions[i]  # 全部累加，包括 gripper
```

gripper 值被错误累加：`0.8 + 0.695 + 0.6 + 0.605 + ... → 31.3`

**Bug 15b: 错误的 delta_mode**

模型使用 `delta_mode: "relative"`（每步 delta 相对于 current_state），但旧代码使用链式累加（frame_diff 模式）：

```python
# 旧代码（错误）— 链式累加
absolute_actions[i] = absolute_actions[i-1] + delta_actions[i]

# 正确 — relative 模式
absolute_actions[i] = current_state + delta_actions[i]
```

链式累加导致 delta 在 chunk 内指数增长：
```
action[0] = state + 0.02 rad (1.1°)    ← 正常
action[1] = action[0] + 0.02 = state + 0.04 rad (2.3°)
action[2] = action[1] + 0.02 = state + 0.06 rad (3.4°)
...
action[49] = state + 1.0 rad (57°)     ← 完全失控
```

而 relative 模式下每个 action 都是 `state + delta[i]`，delta 保持在 1~4° 范围内。

### 修复（3 处）

**1. 更新 .pth 文件**

```bash
# 修改前
cat __editable__.robocoin-0.1.0.1.pth
# → /home/ubuntu/Desktop/Workspace/lerobot_policy_deploy/RoboCOIN/src

# 修改后
echo "/home/ubuntu/Desktop/Workspace/lerobot_policy_deploy/lerobotv3/RoboCOIN/src" > \
  /home/ubuntu/.local/lib/python3.12/site-packages/__editable__.robocoin-0.1.0.1.pth
```

**2. base_adapter.py — 重写 convert_delta_to_absolute()**

```python
def convert_delta_to_absolute(self, delta_actions, current_state, delta_mask=None, delta_mode="relative"):
    if delta_mask is None:
        delta_mask = np.ones(delta_actions.shape[1], dtype=bool)

    absolute_actions = np.zeros_like(delta_actions)

    if delta_mode == "relative":
        # relative: 每步 delta 相对于 current_state
        for i in range(len(delta_actions)):
            absolute_actions[i] = np.where(
                delta_mask,
                current_state + delta_actions[i],   # delta 维度：加 current_state
                delta_actions[i]                      # absolute 维度：直接使用
            )
    else:
        # frame_diff: 逐帧累加（向后兼容旧模型）
        absolute_actions[0] = np.where(delta_mask, current_state + delta_actions[0], delta_actions[0])
        for i in range(1, len(delta_actions)):
            absolute_actions[i] = np.where(
                delta_mask,
                absolute_actions[i-1] + delta_actions[i],
                delta_actions[i]
            )
    return absolute_actions
```

新增方法：
- `_infer_delta_mode()`: 从 `train_config.json` 读取 `delta_mode`
- `_load_delta_mask()`: 从 `train_config.json` 读取 `delta_mask`

**3. act_adapter.py — 传入 delta_mask 和 delta_mode**

```python
if self.is_delta_model:
    delta_mask = self._load_delta_mask()
    action_np = self.convert_delta_to_absolute(
        action_np, current_state, delta_mask, delta_mode=self.delta_mode
    )
```

### 验证方法

PolicyServer 启动时检查日志：

```
[Adapter] 从 train_config 读取 delta_mode: relative          ← 必须是 relative
[Adapter] 从 train_config 加载 delta_mask: [ True  True  True  True  True  True  True False]
[Adapter]    Delta 维度（累加）: [0, 1, 2, 3, 4, 5, 6]       ← 7 个关节
[Adapter]    绝对值维度（直接使用）: [7]                       ← gripper
```

运行时检查：
- gripper 值应在 0~1 范围内（如 0.605, 0.624）
- 关节 delta 应在 1~4° 范围内（不应超过 10°）

### 快速诊断

如果机械臂运动失控，按以下顺序排查：

```bash
# 1. 检查 .pth 指向
cat /home/ubuntu/.local/lib/python3.12/site-packages/__editable__.robocoin-0.1.0.1.pth
# 应该指向 lerobotv3/RoboCOIN/src

# 2. 检查 PolicyServer 日志中的 delta_mode 和 delta_mask
# 应该看到 "delta_mode: relative" 和 "Delta Mask: [T,T,T,T,T,T,T,F]"

# 3. 检查推理输出
# gripper > 1 → delta_mask 未生效（gripper 被累加）
# 关节 delta > 10° → delta_mode 错误（链式累加而非 relative）
```

### 经验教训

1. **editable install (.pth) 是隐形陷阱**。`pip install -e .` 创建的 `.pth` 文件决定实际加载的代码路径。复制项目目录后，`.pth` 仍指向旧路径，修改新目录的代码不会生效。排查时必须确认 `.pth` 指向正确。

2. **delta 转换必须尊重 train_config.json 的完整配置**。`delta_mask` 和 `delta_mode` 是模型训练时的关键参数，推理时必须完全一致。缺少任何一个都会导致动作输出完全错误。

3. **gripper 值 >1 是 delta_mask 缺失的标志性症状**。正常 gripper 范围是 0~1（absolute），如果看到 >1 的值，几乎一定是 gripper 被错误地当作 delta 累加了。

4. **关节 delta 指数增长是 delta_mode 错误的标志性症状**。relative 模式下每步 delta 应该稳定在训练时的范围（1~4°），如果看到 delta 随 chunk 内 index 增长，说明使用了错误的链式累加模式。

---

## <a name="pitfall-016"></a>PITFALL-016: HuggingFace datasets Arrow 缓存累积撑满磁盘

**发现日期**: 2026-02-12
**影响范围**: DAgger 录制 + VR 数据采集（所有使用 LeRobotDataset 的场景）
**严重程度**: 高（可导致磁盘空间耗尽，系统不可用）

### 现象

- 长时间运行后磁盘被占满
- `~/.cache/huggingface/datasets/` 目录下出现大量以哈希命名的文件夹
- 文件夹内是 `.arrow` 格式的缓存文件，单个可达数百 MB

### 根本原因

HuggingFace `datasets` 库在以下操作时会在 `~/.cache/huggingface/datasets/` 下创建 Arrow 格式缓存，且**永远不会自动清理**：

1. **`_save_episode_table()`**（`lerobot_dataset.py:1053`）：每次保存 episode 时调用 `datasets.Dataset.from_dict()` + `concatenate_datasets()`，每次 `concatenate_datasets` 都会创建一个包含所有已录制数据的新 Arrow 文件。

2. **`load_hf_dataset()`**（`lerobot_dataset.py:619`）：调用 `load_dataset("parquet", data_dir=path, split="train")`，将 parquet 文件转为 Arrow 缓存。

**增长模式**：录制 N 个 episode 后，缓存中有 N 个递增大小的 Arrow 文件（第 i 个包含前 i 个 episode 的合并数据），总大小约 O(N²)。

### 涉及代码

```
# DAgger 录制
dagger/core/data_recorder.py → DAggerRecorder → LeRobotDataset.create()

# VR 数据采集
vr_teleop/.../dataset_recorder_node.py → LeRobotDataset.create()

# LeRobot 核心
RoboCOIN/src/lerobot/datasets/lerobot_dataset.py:
  - L619: load_dataset("parquet", ...) → Arrow 缓存
  - L1053: concatenate_datasets([...]) → Arrow 缓存
```

### 解决方案

**定期手动清理**（推荐）：

```bash
# 安全删除 HF datasets 缓存（不影响已保存的数据集）
rm -rf ~/.cache/huggingface/datasets/

# 检查缓存大小
du -sh ~/.cache/huggingface/datasets/
```

**录制结束后清理**（可选，加入脚本）：

```python
import shutil
from pathlib import Path

cache_dir = Path.home() / ".cache/huggingface/datasets"
if cache_dir.exists():
    shutil.rmtree(cache_dir)
    print(f"已清理 HF datasets 缓存: {cache_dir}")
```

### 注意事项

- `~/.cache/huggingface/datasets/` 是 HF `datasets` 库的内部缓存，与你的数据集存储目录（`dataset_root`）无关，删除不会丢失已录制的数据
- `~/.cache/huggingface/hub/` 是模型缓存，不要误删
- 长时间录制（如过夜测试）前建议先清理一次缓存，确保磁盘有足够空间

5. **两个 bug 叠加时现象更难定位**。单独看 delta_mask 缺失（gripper 爆炸）或 delta_mode 错误（关节增长），各自的症状是明确的。但叠加后产生的"完全失控"现象容易让人误以为是更底层的问题（如模型损坏、通信错误）。排查时应逐一检查每个环节。

---

<a id="bug-016"></a>
## BUG-016: DAggerRecorder lazy init 阻塞 SingleThreadedExecutor 导致观测冻结 ~1.7s

**严重程度**: 高
**发现日期**: 2026-02-12
**状态**: 已修复（后台线程初始化）

### 现象

UI 路径（dagger_node）启动 POLICY 模式后，机械臂前几步出现明显的跳变/回退动作。诊断日志显示：

```
infer#0: obs_age=1.6ms,   obs_count=2   ← 正常（首次观测）
infer#1: obs_age=73.2ms,  obs_count=2   ← obs_count 不变！
infer#2: obs_age=823.3ms, obs_count=2   ← 观测冻结 823ms
infer#3: obs_age=1712.3ms,obs_count=2   ← 冻结 1.7s
infer#4: obs_age=1.4ms,   obs_count=20, max_delta=20.8°  ← 突然恢复，巨大跳变
```

obs_count 在 infer#0~#3 期间始终为 2，说明 `_update_inference_observation()` 在 ~1.7s 内从未被调用。

### 根因

`DAggerRecorder` 的 lazy init 在 `_on_recording_sync_inner()` 回调中执行，该回调运行在 ROS2 SingleThreadedExecutor 线程上。`DAggerRecorder()` 构造函数内部调用 `LeRobotDataset.create()`，涉及目录创建、元数据写入等 I/O 操作，耗时 ~1.7s。

在此期间，SingleThreadedExecutor 被完全阻塞：
- `_execution_tick()` 定时器无法触发 → `_update_inference_observation()` 不被调用
- `_on_arm_state()` 订阅回调无法触发 → 状态缓存不更新
- `_on_image()` 订阅回调无法触发 → 图像缓存不更新

推理线程（独立线程）继续运行，但只能读到冻结的观测（obs_count=2），基于过时状态生成 action。当 recorder init 完成后，executor 恢复，新鲜观测涌入，模型发现机械臂已移动很远（max_delta=20.8°），产生大幅修正动作。

### 时间线证据

```
T=457.897: exec loop 启动，warmup 完成
T=457.932: exec#0 开始执行（观测正常）
T=459.583: DAggerRecorder initialized (from sync)  ← 1.7s 后！
           obs_age=1712ms ≈ 1.7s，完美匹配
```

### 修复

将 `DAggerRecorder` 初始化移到后台线程，不阻塞 executor：

```python
# 修复前（阻塞 executor ~1.7s）
if not self._recorder_initialized:
    self._init_recorder_from_sync(state_14d, images_rgb)  # 阻塞！
    ...

# 修复后（后台线程初始化）
if not self._recorder_initialized and not self._recorder_init_in_progress and not self._recorder_init_failed:
    self._recorder_init_in_progress = True

    def _background_init():
        self._init_recorder_from_sync(state_14d, images_rgb)
        if self._recorder is not None:
            with self._recorder_lock:
                self._recorder.start_episode(task=self.recording_task)
        self._recorder_init_in_progress = False

    init_thread = threading.Thread(target=_background_init, daemon=True, name="RecorderInit")
    init_thread.start()
    return  # 立即返回，不阻塞 executor

# 初始化进行中，跳过本帧录制
if self._recorder_init_in_progress:
    return
```

新增 `_recorder_init_in_progress` 标志防止多次触发后台初始化。初始化期间跳过录制帧（丢失 ~1.7s 的录制数据，可接受）。

### 经验教训

1. **ROS2 SingleThreadedExecutor 中的任何回调都不能执行耗时操作**。所有定时器、订阅、服务回调共享同一线程，任何一个阻塞都会冻结全部。耗时操作必须移到后台线程。

2. **obs_count 不变是观测冻结的标志性症状**。如果推理线程的 obs_count 在多次推理间不增长，说明 `update_observation()` 没有被调用，应排查 executor 是否被阻塞。

3. **obs_age 与阻塞时间的匹配是定位根因的关键**。1712ms obs_age ≈ 1.7s recorder init 时间，这种精确匹配直接指向根因。

4. **lazy init 在高频回调中是危险模式**。虽然 lazy init 可以延迟开销，但如果初始化本身很重（如创建数据集目录），必须确保不在关键路径上执行。

---

<a id="bug-017"></a>
## BUG-017: gripper_deadband 配置不生效——dagger_config vs teleop_config 的两条路径

**严重程度**: 中
**发现日期**: 2026-02-12
**状态**: 已修复

### 现象

dagger_node 在 POLICY 模式下夹爪持续振荡（反复开合），driver 日志显示 223 帧中发送了 151 次夹爪指令（67.7%）。而 async_inference_client 使用同一 PolicyServer + 同一模型时夹爪行为正常。

用户已在 `dagger/config/dagger_params.yaml` 中配置了 `gripper_deadband: 20`，但夹爪振荡仍然存在。

### 根因

**`dagger_params.yaml` 的 `gripper_deadband: 20` 只被 `async_inference_client` 读取，dagger_node 走的是完全不同的控制路径，该配置根本不生效。**

两条夹爪控制路径：

| 路径 | 配置来源 | deadband 值 | 过滤位置 |
|------|---------|------------|---------|
| async_inference_client | `dagger_params.yaml` → `GripperAsyncController(deadband=20)` | **20** | client 内部，JSON socket 直连夹爪 |
| dagger_node → driver | `teleop_params.yaml` → `realman_driver_node` ROS2 参数 | **5** | driver `_on_action()` 回调 |

dagger_node 通过 `_publish_joint_action()` 发布 ROS2 JointState 消息（包含 joints + gripper），由 `realman_driver_node` 的 `_on_action()` 回调接收并处理。driver 使用自己的 `gripper_deadband` ROS2 参数（来自 `teleop_params.yaml`），**不读取 `dagger_params.yaml`**。

`teleop_params.yaml` 中的配置：

```yaml
realman_driver_node:
  ros__parameters:
    gripper_deadband: 5   # ← 实际生效的值！
```

`realman_driver_node.py` 中的默认值：

```python
self.declare_parameter("gripper_deadband", 10)  # ← yaml 覆盖后实际为 5
```

**deadband=5 意味着 gripper SDK 值变化 ≥5（即归一化值 ≥0.005）就发送指令**，几乎没有过滤效果。模型输出的 gripper 微小波动（±0.01）每帧都会触发夹爪动作。

### 影响链

```
模型输出 gripper 微小波动（±0.01）
  → dagger_node 每帧发布 JointState（含 gripper）
  → driver _on_action() 接收
  → deadband=5，变化 ≥0.005 就放行
  → 67.7% 的帧都发送夹爪指令
  → 夹爪持续振荡
```

### 修复（2 处）

**1. `teleop_params.yaml` — 将 driver 的 gripper_deadband 改为 20**

```yaml
realman_driver_node:
  ros__parameters:
    # 与 async_client 的 GripperAsyncController(deadband=20) 保持一致
    gripper_deadband: 20
```

**2. `realman_driver_node.py` — 默认值也改为 20（兜底）**

```python
# deadband=20 与 async_client 的 GripperAsyncController 一致，避免夹爪振荡
self.declare_parameter("gripper_deadband", 20)
```

### 为什么 async_inference_client 不受影响？

async_inference_client 直接通过 `GripperAsyncController`（JSON socket）控制夹爪，不经过 ROS2 driver。`GripperAsyncController` 的 deadband 来自 `dagger_params.yaml` 的 `gripper_deadband: 20`，过滤效果正常。

```
async_client 路径:
  model output → gripper_normalized_to_sdk() → GripperAsyncController.send_async(deadband=20) → JSON socket

dagger_node 路径:
  model output → _publish_joint_action() → ROS2 topic → driver _on_action(deadband=5) → gripper queue → JSON socket
```

### 诊断方法

1. **driver 日志**：检查夹爪发送频率。如果 `累计发送` 数接近总帧数的 50% 以上，说明 deadband 过小。

```
[夹爪] 位置=996, 结果=True, 累计发送=1
[夹爪] 位置=426, 结果=True, 累计发送=51    ← 8s 内发了 50 次
[夹爪] 位置=434, 结果=True, 累计发送=101   ← 2.6s 内又发了 50 次
```

2. **确认实际 deadband 值**：

```bash
ros2 param get /realman_driver_node gripper_deadband
# 应该返回 20，如果返回 5 或 10 说明配置未生效
```

### 经验教训

1. **dagger_node 和 async_inference_client 的夹爪控制路径完全不同**。async_client 直连夹爪（JSON socket），dagger_node 通过 ROS2 driver 中转。两条路径的 deadband 配置来源不同，必须分别设置。

2. **在 `dagger_params.yaml` 中设置的参数不会自动传递给 ROS2 driver**。driver 的参数来自 `teleop_params.yaml`，通过 launch 文件的 `parameters=[teleop_config]` 加载。

---

<a id="bug-018"></a>
## BUG-018: ⚠️ 错误修复——用 TCP rm_get_gripper_state() 替代 _last_gripper_cmd（已撤回）

**严重程度**: 高
**发现日期**: 2026-02-12
**状态**: ⚠️ 已撤回（修复方向错误，见 BUG-019）

### 背景

BUG-017 修复 deadband 后，夹爪振荡频率降低但仍存在。分析认为 `_last_gripper_cmd`（命令值）作为夹爪观测会导致瞬时跳变，于是改用 TCP `rm_get_gripper_state()` 读取 `actpos` 作为夹爪观测。

### 错误的修复

```python
# BUG-018 修复（错误）: 在 _on_timer 中用 TCP 读取夹爪
ret_code, grip_info = self.driver.arm.rm_get_gripper_state()
if ret_code == 0:
    gripper_open = float(grip_info['actpos']) / 1000.0  # ← 读的是 Modbus 夹爪！
```

### 为什么是错误的

`rm_get_gripper_state()` 是 SDK 的标准 Modbus 夹爪读取接口，读的是**标准 Modbus 夹爪子系统**。而 dagger_node 通过 RM+ JSON 协议（`set_gripper_position`）控制的是 **RM+ 夹爪子系统**。两个是不同的子系统：

| 子系统 | 控制方式 | 读取方式 | 延迟 |
|--------|---------|---------|------|
| 标准 Modbus 夹爪 | `rm_set_gripper_position()` (SDK) | `rm_get_gripper_state()` (SDK) | ~1000ms |
| RM+ 夹爪 | `set_gripper_position` (JSON TCP) | UDP `rm_plus_state.pos[0]` | ~5ms |

**控制用 RM+ JSON，读取用 Modbus SDK = 读错子系统**。Modbus 夹爪始终返回 `actpos=1000`（全开），不反映 RM+ 夹爪的实际位置。

### 日志证据（18:36 session）

```
#100:  gripper_open=1.0000, source=TCP_actpos, gripper_open_udp=0.0000  (UDP 还没收到 RM+ 数据)
#1000: gripper_open=1.0000, source=TCP_actpos, gripper_open_udp=0.6060  (UDP 正确！TCP 错误！)
```

- `TCP_actpos=1.0000`：Modbus 夹爪始终全开（因为没人通过 Modbus 控制它）
- `gripper_open_udp=0.6060`：RM+ 夹爪已关到 60%（正确反映物理状态）

### 为什么 async_inference_client 没有这个问题

async_client 的 `_update_state()` 也调用 `rm_get_gripper_state()`，但它的 UDP 接收器能正确读到 `gripper_raw_udp`（非 None），所以走的是 UDP 路径：

```python
# async_client: UDP gripper 有值时直接用 UDP
if gripper_raw is not None:
    self.current_gripper = float(gripper_raw) / 1000.0  # ← 走这里
else:
    ret_code, grip = self.robot.arm.rm_get_gripper_state()  # ← 不走这里
```

async_client 日志证实：`#200: gripper=0.4330, gripper_raw_udp=433, source=UDP`

### 经验教训

1. **控制和读取必须使用同一个子系统**。用 RM+ JSON 控制夹爪，就必须用 RM+ 数据（UDP `rm_plus_state.pos[0]`）读取。
2. **不要假设 SDK API 能读到所有夹爪状态**。`rm_get_gripper_state()` 只读 Modbus 夹爪，对 RM+ 夹爪无效。

---

<a id="bug-019"></a>
## BUG-019: rm_get_gripper_state() 读的是 Modbus 夹爪，不反映 RM+ 夹爪位置

**严重程度**: 高
**发现日期**: 2026-02-12
**状态**: 修复中（BUG-018v2: 改用 UDP RM+ gripper 数据）

### 现象

dagger_node 在 POLICY 模式下，夹爪观测始终为 1.0000（全开），即使 policy 发送关闭命令且夹爪物理上已经关闭。导致 policy 持续输出关闭命令，夹爪行为异常。

### 根因

见 BUG-018 详细分析。核心问题：**Realman 机械臂有两套独立的夹爪子系统**：

1. **标准 Modbus 夹爪**：通过 SDK `rm_set_gripper_position()` / `rm_get_gripper_state()` 操作
2. **RM+ 生态夹爪**：通过 JSON TCP `set_gripper_position` 控制，通过 UDP `rm_plus_state.pos[0]` 读取

driver_node 用 RM+ JSON 控制夹爪，但 BUG-018 修复错误地用 Modbus SDK 读取状态。

### 修复（BUG-018v2）

在 `realman_driver_node.py` 的 `_on_timer()` 中，直接使用 UDP RM+ gripper 数据：

```python
# BUG-018v2: 直接用 UDP RM+ gripper 数据
gripper_source = "UDP_rmplus"
gripper_open = gripper_open_udp
# 回退: UDP gripper 为 0 且有命令记录时，用 last_cmd
if gripper_act == 0 and self._last_gripper_cmd is not None:
    gripper_open = gripper_open_from_pos(self._last_gripper_cmd)
    gripper_source = "last_cmd_fallback"
```

移除了整个 `rm_get_gripper_state()` TCP 调用块。

### 验证方法

运行 dagger launch，启动 POLICY 模式，观察 driver 日志：

```
[DIAG-GRIP-v2] #200: PUBLISH=0.6060, source=UDP_rmplus, udp_raw=606, ...
```

- `PUBLISH` 值应随夹爪物理位置变化（不再固定 1.0000）
- `source` 应为 `UDP_rmplus`（不再是 `TCP_actpos`）
- `tcp_actpos` 仅作诊断对比，每 200 帧读一次，不影响发布值

### 已知限制

- UDP RM+ gripper 数据在启动初期可能为 0（RM+ 协议尚未就绪），此时回退到 `last_cmd`
- UDP RM+ gripper 可能有轻微延迟（比 TCP Modbus 的"始终 1000"好得多）


3. **ROS2 `declare_parameter()` 的默认值会被 yaml 配置覆盖**。即使代码中默认值是 10，如果 yaml 中写了 5，实际运行时就是 5。排查时不能只看代码默认值，必须同时检查 yaml 配置。

4. **夹爪振荡的快速诊断**：driver 日志中夹爪发送频率 >50% 帧率 → deadband 过小。正常情况下（模型输出稳定时），夹爪指令应该很少发送（<10% 帧率）。

---

<a id="bug-020"></a>
## BUG-020: camera_node 和 dagger_node 双重旋转导致图像方向错误

**严重程度**: 致命
**发现日期**: 2026-02-12
**状态**: 已修复

### 现象

dagger_node（UI 控制路径）在 POLICY 模式下：
- 夹取方向偏差（gripper 方向与预期不符）
- 无法成功抓取物体
- 同一 PolicyServer + 同一模型，async_inference_client 工作正常

关键线索：用户在 UI 中看到的相机图像是正常的（正向），但 policy 行为异常。

### 根因

**cam0 图像被双重旋转 180° + 180° = 360°，导致 policy 看到的是倒置的原始图像（与训练数据不一致）。**

**物理事实**：腕部相机（cam0）安装时是倒着的（上下颠倒），需要旋转 180° 才能变成正向。

**问题链**：
```
RealSense 原始图像（倒置）
  → camera_node 旋转 180°（变正向）
  → dagger_node._on_image() 又旋转 180°（又变倒置）
  → policy 看到倒置图像
  → 与训练数据（正向）不一致
  → 抓取方向错误
```

#### 图像处理的三条路径

1. **训练数据录制**（dataset_recorder_node）：
   ```
   RealSense 原始图像（倒置）
     → camera_node 旋转 180°（正向）
     → ROS2 topic → recorder 保存
   → 训练数据中 cam0 是正向的 ✅
   ```

2. **async_inference_client**（工作正常）：
   ```
   RealSense 原始图像（倒置）
     → 直接读取 → rotate_image(180°)（正向）
     → policy
   → policy 看到正向图像（与训练数据一致）✅
   ```

3. **dagger_node**（修复前，双重旋转）：
   ```
   RealSense 原始图像（倒置）
     → camera_node 旋转 180°（正向）
     → ROS2 topic → UI 显示（正向）✅

   同时：
   ROS2 topic（正向）
     → dagger_node._on_image() 再旋转 180°（倒置）❌
     → policy
   → policy 看到倒置图像（与训练数据不一致）❌
   ```

#### 配置证据

**teleop_params.yaml** (line 76):
```yaml
camera_node_cam0:
  ros__parameters:
    rotate_deg: 180  # ← camera_node 已经旋转了 180°
```

**dagger_node.py** `_on_image()` (修复前，line 532-533):
```python
rot_deg = self._camera_rotations.get(cam_name, 0)
if rot_deg == 180:
    img = np.rot90(img, k=2).copy()  # ← 又旋转了 180°
```

**dagger_params.yaml** (line 72-73):
```yaml
camera_rotations:
  cam0_rgb: 180  # ← 配置了旋转，但不应该在 dagger_node 中再次旋转
```

### 为什么 UI 上看起来正常？

**UI 显示的图像和 policy 收到的图像是不同的路径**：

- **UI 图像流**：直接订阅 camera_node 的 ROS2 topic（已旋转 180°，正向）✅
- **policy 输入**：经过 dagger_node `_on_image()` 处理（又旋转 180°，变回倒置）❌

所以用户在 UI 上看到的图像是正向的（正确），但 policy 实际收到的是双重旋转后的倒置图像（与训练数据不一致）。

### 修复

**移除 dagger_node `_on_image()` 中的旋转逻辑**，因为 camera_node 已经处理了旋转：

```python
def _on_image(self, msg: Image, cam_name: str):
    """
    Receive camera image, cache as RGB numpy array.

    IMPORTANT: 不在此处旋转图像！camera_node 已经根据 teleop_params.yaml 的 rotate_deg
    参数旋转过了（cam0: 180°）。如果这里再旋转，会导致双重旋转（360°），图像方向错误。

    与 async_inference_client 的差异：
    - async_client: 直接从 RealSense 读取原始图像 → 需要旋转 180°
    - dagger_node: 从 camera_node ROS2 话题接收 → camera_node 已旋转，不需要再旋转
    """
    try:
        img = self._cv_bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        # 不旋转！camera_node 已经旋转过了
        with self._image_lock:
            self._cached_images[cam_name] = img
    except Exception as e:
        self.get_logger().warn(
            f"Failed to convert image from {cam_name}: {e}",
            throttle_duration_sec=5.0,
        )
```

### 架构差异总结

| 维度 | async_inference_client | dagger_node（修复前） | dagger_node（修复后） |
|------|----------------------|---------------------|---------------------|
| 图像来源 | 直接读 RealSense SDK | 订阅 camera_node ROS2 topic | 订阅 camera_node ROS2 topic |
| camera_node 旋转 | 不经过 camera_node | 经过（倒→正，180°） | 经过（倒→正，180°） |
| 客户端旋转 | 需要（倒→正，180°） | 又旋转（正→倒，180°）❌ | 不旋转 ✅ |
| policy 看到的图像 | 正向 ✅ | 倒置（与训练数据不一致）❌ | 正向 ✅ |

### 诊断方法

1. **检查 camera_node 配置**：
   ```bash
   grep "rotate_deg" vr_teleop/.../config/teleop_params.yaml
   # 如果 cam0 配置了 rotate_deg: 180，说明 camera_node 已经旋转
   ```

2. **检查 dagger_node 是否再次旋转**：
   ```bash
   grep -A 5 "_on_image" dagger/dagger_node.py | grep "rot90\|rotate"
   # 如果有 np.rot90 或 cv2.rotate 调用，说明双重旋转
   ```

3. **对比 async_client 和 dagger_node 的图像处理**：
   - async_client 应该有旋转逻辑（因为直接读 RealSense）
   - dagger_node 不应该有旋转逻辑（因为 camera_node 已旋转）

### 经验教训

1. **图像旋转必须只做一次**。如果 camera_node 已经旋转，下游节点不应该再旋转。

2. **UI 显示正常 ≠ policy 输入正确**。UI 订阅的是 camera_node 的原始 topic，而 policy 收到的是经过客户端处理后的图像，两者可能不同。

3. **不同的图像获取路径需要不同的处理逻辑**：
   - 直接读硬件（async_client）→ 需要自己旋转
   - 订阅 ROS2 topic（dagger_node）→ 不需要旋转（camera_node 已处理）

4. **双重旋转 360° 的隐蔽性**。原始图像是倒置的，旋转 180° 变正向（训练数据），再旋转 180° 又变回倒置。倒置的图像不像旋转 90° 那样明显错误（左右仍然正确，只是上下颠倒），但对于训练在正向图像上的 policy 来说，这是完全错误的输入，会导致抓取方向错误。

5. **排查图像相关问题时，必须追踪完整的图像处理链路**：
   - 硬件 → camera_node → ROS2 topic → 客户端回调 → policy
   - 每一步的旋转/变换都要确认，确保最终 policy 看到的图像与训练数据一致

6. **配置文件中的 camera_rotations 参数含义取决于上下文**：
   - async_client: 表示"需要旋转多少度"（因为读的是原始图像）
   - dagger_node: 应该表示"camera_node 已经旋转了多少度"（仅作记录，不应再次旋转）

### 相关文件

- `vr_teleop/.../config/teleop_params.yaml` — camera_node 旋转配置
- `dagger/config/dagger_params.yaml` — camera_rotations 配置（仅供参考）
- `dagger/dagger_node.py` — `_on_image()` 回调（修复：移除旋转）
- `dagger/async_inference_client.py` — 参考实现（直接读硬件，需要旋转）

---

<a id="pitfall-021"></a>
## PITFALL-021: 录制帧率节流过激导致 ~30% 丢帧

**严重程度**: 高
**发现日期**: 2026-02-23
**状态**: 已修复（系数从 0.9 改为 0.5）

### 现象

录制设置 30fps，实际推理 18 秒，预期 540 帧，但只录到 ~410 帧（76%）。REC-STATS 诊断日志显示：

```
[REC-STATS] 10.0s: triggered=295 throttled=92 queue_full=0 recorded=203 fps=20.3
```

- `triggered=295`：10 秒内触发 295 次（~30Hz，正常）
- `throttled=92`：其中 92 次被节流丢弃（31%）
- `recorded=203`：实际只录了 203 帧（fps=20.3）

### 根因

录制系统有一个"最小帧间隔"保护机制（`_on_recording_sync` 中），防止录制频率超过目标 fps：

```python
# dagger_node.py line 296
self._min_frame_interval = 0.9 / self.recording_fps  # = 0.9 / 30 = 30ms
```

每次录制一帧时，检查距上一帧的时间间隔是否 ≥ 30ms，不足则丢弃。

问题在于：录制触发源是 ROS2 `ApproximateTimeSynchronizer`，消息到达时间有抖动。理想间隔 33.3ms（30Hz），但实际到达时间波动 ±5ms：

```
帧1: t=0ms
帧2: t=28ms  ← 距帧1只有28ms < 30ms阈值 → 被节流丢弃！
帧3: t=62ms  ← 距帧1有62ms > 30ms → 录制
帧4: t=95ms  ← 距帧3有33ms > 30ms → 录制
帧5: t=123ms ← 距帧4有28ms < 30ms → 被节流丢弃！
```

30ms 阈值对于 33.3ms 的理想间隔来说太紧——消息只要早到 3.3ms（很常见的抖动），就会被丢弃。

### 修复

将系数从 `0.9` 改为 `0.5`：

```python
# 修复后：0.5 / 30 = 16.7ms，只过滤明显的重复帧（>60Hz），不误杀正常帧
self._min_frame_interval = 0.5 / self.recording_fps if self.recording_fps > 0 else 0.0
```

`0.5 / 30 = 16.7ms` 阈值只会过滤真正的重复帧（同一帧触发两次，间隔 <16.7ms），不会误杀正常的 30Hz 帧。

### 系数选择参考

| 系数 | 阈值 (30fps) | 效果 |
|------|-------------|------|
| 0.9 | 30.0ms | 过激：31% 丢帧 |
| 0.7 | 23.3ms | 保守折中 |
| 0.5 | 16.7ms | 推荐：只过滤 >60Hz 的重复帧 |
| 0.0 | 0ms | 完全关闭节流 |

### 诊断方法

查看 REC-STATS 日志中 `throttled / triggered` 的比例：
- `< 5%`：正常
- `> 20%`：节流过激，需要降低系数

### 经验教训

1. **ROS2 消息到达时间有抖动**。不能假设消息以精确的固定间隔到达，节流阈值必须留足余量。
2. **节流的目的是防止超频，不是精确控频**。ROS2 sync 本身就是 ~30Hz，节流只需要防止偶尔的消息重复（同一帧触发两次），不需要精确限制到 30fps。
3. **REC-STATS 诊断日志是排查录制丢帧的关键工具**。`triggered/throttled/queue_full/recorded` 四个计数器可以快速定位丢帧发生在哪个环节。