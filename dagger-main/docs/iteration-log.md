# DAgger 项目迭代记录

---

## Iteration #1 — VR 介入模式诊断与夹爪跳变修复

**日期**: 2026-02-26
**分析日志**: `dagger/logs/2026-02-26_11-44-52`（78600 行 dagger_node.log + 663 行 inference_bridge.log）

---

### 问题描述

用户在 DAgger 训练中使用 VR 手柄介入（POLICY→HUMAN 模式切换）时，观察到以下异常：

1. **第一个 episode**：按下介入键（trigger）后，机械臂不响应 VR 手柄的移动指令，但夹爪会突然开/关
2. **第二个 episode 开始**：第一次介入时出现卡顿，随后机械臂突然移动（动作过冲）
3. **同一 episode 后续介入**：机械臂再次不响应移动
4. **夹爪异常**：每次按介入键时夹爪状态突然跳变

---

### 分析过程

#### 1. 日志中的关键发现

- 日志中有 **15 次 POLICY→HUMAN 转换**（trigger press 正常触发）
- **0 次 HUMAN→POLICY 转换日志**（`_transition_to_policy()` 没有 diag log，无法确认是否执行）
- HUMAN 模式持续时间约 1.4-3 秒，期间 **没有任何 `[PUB-ACTION]` 日志**

#### 2. 为什么 HUMAN 模式下没有 action 日志

`_tick_human()` 发布 pose action 通过 `_publish_pose_action()` → `_publish_pose_action_with_gripper()`，但该方法 **没有 `_diag_log` 输出**（与 `_publish_joint_action` 不同）。因此无法从日志判断 HUMAN 模式是否在正常发布 action。

这是一个诊断盲区——HUMAN 模式的整个 action 发布链路没有任何可观测的日志。

#### 3. `_tick_human()` 的 4 个 return 路径

```python
def _tick_human(self):
    with self._vr_lock:
        if time.time() - self._vr_last_update > 0.5:  # 检查1: VR数据超时
            return
        if not self.vr_state.is_active:                # 检查2: trigger未按下
            self._sync_arm_state()
            return
        if not self.vr_state.is_initialized:           # 检查3: 未初始化
            return
        target_pose = self._calculate_target_pose()
        if target_pose is None:                        # 检查4: 计算失败（实际不会发生）
            return
    self._publish_pose_action(target_pose)
```

由于没有诊断日志，无法确认是哪个检查点导致 HUMAN 模式下 action 未发布。

#### 4. 夹爪跳变的根因

分析日志中 POLICY→HUMAN 转换前后的 gripper 值：

```
转换前（POLICY 模式）: grip_act=0.4598（policy 正在关闭夹爪，gripper ≈ 0.46）
转换后（HUMAN 模式）:  _current_grip=1.0（默认 open）
```

**根因**：`_current_grip` 初始值为 1.0（open），`_gripper_closed` 初始值为 False。当从 POLICY 切换到 HUMAN 时，这两个值没有与当前机械臂的实际夹爪状态同步。如果 policy 正在执行夹爪关闭动作（gripper ≈ 0.4），切换到 HUMAN 后立即发送 gripper=1.0（全开），导致夹爪突然松开。

反之，如果 policy 保持夹爪全开（gripper=1.0），而之前某次 grip toggle 将 `_gripper_closed` 设为 True，切换后会发送 gripper=0.0（全关），导致夹爪突然关闭。

---

### 修改内容

#### 修改 1：`_tick_human()` 添加诊断日志

**文件**: `dagger/dagger_node.py`
**位置**: `_tick_human()` 方法

在每个 return 路径添加 `_diag_log`，输出具体的跳过原因：

- `[TICK-HUMAN] #N: SKIP vr_timeout (Xs)` — VR 数据超时
- `[TICK-HUMAN] #N: SKIP is_active=False` — trigger 未按下
- `[TICK-HUMAN] #N: SKIP is_initialized=False` — 未初始化
- `[TICK-HUMAN] #N: SKIP target_pose=None` — 位姿计算失败
- `[TICK-HUMAN] #N: PUB pose grip=X.XX pos=[x,y,z]` — 成功发布

#### 修改 2：`_publish_pose_action_with_gripper()` 添加诊断日志

**文件**: `dagger/dagger_node.py`
**位置**: `_publish_pose_action_with_gripper()` 方法

添加 `[PUB-POSE] #N: gripper=X.XXXX, pose_dim=N` 日志，与 `[PUB-ACTION]`（joint action）对应，消除 HUMAN 模式的诊断盲区。

#### 修改 3：POLICY→HUMAN 切换时同步夹爪状态

**文件**: `dagger/dagger_node.py`
**位置**: `_transition_to_human()` 方法

在模式切换时，从 `_on_state_last_gripper`（机械臂实际夹爪反馈值）读取当前夹爪状态，初始化 `_gripper_closed` 和 `_current_grip`：

```python
current_gripper = getattr(self, '_on_state_last_gripper', None)
if current_gripper is not None:
    self._gripper_closed = current_gripper <= 0.5
    self._current_grip = 0.0 if self._gripper_closed else 1.0
    self.vr_state.current_grip = self._current_grip
```

同时添加 `[GRIP-SYNC]` 诊断日志，记录同步前后的 gripper 值变化。

---

### 待验证

1. **HUMAN 模式不响应移动**：需要在下次测试中观察 `[TICK-HUMAN]` 日志，确认是哪个检查点导致 action 未发布。可能的原因：
   - VR pose 数据超时（`_vr_last_update` 过期）
   - `is_active=False`（trigger 状态异常）
   - `is_initialized=False`（`_activate_control()` 中 arm state 获取失败）
2. **夹爪跳变**：修改 3 应该能解决。需要验证 `_on_state_last_gripper` 在转换时刻的值是否准确反映实际夹爪状态。
3. **第二个 episode 的过冲**：与 Motion Gating + buffer 清空后一次性填入整个 chunk 有关，本次未修复，待后续迭代处理。
