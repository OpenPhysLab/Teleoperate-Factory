# 18 号夹爪与录像

入口：src/oculus_reader/scripts/rm18_gripper_video.py

- A：开始录像；正在录像时重复按 A 不会停止或新建片段。
- X：结束当前录像并保存；未录像时按 X 无操作。保存后再按 A 新建下一段。
- 同时按 A+X 时优先结束，不会开始新片段。
- B：切换 192.168.1.18 的夹爪开合，未录像时也始终响应。
- Ctrl+C、预览窗口 Q 或 Esc：结束程序，保存正在录制的片段。
- 不要求按住按键；每次按下后松开，长按只触发一次。
- 启动时已按住的键会被忽略，松开后重新按才触发。
- 启动只读取夹爪当前状态，退出保持最后夹爪状态。程序没有关节控制和回位调用。
- 初始夹爪模式/位置用于判断首次 B 应打开还是闭合；SDK 拒绝的指令不会切换逻辑状态，下次 B 可重连重试。

默认保存到 /media/lyj/Data/GaussianRDT_Videos/gripper18_时间戳。
每个 episode 包含三路 mp4v 编码的 MP4 彩色视频、高位 D435 的 depth/*.png
16 位彩色对齐深度、manifest.jsonl、summary.json。
深度记录比例/内外参/时钟域；距离（米）= PNG 像素值 × scale_meters_per_unit。
两台 D405 保持彩色视频。每个 MP4 的帧索引由 manifest 的 videos 对应，
视频按配置 fps 播放，真实采集时间以 manifest 时间戳为准。
会话 events.jsonl 记录开始/停止和夹爪指令，未录制期间的操作也记录。
夹爪字段 commanded/accepted 代表指令状态，实际到位情况看 gripper_feedback。

从工作站项目目录启动（反斜杠表示该条命令续行）：

```bash
source scripts/activate_rm75_api2.sh
python3 src/oculus_reader/scripts/rm18_gripper_video.py \
  --quest-ip 172.16.205.198
```

--check 只连接并读取夹爪状态后退出，不打开相机，也不发开合指令。
--dry-run 模拟夹爪，但照常读取 Quest/相机和录制；--no-display 关闭预览。
--no-depth 临时关闭深度；--duration N 运行 N 秒后保存退出，默认持续运行。
相机不可与旧数采或监控服务同时独占打开，切换前需先正常结束旧采集。
