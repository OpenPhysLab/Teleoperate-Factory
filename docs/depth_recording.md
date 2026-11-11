# 高位 D435 深度记录

高位相机为 D435（SDK 序列号 233622075512）。cameras.yaml 中
record_depth: true 默认启用；两台 D405 保持只采彩色。
数采入口 rm75_teleop_dataset.py 和独立相机入口 synchronized_cameras.py
均支持 --no-depth 临时关闭深度。改用 V4L2 时必须关闭深度，不会静默漏录。

每个采集帧在当前 episode 的 depth/ 子目录下保存
00000000_d435.png 等 16 位单通道无损 PNG。SDK 在同一 frameset 内将
深度对齐到彩色像素网格；分辨率与彩色一致，当前为 640×480。
未执行伪彩色转换、毫米取整或有损视频编码。零值表示无效深度。

manifest.jsonl 的 depth.d435 保存 path、scale_meters_per_unit、
深度源 timestamp_ns / timestamp_domain / sequence、source_serial、
彩色像素网格 intrinsics、native_depth_intrinsics，以及
depth_to_color_extrinsics（旋转按 SDK 的列优先顺序，平移单位米）。
距离（米）= PNG 像素原始 uint16 值 × scale_meters_per_unit。
深度时间戳仅在其 timestamp_domain 对应的时钟域内解释，原有
timestamps_ns / host_timestamps_ns / timestamp_domains 仍描述彩色帧。

深度与被选中的同步彩色帧一起入队和保存，只在当前条采集激活、
通过同步与频率筛选后落盘。--no-save-images 仅关闭 RGB JPEG，
不关闭深度；--no-cameras 则不采集任何相机。缺少深度帧的 D435
frameset 不进入队列；启动深度失败会报错，不冒充成功。
启动和停止机器人仍沿用现有流程，单独验证相机不会驱动机器人。
