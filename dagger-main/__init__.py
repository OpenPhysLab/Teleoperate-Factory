"""
DAgger 系统 — 异步推理 + VR 接管 + 数据录制

Phase 1: 异步推理框架（ring buffer + 固定频率执行 + gRPC 推理线程）
Phase 2: 推理时数据录制（LeRobot v2.1 格式）
Phase 3: VR 人类接管（ROS2 + VR trigger 控制权切换）
"""
