#!/usr/bin/env python3
"""
诊断脚本：精确测量 dagger_node 的 action 发布频率。

用法 1 — 订阅 ROS2 话题测量（需要 dagger launch 运行中）：
    python dagger/tests/test_exec_frequency_diag.py --ros

用法 2 — 直接测量 _policy_exec_loop 的 timing（无需 ROS2）：
    python dagger/tests/test_exec_frequency_diag.py --standalone

用法 3 — 对比 async_inference_client 的 timing：
    python dagger/tests/test_exec_frequency_diag.py --async-client
"""
import argparse
import time
import threading
import numpy as np
from unittest.mock import MagicMock
from collections import deque


def test_standalone_exec_loop_timing(f_exec=10.0, duration=5.0):
    """
    模拟 _policy_exec_loop 的 timing，不需要 ROS2。
    精确测量 while+sleep 循环的实际频率。
    """
    print(f"\n{'='*60}")
    print(f"Standalone exec loop timing test")
    print(f"  f_exec={f_exec} Hz, T_step={1.0/f_exec:.4f}s, duration={duration}s")
    print(f"{'='*60}")

    T_step = 1.0 / f_exec
    timestamps = []
    stop = threading.Event()

    def exec_loop():
        while not stop.is_set():
            t_start = time.monotonic()
            timestamps.append(t_start)

            # Simulate action execution (publish) — ~0.1ms
            time.sleep(0.0001)

            elapsed = time.monotonic() - t_start
            sleep_time = T_step - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    t = threading.Thread(target=exec_loop, daemon=True)
    t.start()
    time.sleep(duration)
    stop.set()
    t.join(timeout=2.0)

    if len(timestamps) < 2:
        print("ERROR: Not enough samples")
        return

    intervals = np.diff(timestamps)
    actual_hz = 1.0 / np.mean(intervals)
    print(f"\nResults:")
    print(f"  Samples: {len(timestamps)}")
    print(f"  Expected Hz: {f_exec}")
    print(f"  Actual Hz:   {actual_hz:.2f}")
    print(f"  Mean interval: {np.mean(intervals)*1000:.2f} ms")
    print(f"  Std interval:  {np.std(intervals)*1000:.2f} ms")
    print(f"  Min interval:  {np.min(intervals)*1000:.2f} ms")
    print(f"  Max interval:  {np.max(intervals)*1000:.2f} ms")
    print(f"  Drift: {abs(actual_hz - f_exec) / f_exec * 100:.2f}%")


def test_standalone_with_observation_load(f_exec=10.0, obs_time_ms=5.0, duration=5.0):
    """
    模拟 _policy_exec_loop + _update_inference_observation 的 timing。
    obs_time_ms: 模拟 observation 更新耗时（毫秒）
    """
    print(f"\n{'='*60}")
    print(f"Exec loop + observation load timing test")
    print(f"  f_exec={f_exec} Hz, obs_time={obs_time_ms}ms, duration={duration}s")
    print(f"{'='*60}")

    T_step = 1.0 / f_exec
    timestamps = []
    stop = threading.Event()

    def exec_loop():
        while not stop.is_set():
            t_start = time.monotonic()
            timestamps.append(t_start)

            # Simulate _update_inference_observation
            time.sleep(obs_time_ms / 1000.0)

            # Simulate action execution
            time.sleep(0.0001)

            elapsed = time.monotonic() - t_start
            sleep_time = T_step - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    t = threading.Thread(target=exec_loop, daemon=True)
    t.start()
    time.sleep(duration)
    stop.set()
    t.join(timeout=2.0)

    if len(timestamps) < 2:
        print("ERROR: Not enough samples")
        return

    intervals = np.diff(timestamps)
    actual_hz = 1.0 / np.mean(intervals)
    print(f"\nResults:")
    print(f"  Samples: {len(timestamps)}")
    print(f"  Expected Hz: {f_exec}")
    print(f"  Actual Hz:   {actual_hz:.2f}")
    print(f"  Mean interval: {np.mean(intervals)*1000:.2f} ms")
    print(f"  Std interval:  {np.std(intervals)*1000:.2f} ms")


def test_dagger_node_actual_publish_rate():
    """
    模拟完整的 dagger_node POLICY 模式：
    - timer callback at control_hz (feeds observations)
    - _policy_exec_loop thread at f_exec (executes actions)

    测量 action publish 的实际频率。
    """
    print(f"\n{'='*60}")
    print(f"Full dagger_node simulation (timer + exec loop)")
    print(f"{'='*60}")

    control_hz = 10.0  # from config vr_control.control_hz
    f_exec = 10.0      # from config f_exec
    T_step = 1.0 / f_exec
    timer_period = 1.0 / control_hz
    duration = 5.0

    publish_timestamps = []
    obs_timestamps = []
    stop = threading.Event()

    # Simulate _execution_tick timer (observation feeding only in POLICY mode)
    def timer_loop():
        while not stop.is_set():
            t_start = time.monotonic()
            obs_timestamps.append(t_start)
            # Simulate _update_inference_observation (~2ms)
            time.sleep(0.002)
            elapsed = time.monotonic() - t_start
            sleep_time = timer_period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    # Simulate _policy_exec_loop (action execution)
    def exec_loop():
        while not stop.is_set():
            t_start = time.monotonic()

            # _update_inference_observation (also called here)
            time.sleep(0.002)

            # get_next_action + execute
            time.sleep(0.0001)
            publish_timestamps.append(time.monotonic())

            elapsed = time.monotonic() - t_start
            sleep_time = T_step - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    timer_thread = threading.Thread(target=timer_loop, daemon=True)
    exec_thread = threading.Thread(target=exec_loop, daemon=True)
    timer_thread.start()
    exec_thread.start()
    time.sleep(duration)
    stop.set()
    timer_thread.join(timeout=2.0)
    exec_thread.join(timeout=2.0)

    if len(publish_timestamps) < 2:
        print("ERROR: Not enough publish samples")
        return

    pub_intervals = np.diff(publish_timestamps)
    obs_intervals = np.diff(obs_timestamps)

    print(f"\nAction publish rate (from exec loop):")
    print(f"  Samples: {len(publish_timestamps)}")
    print(f"  Expected Hz: {f_exec}")
    print(f"  Actual Hz:   {1.0/np.mean(pub_intervals):.2f}")
    print(f"  Mean interval: {np.mean(pub_intervals)*1000:.2f} ms")

    print(f"\nObservation feed rate (from timer):")
    print(f"  Samples: {len(obs_timestamps)}")
    print(f"  Expected Hz: {control_hz}")
    print(f"  Actual Hz:   {1.0/np.mean(obs_intervals):.2f}")
    print(f"  Mean interval: {np.mean(obs_intervals)*1000:.2f} ms")


def test_ros_topic_rate():
    """
    订阅 /rm/action_pose 话题，测量实际发布频率。
    需要 dagger launch 运行中。
    """
    try:
        import rclpy
        from rclpy.node import Node as RosNode
        from sensor_msgs.msg import JointState
    except ImportError:
        print("ERROR: rclpy not available. Run with system Python or source ROS2.")
        return

    print(f"\n{'='*60}")
    print(f"ROS2 topic rate measurement: /rm/action_pose")
    print(f"Listening for 10 seconds...")
    print(f"{'='*60}")

    timestamps = deque(maxlen=1000)

    rclpy.init()

    class RateMonitor(RosNode):
        def __init__(self):
            super().__init__("rate_monitor")
            self.sub = self.create_subscription(
                JointState, "/rm/action_pose", self.cb, 10
            )
            self.timer = self.create_timer(1.0, self.report)
            self.count = 0

        def cb(self, msg):
            timestamps.append(time.monotonic())
            self.count += 1

        def report(self):
            if len(timestamps) < 2:
                print(f"  Received {self.count} messages so far...")
                return
            recent = list(timestamps)
            intervals = np.diff(recent[-100:])  # last 100
            hz = 1.0 / np.mean(intervals) if len(intervals) > 0 else 0
            print(f"  Messages: {self.count}, Rate: {hz:.1f} Hz, "
                  f"Interval: {np.mean(intervals)*1000:.1f} ms")

    node = RateMonitor()
    try:
        end_time = time.time() + 10.0
        while time.time() < end_time:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

    if len(timestamps) >= 2:
        all_intervals = np.diff(list(timestamps))
        print(f"\nFinal results:")
        print(f"  Total messages: {len(timestamps)}")
        print(f"  Overall Hz: {1.0/np.mean(all_intervals):.2f}")
        print(f"  Mean interval: {np.mean(all_intervals)*1000:.2f} ms")
        print(f"  Std interval:  {np.std(all_intervals)*1000:.2f} ms")
    else:
        print("\nNo messages received on /rm/action_pose")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DAgger exec frequency diagnostic")
    parser.add_argument("--ros", action="store_true", help="Subscribe to ROS2 topic")
    parser.add_argument("--standalone", action="store_true", help="Standalone timing test")
    parser.add_argument("--async-client", action="store_true", help="Compare async_client timing")
    parser.add_argument("--full-sim", action="store_true", help="Full dagger_node simulation")
    parser.add_argument("--f-exec", type=float, default=10.0, help="Execution frequency")
    parser.add_argument("--duration", type=float, default=5.0, help="Test duration (seconds)")
    args = parser.parse_args()

    if args.ros:
        test_ros_topic_rate()
    elif args.full_sim:
        test_dagger_node_actual_publish_rate()
    elif args.standalone:
        test_standalone_exec_loop_timing(f_exec=args.f_exec, duration=args.duration)
        test_standalone_with_observation_load(f_exec=args.f_exec, obs_time_ms=5.0, duration=args.duration)
    else:
        # Default: run all non-ROS tests
        test_standalone_exec_loop_timing(f_exec=args.f_exec, duration=args.duration)
        test_standalone_with_observation_load(f_exec=args.f_exec, obs_time_ms=5.0, duration=args.duration)
        test_dagger_node_actual_publish_rate()
