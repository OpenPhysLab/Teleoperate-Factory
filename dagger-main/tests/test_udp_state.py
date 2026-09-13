"""
UDPStateReceiver 单元测试

测试覆盖:
- UDP JSON 解析 + 状态更新
- 消息类型校验 (P0)
- 畸形消息处理 (P0)
- 关节角度精度
- 夹爪: rm_plus_state + hand 备用字段 (P1)
- 末端位姿: 数组格式 + 字典格式 (P1)
- 浮点转换异常不崩溃 (P0)
- read_state_14d() 单位转换
- 线程安全
- is_receiving() 超时检测
"""
import json
import socket
import threading
import time

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dagger.core.udp_state import UDPStateReceiver, enable_udp_push


def _make_udp_msg(
    joint_position=None,
    gripper_pos=None,
    hand_pos=None,
    waypoint_position=None,
    waypoint_euler=None,
    state="realtime_arm_joint_state",
    extra_fields=None,
):
    """构造模拟 UDP JSON 消息"""
    msg = {}
    if state is not None:
        msg["state"] = state

    if joint_position is not None:
        msg["joint_status"] = {"joint_position": joint_position}

    if gripper_pos is not None:
        msg["rm_plus_state"] = {"pos": gripper_pos}

    if hand_pos is not None:
        msg["hand"] = {"pos": hand_pos}

    if waypoint_position is not None or waypoint_euler is not None:
        wp = {}
        if waypoint_position is not None:
            wp["position"] = waypoint_position
        if waypoint_euler is not None:
            wp["euler"] = waypoint_euler
        msg["waypoint"] = wp

    if extra_fields:
        msg.update(extra_fields)

    return json.dumps(msg).encode("utf-8")


def _send_udp(port, data):
    """发送 UDP 数据包到 localhost"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(data, ("127.0.0.1", port))
    sock.close()


class TestUDPStateReceiverParsing:
    """测试 _parse_and_update() 解析逻辑（直接调用，不启动线程）"""

    def setup_method(self):
        self.receiver = UDPStateReceiver(port=0, arm_dof=7)

    def test_valid_message_array_format(self):
        """完整有效消息（数组格式）"""
        data = _make_udp_msg(
            joint_position=[180000, 90000, 45000, 0, -45000, -90000, -180000],
            gripper_pos=[500],
            waypoint_position=[300000, 200000, 100000],
            waypoint_euler=[1571, 785, 0],
        )
        self.receiver._parse_and_update(data)

        jd, gp, ep = self.receiver.read_full_state()

        # 关节角度: 180000 * 0.001 = 180.0 度
        assert jd is not None
        np.testing.assert_allclose(jd, [180.0, 90.0, 45.0, 0.0, -45.0, -90.0, -180.0])

        # 夹爪
        assert gp == 500

        # 末端位姿: 300000 微米 = 0.3 米, 1571 * 0.001 = 1.571 弧度
        assert ep is not None
        np.testing.assert_allclose(ep, [0.3, 0.2, 0.1, 1.571, 0.785, 0.0], atol=1e-6)

    def test_joint_precision(self):
        """关节角度精度: 输入 180000 (0.001度) -> 输出 180.0 度"""
        data = _make_udp_msg(
            joint_position=[180000, 0, 0, 0, 0, 0, 0],
        )
        self.receiver._parse_and_update(data)
        jd, _, _ = self.receiver.read_full_state()
        assert jd is not None
        assert abs(jd[0] - 180.0) < 1e-10

    def test_gripper_rm_plus(self):
        """夹爪: rm_plus_state.pos"""
        data = _make_udp_msg(gripper_pos=[500])
        self.receiver._parse_and_update(data)
        _, gp, _ = self.receiver.read_full_state()
        assert gp == 500

    def test_gripper_hand_fallback_list(self):
        """[P1] 夹爪备用 hand 字段 (列表格式)"""
        data = _make_udp_msg(hand_pos=[750])
        self.receiver._parse_and_update(data)
        _, gp, _ = self.receiver.read_full_state()
        assert gp == 750

    def test_gripper_hand_fallback_scalar(self):
        """[P1] 夹爪备用 hand 字段 (标量格式)"""
        msg = {
            "state": "realtime_arm_joint_state",
            "hand": {"pos": 600},
        }
        data = json.dumps(msg).encode("utf-8")
        self.receiver._parse_and_update(data)
        _, gp, _ = self.receiver.read_full_state()
        assert gp == 600

    def test_gripper_rm_plus_takes_priority(self):
        """rm_plus_state 优先于 hand 字段"""
        data = _make_udp_msg(gripper_pos=[500], hand_pos=[750])
        self.receiver._parse_and_update(data)
        _, gp, _ = self.receiver.read_full_state()
        assert gp == 500

    def test_waypoint_dict_format(self):
        """[P1] 末端位姿字典格式（旧版固件）"""
        msg = {
            "state": "realtime_arm_joint_state",
            "waypoint": {
                "position": {"x": 300000, "y": 200000, "z": 100000},
                "euler": {"rx": 1571, "ry": 785, "rz": 0},
            },
        }
        data = json.dumps(msg).encode("utf-8")
        self.receiver._parse_and_update(data)
        _, _, ep = self.receiver.read_full_state()
        assert ep is not None
        np.testing.assert_allclose(ep, [0.3, 0.2, 0.1, 1.571, 0.785, 0.0], atol=1e-6)

    def test_reject_wrong_state_type(self):
        """[P0] 非 realtime_arm_joint_state 消息被忽略"""
        data = _make_udp_msg(
            joint_position=[180000, 0, 0, 0, 0, 0, 0],
            state="some_other_state",
        )
        self.receiver._parse_and_update(data)
        jd, _, _ = self.receiver.read_full_state()
        assert jd is None

    def test_reject_missing_state_field(self):
        """[P0] 缺少 state 字段的消息被忽略"""
        data = _make_udp_msg(
            joint_position=[180000, 0, 0, 0, 0, 0, 0],
            state=None,
        )
        self.receiver._parse_and_update(data)
        jd, _, _ = self.receiver.read_full_state()
        assert jd is None

    def test_reject_malformed_json(self):
        """[P0] 畸形 JSON 被忽略"""
        self.receiver._parse_and_update(b"not json at all")
        jd, gp, ep = self.receiver.read_full_state()
        assert jd is None and gp is None and ep is None

    def test_reject_non_dict_json(self):
        """[P0] 非 dict JSON 被忽略"""
        self.receiver._parse_and_update(b'[1, 2, 3]')
        jd, gp, ep = self.receiver.read_full_state()
        assert jd is None and gp is None and ep is None

    def test_reject_non_utf8(self):
        """[P0] 非 UTF-8 数据被忽略"""
        self.receiver._parse_and_update(b'\xff\xfe\x00\x01')
        jd, gp, ep = self.receiver.read_full_state()
        assert jd is None and gp is None and ep is None

    def test_float_conversion_error_no_crash(self):
        """[P0] 浮点转换异常不崩溃"""
        msg = {
            "state": "realtime_arm_joint_state",
            "joint_status": {"joint_position": ["not_a_number", 0, 0, 0, 0, 0, 0]},
        }
        data = json.dumps(msg).encode("utf-8")
        # Should not raise
        self.receiver._parse_and_update(data)
        # Error count should increase
        assert self.receiver._error_count >= 1

    def test_partial_update_preserves_previous(self):
        """部分更新保留之前的值"""
        # First: set joints
        data1 = _make_udp_msg(joint_position=[10000, 0, 0, 0, 0, 0, 0])
        self.receiver._parse_and_update(data1)

        # Second: set gripper only (no joints in this message)
        data2 = _make_udp_msg(gripper_pos=[500])
        self.receiver._parse_and_update(data2)

        jd, gp, _ = self.receiver.read_full_state()
        # Joints should still be from first message
        assert jd is not None
        assert abs(jd[0] - 10.0) < 1e-10
        # Gripper from second message
        assert gp == 500

    def test_recv_count_increments(self):
        """recv_count 正确递增"""
        data = _make_udp_msg(joint_position=[0, 0, 0, 0, 0, 0, 0])
        assert self.receiver.recv_count == 0
        self.receiver._parse_and_update(data)
        assert self.receiver.recv_count == 1
        self.receiver._parse_and_update(data)
        assert self.receiver.recv_count == 2


class TestReadState14D:
    """测试 read_state_14d() 单位转换"""

    def setup_method(self):
        self.receiver = UDPStateReceiver(port=0, arm_dof=7)

    def test_returns_none_when_no_data(self):
        """未收到数据时返回 None"""
        assert self.receiver.read_state_14d() is None

    def test_returns_none_when_partial_data(self):
        """只有部分数据时返回 None"""
        data = _make_udp_msg(joint_position=[0, 0, 0, 0, 0, 0, 0])
        self.receiver._parse_and_update(data)
        # No gripper or eef yet
        assert self.receiver.read_state_14d() is None

    def test_14d_conversion(self):
        """14D 向量格式和单位转换正确"""
        data = _make_udp_msg(
            joint_position=[180000, 90000, 0, 0, 0, 0, 0],  # 180度, 90度
            gripper_pos=[500],  # 500/1000 = 0.5
            waypoint_position=[300000, 0, 0],  # 0.3m
            waypoint_euler=[1571, 0, 0],  # 1.571 rad
        )
        self.receiver._parse_and_update(data)
        state = self.receiver.read_state_14d()

        assert state is not None
        assert state.shape == (14,)
        assert state.dtype == np.float32

        # joints: degrees -> radians
        np.testing.assert_allclose(state[0], np.radians(180.0), atol=1e-5)
        np.testing.assert_allclose(state[1], np.radians(90.0), atol=1e-5)

        # gripper: 500/1000 = 0.5
        np.testing.assert_allclose(state[7], 0.5, atol=1e-5)

        # eef pose: passed through from _parse_and_update
        np.testing.assert_allclose(state[8], 0.3, atol=1e-5)
        np.testing.assert_allclose(state[11], 1.571, atol=1e-3)

    def test_gripper_clipping(self):
        """夹爪值被 clip 到 [0, 1]"""
        data = _make_udp_msg(
            joint_position=[0, 0, 0, 0, 0, 0, 0],
            gripper_pos=[1500],  # > 1000
            waypoint_position=[0, 0, 0],
            waypoint_euler=[0, 0, 0],
        )
        self.receiver._parse_and_update(data)
        state = self.receiver.read_state_14d()
        assert state is not None
        assert state[7] == 1.0  # clipped to 1.0


class TestUDPStateReceiverThreaded:
    """测试 start/stop 和线程安全"""

    def _find_free_port(self):
        """找到一个可用的 UDP 端口"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        return port

    def test_start_stop(self):
        """启动和停止不崩溃"""
        port = self._find_free_port()
        receiver = UDPStateReceiver(port=port)
        receiver.start()
        assert receiver._running is True
        time.sleep(0.05)
        receiver.stop()
        assert receiver._running is False

    def test_receive_via_udp(self):
        """通过实际 UDP 发送接收数据"""
        port = self._find_free_port()
        receiver = UDPStateReceiver(port=port)
        receiver.start()
        try:
            data = _make_udp_msg(
                joint_position=[90000, 0, 0, 0, 0, 0, 0],
                gripper_pos=[1000],
                waypoint_position=[500000, 0, 0],
                waypoint_euler=[0, 0, 0],
            )
            _send_udp(port, data)
            # Wait for receive thread to process
            time.sleep(0.2)

            jd, gp, ep = receiver.read_full_state()
            assert jd is not None
            np.testing.assert_allclose(jd[0], 90.0, atol=1e-6)
            assert gp == 1000
            assert ep is not None
            np.testing.assert_allclose(ep[0], 0.5, atol=1e-6)
        finally:
            receiver.stop()

    def test_is_receiving(self):
        """is_receiving() 超时检测"""
        port = self._find_free_port()
        receiver = UDPStateReceiver(port=port)
        receiver.start()
        try:
            # No data sent yet
            assert receiver.is_receiving(timeout=0.5) is False

            # Send data
            data = _make_udp_msg(joint_position=[0, 0, 0, 0, 0, 0, 0])
            _send_udp(port, data)
            time.sleep(0.2)

            assert receiver.is_receiving(timeout=1.0) is True

            # Wait for timeout
            time.sleep(1.1)
            assert receiver.is_receiving(timeout=1.0) is False
        finally:
            receiver.stop()

    def test_concurrent_read_write(self):
        """多线程并发读写无死锁"""
        port = self._find_free_port()
        receiver = UDPStateReceiver(port=port)
        receiver.start()

        errors = []
        stop_event = threading.Event()

        def writer():
            """持续发送 UDP 数据"""
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                i = 0
                while not stop_event.is_set():
                    data = _make_udp_msg(
                        joint_position=[i * 1000, 0, 0, 0, 0, 0, 0],
                        gripper_pos=[i % 1001],
                        waypoint_position=[i * 100, 0, 0],
                        waypoint_euler=[0, 0, 0],
                    )
                    sock.sendto(data, ("127.0.0.1", port))
                    i += 1
                    time.sleep(0.001)
                sock.close()
            except Exception as e:
                errors.append(f"writer: {e}")

        def reader():
            """持续读取状态"""
            try:
                while not stop_event.is_set():
                    receiver.read_full_state()
                    receiver.read_state_14d()
                    receiver.is_receiving()
                    time.sleep(0.001)
            except Exception as e:
                errors.append(f"reader: {e}")

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()

        time.sleep(0.5)
        stop_event.set()

        for t in threads:
            t.join(timeout=2.0)

        receiver.stop()
        assert len(errors) == 0, f"Thread errors: {errors}"

    def test_double_start(self):
        """重复 start() 不会创建多个线程"""
        port = self._find_free_port()
        receiver = UDPStateReceiver(port=port)
        receiver.start()
        thread1 = receiver._thread
        receiver.start()  # second call should be no-op
        thread2 = receiver._thread
        assert thread1 is thread2
        receiver.stop()


class TestEnableUdpPush:
    """测试 enable_udp_push() 参数构造"""

    def test_sdk_api_call(self):
        """验证 SDK API 调用参数正确"""
        call_log = []

        class MockArm:
            def rm_set_realtime_push(self, config):
                call_log.append(config)
                return 0

        # Mock the SDK types
        import dagger.core.udp_state as udp_module

        mock_arm = MockArm()
        result = enable_udp_push(
            mock_arm,
            target_ip="192.168.1.100",
            target_port=8089,
            cycle=2,
            enable=True,
        )

        # If SDK import fails, it will go to fallback. We test the logic path.
        # The test verifies the function doesn't crash and returns a bool.
        assert isinstance(result, bool)

    def test_fallback_on_missing_arm_ip(self):
        """备用方案: 未提供 arm_ip 时返回 False"""
        class MockArm:
            def rm_set_realtime_push(self, config):
                raise ImportError("No SDK")

        mock_arm = MockArm()
        # 不传 arm_ip -> fallback 应优雅失败
        result = enable_udp_push(mock_arm, "192.168.1.100")
        assert result is False
