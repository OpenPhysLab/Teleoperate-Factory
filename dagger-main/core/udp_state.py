"""
UDP 状态接收器 -- 从睿尔曼机械臂接收实时状态推送。

复制自 vr_teleop/.../hardware.py 的 UDPStateReceiver 类。
原因: AsyncInferenceClient 运行在纯 Python 环境，不依赖 ROS2 workspace。

数据格式参考: https://develop.realman-robotics.com/robot4th/json/udpConfig/
"""
import json
import socket
import threading
import time
from typing import Optional, Tuple

import numpy as np


class UDPStateReceiver:
    """
    睿尔曼机械臂 UDP 状态接收器。

    通过 UDP 协议接收机械臂主动上报的状态数据:
    - 不阻塞: 独立 UDP 端口，与控制指令完全分离
    - 高频率: 最高 200Hz (5ms 周期)
    - 低延迟: 数据主动推送，read_full_state() 仅读缓存

    使用方法:
    1. 通过 enable_udp_push() 配置机械臂开启 UDP 上报
    2. 调用 start() 启动后台接收线程
    3. 调用 read_full_state() 获取最新状态
    4. 程序结束前调用 stop()
    """

    def __init__(self, port: int = 8089, arm_dof: int = 7):
        self.port = port
        self.arm_dof = arm_dof

        # 最新状态（线程安全）
        self._joint_deg: Optional[np.ndarray] = None
        self._gripper_pos: Optional[int] = None
        self._eef_pose: Optional[np.ndarray] = None  # [x,y,z,rx,ry,rz] 米/弧度
        self._last_recv_ts: Optional[float] = None
        self._lock = threading.Lock()

        # 后台线程
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._socket: Optional[socket.socket] = None

        # 统计
        self._recv_count = 0
        self._error_count = 0

    def start(self) -> None:
        """启动 UDP 接收线程"""
        if self._running:
            return
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # 允许多个进程绑定同一端口（Linux 3.9+）
        if hasattr(socket, 'SO_REUSEPORT'):
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        self._socket.bind(("0.0.0.0", self.port))
        self._socket.settimeout(0.1)
        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()
        print(f"[UDPStateReceiver] 已启动，监听端口 {self.port}")

    def _recv_loop(self) -> None:
        """后台接收循环"""
        while self._running:
            try:
                data, addr = self._socket.recvfrom(65536)
                self._parse_and_update(data)
            except socket.timeout:
                continue
            except Exception as exc:
                self._error_count += 1
                if self._error_count <= 5:
                    print(f"[UDPStateReceiver] 接收错误: {exc}")

    def _parse_and_update(self, data: bytes) -> None:
        """
        解析 UDP JSON 并更新状态。

        v2 修正:
        - [P0] 校验 msg["state"] == "realtime_arm_joint_state"
        - [P1] 兼容 waypoint position/euler 的数组和字典两种格式
        - [P1] 增加 hand 字段作为夹爪备用路径
        - [P0] 增加浮点转换异常处理

        数据格式 (睿尔曼文档):
        - state: "realtime_arm_joint_state" (消息类型标识)
        - joint_status.joint_position: 关节角度，精度 0.001 度
        - rm_plus_state.pos: 夹爪位置数组 (0-1000)
        - hand.pos: 夹爪位置备用字段 (部分固件版本)
        - waypoint.position: 末端位置 [x,y,z]，单位微米 (数组格式)
                          或 {"x":..,"y":..,"z":..} (字典格式，旧版固件)
        - waypoint.euler: 末端姿态 [rx,ry,rz]，单位 0.001 弧度 (数组格式)
                       或 {"rx":..,"ry":..,"rz":..} (字典格式，旧版固件)
        """
        try:
            msg = json.loads(data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return

        # [P0] 消息类型校验 — 参考 hardware.py:807
        if not isinstance(msg, dict):
            return
        if msg.get("state") != "realtime_arm_joint_state":
            return

        try:
            with self._lock:
                # --- 关节角度 (0.001 度 -> 度) ---
                js = msg.get("joint_status", {})
                jp = js.get("joint_position")
                if jp is not None and len(jp) >= self.arm_dof:
                    self._joint_deg = np.array(
                        [float(p) * 0.001 for p in jp[:self.arm_dof]],
                        dtype=np.float64
                    )

                # --- 夹爪 (RM+ 协议，备用 hand 字段) ---
                # 注意: rm_plus_state/hand 可能是 "disable" 字符串（未启用时）
                gripper_found = False
                rm_plus = msg.get("rm_plus_state")
                if isinstance(rm_plus, dict):
                    pos_arr = rm_plus.get("pos")
                    if pos_arr is not None and len(pos_arr) > 0:
                        self._gripper_pos = int(pos_arr[0])
                        gripper_found = True

                if not gripper_found:
                    # [P1] 备用: hand 字段 (部分固件版本)
                    hand = msg.get("hand")
                    if not isinstance(hand, dict):
                        hand = {}
                    hand_pos = hand.get("pos")
                    if hand_pos is not None:
                        if isinstance(hand_pos, list) and len(hand_pos) > 0:
                            self._gripper_pos = int(hand_pos[0])
                        elif isinstance(hand_pos, (int, float)):
                            self._gripper_pos = int(hand_pos)

                # --- 末端位姿 ---
                wp = msg.get("waypoint", {})
                wp_pos = wp.get("position")
                wp_euler = wp.get("euler")
                if wp_pos is not None and wp_euler is not None:
                    eef_pose = self._parse_waypoint(wp_pos, wp_euler)
                    if eef_pose is not None:
                        self._eef_pose = eef_pose

                self._last_recv_ts = time.monotonic()
                self._recv_count += 1

        except (ValueError, TypeError, IndexError) as exc:
            # [P0] 浮点转换/数组越界异常处理
            self._error_count += 1
            if self._error_count <= 5:
                print(f"[UDPStateReceiver] 解析错误: {exc}")

    def _parse_waypoint(self, pos, euler) -> Optional[np.ndarray]:
        """
        解析 waypoint 的 position 和 euler，兼容数组和字典两种格式。

        数组格式 (新版固件):
            position: [x_um, y_um, z_um]  (微米)
            euler: [rx_mrad, ry_mrad, rz_mrad]  (0.001 弧度)

        字典格式 (旧版固件, 参考 hardware.py:842-851):
            position: {"x": x_um, "y": y_um, "z": z_um}
            euler: {"rx": rx_mrad, "ry": ry_mrad, "rz": rz_mrad}
        """
        try:
            if isinstance(pos, list) and isinstance(euler, list):
                if len(pos) >= 3 and len(euler) >= 3:
                    x_m = float(pos[0]) * 1e-6
                    y_m = float(pos[1]) * 1e-6
                    z_m = float(pos[2]) * 1e-6
                    rx = float(euler[0]) * 0.001
                    ry = float(euler[1]) * 0.001
                    rz = float(euler[2]) * 0.001
                    return np.array([x_m, y_m, z_m, rx, ry, rz], dtype=np.float64)

            elif isinstance(pos, dict) and isinstance(euler, dict):
                # [P1] 字典格式兼容 (旧版固件)
                x_m = float(pos.get("x", 0)) * 1e-6
                y_m = float(pos.get("y", 0)) * 1e-6
                z_m = float(pos.get("z", 0)) * 1e-6
                rx = float(euler.get("rx", 0)) * 0.001
                ry = float(euler.get("ry", 0)) * 0.001
                rz = float(euler.get("rz", 0)) * 0.001
                return np.array([x_m, y_m, z_m, rx, ry, rz], dtype=np.float64)

        except (ValueError, TypeError):
            pass
        return None

    def read_full_state(self) -> Tuple[Optional[np.ndarray], Optional[int], Optional[np.ndarray]]:
        """
        读取最新完整状态（线程安全）。

        返回:
            (joint_deg, gripper_pos, eef_pose)
            - joint_deg: np.ndarray (arm_dof,) 关节角度（度）, 或 None
            - gripper_pos: int 夹爪位置 (0-1000), 或 None
            - eef_pose: np.ndarray (6,) [x,y,z,rx,ry,rz] 米/弧度, 或 None
        """
        with self._lock:
            jd = self._joint_deg.copy() if self._joint_deg is not None else None
            gp = self._gripper_pos
            ep = self._eef_pose.copy() if self._eef_pose is not None else None
        return jd, gp, ep

    def read_state_14d(self) -> Optional[np.ndarray]:
        """
        读取 14D 状态向量（与 VR 遥操格式一致）。

        重要: 必须使用 read_full_state() 一次性读取所有字段，
        不可拆分为多次读取，否则字段间可能不一致（被 UDP 接收线程更新）。

        返回:
            np.ndarray (14,) = [7 joints_rad, 1 gripper_01, 6 eef_pose]
            如果任何字段尚未收到数据，返回 None。
        """
        jd, gp, ep = self.read_full_state()
        if jd is None or gp is None or ep is None:
            return None
        joint_rad = np.radians(jd)
        gripper_01 = np.clip(gp / 1000.0, 0.0, 1.0)
        return np.concatenate([
            joint_rad, [gripper_01], ep
        ]).astype(np.float32)

    def is_receiving(self, timeout: float = 1.0) -> bool:
        """检查是否在 timeout 秒内收到过数据"""
        with self._lock:
            if self._last_recv_ts is None:
                return False
            return (time.monotonic() - self._last_recv_ts) < timeout

    def stop(self) -> None:
        """停止接收线程"""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._socket is not None:
            self._socket.close()
        print(
            f"[UDPStateReceiver] 已停止 "
            f"(收到 {self._recv_count} 包, {self._error_count} 错误)"
        )

    @property
    def recv_count(self) -> int:
        return self._recv_count


def enable_rm_plus(arm_ip: str, arm_port: int = 8080, baud: int = 115200) -> bool:
    """
    通过 JSON TCP 命令开启 RM+ 生态协议。

    必须在 enable_udp_push() 之前调用，否则 UDP 上报中 rm_plus_state 为 "disable"，
    无法获取夹爪数据。

    参考: hardware.py:258-285 _enable_rm_plus_json()

    参数:
        arm_ip: 机械臂 IP 地址
        arm_port: 机械臂 TCP 端口（默认 8080）
        baud: RM+ 串口波特率（0=禁用, 9600/115200/256000/460800）

    返回:
        bool: 是否成功
    """
    if baud <= 0:
        return False

    cmd = {"command": "set_rm_plus_mode", "mode": int(baud)}
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2.0)
        sock.connect((arm_ip, arm_port))
        sock.sendall(json.dumps(cmd).encode("utf-8"))
        response = sock.recv(1024).decode("utf-8")
        sock.close()

        resp_json = json.loads(response)
        if resp_json.get("set_state") is True:
            print(f"[RM+ Protocol] 已开启 RM+ 生态协议 (波特率={baud})")
            return True
        else:
            print(f"[RM+ Protocol] 开启失败，响应: {response}")
            return False
    except Exception as exc:
        print(f"[RM+ Protocol] 开启异常: {exc}")
        return False


def enable_udp_push(robot_arm, target_ip: str, target_port: int = 8089,
                    cycle: int = 5, enable: bool = True,
                    arm_ip: str = None, arm_port: int = 8080) -> bool:
    """
    通过 SDK API 配置机械臂开启/关闭 UDP 实时状态推送。

    v3 修正: SDK 失败（返回 -2 超时等）时自动走 TCP fallback。
    新增 arm_ip/arm_port 参数供 fallback 使用（RoboticArm 对象不暴露连接信息）。

    参数:
        robot_arm: RoboticArm 实例（self.robot.arm）
                   必须是已连接的 RoboticArm 对象，继承了 UdpConfig 类
        target_ip: 接收 UDP 数据的主机 IP
        target_port: UDP 端口（默认 8089）
        cycle: 推送周期（1=5ms/200Hz, 2=10ms/100Hz, 5=25ms/40Hz）
        enable: True=开启, False=关闭
        arm_ip: 机械臂 IP 地址（fallback 用，SDK 不暴露此信息）
        arm_port: 机械臂 TCP 端口（默认 8080）

    返回:
        bool: 是否成功
    """
    try:
        from Robotic_Arm.rm_ctypes_wrap import rm_realtime_push_config_t, rm_udp_custom_config_t

        # 构造自定义上报项配置
        custom = rm_udp_custom_config_t(
            joint_speed=1,           # 上报关节速度
            arm_current_status=1,    # 上报机械臂当前状态
            plus_base=1,             # 上报 RM+ 末端设备基础信息
            plus_state=1,            # 上报 RM+ 末端设备状态（含夹爪）
        )

        # 构造 UDP 推送配置
        config = rm_realtime_push_config_t(
            cycle=cycle,
            enable=enable,
            port=target_port,
            force_coordinate=0,      # 传感器坐标系
            ip=target_ip,
            custom_config=custom,
        )

        # 调用 SDK API
        result = robot_arm.rm_set_realtime_push(config)

        if result == 0:
            action = "开启" if enable else "关闭"
            print(
                f"[UDP Push] 已{action} UDP 推送: "
                f"{target_ip}:{target_port}, 周期={cycle} (={cycle*5}ms)"
            )
            return True
        else:
            # SDK 失败（-2=超时, -1=发送失败等），尝试 TCP fallback
            print(f"[UDP Push] SDK 返回码: {result}，尝试 TCP fallback...")
            return _enable_udp_push_fallback(
                arm_ip, arm_port, target_ip, target_port, cycle, enable
            )

    except Exception as e:
        print(f"[UDP Push] 配置异常: {e}")
        # 备用方案: 直接 TCP 发送 JSON 命令
        return _enable_udp_push_fallback(
            arm_ip, arm_port, target_ip, target_port, cycle, enable
        )


def _enable_udp_push_fallback(arm_ip, arm_port, target_ip, target_port, cycle, enable):
    """
    备用方案: 直接通过 TCP 发送 JSON 命令。
    参考 hardware.py:617-655 的 _send_json_command_direct()。
    当 SDK API 不可用或返回错误时使用。
    """
    if arm_ip is None:
        print("[UDP Push] fallback 失败: 未提供 arm_ip 参数")
        return False

    cmd = {
        "command": "set_realtime_push",
        "enable": enable,
        "port": target_port,
        "cycle": cycle,
        "ip": target_ip,
        "force_coordinate": 0,
        "custom": {
            "joint_speed": True,
            "arm_current_status": True,
            "rm_plus_state": True,
            "rm_plus_base": True,
        },
    }
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2.0)
        sock.connect((arm_ip, arm_port))
        sock.sendall(json.dumps(cmd).encode("utf-8"))
        response = sock.recv(1024).decode("utf-8")
        sock.close()

        resp_json = json.loads(response)
        if resp_json.get("state") is True or resp_json.get("set_realtime_push") is True:
            action = "开启" if enable else "关闭"
            print(f"[UDP Push] 已{action}（TCP fallback）: {target_ip}:{target_port}")
            return True
        else:
            print(f"[UDP Push] fallback 失败，响应: {response}")
            return False
    except Exception as exc:
        print(f"[UDP Push] fallback 异常: {exc}")
        return False
