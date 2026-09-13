"""
夹爪异步控制器（复制自 RoboCOIN/src/lerobot/extensions/unified_deploy/client/gripper_controller.py）

使用 JSON socket + RM+ 生态协议控制 Realman 夹爪。
"""

import json
import socket
import threading
from queue import Queue, Empty
from typing import Optional


class GripperAsyncController:
    """
    夹爪异步控制器

    使用 JSON socket + RM+ 生态协议异步控制 Realman 夹爪。
    特点：
    - 异步发送，不阻塞主控制循环
    - 带死区判断，避免频繁发送相近指令
    - 自动管理 RM+ 生态模式
    """

    def __init__(
        self,
        ip: str,
        port: int,
        baud: int = 115200,
        deadband: int = 20,
        dry_run: bool = False
    ):
        self.ip = ip
        self.port = port
        self.baud = baud
        self.deadband = deadband
        self.dry_run = dry_run

        self._last_cmd: Optional[int] = None
        self._queue: Queue = Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._sock: Optional[socket.socket] = None

        if not dry_run:
            self._create_socket()
            self._enable_rm_plus()
            self._thread = threading.Thread(target=self._control_loop, daemon=True)
            self._thread.start()

    def _create_socket(self) -> bool:
        """创建 JSON socket 连接"""
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.settimeout(2.0)
            self._sock.connect((self.ip, self.port))
            self._sock.settimeout(0.5)
            print(f"[Gripper] JSON socket 已连接 {self.ip}:{self.port}")
            return True
        except Exception as e:
            print(f"[Gripper] 连接失败: {e}")
            self._sock = None
            return False

    def _send_json(self, cmd: dict) -> Optional[dict]:
        """发送 JSON 命令并接收响应"""
        if self._sock is None:
            return None
        try:
            self._sock.sendall(json.dumps(cmd).encode("utf-8"))
            return json.loads(self._sock.recv(4096).decode("utf-8"))
        except Exception:
            return None

    def _enable_rm_plus(self) -> bool:
        """启用 RM+ 生态模式"""
        if self._sock is None:
            return False
        resp = self._send_json({"command": "set_rm_plus_mode", "mode": self.baud})
        if resp and (resp.get("arm_err") == 0 or resp.get("set_state") == True):  # noqa: E712
            print(f"[Gripper] RM+ 生态已开启 (波特率={self.baud})")
            return True
        print(f"[Gripper] RM+ 开启失败: {resp}")
        return False

    def _control_loop(self):
        """异步控制循环"""
        while not self._stop_event.is_set():
            try:
                pos = self._queue.get(timeout=0.1)
                self._send_json({
                    "command": "set_gripper_position",
                    "position": int(pos),
                    "block": False
                })
            except Empty:
                continue

    def send_async(self, position: int):
        """异步发送夹爪位置（带死区判断）"""
        if self.dry_run:
            return

        if self._last_cmd is not None and abs(position - self._last_cmd) < self.deadband:
            return

        try:
            while not self._queue.empty():
                self._queue.get_nowait()
            self._queue.put_nowait(position)
            self._last_cmd = position
        except Exception:
            pass

    def close(self):
        """关闭控制器"""
        self._stop_event.set()
        if not self.dry_run and self._sock:
            self._send_json({"command": "set_rm_plus_mode", "mode": 0})
            try:
                self._sock.close()
            except Exception:
                pass
            print("[Gripper] 控制器已关闭")
