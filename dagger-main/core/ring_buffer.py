"""动作环形缓冲区 — 线程安全的 ActionChunk 管理器

推理线程调用 update() 批量写入新的 action chunk。
执行线程调用 pop_current() 逐帧取出当前 action。

关键特性:
- update() 替换整个缓冲区（新推理基于更新的观测，旧 action 过时）
- pop_current() 返回 None 时，调用方应使用 last_popped 做 hold
- action_available Event: update() 时 set()，buffer 为空时 clear()
  exec loop 可用 wait_for_action() 替代 time.sleep() 减少 GIL 竞争
- 所有公开方法均线程安全
"""

import threading
from collections import deque


class ActionRingBuffer:
    """
    线程安全的动作环形缓冲区。

    推理线程调用 update() 批量写入新的 action chunk。
    执行线程调用 pop_current() 逐帧取出当前 action。
    """

    def __init__(self, capacity: int = 200):
        self._buffer = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._last_popped = None  # hold-last-action 用
        # Event 机制：减少 exec loop 空转时的 GIL 竞争
        # update() 时 set()，buffer 为空时 clear()
        self.action_available = threading.Event()

    def update(self, actions: list) -> int:
        """批量替换缓冲区（推理线程调用）。返回新 buffer 大小。"""
        with self._lock:
            self._buffer.clear()
            self._buffer.extend(actions)
            size = len(self._buffer)
        # 在锁外 set Event，唤醒等待中的 exec loop
        if size > 0:
            self.action_available.set()
        return size

    def pop_current(self):
        """取出当前 action（执行线程调用）。空则返回 None 并 clear Event。"""
        with self._lock:
            if not self._buffer:
                self.action_available.clear()
                return None
            action = self._buffer.popleft()
            self._last_popped = action
            # buffer 取空后 clear Event
            if not self._buffer:
                self.action_available.clear()
            return action

    def peek_current(self):
        """查看当前 action 但不取出。"""
        with self._lock:
            return self._buffer[0] if self._buffer else None

    @property
    def last_popped(self):
        """最近一次 pop 出的 action（用于 hold position 和录制）"""
        with self._lock:
            return self._last_popped

    def wait_for_action(self, timeout: float) -> bool:
        """等待新 action 到达（释放 GIL，不做忙等）。

        Args:
            timeout: 最大等待秒数

        Returns:
            True 表示有 action 可用，False 表示超时
        """
        return self.action_available.wait(timeout=timeout)

    def clear(self):
        """清空缓冲区，但保留 last_popped 作为 hold 过渡。

        模式切换时调用：丢弃过时的 action chunk，但保留最后一个已执行的
        action 供 hold_on_empty 使用，避免切换后出现无 action 的空窗期。
        完全清空（含 last_popped）请用 reset()。
        """
        with self._lock:
            self._buffer.clear()
            # 注意：不清空 _last_popped，让 exec loop 在新推理完成前能 hold 旧 action
            self.action_available.clear()

    def reset(self):
        """完全重置：清空缓冲区 + last_popped + Event。用于 session 边界。"""
        with self._lock:
            self._buffer.clear()
            self._last_popped = None
            self.action_available.clear()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._buffer)

    @property
    def is_empty(self) -> bool:
        with self._lock:
            return len(self._buffer) == 0
