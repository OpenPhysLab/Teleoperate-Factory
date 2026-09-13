"""
ring_buffer.py 单元测试

验证项:
- 空 buffer pop 返回 None
- update 后 size 正确
- pop_current 按 FIFO 顺序
- clear 后 is_empty 为 True
- last_popped 在 clear 后仍保留
- 多线程并发读写无死锁
"""

import sys
import os
import threading
import time

# 确保 dagger 包可导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dagger.core.ring_buffer import ActionRingBuffer


def test_empty_pop_returns_none():
    """空 buffer pop 返回 None"""
    buf = ActionRingBuffer()
    assert buf.pop_current() is None
    assert buf.is_empty
    assert buf.size == 0
    print("  [PASS] test_empty_pop_returns_none")


def test_update_size():
    """update 后 size 正确"""
    buf = ActionRingBuffer()
    size = buf.update([1, 2, 3, 4, 5])
    assert size == 5
    assert buf.size == 5
    assert not buf.is_empty
    print("  [PASS] test_update_size")


def test_pop_fifo_order():
    """pop_current 按 FIFO 顺序"""
    buf = ActionRingBuffer()
    buf.update(["a", "b", "c", "d"])
    assert buf.pop_current() == "a"
    assert buf.pop_current() == "b"
    assert buf.pop_current() == "c"
    assert buf.pop_current() == "d"
    assert buf.pop_current() is None
    print("  [PASS] test_pop_fifo_order")


def test_update_replaces_buffer():
    """update 替换整个缓冲区"""
    buf = ActionRingBuffer()
    buf.update([1, 2, 3])
    buf.pop_current()  # pop 1
    buf.update([10, 20, 30])  # 替换
    assert buf.size == 3
    assert buf.pop_current() == 10
    assert buf.pop_current() == 20
    assert buf.pop_current() == 30
    print("  [PASS] test_update_replaces_buffer")


def test_clear():
    """clear 后 is_empty 为 True"""
    buf = ActionRingBuffer()
    buf.update([1, 2, 3])
    buf.clear()
    assert buf.is_empty
    assert buf.size == 0
    assert buf.pop_current() is None
    print("  [PASS] test_clear")


def test_last_popped_survives_clear():
    """last_popped 在 clear 后仍保留"""
    buf = ActionRingBuffer()
    buf.update([1, 2, 3])
    buf.pop_current()  # pop 1
    buf.pop_current()  # pop 2
    assert buf.last_popped == 2
    buf.clear()
    assert buf.last_popped == 2  # clear 不清空 last_popped
    print("  [PASS] test_last_popped_survives_clear")


def test_last_popped_initially_none():
    """last_popped 初始为 None"""
    buf = ActionRingBuffer()
    assert buf.last_popped is None
    print("  [PASS] test_last_popped_initially_none")


def test_peek_current():
    """peek_current 查看但不取出"""
    buf = ActionRingBuffer()
    buf.update([1, 2, 3])
    assert buf.peek_current() == 1
    assert buf.peek_current() == 1  # 不变
    assert buf.size == 3  # 不变
    buf.pop_current()
    assert buf.peek_current() == 2
    print("  [PASS] test_peek_current")


def test_capacity():
    """超过 capacity 时自动丢弃旧数据"""
    buf = ActionRingBuffer(capacity=3)
    buf.update([1, 2, 3, 4, 5])  # 超过 capacity=3
    assert buf.size == 3
    assert buf.pop_current() == 3  # deque(maxlen=3) 保留最后 3 个
    assert buf.pop_current() == 4
    assert buf.pop_current() == 5
    print("  [PASS] test_capacity")


def test_concurrent_read_write():
    """多线程并发读写无死锁（10 写线程 + 10 读线程，1 秒）"""
    buf = ActionRingBuffer(capacity=100)
    errors = []
    stop_event = threading.Event()

    def writer(thread_id):
        count = 0
        while not stop_event.is_set():
            try:
                buf.update(list(range(thread_id * 100, thread_id * 100 + 10)))
                count += 1
            except Exception as e:
                errors.append(f"Writer {thread_id}: {e}")
                break

    def reader(thread_id):
        count = 0
        while not stop_event.is_set():
            try:
                action = buf.pop_current()
                _ = buf.last_popped
                _ = buf.size
                _ = buf.is_empty
                count += 1
            except Exception as e:
                errors.append(f"Reader {thread_id}: {e}")
                break

    threads = []
    for i in range(10):
        t = threading.Thread(target=writer, args=(i,))
        threads.append(t)
    for i in range(10):
        t = threading.Thread(target=reader, args=(i,))
        threads.append(t)

    for t in threads:
        t.start()

    time.sleep(1.0)
    stop_event.set()

    for t in threads:
        t.join(timeout=3.0)

    # 检查是否有线程还在运行（死锁）
    alive = [t for t in threads if t.is_alive()]
    assert len(alive) == 0, f"Deadlock detected: {len(alive)} threads still alive"
    assert len(errors) == 0, f"Errors: {errors}"
    print("  [PASS] test_concurrent_read_write (10W + 10R, 1s)")


def run_all():
    print("\n=== ring_buffer.py Unit Tests ===\n")
    test_empty_pop_returns_none()
    test_update_size()
    test_pop_fifo_order()
    test_update_replaces_buffer()
    test_clear()
    test_last_popped_survives_clear()
    test_last_popped_initially_none()
    test_peek_current()
    test_capacity()
    test_concurrent_read_write()
    print(f"\n=== All tests passed ===\n")


if __name__ == "__main__":
    run_all()
