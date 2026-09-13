"""
Shared diagnostic logging utility.

Each session creates a timestamped subdirectory under dagger/logs/:
    dagger/logs/2025-01-15_14-30-22/
        dagger_node.log
        inference_bridge.log
        async_client.log

All modules that import from here share the same session directory,
so logs from the same run are grouped together.

Usage:
    from dagger.core.diag_log import create_diag_logger

    _diag_log = create_diag_logger("dagger_node")
    _diag_log("some message")
"""

import os
import time
import threading

_LOGS_BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")

# Session directory (shared across all loggers in the same process)
_session_dir: str | None = None
_session_lock = threading.Lock()


def _get_session_dir() -> str:
    """Get or create the session log directory (once per process)."""
    global _session_dir
    if _session_dir is not None:
        return _session_dir
    with _session_lock:
        if _session_dir is not None:
            return _session_dir
        ts = time.strftime("%Y-%m-%d_%H-%M-%S")
        _session_dir = os.path.join(_LOGS_BASE, ts)
        os.makedirs(_session_dir, exist_ok=True)
    return _session_dir


def get_session_dir() -> str:
    """Public accessor for the current session log directory."""
    return _get_session_dir()


def create_diag_logger(name: str, print_to_stdout: bool = True):
    """
    Create a diagnostic logger that writes to dagger/logs/<session>/<name>.log.

    Args:
        name: Log file name (without .log extension), e.g. "dagger_node"
        print_to_stdout: Also print to stdout (default True)

    Returns:
        A callable _diag_log(msg, throttle_key=None, throttle_sec=0) function.
        When throttle_key is provided, the message is only logged once per throttle_sec seconds.
    """
    _file_handle = None
    _write_count = 0
    _last_flush_time = 0.0
    _throttle_map: dict[str, float] = {}  # key -> last_log_monotonic

    def _diag_log(msg: str, throttle_key: str | None = None, throttle_sec: float = 0.0):
        nonlocal _file_handle, _write_count, _last_flush_time
        # Throttle 检查
        if throttle_key is not None and throttle_sec > 0:
            now_mono = time.monotonic()
            last = _throttle_map.get(throttle_key, 0.0)
            if (now_mono - last) < throttle_sec:
                return
            _throttle_map[throttle_key] = now_mono

        if _file_handle is None:
            session_dir = _get_session_dir()
            log_path = os.path.join(session_dir, f"{name}.log")
            _file_handle = open(log_path, "a")
            _file_handle.write(f"{'='*60}\n")
            _file_handle.write(f"[{name}] Session started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            _file_handle.write(f"{'='*60}\n")
            _last_flush_time = time.monotonic()
        # 使用 time.time() 获取完整的 wall clock 时间戳（含毫秒）
        _now = time.time()
        _ms = int((_now % 1) * 1000)
        line = f"[{time.strftime('%H:%M:%S', time.localtime(_now))}.{_ms:03d}] {msg}\n"
        _file_handle.write(line)
        # 定期 flush：每 50 次写入或每 0.5 秒，避免每次 flush 的 I/O 开销
        _write_count += 1
        now = time.monotonic()
        if _write_count >= 50 or (now - _last_flush_time) >= 0.5:
            _file_handle.flush()
            _write_count = 0
            _last_flush_time = now
        if print_to_stdout:
            print(msg)

    return _diag_log
