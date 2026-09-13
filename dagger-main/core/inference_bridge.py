"""
Inference Bridge - gRPC inference layer for DAgger node (Phase 3B).

Extracted from AsyncInferenceClient: handles gRPC communication, observation
building, action parsing, and ring buffer management. Does NOT manage robot
connections (no RoboticArm, no camera init).

Thread model:
  - Main thread: calls update_observation() with latest state + images
  - Inference thread: reads observation snapshot, sends gRPC, updates ring buffer
  - Execution thread: calls get_next_action() to pop from ring buffer

Usage:
    bridge = InferenceBridge(config)
    bridge.connect()
    bridge.start_inference_loop()
    # ... in control loop:
    bridge.update_observation(state_14d, images)
    action = bridge.get_next_action()
    # ... on mode switch:
    bridge.pause()
    bridge.clear_buffer()
    bridge.resume()
    # ... on shutdown:
    bridge.stop_inference_loop()
    bridge.disconnect()
"""

import math
import os
import sys
import pickle
import threading
import time
from dataclasses import dataclass
from typing import Optional, Dict, List

import numpy as np

import grpc

# === Path setup ===
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ROBOCOIN_SRC = os.path.join(_PROJECT_ROOT, "RoboCOIN", "src")
if _ROBOCOIN_SRC not in sys.path:
    sys.path.insert(0, _ROBOCOIN_SRC)

from dagger.core.ring_buffer import ActionRingBuffer

# === 诊断日志（按 session 时间戳分目录）===
from dagger.core.diag_log import create_diag_logger
_diag_log = create_diag_logger("inference_bridge", print_to_stdout=True)

# Data types: must use original module path for pickle serialization
from lerobot.extensions.unified_deploy.core.data_types import (
    UnifiedObservation,
    UnifiedAction,
    ActionChunk,
)

# gRPC transport
import lerobot.transport.services_pb2 as services_pb2
import lerobot.transport.services_pb2_grpc as services_pb2_grpc
from lerobot.transport.utils import send_bytes_in_chunks, grpc_channel_options


@dataclass
class InferenceBridgeConfig:
    """Configuration for InferenceBridge."""
    server_address: str = "127.0.0.1:50051"
    task: str = ""
    f_exec: float = 30.0
    n_action_steps: int = 0
    T_inter: float = 0.0
    t_inf: float = 0.0
    chunk_size_threshold: int = 0
    hold_on_empty: bool = True


class InferenceBridge:
    """
    gRPC inference bridge for DAgger node.

    Manages the inference thread, ring buffer, and gRPC communication.
    Does NOT manage robot connections or cameras.
    """

    def __init__(self, config: InferenceBridgeConfig):
        self.config = config

        # gRPC
        self.channel = None
        self.stub = None

        # Ring buffer
        self.ring_buffer = ActionRingBuffer(capacity=200)

        # Threading
        self.shutdown_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # Start unpaused
        self._inference_thread: Optional[threading.Thread] = None

        # Observation snapshot (thread-safe)
        self._obs_lock = threading.Lock()
        self._obs_state_14d: Optional[np.ndarray] = None
        self._obs_images: Dict[str, np.ndarray] = {}
        self._obs_raw_image_msgs: Optional[Dict] = None  # 轻量模式：raw ROS2 Image msg 引用
        self._image_converter = None  # 注入的图像转换函数：raw_msgs -> {cam: numpy_rgb}
        self._obs_timestamp: float = 0.0
        self._obs_updated = threading.Event()
        self._obs_update_count = 0  # DIAG: 观测更新计数
        self._obs_update_mono: float = 0.0  # DIAG: 上次更新的 monotonic 时间

        # Server metadata
        self.server_metadata: Optional[Dict] = None
        self.server_task: Optional[str] = None
        self.server_action_space: Optional[str] = None

        # Inference stats
        self._model_chunk_size: Optional[int] = None
        self._policy_action_dim: int = 8
        self._infer_count = 0
        self._t_inf_avg = 0.0
        self._t_inf_ema = 0.0
        self._t_inf_ema_alpha = 0.3
        self._t_inf_ema_initialized = False
        self.timestep = 0

        # Warmup complete flag
        self._warmup_done = threading.Event()

    # =========================================================================
    # Connection management
    # =========================================================================

    def connect(self, timeout: float = 180.0, retry_interval: float = 2.0):
        """Connect to gRPC PolicyServer with retry.

        Args:
            timeout: Maximum seconds to wait for PolicyServer to become ready.
                     Default 180s to support large models (e.g. Pi0 ~2min load).
            retry_interval: Seconds between retry attempts.
        """
        print(f"[InferenceBridge] Connecting to PolicyServer: {self.config.server_address}")
        self.channel = grpc.insecure_channel(
            self.config.server_address,
            options=grpc_channel_options(),
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)

        deadline = time.monotonic() + timeout
        attempt = 0
        while True:
            attempt += 1
            try:
                self.stub.Ready(services_pb2.Empty())
                print(f"[InferenceBridge] PolicyServer connected (attempt {attempt})")
                return
            except grpc.RpcError as e:
                if time.monotonic() >= deadline:
                    raise ConnectionError(
                        f"Cannot connect to PolicyServer after {timeout:.0f}s: {e}"
                    )
                remaining = deadline - time.monotonic()
                print(
                    f"[InferenceBridge] Waiting for PolicyServer ({attempt}, "
                    f"{remaining:.0f}s remaining): {e}"
                )
                if self.shutdown_event.is_set():
                    raise ConnectionError("Shutdown requested during connect")
                time.sleep(retry_interval)

    @staticmethod
    def probe_server(server_address: str, timeout_sec: float = 1.0) -> bool:
        """Quick non-blocking check if PolicyServer is reachable and ready.

        Creates a temporary gRPC channel, calls Ready(), and closes.
        Designed for periodic polling from IDLE mode (no persistent connection).

        Args:
            server_address: gRPC address (e.g. "127.0.0.1:50051")
            timeout_sec: Max seconds to wait for response.

        Returns:
            True if server responded to Ready(), False otherwise.
        """
        try:
            channel = grpc.insecure_channel(server_address)
            stub = services_pb2_grpc.AsyncInferenceStub(channel)
            stub.Ready(services_pb2.Empty(), timeout=timeout_sec)
            channel.close()
            return True
        except Exception:
            return False

    def disconnect(self):
        """Disconnect from PolicyServer."""
        self.stop_inference_loop()
        if self.channel:
            self.channel.close()
            self.channel = None
            self.stub = None
        print("[InferenceBridge] Disconnected")

    # =========================================================================
    # Image converter injection
    # =========================================================================

    def set_image_converter(self, converter):
        """注入图像转换函数，由 dagger_node 提供。

        Args:
            converter: callable, raw_msgs dict -> {cam_name: numpy_rgb} dict
        """
        self._image_converter = converter

    # =========================================================================
    # Observation update (called from main/control thread)
    # =========================================================================

    def update_observation(self, state_14d: np.ndarray, images: Optional[Dict[str, np.ndarray]] = None,
                           raw_image_msgs: Optional[Dict] = None):
        """
        Update the observation snapshot (thread-safe).

        Called from the main control thread with latest robot state and images.
        支持两种模式：
        - 轻量模式：传 raw_image_msgs（只存 msg 引用 + state copy，<1ms）
        - 传统模式：传 images（存 image 引用，不再 .copy()）

        Args:
            state_14d: 14-dim state [7 joints_rad, 1 gripper_norm, 6 eef_pose]
            images: dict of camera_name -> RGB image (np.ndarray)，传统模式
            raw_image_msgs: dict of camera_name -> raw ROS2 Image msg，轻量模式
        """
        with self._obs_lock:
            self._obs_state_14d = state_14d.copy()
            if raw_image_msgs is not None:
                # 轻量模式：只存 msg 引用的浅拷贝
                self._obs_raw_image_msgs = dict(raw_image_msgs)
            elif images is not None:
                # 传统模式：存 image 引用（不再 .copy()）
                self._obs_images = images
                self._obs_raw_image_msgs = None
            self._obs_timestamp = time.time()
            self._obs_update_mono = time.monotonic()
            self._obs_update_count += 1
        self._obs_updated.set()

    def _build_observation(self) -> Optional[UnifiedObservation]:
        """Build UnifiedObservation from the current snapshot.

        轻量模式下在推理线程中调用 _image_converter 做图像转换。
        """
        with self._obs_lock:
            if self._obs_state_14d is None:
                return None
            state = self._obs_state_14d.copy()
            raw_msgs = dict(self._obs_raw_image_msgs) if self._obs_raw_image_msgs else None
            images = {k: v for k, v in self._obs_images.items()} if raw_msgs is None else {}
            ts = self._obs_timestamp
            obs_age_ms = (time.monotonic() - self._obs_update_mono) * 1000
            obs_count = self._obs_update_count

        # 轻量模式：在推理线程中做图像转换
        if raw_msgs is not None:
            if self._image_converter is not None:
                images = self._image_converter(raw_msgs)
            elif self._obs_images:
                # image_converter 未注入，回退到传统模式缓存
                images = {k: v for k, v in self._obs_images.items()}
            else:
                return None

        joints = state[:7].astype(np.float32)
        gripper = float(np.clip(state[7], 0.0, 1.0))
        ee_pose = state[8:14].astype(np.float32)

        # DIAG: 保存诊断信息供推理循环使用
        self._diag_obs_age_ms = obs_age_ms
        self._diag_obs_count = obs_count
        self._diag_obs_gripper_raw = float(state[7])
        self._diag_obs_gripper_clipped = gripper

        # === DIAG-COMPARE: 每帧观测构建写文件（obs_age 是关键指标）===
        if not hasattr(self, '_build_obs_count'):
            self._build_obs_count = 0
        self._build_obs_count += 1
        _diag_log(
            f"[BRIDGE-OBS] #{self._build_obs_count}: "
            f"gripper={gripper:.4f}, "
            f"obs_age={obs_age_ms:.1f}ms, "
            f"obs_count={obs_count}, "
            f"buf={self.ring_buffer.size}, "
            f"must_go={self.ring_buffer.is_empty}"
        )

        return UnifiedObservation(
            timestamp=ts,
            timestep=self.timestep,
            must_go=self.ring_buffer.is_empty,
            joints=joints,
            ee_pose=ee_pose,
            gripper=gripper,
            images=images,
            task=self.config.task,
        )

    # =========================================================================
    # Action retrieval (called from control thread)
    # =========================================================================

    def get_next_action(self) -> Optional[UnifiedAction]:
        """Pop the next action from the ring buffer."""
        return self.ring_buffer.pop_current()

    @property
    def last_action(self) -> Optional[UnifiedAction]:
        """Last popped action (for hold-last-action)."""
        return self.ring_buffer.last_popped

    @property
    def buffer_size(self) -> int:
        return self.ring_buffer.size

    @property
    def buffer_empty(self) -> bool:
        return self.ring_buffer.is_empty

    def clear_buffer(self):
        """Clear the ring buffer (e.g., on mode switch)."""
        self.ring_buffer.clear()

    def reset_buffer(self):
        """完全重置 buffer + last_popped。用于 session 边界，防止 hold 旧 action 导致跳变。"""
        self.ring_buffer.reset()

    # =========================================================================
    # Pause / Resume
    # =========================================================================

    def pause(self):
        """Pause the inference thread."""
        self._pause_event.clear()
        print("[InferenceBridge] Inference paused")

    def resume(self):
        """Resume the inference thread."""
        self._pause_event.set()
        print("[InferenceBridge] Inference resumed")

    @property
    def is_paused(self) -> bool:
        return not self._pause_event.is_set()

    @property
    def is_connected(self) -> bool:
        """gRPC 通道是否已连接（channel 和 stub 均存在）。"""
        return self.channel is not None and self.stub is not None

    @property
    def warmup_done(self) -> bool:
        return self._warmup_done.is_set()

    @property
    def policy_action_dim(self) -> int:
        return self._policy_action_dim

    # =========================================================================
    # Inference thread management
    # =========================================================================

    def start_inference_loop(self):
        """Start the background inference thread."""
        if self._inference_thread is not None and self._inference_thread.is_alive():
            print("[InferenceBridge] Inference thread already running")
            return

        # 重置 warmup 状态和 buffer，确保新推理线程从干净状态开始
        # （第二次 start 时 _warmup_done 可能还是上次的 set 状态）
        self._warmup_done.clear()
        self.ring_buffer.reset()  # 完全重置，包括 last_popped

        # warmup 竞态修复：推理线程 warmup 完成后进入主循环时被 _pause_event.wait() 阻塞，
        # 直到 _connect_and_start_inference 完成 obs 刷新 + buffer 清理后才 resume。
        self._pause_event.clear()

        self.shutdown_event.clear()
        self._inference_thread = threading.Thread(
            target=self._inference_loop, daemon=True
        )
        self._inference_thread.start()

    def stop_inference_loop(self):
        """Stop the background inference thread."""
        self.shutdown_event.set()
        self._pause_event.set()  # Unblock if paused
        self._obs_updated.set()  # Unblock if waiting for obs
        if self._inference_thread is not None:
            self._inference_thread.join(timeout=5.0)
            self._inference_thread = None

    # =========================================================================
    # Warmup and parameter derivation (from AsyncInferenceClient)
    # =========================================================================

    def _measure_inference_time(self, n_warmup: int = 5) -> float:
        """
        Warmup: measure stable t_inf via multiple inferences.

        If config.t_inf > 0, uses configured value (still runs 1 inference for chunk_size).
        Otherwise runs n_warmup inferences, discards first (GPU cold start), averages rest.
        """
        n_warmup = max(n_warmup, 2)

        if self.config.t_inf > 0:
            print(f"[Warmup] Using configured t_inf={self.config.t_inf:.3f}s")
            self._warmup_single(tag="chunk_size probe")
            return self.config.t_inf

        print(f"[Warmup] Measuring inference latency ({n_warmup} runs, discarding first)...")

        # gRPC ready 已由 connect() 保证，直接进入观测等待

        # Wait for first observation
        print("[Warmup] Waiting for first observation...")
        deadline = time.monotonic() + 10.0
        while not self.shutdown_event.is_set():
            with self._obs_lock:
                has_obs = self._obs_state_14d is not None
            if has_obs:
                break
            if time.monotonic() > deadline:
                print("[Warmup] ERROR: No observation received within 10s")
                return 0.0
            time.sleep(0.1)

        t_inf_list = []
        last_actions = None

        for i in range(n_warmup):
            if self.shutdown_event.is_set():
                return 0.0
            result = self._warmup_single(tag=f"run {i+1}/{n_warmup}")
            if result is None:
                continue
            t_inf_i, actions = result
            t_inf_list.append(t_inf_i)
            last_actions = actions

        if not t_inf_list:
            print("[Warmup] ERROR: All warmup inferences failed — "
                  "PolicyServer 可能未正确加载模型或推理异常")
            _diag_log("[WARMUP-FATAL] 所有 warmup 推理均失败，"
                      "请检查 PolicyServer 终端输出是否有异常信息")
            return 0.0

        if len(t_inf_list) >= 2:
            t_inf_cold = t_inf_list[0]
            t_inf_stable = t_inf_list[1:]
            t_inf = sum(t_inf_stable) / len(t_inf_stable)
            print(f"[Warmup] Cold start t_inf={t_inf_cold:.3f}s (discarded)")
            print(f"[Warmup] Stable: {[f'{t:.3f}s' for t in t_inf_stable]}")
            print(f"[Warmup] Average t_inf={t_inf:.3f}s ({len(t_inf_stable)} runs)")
        else:
            t_inf = t_inf_list[0]
            print(f"[Warmup] Single measurement t_inf={t_inf:.3f}s (may include cold start)")

        # Write last warmup result to buffer
        if last_actions is not None:
            if self.config.n_action_steps > 0:
                last_actions = last_actions[:self.config.n_action_steps]
            K_inf = int(t_inf * self.config.f_exec)
            if K_inf < len(last_actions):
                valid_actions = last_actions[K_inf:]
            else:
                valid_actions = last_actions[-1:]
            self.ring_buffer.update(valid_actions)

        return t_inf

    def _warmup_single(self, tag: str = "") -> Optional[tuple]:
        """Execute a single warmup inference."""
        try:
            obs = self._build_observation()
            if obs is None:
                print(f"[Warmup] {tag}: No observation available")
                return None

            obs_bytes = pickle.dumps(obs)

            t_start = time.monotonic()
            obs_iterator = send_bytes_in_chunks(
                obs_bytes,
                services_pb2.Observation,
                log_prefix=f"[Warmup:{tag}] Observation",
                silent=True,
            )
            self.stub.SendObservations(obs_iterator)
            response = self.stub.GetActions(services_pb2.Empty())
            t_inf = time.monotonic() - t_start

            actions = self._parse_response(response)
            if not actions:
                _resp_len = len(response.data) if response else -1
                print(f"[Warmup] {tag}: Empty actions (response.data length={_resp_len})")
                _diag_log(f"[WARMUP-FAIL] {tag}: response.data={_resp_len}B, "
                          f"可能原因: PolicyServer 推理异常，请检查 Server 终端输出")
                return None

            chunk_size = len(actions)
            if self._model_chunk_size is None:
                self._model_chunk_size = chunk_size
                if actions:
                    a0 = actions[0]
                    # +1 for gripper: gRPC returns arm-only actions (e.g. 6D or 7D),
                    # but the full policy action includes a gripper dimension appended
                    # during action parsing (_parse_actions). This matches the convention
                    # used by DAggerRecorder and the execution loop.
                    self._policy_action_dim = len(a0.data) + 1
            elif chunk_size != self._model_chunk_size:
                print(f"[Warmup] WARNING: chunk_size mismatch: "
                      f"prev={self._model_chunk_size}, now={chunk_size}")

            print(f"[Warmup] {tag}: t_inf={t_inf:.3f}s, chunk_size={chunk_size}")
            return (t_inf, actions)

        except Exception as e:
            print(f"[Warmup] {tag} failed: {e}")
            return None

    def _derive_parameters(self, t_inf: float) -> dict:
        """Derive runtime parameters from measured t_inf."""
        f_exec = self.config.f_exec
        T_step = 1.0 / f_exec

        L = self._model_chunk_size
        if not L or L == 0:
            print("[Parameters] ERROR: chunk_size unknown, degrading to passive mode")
            return {
                "L": 0, "L_eff": 0, "K_inf": 0, "L_valid": 0,
                "T_inter_max": 0.0, "T_inter": 0.0,
                "chunk_size_threshold": 2, "feasible": False,
            }

        n_action_steps = self.config.n_action_steps
        L_eff = min(n_action_steps, L) if n_action_steps > 0 else L
        K_inf = int(math.ceil(t_inf * f_exec))
        L_valid = L_eff - K_inf
        T_inter_max = L_eff * T_step - t_inf

        if self.config.T_inter > 0:
            T_inter = min(self.config.T_inter, T_inter_max) if T_inter_max > 0 else self.config.T_inter
        else:
            T_inter = T_inter_max * 0.8 if T_inter_max > 0 else 0.0

        if self.config.chunk_size_threshold > 0:
            chunk_size_threshold = self.config.chunk_size_threshold
        else:
            chunk_size_threshold = max(2, K_inf)

        feasible = L_valid > 0 and T_inter_max > 0

        print(f"\n{'='*60}")
        print(f"[InferenceBridge] Parameter derivation")
        print(f"{'='*60}")
        print(f"  chunk_size (L):        {L}")
        print(f"  effective (L_eff):     {L_eff}")
        print(f"  f_exec:                {f_exec} Hz")
        print(f"  t_inf:                 {t_inf:.3f}s")
        print(f"  K_inf:                 {K_inf}")
        print(f"  L_valid:               {L_valid}")
        print(f"  T_inter_max:           {T_inter_max:.3f}s")
        print(f"  T_inter:               {T_inter:.3f}s")
        print(f"  threshold:             {chunk_size_threshold}")
        print(f"  feasible:              {'PASS' if feasible else 'FAIL'}")
        if not feasible:
            if L_valid <= 0:
                print(f"  WARNING: L_valid={L_valid} <= 0, inference too slow for chunk size")
            if T_inter_max <= 0:
                print(f"  WARNING: T_inter_max={T_inter_max:.3f}s <= 0")
        print(f"{'='*60}\n")

        return {
            "L": L, "L_eff": L_eff, "K_inf": K_inf, "L_valid": L_valid,
            "T_inter_max": T_inter_max, "T_inter": T_inter,
            "chunk_size_threshold": chunk_size_threshold, "feasible": feasible,
        }

    # =========================================================================
    # Inference loop (background thread)
    # =========================================================================

    def _inference_loop(self):
        """
        Background inference thread.

        Two modes:
        - Feasible: timed T_inter + safety-net polling
        - Degraded: passive wait for buffer <= threshold
        """
        print("[InferenceBridge] Inference thread started")

        # Warmup + parameter derivation
        t_inf_measured = self._measure_inference_time()
        t_inf = self.config.t_inf if self.config.t_inf > 0 else t_inf_measured
        params = self._derive_parameters(t_inf)

        T_inter = params["T_inter"]
        threshold = params["chunk_size_threshold"]
        feasible = params["feasible"]

        if not feasible:
            print("[InferenceBridge] WARN: System not feasible, degrading to passive mode")
            threshold = 2
            T_inter = 0

        # 关键参数记录到 _diag_log，确保被捕获到日志文件
        _diag_log(f"[INFER-PARAMS] t_inf={t_inf*1000:.1f}ms, T_inter={T_inter*1000:.0f}ms, "
                  f"threshold={threshold}, feasible={feasible}, "
                  f"L={params['L']}, L_eff={params['L_eff']}, K_inf={params['K_inf']}, "
                  f"L_valid={params['L_valid']}")

        self._warmup_done.set()

        # Main loop
        while not self.shutdown_event.is_set():
            # Check pause
            self._pause_event.wait()
            if self.shutdown_event.is_set():
                break

            t_start = time.monotonic()

            # Degraded mode: wait for buffer drain
            # 使用 Event.wait 替代短间隔轮询，减少 GIL 竞争
            if not feasible:
                while not self.shutdown_event.is_set():
                    if self.ring_buffer.size <= threshold:
                        break
                    self._pause_event.wait(timeout=0.02)
                if self.shutdown_event.is_set():
                    break
                t_start = time.monotonic()

            try:
                # 1. Build observation
                _t_obs = time.monotonic()
                obs = self._build_observation()
                _t_obs_done = time.monotonic()
                if obs is None:
                    time.sleep(0.05)
                    continue

                # === DEBUG: 打印发送给 server 的观测 + 新鲜度 ===
                if True:
                    import numpy as _np
                    obs_joints_deg = _np.degrees(obs.joints) if obs.joints is not None else None
                    obs_age = getattr(self, '_diag_obs_age_ms', -1)
                    obs_count = getattr(self, '_diag_obs_count', -1)

                    # 计算与上次推理的 joints 变化量
                    if hasattr(self, '_last_obs_joints') and self._last_obs_joints is not None:
                        delta_deg = _np.degrees(_np.abs(obs.joints - self._last_obs_joints))
                        max_delta = _np.max(delta_deg)
                        delta_str = f"max_delta={max_delta:.3f}deg"
                    else:
                        delta_str = "first_obs"
                    self._last_obs_joints = obs.joints.copy()

                    _diag_log(f"[DIAG-OBS] infer#{self._infer_count} "
                          f"joints(deg): {[f'{v:.2f}' for v in obs_joints_deg] if obs_joints_deg is not None else 'None'}, "
                          f"gripper: {obs.gripper:.3f}, "
                          f"obs_age={obs_age:.1f}ms, obs_count={obs_count}, {delta_str}, "
                          f"images: {list(obs.images.keys())}, "
                          f"img_shapes: {[obs.images[k].shape for k in obs.images]}")

                _t_pickle = time.monotonic()
                obs_bytes = pickle.dumps(obs)
                _t_pickle_done = time.monotonic()

                # 2. gRPC inference
                _t_grpc = time.monotonic()
                response = self._grpc_infer(obs_bytes)
                _t_grpc_done = time.monotonic()
                if response is None:
                    time.sleep(0.1)
                    continue

                # 3. Parse ActionChunk
                _t_parse = time.monotonic()
                actions = self._parse_response(response)
                _t_parse_done = time.monotonic()
                if not actions:
                    self._consecutive_empty = getattr(self, '_consecutive_empty', 0) + 1
                    backoff = min(0.1 * self._consecutive_empty, 2.0)  # 最大 2 秒
                    if self._consecutive_empty <= 3:
                        _diag_log(f"[INFER-RETRY] 连续空响应 #{self._consecutive_empty}, "
                                  f"退避 {backoff*1000:.0f}ms")
                    time.sleep(backoff)
                    continue

                if self.config.n_action_steps > 0:
                    actions = actions[:self.config.n_action_steps]

                # 成功收到 actions，重置连续空响应计数器
                self._consecutive_empty = 0
                t_inf_actual = time.monotonic() - t_start

                if not self._t_inf_ema_initialized:
                    self._t_inf_ema = t_inf_actual
                    self._t_inf_ema_initialized = True
                else:
                    self._t_inf_ema = (
                        self._t_inf_ema_alpha * t_inf_actual
                        + (1 - self._t_inf_ema_alpha) * self._t_inf_ema
                    )

                t_inf_for_clip = max(t_inf_actual, self._t_inf_ema)
                K_inf_actual = int(t_inf_for_clip * self.config.f_exec) + 15
                if K_inf_actual < len(actions):
                    valid_actions = actions[K_inf_actual:]
                else:
                    valid_actions = actions[-1:]

                # 5. Update ring buffer
                _t_buf = time.monotonic()
                buf_size = self.ring_buffer.update(valid_actions)
                _t_buf_done = time.monotonic()

                # === 关键诊断：记录 observation gripper 和 action chunk 的 gripper 范围 ===
                _obs_grip = getattr(self, '_diag_obs_gripper_clipped', -1)
                _act_grippers = [a.gripper for a in valid_actions]
                _act_grip_min = min(_act_grippers) if _act_grippers else -1
                _act_grip_max = max(_act_grippers) if _act_grippers else -1
                _act_grip_first = _act_grippers[0] if _act_grippers else -1
                _diag_log(
                    f"[CHUNK-PUSH] infer#{self._infer_count}: "
                    f"obs_grip={_obs_grip:.4f}, "
                    f"act_grip_range=[{_act_grip_min:.4f}, {_act_grip_max:.4f}], "
                    f"act_grip_first={_act_grip_first:.4f}, "
                    f"chunk_size={len(valid_actions)}, "
                    f"buf_after={buf_size}"
                )

                # === TIMING: 推理线程各阶段耗时 ===
                _obs_age = getattr(self, '_diag_obs_age_ms', -1)
                _obs_cnt = getattr(self, '_diag_obs_count', -1)
                if self._infer_count <= 10 or self._infer_count % 10 == 0:
                    print(f"[TIMING-INFER-BRIDGE] #{self._infer_count}: "
                          f"obs_build={(_t_obs_done-_t_obs)*1000:.2f}ms, "
                          f"pickle={(_t_pickle_done-_t_pickle)*1000:.2f}ms, "
                          f"grpc={(_t_grpc_done-_t_grpc)*1000:.2f}ms, "
                          f"parse={(_t_parse_done-_t_parse)*1000:.2f}ms, "
                          f"buf_update={(_t_buf_done-_t_buf)*1000:.2f}ms, "
                          f"t_inf_total={t_inf_actual*1000:.1f}ms, "
                          f"K_inf={K_inf_actual}, valid={len(valid_actions)}, "
                          f"buf={buf_size}, ema={self._t_inf_ema*1000:.1f}ms, "
                          f"obs_age={_obs_age:.1f}ms, obs_cnt={_obs_cnt}, "
                          f"T_inter={T_inter*1000:.0f}ms, feasible={feasible}")

                # === DEBUG: 打印 action chunk 和 clipping 信息 ===
                if True:
                    import numpy as _np
                    a0 = actions[0]
                    a0_deg = _np.degrees(a0.data[:7]) if a0.action_space == "joints" else a0.data[:6]
                    v0 = valid_actions[0]
                    v0_deg = _np.degrees(v0.data[:7]) if v0.action_space == "joints" else v0.data[:6]
                    a_last = actions[-1]
                    al_deg = _np.degrees(a_last.data[:7]) if a_last.action_space == "joints" else a_last.data[:6]
                    # gripper 链路诊断
                    _obs_grip_raw = getattr(self, '_diag_obs_gripper_raw', -1)
                    _obs_grip_clip = getattr(self, '_diag_obs_gripper_clipped', -1)
                    _grippers_all = [a.gripper for a in actions]
                    _grippers_valid = [a.gripper for a in valid_actions]
                    _diag_log(f"[DIAG-ACT] infer#{self._infer_count}: "
                          f"t_inf={t_inf_actual:.3f}s, K_inf={K_inf_actual}, "
                          f"chunk={len(actions)}, valid={len(valid_actions)}\n"
                          f"  action[0](deg):     {[f'{v:.2f}' for v in a0_deg]}\n"
                          f"  action[-1](deg):    {[f'{v:.2f}' for v in al_deg]}\n"
                          f"  valid[0](deg):      {[f'{v:.2f}' for v in v0_deg]}\n"
                          f"  action_space={a0.action_space}, action_type={a0.action_type}\n"
                          f"  obs_gripper: raw={_obs_grip_raw:.4f}, clipped={_obs_grip_clip:.4f}\n"
                          f"  gripper_all:   min={min(_grippers_all):.4f}, max={max(_grippers_all):.4f}, "
                          f"first5={[f'{g:.3f}' for g in _grippers_all[:5]]}\n"
                          f"  gripper_valid: min={min(_grippers_valid):.4f}, max={max(_grippers_valid):.4f}, "
                          f"first5={[f'{g:.3f}' for g in _grippers_valid[:5]]}")
                elif self._infer_count % 10 == 0:
                    _obs_grip_raw = getattr(self, '_diag_obs_gripper_raw', -1)
                    _grippers_valid = [a.gripper for a in valid_actions]
                    _diag_log(f"[DIAG-ACT-GRIP] infer#{self._infer_count}: "
                          f"obs_grip={_obs_grip_raw:.4f}, "
                          f"K_inf={K_inf_actual}, valid={len(valid_actions)}, "
                          f"grip_valid: min={min(_grippers_valid):.4f}, max={max(_grippers_valid):.4f}, "
                          f"first3={[f'{g:.3f}' for g in _grippers_valid[:3]]}")

                self._infer_count += 1
                self._t_inf_avg = (
                    self._t_inf_avg * (self._infer_count - 1) + t_inf_actual
                ) / self._infer_count
                self.timestep += 1

                if self._infer_count % 10 == 0:
                    print(f"[InferenceBridge] #{self._infer_count}: "
                          f"t_inf={t_inf_actual:.3f}s, K_inf={K_inf_actual}, "
                          f"chunk={len(actions)}, valid={len(valid_actions)}, "
                          f"buf={buf_size}")

            except Exception as e:
                print(f"[InferenceBridge] Error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.5)
                continue

            # 6. Timed wait + 强制 buffer drain
            # 无论 T_inter 是否已过期，都必须等待 buffer drain 到 threshold 以下
            # 再开始下一次推理。否则推理线程连续运行会饿死 exec loop 和 ROS2 executor。
            elapsed = time.monotonic() - t_start
            remaining = T_inter - elapsed
            _wait_start = time.monotonic()
            _wait_reason = "no_wait"
            T_step = 1.0 / self.config.f_exec
            poll_interval = min(T_step * 0.3, 0.01)

            # Phase A: timed wait（如果 T_inter 未过期）
            if remaining > 0:
                while remaining > 0 and not self.shutdown_event.is_set():
                    if not self._pause_event.is_set():
                        _wait_reason = "paused"
                        break
                    if self.ring_buffer.size <= threshold:
                        _wait_reason = f"buf_drain({self.ring_buffer.size}<={threshold})"
                        break
                    time.sleep(min(remaining, poll_interval))
                    remaining = T_inter - (time.monotonic() - t_start)
                if remaining <= 0:
                    _wait_reason = "T_inter_expired"

            # Phase B: 强制 buffer drain（即使 T_inter 已过期）
            # 防止推理线程在 buffer 满时立即开始下一次推理导致 GIL 饥饿
            # 无超时——必须等 buffer 消耗到 threshold 以下才开始下一次推理，
            # 否则 update() 会替换整个 buffer，丢弃大量未执行的 action，
            # 导致机械臂轨迹不连续，触发 Realman SDK 保护性回 home。
            if (self.ring_buffer.size > threshold
                    and _wait_reason != "paused"
                    and not self.shutdown_event.is_set()):
                _wait_reason_b = _wait_reason  # 保留 Phase A 的原因
                while not self.shutdown_event.is_set():
                    if not self._pause_event.is_set():
                        _wait_reason = "paused"
                        break
                    if self.ring_buffer.size <= threshold:
                        _wait_reason = f"forced_drain({self.ring_buffer.size}<={threshold})"
                        break
                    time.sleep(poll_interval)

            _wait_ms = (time.monotonic() - _wait_start) * 1000
            _total_cycle = (time.monotonic() - t_start) * 1000

            # DIAG: 推理周期总耗时（前 10 次 + 每 10 次）
            if self._infer_count <= 10 or self._infer_count % 10 == 0:
                _diag_log(f"[DIAG-CYCLE] #{self._infer_count - 1}: "
                      f"infer={elapsed*1000:.1f}ms, wait={_wait_ms:.1f}ms, "
                      f"total_cycle={_total_cycle:.1f}ms, "
                      f"reason={_wait_reason}, "
                      f"T_inter={T_inter*1000:.0f}ms, feasible={feasible}")

        print("[InferenceBridge] Inference thread stopped")

    def _grpc_infer(self, obs_bytes: bytes):
        """gRPC inference call."""
        try:
            obs_iterator = send_bytes_in_chunks(
                obs_bytes,
                services_pb2.Observation,
                log_prefix="[InferenceBridge] Observation",
                silent=True,
            )
            self.stub.SendObservations(obs_iterator, timeout=30.0)
            response = self.stub.GetActions(services_pb2.Empty(), timeout=30.0)
            return response
        except grpc.RpcError as e:
            if not self.shutdown_event.is_set():
                print(f"[InferenceBridge] gRPC error: {e}")
            return None

    def _parse_response(self, response) -> List[UnifiedAction]:
        """Parse gRPC response into UnifiedAction list."""
        if len(response.data) == 0:
            self._empty_response_count = getattr(self, '_empty_response_count', 0) + 1
            if self._empty_response_count <= 5 or self._empty_response_count % 50 == 0:
                _diag_log(f"[GRPC-EMPTY] #{self._empty_response_count}: "
                          f"Server 返回空响应，可能原因: 模型未加载/推理异常/观测反序列化失败")
            return []

        try:
            action_chunk = pickle.loads(response.data)
        except Exception as e:
            print(f"[InferenceBridge] pickle.loads() failed: {e}")
            return []

        # 检查是否是 Server 端错误信息
        if isinstance(action_chunk, dict) and "error" in action_chunk:
            _err = action_chunk["error"]
            _tb = action_chunk.get("traceback", "")
            _diag_log(f"[SERVER-ERROR] PolicyServer 推理异常: {_err}\n{_tb}")
            print(f"[InferenceBridge] SERVER ERROR: {_err}")
            return []

        if self.server_metadata is None and isinstance(action_chunk, ActionChunk):
            if action_chunk.metadata:
                self.server_metadata = action_chunk.metadata
                self.server_task = action_chunk.metadata.get("task", "")
                self.server_action_space = action_chunk.metadata.get("action_space", "")
                print(f"[InferenceBridge] Server config received:")
                print(f"  task: {self.server_task}")
                print(f"  action_space: {self.server_action_space}")

        if isinstance(action_chunk, ActionChunk):
            actions = action_chunk.actions
        elif isinstance(action_chunk, list):
            actions = action_chunk
        else:
            actions = self._convert_timed_actions(action_chunk)

        return actions

    def _convert_timed_actions(self, timed_actions) -> List[UnifiedAction]:
        """Convert legacy TimedAction format to UnifiedAction."""
        actions = []
        for ta in timed_actions:
            action_data = ta.get_action()
            if hasattr(action_data, 'numpy'):
                action_data = action_data.numpy()

            if len(action_data) >= 8:
                data = action_data[:7]
                gripper = action_data[7]
                action_space = "joints"
            else:
                data = action_data[:6]
                gripper = action_data[6] if len(action_data) > 6 else 0.0
                action_space = "pose"

            actions.append(UnifiedAction(
                timestamp=ta.get_timestamp(),
                timestep=ta.get_timestep(),
                action_space=action_space,
                action_type="absolute",
                data=np.array(data, dtype=np.float32),
                gripper=float(gripper),
            ))
        return actions
