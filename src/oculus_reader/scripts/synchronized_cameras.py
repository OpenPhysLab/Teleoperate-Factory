#!/usr/bin/env python3
"""ROS-free synchronized capture from the configured RealSense cameras.

The default RealSense backend uses comparable global hardware timestamps.  A
V4L2 backend remains available as a fallback and uses host arrival timestamps.
Each physical camera is read by its own thread, and a synchronized group is
formed only when all source timestamps are within the configured window.

Example:
  python3 src/oculus_reader/scripts/synchronized_cameras.py \
      --frames 100 --save-dir recordings/session01

Use ``--display`` for a live preview, or ``--frames 1`` as a quick smoke test.
The script writes ``manifest.jsonl`` when --save-dir is supplied.  Every line
contains one frame_id, the group timestamp, each source timestamp, and the
measured synchronization span.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
from itertools import product
import os
import queue
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Optional, Sequence, Tuple


@dataclass
class Frame:
    timestamp_ns: int
    image: object
    sequence: int
    host_timestamp_ns: int = 0
    timestamp_domain: str = "host_monotonic"


class CameraWorker(threading.Thread):
    """Read one V4L2 node continuously into a bounded queue."""

    def __init__(self, name: str, devices: Sequence[str], fps: float, width: int,
                 height: int, queue_size: int, backend: str = "v4l2",
                 fourcc: str = "MJPG"):
        super().__init__(name="camera-{}".format(name), daemon=True)
        self.camera_name = name
        self.devices = list(devices)
        self.device = self.devices[0] if self.devices else ""
        self.fps = fps
        self.width = width
        self.height = height
        self.backend = backend.lower()
        self.fourcc = fourcc.upper()
        self.frames: queue.Queue[Frame] = queue.Queue(maxsize=max(1, queue_size))
        self.stop_event = threading.Event()
        self.ready_event = threading.Event()
        self.error: Optional[str] = None
        self.read_count = 0
        self.drop_count = 0
        self._capture = None
        self.source_serial = "unknown"
        self.timestamp_domain = "host_monotonic"

    def _put_latest(self, frame: Frame) -> None:
        try:
            self.frames.put_nowait(frame)
            return
        except queue.Full:
            pass
        # Dropping the oldest frame bounds latency when the synchronizer is
        # slower than a camera.  A fresh frame is always preferred.
        try:
            self.frames.get_nowait()
            self.drop_count += 1
        except queue.Empty:
            return
        try:
            self.frames.put_nowait(frame)
        except queue.Full:
            self.drop_count += 1

    def run(self) -> None:  # pragma: no cover - exercised against hardware
        try:
            import cv2
        except ImportError as exc:
            self.error = "OpenCV is required: {}".format(exc)
            self.ready_event.set()
            return

        backend_id = cv2.CAP_V4L2 if self.backend in ("v4l2", "cv2") else cv2.CAP_ANY
        cap = None
        for device in self.devices:
            candidate = cv2.VideoCapture(device, backend_id)
            if candidate.isOpened():
                cap = candidate
                self.device = device
                break
            candidate.release()
        self._capture = cap
        if cap is None:
            self.error = "cannot open any of {}".format(
                ", ".join(self.devices) or "<none>")
            self.ready_event.set()
            return
        if len(self.fourcc) == 4:
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
            except Exception:
                pass
        if self.width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps > 0:
            cap.set(cv2.CAP_PROP_FPS, self.fps)
        # A small V4L2 buffer reduces latency.  Some drivers ignore this.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        self.ready_event.set()

        sequence = 0
        while not self.stop_event.is_set():
            ok, image = cap.read()
            if not ok or image is None:
                if self.stop_event.wait(0.01):
                    break
                continue
            # Timestamp immediately after read so the timestamp describes the
            # arrival of this frame at the host, independent of USB ordering.
            timestamp_ns = time.monotonic_ns()
            sequence += 1
            self.read_count += 1
            self._put_latest(Frame(timestamp_ns, image, sequence,
                                   timestamp_ns, "host_monotonic"))

        cap.release()
        self._capture = None

    def stop(self) -> None:
        self.stop_event.set()
        # release() is safe for OpenCV backends and helps unblock read() on
        # drivers that wait during shutdown.
        cap = self._capture
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass


def _video_interface_path(device: str) -> Optional[str]:
    """Return the USB video interface path used to identify an RS device."""
    node = os.path.basename(device)
    path = "/sys/class/video4linux/{}/device".format(node)
    try:
        return os.path.realpath(path)
    except OSError:
        return None


class RealSenseWorker(CameraWorker):
    """RealSense SDK worker using comparable global hardware timestamps."""

    def __init__(self, name: str, devices: Sequence[str], serial: Optional[str],
                 fps: float, width: int, height: int, queue_size: int):
        super().__init__(name, devices, fps, width, height, queue_size,
                         backend="realsense")
        self.requested_serial = str(serial) if serial else None
        self._pipeline = None

    def _select_device(self, rs, context):
        target_interfaces = {
            _video_interface_path(device) for device in self.devices
            if _video_interface_path(device)
        }
        for device in context.query_devices():
            serial = device.get_info(rs.camera_info.serial_number)
            if self.requested_serial and serial == self.requested_serial:
                return device
            try:
                physical_port = device.get_info(rs.camera_info.physical_port)
                interface = physical_port.rsplit("/video4linux", 1)[0]
            except Exception:
                interface = None
            if interface in target_interfaces:
                return device
        return None

    def run(self) -> None:  # pragma: no cover - exercised against hardware
        try:
            import numpy as np
            import pyrealsense2 as rs
        except ImportError as exc:
            self.error = "pyrealsense2 and numpy are required: {}".format(exc)
            self.ready_event.set()
            return
        try:
            context = rs.context()
            device = self._select_device(rs, context)
            if device is None:
                self.error = "cannot map {} to a RealSense device".format(
                    ", ".join(self.devices) or "<none>")
                self.ready_event.set()
                return
            self.source_serial = device.get_info(rs.camera_info.serial_number)
            config = rs.config()
            config.enable_device(self.source_serial)
            # The current D435/D405 color profiles expose 30 Hz at this
            # resolution; 25 Hz is produced by the timestamp-based output
            # decimator below rather than requesting an unsupported profile.
            stream_fps = 30 if self.fps <= 30 else int(round(self.fps))
            config.enable_stream(rs.stream.color, self.width, self.height,
                                 rs.format.bgr8, stream_fps)
            pipeline = rs.pipeline(context)
            pipeline.start(config)
            self._pipeline = pipeline
            self.ready_event.set()
            sequence = 0
            while not self.stop_event.is_set():
                frames = pipeline.wait_for_frames(1000)
                color = frames.get_color_frame()
                if not color:
                    continue
                host_timestamp_ns = time.monotonic_ns()
                hardware_timestamp_ns = int(round(color.get_timestamp() * 1e6))
                domain = str(color.get_frame_timestamp_domain()).split(".")[-1]
                # Global-time timestamps are comparable between the cameras.
                # If firmware exposes only a device-local clock, fall back to
                # host arrival time rather than falsely claiming synchronization.
                timestamp_ns = (hardware_timestamp_ns if domain == "global_time"
                                else host_timestamp_ns)
                sequence += 1
                self.read_count += 1
                self.timestamp_domain = domain
                self._put_latest(Frame(timestamp_ns,
                                       np.asanyarray(color.get_data()).copy(),
                                       sequence, host_timestamp_ns, domain))
        except Exception as exc:
            self.error = "RealSense {}: {}".format(self.device, exc)
            self.ready_event.set()
        finally:
            pipeline = self._pipeline
            self._pipeline = None
            if pipeline is not None:
                try:
                    pipeline.stop()
                except Exception:
                    pass

    def stop(self) -> None:
        self.stop_event.set()
        pipeline = self._pipeline
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                pass


def _load_config(path: str) -> Tuple[dict, List[dict]]:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required: {}".format(exc))
    with open(path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    capture = config.get("capture", {}) or {}
    cameras = config.get("cameras", []) or []
    if not cameras:
        raise ValueError("no cameras found in {}".format(path))
    return capture, cameras


def _camera_device_candidates(camera: dict) -> List[str]:
    """Return preferred capture node followed by configured fallbacks."""
    configured = []
    if camera.get("capture_device"):
        configured.append(camera["capture_device"])
    configured.extend(camera.get("preferred_nodes") or [])
    configured.extend(camera.get("video_devices") or [])
    candidates = []
    for device in configured:
        device = str(device)
        if device and device not in candidates:
            candidates.append(device)
    if not candidates:
        raise ValueError("camera {} has no capture_device or nodes".format(camera.get("name")))
    return candidates


def _pop_synchronized(buffers: Sequence[Deque[Frame]], window_ns: int):
    """Return the lowest-span group currently present in all buffers.

    Looking only at queue heads is simple but can pair a frame with a nearby
    frame that is not its best counterpart.  The queues are deliberately
    small, so searching their bounded Cartesian product is inexpensive and
    gives the lowest possible timestamp span without reusing a frame.
    """
    if any(not buffer for buffer in buffers):
        return None
    choices = [tuple(buffer) for buffer in buffers]
    best = min(product(*choices), key=lambda group: (
        max(frame.timestamp_ns for frame in group)
        - min(frame.timestamp_ns for frame in group),
        -max(frame.timestamp_ns for frame in group)))
    best_span = max(frame.timestamp_ns for frame in best) - min(
        frame.timestamp_ns for frame in best)
    if best_span <= window_ns:
        # Remove stale frames before each selected frame, then remove the
        # selected frame itself.  Identity comparison avoids NumPy array
        # equality when Frame.image is a cv2 image.
        for buffer, selected in zip(buffers, best):
            while buffer and buffer[0] is not selected:
                buffer.popleft()
            if buffer:
                buffer.popleft()
        return list(best)

    # No combination fits the window.  The globally oldest head cannot be in
    # a future valid group, so discard it and wait for the next iteration.
    oldest = min(buffer[0].timestamp_ns for buffer in buffers)
    for buffer in buffers:
        if buffer and buffer[0].timestamp_ns == oldest:
            buffer.popleft()
    return None


def _drain(worker: CameraWorker, buffer: Deque[Frame]) -> None:
    while True:
        try:
            buffer.append(worker.frames.get_nowait())
        except queue.Empty:
            return


def _save_group(save_dir: Path, frame_id: int, group: Sequence[Frame],
                names: Sequence[str], span_ns: int, jpeg_quality: int) -> dict:
    import cv2

    images = {}
    for name, frame in zip(names, group):
        filename = "{:08d}_{}.jpg".format(frame_id, name)
        path = save_dir / filename
        if not cv2.imwrite(str(path), frame.image,
                           [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]):
            raise IOError("failed to write {}".format(path))
        images[name] = filename
    return {
        "frame_id": frame_id,
        "timestamp_ns": max(frame.timestamp_ns for frame in group),
        "timestamps_ns": {name: frame.timestamp_ns for name, frame in zip(names, group)},
        "host_timestamps_ns": {name: frame.host_timestamp_ns
                               for name, frame in zip(names, group)},
        "timestamp_domains": {name: frame.timestamp_domain
                              for name, frame in zip(names, group)},
        "sequences": {name: frame.sequence for name, frame in zip(names, group)},
        "sync_span_ms": span_ns / 1e6,
        "images": images,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="src/oculus_reader/config/cameras.yaml")
    parser.add_argument("--frames", type=int, default=0,
                        help="number of synchronized groups (0 means use --duration)")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="seconds to run when --frames is 0")
    parser.add_argument("--fps", type=float, default=None,
                        help="capture rate override; default from config")
    parser.add_argument("--backend", choices=("realsense", "v4l2", "auto"),
                        default=None, help="timestamp/capture backend override")
    parser.add_argument("--sync-window-ms", type=float, default=None,
                        help="maximum timestamp span per group; default from config")
    parser.add_argument("--queue-size", type=int, default=None,
                        help="per-camera queue size; default from config")
    parser.add_argument("--save-dir", default="",
                        help="write JPEGs and manifest.jsonl to this directory")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--display", action="store_true",
                        help="show a concatenated preview; press q or ESC to stop")
    parser.add_argument("--startup-timeout", type=float, default=3.0)
    parser.add_argument("--poll-ms", type=float, default=2.0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        capture, camera_config = _load_config(args.config)
        fps = (float(args.fps) if args.fps is not None
               else float(capture.get("fps", 15)))
        width = int(capture.get("width", 640))
        height = int(capture.get("height", 480))
        window_ms = (float(args.sync_window_ms) if args.sync_window_ms is not None
                     else float(capture.get("sync_window_ms", 20)))
        queue_size = (int(args.queue_size) if args.queue_size is not None
                      else int(capture.get("queue_size", 5)))
        backend = str(args.backend or capture.get("backend", "realsense")).lower()
        fourcc = str(capture.get("fourcc", "MJPG"))
        names = [str(camera.get("name", "camera{}".format(i)))
                 for i, camera in enumerate(camera_config)]
        device_candidates = [_camera_device_candidates(camera) for camera in camera_config]
    except (OSError, ValueError, TypeError) as exc:
        print("configuration error: {}".format(exc), file=sys.stderr)
        return 2
    if len(set(names)) != len(names):
        print("configuration error: camera names must be unique", file=sys.stderr)
        return 2
    if backend == "auto":
        try:
            import pyrealsense2  # noqa: F401
            backend = "realsense"
        except ImportError:
            backend = "v4l2"
    if backend not in ("realsense", "v4l2"):
        print("backend must be realsense, v4l2, or auto", file=sys.stderr)
        return 2
    if window_ms <= 0 or queue_size <= 0:
        print("sync window and queue size must be positive", file=sys.stderr)
        return 2

    save_dir = Path(args.save_dir).expanduser() if args.save_dir else None
    manifest = None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        manifest = open(save_dir / "manifest.jsonl", "w", encoding="utf-8")

    if backend == "realsense":
        workers = [RealSenseWorker(
            name, candidates,
            camera.get("realsense_serial") or camera.get("sdk_serial"),
            fps, width, height, queue_size)
            for name, candidates, camera in zip(names, device_candidates, camera_config)]
    else:
        workers = [CameraWorker(name, candidates, fps, width, height, queue_size,
                                backend, fourcc)
                   for name, candidates in zip(names, device_candidates)]
    buffers: List[Deque[Frame]] = [deque(maxlen=max(2, queue_size * 2)) for _ in workers]
    stop_event = threading.Event()

    def request_stop(*_args):
        stop_event.set()

    old_sigint = signal.signal(signal.SIGINT, request_stop)
    old_sigterm = signal.signal(signal.SIGTERM, request_stop)
    try:
        for worker in workers:
            worker.start()
        deadline = time.monotonic() + max(0.1, args.startup_timeout)
        while time.monotonic() < deadline and not stop_event.is_set():
            if all(worker.ready_event.is_set() for worker in workers):
                break
            time.sleep(0.01)
        failed = ["{} ({}): {}".format(w.camera_name, w.device, w.error)
                  for w in workers if w.error]
        if failed:
            print("camera startup failed:", file=sys.stderr)
            for item in failed:
                print("  {}".format(item), file=sys.stderr)
            return 1
        if not all(worker.ready_event.is_set() for worker in workers):
            print("camera startup timeout", file=sys.stderr)
            return 1
        print("backend: {} (timestamp source: {})".format(
            backend, "RealSense global_time" if backend == "realsense"
            else "host monotonic clock"))
        print("capturing {} cameras: {}".format(len(workers), ", ".join(
            "{}={}".format(worker.camera_name, worker.device)
            for worker in workers)))
        print("sync window: {:.1f} ms; target fps: {:.1f}".format(window_ms, fps))

        import cv2
        start = time.monotonic()
        frame_id = 0
        matched = 0
        rate_dropped = 0
        last_group_timestamp_ns = None
        rate_accumulator = 1.0
        window_ns = int(window_ms * 1e6)
        while not stop_event.is_set():
            if args.frames > 0 and matched >= args.frames:
                break
            if args.frames <= 0 and time.monotonic() - start >= max(0.0, args.duration):
                break
            for worker, buffer in zip(workers, buffers):
                _drain(worker, buffer)
            group = _pop_synchronized(buffers, window_ns)
            if group is None:
                if args.display:
                    key = cv2.waitKey(1) & 0xff
                    if key in (27, ord("q")):
                        break
                time.sleep(max(0.0001, args.poll_ms / 1000.0))
                continue
            span_ns = max(frame.timestamp_ns for frame in group) - min(
                frame.timestamp_ns for frame in group)
            group_timestamp_ns = max(frame.timestamp_ns for frame in group)
            if last_group_timestamp_ns is not None and fps > 0:
                elapsed_ns = max(0, group_timestamp_ns - last_group_timestamp_ns)
                rate_accumulator += elapsed_ns * fps / 1e9
            last_group_timestamp_ns = group_timestamp_ns
            if fps > 0 and rate_accumulator < 1.0:
                rate_dropped += 1
                continue
            if fps > 0:
                rate_accumulator -= 1.0
            matched += 1
            record = {"frame_id": frame_id,
                      "timestamp_ns": group_timestamp_ns,
                      "timestamps_ns": {name: frame.timestamp_ns for name, frame in zip(names, group)},
                      "host_timestamps_ns": {name: frame.host_timestamp_ns
                                             for name, frame in zip(names, group)},
                      "timestamp_domains": {name: frame.timestamp_domain
                                            for name, frame in zip(names, group)},
                      "sequences": {name: frame.sequence for name, frame in zip(names, group)},
                      "sync_span_ms": span_ns / 1e6}
            if save_dir:
                saved = _save_group(save_dir, frame_id, group, names, span_ns,
                                    max(1, min(100, args.jpeg_quality)))
                record["images"] = saved["images"]
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                manifest.flush()
            if args.display:
                preview = cv2.hconcat([frame.image for frame in group])
                cv2.imshow("synchronized cameras", preview)
                key = cv2.waitKey(1) & 0xff
                if key in (27, ord("q")):
                    break
            if matched == 1 or matched % 30 == 0:
                print("group {:6d}: span={:.2f} ms".format(frame_id, span_ns / 1e6))
            frame_id += 1
        print("matched synchronized groups: {}".format(matched))
        print("rate-limited groups: {}".format(rate_dropped))
        print("camera reads: {}".format(", ".join(
            "{}={}".format(worker.camera_name, worker.read_count) for worker in workers)))
        print("source serials: {}".format(", ".join(
            "{}={}".format(worker.camera_name, worker.source_serial)
            for worker in workers)))
        print("camera queue drops: {}".format(", ".join(
            "{}={}".format(worker.camera_name, worker.drop_count) for worker in workers)))
        return 0 if matched > 0 else 1
    finally:
        stop_event.set()
        for worker in workers:
            worker.stop()
        for worker in workers:
            worker.join(timeout=2.0)
        if manifest:
            manifest.close()
        if args.display:
            try:
                import cv2
                cv2.destroyAllWindows()
            except Exception:
                pass
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)


if __name__ == "__main__":
    sys.exit(main())
