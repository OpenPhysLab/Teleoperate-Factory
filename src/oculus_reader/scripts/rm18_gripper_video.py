#!/usr/bin/env python3
"""Quest A toggles RGB video + overhead depth recording; B toggles RM18 gripper."""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
import json
from pathlib import Path
import signal
import time

from oculus_reader import OculusReader
from synchronized_cameras import (
    CameraWorker, RealSenseWorker, _camera_device_candidates, _drain,
    _load_config, _pop_synchronized, _save_depth_frames,
)

ARM_IP = "192.168.1.18"


class Gripper:
    """Only gripper commands are exposed; connecting never moves the robot."""

    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        self.sdk = None
        self.closed = False
        self.state = None
        if not dry_run:
            self.connect()

    def connect(self):
        from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e
        self.sdk = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        try:
            handle = self.sdk.rm_create_robot_arm(ARM_IP, 8080, 3)
            if handle is None or handle.id <= 0:
                raise RuntimeError("cannot connect to RM18")
            self.state = self.read_state()
            mode, position = self.state.get("mode"), self.state.get("actpos")
            self.closed = mode in (2, 4, 6) or (
                mode not in (1, 5) and isinstance(position, int) and position < 500)
        except Exception:
            self.close()
            raise

    def read_state(self):
        if self.dry_run:
            self.state = {"actpos": 1 if self.closed else 1000, "dry_run": True}
        elif self.sdk is not None:
            status, state = self.sdk.rm_get_gripper_state()
            if status != 0:
                raise RuntimeError("gripper feedback status {}".format(status))
            if state.get("error", 0) or state.get("status") != 1 or state.get("enable_state") != 1:
                raise RuntimeError("gripper is not ready: {}".format(state))
            self.state = state
        return self.state

    def toggle(self):
        if not self.dry_run and self.sdk is None:
            self.connect()
        target = not self.closed
        try:
            result = 0 if self.dry_run else self.sdk.rm_set_gripper_position(
                1 if target else 1000, False, 1)
            if result != 0:
                raise RuntimeError("gripper command status {}".format(result))
        except Exception:
            self.close()
            raise
        self.closed = target
        return {"closed": target, "position": 1 if target else 1000,
                "accepted": True, "dry_run": self.dry_run}

    def close(self):
        if self.sdk is not None:
            try:
                self.sdk.rm_delete_robot_arm()
            finally:
                self.sdk = None


class ButtonEdges:
    def __init__(self):
        self.previous = None

    def update(self, buttons):
        if not buttons:
            return set()
        current = {key: bool(buttons.get(key, False)) for key in ("A", "B")}
        # An already-held button at startup is consumed until released.
        edges = set() if self.previous is None else {
            key for key in current if current[key] and not self.previous[key]}
        self.previous = current
        return edges


class Recording:
    def __init__(self, session, fps):
        self.session, self.fps = Path(session), fps
        self.index = 0
        self.active = False
        self.folder = None
        self.manifest = None
        self.writers = {}
        self.shapes = {}
        self.frame_id = 0

    def start(self):
        self.index += 1
        self.folder = self.session / "episode_{:04d}".format(self.index)
        self.folder.mkdir(exist_ok=False)
        self.manifest = open(self.folder / "manifest.jsonl", "x", encoding="utf-8")
        self.writers, self.shapes, self.frame_id = {}, {}, 0
        self.active = True
        print("RECORDING_START " + str(self.folder), flush=True)

    def write(self, group, names, gripper, buttons):
        import cv2
        if not self.active:
            return
        # Check paired depth and save it before writing this group's RGB frames.
        depths = _save_depth_frames(self.folder, self.frame_id, group, names)
        videos = {}
        for name, frame in zip(names, group):
            shape = frame.image.shape[:2]
            if name not in self.writers:
                height, width = shape
                path = self.folder / (name + ".mp4")
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                         self.fps, (width, height))
                if not writer.isOpened():
                    writer.release()
                    raise RuntimeError("cannot open video writer " + str(path))
                self.writers[name], self.shapes[name] = writer, shape
            if self.shapes[name] != shape:
                raise RuntimeError("camera resolution changed for " + name)
            self.writers[name].write(frame.image)
            videos[name] = {"path": name + ".mp4", "frame_index": self.frame_id}
        stamps = [frame.timestamp_ns for frame in group]
        record = {
            "frame_id": self.frame_id, "timestamp_ns": max(stamps),
            "host_timestamp_ns": time.monotonic_ns(),
            "timestamps_ns": {n: f.timestamp_ns for n, f in zip(names, group)},
            "host_timestamps_ns": {n: f.host_timestamp_ns for n, f in zip(names, group)},
            "timestamp_domains": {n: f.timestamp_domain for n, f in zip(names, group)},
            "sequences": {n: f.sequence for n, f in zip(names, group)},
            "sync_span_ms": (max(stamps) - min(stamps)) / 1e6,
            "videos": videos, "depth": depths,
            "gripper_ip": ARM_IP, "gripper_commanded_closed": gripper.closed,
            "gripper_feedback": gripper.state,
            "buttons": {key: bool(buttons.get(key, False)) for key in ("A", "B")},
        }
        self.manifest.write(json.dumps(record) + "\n")
        self.manifest.flush()
        self.frame_id += 1

    def stop(self):
        if not self.active:
            return
        self.active = False
        try:
            for writer in self.writers.values():
                writer.release()
        finally:
            self.writers = {}
            if self.manifest is not None:
                self.manifest.close()
                self.manifest = None
        (self.folder / "summary.json").write_text(json.dumps({
            "frames": self.frame_id, "video_fps": self.fps,
            "gripper_ip": ARM_IP, "completed": True}, indent=2))
        print("RECORDING_SAVED {} frames={}".format(self.folder, self.frame_id), flush=True)


def make_workers(config_path, no_depth):
    capture, cameras = _load_config(config_path)
    fps = float(capture.get("fps", 25))
    if fps <= 0:
        raise ValueError("camera fps must be positive")
    width, height = int(capture.get("width", 640)), int(capture.get("height", 480))
    queue_size = int(capture.get("queue_size", 5))
    backend = capture.get("backend", "realsense")
    if backend not in ("realsense", "v4l2"):
        raise ValueError("use realsense or v4l2 backend")
    if backend != "realsense" and not no_depth and any(c.get("record_depth") for c in cameras):
        raise ValueError("depth requires realsense; use --no-depth for V4L2")
    names, workers = [], []
    for camera in cameras:
        name = str(camera["name"])
        if not name or name in names or Path(name).name != name or name in (".", ".."):
            raise ValueError("camera names must be unique safe filenames")
        names.append(name)
        devices = _camera_device_candidates(camera)
        if backend == "realsense":
            worker = RealSenseWorker(name, devices,
                camera.get("realsense_serial") or camera.get("sdk_serial"),
                fps, width, height, queue_size,
                record_depth=bool(camera.get("record_depth", False) and not no_depth))
        else:
            worker = CameraWorker(name, devices, fps, width, height, queue_size,
                                  backend, capture.get("fourcc", "MJPG"))
        workers.append(worker)
    return fps, float(capture.get("sync_window_ms", 18)) * 1e6, names, workers


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quest-ip", default=None, help="omit for workstation USB")
    parser.add_argument("--camera-config", default=str(
        Path(__file__).resolve().parents[1] / "config" / "cameras.yaml"))
    parser.add_argument("--save-dir", default=None, help="new session directory")
    parser.add_argument("--no-depth", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="simulate gripper commands")
    parser.add_argument("--check", action="store_true", help="only read RM18 gripper state and exit")
    parser.add_argument("--duration", type=float, default=0, help="runtime seconds; 0 runs until Ctrl+C")
    args = parser.parse_args(argv)
    if args.duration < 0:
        parser.error("--duration must be non-negative")
    if args.check:
        gripper = Gripper(args.dry_run)
        try:
            print("GRIPPER_CHECK", ARM_IP, gripper.read_state(), flush=True)
        finally:
            gripper.close()
        return 0

    fps, window_ns, names, workers = make_workers(args.camera_config, args.no_depth)
    data_root = Path("/media/lyj/Data")
    session = Path(args.save_dir).expanduser() if args.save_dir else (
        data_root / "GaussianRDT_Videos" / ("gripper18_" + datetime.now().strftime("%Y%m%d_%H%M%S")))
    if session.resolve().is_relative_to(data_root) and not data_root.is_mount():
        raise RuntimeError("Data disk is not mounted; refusing to write onto its mount directory")
    session.mkdir(parents=True, exist_ok=False)
    recording = Recording(session, fps)
    buffers = [deque(maxlen=5) for _ in workers]
    stopped = False
    def stop(*_):
        nonlocal stopped
        stopped = True
    old_handlers = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
    reader, gripper, events = None, None, None
    try:
        events = open(session / "events.jsonl", "x", encoding="utf-8")
        def event(kind, **fields):
            events.write(json.dumps(dict(
                event=kind, host_timestamp_ns=time.monotonic_ns(), **fields)) + "\n")
            events.flush()
        reader = OculusReader(ip_address=args.quest_ip)
        gripper = Gripper(args.dry_run)
        for worker in workers:
            worker.start()
        deadline = time.monotonic() + 10
        while not all(w.ready_event.is_set() for w in workers):
            if stopped or time.monotonic() > deadline:
                raise RuntimeError("camera startup interrupted or timed out")
            time.sleep(0.01)
        failures = [w.error for w in workers if w.error]
        if failures:
            raise RuntimeError("camera startup failed: " + "; ".join(failures))
        print("READY: A=start/stop recording; B=toggle RM18 gripper; Ctrl+C=save and quit",
              flush=True)
        print("SESSION " + str(session), flush=True)
        event("ready", arm_ip=ARM_IP, initial_gripper=gripper.state, dry_run=args.dry_run)
        edges, last_group, accumulator = ButtonEdges(), None, 1.0
        start, next_feedback, next_poll = time.monotonic(), 0.0, time.monotonic()
        latest_buttons = {}
        while not stopped and (args.duration == 0 or time.monotonic() - start < args.duration):
            now = time.monotonic()
            if now >= next_poll:
                _, latest_buttons = reader.get_transformations_and_buttons()
                latest_buttons = latest_buttons or {}
                pressed = edges.update(latest_buttons)
                if "B" in pressed:
                    try:
                        command = gripper.toggle()
                        event("gripper", **command)
                        print("GRIPPER " + ("CLOSED" if gripper.closed else "OPEN"), flush=True)
                    except Exception as exc:
                        event("gripper_error", message=str(exc))
                        print("GRIPPER_ERROR " + str(exc), flush=True)
                if "A" in pressed:
                    if recording.active:
                        recording.stop()
                        event("recording_stop", episode=recording.index)
                    else:
                        recording.start()
                        for worker, buffer in zip(workers, buffers):
                            _drain(worker, buffer)
                            buffer.clear()
                        last_group, accumulator = None, 1.0
                        event("recording_start", episode=recording.index)
                next_poll = now + 1 / 30
            if now >= next_feedback:
                try:
                    gripper.read_state()
                except Exception as exc:
                    event("feedback_error", message=str(exc))
                    gripper.close()
                next_feedback = now + 0.5
            for worker, buffer in zip(workers, buffers):
                if worker.error:
                    raise RuntimeError("camera failed: " + worker.error)
                _drain(worker, buffer)
            group = _pop_synchronized(buffers, int(window_ns))
            if group is not None:
                stamp = max(f.timestamp_ns for f in group)
                if last_group is not None:
                    accumulator += max(0, stamp - last_group) * fps / 1e9
                last_group = stamp
                if accumulator >= 1:
                    accumulator = min(accumulator - 1, 1)
                    recording.write(group, names, gripper, latest_buttons)
                    if not args.no_display:
                        import cv2
                        preview = cv2.hconcat([f.image for f in group])
                        text = "REC" if recording.active else "IDLE"
                        cv2.putText(preview, text + " | A: video  B: gripper18",
                                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        cv2.namedWindow("RM18 gripper recorder", cv2.WINDOW_NORMAL)
                        cv2.imshow("RM18 gripper recorder", preview)
            if not args.no_display:
                import cv2
                if cv2.waitKey(1) & 0xff in (27, ord("q")):
                    stopped = True
            time.sleep(0.002)
        return 0
    finally:
        try:
            recording.stop()
        finally:
            if reader is not None:
                reader.stop()
            if gripper is not None:
                gripper.close()
            for worker in workers:
                worker.stop()
            for worker in workers:
                if worker.ident is not None:
                    worker.join(timeout=2)
            if events is not None:
                events.close()
            if not args.no_display:
                import cv2
                cv2.destroyAllWindows()
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
