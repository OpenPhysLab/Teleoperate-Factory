#!/usr/bin/env python3
"""Calibration capture: Quest hand pose, RM75 FK/IK diagnostics, and synchronized cameras.

This is the dataset-oriented entry point for the existing ROS-free Quest
teleoperation stack.  It records camera groups, Quest transforms/buttons and
the latest RM75 joint feedback into one JSONL record.  The robot control path
is the same API2/IK path used by ``rm_api2_teleop.py``; ``--no-robot`` and
``--dry-run`` are provided for safe Quest/camera bring-up first.

USB Quest:
  python3 src/oculus_reader/scripts/rm75_teleop_dataset.py --no-robot \
      --duration 30 --save-dir recordings/quest_test

Double-arm collection:
  python3 src/oculus_reader/scripts/rm75_teleop_dataset.py --mode double \
      --left-ip 192.168.1.18 --right-ip 192.168.1.19 \
      --duration 60 --save-dir recordings/session01

Quest lifecycle controls: right joystick click starts, left joystick click ends
and saves, both joystick clicks together discard and end.  B/Y hold robot
following, A/X recalibrate the neutral pose, and the index triggers control the
grippers.  Three camera preview windows are shown by default.
"""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
import json
import queue
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np

import rm_api2_teleop as teleop
from oculus_reader import OculusReader
from synchronized_cameras import (CameraWorker, RealSenseWorker,
                                  _camera_device_candidates, _drain,
                                  _load_config, _pop_synchronized)

DEFAULT_OUTPUT_ROOT = Path("/media/lyj/Data/GaussianRDT_Videos")


STOP = False


def _stop_handler(signum, frame):
    del signum, frame
    global STOP
    STOP = True


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _make_workers(capture, cameras, args):
    fps = float(args.camera_fps if args.camera_fps is not None
                else capture.get("fps", 25))
    width = int(capture.get("width", 640))
    height = int(capture.get("height", 480))
    queue_size = int(args.queue_size if args.queue_size is not None
                     else capture.get("queue_size", 5))
    backend = str(args.backend or capture.get("backend", "realsense")).lower()
    if backend == "auto":
        try:
            import pyrealsense2  # noqa: F401
            backend = "realsense"
        except ImportError:
            backend = "v4l2"
    if backend not in ("realsense", "v4l2"):
        raise ValueError("backend must be realsense, v4l2, or auto")
    names = [str(camera.get("name", "camera{}".format(index)))
             for index, camera in enumerate(cameras)]
    candidates = [_camera_device_candidates(camera) for camera in cameras]
    if backend == "realsense":
        workers = [RealSenseWorker(
            name, paths, camera.get("realsense_serial") or camera.get("sdk_serial"),
            fps, width, height, queue_size)
            for name, paths, camera in zip(names, candidates, cameras)]
    else:
        fourcc = str(capture.get("fourcc", "MJPG"))
        workers = [CameraWorker(name, paths, fps, width, height, queue_size,
                                backend, fourcc)
                   for name, paths in zip(names, candidates)]
    return backend, fps, names, workers, queue_size


def _start_workers(workers, timeout):
    for worker in workers:
        worker.start()
    deadline = time.monotonic() + max(0.1, timeout)
    while time.monotonic() < deadline:
        if all(worker.ready_event.is_set() for worker in workers):
            break
        time.sleep(0.01)
    failures = ["{}: {}".format(worker.camera_name, worker.error)
                for worker in workers if worker.error]
    if failures:
        raise RuntimeError("camera startup failed: {}".format("; ".join(failures)))
    if not all(worker.ready_event.is_set() for worker in workers):
        raise RuntimeError("camera startup timeout")


def _controller_setup(args, config):
    """Create controllers/IK/base poses using the tested teleop code."""
    if teleop._IK_IMPORT_ERROR is not None:
        raise RuntimeError("RM75 dataset control requires Pinocchio: {}".format(
            teleop._IK_IMPORT_ERROR))
    controllers = {}
    if args.mode == "single":
        controller = teleop.make_controller(config, args)
        controllers["right"] = controller
        ik = {"right": teleop.Arm_IK(params=config)}
        if not args.skip_init_pose:
            controller.init_pose(config.get("initial_gripper_width", 0.0))
        poses = {
            "right": {
                "base": list(config.get("base_pose", [0.19, 0.0, 0.2, 0, 0, 0])),
                "zero": list(config.get("right_teleop_zero_pose",
                                       config.get("teleop_zero_pose",
                                                  [0.0, 0.0, 0.8505, 0, 0, 0]))),
            }
        }
    else:
        right = teleop.make_controller(config, args, "right")
        left = teleop.make_controller(config, args, "left")
        controllers = {"right": right, "left": left}
        ik = {"right": teleop.Arm_IK("right", config),
              "left": teleop.Arm_IK("left", config)}
        if not args.skip_init_pose:
            right.init_pose(config.get("initial_gripper_width_right",
                                       config.get("initial_gripper_width", 0.0)))
            left.init_pose(config.get("initial_gripper_width_left",
                                      config.get("initial_gripper_width", 0.0)))
        default_base = [0.19, 0.0, 0.2, 0, 0, 0]
        default_zero = [0.0, 0.0, 0.8505, 0, 0, 0]
        poses = {
            "right": {"base": list(config.get("right_base_pose",
                                             config.get("base_pose", default_base))),
                       "zero": list(config.get("right_teleop_zero_pose",
                                             config.get("teleop_zero_pose", default_zero)))},
            "left": {"base": list(config.get("left_base_pose",
                                            config.get("base_pose", default_base))),
                      "zero": list(config.get("left_teleop_zero_pose",
                                            config.get("teleop_zero_pose", default_zero)))},
        }
    return controllers, ik, poses


def _control_once(args, config, reader, controllers, ik, poses, previous_reset,
                  allow_motion=True):
    transforms, buttons = reader.get_transformations_and_buttons()
    sample_timestamp_ns = time.monotonic_ns()
    transforms = dict(transforms or {})
    buttons = dict(buttons or {})
    # Lightweight diagnostics: make received Quest button edges visible even
    # while the dataset is paused (essential for wireless input bring-up).
    active_keys = tuple(k for k in ("A", "B", "X", "Y", "RJ", "LJ")
                        if bool(buttons.get(k, False)))
    if active_keys != getattr(reader, "_last_diag_buttons", ()):
        if active_keys:
            print("VR buttons: {}".format(",".join(active_keys)), flush=True)
        reader._last_diag_buttons = active_keys
    if controllers:
        for hand, transform_key, reset_key, enable_key, trigger_key in (
                ("right", "r", "A", "B", "rightTrig"),
                ("left", "l", "X", "Y", "leftTrig")):
            if hand not in controllers or transform_key not in transforms:
                continue
            reset = bool(buttons.get(reset_key, False))
            # A/X are also used as session controls on Quest builds without
            # RJ/LJ. Do not re-home on the same edge; the saved initial pose is
            # restored only at startup and shutdown.
            if reset and not previous_reset.get(hand, False) and not getattr(args, "lifecycle_ax", True):
                controllers[hand].init_pose(
                    config.get("initial_gripper_width_{}".format(hand),
                               config.get("initial_gripper_width", 0.0)))
                poses[hand]["base"] = teleop.matrix_to_xyzrpy(
                    teleop.adjustment_matrix(transforms[transform_key]))
            if allow_motion:
                # Digital toggle: click A+B (right) or X+Y (left) once to
                # close, click the same pair again to open.
                combo = bool(buttons.get(reset_key, False) and
                             buttons.get(enable_key, False))
                combo_prev = getattr(reader, "_combo_prev", {}).get(hand, False)
                grip_state = getattr(reader, "_grip_closed", {})
                if combo and not combo_prev:
                    grip_state[hand] = not grip_state.get(hand, False)
                reader._combo_prev = dict(getattr(reader, "_combo_prev", {}),
                                          **{hand: combo})
                reader._grip_closed = grip_state
                teleop.solve_and_send(
                    controllers[hand], ik[hand], transforms[transform_key],
                    poses[hand]["base"], poses[hand]["zero"],
                    1.0 if grip_state.get(hand, False) else 0.0,
                    bool(buttons.get(enable_key, False)),
                    not args.allow_collision)
                if not bool(buttons.get(enable_key, False)):
                    # Gripper toggle is independent of arm-follow clutch.
                    controllers[hand].command_gripper(
                        0.0 if grip_state.get(hand, False) else
                        controllers[hand].gripper_max_width)
            previous_reset[hand] = reset
    return transforms, buttons, sample_timestamp_ns


def _vr_action(buttons, previous_buttons):
    """Map lifecycle edges, with A/X fallback for APKs without RJ/LJ."""
    # Some Quest APK builds expose joystick clicks as RJ/LJ, while the current
    # Quest 3 build reliably exposes A/X. Accept both, with A/X as fallback.
    # X+Y/A+B are gripper toggles; do not interpret those combinations as
    # lifecycle save/start actions.  A or X alone remain the APK fallback.
    r_now = bool(buttons.get("RJ", False) or
                 (buttons.get("A", False) and not buttons.get("B", False)))
    l_now = bool(buttons.get("LJ", False) or
                 (buttons.get("X", False) and not buttons.get("Y", False)))
    r_old = bool(previous_buttons.get("_right_lifecycle", False))
    l_old = bool(previous_buttons.get("_left_lifecycle", False))
    previous_buttons["_right_lifecycle"] = r_now
    previous_buttons["_left_lifecycle"] = l_now
    if r_now and l_now and not (r_old and l_old):
        return "finish"
    if r_now and not r_old:
        return "start"
    if l_now and not l_old:
        return "episode_end"
    return None


def _show_views(group, names, enabled=True):
    if not enabled:
        return
    import cv2
    frames = []
    for name, frame in zip(names, group):
        image = frame.image
        # Keep one consistent display height so all three views fit in one
        # window even when camera profiles differ after a reboot.
        if image is None or image.size == 0:
            continue
        if image.shape[0] != 360:
            scale = 360.0 / image.shape[0]
            image = cv2.resize(image, (int(image.shape[1] * scale), 360))
        cv2.putText(image, name, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2, cv2.LINE_AA)
        frames.append(image)
    if frames:
        canvas = cv2.hconcat(frames)
        cv2.imshow("RealSense cameras (D435 | D405 | D405)", canvas)
    cv2.waitKey(1)


def _robot_record(controllers):
    result = {}
    for name, controller in controllers.items():
        result[name] = {"joints_rad": controller.current_joint_positions()}
    return result


def _start_command_reader(command_queue):
    """Read one-letter interactive commands without blocking acquisition."""
    def read_commands():
        while True:
            line = sys.stdin.readline()
            if not line:
                return
            command_queue.put(line.strip().lower())
    thread = threading.Thread(target=read_commands, name="dataset-console", daemon=True)
    thread.start()
    return thread


def _open_manifest(save_dir):
    return open(save_dir / "manifest.jsonl", "w", encoding="utf-8")


def _discard_session(save_dir, manifest):
    """Delete only files inside the explicitly selected session directory."""
    manifest.flush()
    manifest.close()
    for child in save_dir.iterdir():
        if child.is_file() or child.is_symlink():
            child.unlink()
        elif child.is_dir():
            # A session normally contains only images and manifest.  Refuse to
            # recurse into nested directories so a bad path cannot erase more.
            raise RuntimeError("refusing to delete nested directory {}".format(child))
    return _open_manifest(save_dir)


def main(argv=None):
    global STOP
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="src/oculus_reader/config/rm75.yaml")
    parser.add_argument("--camera-config", default="src/oculus_reader/config/cameras.yaml")
    parser.add_argument("--mode", choices=("single", "double"), default="double")
    parser.add_argument("--ip", default=None)
    parser.add_argument("--right-ip", default=None)
    parser.add_argument("--left-ip", default=None)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--quest-ip", default=None,
                        help="Quest 3 ADB IP; omit when connected by USB")
    parser.add_argument("--rate", type=float, default=30.0,
                        help="Quest/robot control loop rate")
    parser.add_argument("--camera-fps", type=float, default=None,
                        help="camera target/output rate; default from camera config")
    parser.add_argument("--backend", choices=("realsense", "v4l2", "auto"), default=None)
    parser.add_argument("--queue-size", type=int, default=None)
    parser.add_argument("--sync-window-ms", type=float, default=None,
                        help="camera matching window; default from camera config")
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--frames", type=int, default=0,
                        help="number of synchronized records; 0 means duration")
    parser.add_argument("--save-dir", default=None,
                        help="session output directory (default: /media/lyj/Data/GaussianRDT_Videos/session_TIMESTAMP)")
    parser.add_argument("--interactive", action="store_true",
                        help="legacy console controls; VR controls remain active")
    parser.add_argument("--wait-start", action="store_true",
                        help="with --interactive, wait for 's' before recording/motion")
    parser.add_argument("--no-robot", action="store_true",
                        help="record Quest + cameras without connecting/commanding RM75")
    parser.add_argument("--dry-run", action="store_true",
                        help="run IK and print commands without moving RM75")
    parser.add_argument("--skip-init-pose", action="store_true")
    parser.add_argument("--allow-collision", action="store_true")
    parser.add_argument("--no-display", action="store_true",
                        help="disable the three live camera windows")
    parser.add_argument("--no-save-images", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--save-images", action="store_true",
                        help="save camera JPEGs (default: do not save images)")
    args = parser.parse_args(argv)
    if args.rate <= 0 or args.frames < 0 or args.duration < 0:
        raise SystemExit("rate, frames and duration must be non-negative (rate > 0)")
    if args.frames == 0 and args.duration <= 0:
        raise SystemExit("set --frames or a positive --duration")

    camera_capture, camera_config = _load_config(args.camera_config)
    backend, camera_fps, names, workers, queue_size = _make_workers(
        camera_capture, camera_config, args)
    buffers = [deque(maxlen=max(2, queue_size * 2)) for _ in workers]
    save_dir = (Path(args.save_dir).expanduser() if args.save_dir else
                DEFAULT_OUTPUT_ROOT / ("session_" + datetime.now().strftime("%Y%m%d_%H%M%S")))
    save_dir.mkdir(parents=True, exist_ok=True)
    manifest = _open_manifest(save_dir)
    reader = None
    controllers = {}
    robot_config = {}
    try:
        _start_workers(workers, 5.0)
        print("camera backend: {}; target/output rate: {:.1f} Hz".format(
            backend, camera_fps))
        reader = OculusReader(ip_address=args.quest_ip)
        robot_config = teleop.load_config(args.config)
        if not args.no_robot:
            controllers, ik, poses = _controller_setup(args, robot_config)
        else:
            ik, poses = {}, {}
        print("Quest 3 connected; recording synchronized dataset to {}".format(save_dir))

        previous_reset = {"right": False, "left": False}
        latest_transforms, latest_buttons = {}, {}
        latest_quest_timestamp_ns = 0
        command_queue = queue.Queue()
        if args.interactive:
            _start_command_reader(command_queue)
            print("controls: s=start/resume, p=pause, d=discard session, q=finish")
        # Session lifecycle is controlled by Quest joystick buttons.  Keep
        # startup paused even when --wait-start is omitted.
        recording_active = False
        if args.wait_start and not args.interactive:
            raise RuntimeError("--wait-start requires --interactive")
        next_control = time.monotonic()
        last_wall = time.monotonic()
        active_elapsed = 0.0
        frame_id = 0
        rate_accumulator = 1.0
        last_group_timestamp_ns = None
        previous_lifecycle_buttons = {"RJ": False, "LJ": False}
        end_reason = None
        sync_window_ms = (float(args.sync_window_ms)
                          if args.sync_window_ms is not None
                          else float(camera_capture.get("sync_window_ms", 18)))
        window_ns = int(sync_window_ms * 1e6)
        while not STOP:
            now = time.monotonic()
            active_elapsed += max(0.0, now - last_wall) if recording_active else 0.0
            last_wall = now
            while args.interactive:
                try:
                    command = command_queue.get_nowait()
                except queue.Empty:
                    break
                if command in ("s", "start", "resume"):
                    recording_active = True
                    print("recording/resume")
                elif command in ("p", "pause"):
                    recording_active = False
                    print("paused: robot follow and file output stopped")
                elif command in ("d", "discard", "delete"):
                    recording_active = False
                    manifest = _discard_session(save_dir, manifest)
                    frame_id = 0
                    active_elapsed = 0.0
                    rate_accumulator = 1.0
                    last_group_timestamp_ns = None
                    print("current session discarded; press s to start again")
                elif command in ("q", "quit", "exit", "stop"):
                    STOP = True
                    break
            if args.frames and frame_id >= args.frames:
                break
            if not args.frames and args.duration and active_elapsed >= args.duration:
                break
            if now >= next_control:
                (latest_transforms, latest_buttons,
                 latest_quest_timestamp_ns) = _control_once(
                    args, robot_config, reader, controllers, ik, poses,
                    previous_reset, allow_motion=recording_active)
                action = _vr_action(latest_buttons, previous_lifecycle_buttons)
                if action == "start":
                    recording_active = True
                    # Capture the current Quest pose as the neutral reference
                    # before the first follow command. Without this, the
                    # absolute tracking pose is interpreted as a large motion.
                    for hand, key in (("right", "r"), ("left", "l")):
                        if hand in poses and key in latest_transforms:
                            poses[hand]["base"] = teleop.matrix_to_xyzrpy(
                                teleop.adjustment_matrix(latest_transforms[key]))
                    print("VR: recording started")
                elif action == "episode_end":
                    recording_active = False
                    for controller in controllers.values():
                        try:
                            controller.init_pose(robot_config.get("initial_gripper_width", 0.07))
                        except Exception as exc:
                            print("WARNING: episode reset failed: {}".format(exc))
                    print("VR: current episode ended; saved. Press A to start next episode")
                elif action == "finish":
                    recording_active = False
                    end_reason = "finish"
                    STOP = True
                    print("VR: collection finished")
                next_control += 1.0 / args.rate
                if next_control < now - 1.0 / args.rate:
                    next_control = now
            for worker, buffer in zip(workers, buffers):
                _drain(worker, buffer)
            group = _pop_synchronized(buffers, window_ns)
            if group is None:
                time.sleep(0.001)
                continue
            _show_views(group, names, enabled=not args.no_display)
            if not recording_active:
                continue
            group_timestamp_ns = max(frame.timestamp_ns for frame in group)
            if last_group_timestamp_ns is not None and camera_fps > 0:
                elapsed = max(0, group_timestamp_ns - last_group_timestamp_ns)
                rate_accumulator += elapsed * camera_fps / 1e9
            last_group_timestamp_ns = group_timestamp_ns
            if camera_fps > 0 and rate_accumulator < 1.0:
                continue
            if camera_fps > 0:
                rate_accumulator -= 1.0
            span_ns = group_timestamp_ns - min(frame.timestamp_ns for frame in group)
            record = {
                "frame_id": frame_id,
                "timestamp_ns": group_timestamp_ns,
                "host_timestamp_ns": max(frame.host_timestamp_ns for frame in group),
                "timestamps_ns": {name: frame.timestamp_ns
                                  for name, frame in zip(names, group)},
                "host_timestamps_ns": {name: frame.host_timestamp_ns
                                       for name, frame in zip(names, group)},
                "timestamp_domains": {name: frame.timestamp_domain
                                      for name, frame in zip(names, group)},
                "camera_serials": {worker.camera_name: worker.source_serial
                                   for worker in workers},
                "sync_span_ms": span_ns / 1e6,
                "quest_host_timestamp_ns": latest_quest_timestamp_ns,
                "quest_transforms": _jsonable(latest_transforms),
                "quest_buttons": _jsonable(latest_buttons),
                "calibration": {"ik_status": {}, "ik_residual": {}, "robot_fk_pose": {}},
                "images": {},
            }
            if args.save_images and not args.no_save_images:
                import cv2
                for name, frame in zip(names, group):
                    filename = "{:08d}_{}.jpg".format(frame_id, name)
                    if not cv2.imwrite(str(save_dir / filename), frame.image,
                                       [cv2.IMWRITE_JPEG_QUALITY, 95]):
                        raise IOError("failed to write {}".format(filename))
                    record["images"][name] = filename
            if controllers:
                robot_timestamp_ns = time.monotonic_ns()
                record["robot_state_host_timestamp_ns"] = robot_timestamp_ns
                record["robot"] = _jsonable(_robot_record(controllers))
                for hand, controller in controllers.items():
                    try:
                        q_now = controller.current_joint_positions(strict=True)
                        fk = ik[hand].forward_pose(q_now)
                        record["calibration"]["robot_fk_pose"][hand] = _jsonable(fk)
                        follow_key = "B" if hand == "right" else "Y"
                        record["calibration"]["ik_status"][hand] = (
                            "feedback_ok_follow" if bool(latest_buttons.get(follow_key, False))
                            else "feedback_ok_idle")
                    except Exception as exc:
                        record["calibration"]["ik_status"][hand] = "feedback_error: {}".format(exc)
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
            manifest.flush()
            if frame_id == 0 or frame_id % 25 == 0:
                print("record {:6d}: span={:.2f} ms".format(
                    frame_id, span_ns / 1e6))
            frame_id += 1
        print("saved synchronized records: {} ({})".format(
            frame_id, end_reason or "stopped"))
        return 0 if frame_id > 0 else 1
    finally:
        if reader is not None:
            reader.stop()
        for controller in controllers.values():
            try:
                controller.init_pose(robot_config.get(
                    "initial_gripper_width", 0.07))
            except Exception as exc:
                print("WARNING: failed to restore initial pose: {}".format(exc),
                      file=sys.stderr)
            controller.close()
        for worker in workers:
            worker.stop()
        for worker in workers:
            worker.join(timeout=2.0)
        manifest.close()
        if not args.no_display:
            try:
                import cv2
                cv2.destroyAllWindows()
            except Exception:
                pass


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _stop_handler)
    signal.signal(signal.SIGTERM, _stop_handler)
    sys.exit(main())
