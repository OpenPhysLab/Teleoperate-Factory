"""Optional live-camera video recording, independent of the replay control loop."""

import json
import math
import queue
import threading
import time
from pathlib import Path

from synchronized_cameras import (CameraWorker, RealSenseWorker,
                                  _camera_device_candidates, _load_config)


class ReplayVideoRecorder:
    """Write one MP4 per camera at wall-clock speed, with source timestamps in JSONL.

    Views use the latest available frames, not hardware-synchronized exposure.
    Repeated frames retain their source timestamps for identifying duplicates.
    """

    def __init__(self, output_dir, camera_config):
        capture, cameras = _load_config(str(camera_config))
        self.fps = float(capture.get("fps", 25))
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("camera fps must be positive and finite")
        width, height = int(capture.get("width", 640)), int(capture.get("height", 480))
        backend = capture.get("backend", "realsense")
        if backend not in ("realsense", "v4l2"):
            raise ValueError("replay video backend must be realsense or v4l2")
        self.workers = []
        for camera in cameras:
            name = str(camera["name"])
            devices = _camera_device_candidates(camera)
            if backend == "realsense":
                worker = RealSenseWorker(
                    name, devices, camera.get("realsense_serial") or camera.get("sdk_serial"),
                    self.fps, width, height, 2)
            else:
                worker = CameraWorker(name, devices, self.fps, width, height, 2,
                                      backend, capture.get("fourcc", "MJPG"))
            self.workers.append(worker)
        self.output_dir = Path(output_dir)
        # Never overwrite an earlier replay or a dataset directory.
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.stop_event = threading.Event()
        self.ready = threading.Event()
        self.error = None
        self.replay_row = None
        self.thread = threading.Thread(target=self._run, name="replay-video", daemon=True)

    def start(self):
        self.thread.start()
        if not self.ready.wait(15):
            raise RuntimeError("replay video startup timed out")
        self.check()
        print("Recording live replay videos in: {}".format(self.output_dir))

    def check(self):
        if self.error is not None:
            raise RuntimeError("replay video failed: {}".format(self.error))

    def close(self):
        self.stop_event.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=5)
            if self.thread.is_alive():
                raise RuntimeError("video recorder did not stop; video finalization unconfirmed")
        self.check()

    def _run(self):
        writers = {}
        started = []
        cv2 = None
        try:
            import cv2
            for worker in self.workers:
                worker.start()
                started.append(worker)
            latest = [None] * len(self.workers)
            next_tick = None
            frame_id = 0
            with (self.output_dir / "video_timestamps.jsonl").open("x") as timeline:
                while not self.stop_event.is_set():
                    for i, worker in enumerate(self.workers):
                        if worker.error:
                            raise RuntimeError("{}: {}".format(worker.camera_name, worker.error))
                        while True:
                            try:
                                latest[i] = worker.frames.get_nowait()
                            except queue.Empty:
                                break
                    if any(frame is None for frame in latest):
                        self.stop_event.wait(0.005)
                        continue
                    now_ns = time.monotonic_ns()
                    if any(now_ns - frame.host_timestamp_ns > 2_000_000_000 for frame in latest):
                        raise RuntimeError("camera frame stale for more than 2 seconds")
                    for worker, frame in zip(self.workers, latest):
                        h, w = frame.image.shape[:2]
                        view = cv2.resize(frame.image,
                                          (max(2, int(w * 360 / h) // 2 * 2), 360))
                        if worker.camera_name not in writers:
                            path = self.output_dir / (worker.camera_name + ".mp4")
                            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                                     self.fps, (view.shape[1], view.shape[0]))
                            if not writer.isOpened():
                                raise RuntimeError("cannot open MP4 writer for {}".format(worker.camera_name))
                            writers[worker.camera_name] = writer
                        writers[worker.camera_name].write(view)
                        # Preview is a single window; files remain one per
                        # camera.  waitKey also lets the window repaint while
                        # the replay loop is running.
                        if worker is self.workers[-1]:
                            preview = cv2.hconcat([
                                cv2.resize(item.image, (max(2, int(item.image.shape[1] * 360 / item.image.shape[0]) // 2 * 2), 360))
                                for item in latest])
                            cv2.imshow("Replay cameras (live)", preview)
                            cv2.waitKey(1)
                    if next_tick is None:
                        next_tick = time.monotonic()
                    timeline.write(json.dumps({
                        "video_frame": frame_id, "video_time_s": frame_id / self.fps,
                        "host_timestamp_ns": now_ns, "replay_row": self.replay_row,
                        "cameras": {worker.camera_name: {
                            "serial": worker.source_serial,
                            "host_timestamp_ns": frame.host_timestamp_ns,
                            "timestamp_ns": frame.timestamp_ns,
                            "timestamp_domain": frame.timestamp_domain,
                        } for worker, frame in zip(self.workers, latest)},
                    }) + "\n")
                    timeline.flush()
                    self.ready.set()
                    frame_id += 1
                    next_tick += 1 / self.fps
                    if time.monotonic() - next_tick > 2:
                        raise RuntimeError("video encoding cannot keep up with camera fps")
                    self.stop_event.wait(max(0, next_tick - time.monotonic()))
        except Exception as exc:
            self.error = str(exc)
        finally:
            try:
                for writer in writers.values():
                    writer.release()
                if cv2 is not None:
                    try:
                        cv2.destroyWindow("Replay cameras (live)")
                    except Exception:
                        pass
            finally:
                for worker in started:
                    worker.stop()
                for worker in started:
                    worker.join(timeout=2)
                self.ready.set()
