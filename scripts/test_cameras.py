#!/usr/bin/env python3
"""Non-ROS V4L2 health check for the three configured RealSense cameras."""

import argparse
import multiprocessing as mp
import os
import sys
import time


def _node_name(path):
    node = os.path.basename(path)
    try:
        with open('/sys/class/video4linux/{}/name'.format(node), encoding='utf-8') as stream:
            return stream.read().strip()
    except OSError:
        return 'unknown'


def _probe(path, frames, warmup, queue):
    """Probe one V4L2 node in a child so a bad node cannot hang the check."""
    try:
        import cv2
        cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
        if not cap.isOpened():
            queue.put((False, 0, None))
            cap.release()
            return
        time.sleep(max(0.0, warmup))
        count, shape = 0, None
        for _ in range(max(1, frames)):
            ret, frame = cap.read()
            if ret and frame is not None:
                count += 1
                shape = tuple(frame.shape)
        cap.release()
        queue.put((count > 0, count, shape))
    except Exception:
        queue.put((False, 0, None))


def probe(path, frames, warmup, timeout):
    queue = mp.Queue()
    process = mp.Process(target=_probe, args=(path, frames, warmup, queue))
    process.start()
    process.join(max(0.1, timeout))
    if process.is_alive():
        process.terminate()
        process.join(1.0)
        return False, 0, None
    if not queue.empty():
        return queue.get()
    return False, 0, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='src/oculus_reader/config/cameras.yaml')
    parser.add_argument('--frames', type=int, default=2,
                        help='frames to read from each node')
    parser.add_argument('--warmup', type=float, default=0.15)
    parser.add_argument('--timeout', type=float, default=2.0,
                        help='maximum seconds allowed per V4L2 node')
    parser.add_argument('--min-readable', type=int, default=1,
                        help='minimum readable V4L2 nodes per physical camera')
    parser.add_argument('--preferred-only', action='store_true',
                        help='test only preferred_nodes from the config')
    args = parser.parse_args()
    try:
        import yaml
        import cv2  # noqa: F401
    except ImportError as exc:
        raise SystemExit('camera test requires PyYAML and opencv-python: {}'.format(exc))

    with open(args.config, encoding='utf-8') as stream:
        cameras = yaml.safe_load(stream).get('cameras', [])
    total_ok = 0
    failed_cameras = 0
    total_nodes = 0
    for camera in cameras:
        print('\n{} ({}, serial={})'.format(
            camera['name'], camera.get('model', 'unknown'), camera.get('serial', 'unknown')))
        camera_ok = 0
        nodes = camera.get('preferred_nodes', []) if args.preferred_only else camera.get('video_devices', [])
        for path in nodes:
            total_nodes += 1
            healthy, frames, shape = probe(
                path, args.frames, args.warmup, args.timeout)
            healthy = frames > 0
            if healthy:
                camera_ok += 1
                total_ok += 1
            print('  {:12s} {:5s} frames={}/{} shape={} name={}'.format(
                path, 'OK' if healthy else 'FAIL', frames, max(1, args.frames),
                shape, _node_name(path)))
        camera_pass = camera_ok >= args.min_readable
        if not camera_pass:
            failed_cameras += 1
        print('  camera result: {}/{} readable nodes -> {}'.format(
            camera_ok, len(nodes),
            'OK' if camera_pass else 'FAIL'))

    print('\nTOTAL readable nodes: {}/{}; failed cameras: {}'.format(
        total_ok, total_nodes, failed_cameras))
    return 0 if failed_cameras == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
