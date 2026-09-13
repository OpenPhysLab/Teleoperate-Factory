# Quest VR Teleoperation for RealMan RM75-6F

This repository provides a ROS-free Python API2 teleoperation and dataset collection pipeline for Meta Quest 3/3S and one or two RealMan RM75-6F arms. RealSense cameras can be recorded alongside Quest poses, robot joints, and gripper commands.

## Hardware defaults

- Left RM75: `192.168.1.18:8080`
- Right RM75: `192.168.1.19:8080`
- Quest 3/3S: `172.16.204.100`
- RealSense configuration: `src/oculus_reader/config/cameras.yaml`

The arm model and API2 settings are in `src/oculus_reader/config/rm75.yaml`; the official RM75-6F URDF is under `src/oculus_reader/urdf/`.

## Installation

```bash
sudo apt install android-tools-adb
./scripts/setup_rm75_api2.sh
source scripts/activate_rm75_api2.sh
```

The environment needs NumPy, OpenCV, PyYAML, Pinocchio (or the numerical IK fallback), pyrealsense2, pure-python-adb, and the RealMan `Robotic_Arm` API2 package. Enable Quest Developer Mode and USB debugging, or connect over the configured network address.

## Dataset collection

```bash
source scripts/activate_rm75_api2.sh
python3 src/oculus_reader/scripts/rm75_teleop_dataset.py \
  --mode double --left-ip 192.168.1.18 --right-ip 192.168.1.19 \
  --quest-ip 172.16.204.100 --duration 3600
```

The default output is `/media/lyj/Data/GaussianRDT_Videos/session_TIMESTAMP/`. Each episode is stored in its own `episode_0001/`, `episode_0002/`, etc. directory. JSONL state recording is enabled by default; JPEG images are not written unless `--save-images` is supplied. The live camera preview remains available. Use `--dry-run` for a non-moving control test and `--no-robot` for Quest/camera-only capture.

### Quest controls

- Hold `B` (right) or `Y` (left) to enable that arm's following.
- `A+B` toggles the right gripper; `X+Y` toggles the left gripper.
- `A` starts the next episode after the previous one has ended.
- Release and press `X` three times within two seconds to save and end the current episode and restore the initial pose.
- `A+X` finishes the complete collection process.

Each manifest records Quest transforms/buttons, camera timestamps, robot joint feedback, and commanded gripper width. Begin with small motions before long captures.

## Replay

```bash
python3 src/oculus_reader/scripts/replay_rm75_joints.py \
  /media/lyj/Data/GaussianRDT_Videos/session_TIMESTAMP/episode_0001/manifest.jsonl \
  --mode double
```

Replay uses recorded joint and gripper values. By default it opens one live three-camera preview and writes one independent MP4 per camera to `/media/lyj/Data/GaussianRDT_Videos/replay_video_TIMESTAMP/`, together with `video_timestamps.jsonl`. Use `--no-video` to disable cameras, `--video-dir PATH` to select an output directory, and `--dry-run` to avoid commanding the real arms.

Other useful entry points are `rm75_teleop_calibration.py` for FK/IK diagnostics, `replay_video.py` for camera recording, and `rm_api2_control.py` for the API2 interface.

## Safety

Verify network addresses, enable state, initial poses, and camera identity before commanding a robot. Start with `--dry-run` and short, low-amplitude motions. Keep an emergency stop available and release the follow button whenever motion is unexpected.

Collected sessions, images, videos, JSONL manifests, logs, caches, and Python build artifacts are ignored by `.gitignore`; source code, configuration, URDF, and documentation remain version controlled.
