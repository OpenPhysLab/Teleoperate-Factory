# DAgger Dependencies

External dependencies copied into this directory to keep `dagger/` self-contained.
**Do not modify these files** — they are copies from the source projects.

## Source Mapping

| File | Source | Copy Date |
| :--- | :--- | :--- |
| `constants.py` | `RoboCOIN/src/lerobot/extensions/unified_deploy/core/constants.py` | 2026-02-08 |
| `data_types.py` | `RoboCOIN/src/lerobot/extensions/unified_deploy/core/data_types.py` | 2026-02-08 |
| `utils.py` | `RoboCOIN/src/lerobot/extensions/unified_deploy/core/utils.py` | 2026-02-08 |
| `configs.py` | `RoboCOIN/src/lerobot/extensions/unified_deploy/client/configs.py` | 2026-02-08 |
| `gripper_controller.py` | `RoboCOIN/src/lerobot/extensions/unified_deploy/client/gripper_controller.py` | 2026-02-08 |

## gRPC Transport (NOT copied)

The gRPC generated files (`services_pb2.py`, `services_pb2_grpc.py`) and `transport/utils.py`
are referenced via `sys.path` pointing to `RoboCOIN/src/` because:
1. Generated protobuf code has hardcoded import paths (`from lerobot.transport import ...`)
2. Modifying generated code is fragile and error-prone
3. The `lerobot.transport` package also depends on `lerobot.utils.transition`

The `async_inference_client.py` adds `RoboCOIN/src/` to `sys.path` at startup.

## Realman SDK (NOT copied)

`Robotic_Arm/rm_robot_interface.py` is a large C-wrapper SDK (295KB).
Referenced via `sys.path` pointing to the project root.
