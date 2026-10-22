# UR5 (CB3) setup

The `ur5` backend drives one CB3 UR5 through `ur_rtde`, with a Robotiq 2F gripper on the
tool connector and a RealSense wrist camera. Planning, retiming, settling and the tool
set are shared with the YAM backend; only the hardware layer differs
(`hardware/ur5.py`, `hardware/ur5_ik.py`, `hardware/robotiq_ur.py`).

```bash
python scripts/install_drivers.py ur5 realsense
gpt-policy --config configs/machines/lab-ur5.json --check   # opens no hardware
```

Copy `configs/examples/ur5-local.json` to a machine profile and set the values below.

## Before the first run

| Item | Where | Notes |
| --- | --- | --- |
| Robot IP | `runtime.interface` | `right_interface` must stay `null` (one arm). |
| Pendant | PolyScope 3.x | Remote control on, External Control program playing, **controller TCP set to zero** (poses are flange poses; the fingertip offset comes from `link6_from_tcp`). |
| Safety | pendant | Set safety planes and speed limits. IK acceptance does not prove collision-free motion. |
| Fingertip TCP | `calibration.link6_from_tcp` | The example ships an approximate 0.15 m z offset. Measure yours. |
| Wrist camera | `cameras.left`, `vision.camera_intrinsics.left`, `calibration.link6_from_camera.left` | The wrist camera is named `left` so it maps to the only arm. Intrinsics and hand-eye are placeholders until calibrated. |
| Robotiq script | `robot.robotiq_preamble` | Path to a Python file defining `ROBOTIQ_PREAMBLE` (the ur_rtde Robotiq preamble). If unset, `ur7e_recorder` is imported when installed. |

At startup the adapter compares its DH model with the controller's own forward
kinematics and refuses to run if they disagree by more than 1 mm, so a wrong robot
model or a non-zero controller TCP offset fails early.

## Behaviour to know

- **Frames.** Poses are in the controller base frame (what RTDE reports). The model sees the calibrated fingertip TCP.
- **Streaming.** `servoJ` at `runtime.trajectory_hz` (125 Hz on CB3) with `robot.servo_lookahead_s` and `robot.servo_gain`. Defaults are slow (`tcp_velocity_m_s` 0.05); raise them only after dry runs with `check_path`.
- **Gripper is open loop.** The Robotiq script cannot report finger position, so `gripper_normalized` is the last command (`1` open, `0` closed; the Robotiq register is inverted internally). Each gripper call stops the arm, sends a custom script (which resets the RTDE session) and resynchronizes the receive link.
- **Faults.** An emergency or protective stop latches a fault for the rest of the process; restart after clearing it.
- **Home.** `return_home` goes back to the joint pose the arm had at startup. Start each run from a safe pose.
