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

## Using `phyai_vla-ur7` recordings as demonstrations

The recorder's LeRobot v3 datasets (7-D joint state/action, `1` = closed gripper, wrist
video) convert to a reviewed `demo.json` bundle that `--demo` loads. The converter reads
the dataset directly, needs `ffmpeg` and `pip install -e '.[lerobot]'`, and calls no model.

```bash
python -m gpt_policy.input.ur_lerobot_demo ~/github/phyai_vla-ur7/open_trashcan \
    --config configs/machines/lab-ur5.json \
    --episodes 0-3 --task "Open the trash can lid" --output var/demos/open_trashcan

gpt-policy "Open the trash can lid" --demo var/demos/open_trashcan/episode-0001
```

- `--config` must be the machine profile you run the robot with. Joint angles are turned into fingertip TCP poses with its `link6_from_tcp`, so demonstrated and live poses share one frame. The frame is only valid while the robot base has not moved relative to the scene.
- **Keyframes** (default 12, at most 24) are the first and last frame, both sides of each gripper change, and frames spread evenly along the joint path. The numeric context keeps every turn, lift and gripper change (5 mm / 5 degree tolerance) and otherwise samples at 1 Hz.
- **Stage labels are mechanical** ("start", "after: gripper closes", "transit, 37% of joint path"). Edit `demo.json` to add descriptions before relying on it; set `--outcome success` only for episodes you have checked.
- **Gripper** values are converted to `0` closed, `1` open and labelled as the recorded open-loop command, not a measured opening or proof of a grasp.
- Episodes whose joints moved less than 0.01 rad are refused (a frozen RTDE feed during recording); `--allow-frozen` overrides. A dataset that kept the recorder's placeholder task text needs `--task`.
- `conversion.json` in each bundle records source file hashes, the chosen frames and the calibration file.
- With several cameras, name them with `--camera observation.images.KEY=NAME`. A single camera is named `left`, matching the wrist camera in the machine profile.
