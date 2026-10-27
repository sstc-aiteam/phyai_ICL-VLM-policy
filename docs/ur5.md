# UR5 (CB3) setup

The `ur5` backend drives one CB3 UR5 through `ur_rtde`, with a Robotiq 2F gripper on the
tool connector and a RealSense wrist camera. Planning, retiming, settling and the tool
set are shared with the YAM backend; only the hardware layer differs
(`hardware/ur5.py`, `hardware/ur5_ik.py`, `hardware/robotiq_ur.py`).

```bash
python scripts/install_drivers.py ur5 realsense
gpt-policy --config configs/machines/lab-ur5.json --check   # opens no hardware
python -m gpt_policy.hardware.ur5_check --config configs/machines/lab-ur5.json   # read-only robot check
```

Copy `configs/examples/ur5-local.json` to a machine profile and set the values below. If you move it out of `configs/examples/`, change its `extends` paths (`../examples/simulated.json`, `../robots/ur5.json`), which are relative to the file.

## Before the first run

| Item | Where | Notes |
| --- | --- | --- |
| Robot IP | `runtime.interface` | `right_interface` must stay `null` (one arm). |
| Pendant | PolyScope 3.x | Remote control on, External Control program playing, **controller TCP set to zero** (poses are flange poses; the fingertip offset comes from `link6_from_tcp`). |
| Safety | pendant | Set safety planes and speed limits. IK acceptance does not prove collision-free motion. |
| Fingertip TCP | `calibration.link6_from_tcp` | The example ships an approximate 0.15 m z offset. Measure yours. |
| Wrist camera | `cameras.left`, `vision.camera_intrinsics.left`, `calibration.link6_from_camera.left` | The wrist camera is named `left` so it maps to the only arm. Intrinsics and hand-eye are placeholders until calibrated. |
| Arm kinematics | `robot.kinematics_calibration` | This arm's factory DH calibration. Every UR arm differs from the nominal model by millimetres, enough to fail the startup check. Run `python -m gpt_policy.hardware.ur5_kinematics <robot ip>` (read-only) and paste the block into the profile. |
| Robotiq script | `robot.robotiq_preamble` | Path to a Python file defining `ROBOTIQ_PREAMBLE` (the ur_rtde Robotiq preamble). If unset, `ur7e_recorder` is imported when installed. |

At startup the adapter compares its DH model with the controller's own forward
kinematics and refuses to run if they disagree by more than 1 mm, so a wrong robot
model or a non-zero controller TCP offset fails early.

## Bring-up check

Run this before the first agent run, and again after changing calibration or cabling:

```bash
python -m gpt_policy.hardware.ur5_check --config configs/machines/lab-ur5.json --save-frame var/wrist.png
```

It sends no arm motion command. It reads the PolyScope version, robot mode and safety
status from the Dashboard, then opens a **receive-only** RTDE connection (no control
script is uploaded). It reports FAIL, WARN or PASS for:

| Check | Catches |
| --- | --- |
| RTDE feed rate | A stuck or slow receive link (the same fault that froze some recorded episodes). |
| Stops, robot and safety mode | Emergency or protective stop, brakes not released, reduced mode. |
| Joint limits and temperatures | Starting outside `robot.joint_limits_rad`, hot motors. |
| Kinematics vs controller | The DH model not matching the robot, or a rotated controller TCP (FAIL); a non-zero controller TCP offset is a WARN with the measured offset. |
| IK round trip | The solver failing at the arm's current pose. |
| Fingertip TCP pose | Prints the pose the model will see, for you to sanity-check against reality. |
| Camera | A missing or dark frame; prints the device's own intrinsics to paste into `vision.camera_intrinsics` when the profile still has placeholders. |
| Calibration placeholders | Identity TCP, hand-eye or intrinsics, and placeholder serials left in the profile. |

`--no-network` and `--no-camera` limit it to configuration checks. `--test-gripper`
(asks for confirmation unless `--yes`) is the one option that does more: it opens the
control interface and cycles the Robotiq open, half, open. The arm still does not move.

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
