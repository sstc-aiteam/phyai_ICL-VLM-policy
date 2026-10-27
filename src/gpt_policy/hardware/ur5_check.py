"""Bring-up check for a CB3 UR5: connects, reads and compares; never moves the arm.

    python -m gpt_policy.hardware.ur5_check --config configs/machines/lab-ur5.json

By default this opens only a receive-side RTDE connection plus read-only Dashboard
queries, so it sends no motion command and uploads no program. ``--test-gripper``
is the one exception: it opens the control interface and cycles the Robotiq fingers
(the arm still does not move).
"""
from __future__ import annotations

import argparse
import json
import re
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..geometry.frames import FrameCalibration
from ..geometry.poses import pose_to_matrix, rotation_distance
from ..settings import load_settings
from .robotiq_ur import RobotiqUR, load_preamble
from .ur5 import UR5_JOINT_LIMITS, UR5Link
from .ur5_ik import UR5IK, chain_from_settings, rtde_pose_to_pose

DASHBOARD_PORT = 29999
ROBOT_MODE_RUNNING = 7
SAFETY_NORMAL = 1
MIN_FEED_HZ = 50.0
POSITION_TOLERANCE_M = 1e-3
ROTATION_TOLERANCE_RAD = 5e-3


@dataclass
class Result:
    status: str  # PASS | WARN | FAIL | SKIP
    name: str
    detail: str = ""

    def line(self) -> str:
        return f"[{self.status}] {self.name}" + (f": {self.detail}" if self.detail else "")


def _placeholder(value) -> bool:
    return not value or (isinstance(value, str) and value.strip().upper().startswith("REPLACE"))


def _identity(matrix) -> bool:
    return bool(np.allclose(np.asarray(matrix, dtype=float), np.eye(np.asarray(matrix).shape[0])))


def check_config(settings, path: Path) -> list[Result]:
    results = []
    runtime = settings.get("runtime", {})
    robot = settings.get("robot", {})
    calibration = settings.get("calibration", {})
    if settings.get("backend") != "ur5":
        results.append(Result("FAIL", "backend", f"expected 'ur5', got {settings.get('backend')!r}"))
    ip = runtime.get("interface")
    results.append(Result("FAIL", "robot IP", "runtime.interface is not set") if _placeholder(ip)
                   else Result("PASS", "robot IP", ip))
    try:
        from ..preflight import check_configuration
        check_configuration(settings, path)
        results.append(Result("PASS", "configuration", "structure and calibration are valid"))
    except Exception as exc:
        results.append(Result("FAIL", "configuration", str(exc)))
    if robot.get("gripper_kind", "robotiq") == "robotiq":
        try:
            load_preamble(robot.get("robotiq_preamble"))
            results.append(Result("PASS", "Robotiq preamble", "loaded"))
        except Exception as exc:
            results.append(Result("FAIL", "Robotiq preamble", str(exc)))
    else:
        results.append(Result("SKIP", "Robotiq preamble", "gripper_kind is not robotiq"))
    placeholders = []
    if _identity(calibration.get("link6_from_tcp", np.eye(4))):
        placeholders.append("link6_from_tcp is identity (TCP at the flange)")
    for name, matrix in calibration.get("link6_from_camera", {}).items():
        if _identity(matrix):
            placeholders.append(f"hand-eye link6_from_camera.{name} is identity")
    for name, matrix in settings.get("vision", {}).get("camera_intrinsics", {}).items():
        if _identity(matrix):
            placeholders.append(f"camera_intrinsics.{name} is identity")
    for name, camera in settings.get("cameras", {}).items():
        if _placeholder(camera.get("serial")):
            placeholders.append(f"cameras.{name}.serial is a placeholder")
    results.append(Result("WARN", "calibration placeholders", "; ".join(placeholders)) if placeholders
                   else Result("PASS", "calibration placeholders", "none left"))
    return results


def check_dashboard(ip: str, connect=socket.create_connection, timeout=2.0) -> list[Result]:
    """Read-only Dashboard queries: PolyScope version, robot mode, safety status."""
    try:
        with connect((ip, DASHBOARD_PORT), timeout=timeout) as sock:
            reader = sock.makefile("r", encoding="utf-8", newline="\n")
            reader.readline()
            replies = {}
            for command in ("PolyscopeVersion", "robotmode", "safetystatus"):
                sock.sendall(f"{command}\n".encode())
                replies[command] = reader.readline().strip()
    except OSError as exc:
        return [Result("FAIL", "Dashboard", f"cannot reach {ip}:{DASHBOARD_PORT} ({exc})")]
    results = []
    match = re.search(r"\d+(?:\.\d+){2,}", replies["PolyscopeVersion"])
    if not match:
        results.append(Result("WARN", "PolyScope version", f"unparsed reply {replies['PolyscopeVersion']!r}"))
    elif int(match.group().split(".")[0]) < 5:
        results.append(Result("PASS", "PolyScope version", f"{match.group()} (CB3)"))
    else:
        results.append(Result("WARN", "PolyScope version", f"{match.group()} is not CB3; this adapter was written for CB3"))
    mode = replies["robotmode"]
    results.append(Result("PASS" if "RUNNING" in mode.upper() else "WARN", "robot mode", mode))
    safety = replies["safetystatus"]
    results.append(Result("PASS" if "NORMAL" in safety.upper() else "WARN", "safety status", safety))
    return results


def check_rtde(settings, receive, seconds=1.0, clock=time.monotonic, sleep=time.sleep) -> list[Result]:
    results = []
    robot = settings.get("robot", {})
    limits = np.asarray(robot.get("joint_limits_rad", UR5_JOINT_LIMITS), dtype=float)

    stamps, started = [], clock()
    while clock() - started < seconds:
        stamp = float(receive.getTimestamp())
        if not stamps or stamp != stamps[-1]:
            stamps.append(stamp)
        sleep(0.002)
    span = stamps[-1] - stamps[0] if len(stamps) > 1 else 0.0
    rate = (len(stamps) - 1) / span if span > 0 else 0.0
    if len(stamps) < 2:
        results.append(Result("FAIL", "RTDE feed", "controller timestamp did not advance (stuck receive link)"))
        return results
    results.append(Result("PASS" if rate >= MIN_FEED_HZ else "WARN", "RTDE feed", f"{rate:.0f} Hz"))

    if receive.isEmergencyStopped():
        results.append(Result("FAIL", "emergency stop", "robot is emergency stopped"))
    if receive.isProtectiveStopped():
        results.append(Result("FAIL", "protective stop", "robot is protective stopped"))
    mode, safety = int(receive.getRobotMode()), int(receive.getSafetyMode())
    results.append(Result("PASS" if mode == ROBOT_MODE_RUNNING else "WARN", "robot mode",
                          f"{mode} ({'RUNNING' if mode == ROBOT_MODE_RUNNING else 'not RUNNING: power on / release brakes'})"))
    results.append(Result("PASS" if safety == SAFETY_NORMAL else "WARN", "safety mode", str(safety)))

    q = np.asarray(receive.getActualQ(), dtype=float)
    outside = [i + 1 for i in range(6) if not limits[i, 0] <= q[i] <= limits[i, 1]]
    results.append(Result("FAIL", "joint limits", f"joints {outside} are outside robot.joint_limits_rad") if outside
                   else Result("PASS", "joint limits", "current joints are inside the configured range"))
    temperatures = list(receive.getJointTemperatures())
    limit = float(robot.get("motor_temp_limit_c", 80.0))
    results.append(Result("WARN" if max(temperatures) >= limit else "PASS", "joint temperatures",
                          f"max {max(temperatures):.0f} C (limit {limit:.0f} C)"))

    ik = UR5IK(settings.get("motion", {}), limits, chain_from_settings(settings))
    flange = ik.forward_kinematics(q)
    controller = rtde_pose_to_pose(receive.getActualTCPPose())
    rotation_error = rotation_distance(flange[3:], controller[3:])
    offset = (np.linalg.inv(pose_to_matrix(flange)) @ pose_to_matrix(controller))[:3, 3]
    if rotation_error > ROTATION_TOLERANCE_RAD:
        results.append(Result("FAIL", "kinematics vs controller",
                              f"orientation differs by {rotation_error:.3f} rad: the kinematic model does not match this robot "
                              "(save its factory calibration with `python -m gpt_policy.hardware.ur5_kinematics`), "
                              "or the controller TCP has a rotation (set it to zero on the pendant and rerun)"))
    elif np.linalg.norm(offset) > POSITION_TOLERANCE_M:
        results.append(Result("WARN", "kinematics vs controller",
                              "orientation matches; controller TCP offset is "
                              f"[{offset[0]:.4f}, {offset[1]:.4f}, {offset[2]:.4f}] m. The adapter plans from joints and "
                              "does not need this, but zero it so RTDE poses are flange poses"))
    else:
        results.append(Result("PASS", "kinematics vs controller",
                              f"flange pose agrees (position {np.linalg.norm(offset) * 1000:.2f} mm, "
                              f"orientation {rotation_error * 1000:.2f} mrad)"))
    try:
        solved = ik.solve(flange, q, 0, 0)
        error = float(np.max(np.abs(solved - q)))
        results.append(Result("PASS" if error < 1e-3 else "WARN", "IK round trip",
                              f"max joint difference {error:.2e} rad at the current pose"))
    except Exception as exc:
        results.append(Result("WARN", "IK round trip", f"no solution at the current pose ({exc})"))

    try:
        frames = FrameCalibration(settings)
        tcp = frames.sdk_to_tcp(flange)
        results.append(Result("PASS", "fingertip TCP pose",
                              "xyz [" + ", ".join(f"{v:.3f}" for v in tcp[:3]) + "] m, rpy ["
                              + ", ".join(f"{v:.3f}" for v in tcp[3:]) + "] rad (what the model will see)"))
    except Exception as exc:
        results.append(Result("WARN", "fingertip TCP pose", str(exc)))
    return results


class RealSenseProbe:
    """Real-device adapter: ``capture() -> (rgb, ...)`` and ``intrinsics()`` for one camera."""

    def __init__(self, name, serial, width, height):
        from .realsense import RealSenseCameraSet
        self.set = RealSenseCameraSet([(name, serial)], width, height)
        self.name, self.width, self.height = name, width, height

    def capture(self, frames=15):
        image = None
        for _ in range(frames):  # let exposure settle; keep the last frame
            image = self.set.capture()[self.name]
        return np.frombuffer(image.rgb_data, dtype=np.uint8).reshape(image.height, image.width, 3)

    def intrinsics(self):
        rs = self.set.rs
        profile = self.set.cameras[0].pipeline.get_active_profile().get_stream(rs.stream.color)
        i = profile.as_video_stream_profile().get_intrinsics()
        return {"fx": i.fx, "fy": i.fy, "cx": i.ppx, "cy": i.ppy, "distortion": list(i.coeffs), "model": str(i.model)}

    def close(self):
        self.set.close()


def check_camera(settings, name="left", probe_factory=RealSenseProbe, save_frame: Path | None = None) -> list[Result]:
    camera = settings.get("cameras", {}).get(name)
    if camera is None or _placeholder(camera.get("serial")):
        return [Result("FAIL", f"camera {name}", "no RealSense serial configured")]
    runtime = settings.get("runtime", {})
    try:
        probe = probe_factory(name, camera["serial"], runtime.get("camera_width", 640), runtime.get("camera_height", 480))
    except Exception as exc:
        return [Result("FAIL", f"camera {name}", f"cannot open {camera['serial']}: {exc}")]
    results = []
    try:
        rgb = probe.capture()
        mean, spread = float(rgb.mean()), float(rgb.std())
        usable = 10 <= mean <= 245 and spread >= 5
        results.append(Result("PASS" if usable else "WARN", f"camera {name} frame",
                              f"{rgb.shape[1]}x{rgb.shape[0]}, mean {mean:.0f}, std {spread:.0f}"
                              + ("" if usable else " (looks black, blown out or uniform: lens covered?)")))
        if save_frame is not None:
            from PIL import Image
            save_frame.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb).save(save_frame)
            results.append(Result("PASS", "saved frame", str(save_frame)))
        device = probe.intrinsics()
        configured = np.asarray(settings.get("vision", {}).get("camera_intrinsics", {}).get(name, np.eye(3)), dtype=float)
        measured = np.array([[device["fx"], 0, device["cx"]], [0, device["fy"], device["cy"]], [0, 0, 1]])
        snippet = json.dumps({"camera_intrinsics": {name: measured.round(3).tolist()},
                              "distortion_coefficients": {name: [round(c, 6) for c in device["distortion"]]}})
        if configured.shape != (3, 3):
            results.append(Result("WARN", "camera intrinsics",
                                  f"config matrix has shape {configured.shape}, expected 3x3; device reports: " + snippet))
        elif _identity(configured):
            results.append(Result("WARN", "camera intrinsics", "config has placeholders; device reports: " + snippet))
        elif np.max(np.abs(configured - measured)[:2]) > 2.0:
            results.append(Result("WARN", "camera intrinsics",
                                  "config differs from the device by more than 2 px/unit; device reports: " + snippet))
        else:
            results.append(Result("PASS", "camera intrinsics", "match the device's factory values"))
    except Exception as exc:
        results.append(Result("FAIL", f"camera {name}", str(exc)))
    finally:
        try:
            probe.close()
        except Exception:
            pass
    return results


def run_gripper_test(settings, link_factory=None, preamble_loader=load_preamble) -> list[Result]:
    """Cycle the Robotiq open, half, open. Moves the fingers only."""
    runtime, robot = settings.get("runtime", {}), settings.get("robot", {})
    link = None
    try:
        link = (link_factory or (lambda: UR5Link(runtime["interface"], hz=runtime.get("trajectory_hz", 125.0))))()
        gripper = RobotiqUR(link, preamble_loader(robot.get("robotiq_preamble")),
                            speed_percent=robot.get("gripper_speed_percent", 50),
                            force_percent=robot.get("gripper_force_percent", 50),
                            settle_s=robot.get("gripper_settle_s", 0.2))
        moves = [gripper.move(opening) for opening in (1.0, 0.5, 1.0)]
        accepted = all(m["script_accepted"] for m in moves)
        return [Result("PASS" if accepted else "WARN", "gripper cycle",
                       "open, half, open sent" + ("" if accepted else "; the controller did not confirm every script"))]
    except Exception as exc:
        return [Result("FAIL", "gripper cycle", str(exc))]
    finally:
        if link is not None:
            link.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="UR5 machine profile")
    parser.add_argument("--no-camera", action="store_true")
    parser.add_argument("--no-network", action="store_true", help="Skip every robot connection (config checks only)")
    parser.add_argument("--camera", default="left", help="Camera name in the profile (default left)")
    parser.add_argument("--rtde-seconds", type=float, default=1.0)
    parser.add_argument("--save-frame", type=Path, help="Write the captured camera frame to this path")
    parser.add_argument("--test-gripper", action="store_true", help="Cycle the Robotiq fingers (arm does not move)")
    parser.add_argument("--yes", action="store_true", help="Do not ask before --test-gripper")
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    results = check_config(settings, args.config)
    ip = settings.get("runtime", {}).get("interface")
    if args.no_network or _placeholder(ip):
        results.append(Result("SKIP", "robot connection", "no network checks" if args.no_network else "no robot IP"))
    else:
        results += check_dashboard(ip)
        try:
            import rtde_receive
            receive = rtde_receive.RTDEReceiveInterface(ip)
        except Exception as exc:
            results.append(Result("FAIL", "RTDE receive", f"cannot connect to {ip}: {exc}"))
        else:
            try:
                results += check_rtde(settings, receive, args.rtde_seconds)
            finally:
                try:
                    receive.disconnect()
                except Exception:
                    pass
    if args.no_camera:
        results.append(Result("SKIP", "camera", "--no-camera"))
    else:
        results += check_camera(settings, args.camera, save_frame=args.save_frame)
    if args.test_gripper and not args.no_network and not _placeholder(ip):
        if args.yes or input("Robotiq fingers will open and close. Keep them clear and type 'yes': ").strip() == "yes":
            results += run_gripper_test(settings)
        else:
            results.append(Result("SKIP", "gripper cycle", "not confirmed"))
    else:
        results.append(Result("SKIP", "gripper cycle", "pass --test-gripper to cycle the fingers"))

    for result in results:
        print(result.line())
    failed = sum(r.status == "FAIL" for r in results)
    warned = sum(r.status == "WARN" for r in results)
    print(f"\n{failed} failed, {warned} warnings. No arm motion command was sent.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
