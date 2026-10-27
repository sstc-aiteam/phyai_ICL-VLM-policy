"""UR5 (CB3) adapter: ur_rtde servoJ streaming under the shared TCP planner.

Planning, retiming, settling, homing and tool dispatch are inherited from the
YAM adapter because both are six-joint arms driven by streamed joint targets.
Only the pieces that talk to hardware are replaced: the RTDE link, a feedback
snapshot, a blocking Robotiq gripper and fault latching.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from ..geometry.poses import rotation_distance
from .motion_control import MotionFault
from .robotiq_ur import NoGripper, RobotiqUR, load_preamble
from .ur5_ik import UR5IK, chain_from_settings, rtde_pose_to_pose
from .yam import YamRobot

UR5_JOINT_LIMITS = [[-2 * np.pi, 2 * np.pi]] * 6
STALE_FEED_S = 0.5


class UR5Link:
    """Owns the RTDE control/receive pair and exposes the driver surface YamRobot uses."""

    _server_thread = None  # YamRobot probes this for the I2RT command thread.

    def __init__(self, ip, *, hz=125.0, servo_lookahead_s=0.1, servo_gain=300,
                 joint_limits=None, control=None, receive=None, receive_factory=None,
                 attempts=4, retry_delay_s=5.0):
        self.ip, self.hz = ip, float(hz)
        self.lookahead, self.gain = float(servo_lookahead_s), float(servo_gain)
        if not 0.03 <= self.lookahead <= 0.2 or not 100 <= self.gain <= 2000:
            raise ValueError("servoJ needs lookahead in [0.03, 0.2] s and gain in [100, 2000]")
        limits = np.asarray(joint_limits if joint_limits is not None else UR5_JOINT_LIMITS, dtype=float)
        self.joint_limits = np.vstack([limits, [0.0, 1.0]])  # six joints + normalized gripper
        self._receive_factory = receive_factory
        self.control, self.receive = control, receive
        self._lock = threading.RLock()
        self._closed = False
        if self.control is None:
            self._connect(attempts, retry_delay_s)

    def _connect(self, attempts, retry_delay_s):
        import rtde_control
        import rtde_receive

        self._receive_factory = lambda: rtde_receive.RTDEReceiveInterface(self.ip)
        for attempt in range(1, attempts + 1):
            try:
                # Control first: constructing it uploads the External Control script
                # and resets the controller's RTDE session.
                self.control = rtde_control.RTDEControlInterface(self.ip)
                self.receive = self._receive_factory()
                return
            except RuntimeError as exc:
                self._release()
                if attempt == attempts:
                    raise RuntimeError(
                        f"Could not open RTDE to {self.ip} after {attempts} attempts ({exc}). "
                        "A previous run may still hold the session; stop any pendant program "
                        "and wait about a minute.") from exc
                time.sleep(retry_delay_s * attempt)

    def _release(self):
        for owner, calls in ((self.control, ("servoStop", "stopScript", "disconnect")),
                             (self.receive, ("disconnect",))):
            for call in calls if owner is not None else ():
                try:
                    getattr(owner, call)()
                except Exception:
                    pass
        self.control = self.receive = None

    def get_robot_info(self):
        return {"joint_limits": self.joint_limits}

    def command_joint_pos(self, command):
        """servoJ to the six joint targets; the seventh value (gripper) is not an RTDE joint."""
        q = np.asarray(command, dtype=float)[:6]
        if not np.isfinite(q).all():
            raise ValueError("non-finite joint command")
        if not self.control.isConnected():
            raise RuntimeError("RTDE control connection lost")
        if not self.control.servoJ(q.tolist(), 0.0, 0.0, 1.0 / self.hz, self.lookahead, self.gain):
            raise RuntimeError("servoJ was rejected (protective stop or script stopped?)")

    def stop(self):
        self.control.servoStop()

    def read(self):
        r = self.receive
        if not r.isConnected():
            self.resync_receive()
            r = self.receive
        try:
            temperatures = list(r.getJointTemperatures())
        except Exception:
            temperatures = [0.0] * 6
        return {
            "q": list(r.getActualQ()), "dq": list(r.getActualQd()),
            "controller_time_s": float(r.getTimestamp()),
            "tcp_pose_rtde": list(r.getActualTCPPose()),
            "temperature_c": temperatures,
            "emergency_stopped": bool(r.isEmergencyStopped()),
            "protective_stopped": bool(r.isProtectiveStopped()),
            "robot_mode": int(r.getRobotMode()),
        }

    def resync_receive(self):
        """Rebuild the receive side; a custom script can leave it replaying a stale packet."""
        if self.receive is not None:
            try:
                self.receive.disconnect()
            except Exception:
                pass
        self.receive = self._receive_factory()

    def run_script(self, name, text):
        """Run a custom URScript function, then restore RTDE state it may have reset."""
        with self._lock:
            self.control.servoStop()
            ok = self.control.sendCustomScriptFunction(name, text)
            self.resync_receive()
            if not self.control.isProgramRunning() and hasattr(self.control, "reuploadScript"):
                self.control.reuploadScript()
            return ok

    # CB3 3.13 never answers get_forward_kin when the offset is exactly zero, and joints alone return a
    # stale tool offset; 1 micrometre along z is the same flange pose to well inside the 1 mm check.
    _FK_PROBE_OFFSET = [0.0, 0.0, 1e-6, 0.0, 0.0, 0.0]

    def controller_flange_pose(self, joints):
        """Controller FK with a (numerically) zero tool offset, as xyzrpy (kinematics cross-check)."""
        return rtde_pose_to_pose(self.control.getForwardKinematics(list(joints), self._FK_PROBE_OFFSET))

    def close(self):
        if self._closed:
            return
        self._closed = True
        with self._lock:
            self._release()


class UR5Robot(YamRobot):
    """One UR5; ``interface`` is the controller IP address."""

    dof = 6

    def __init__(self, ip, settings, side="left", *, link=None, ik=None, gripper=None):
        robot = settings.get("robot", {})
        runtime = settings.get("runtime", {})
        motion = settings.get("motion", {})
        hz = float(runtime.get("trajectory_hz", 125))
        limits = robot.get("joint_limits_rad")
        self._gripper_impl = gripper
        self._last_feed = (None, time.monotonic())
        link = link or UR5Link(
            ip, hz=hz, servo_lookahead_s=robot.get("servo_lookahead_s", 0.1),
            servo_gain=robot.get("servo_gain", 300), joint_limits=limits)
        try:
            if self._gripper_impl is None:
                self._gripper_impl = self._open_gripper(link, robot)
            super().__init__(ip, settings, side, driver=link,
                             ik=ik or UR5IK(motion, limits, chain_from_settings(settings)))
            self._verify_kinematics()
        except BaseException:
            link.close()
            raise

    @staticmethod
    def _open_gripper(link, robot):
        kind = robot.get("gripper_kind", "robotiq")
        if kind == "none":
            return NoGripper()
        if kind != "robotiq":
            raise ValueError(f"Unknown UR5 gripper_kind: {kind}")
        return RobotiqUR(
            link, load_preamble(robot.get("robotiq_preamble")),
            speed_percent=robot.get("gripper_speed_percent", 50),
            force_percent=robot.get("gripper_force_percent", 50),
            settle_s=robot.get("gripper_settle_s", 0.2))

    def _verify_kinematics(self):
        """Fail at startup if the DH model is not this controller's robot."""
        if not hasattr(self.driver, "controller_flange_pose"):
            return
        q = self._feedback()["q"][:6]
        model = self.ik.forward_kinematics(q)
        controller = self.driver.controller_flange_pose(q)
        if (np.linalg.norm(model[:3] - controller[:3]) > 1e-3
                or rotation_distance(model[3:], controller[3:]) > 5e-3):
            raise ValueError(
                "UR5 kinematics disagree with the controller's own forward kinematics; "
                "is this a CB3 UR5 with default calibration?")

    def _fault(self, details):
        self.motion.fault = details
        raise MotionFault(details)

    def _feedback(self):
        raw = self.driver.read()
        if raw["emergency_stopped"]:
            self._fault({"reason": "emergency_stop"})
        if raw["protective_stopped"]:
            self._fault({"reason": "protective_stop"})
        stamp, seen = self._last_feed
        if raw["controller_time_s"] != stamp:
            self._last_feed = (raw["controller_time_s"], time.monotonic())
        elif time.monotonic() - seen > STALE_FEED_S:
            raise RuntimeError("RTDE state feed has not updated; the receive link is stale")
        q = np.r_[raw["q"], self._gripper_impl.opening]
        dq = np.r_[raw["dq"], 0.0]
        temperature = np.r_[raw["temperature_c"], 0.0]
        hot = np.flatnonzero(temperature >= self.temperature_limit)
        over_limit = {"temp_rotor": [{"motor_index": int(i), "value_c": float(temperature[i])} for i in hot]}
        return {
            "q": q, "dq": dq, "effort": np.zeros(7), "timestamp": raw["controller_time_s"],
            "temp_mos": temperature.copy(), "temp_rotor": temperature,
            "temperature_over_limit": {"over_limit": over_limit, "limit_c": self.temperature_limit} if hot.size else None,
            "tcp_pose_rtde": raw["tcp_pose_rtde"], "robot_mode": raw["robot_mode"],
        }

    def state(self):
        s = self._feedback()
        result = super().state()
        # super().state() re-reads feedback; keep the fields that are specific to UR.
        result.update({
            "gravity_compensation": False,
            "gripper_torque_nm": None,
            "gripper_feedback": "open_loop_last_command",
            "controller_tcp_pose_rtde": s["tcp_pose_rtde"],
            "robot_mode": s["robot_mode"],
            "timestamp_source": "ur_rtde_controller_time",
        })
        return result

    def _set_gripper(self, position):
        if not np.isfinite(position) or not 0 <= position <= 1:
            raise ValueError("gripper must be in [0, 1]")
        self.wait_reference_complete()
        self.motion.check()
        # The gripper script replaces the controller program and blocks until the
        # fingers stop; the arm is stationary and must not be mid-servo.
        info = self._gripper_impl.move(float(position))
        self._gripper_target = float(position)
        return {"active_command_normalized": position, **info,
                "force_limiter": "Robotiq force setting"}

    def _hold(self):
        q = self._feedback()["q"]
        self.driver.stop()
        self._target = q[:6].copy()
        self._gripper_target = float(q[6])

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._pool.shutdown(wait=True, cancel_futures=True)
        if self.driver is not None:
            try:
                self.driver.stop()
            except Exception:
                pass
            self.driver.close()
