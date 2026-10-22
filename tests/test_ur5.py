"""UR5 kinematics and adapter behaviour against a simulated RTDE link."""
import numpy as np
import pytest

from gpt_policy.geometry.poses import matrix_to_pose, pose_to_matrix
from gpt_policy.hardware.motion_control import MotionFault
from gpt_policy.hardware.robotiq_ur import NoGripper, RobotiqUR, opening_to_raw
from gpt_policy.hardware.ur5 import UR5Link, UR5Robot
from gpt_policy.hardware.ur5_ik import UR5IK, analytic_solutions, forward_matrix
from gpt_policy.motion.coordination import TrajectoryIKError

HOME = np.array([0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])


def rotvec(matrix):
    angle = np.arccos(np.clip((np.trace(matrix) - 1) / 2, -1, 1))
    axis = np.array([matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]])
    return axis / np.linalg.norm(axis) * angle if np.linalg.norm(axis) > 1e-12 else np.zeros(3)


def rtde_pose(joints):
    transform = forward_matrix(joints)
    return [*transform[:3, 3], *rotvec(transform[:3, :3])]


def test_zero_pose_matches_published_ur5_flange_position():
    assert forward_matrix(np.zeros(6))[:3, 3] == pytest.approx([-0.81725, -0.19145, -0.00549], abs=1e-4)


def test_every_closed_form_branch_reproduces_the_target():
    rng = np.random.default_rng(7)
    for _ in range(100):
        target = forward_matrix(rng.uniform(-np.pi, np.pi, 6))
        branches = analytic_solutions(target)
        assert branches
        for branch in branches:
            assert np.abs(forward_matrix(branch) - target).max() < 1e-8


def test_solve_stays_on_the_seed_branch_along_a_path():
    ik = UR5IK()
    joints = HOME.copy()
    for step in range(40):
        previous = joints
        joints = previous + np.array([0.01, 0.005, -0.004, 0.006, 0.0, -0.01])
        solved = ik.solve(ik.forward_kinematics(joints), previous, 1, step)
        assert np.abs(solved - joints).max() < 1e-4  # no elbow/wrist flip


def test_solve_honours_joint_limits_and_reports_unreachable_targets():
    ik = UR5IK(joint_limits=[[-0.2, 0.2]] + [[-6.28, 6.28]] * 5)
    target = ik.forward_kinematics(HOME + [1.0, 0, 0, 0, 0, 0])
    with pytest.raises(TrajectoryIKError):
        ik.solve(target, HOME, 1, 1)
    far = np.r_[2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0][:6]
    with pytest.raises(TrajectoryIKError):
        UR5IK().solve(far, HOME, 1, 1)


def test_opening_maps_to_robotiq_register():
    assert (opening_to_raw(1.0), opening_to_raw(0.0), opening_to_raw(0.5)) == (0, 255, 128)
    with pytest.raises(ValueError):
        opening_to_raw(1.2)


class FakeReceive:
    def __init__(self, q):
        self.q, self.time, self.estop, self.pstop = np.asarray(q, float), 1000.0, False, False

    def isConnected(self):
        return True

    def getActualQ(self):
        return self.q.tolist()

    def getActualQd(self):
        return [0.0] * 6

    def getTimestamp(self):
        self.time += 0.008
        return self.time

    def getActualTCPPose(self):
        return rtde_pose(self.q)

    def getJointTemperatures(self):
        return [30.0] * 6

    def isEmergencyStopped(self):
        return self.estop

    def isProtectiveStopped(self):
        return self.pstop

    def getRobotMode(self):
        return 7

    def disconnect(self):
        pass


class FakeControl:
    def __init__(self, receive, kinematics_offset=0.0):
        self.receive, self.offset = receive, kinematics_offset
        self.scripts, self.servo_calls, self.stops = [], 0, 0

    def isConnected(self):
        return True

    def isProgramRunning(self):
        return True

    def servoJ(self, q, velocity, acceleration, dt, lookahead, gain):
        assert (velocity, acceleration, dt) == (0.0, 0.0, 1 / 125) and 0.03 <= lookahead <= 0.2
        self.receive.q = np.asarray(q, float)
        self.servo_calls += 1
        return True

    def servoStop(self):
        self.stops += 1

    def sendCustomScriptFunction(self, name, text):
        self.scripts.append((name, text))
        return True

    def getForwardKinematics(self, q, tcp_offset):
        pose = rtde_pose(q)
        pose[2] += self.offset
        return pose

    def stopScript(self):
        pass

    def disconnect(self):
        pass


IDENTITY = np.eye(4).tolist()
SETTINGS = {
    "calibration": {"link6_from_sdk_eef": IDENTITY, "link6_from_tcp": IDENTITY},
    "runtime": {"trajectory_hz": 125.0},
    "robot": {"gripper_kind": "none", "gripper_width_m": 0.085},
    "motion": {"tcp_velocity_m_s": 0.5, "tcp_angular_velocity_rad_s": 2.0, "joint_velocity_rad_s": 1.5,
               "joint_acceleration_rad_s2": 6.0, "joint_jerk_rad_s3": 60.0, "settle_timeout_s": 2.0},
}


def build(kinematics_offset=0.0, gripper=None):
    receive = FakeReceive(HOME)
    control = FakeControl(receive, kinematics_offset)
    link = UR5Link("sim", hz=125.0, control=control, receive=receive, receive_factory=lambda: receive)
    return UR5Robot("sim", SETTINGS, link=link, gripper=gripper or NoGripper()), control, receive


def test_move_to_streams_servoj_and_reaches_the_target():
    robot, control, receive = build()
    try:
        before = robot.state()["tcp_xyzquat"]
        target = [before[0], before[1], before[2] - 0.03, *before[3:]]
        result = robot.execute("move_to", {"target": {"pose_xyzquat": target}, "note": "descend 3 cm"})
        assert control.servo_calls > 10
        assert result["tcp_xyzquat"][2] == pytest.approx(before[2] - 0.03, abs=2e-3)
        assert result["execution_feedback"]["settle"]["settled"]
    finally:
        robot.close()


def test_check_path_sends_no_commands():
    robot, control, _ = build()
    try:
        tcp = robot.state()["tcp_xyzquat"]
        robot.execute("check_path", {"poses": [{"pose_xyzquat": [tcp[0], tcp[1], tcp[2] - 0.02, *tcp[3:]]}], "note": "dry run"})
        assert control.servo_calls == 0
    finally:
        robot.close()


def test_set_gripper_uses_the_gripper_and_keeps_the_arm_still():
    robot, control, receive = build(gripper=RobotiqUR(
        UR5Link("sim", hz=125.0, control=(c := FakeControl(FakeReceive(HOME))), receive=c.receive,
                receive_factory=lambda: c.receive),
        "PRE\n", activate=False, settle_s=0))
    try:
        result = robot.execute("set_gripper", {"gripper": 0.25, "note": "grip"})
        assert result["gripper_normalized"] == pytest.approx(0.25)
        assert result["execution_feedback"]["motion"]["raw_position"] == 191
        assert result["execution_feedback"]["motion"]["feedback"] == "open_loop_last_command"
        assert c.scripts == [("ROBOTIQ_MOVE", "PRE\nrq_move_and_wait(191)")]
        assert c.stops >= 1 and control.servo_calls == 0
    finally:
        robot.close()


def test_protective_stop_latches_a_fault():
    robot, _, receive = build()
    try:
        receive.pstop = True
        with pytest.raises(MotionFault):
            robot.state()
        with pytest.raises(MotionFault):
            robot.execute("move_to", {"target": {"pose_xyzquat": [0, 0, 0, 0, 0, 0, 1]}, "note": "x"})
    finally:
        robot.close()


def test_startup_rejects_a_robot_whose_kinematics_differ():
    with pytest.raises(ValueError, match="kinematics disagree"):
        build(kinematics_offset=0.02)
