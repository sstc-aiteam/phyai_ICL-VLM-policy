"""UR5 kinematics and adapter behaviour against a simulated RTDE link."""
import numpy as np
import pytest

from gpt_policy.geometry.poses import matrix_to_pose, pose_to_matrix
from gpt_policy.hardware.motion_control import MotionFault
from gpt_policy.hardware.robotiq_ur import NoGripper, RobotiqUR, opening_to_raw
from gpt_policy.hardware.ur5 import UR5Link, UR5Robot
from gpt_policy.hardware.ur5_ik import NOMINAL_CHAIN, UR5IK, Chain, analytic_solutions, chain_from_settings, forward_matrix
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


# A real CB3 arm's factory calibration; joints 2/3 use UR's redundant large-d2/d3 parameterization.
CALIBRATION = {
    "dh_theta": [1.4046709602427362e-05, -1.0667405705515913, 1.0597476489326159, 0.006871914746597391,
                 1.217511230737771e-05, 4.844992693619869e-05],
    "dh_a": [7.105856908529943e-05, -0.20551391937761987, -0.39221713329545993, 7.873746432614875e-05,
             3.9321789037181836e-05, 0.0],
    "dh_d": [0.08922112926639071, 227.85419881246582, -228.1449493293873, 0.40092686041540593,
             0.09480852379818619, 0.08258423411251604],
    "dh_alpha": [1.5704928138799201, -0.0016351448055779852, 0.009384731158707702, 1.5701325921139673,
                 -1.5703626259402443, 0.0],
}


def test_calibrated_chain_stays_within_millimetres_of_nominal_and_nominal_chain_is_identity():
    calibrated = Chain.from_calibration(CALIBRATION)
    assert np.abs(forward_matrix(HOME, NOMINAL_CHAIN) - forward_matrix(HOME)).max() == 0.0
    assert 1e-4 < np.linalg.norm(forward_matrix(HOME, calibrated)[:3, 3] - forward_matrix(HOME)[:3, 3]) < 5e-3


def test_chain_is_read_from_the_robot_settings_and_validated():
    assert chain_from_settings({"robot": {}}) is NOMINAL_CHAIN
    assert chain_from_settings({"robot": {"kinematics_calibration": CALIBRATION}}).d[1] == CALIBRATION["dh_d"][1]
    with pytest.raises(ValueError):
        chain_from_settings({"robot": {"kinematics_calibration": {"dh_theta": [0.0] * 6}}})
    with pytest.raises(ValueError):
        Chain([0.0] * 5, [0.0] * 6, [0.0] * 6, [0.0] * 6)


def test_calibrated_solve_hits_targets_the_nominal_model_would_miss_by_millimetres():
    chain = Chain.from_calibration(CALIBRATION)
    calibrated, nominal = UR5IK(chain=chain), UR5IK()
    joints = HOME.copy()
    for step in range(40):
        previous = joints
        joints = previous + np.array([0.01, 0.005, -0.004, 0.006, 0.0, -0.01])
        target = calibrated.forward_kinematics(joints)
        solved = calibrated.solve(target, previous, 1, step)
        assert np.abs(solved - joints).max() < 1e-3  # same branch, no flip
        assert calibrated.error(target, solved)[0] < 1e-4
    assert nominal.error(target, solved)[0] > 5e-4  # the nominal model really is off for this arm


def test_calibrated_solve_returns_the_seed_for_its_own_pose_even_near_a_wrist_singularity():
    ik = UR5IK(chain=Chain.from_calibration(CALIBRATION))
    seed = np.array([0.0622, -1.6475, 0.1542, -1.5312, -0.0182, -1.6651])
    assert np.abs(ik.solve(ik.forward_kinematics(seed), seed, 0, 0) - seed).max() < 1e-9


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
        self.last_tcp_offset = list(tcp_offset)
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


def test_startup_kinematics_check_never_sends_an_all_zero_tool_offset():
    # CB3 3.13 never answers get_forward_kin with an exactly-zero offset, so the check would hang.
    robot, control, _ = build()
    try:
        assert any(control.last_tcp_offset) and max(abs(v) for v in control.last_tcp_offset) < 1e-5
    finally:
        robot.close()


def test_targets_below_the_fingertip_floor_are_refused_before_any_motion():
    settings = {**SETTINGS, "robot": {**SETTINGS["robot"], "min_tcp_z_m": 0.0}}
    receive = FakeReceive(HOME)
    control = FakeControl(receive, 0.0)
    link = UR5Link("sim", hz=125.0, control=control, receive=receive, receive_factory=lambda: receive)
    robot = UR5Robot("sim", settings, link=link, gripper=NoGripper())
    try:
        tcp = robot.state()["tcp_xyzquat"]
        floor = tcp[2] - 0.02
        robot.min_tcp_z = floor
        low = [tcp[0], tcp[1], floor - 0.005, *tcp[3:]]
        for tool, arguments in (("check_path", {"poses": [{"pose_xyzquat": low}]}),
                                ("move_to", {"target": {"pose_xyzquat": low}}),
                                ("move_eef_chunk", {"poses": [{"pose_xyzquat": tcp}, {"pose_xyzquat": low}]})):
            with pytest.raises(ValueError, match="fingertip floor"):
                robot.execute(tool, {**arguments, "note": "too low"})
        assert control.servo_calls == 0
        ok = [tcp[0], tcp[1], floor + 0.005, *tcp[3:]]
        robot.execute("check_path", {"poses": [{"pose_xyzquat": ok}], "note": "above the floor"})
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
