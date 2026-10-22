"""The UR5 bring-up check reports real problems and sends nothing to the robot."""
import json

import numpy as np
import pytest

from gpt_policy.hardware import ur5_check as check
from gpt_policy.hardware.ur5_ik import forward_matrix

HOME = np.array([0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
IDENTITY = np.eye(4).tolist()


def rotvec(matrix):
    angle = np.arccos(np.clip((np.trace(matrix) - 1) / 2, -1, 1))
    axis = np.array([matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]])
    return axis / np.linalg.norm(axis) * angle if np.linalg.norm(axis) > 1e-12 else np.zeros(3)


def rtde(transform):
    return [*transform[:3, 3], *rotvec(transform[:3, :3])]


class FakeReceive:
    def __init__(self, q=HOME, tcp_offset=(0, 0, 0), tilt=0.0, frozen=False):
        self.q, self.offset, self.tilt, self.frozen = np.asarray(q, float), tcp_offset, tilt, frozen
        self.time, self.estop, self.pstop, self.mode, self.safety, self.temps = 500.0, False, False, 7, 1, [35.0] * 6

    def getTimestamp(self):
        if not self.frozen:
            self.time += 0.008
        return self.time

    def isEmergencyStopped(self):
        return self.estop

    def isProtectiveStopped(self):
        return self.pstop

    def getRobotMode(self):
        return self.mode

    def getSafetyMode(self):
        return self.safety

    def getActualQ(self):
        return self.q.tolist()

    def getJointTemperatures(self):
        return self.temps

    def getActualTCPPose(self):
        transform = forward_matrix(self.q)
        shift = np.eye(4)
        shift[:3, 3] = self.offset
        c, s = np.cos(self.tilt), np.sin(self.tilt)
        shift[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        return rtde(transform @ shift)


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def settings(**extra):
    base = {"backend": "ur5", "runtime": {"interface": "10.0.0.2"}, "robot": {"gripper_kind": "none"},
            "calibration": {"link6_from_sdk_eef": IDENTITY, "link6_from_tcp": IDENTITY}}
    base.update(extra)
    return base


def run(receive, **kwargs):
    clock = Clock()
    return {r.name: r for r in check.check_rtde(settings(), receive, 1.0, clock=clock, sleep=clock.sleep)}


def test_healthy_robot_passes_every_rtde_check():
    results = run(FakeReceive())
    assert {r.status for r in results.values()} == {"PASS"}
    assert "kinematics vs controller" in results and "IK round trip" in results
    assert float(results["RTDE feed"].detail.split()[0]) >= 100


def test_nonzero_controller_tcp_is_a_warning_with_the_measured_offset():
    result = run(FakeReceive(tcp_offset=(0, 0, 0.15)))["kinematics vs controller"]
    assert result.status == "WARN" and "0.1500" in result.detail


def test_rotated_controller_tcp_or_wrong_model_fails():
    result = run(FakeReceive(tilt=0.3))["kinematics vs controller"]
    assert result.status == "FAIL" and "zero on the pendant" in result.detail


def test_stuck_feed_fails_immediately():
    results = check.check_rtde(settings(), FakeReceive(frozen=True), 1.0, clock=(c := Clock()), sleep=c.sleep)
    assert [r.status for r in results] == ["FAIL"] and "did not advance" in results[0].detail


def test_stops_limits_and_temperature_are_reported():
    receive = FakeReceive(q=HOME + [7.0, 0, 0, 0, 0, 0])  # shoulder beyond +-2*pi
    receive.pstop, receive.mode, receive.temps = True, 5, [85.0] * 6
    results = run(receive)
    assert results["protective stop"].status == "FAIL"
    assert results["joint limits"].status == "FAIL"
    assert results["robot mode"].status == "WARN"
    assert results["joint temperatures"].status == "WARN"


class FakeSocket:
    def __init__(self, replies):
        self.replies, self.sent = list(replies), []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def makefile(self, *args, **kwargs):
        outer = self

        class Reader:
            def readline(self):
                return outer.replies.pop(0) + "\n"
        return Reader()

    def sendall(self, data):
        self.sent.append(data.decode().strip())


def test_dashboard_reads_only_queries_and_recognises_cb3():
    sock = FakeSocket(["Connected: Universal Robots Dashboard Server", "URSoftware 3.13.0.10253 (Mar 22 2020)",
                       "Robotmode: RUNNING", "Safetystatus: NORMAL"])
    results = check.check_dashboard("10.0.0.2", connect=lambda *a, **k: sock)
    assert [r.status for r in results] == ["PASS"] * 3 and "CB3" in results[0].detail
    assert sock.sent == ["PolyscopeVersion", "robotmode", "safetystatus"]


def test_dashboard_flags_eseries_and_unreachable_robots():
    sock = FakeSocket(["hi", "5.9.4.10151470 (Aug 2 2022)", "Robotmode: IDLE", "Safetystatus: NORMAL"])
    results = check.check_dashboard("x", connect=lambda *a, **k: sock)
    assert results[0].status == "WARN" and results[1].status == "WARN"

    def refuse(*args, **kwargs):
        raise OSError("timed out")
    assert check.check_dashboard("x", connect=refuse)[0].status == "FAIL"


class FakeProbe:
    def __init__(self, frame, intrinsics):
        self.frame, self._intrinsics, self.closed = frame, intrinsics, False

    def capture(self):
        return self.frame

    def intrinsics(self):
        return self._intrinsics

    def close(self):
        self.closed = True


DEVICE = {"fx": 380.5, "fy": 380.2, "cx": 320.1, "cy": 240.4, "distortion": [0.0] * 5, "model": "brown"}


def camera_results(frame, vision=None, tmp=None, **kwargs):
    probe = FakeProbe(frame, DEVICE)
    cfg = settings(cameras={"left": {"serial": "123"}}, vision=vision or {})
    results = check.check_camera(cfg, "left", probe_factory=lambda *a: probe, save_frame=tmp, **kwargs)
    assert probe.closed
    return {r.name: r for r in results}


def test_camera_placeholder_intrinsics_print_the_device_values():
    frame = np.random.default_rng(0).integers(40, 200, (48, 64, 3), dtype=np.uint8)
    results = camera_results(frame, {"camera_intrinsics": {"left": np.eye(3).tolist()}})
    assert results["camera left frame"].status == "PASS"
    assert results["camera intrinsics"].status == "WARN" and "380.5" in results["camera intrinsics"].detail


def test_camera_malformed_intrinsics_are_reported_not_raised():
    frame = np.full((48, 64, 3), 100, np.uint8)
    frame[:, ::2] = 150
    result = camera_results(frame, {"camera_intrinsics": {"left": IDENTITY[:3]}})["camera intrinsics"]
    assert result.status == "WARN" and "expected 3x3" in result.detail


def test_camera_matching_intrinsics_pass_and_black_frame_warns(tmp_path):
    intrinsics = [[380.5, 0, 320.1], [0, 380.2, 240.4], [0, 0, 1]]
    results = camera_results(np.zeros((48, 64, 3), np.uint8), {"camera_intrinsics": {"left": intrinsics}}, tmp_path / "f.png")
    assert results["camera intrinsics"].status == "PASS"
    assert results["camera left frame"].status == "WARN" and (tmp_path / "f.png").is_file()


def test_camera_missing_serial_or_open_failure_fails():
    assert check.check_camera(settings(cameras={"left": {"serial": "REPLACE_ME"}}))[0].status == "FAIL"

    def broken(*args):
        raise RuntimeError("No device connected")
    result = check.check_camera(settings(cameras={"left": {"serial": "1"}}), probe_factory=broken)[0]
    assert result.status == "FAIL" and "No device" in result.detail


class FakeLink:
    def __init__(self):
        self.scripts, self.closed = [], False

    def run_script(self, name, text):
        self.scripts.append((name, text))
        return True

    def close(self):
        self.closed = True


def test_gripper_cycle_sends_three_moves_and_closes_the_link(monkeypatch):
    monkeypatch.setattr("gpt_policy.hardware.robotiq_ur.time.sleep", lambda s: None)
    link = FakeLink()
    results = check.run_gripper_test(settings(), link_factory=lambda: link, preamble_loader=lambda path: "PRE\n")
    moves = [text for name, text in link.scripts if name == "ROBOTIQ_MOVE"]
    assert results[0].status == "PASS" and link.closed
    assert moves == ["PRE\nrq_move_and_wait(0)", "PRE\nrq_move_and_wait(128)", "PRE\nrq_move_and_wait(0)"]


def preamble_file(tmp_path):
    path = tmp_path / "preamble.py"
    path.write_text('ROBOTIQ_PREAMBLE = "def rq_move_and_wait(pos):\\n  sleep(0)\\nend\\n"\n')
    return path


def test_config_check_flags_placeholders_but_not_a_good_profile(tmp_path):
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    config = tmp_path / "machine.json"
    config.write_text(json.dumps({
        "extends": str(root / "configs/examples/ur5-local.json"),
        "robot": {"robotiq_preamble": str(preamble_file(tmp_path))}}))
    from gpt_policy.settings import load_settings
    results = {r.name: r for r in check.check_config(load_settings(config), config)}
    assert results["configuration"].status == "PASS" and results["Robotiq preamble"].status == "PASS"
    warning = results["calibration placeholders"]
    assert warning.status == "WARN" and "hand-eye" in warning.detail and "serial" in warning.detail


def test_missing_preamble_and_robot_ip_are_failures(tmp_path):
    cfg = settings(robot={"gripper_kind": "robotiq", "robotiq_preamble": str(tmp_path / "nope.py")},
                   runtime={"interface": "REPLACE_WITH_ROBOT_IP"})
    results = {r.name: r for r in check.check_config(cfg, tmp_path / "x.json")}
    assert results["Robotiq preamble"].status == "FAIL" and results["robot IP"].status == "FAIL"


def test_cli_offline_mode_opens_no_connection_and_reports_exit_code(tmp_path, capsys, monkeypatch):
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    config = tmp_path / "machine.json"
    config.write_text(json.dumps({
        "extends": str(root / "configs/examples/ur5-local.json"),
        "robot": {"robotiq_preamble": str(preamble_file(tmp_path))}}))
    monkeypatch.setattr(check.socket, "create_connection", lambda *a, **k: pytest.fail("network opened"))
    code = check.main(["--config", str(config), "--no-network", "--no-camera"])
    out = capsys.readouterr().out
    assert code == 0 and "No arm motion command was sent" in out and "[WARN] calibration placeholders" in out
