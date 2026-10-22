"""UR recorder LeRobot episodes become reviewed demo bundles that the loader accepts."""
import json
import shutil
import subprocess

import numpy as np
import pytest
from PIL import Image

pq = pytest.importorskip("pyarrow.parquet")
pa = pytest.importorskip("pyarrow")

from gpt_policy.geometry.frames import FrameCalibration
from gpt_policy.hardware.ur5_ik import forward_matrix
from gpt_policy.input import ur_lerobot_demo as ur
from gpt_policy.input.demonstration import prepare_demonstration
from gpt_policy.input.request import RunInput

HOME = np.array([0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
FPS = 5
KEY = "observation.images.cam_wrist"
IDENTITY = np.eye(4).tolist()
LENGTHS = {0: 12, 1: 15, 2: 6}  # episode 2 is a frozen RTDE feed


def gray(global_frame):
    return 10 + 7 * global_frame


def has_h264():
    if shutil.which("ffmpeg") is None:
        return False
    encoders = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    return "libx264" in encoders


pytestmark = pytest.mark.skipif(not has_h264(), reason="needs ffmpeg with libx264")


def trajectory(episode, length):
    t = np.arange(length)[:, None]
    joints = np.repeat(HOME[None], length, axis=0)
    if episode != 2:
        joints = joints + t * np.array([0.02, 0.01, -0.015, 0.0, 0.01, 0.0])
        joints[length // 2:, 1] -= 0.15  # a lift that a straight chord would miss
    grip = np.zeros(length)
    if episode == 1:
        grip[5:11] = 1.0  # closed command, then released
    return np.c_[joints, grip]


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "ds"
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    rows = {k: [] for k in ("observation.state", "action", "timestamp", "frame_index", "episode_index", "index", "task_index")}
    episodes, start = [], 0
    for episode, length in LENGTHS.items():
        data = trajectory(episode, length)
        for i in range(length):
            rows["observation.state"].append(data[i].tolist())
            rows["action"].append(data[i].tolist())
            rows["timestamp"].append(i / FPS)
            rows["frame_index"].append(i)
            rows["episode_index"].append(episode)
            rows["index"].append(start + i)
            rows["task_index"].append(0)
            Image.new("RGB", (64, 48), (gray(start + i),) * 3).save(frames_dir / f"{start + i:03d}.png")
        episodes.append({
            "episode_index": episode, "tasks": ["Open the lid"], "length": length,
            "data/chunk_index": 0, "data/file_index": 0, "dataset_from_index": start, "dataset_to_index": start + length,
            f"videos/{KEY}/chunk_index": 0, f"videos/{KEY}/file_index": 0,
            f"videos/{KEY}/from_timestamp": start / FPS, f"videos/{KEY}/to_timestamp": (start + length) / FPS})
        start += length
    for path in ("meta/episodes/chunk-000", "data/chunk-000", f"videos/{KEY}/chunk-000"):
        (root / path).mkdir(parents=True)
    (root / "meta/info.json").write_text(json.dumps({
        "codebase_version": "v3.0", "robot_type": "ur5", "fps": FPS,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {"observation.state": {"dtype": "float32", "shape": [7]},
                     "action": {"dtype": "float32", "shape": [7]},
                     KEY: {"dtype": "video", "shape": [48, 64, 3]}}}))
    pq.write_table(pa.Table.from_pylist(episodes), root / "meta/episodes/chunk-000/file-000.parquet")
    pq.write_table(pa.table(rows), root / "data/chunk-000/file-000.parquet")
    subprocess.run(["ffmpeg", "-v", "error", "-framerate", str(FPS), "-i", str(frames_dir / "%03d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "4",
                    str(root / f"videos/{KEY}/chunk-000/file-000.mp4")], check=True)
    return root


def frames(tcp_z=0.0):
    transform = np.eye(4)
    transform[2, 3] = tcp_z
    return FrameCalibration({"calibration": {"link6_from_sdk_eef": IDENTITY, "link6_from_tcp": transform.tolist()}})


def convert(dataset, tmp_path, episode, **kwargs):
    ep = ur.load_episode(dataset, episode)
    return ep, ur.build_bundle(ep, frames(0.15), tmp_path / f"out{episode}", config_label="test.json",
                               camera_names={KEY: "left"}, **kwargs)


def test_keyframe_images_are_exactly_the_source_frames(dataset, tmp_path):
    ep, report = convert(dataset, tmp_path, 1, keyframes=8)
    demo = json.loads((tmp_path / "out1/demo.json").read_text())
    start = LENGTHS[0]
    assert len(demo["keyframes"]) == report["keyframes"] <= 8
    for frame in demo["keyframes"]:
        pixel = np.asarray(Image.open(tmp_path / "out1" / frame["images"]["left"]))[24, 32]
        assert abs(int(pixel[0]) - gray(start + frame["source_frame_index"])) <= 3


def test_last_frame_of_the_last_episode_is_extractable(dataset, tmp_path):
    ep = ur.load_episode(dataset, 2)
    report = ur.build_bundle(ep, frames(), tmp_path / "frozen", config_label="t", camera_names={KEY: "left"},
                             keyframes=3, allow_frozen=True)
    demo = json.loads((tmp_path / "frozen/demo.json").read_text())
    last = demo["keyframes"][-1]
    pixel = np.asarray(Image.open(tmp_path / "frozen" / last["images"]["left"]))[24, 32]
    assert last["source_frame_index"] == LENGTHS[2] - 1
    assert abs(int(pixel[0]) - gray(sum(LENGTHS.values()) - 1)) <= 3


def test_bundle_is_accepted_by_the_demonstration_loader(dataset, tmp_path):
    convert(dataset, tmp_path, 1, keyframes=8)
    prepare_demonstration(RunInput("Open the lid", "test"), tmp_path / "out1", "video+action", tmp_path / "prepared")
    content = json.loads((tmp_path / "prepared/input.json").read_text())["content"]
    header = json.loads(content[1])
    assert header["numeric_reference_semantics"] == "recorded_joint_states_with_commanded_gripper"
    assert ["left", "gripper_command"] in header["action_sample_encoding"]["columns"]
    assert "open_loop" in header["coordinate_frame"] or "open-loop" in header["coordinate_frame"]
    compiled = json.loads((tmp_path / "prepared/demo.json").read_text())
    assert compiled["keyframes"][0]["state"]["left"]["gripper_command"] == 1.0
    # video-only mode must drop the numbers but still load.
    prepare_demonstration(RunInput("Open the lid", "test"), tmp_path / "out1", "video", tmp_path / "prepared-video")


def test_gripper_is_converted_to_open_equals_one_and_events_are_keyframes(dataset, tmp_path):
    ep, report = convert(dataset, tmp_path, 1, keyframes=8)
    demo = json.loads((tmp_path / "out1/demo.json").read_text())
    by_frame = {f["source_frame_index"]: f for f in demo["keyframes"]}
    assert by_frame[0]["state"]["left"]["gripper_command"] == 1.0  # recorded 0 = open
    assert {4, 5, 10, 11} <= set(by_frame)  # both sides of close (frame 5) and open (frame 11)
    assert by_frame[5]["state"]["left"]["gripper_command"] == 0.0
    assert "after: gripper closes" in by_frame[5]["stage"] and "after: gripper opens" in by_frame[11]["stage"]


def test_poses_are_in_the_calibrated_tcp_frame(dataset):
    ep = ur.load_episode(dataset, 1)
    flange = ur.tcp_poses(ep.state[:, :6], frames(0.0))
    tool = ur.tcp_poses(ep.state[:, :6], frames(0.15))
    assert flange[0, :3] == pytest.approx(forward_matrix(ep.state[0, :6])[:3, 3])
    assert np.linalg.norm(tool[:, :3] - flange[:, :3], axis=1) == pytest.approx(0.15)
    assert np.linalg.norm(tool[:, 3:], axis=1) == pytest.approx(1.0)
    assert np.all(np.sum(tool[1:, 3:] * tool[:-1, 3:], axis=1) > 0)  # continuous quaternion sign


def test_frozen_episode_and_placeholder_task_are_refused(dataset, tmp_path):
    ep = ur.load_episode(dataset, 2)
    with pytest.raises(ValueError, match="frozen"):
        ur.build_bundle(ep, frames(), tmp_path / "x", config_label="t", camera_names={KEY: "left"})
    ep.task = "task description in here"
    moving = ur.load_episode(dataset, 1)
    moving.task = "task description in here"
    with pytest.raises(ValueError, match="placeholder"):
        ur.build_bundle(moving, frames(), tmp_path / "y", config_label="t", camera_names={KEY: "left"})
    assert not (tmp_path / "x").exists() and not (tmp_path / "y").exists()
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".ur-demo")]


def test_geometry_anchors_keep_turns_and_gripper_changes():
    times = np.arange(11) / 5
    line = np.zeros((11, 7))
    line[:, 0] = np.linspace(0, 0.5, 11)
    line[:, 6] = 1
    assert ur.geometry_anchors(line, np.ones(11), times) == [0, 10]
    bent = line.copy()
    bent[6:, 0] = bent[6, 0]
    bent[6:, 2] = np.linspace(0, 0.2, 5)  # corner at frame 5/6
    assert {0, 10} < set(ur.geometry_anchors(bent, np.ones(11), times))
    grip = np.r_[np.ones(4), np.zeros(7)]
    assert {3, 4} <= set(ur.geometry_anchors(line, grip, times))


def test_select_keyframes_respects_budget_and_thins_gripper_changes():
    joints = np.cumsum(np.full((60, 6), 0.01), axis=0)
    opening = np.ones(60)
    for start in (10, 20, 30, 40, 50):  # five closes -> too many pairs for a small budget
        opening[start:start + 3] = 0
    chosen, labels = ur.select_keyframes(joints, opening, 6)
    assert chosen[0] == 0 and chosen[-1] == 59 and len(chosen) <= 6
    assert set(chosen) == set(labels) or set(chosen) <= set(labels)


def test_cli_converts_several_episodes_and_reports_skips(dataset, tmp_path, capsys):
    config = tmp_path / "machine.json"
    config.write_text(json.dumps({"calibration": {"link6_from_sdk_eef": IDENTITY, "link6_from_tcp": IDENTITY}}))
    code = ur.main([str(dataset), "--config", str(config), "--output", str(tmp_path / "bundles"),
                    "--episodes", "0-2", "--keyframes", "5"])
    out = capsys.readouterr().out
    assert code == 1 and "episode 2: skipped" in out
    assert (tmp_path / "bundles/episode-0000/demo.json").is_file() and (tmp_path / "bundles/episode-0001/demo.json").is_file()
    assert not (tmp_path / "bundles/episode-0002").exists()
