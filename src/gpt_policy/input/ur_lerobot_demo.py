"""Convert a UR recorder LeRobot v3 episode into a reviewed GPT-Policy demo bundle.

The source is a dataset written by ``phyai_vla-ur7`` (free-drive recording):
``observation.state`` / ``action`` are 7-vectors (six UR joints in radians plus a
binary gripper, ``1`` = closed) with one or more video cameras. The output is a
``demo.json`` bundle that ``gpt-policy --demo`` loads without a model call.

No model or vision selector runs here. Keyframes are chosen from the numeric
trajectory, and their stage labels are mechanical; edit ``demo.json`` to add
human-reviewed descriptions before relying on it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..geometry.frames import FrameCalibration
from ..geometry.poses import matrix_to_pose, rpy_to_quaternion
from ..hardware.ur5_ik import forward_matrix
from ..settings import load_settings
from .action_sampling import gripper_events

PLACEHOLDER_TASKS = {"task description in here"}
FROZEN_RANGE_RAD = 0.01  # Same threshold as the recorder's own frozen-feed check.
MAX_KEYFRAMES = 24
MAX_IMAGES = 48


@dataclass
class Episode:
    root: Path
    index: int
    task: str
    fps: float
    robot_type: str
    state: np.ndarray            # (T, 7)
    action: np.ndarray           # (T, 7)
    dataset_index: np.ndarray    # (T,)
    videos: dict                 # video key -> (mp4 path, from_timestamp_s)
    sources: dict                # relative path -> sha256

    @property
    def times(self) -> np.ndarray:
        return np.arange(len(self.state)) / self.fps


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _parquet():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Reading LeRobot datasets needs pyarrow: pip install 'gpt-policy[lerobot]'") from exc
    return pq


def load_episode(root, episode: int, video_keys=None) -> Episode:
    pq = _parquet()
    root = Path(root).expanduser().resolve()
    info = json.loads((root / "meta/info.json").read_text())
    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"Expected a LeRobot v3.0 dataset, got {info.get('codebase_version')!r}")
    features = info["features"]
    for key in ("observation.state", "action"):
        if tuple(features[key]["shape"]) != (7,):
            raise ValueError(f"{key} must be 7-D (six UR joints + gripper), got {features[key]['shape']}")
    available = [k for k, f in features.items() if f["dtype"] == "video"]
    video_keys = list(video_keys or available)
    if not video_keys or any(k not in available for k in video_keys):
        raise ValueError(f"Dataset has video features {available}; none usable for {video_keys}. "
                         "A demonstration needs images.")

    columns = ["episode_index", "tasks", "length", "data/chunk_index", "data/file_index",
               "dataset_from_index", "dataset_to_index"]
    for key in video_keys:
        columns += [f"videos/{key}/{name}" for name in ("chunk_index", "file_index", "from_timestamp")]
    rows = []
    for path in sorted((root / "meta/episodes").glob("chunk-*/file-*.parquet")):
        rows += pq.read_table(path, columns=columns, filters=[("episode_index", "=", episode)]).to_pylist()
    if len(rows) != 1:
        raise ValueError(f"Episode {episode} not found exactly once in {root}")
    meta = rows[0]

    data_file = root / info["data_path"].format(chunk_index=meta["data/chunk_index"], file_index=meta["data/file_index"])
    table = pq.read_table(data_file, filters=[("episode_index", "=", episode)]).to_pydict()
    state = np.asarray(table["observation.state"], dtype=float)
    action = np.asarray(table["action"], dtype=float)
    frame_index = np.asarray(table["frame_index"])
    dataset_index = np.asarray(table["index"])
    length = int(meta["length"])
    if len(state) != length or not np.array_equal(frame_index, np.arange(length)):
        raise ValueError("Episode rows do not match its metadata length/frame order")
    if (dataset_index[0] != meta["dataset_from_index"] or dataset_index[-1] + 1 != meta["dataset_to_index"]):
        raise ValueError("Episode dataset index bounds do not match its metadata")
    fps = float(info["fps"])
    if not np.allclose(np.asarray(table["timestamp"], dtype=float), frame_index / fps, atol=1e-3):
        raise ValueError("Episode timestamps do not follow frame_index / fps")
    if not (np.isfinite(state).all() and np.isfinite(action).all()):
        raise ValueError("Non-finite state/action in episode")

    videos, sources = {}, {str(data_file.relative_to(root)): _sha256(data_file)}
    for key in video_keys:
        path = root / info["video_path"].format(
            video_key=key, chunk_index=meta[f"videos/{key}/chunk_index"], file_index=meta[f"videos/{key}/file_index"])
        if not path.is_file():
            raise FileNotFoundError(path)
        videos[key] = (path, float(meta[f"videos/{key}/from_timestamp"]))
        sources[str(path.relative_to(root))] = _sha256(path)
    tasks = meta["tasks"]
    return Episode(root, episode, str(tasks[0]) if len(tasks) == 1 else " / ".join(map(str, tasks)),
                   fps, str(info.get("robot_type", "ur")), state, action, dataset_index, videos, sources)


def tcp_poses(joints: np.ndarray, frames: FrameCalibration) -> np.ndarray:
    """Per-frame ``[x, y, z, qx, qy, qz, qw]`` of the calibrated fingertip TCP in the UR base frame."""
    result, previous = [], None
    for q in joints:
        tcp = frames.sdk_to_tcp(matrix_to_pose(forward_matrix(q)))
        quaternion = np.asarray(rpy_to_quaternion(tcp[3:]))
        # q and -q are one rotation; keep the sequence continuous.
        if (previous is None and quaternion[3] < 0) or (previous is not None and np.dot(previous, quaternion) < 0):
            quaternion = -quaternion
        previous = quaternion
        result.append(np.r_[tcp[:3], quaternion])
    return np.asarray(result)


def _quaternion_angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 2 * np.arccos(np.clip(np.abs(np.sum(a * b, axis=-1)), 0, 1))


def _slerp(a: np.ndarray, b: np.ndarray, fraction: np.ndarray) -> np.ndarray:
    dot = float(np.dot(a, b))
    if dot < 0:
        b, dot = -b, -dot
    theta = math.acos(min(1.0, dot))
    if theta < 1e-8:
        return np.repeat(a[None], len(fraction), axis=0)
    return (np.sin((1 - fraction) * theta)[:, None] * a + np.sin(fraction * theta)[:, None] * b) / math.sin(theta)


def geometry_anchors(poses: np.ndarray, gripper_command: np.ndarray, times: np.ndarray,
                     position_m=0.005, rotation_rad=np.pi / 36) -> list[int]:
    """Frames needed to keep the route's turns, lifts and gripper changes.

    Recursive pivot search (Douglas-Peucker style) on position and orientation
    against the straight-line/slerp interpolation. Tolerances describe how
    faithfully the context reproduces the demonstration, not control accuracy.
    """
    count = len(poses)
    keep = {0, count - 1}
    samples = [{"left": {"gripper_command": float(g)}} for g in gripper_command]
    keep |= gripper_events(samples, ("gripper_command",), 0.1)
    pending = list(zip(sorted(keep), sorted(keep)[1:]))
    while pending:
        lo, hi = pending.pop()
        if hi - lo < 2:
            continue
        fraction = ((times[lo + 1:hi] - times[lo]) / (times[hi] - times[lo]))
        predicted = poses[lo, :3] + fraction[:, None] * (poses[hi, :3] - poses[lo, :3])
        position = np.linalg.norm(poses[lo + 1:hi, :3] - predicted, axis=1) / position_m
        rotation = _quaternion_angle(_slerp(poses[lo, 3:], poses[hi, 3:], fraction), poses[lo + 1:hi, 3:]) / rotation_rad
        error = np.maximum(position, rotation)
        pivot = int(np.argmax(error))
        if error[pivot] > 1:
            mid = lo + 1 + pivot
            keep.add(mid)
            pending.extend(((lo, mid), (mid, hi)))
    return sorted(keep)


def select_keyframes(joints: np.ndarray, opening: np.ndarray, count: int) -> tuple[list[int], dict[int, str]]:
    """Endpoints, both sides of every gripper change, then frames spread along the joint path."""
    total = len(joints)
    if not 2 <= count <= MAX_KEYFRAMES:
        raise ValueError(f"keyframes must be between 2 and {MAX_KEYFRAMES}")
    labels = {0: "start", total - 1: "end"}
    events = []
    for i in range(1, total):
        was, now = opening[i - 1] > 0.5, opening[i] > 0.5
        if was != now:
            events.append((i, "gripper opens" if now else "gripper closes"))
    required = {0, total - 1}
    slots = count - len(required)
    pairs = [(i - 1, i, name) for i, name in events]
    while pairs and 2 * len(pairs) > slots:  # too many changes: drop from the middle, keep first and last
        pairs.pop(len(pairs) // 2)
    for before, after, name in pairs:
        required |= {before, after}
        labels.setdefault(before, f"before: {name}")
        labels[after] = f"after: {name}"
    progress = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(joints, axis=0), axis=1))]
    chosen = sorted(required)
    while len(chosen) < count:
        gap = np.min(np.abs(progress[:, None] - progress[chosen][None, :]), axis=1)
        gap[chosen] = -1
        best = int(np.argmax(gap))
        if gap[best] <= 1e-9:
            break
        chosen = sorted([*chosen, best])
    for i in chosen:
        share = progress[i] / progress[-1] if progress[-1] > 0 else 0.0
        labels.setdefault(i, f"transit, {share:.0%} of joint path")
    return chosen, labels


def extract_frame(video: Path, seconds: float, destination: Path, ffmpeg="ffmpeg") -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([ffmpeg, "-v", "error", "-y", "-ss", f"{seconds:.4f}", "-i", str(video),
                    "-frames:v", "1", "-q:v", "2", str(destination)], check=True)
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg produced no frame at {seconds:.3f}s from {video}")


def _arm_sample(ep: Episode, poses, opening, i, include_targets):
    sample = {
        "joint_measured_rad": ep.state[i, :6].tolist(),
        "eef_measured_xyz_xyzw": poses[i].tolist(),
        "gripper_command": float(opening[i]),
    }
    if include_targets:
        sample["joint_target_rad"] = ep.action[i, :6].tolist()
    return sample


def build_bundle(ep: Episode, frames: FrameCalibration, destination: Path, *, config_label: str,
                 camera_names: dict, keyframes=12, task=None, outcome="unverified",
                 robot_model=None, allow_frozen=False, ffmpeg="ffmpeg") -> dict:
    """Write ``demo.json`` and images into ``destination`` (must not exist). Returns a report."""
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    title = (task or ep.task).strip()
    if not task and title in PLACEHOLDER_TASKS:
        raise ValueError("The dataset's task text is the recorder placeholder; pass --task with the real instruction")
    joint_range = float(np.max(np.ptp(ep.state[:, :6], axis=0)))
    if joint_range < FROZEN_RANGE_RAD and not allow_frozen:
        raise ValueError(
            f"Episode {ep.index}: arm joints moved only {joint_range:.4f} rad; the RTDE feed was probably frozen "
            "while recording. Use another episode, or --allow-frozen to convert anyway.")
    if len(camera_names) * keyframes > MAX_IMAGES:
        raise ValueError(f"{len(camera_names)} cameras x {keyframes} keyframes exceeds {MAX_IMAGES} images")

    opening = 1.0 - ep.state[:, 6]  # Recorder: 1 = closed. Here: 0 closed, 1 open.
    include_targets = bool(np.max(np.abs(ep.action[:, :6] - ep.state[:, :6])) > 1e-6)
    poses = tcp_poses(ep.state[:, :6], frames)
    times = ep.times
    chosen, labels = select_keyframes(ep.state[:, :6], opening, keyframes)
    anchors = set(geometry_anchors(poses, opening, times))

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".ur-demo-", dir=destination.parent))
    try:
        entries = []
        for n, i in enumerate(chosen):
            images = {}
            for key, name in camera_names.items():
                video, offset = ep.videos[key]
                relative = f"images/kf-{n:02d}-{name}.jpg"
                # ffmpeg -ss drops frames earlier than the seek point; land a quarter frame
                # before frame i's own timestamp so exactly frame i is the first one kept.
                extract_frame(video, max(0.0, offset + (i - 0.25) / ep.fps), temp / relative, ffmpeg)
                images[name] = relative
            entry = {
                "t_s": float(times[i]), "source_frame_index": i, "stage": labels[i], "images": images,
                "state": {"left": _arm_sample(ep, poses, opening, i, include_targets)},
                "alignment": {
                    "method": "LeRobot episode frame index; video frame at from_timestamp + frame_index / fps",
                    "episode_index": ep.index, "source_frame_index": i, "dataset_index": int(ep.dataset_index[i]),
                    "timestamp_s": float(times[i]), "hardware_synchronized": False},
            }
            if n + 1 < len(chosen):
                end = chosen[n + 1]
                span = range(i, end + 1)
                entry["action"] = {
                    "kind": "historical_measured_state_sequence", "time_basis": "episode_elapsed_s",
                    "samples": [{"t_s": float(times[j]), "source_frame_index": j,
                                 "left": _arm_sample(ep, poses, opening, j, include_targets)} for j in span],
                    "preserve_sample_indices": [k for k, j in enumerate(span) if j in anchors]}
            entries.append(entry)
        demo = {
            "schema_version": 1,
            "title": title,
            "source": f"LeRobot v3 dataset {ep.root}, episode {ep.index}",
            "demonstrator": "human_freedrive",
            "coverage": "sampled",
            "outcome": outcome,
            "summary": (f"{len(ep.state)} frames at {ep.fps:g} Hz over {times[-1]:.1f} s; "
                        f"{len(chosen)} keyframes chosen from the joint path and gripper changes. "
                        "Stage labels are mechanical, not reviewed descriptions."),
            "source_robot": {"model": robot_model or ep.robot_type, "controller": "UR CB3 via RTDE",
                             "calibration": config_label},
            "coordinate_frame": (
                "UR base frame as reported by RTDE, metres, radians, quaternion xyzw. TCP is the calibrated fingertip "
                f"frame from {config_label} (link6_from_tcp) computed by forward kinematics from the recorded joints. "
                "Valid for the live robot only if its base has not moved relative to the scene. Joints are ordered "
                "base to wrist 3. Gripper normalized: 0 closed, 1 open. It was recorded as a binary open-loop Robotiq "
                "command with no finger feedback, so it shows the commanded state, not a measured opening or a grasp. "
                "Samples are recorded joint states, not issued controls."),
            "numeric_reference_semantics": "recorded_joint_states_with_commanded_gripper",
            "keyframes": entries,
        }
        (temp / "demo.json").write_text(json.dumps(demo, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        (temp / "conversion.json").write_text(json.dumps({
            "source_sha256": ep.sources, "episode": ep.index, "fps": ep.fps, "keyframe_frames": chosen,
            "geometry_anchor_frames": sorted(anchors), "joint_range_rad": joint_range,
            "action_differs_from_state": include_targets, "calibration": config_label,
            "cameras": camera_names}, indent=2) + "\n")
        temp.rename(destination)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return {"output": str(destination), "task": title, "frames": len(ep.state), "keyframes": len(chosen),
            "geometry_anchors": len(anchors), "joint_range_rad": joint_range,
            "stages": [labels[i] for i in chosen]}


def parse_episodes(spec: str) -> list[int]:
    result = []
    for part in spec.split(","):
        lo, _, hi = part.strip().partition("-")
        result += range(int(lo), int(hi or lo) + 1)
    if not result:
        raise ValueError("No episodes selected")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path, help="LeRobot v3 dataset directory written by the UR recorder")
    parser.add_argument("--config", type=Path, required=True,
                        help="Machine profile whose calibration.link6_from_tcp matches the live UR5")
    parser.add_argument("--output", type=Path, required=True, help="New directory; one episode-NNNN subdirectory per episode")
    parser.add_argument("--episodes", default="0", help="Comma list/ranges, e.g. 0,3-5 (default 0)")
    parser.add_argument("--task", help="Instruction text; required when the dataset kept the recorder placeholder")
    parser.add_argument("--keyframes", type=int, default=12, help=f"Keyframes per episode, 2-{MAX_KEYFRAMES} (default 12)")
    parser.add_argument("--camera", action="append", metavar="KEY=NAME",
                        help="Video feature to include as camera NAME (repeatable). Default: every video feature; "
                             "a single camera is named 'left' to match the wrist camera in the machine profile.")
    parser.add_argument("--outcome", choices=("success", "failed", "unverified"), default="unverified")
    parser.add_argument("--robot-model", help="Overrides the dataset's robot_type in the bundle")
    parser.add_argument("--allow-frozen", action="store_true", help="Convert episodes whose joints barely moved")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    frames = FrameCalibration(settings)
    episodes = parse_episodes(args.episodes)
    if args.output.exists():
        parser.error(f"{args.output} already exists")
    multiple = len(episodes) > 1
    exit_code = 0
    for index in episodes:
        keys = [pair.split("=", 1)[0] for pair in args.camera] if args.camera else None
        episode = load_episode(args.dataset, index, keys)
        if args.camera:
            names = dict(pair.split("=", 1) for pair in args.camera)
        elif len(episode.videos) == 1:
            names = {next(iter(episode.videos)): "left"}
        else:
            names = {key: key.rsplit(".", 1)[-1] for key in episode.videos}
        destination = args.output / f"episode-{index:04d}" if multiple else args.output
        try:
            report = build_bundle(
                episode, frames, destination, config_label=str(args.config), camera_names=names,
                keyframes=args.keyframes, task=args.task, outcome=args.outcome,
                robot_model=args.robot_model, allow_frozen=args.allow_frozen, ffmpeg=args.ffmpeg)
        except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as exc:
            print(f"episode {index}: skipped: {exc}")
            exit_code = 1
            continue
        print(f"episode {index}: {report['output']}  ({report['keyframes']} keyframes, "
              f"{report['geometry_anchors']} geometry anchors, joint range {report['joint_range_rad']:.2f} rad)")
        print(f"  gpt-policy \"<task>\" --demo {report['output']}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
