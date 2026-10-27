"""Wrist-camera hand-eye calibration for the UR5 from a chessboard.

    python -m gpt_policy.hardware.ur5_handeye --config configs/machines/lab-ur5.json --cols 8 --rows 5

Put the arm where the wrist camera sees the board (jog it by hand), then run. By default this only
looks: it detects the board, plans the poses and prints every joint move, and sends nothing.
``--execute`` then moves the arm through the planned poses at low speed, each a small rotation of the
camera about the board with the board kept in view, and captures a frame at each. Clear the workspace
and keep a hand on the e-stop. The solve prints the ``calibration.link6_from_camera`` block to paste
into the profile; it never edits the profile.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..geometry.poses import matrix_to_pose
from ..settings import load_settings
from .ur5_ik import UR5IK, chain_from_settings, forward_matrix

# (tilt about flange x deg, tilt about flange y deg, roll about flange z deg, distance scale)
SCHEDULE = (
    (20, 0, 0, 1.0), (-20, 0, 0, 1.0), (0, 20, 0, 1.0), (0, -20, 0, 1.0),
    (15, 15, 10, 0.9), (-15, 15, -10, 0.9), (15, -15, -10, 1.1), (-15, -15, 10, 1.1),
    (25, 0, 25, 1.0), (-25, 0, -25, 1.0), (0, 25, -25, 0.9), (0, -25, 25, 1.1),
    (10, -10, 30, 1.0), (-10, 10, -30, 1.0), (0, 0, 35, 1.0),
)
MIN_POSES = 8
MAX_REPROJECTION_PX = 1.0


def rotation(axis: int, degrees: float) -> np.ndarray:
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    matrix = np.eye(3)
    i, j = [(1, 2), (2, 0), (0, 1)][axis]
    matrix[i, i], matrix[i, j], matrix[j, i], matrix[j, j] = c, -s, s, c
    return matrix


def rotation_log(matrix: np.ndarray) -> np.ndarray:
    """Axis-angle vector of a rotation matrix."""
    angle = np.arccos(np.clip((np.trace(matrix) - 1) / 2, -1.0, 1.0))
    axis = np.array([matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]])
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.zeros(3)
    return axis / norm * angle


def solve_hand_eye(flanges, boards) -> np.ndarray:
    """``flange_from_camera`` (X) from flange poses F_i and board-in-camera poses C_i, F_i X C_i = const.

    Pairs give A X = X B with A = F_j^-1 F_i and B = C_j C_i^-1; rotation by Kabsch on the
    axis-angle vectors, translation by linear least squares.
    """
    alphas, betas, pairs = [], [], []
    for i in range(len(flanges)):
        for j in range(i + 1, len(flanges)):
            a = np.linalg.inv(flanges[j]) @ flanges[i]
            b = boards[j] @ np.linalg.inv(boards[i])
            if np.linalg.norm(rotation_log(a[:3, :3])) < np.radians(3):
                continue
            alphas.append(rotation_log(a[:3, :3]))
            betas.append(rotation_log(b[:3, :3]))
            pairs.append((a, b))
    if len(pairs) < 3:
        raise ValueError("Not enough distinct camera rotations to solve the hand-eye transform")
    h = np.asarray(betas).T @ np.asarray(alphas)
    u, _, vt = np.linalg.svd(h)
    rotation_x = vt.T @ np.diag([1.0, 1.0, np.linalg.det(vt.T @ u.T)]) @ u.T
    lhs = np.vstack([a[:3, :3] - np.eye(3) for a, _ in pairs])
    rhs = np.concatenate([rotation_x @ b[:3, 3] - a[:3, 3] for a, b in pairs])
    translation = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = rotation_x, translation
    return result


def board_consistency(flanges, boards, flange_from_camera) -> tuple[float, float]:
    """Spread of the board pose in the base frame across poses: (position std mm, rotation std deg)."""
    poses = [f @ flange_from_camera @ c for f, c in zip(flanges, boards)]
    positions = np.array([p[:3, 3] for p in poses])
    reference = poses[0][:3, :3]
    angles = [np.linalg.norm(rotation_log(reference.T @ p[:3, :3])) for p in poses]
    spread = np.sqrt(np.mean(np.sum((positions - positions.mean(axis=0)) ** 2, axis=1)))
    return float(spread * 1000), float(np.degrees(np.sqrt(np.mean(np.square(angles)))))


def plan_poses(start_flange: np.ndarray, pivot_base: np.ndarray, schedule=SCHEDULE) -> list[np.ndarray]:
    """Flange targets that rotate about ``pivot_base`` (the board) and scale the distance to it."""
    plan = []
    for tilt_x, tilt_y, roll, scale in schedule:
        local = rotation(2, roll) @ rotation(1, tilt_y) @ rotation(0, tilt_x)
        world = start_flange[:3, :3] @ local @ start_flange[:3, :3].T
        target = np.eye(4)
        target[:3, :3] = world @ start_flange[:3, :3]
        target[:3, 3] = pivot_base + scale * (world @ (start_flange[:3, 3] - pivot_base))
        plan.append(target)
    return plan


def _pose_vector(matrix: np.ndarray) -> np.ndarray:
    import cv2

    # Rodrigues, not rotation_log: the board is often ~180 deg from the base frame, where the log degenerates.
    return np.r_[cv2.Rodrigues(matrix[:3, :3])[0].ravel(), matrix[:3, 3]]


def _pose_matrix(vector: np.ndarray) -> np.ndarray:
    import cv2

    result = np.eye(4)
    result[:3, :3], result[:3, 3] = cv2.Rodrigues(np.asarray(vector[:3], dtype=float))[0], vector[3:]
    return result


def refine_hand_eye(flanges, corner_sets, camera_matrix, distortion, points, flange_from_camera, camera_from_board,
                    base_from_board=None):
    """Fit the hand-eye transform and the (fixed) board pose to every corner pixel at once.

    With ``base_from_board`` given (for example registered by touching the board's corners), the board
    is held there and only the hand-eye transform is fitted.

    Per-image PnP of a small board is noisy in tilt; this uses the pixels directly. Levenberg-Marquardt
    over [X, base_from_board]. Returns ``(flange_from_camera, rms_px, std)`` where ``std`` is the 1-sigma
    ``(translation mm, rotation deg)`` of X from the Jacobian.
    """
    import cv2

    fixed = base_from_board
    if fixed is None:
        base_from_board = flanges[0] @ flange_from_camera @ camera_from_board
    params = np.r_[_pose_vector(flange_from_camera), _pose_vector(base_from_board)] if fixed is None \
        else _pose_vector(flange_from_camera)
    observed = np.concatenate([np.asarray(c, dtype=float).reshape(-1, 2) for c in corner_sets])

    def residual(vector):
        x = _pose_matrix(vector[:6])
        board = fixed if fixed is not None else _pose_matrix(vector[6:])
        out = []
        for flange in flanges:
            pose = np.linalg.inv(flange @ x) @ board
            out.append(cv2.projectPoints(points, cv2.Rodrigues(pose[:3, :3])[0], pose[:3, 3],
                                         camera_matrix, distortion)[0].reshape(-1, 2))
        return (np.concatenate(out) - observed).ravel()

    def jacobian(vector):
        base = residual(vector)
        columns = []
        for i in range(len(vector)):
            step = np.zeros(len(vector))
            step[i] = 1e-6
            columns.append((residual(vector + step) - base) / 1e-6)
        return np.array(columns).T

    damping = 1e-3
    current = residual(params)
    for _ in range(60):
        j = jacobian(params)
        step = np.linalg.solve(j.T @ j + damping * np.eye(len(params)), -j.T @ current)
        trial = residual(params + step)
        if trial @ trial < current @ current:
            params, current, damping = params + step, trial, damping / 3
            if np.linalg.norm(step) < 1e-10:
                break
        else:
            damping *= 4
    rms = float(np.sqrt(np.mean(current ** 2) * 2))  # per-corner distance, not per-coordinate
    covariance = np.linalg.inv(jacobian(params).T @ jacobian(params)) * np.mean(current ** 2)
    sigma = np.sqrt(np.diag(covariance))
    return _pose_matrix(params[:6]), rms, (float(np.linalg.norm(sigma[3:6]) * 1000), float(np.degrees(np.linalg.norm(sigma[:3]))))


def fit_with_rejection(samples, camera_matrix, distortion, points, max_rms_px=1.5):
    """Corner-pixel fit, dropping the worst pose while it explains most of a poor residual.

    A pose is dropped only if removing it cuts the RMS by 30% or more, and never below MIN_POSES.
    Returns ``(flange_from_camera, rms_px, (sigma_mm, sigma_deg), kept_poses, dropped_poses)``.
    """
    def fit(indices):
        flanges = [np.asarray(samples[i]["flange"]) for i in indices]
        boards = [np.asarray(samples[i]["camera_from_board"]) for i in indices]
        initial = solve_hand_eye(flanges, boards)
        return refine_hand_eye(flanges, [samples[i]["corners"] for i in indices], camera_matrix, distortion, points,
                               initial, boards[0])

    kept, dropped = list(range(len(samples))), []
    result = fit(kept)
    while result[1] > max_rms_px and len(kept) > MIN_POSES:
        trials = [(fit([i for i in kept if i != k]), k) for k in kept]
        best, worst = min(trials, key=lambda item: item[0][1])
        if best[1] > 0.7 * result[1]:
            break
        kept.remove(worst)
        dropped.append(samples[worst]["pose"])
        result = best
    return (*result, [samples[i]["pose"] for i in kept], dropped)


def board_object_points(cols: int, rows: int, square_m: float) -> np.ndarray:
    grid = np.zeros((rows * cols, 3))
    grid[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_m
    return grid


def detect_board(rgb: np.ndarray, cols: int, rows: int):
    import cv2

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    found, corners = cv2.findChessboardCornersSB(gray, (cols, rows), cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
    return corners if found else None


def board_pose(corners, camera_matrix, distortion, cols, rows, square_m) -> tuple[np.ndarray, float]:
    """``camera_from_board`` (OpenCV optical frame) and the RMS reprojection error in pixels."""
    import cv2

    points = board_object_points(cols, rows, square_m)
    corners = np.asarray(corners, dtype=np.float32).reshape(-1, 2)  # OpenCV 5 returns (N, 2), older (N, 1, 2)
    found, rvec, tvec = cv2.solvePnP(points, corners, camera_matrix, distortion, flags=cv2.SOLVEPNP_IPPE)
    if not found:
        raise ValueError("solvePnP failed")
    rvec, tvec = cv2.solvePnPRefineLM(points, corners, camera_matrix, distortion, rvec, tvec)
    projected = cv2.projectPoints(points, rvec, tvec, camera_matrix, distortion)[0].reshape(-1, 2)
    error = float(np.sqrt(np.mean(np.sum((projected - corners) ** 2, axis=1))))
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = cv2.Rodrigues(rvec)[0], tvec.ravel()
    return transform, error


def _intrinsics(settings, name):
    vision = settings.get("vision", {})
    return (np.asarray(vision["camera_intrinsics"][name], dtype=float),
            np.asarray(vision.get("distortion_coefficients", {}).get(name, np.zeros(5)), dtype=float))


def _describe(index, q, start_q, flange, start_flange):
    delta = np.degrees(np.abs(q - start_q)).max()
    shift = np.linalg.norm(flange[:3, 3] - start_flange[:3, 3]) * 100
    return (f"  pose {index:2d}: max joint move {delta:5.1f} deg, flange moves {shift:5.1f} cm, "
            f"flange z {flange[2, 3]:.3f} m")


def build_plan(settings, ik, start_q, camera_from_board, args, flange_from_camera):
    start = forward_matrix(start_q, ik.chain)
    pivot = (start @ flange_from_camera @ camera_from_board)[:3, 3]
    floor = args.min_flange_z
    joints, kept, notes = [], [], []
    for number, target in enumerate(plan_poses(start, pivot), 1):
        reason = None
        if np.linalg.norm(target[:3, 3] - start[:3, 3]) > args.max_translation_m:
            reason = f"moves more than {args.max_translation_m * 100:.0f} cm from the start"
        elif target[2, 3] < floor:
            reason = f"flange would go below z = {floor:.3f} m"
        else:
            try:
                q = ik.solve(matrix_to_pose(target), start_q, number, 0)
            except Exception as exc:  # TrajectoryIKError: unreachable or outside the joint limits
                reason = f"no IK solution ({exc})"
            else:
                if np.abs(q - start_q).max() > args.max_joint_move_rad:
                    reason = f"a joint would move more than {np.degrees(args.max_joint_move_rad):.0f} deg"
        if reason:
            notes.append(f"  pose {number:2d}: skipped, {reason}")
        else:
            joints.append(q)
            kept.append(number)
            notes.append(_describe(number, q, start_q, target, start))
    return joints, kept, notes


def capture_session(settings, args, execute: bool):
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface

    from .ur5_check import RealSenseProbe

    robot, runtime = settings.get("robot", {}), settings.get("runtime", {})
    chain = chain_from_settings(settings)
    ik = UR5IK(settings.get("motion", {}), robot.get("joint_limits_rad"), chain)
    camera_matrix, distortion = _intrinsics(settings, args.camera)
    square_m = args.square_mm / 1000.0
    receive = RTDEReceiveInterface(runtime["interface"])
    probe = RealSenseProbe(args.camera, settings["cameras"][args.camera]["serial"],
                           runtime.get("camera_width", 640), runtime.get("camera_height", 480))
    control = None
    out = args.output / time.strftime("%Y%m%d-%H%M%S")
    try:
        start_q = np.asarray(receive.getActualQ(), dtype=float)
        rgb = probe.capture()
        corners = detect_board(rgb, args.cols, args.rows)
        if corners is None:
            raise SystemExit("The board is not visible from the current pose. Jog the arm so the whole "
                             f"{args.cols}x{args.rows} inner-corner grid is in view (7-50 cm), then rerun.")
        camera_from_board, error = board_pose(corners, camera_matrix, distortion, args.cols, args.rows, square_m)
        print(f"Board seen at {np.linalg.norm(camera_from_board[:3, 3]) * 100:.1f} cm, reprojection {error:.2f} px")
        flange_from_camera = np.asarray(settings.get("calibration", {}).get("link6_from_camera", {}).get(args.camera, np.eye(4)))
        if args.guess_from:
            points = board_object_points(args.cols, args.rows, square_m)
            flange_from_camera = fit_with_rejection(_load_session(args.guess_from, args), camera_matrix, distortion, points)[0]
            print("Aiming poses with the camera pose from", args.guess_from)
        joints, kept, notes = build_plan(settings, ik, start_q, camera_from_board, args, flange_from_camera)
        print(f"Plan: {len(joints)} of {len(SCHEDULE)} poses, speed {args.speed} rad/s, acceleration {args.acceleration} rad/s^2")
        print("\n".join(notes))
        if len(joints) < MIN_POSES:
            raise SystemExit(f"Only {len(joints)} usable poses (need {MIN_POSES}). Move the arm further from "
                             "the table/limits, or raise --max-joint-move-rad / --max-translation-m.")
        if not execute:
            print("\nDry run: no motion sent. Add --execute to move the arm.")
            return None
        if not args.yes and input("The arm will move through these poses. Workspace clear, e-stop in reach? Type MOVE: ") != "MOVE":
            raise SystemExit("Not confirmed; nothing moved.")
        control = RTDEControlInterface(runtime["interface"])
        out.mkdir(parents=True)
        samples = []
        for number, q in [(0, start_q), *zip(kept, joints)]:
            if number and not control.moveJ([float(v) for v in q], args.speed, args.acceleration):
                raise RuntimeError(f"moveJ to pose {number} failed (protective stop?)")
            time.sleep(args.settle_s)
            before = np.asarray(receive.getActualQ(), dtype=float)
            rgb = probe.capture(frames=30)  # flush the camera's frame queue so the image is current
            measured = np.asarray(receive.getActualQ(), dtype=float)
            if np.abs(measured - before).max() > 2e-4:
                print(f"  pose {number:2d}: arm still moving during the exposure, skipped")
                continue
            corners = detect_board(rgb, args.cols, args.rows)
            if corners is None:
                print(f"  pose {number:2d}: board not detected, skipped")
                continue
            camera_from_board, error = board_pose(corners, camera_matrix, distortion, args.cols, args.rows, square_m)
            if error > MAX_REPROJECTION_PX:
                print(f"  pose {number:2d}: reprojection {error:.2f} px, skipped")
                continue
            from PIL import Image
            Image.fromarray(rgb).save(out / f"pose_{number:02d}.png")
            samples.append({"pose": number, "q": measured.tolist(), "flange": forward_matrix(measured, chain).tolist(),
                            "camera_from_board": camera_from_board.tolist(), "reprojection_px": error,
                            "corners": np.asarray(corners).reshape(-1, 2).tolist()})
            print(f"  pose {number:2d}: ok (reprojection {error:.2f} px)")
        control.moveJ([float(v) for v in start_q], args.speed, args.acceleration)
        (out / "session.json").write_text(json.dumps(samples, indent=2))
        print(f"Saved {len(samples)} samples to {out}")
        return samples
    except BaseException:
        if control is not None:
            try:
                control.stopJ(2.0)
            except Exception:
                pass
        raise
    finally:
        if control is not None:
            control.stopScript()
            control.disconnect()
        receive.disconnect()
        probe.close()


def solve_and_report(samples, settings, args) -> int:
    if len(samples) < MIN_POSES:
        print(f"Only {len(samples)} usable samples (need {MIN_POSES}); not solving.")
        return 1
    camera_matrix, distortion = _intrinsics(settings, args.camera)
    points = board_object_points(args.cols, args.rows, args.square_mm / 1000.0)
    refined, rms, (sigma_mm, sigma_deg), kept, dropped = fit_with_rejection(samples, camera_matrix, distortion, points)
    if dropped:
        print(f"Dropped outlier poses {dropped}; fitted {len(kept)} poses")
    print(f"Corner fit: {rms:.2f} px RMS; camera in flange frame xyz "
          "[{:.4f}, {:.4f}, {:.4f}] m (1-sigma {:.2f} mm, {:.2f} deg)".format(*refined[:3, 3], sigma_mm, sigma_deg))
    good = rms < 1.0 and sigma_mm < 1.0 and sigma_deg < 0.5
    print("Quality: " + ("good" if good else "poor: recapture with more tilt, a bigger board, or a closer camera"))
    print(json.dumps({"calibration": {"link6_from_camera": {args.camera: np.round(refined, 6).tolist()}}}))
    return 0 if good else 2


def _load_session(path: Path, args) -> list[dict]:
    """Samples from a session.json; sessions saved without corners are re-detected from their images."""
    from PIL import Image

    samples = json.loads(path.read_text())
    for sample in samples:
        if "corners" not in sample:
            image = np.asarray(Image.open(path.parent / f"pose_{sample['pose']:02d}.png").convert("RGB"))
            sample["corners"] = np.asarray(detect_board(image, args.cols, args.rows)).reshape(-1, 2).tolist()
    return samples


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--camera", default="left")
    parser.add_argument("--cols", type=int, required=True, help="Inner corners along the board's width")
    parser.add_argument("--rows", type=int, required=True, help="Inner corners along the board's height")
    parser.add_argument("--square-mm", type=float, default=14.0,
                        help="Printed square size; the lab board measures 14.0 mm (depth and motion agree), not its 10 mm file size")
    parser.add_argument("--output", type=Path, default=Path("var/handeye"))
    parser.add_argument("--execute", action="store_true", help="Move the arm (default: plan only)")
    parser.add_argument("--yes", action="store_true", help="Skip the MOVE confirmation")
    parser.add_argument("--speed", type=float, default=0.15, help="moveJ joint speed, rad/s")
    parser.add_argument("--acceleration", type=float, default=0.3, help="moveJ joint acceleration, rad/s^2")
    parser.add_argument("--settle-s", type=float, default=1.5, help="Wait after each move before the exposure")
    parser.add_argument("--guess-from", type=Path, help="A previous session.json: aim the poses with its camera pose")
    parser.add_argument("--max-joint-move-rad", type=float, default=0.9)
    parser.add_argument("--max-translation-m", type=float, default=0.15)
    parser.add_argument("--min-flange-z", type=float, default=0.4, help="Lowest allowed flange z in the base frame, metres")
    parser.add_argument("--solve", type=Path, help="Solve from a saved session.json instead of capturing")
    args = parser.parse_args(argv)
    settings = load_settings(args.config)
    if args.solve:
        return solve_and_report(_load_session(args.solve, args), settings, args)
    samples = capture_session(settings, args, args.execute)
    return 0 if samples is None else solve_and_report(samples, settings, args)


if __name__ == "__main__":
    raise SystemExit(main())
