"""Refit the wrist camera's hand-eye transform against a board registered by touching its corners.

    python -m gpt_policy.hardware.ur5_handeye_touch --config configs/machines/lab-ur5.json \
        --sessions var/handeye/<run1>/session.json var/handeye/<run2>/session.json

Hand-guide the closed fingertips onto the board's four outer inner corners (the first and last of the
first and last rows: 0, 7, 32, 39 for an 8x5 grid), tool roughly vertical, pressing Enter at each. Count
from the first corner and not from the board edge: the extremes are the 1st and 8th of 8 in a row, the
1st and 5th of 5 rows, so a row spans 7 squares and a column 4. If the touches were taken on other
corners, pass them with ``--touches ... --indices``. The arm's own pose then gives the board's
position, orientation and square size in the base frame, independent of the camera. The camera pose is
refitted to the corner pixels already captured with the board held at that position. It only reads the
arm; nothing is commanded.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..settings import load_settings
from .ur5_handeye import board_object_points, refine_hand_eye, rotation_log
from .ur5_ik import chain_from_settings, forward_matrix

OUTER_CORNERS = lambda cols, rows: (0, cols - 1, (rows - 1) * cols, rows * cols - 1)  # noqa: E731
MAX_TOUCH_RESIDUAL_MM = 4.0


def register_board(tips: np.ndarray, indices, cols: int):
    """``(square_m, base_from_board, residuals_mm)``: similarity fit of the board grid to touched tips (Umeyama)."""
    grid = np.array([[index % cols, index // cols, 0.0] for index in indices])
    tips = np.asarray(tips, dtype=float)
    grid_mean, tip_mean = grid.mean(axis=0), tips.mean(axis=0)
    covariance = (tips - tip_mean).T @ (grid - grid_mean) / len(grid)
    u, singular, vt = np.linalg.svd(covariance)
    flip = np.diag([1.0, 1.0, np.sign(np.linalg.det(u) * np.linalg.det(vt))])
    rotation = u @ flip @ vt
    scale = float(np.trace(np.diag(singular) @ flip) / np.mean(np.sum((grid - grid_mean) ** 2, axis=1)))
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = rotation, tip_mean - scale * rotation @ grid_mean
    fitted = (transform[:3, :3] @ (scale * grid).T).T + transform[:3, 3]
    return scale, transform, list(np.linalg.norm(fitted - tips, axis=1) * 1000)


def _pose_rms(flange, corners, flange_from_camera, base_from_board, points, camera_matrix, distortion) -> float:
    """RMS pixel error of one pose's corners under a given (not refitted) camera pose."""
    import cv2

    pose = np.linalg.inv(flange @ flange_from_camera) @ base_from_board
    projected = cv2.projectPoints(points, cv2.Rodrigues(pose[:3, :3])[0], pose[:3, 3], camera_matrix, distortion)[0]
    return float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - np.asarray(corners, dtype=float).reshape(-1, 2)) ** 2, axis=1))))


def refit_camera(samples, flange_from_camera, base_from_board, square_m, camera_matrix, distortion, cols, rows):
    """Fit X with the board fixed; drops samples whose pixels the fit cannot explain. Returns X, rms, sigma, kept."""
    points = board_object_points(cols, rows, square_m)
    kept = list(range(len(samples)))
    for _ in range(3):
        flanges = [np.asarray(samples[i]["flange"]) for i in kept]
        result = refine_hand_eye(flanges, [samples[i]["corners"] for i in kept], camera_matrix, distortion, points,
                                 flange_from_camera, None, base_from_board)
        per_sample = [_pose_rms(np.asarray(samples[i]["flange"]), samples[i]["corners"], result[0], base_from_board,
                                points, camera_matrix, distortion) for i in kept]
        limit = max(2.0, 3 * float(np.median(per_sample)))
        survivors = [i for i, rms in zip(kept, per_sample) if rms <= limit]
        if len(survivors) == len(kept) or len(survivors) < 8:
            break
        kept = survivors
    return (*result, kept)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sessions", type=Path, nargs="+", required=True, help="session.json files with corner pixels")
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--camera", default="left")
    parser.add_argument("--touches", type=Path, help="Reuse a saved touches file instead of touching again")
    parser.add_argument("--indices", type=int, nargs=4, help="Corner indices the saved touches were really taken on")
    parser.add_argument("--save-touches", type=Path, default=Path("var/handeye/touches.json"))
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    indices = list(OUTER_CORNERS(args.cols, args.rows))
    if args.touches:
        saved = json.loads(args.touches.read_text())
        tips = np.asarray(saved["tips"])
        indices = args.indices or saved["indices"]
    else:
        from rtde_receive import RTDEReceiveInterface

        chain = chain_from_settings(settings)
        link6_from_tcp = np.asarray(settings["calibration"]["link6_from_tcp"], dtype=float)
        receive = RTDEReceiveInterface(settings["runtime"]["interface"])
        tips = []
        try:
            for index in indices:
                while True:
                    row, column = divmod(index, args.cols)
                    input(f"Touch inner corner {index}: the {column + 1}th of {args.cols} from the left on the "
                          f"{row + 1}th of {args.rows} rows from the top (corner 1 is at the very start), "
                          "with the closed fingertips, hold still, press Enter: ")
                    first = np.asarray(receive.getActualQ(), dtype=float)
                    time.sleep(0.5)
                    second = np.asarray(receive.getActualQ(), dtype=float)
                    if np.abs(first - second).max() <= 2e-4:
                        break
                    print("  the arm moved while reading; hold still and try again")
                tips.append((forward_matrix(second, chain) @ link6_from_tcp)[:3, 3])
                print(f"  tip at {np.round(tips[-1], 4)} m")
        finally:
            receive.disconnect()
        tips = np.asarray(tips)
        args.save_touches.parent.mkdir(parents=True, exist_ok=True)
        args.save_touches.write_text(json.dumps({"indices": indices, "tips": tips.tolist()}))
        print(f"Saved touches to {args.save_touches}")

    square_m, base_from_board, residuals = register_board(tips, indices, args.cols)
    print(f"\nBoard registered by touch: squares {square_m * 1000:.2f} mm; corner residuals (mm): "
          + ", ".join(f"{r:.1f}" for r in residuals))
    if max(residuals) > MAX_TOUCH_RESIDUAL_MM:
        print(f"WARNING: a touch is more than {MAX_TOUCH_RESIDUAL_MM:.0f} mm off the rigid board; redo the touches before trusting the refit.")

    samples = [sample for path in args.sessions for sample in json.loads(path.read_text())]
    from .ur5_handeye import _intrinsics

    camera_matrix, distortion = _intrinsics(settings, args.camera)
    current = np.asarray(settings["calibration"]["link6_from_camera"][args.camera], dtype=float)
    refined, rms, (sigma_mm, sigma_deg), kept = refit_camera(
        samples, current, base_from_board, square_m, camera_matrix, distortion, args.cols, args.rows)
    shift_mm = np.linalg.norm(refined[:3, 3] - current[:3, 3]) * 1000
    turn_deg = np.degrees(np.linalg.norm(rotation_log(current[:3, :3].T @ refined[:3, :3])))
    print(f"Camera refit on {len(kept)} of {len(samples)} poses: {rms:.2f} px RMS, "
          f"xyz [{refined[0, 3]:.4f}, {refined[1, 3]:.4f}, {refined[2, 3]:.4f}] m (1-sigma {sigma_mm:.2f} mm, {sigma_deg:.2f} deg)")
    print(f"Change from the profile's camera pose: {shift_mm:.1f} mm, {turn_deg:.2f} deg")
    print(json.dumps({"calibration": {"link6_from_camera": {args.camera: np.round(refined, 6).tolist()}}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
