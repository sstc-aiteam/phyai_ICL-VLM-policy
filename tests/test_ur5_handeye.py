"""Hand-eye solve, pose planning and board PnP, against synthetic ground truth (no hardware)."""
import numpy as np
import pytest

from gpt_policy.hardware.ur5_handeye import (
    SCHEDULE, board_consistency, board_object_points, board_pose, fit_with_rejection, plan_poses, refine_hand_eye, rotation, solve_hand_eye)

HOME_FLANGE = np.array([[0, 1, 0, 0.3], [1, 0, 0, -0.35], [0, 0, -1, 0.5], [0, 0, 0, 1.0]])


def transform(rotation_matrix, translation):
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = rotation_matrix, translation
    return result


def test_solve_recovers_the_camera_pose_from_noise_free_observations():
    rng = np.random.default_rng(3)
    flange_from_camera = transform(rotation(2, 20) @ rotation(0, -8), [0.01, -0.05, 0.08])
    base_from_board = transform(rotation(0, 180) @ rotation(2, 30), [0.3, -0.35, 0.3])
    flanges = [transform(HOME_FLANGE[:3, :3] @ rotation(0, rng.uniform(-25, 25)) @ rotation(1, rng.uniform(-25, 25))
                         @ rotation(2, rng.uniform(-30, 30)), HOME_FLANGE[:3, 3] + rng.uniform(-0.05, 0.05, 3))
               for _ in range(15)]
    boards = [np.linalg.inv(f @ flange_from_camera) @ base_from_board for f in flanges]
    solved = solve_hand_eye(flanges, boards)
    assert np.abs(solved - flange_from_camera).max() < 1e-9
    assert board_consistency(flanges, boards, solved)[0] < 1e-6


def test_solve_refuses_poses_without_distinct_rotations():
    flanges = [HOME_FLANGE.copy() for _ in range(10)]
    with pytest.raises(ValueError):
        solve_hand_eye(flanges, [np.eye(4)] * 10)


def test_planned_poses_keep_the_board_pivot_at_the_same_place_in_the_camera_frame():
    pivot = HOME_FLANGE[:3, 3] + HOME_FLANGE[:3, 2] * 0.2
    plan = plan_poses(HOME_FLANGE, pivot)
    assert len(plan) == len(SCHEDULE) == 15
    for (_, _, _, scale), target in zip(SCHEDULE, plan):
        in_flange = np.linalg.inv(target) @ np.r_[pivot, 1.0]
        assert np.linalg.norm(in_flange[:3]) == pytest.approx(0.2 * scale)  # rotation about the pivot, only distance scales
        assert np.linalg.det(target[:3, :3]) == pytest.approx(1.0)


def test_board_pose_recovers_a_projected_board():
    import cv2

    camera_matrix = np.array([[394.58, 0, 319.4], [0, 394.1, 231.1], [0, 0, 1.0]])
    expected = transform(rotation(0, 200) @ rotation(1, 15) @ rotation(2, 10), [-0.02, 0.01, 0.20])
    points = board_object_points(5, 5, 0.013)
    rvec = cv2.Rodrigues(expected[:3, :3])[0]
    corners = cv2.projectPoints(points, rvec, expected[:3, 3], camera_matrix, np.zeros(5))[0]
    for shape in ((-1, 1, 2), (-1, 2)):  # OpenCV 4 detectors return (N, 1, 2), OpenCV 5 (N, 2)
        solved, error = board_pose(corners.reshape(shape), camera_matrix, np.zeros(5), 5, 5, 0.013)
        assert error < 1e-3
        assert np.abs(solved - expected).max() < 1e-4


def test_corner_refinement_recovers_the_camera_pose_from_noisy_pixels_with_the_board_half_a_turn_from_base():
    import cv2

    rng = np.random.default_rng(1)
    camera_matrix = np.array([[394.58, 0, 319.4], [0, 394.1, 231.1], [0, 0, 1.0]])
    points = board_object_points(8, 5, 0.013)
    flange_from_camera = transform(rotation(2, 20) @ rotation(0, -35), [0.06, -0.017, 0.07])
    base_from_board = transform(rotation(0, 180), [0.3, -0.45, 0.30])  # a 180 deg rotation: the log is degenerate here
    flanges = [transform(HOME_FLANGE[:3, :3] @ rotation(0, rng.uniform(-25, 25)) @ rotation(1, rng.uniform(-25, 25))
                         @ rotation(2, rng.uniform(-30, 30)), HOME_FLANGE[:3, 3] + rng.uniform(-0.03, 0.03, 3))
               for _ in range(9)]
    boards = [np.linalg.inv(f @ flange_from_camera) @ base_from_board for f in flanges]
    corners = [cv2.projectPoints(points, cv2.Rodrigues(c[:3, :3])[0], c[:3, 3], camera_matrix, np.zeros(5))[0].reshape(-1, 2)
               + rng.normal(0, 0.3, (40, 2)) for c in boards]
    start = flange_from_camera.copy()
    start[:3, 3] += [0.01, -0.01, 0.01]  # a poor starting guess
    solved, rms, (sigma_mm, sigma_deg) = refine_hand_eye(flanges, corners, camera_matrix, np.zeros(5), points, start, boards[0])
    assert rms < 0.6
    assert np.linalg.norm(solved[:3, 3] - flange_from_camera[:3, 3]) < 2e-3
    assert sigma_mm < 3.0


def test_a_pose_whose_flange_does_not_match_its_image_is_dropped_and_the_rest_still_solve():
    import cv2

    rng = np.random.default_rng(5)
    camera_matrix = np.array([[394.58, 0, 319.4], [0, 394.1, 231.1], [0, 0, 1.0]])
    points = board_object_points(8, 5, 0.013)
    flange_from_camera = transform(rotation(2, 20) @ rotation(0, -35), [0.06, -0.017, 0.07])
    base_from_board = transform(rotation(0, 180), [0.3, -0.45, 0.30])
    flanges = [transform(HOME_FLANGE[:3, :3] @ rotation(0, rng.uniform(-25, 25)) @ rotation(1, rng.uniform(-25, 25))
                         @ rotation(2, rng.uniform(-30, 30)), HOME_FLANGE[:3, 3] + rng.uniform(-0.03, 0.03, 3))
               for _ in range(10)]
    samples = []
    for number, flange in enumerate(flanges):
        board = np.linalg.inv(flange @ flange_from_camera) @ base_from_board
        pixels = cv2.projectPoints(points, cv2.Rodrigues(board[:3, :3])[0], board[:3, 3], camera_matrix, np.zeros(5))[0].reshape(-1, 2)
        recorded = flange.copy()
        if number == 4:
            recorded[:3, 3] += [0.012, -0.008, 0.005]  # the arm had not stopped when this frame was taken
        samples.append({"pose": number, "flange": recorded, "camera_from_board": board,
                        "corners": pixels + rng.normal(0, 0.3, (40, 2))})
    solved, rms, _, kept, dropped = fit_with_rejection(samples, camera_matrix, np.zeros(5), points)
    assert dropped == [4] and len(kept) == 9
    assert rms < 0.6
    assert np.linalg.norm(solved[:3, 3] - flange_from_camera[:3, 3]) < 2e-3
