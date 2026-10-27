"""Board registration by touch and the fixed-board camera refit, against synthetic ground truth."""
import cv2
import numpy as np

from gpt_policy.hardware.ur5_handeye import board_object_points, rotation
from gpt_policy.hardware.ur5_handeye_touch import OUTER_CORNERS, refit_camera, register_board

HOME_FLANGE = np.array([[0, 1, 0, 0.0], [1, 0, 0, -0.5], [0, 0, -1, 0.55], [0, 0, 0, 1.0]])


def transform(rotation_matrix, translation):
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = rotation_matrix, translation
    return result


def test_touched_corners_give_the_board_pose_and_the_true_square_size():
    rng = np.random.default_rng(4)
    base_from_board = transform(rotation(0, 180) @ rotation(2, 12), [0.0, -0.64, 0.25])
    indices = OUTER_CORNERS(8, 5)
    grid = np.array([[i % 8, i // 8, 0.0] for i in indices]) * 0.014
    tips = (base_from_board[:3, :3] @ grid.T).T + base_from_board[:3, 3] + rng.normal(0, 0.0015, (4, 3))
    square, solved, residuals = register_board(tips, indices, 8)
    assert abs(square - 0.014) < 0.0005
    assert np.linalg.norm(solved[:3, 3] - base_from_board[:3, 3]) < 3e-3
    assert max(residuals) < 4.0


def test_camera_refit_with_the_board_held_fixed_recovers_a_camera_pose_that_a_free_board_could_not_pin_down():
    rng = np.random.default_rng(6)
    camera_matrix = np.array([[394.58, 0, 319.4], [0, 394.1, 231.1], [0, 0, 1.0]])
    square = 0.014
    points = board_object_points(8, 5, square)
    flange_from_camera = transform(rotation(2, 20) @ rotation(0, -35), [0.07, -0.011, 0.057])
    base_from_board = transform(rotation(0, 180) @ rotation(2, 10), [0.0, -0.64, 0.25])
    samples = []
    for _ in range(12):
        flange = transform(HOME_FLANGE[:3, :3] @ rotation(0, rng.uniform(-25, 25)) @ rotation(1, rng.uniform(-25, 25))
                           @ rotation(2, rng.uniform(-30, 30)), HOME_FLANGE[:3, 3] + rng.uniform(-0.03, 0.03, 3))
        board = np.linalg.inv(flange @ flange_from_camera) @ base_from_board
        pixels = cv2.projectPoints(points, cv2.Rodrigues(board[:3, :3])[0], board[:3, 3], camera_matrix, np.zeros(5))[0]
        samples.append({"flange": flange, "corners": pixels.reshape(-1, 2) + rng.normal(0, 0.3, (40, 2))})
    samples.append({"flange": samples[0]["flange"] @ transform(rotation(2, 8), [0.02, 0, 0]),  # one bad pose
                    "corners": samples[0]["corners"]})
    start = flange_from_camera.copy()
    start[:3, 3] += [0.012, -0.01, 0.015]
    solved, rms, _, kept = refit_camera(samples, start, base_from_board, square, camera_matrix, np.zeros(5), 8, 5)
    assert len(kept) == 12 and rms < 0.6
    assert np.linalg.norm(solved[:3, 3] - flange_from_camera[:3, 3]) < 1e-3
