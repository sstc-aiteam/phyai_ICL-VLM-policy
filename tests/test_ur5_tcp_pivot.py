"""Pivot calibration of the fingertip offset, against synthetic ground truth."""
import numpy as np
import pytest

from gpt_policy.hardware.ur5_handeye import rotation
from gpt_policy.hardware.ur5_tcp_pivot import solve_pivot


def flange_touching(pivot, tip, rotation_matrix):
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = pivot - rotation_matrix @ tip
    return transform


def test_offset_and_pivot_are_recovered_from_varied_orientations_with_noise():
    rng = np.random.default_rng(2)
    tip, pivot = np.array([0.002, -0.001, 0.174]), np.array([0.0015, -0.6393, 0.249])
    down = rotation(0, 180)
    flanges = [flange_touching(pivot, tip, down @ rotation(0, rx) @ rotation(1, ry) @ rotation(2, rz))
               for rx, ry, rz in ((0, 0, 0), (35, 0, 20), (-35, 0, 60), (0, 35, -40), (0, -35, 90), (25, 25, 140))]
    for flange in flanges:
        flange[:3, 3] += rng.normal(0, 0.0003, 3)
    solved_tip, solved_pivot, residuals, _ = solve_pivot(flanges)
    assert np.linalg.norm(solved_tip - tip) < 2e-3
    assert np.linalg.norm(solved_pivot - pivot) < 2e-3
    assert max(residuals) < 1.5


def test_touches_from_one_orientation_cannot_separate_offset_from_pivot():
    tip, pivot = np.array([0.0, 0.0, 0.17]), np.array([0.0, -0.6, 0.25])
    flanges = [flange_touching(pivot, tip, rotation(0, 180) @ rotation(2, 1.0 * k)) for k in range(5)]
    with pytest.raises(ValueError):
        solve_pivot(flanges)
    with pytest.raises(ValueError):
        solve_pivot(flanges[:3])
