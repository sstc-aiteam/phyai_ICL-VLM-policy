"""Measure the fingertip offset (``calibration.link6_from_tcp``) by pivoting on one fixed point.

    python -m gpt_policy.hardware.ur5_tcp_pivot --config configs/machines/lab-ur5.json

Hand-guide the closed fingertips onto the same fixed point (a printed corner, a pin tip) from at
least four clearly different tool orientations and press Enter at each. The offset that puts the tip
on one point from every orientation is solved by least squares, with no camera involved. It only
reads the arm (nothing is commanded). The solved pivot point is also printed in the base frame, so it
can be compared with a camera-predicted position.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from ..settings import load_settings
from .ur5_ik import chain_from_settings, forward_matrix

MIN_TOUCHES = 4
MIN_SINGULAR_VALUE = 0.2  # of the stacked [R, -I] system; below this the orientations are too alike


def solve_pivot(flanges):
    """``(tip_in_flange, pivot_in_base, per_touch_residual_mm, conditioning)`` from flange poses.

    Each touch gives R_i p + t_i = c for the fixed pivot c and the tip offset p, in the flange frame.
    """
    if len(flanges) < MIN_TOUCHES:
        raise ValueError(f"Need at least {MIN_TOUCHES} touches")
    rows = np.vstack([np.hstack([f[:3, :3], -np.eye(3)]) for f in flanges])
    rhs = -np.concatenate([f[:3, 3] for f in flanges])
    singular = np.linalg.svd(rows / np.sqrt(len(flanges)), compute_uv=False)
    if singular[-1] < MIN_SINGULAR_VALUE:
        raise ValueError("The touches use too similar tool orientations to separate the offset from the pivot; "
                         "tilt the tool by 30 degrees or more in different directions")
    solution = np.linalg.lstsq(rows, rhs, rcond=None)[0]
    tip, pivot = solution[:3], solution[3:]
    residuals = [float(np.linalg.norm(f[:3, :3] @ tip + f[:3, 3] - pivot)) * 1000 for f in flanges]
    return tip, pivot, residuals, float(singular[-1])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--touches", type=int, default=6)
    args = parser.parse_args(argv)
    from rtde_receive import RTDEReceiveInterface

    settings = load_settings(args.config)
    chain = chain_from_settings(settings)
    receive = RTDEReceiveInterface(settings["runtime"]["interface"])
    flanges = []
    try:
        for index in range(1, args.touches + 1):
            input(f"[{index}/{args.touches}] Touch the same point with the closed fingertips from a new orientation, hold still, press Enter: ")
            first = np.asarray(receive.getActualQ(), dtype=float)
            time.sleep(0.5)
            second = np.asarray(receive.getActualQ(), dtype=float)
            if np.abs(first - second).max() > 2e-4:
                print("  the arm moved while reading; hold still and try this touch again")
                continue
            flanges.append(forward_matrix(second, chain))
            print(f"  recorded flange xyz {np.round(flanges[-1][:3, 3], 4)} m, tool axis {np.round(flanges[-1][:3, 2], 2)}")
    finally:
        receive.disconnect()
    tip, pivot, residuals, conditioning = solve_pivot(flanges)
    print(f"\nFingertip offset in the flange frame: x {tip[0] * 1000:.1f}, y {tip[1] * 1000:.1f}, z {tip[2] * 1000:.1f} mm "
          f"(orientation spread {conditioning:.2f})")
    print("Per-touch scatter (mm):", ", ".join(f"{r:.1f}" for r in residuals), f"| RMS {np.sqrt(np.mean(np.square(residuals))):.1f} mm")
    print(f"Pivot point in the base frame: {np.round(pivot, 4)} m")
    print(f'"link6_from_tcp": [[1,0,0,{tip[0]:.4f}],[0,1,0,{tip[1]:.4f}],[0,0,1,{tip[2]:.4f}],[0,0,0,1]]')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
