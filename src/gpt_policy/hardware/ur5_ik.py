"""UR5 (CB3) forward kinematics and closed-form IK with a branch-continuous choice.

Poses are ``[x, y, z, roll, pitch, yaw]`` of the tool flange (``tool0``) in the
controller's base frame, i.e. the frame RTDE reports. The calibrated fingertip
TCP is applied by ``FrameCalibration`` (``link6_from_tcp``), not here.
"""

from __future__ import annotations

import threading

import numpy as np

from ..geometry.poses import matrix_to_pose, pose_to_matrix, rotation_distance
from ..motion.coordination import TrajectoryIKError

# Universal Robots UR5 (CB3) classic DH parameters, metres / radians.
D1, A2, A3, D4, D5, D6 = 0.089159, -0.425, -0.39225, 0.10915, 0.09465, 0.0823
_ALPHA = (np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0)
_A = (0.0, A2, A3, 0.0, 0.0, 0.0)
_D = (D1, 0.0, 0.0, D4, D5, D6)


class Chain:
    """Per-joint classic DH rows ``theta`` (offset added to the joint), ``a``, ``d``, ``alpha``."""

    def __init__(self, theta, a, d, alpha):
        self.theta, self.a, self.d, self.alpha = (
            tuple(float(v) for v in row) for row in (theta, a, d, alpha))
        if any(len(row) != 6 or not np.isfinite(row).all() for row in (self.theta, self.a, self.d, self.alpha)):
            raise ValueError("A UR5 kinematic chain needs six finite values per DH row")

    @classmethod
    def from_calibration(cls, calibration) -> "Chain":
        """Build from the controller's factory kinematics (``robot.kinematics_calibration``)."""
        try:
            return cls(*(calibration[key] for key in ("dh_theta", "dh_a", "dh_d", "dh_alpha")))
        except KeyError as exc:
            raise ValueError(f"kinematics_calibration is missing {exc}") from exc


NOMINAL_CHAIN = Chain((0.0,) * 6, _A, _D, _ALPHA)


def chain_from_settings(settings) -> Chain:
    """This arm's calibrated chain from ``robot.kinematics_calibration``, else the nominal UR5."""
    calibration = (settings.get("robot") or {}).get("kinematics_calibration")
    return Chain.from_calibration(calibration) if calibration else NOMINAL_CHAIN


def _dh(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    ct, st, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0.0, sa, ca, d],
        [0.0, 0.0, 0.0, 1.0],
    ])


def _link(index: int, theta: float, chain: Chain = NOMINAL_CHAIN) -> np.ndarray:
    return _dh(theta + chain.theta[index], chain.d[index], chain.a[index], chain.alpha[index])


def forward_matrix(joints, chain: Chain = NOMINAL_CHAIN) -> np.ndarray:
    transform = np.eye(4)
    for index, theta in enumerate(joints):
        transform = transform @ _link(index, float(theta), chain)
    return transform


def rotvec_to_matrix(rotvec) -> np.ndarray:
    """UR axis-angle (as returned by RTDE) to a rotation matrix."""
    vector = np.asarray(rotvec, dtype=float)
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3)
    x, y, z = vector / angle
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross


def rtde_pose_to_pose(rtde_pose) -> np.ndarray:
    """RTDE ``[x, y, z, rx, ry, rz]`` axis-angle pose to this repo's xyzrpy."""
    transform = np.eye(4)
    transform[:3, :3] = rotvec_to_matrix(rtde_pose[3:])
    transform[:3, 3] = rtde_pose[:3]
    return matrix_to_pose(transform)


def analytic_solutions(target: np.ndarray) -> list[np.ndarray]:
    """Up to eight closed-form joint vectors for a 4x4 flange target.

    Angles are in (-pi, pi]; callers choose the branch and ``2*pi`` offset.
    Degenerate wrist/shoulder cases return fewer candidates; the caller verifies
    every candidate with forward kinematics.
    """
    position = target[:3, 3]
    p05 = position - D6 * target[:3, 2]
    radius = float(np.hypot(p05[0], p05[1]))
    if radius < abs(D4) - 1e-9:
        return []
    offset = np.arccos(np.clip(D4 / max(radius, 1e-12), -1.0, 1.0))
    base_angle = np.arctan2(p05[1], p05[0])
    inverse = np.linalg.inv(target)
    solutions = []
    for th1 in (base_angle + offset + np.pi / 2, base_angle - offset + np.pi / 2):
        s1, c1 = np.sin(th1), np.cos(th1)
        cos5 = (position[0] * s1 - position[1] * c1 - D4) / D6
        if abs(cos5) > 1 + 1e-9:
            continue
        for th5 in (np.arccos(np.clip(cos5, -1, 1)), -np.arccos(np.clip(cos5, -1, 1))):
            s5 = np.sin(th5)
            if abs(s5) < 1e-9:
                # Wrist singularity: only th4+th6 is determined; keep th6 = 0.
                th6 = 0.0
            else:
                th6 = np.arctan2(
                    (-inverse[1, 0] * s1 + inverse[1, 1] * c1) / s5,
                    (inverse[0, 0] * s1 - inverse[0, 1] * c1) / s5,
                )
            t14 = np.linalg.inv(_link(0, th1)) @ target @ np.linalg.inv(_link(5, th6)) @ np.linalg.inv(_link(4, th5))
            p13 = t14[:3, 3] - D4 * t14[:3, 1]
            reach = float(np.linalg.norm(p13[:2]))
            cos3 = (reach**2 - A2**2 - A3**2) / (2 * A2 * A3)
            if abs(cos3) > 1 + 1e-9:
                continue
            for th3 in (np.arccos(np.clip(cos3, -1, 1)), -np.arccos(np.clip(cos3, -1, 1))):
                th2 = -np.arctan2(p13[1], -p13[0]) + np.arcsin(
                    np.clip(A3 * np.sin(th3) / max(reach, 1e-12), -1, 1))
                t34 = np.linalg.inv(_link(1, th2) @ _link(2, th3)) @ t14
                th4 = np.arctan2(t34[1, 0], t34[0, 0])
                solutions.append(np.array([th1, th2, th3, th4, th5, th6]))
    return solutions


def _wrap_near(angles: np.ndarray, seed: np.ndarray, lower: np.ndarray, upper: np.ndarray):
    """Shift each angle by a multiple of 2*pi to the legal value nearest the seed."""
    result = angles.copy()
    for i in range(6):
        best = None
        for k in (-2, -1, 0, 1, 2):
            value = angles[i] + 2 * np.pi * k
            if lower[i] - 1e-9 <= value <= upper[i] + 1e-9 and (
                    best is None or abs(value - seed[i]) < abs(best - seed[i])):
                best = value
        if best is None:
            return None
        result[i] = best
    return result


class UR5IK:
    """IK object matching the planner's ``YamIK`` interface."""

    name = "UR5 CB3 closed-form IK (8 branches, nearest to seed), verified by FK"

    def __init__(self, motion=None, joint_limits=None, chain: Chain | None = None):
        """``chain`` is this arm's calibrated kinematics; without it the nominal UR5 model is used.

        The closed form is nominal, so with a calibrated chain every branch is polished by
        Newton steps on the calibrated FK (a few millimetres at most) before it is verified.
        """
        motion = motion or {}
        self.chain = chain or NOMINAL_CHAIN
        limits = np.asarray(joint_limits if joint_limits is not None
                            else [[-2 * np.pi, 2 * np.pi]] * 6, dtype=float)
        self._set_limits(limits)
        self.execution_translation_tolerance_m = float(motion.get("ik_execution_translation_tolerance_m", 0.002))
        self.execution_rotation_tolerance_rad = float(motion.get("ik_execution_rotation_tolerance_rad", 0.0174532925))
        self.refine_iterations = int(motion.get("ik_refine_iterations", 12))
        self.lock = threading.RLock()
        self.solver = self  # The common planner's FK interface.

    def _set_limits(self, limits: np.ndarray) -> None:
        if limits.shape != (6, 2) or not np.isfinite(limits).all() or np.any(limits[:, 0] >= limits[:, 1]):
            raise ValueError("Expected six valid UR5 joint limits")
        self.lower, self.upper = limits.T.copy()

    def configure_joint_limits(self, bounds) -> None:
        self._set_limits(np.asarray(bounds, dtype=float))

    def forward_kinematics(self, joints) -> np.ndarray:
        joints = np.asarray(joints, dtype=float)
        if joints.shape != (6,) or not np.isfinite(joints).all():
            raise ValueError("UR5 arm joints must be a finite six-vector")
        return matrix_to_pose(forward_matrix(joints, self.chain))

    def error(self, target, joints) -> tuple[float, float]:
        actual = self.forward_kinematics(joints)
        return float(np.linalg.norm(target[:3] - actual[:3])), rotation_distance(target[3:], actual[3:])

    def _within(self, translation: float, rotation: float) -> bool:
        return (translation <= self.execution_translation_tolerance_m
                and rotation <= self.execution_rotation_tolerance_rad)

    def solve(self, target, seed, segment, sample):
        target = np.asarray(target, dtype=float)
        seed = np.asarray(seed, dtype=float)
        with self.lock:
            best, best_cost, nearest = None, np.inf, (1e30, 1e30)
            candidates = analytic_solutions(pose_to_matrix(target))
            if self.chain is not NOMINAL_CHAIN:
                # The closed form is nominal; the seed itself is the exact answer when the target
                # is the seed's own pose, and keeps near-singular wrists from drifting.
                candidates = [self._newton(target, candidate) for candidate in [*candidates, seed]]
            for candidate in candidates:
                joints = _wrap_near(candidate, seed, self.lower, self.upper)
                if joints is None:
                    continue
                translation, rotation = self.error(target, joints)
                if not self._within(translation, rotation):
                    nearest = min(nearest, (translation, rotation))
                    continue
                cost = float(np.max(np.abs(joints - seed)))
                if cost < best_cost:
                    best, best_cost = joints, cost
            if best is not None:
                return best
            refined = self._refine(target, seed)
            if refined is not None:
                return refined
            translation, rotation = nearest
            raise TrajectoryIKError(
                segment=segment, sample=sample, status=-1, status_name="UR5_NO_BRANCH",
                translation_error_m=translation, rotation_error_rad=rotation,
                reason="ik_residual_above_tolerance" if translation < 1e29 else "ik_joint_limit",
            )

    def _refine(self, target: np.ndarray, seed: np.ndarray):
        """Damped least squares from the seed; covers wrist singularities."""
        joints = self._newton(target, seed)
        if np.any(joints < self.lower - 1e-7) or np.any(joints > self.upper + 1e-7):
            return None
        translation, rotation = self.error(target, joints)
        return joints if self._within(translation, rotation) else None

    def _newton(self, target: np.ndarray, start: np.ndarray) -> np.ndarray:
        joints = np.asarray(start, dtype=float).copy()
        wanted = pose_to_matrix(target)
        for _ in range(max(1, 5 * self.refine_iterations)):
            current = forward_matrix(joints, self.chain)
            twist = np.r_[wanted[:3, 3] - current[:3, 3], _rotation_error(current[:3, :3], wanted[:3, :3])]
            if np.linalg.norm(twist[:3]) < 1e-5 and np.linalg.norm(twist[3:]) < 1e-4:
                break
            jacobian = np.empty((6, 6))
            for i in range(6):
                step = np.zeros(6)
                step[i] = 1e-6
                after = forward_matrix(joints + step, self.chain)
                jacobian[:3, i] = (after[:3, 3] - current[:3, 3]) / 1e-6
                jacobian[3:, i] = _rotation_error(current[:3, :3], after[:3, :3]) / 1e-6
            damping = 1e-3
            joints = joints + jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping * np.eye(6), twist)
        return joints


def _rotation_error(current: np.ndarray, wanted: np.ndarray) -> np.ndarray:
    """World-frame rotation vector taking ``current`` to ``wanted``."""
    delta = wanted @ current.T
    cosine = np.clip((np.trace(delta) - 1) / 2, -1, 1)
    angle = np.arccos(cosine)
    axis = np.array([delta[2, 1] - delta[1, 2], delta[0, 2] - delta[2, 0], delta[1, 0] - delta[0, 1]])
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.zeros(3)
    return axis / norm * angle
