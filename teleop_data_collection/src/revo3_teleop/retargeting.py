"""Fail-closed 6-DoF wrist-pose to Tianji joint-target interfaces.

This module intentionally does not ship a robot-specific IK model.  A real
Tianji URDF/kinematic model, tool transform and solver must be injected and
identified by revision.  BrainCo EDU glove flex/IMU/magnetometer telemetry is
not a verified 6-DoF wrist pose and must never enter this path.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol, Sequence

import numpy as np


TIANJI_ARM_DOF = 7


class WristPoseError(ValueError):
    """A wrist pose is malformed, stale, unverified, or frame-incompatible."""


class RetargetingBlocked(RuntimeError):
    """A safe Cartesian or joint target cannot be produced."""


def _text(value: object, *, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return result


def _vector(value: object, size: int, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape ({size},)")
    return result.copy()


def _quaternion(value: object, *, name: str) -> np.ndarray:
    result = _vector(value, 4, name=name)
    norm = float(np.linalg.norm(result))
    if not 0.98 <= norm <= 1.02:
        raise ValueError(f"{name} norm must be within [0.98, 1.02], got {norm}")
    result /= norm
    return result


def _rotation(value: object, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3, 3) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite 3x3 matrix")
    if not np.allclose(result.T @ result, np.eye(3), atol=1e-5):
        raise ValueError(f"{name} must be orthonormal")
    if not np.isclose(np.linalg.det(result), 1.0, atol=1e-5):
        raise ValueError(f"{name} must be a proper rotation with determinant +1")
    return result.copy()


def quaternion_xyzw_to_rotation(quaternion_xyzw: object) -> np.ndarray:
    x, y, z, w = _quaternion(quaternion_xyzw, name="quaternion_xyzw")
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotation_to_quaternion_xyzw(rotation: object) -> np.ndarray:
    matrix = _rotation(rotation, name="rotation")
    # Stable branch formulation; sign is canonicalized to w >= 0 so recorded
    # values do not jump between equivalent q and -q representations.
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.asarray(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.asarray(
                [
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                ]
            )
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.asarray(
                [
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                ]
            )
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.asarray(
                [
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                ]
            )
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[3] < 0.0:
        quaternion *= -1.0
    return quaternion


def rotation_angle_rad(rotation: object) -> float:
    matrix = _rotation(rotation, name="rotation")
    cosine = float(np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0))
    return math.acos(cosine)


@dataclass(frozen=True)
class WristPose6D:
    source_id: str
    source_frame: str
    capture_timestamp_ns: int
    receive_timestamp_ns: int
    clock_domain: str
    position_m: np.ndarray
    quaternion_xyzw: np.ndarray
    calibration_revision: str
    confidence: float = 1.0
    valid: bool = True

    def __post_init__(self) -> None:
        for name in ("source_id", "source_frame", "clock_domain", "calibration_revision"):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))
        if self.capture_timestamp_ns < 0 or self.receive_timestamp_ns < self.capture_timestamp_ns:
            raise ValueError("invalid wrist-pose capture/receive timestamps")
        confidence = float(self.confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be finite in [0, 1]")
        position = _vector(self.position_m, 3, name="position_m")
        quaternion = _quaternion(self.quaternion_xyzw, name="quaternion_xyzw")
        position.setflags(write=False)
        quaternion.setflags(write=False)
        object.__setattr__(self, "position_m", position)
        object.__setattr__(self, "quaternion_xyzw", quaternion)
        object.__setattr__(self, "confidence", confidence)

    def require_fresh(
        self,
        *,
        now_ns: int,
        max_age_ns: int,
        expected_clock_domain: str,
        minimum_confidence: float,
    ) -> None:
        if not self.valid:
            raise WristPoseError("wrist_pose_marked_invalid")
        if self.clock_domain != expected_clock_domain:
            raise WristPoseError("wrist_pose_clock_domain_mismatch")
        now = int(now_ns)
        if now < self.capture_timestamp_ns:
            raise WristPoseError("wrist_pose_timestamp_in_future")
        if now - self.capture_timestamp_ns > int(max_age_ns):
            raise WristPoseError("wrist_pose_stale")
        if self.confidence < float(minimum_confidence):
            raise WristPoseError("wrist_pose_confidence_too_low")


class WristPoseProvider(Protocol):
    """Tracker/MANUS boundary. ``read_pose`` must not synthesize missing axes."""

    provides_wrist_pose: bool

    def read_pose(self) -> WristPose6D: ...


@dataclass(frozen=True)
class CartesianPose:
    frame: str
    child_frame: str
    position_m: np.ndarray
    quaternion_xyzw: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame", _text(self.frame, name="frame"))
        object.__setattr__(self, "child_frame", _text(self.child_frame, name="child_frame"))
        object.__setattr__(self, "position_m", _vector(self.position_m, 3, name="position_m"))
        object.__setattr__(
            self,
            "quaternion_xyzw",
            _quaternion(self.quaternion_xyzw, name="quaternion_xyzw"),
        )


@dataclass(frozen=True)
class WristRetargetCalibration:
    revision: str
    source_frame: str
    robot_base_frame: str
    tool_frame: str
    source_reference: CartesianPose
    robot_reference: CartesianPose
    source_axes_to_robot: np.ndarray
    translation_gain: np.ndarray

    def __post_init__(self) -> None:
        for name in ("revision", "source_frame", "robot_base_frame", "tool_frame"):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))
        if self.source_reference.frame != self.source_frame:
            raise ValueError("source_reference.frame must match source_frame")
        if self.robot_reference.frame != self.robot_base_frame:
            raise ValueError("robot_reference.frame must match robot_base_frame")
        if self.robot_reference.child_frame != self.tool_frame:
            raise ValueError("robot_reference.child_frame must match tool_frame")
        mapping = _rotation(self.source_axes_to_robot, name="source_axes_to_robot")
        gain = _vector(self.translation_gain, 3, name="translation_gain")
        if np.any(gain <= 0.0):
            raise ValueError("translation_gain entries must be positive")
        mapping.setflags(write=False)
        gain.setflags(write=False)
        object.__setattr__(self, "source_axes_to_robot", mapping)
        object.__setattr__(self, "translation_gain", gain)


@dataclass(frozen=True)
class CartesianSafetyBounds:
    min_position_m: np.ndarray
    max_position_m: np.ndarray
    max_translation_from_reference_m: float
    max_orientation_from_reference_rad: float

    def __post_init__(self) -> None:
        lower = _vector(self.min_position_m, 3, name="min_position_m")
        upper = _vector(self.max_position_m, 3, name="max_position_m")
        if np.any(lower >= upper):
            raise ValueError("each min_position_m must be less than max_position_m")
        translation = float(self.max_translation_from_reference_m)
        orientation = float(self.max_orientation_from_reference_rad)
        if not math.isfinite(translation) or translation <= 0.0:
            raise ValueError("max_translation_from_reference_m must be positive")
        if not math.isfinite(orientation) or not 0.0 < orientation <= math.pi:
            raise ValueError("max_orientation_from_reference_rad must be in (0, pi]")
        object.__setattr__(self, "min_position_m", lower)
        object.__setattr__(self, "max_position_m", upper)
        object.__setattr__(self, "max_translation_from_reference_m", translation)
        object.__setattr__(self, "max_orientation_from_reference_rad", orientation)


class RelativeWristRetargeter:
    """Map calibrated relative human wrist motion into a robot tool target."""

    def __init__(
        self,
        calibration: WristRetargetCalibration,
        bounds: CartesianSafetyBounds,
    ) -> None:
        self.calibration = calibration
        self.bounds = bounds

    def map(self, pose: WristPose6D) -> CartesianPose:
        calibration = self.calibration
        if pose.source_frame != calibration.source_frame:
            raise RetargetingBlocked("wrist_pose_source_frame_mismatch")
        source_delta = pose.position_m - calibration.source_reference.position_m
        robot_delta = calibration.source_axes_to_robot @ (
            source_delta * calibration.translation_gain
        )
        target_position = calibration.robot_reference.position_m + robot_delta

        source_rotation = quaternion_xyzw_to_rotation(pose.quaternion_xyzw)
        source_reference_rotation = quaternion_xyzw_to_rotation(
            calibration.source_reference.quaternion_xyzw
        )
        robot_reference_rotation = quaternion_xyzw_to_rotation(
            calibration.robot_reference.quaternion_xyzw
        )
        source_relative = source_rotation @ source_reference_rotation.T
        mapped_relative = (
            calibration.source_axes_to_robot
            @ source_relative
            @ calibration.source_axes_to_robot.T
        )
        target_rotation = mapped_relative @ robot_reference_rotation

        bounds = self.bounds
        if np.any(target_position < bounds.min_position_m) or np.any(
            target_position > bounds.max_position_m
        ):
            raise RetargetingBlocked("cartesian_workspace_violation")
        if (
            np.linalg.norm(target_position - calibration.robot_reference.position_m)
            > bounds.max_translation_from_reference_m
        ):
            raise RetargetingBlocked("cartesian_translation_delta_violation")
        if (
            rotation_angle_rad(target_rotation @ robot_reference_rotation.T)
            > bounds.max_orientation_from_reference_rad
        ):
            raise RetargetingBlocked("cartesian_orientation_delta_violation")
        return CartesianPose(
            frame=calibration.robot_base_frame,
            child_frame=calibration.tool_frame,
            position_m=target_position,
            quaternion_xyzw=rotation_to_quaternion_xyzw(target_rotation),
        )


@dataclass(frozen=True)
class IKSolution:
    q_rad: np.ndarray
    joint_order: tuple[str, ...]
    converged: bool
    position_residual_m: float
    orientation_residual_rad: float
    iterations: int
    solver_revision: str

    def __post_init__(self) -> None:
        q = _vector(self.q_rad, TIANJI_ARM_DOF, name="q_rad")
        order = tuple(_text(name, name="joint_order entry") for name in self.joint_order)
        if len(order) != TIANJI_ARM_DOF or len(set(order)) != TIANJI_ARM_DOF:
            raise ValueError("joint_order must contain seven unique names")
        position = float(self.position_residual_m)
        orientation = float(self.orientation_residual_rad)
        if not math.isfinite(position) or not math.isfinite(orientation):
            raise ValueError("IK residuals must be finite")
        if position < 0.0 or orientation < 0.0 or self.iterations < 0:
            raise ValueError("IK residuals/iterations must be non-negative")
        object.__setattr__(self, "q_rad", q)
        object.__setattr__(self, "joint_order", order)
        object.__setattr__(self, "solver_revision", _text(self.solver_revision, name="solver_revision"))


class IKSolver(Protocol):
    """Robot-specific, versioned solver supplied by the Tianji integration."""

    def solve(self, target: CartesianPose, seed_q_rad: np.ndarray) -> IKSolution: ...


@dataclass(frozen=True)
class PlannedArmTarget:
    q_rad: np.ndarray
    target_timestamp_ns: int
    wrist_capture_timestamp_ns: int
    wrist_source_id: str
    retarget_calibration_revision: str
    ik_solver_revision: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "q_rad", _vector(self.q_rad, TIANJI_ARM_DOF, name="q_rad"))
        if min(self.target_timestamp_ns, self.wrist_capture_timestamp_ns) < 0:
            raise ValueError("target timestamps must be non-negative")


class TianjiJointTargetPlanner:
    """Validate pose, retarget, solve IK and return a 7-D candidate target."""

    def __init__(
        self,
        *,
        retargeter: RelativeWristRetargeter,
        ik_solver: IKSolver,
        expected_joint_order: Sequence[str],
        expected_clock_domain: str,
        max_pose_age_ns: int,
        minimum_pose_confidence: float = 0.8,
        max_ik_position_residual_m: float = 0.005,
        max_ik_orientation_residual_rad: float = math.radians(3.0),
        max_ik_iterations: int = 200,
    ) -> None:
        self.retargeter = retargeter
        self.ik_solver = ik_solver
        self.expected_joint_order = tuple(
            _text(name, name="expected_joint_order entry") for name in expected_joint_order
        )
        if len(self.expected_joint_order) != TIANJI_ARM_DOF or len(
            set(self.expected_joint_order)
        ) != TIANJI_ARM_DOF:
            raise ValueError("expected_joint_order must contain seven unique names")
        self.expected_clock_domain = _text(expected_clock_domain, name="expected_clock_domain")
        self.max_pose_age_ns = int(max_pose_age_ns)
        self.minimum_pose_confidence = float(minimum_pose_confidence)
        self.max_ik_position_residual_m = float(max_ik_position_residual_m)
        self.max_ik_orientation_residual_rad = float(max_ik_orientation_residual_rad)
        self.max_ik_iterations = int(max_ik_iterations)
        if self.max_pose_age_ns <= 0 or self.max_ik_iterations <= 0:
            raise ValueError("age and iteration limits must be positive")
        if not 0.0 <= self.minimum_pose_confidence <= 1.0:
            raise ValueError("minimum_pose_confidence must be in [0, 1]")
        if self.max_ik_position_residual_m <= 0.0 or self.max_ik_orientation_residual_rad <= 0.0:
            raise ValueError("IK residual thresholds must be positive")

    def plan(
        self,
        pose: WristPose6D,
        *,
        seed_q_rad: Sequence[float] | np.ndarray,
        now_ns: int,
    ) -> PlannedArmTarget:
        now = int(now_ns)
        pose.require_fresh(
            now_ns=now,
            max_age_ns=self.max_pose_age_ns,
            expected_clock_domain=self.expected_clock_domain,
            minimum_confidence=self.minimum_pose_confidence,
        )
        cartesian = self.retargeter.map(pose)
        solution = self.ik_solver.solve(
            cartesian,
            _vector(seed_q_rad, TIANJI_ARM_DOF, name="seed_q_rad"),
        )
        if not solution.converged:
            raise RetargetingBlocked("ik_not_converged")
        if solution.joint_order != self.expected_joint_order:
            raise RetargetingBlocked("ik_joint_order_mismatch")
        if solution.position_residual_m > self.max_ik_position_residual_m:
            raise RetargetingBlocked("ik_position_residual_too_large")
        if solution.orientation_residual_rad > self.max_ik_orientation_residual_rad:
            raise RetargetingBlocked("ik_orientation_residual_too_large")
        if solution.iterations > self.max_ik_iterations:
            raise RetargetingBlocked("ik_iteration_limit_exceeded")
        return PlannedArmTarget(
            q_rad=solution.q_rad,
            target_timestamp_ns=now,
            wrist_capture_timestamp_ns=pose.capture_timestamp_ns,
            wrist_source_id=pose.source_id,
            retarget_calibration_revision=self.retargeter.calibration.revision,
            ik_solver_revision=solution.solver_revision,
        )


__all__ = [
    "CartesianPose",
    "CartesianSafetyBounds",
    "IKSolution",
    "IKSolver",
    "PlannedArmTarget",
    "RelativeWristRetargeter",
    "RetargetingBlocked",
    "TianjiJointTargetPlanner",
    "WristPose6D",
    "WristPoseError",
    "WristPoseProvider",
    "WristRetargetCalibration",
    "quaternion_xyzw_to_rotation",
    "rotation_to_quaternion_xyzw",
]
