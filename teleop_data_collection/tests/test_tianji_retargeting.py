from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from revo3_teleop.contracts import NativeSample, SampleHeader  # noqa: E402
from revo3_teleop.retargeting import (  # noqa: E402
    CartesianPose,
    CartesianSafetyBounds,
    IKSolution,
    RelativeWristRetargeter,
    RetargetingBlocked,
    TianjiJointTargetPlanner,
    WristPose6D,
    WristPoseError,
    WristRetargetCalibration,
)
from revo3_teleop.sources import (  # noqa: E402
    ManusRosFrame,
    ManusWristPoseExtractor,
    VerifiedManusWristConfig,
)
from revo3_teleop.tianji_runtime import TianjiTeleopRuntime  # noqa: E402


JOINTS = tuple(f"arm_joint_{index}" for index in range(7))


def pose(timestamp_ns: int = 1_000_000_000) -> WristPose6D:
    return WristPose6D(
        source_id="verified_tracker",
        source_frame="tracker_world",
        capture_timestamp_ns=timestamp_ns,
        receive_timestamp_ns=timestamp_ns + 1_000_000,
        clock_domain="workstation_monotonic",
        position_m=np.asarray([0.1, 0.0, 0.0]),
        quaternion_xyzw=np.asarray([0.0, 0.0, 0.0, 1.0]),
        calibration_revision="tracker-cal-v1",
    )


def planner(solution: IKSolution | None = None) -> TianjiJointTargetPlanner:
    source_reference = CartesianPose(
        frame="tracker_world",
        child_frame="wrist",
        position_m=np.zeros(3),
        quaternion_xyzw=np.asarray([0.0, 0.0, 0.0, 1.0]),
    )
    robot_reference = CartesianPose(
        frame="tianji_base",
        child_frame="revo_tool",
        position_m=np.asarray([0.4, 0.0, 0.5]),
        quaternion_xyzw=np.asarray([0.0, 0.0, 0.0, 1.0]),
    )
    retargeter = RelativeWristRetargeter(
        WristRetargetCalibration(
            revision="wrist-to-arm-v1",
            source_frame="tracker_world",
            robot_base_frame="tianji_base",
            tool_frame="revo_tool",
            source_reference=source_reference,
            robot_reference=robot_reference,
            source_axes_to_robot=np.eye(3),
            translation_gain=np.ones(3),
        ),
        CartesianSafetyBounds(
            min_position_m=np.asarray([0.0, -1.0, 0.0]),
            max_position_m=np.asarray([1.0, 1.0, 1.0]),
            max_translation_from_reference_m=0.2,
            max_orientation_from_reference_rad=0.5,
        ),
    )
    result = solution or IKSolution(
        q_rad=np.linspace(-0.1, 0.1, 7),
        joint_order=JOINTS,
        converged=True,
        position_residual_m=0.001,
        orientation_residual_rad=0.01,
        iterations=12,
        solver_revision="tianji-urdf-hash:abc",
    )

    class FixedIK:
        def solve(self, target, seed_q_rad):
            assert target.frame == "tianji_base"
            np.testing.assert_allclose(target.position_m, [0.5, 0.0, 0.5])
            np.testing.assert_allclose(seed_q_rad, np.zeros(7))
            return result

    return TianjiJointTargetPlanner(
        retargeter=retargeter,
        ik_solver=FixedIK(),
        expected_joint_order=JOINTS,
        expected_clock_domain="workstation_monotonic",
        max_pose_age_ns=50_000_000,
    )


def test_relative_retarget_and_injected_ik_produce_seven_axis_candidate() -> None:
    result = planner().plan(
        pose(),
        seed_q_rad=np.zeros(7),
        now_ns=1_010_000_000,
    )

    np.testing.assert_allclose(result.q_rad, np.linspace(-0.1, 0.1, 7))
    assert result.wrist_capture_timestamp_ns == 1_000_000_000
    assert result.retarget_calibration_revision == "wrist-to-arm-v1"


def test_stale_pose_and_ik_joint_order_mismatch_fail_before_a_target() -> None:
    with pytest.raises(WristPoseError, match="stale"):
        planner().plan(
            pose(),
            seed_q_rad=np.zeros(7),
            now_ns=1_060_000_000,
        )

    wrong = IKSolution(
        q_rad=np.zeros(7),
        joint_order=tuple(reversed(JOINTS)),
        converged=True,
        position_residual_m=0.0,
        orientation_residual_rad=0.0,
        iterations=1,
        solver_revision="wrong-order",
    )
    with pytest.raises(RetargetingBlocked, match="joint_order"):
        planner(wrong).plan(
            pose(),
            seed_q_rad=np.zeros(7),
            now_ns=1_010_000_000,
        )


def _manus_frame(*, node_id: int = 42) -> ManusRosFrame:
    sample = NativeSample(
        SampleHeader(
            source_id="manus_ros:/glove",
            sequence=3,
            capture_timestamp_ns=100,
            receive_timestamp_ns=100,
            clock_domain="host_ros_callback_arrival_monotonic",
        ),
        {
            "raw_node_ids": np.asarray([node_id], dtype=np.int32),
            "raw_node_positions": np.asarray([[100.0, 0.0, 0.0]], dtype=np.float32),
            "raw_node_orientations": np.asarray(
                [[0.0, 0.0, 0.0, 1.0]], dtype=np.float32
            ),
        },
    )
    return ManusRosFrame(
        sample=sample,
        topic="/glove",
        side="right",
        ergonomics_types=(),
        raw_joint_types=("wrist",),
        raw_chain_types=("arm",),
        provides_wrist_pose=True,
    )


def test_manus_wrist_requires_explicit_mapping_and_position_scale() -> None:
    with pytest.raises(WristPoseError, match="not been verified"):
        ManusWristPoseExtractor(
            VerifiedManusWristConfig(
                wrist_node_id=42,
                side="right",
                source_frame="manus_world",
                calibration_revision="cal-v1",
                position_scale_to_m=0.001,
                wrist_node_mapping_verified=False,
            )
        )

    extractor = ManusWristPoseExtractor(
        VerifiedManusWristConfig(
            wrist_node_id=42,
            side="right",
            source_frame="manus_world",
            calibration_revision="cal-v1",
            position_scale_to_m=0.001,
            wrist_node_mapping_verified=True,
        )
    )
    extracted = extractor.extract(_manus_frame())
    np.testing.assert_allclose(extracted.position_m, [0.1, 0.0, 0.0])
    assert extracted.capture_timestamp_ns == 100


def test_runtime_pose_failure_never_reaches_tianji_submit() -> None:
    class Backend:
        def __init__(self):
            self.submit_calls = 0

        def read_state(self):
            return SimpleNamespace(q_rad=np.zeros(7))

        def submit_target(self, **kwargs):
            self.submit_calls += 1
            raise AssertionError("position write must not be reached")

    class BadProvider:
        provides_wrist_pose = True

        def read_pose(self):
            raise WristPoseError("tracker_lost")

    backend = Backend()
    runtime = TianjiTeleopRuntime(
        backend=backend,
        wrist_pose_provider=BadProvider(),
        planner=planner(),
        arm_token="test",
        clock=lambda: 1_010_000_000,
    )

    result = runtime.step("blocked-0")

    assert not result.accepted
    assert "tracker_lost" in result.reason
    assert backend.submit_calls == 0
