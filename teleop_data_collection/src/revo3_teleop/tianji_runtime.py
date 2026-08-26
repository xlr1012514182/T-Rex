"""Composed wrist-pose -> retarget -> IK -> fail-closed Tianji write path."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

from revo3_teleop.backends.tianji import TianjiMarvinBackend
from revo3_teleop.contracts import CommandReceipt
from revo3_teleop.retargeting import (
    PlannedArmTarget,
    TianjiJointTargetPlanner,
    WristPoseProvider,
)


@dataclass(frozen=True)
class TianjiRuntimeStep:
    request_id: str
    accepted: bool
    reason: str
    receipt: CommandReceipt | None = None
    plan: PlannedArmTarget | None = None


class TianjiTeleopRuntime:
    """One explicitly armed Tianji control boundary.

    Planner/provider failures never invoke a position write.  Backend safety
    vetoes remain authoritative and are returned verbatim in the receipt.
    """

    def __init__(
        self,
        *,
        backend: TianjiMarvinBackend,
        wrist_pose_provider: WristPoseProvider,
        planner: TianjiJointTargetPlanner,
        arm_token: str | None,
        clock: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if not bool(getattr(wrist_pose_provider, "provides_wrist_pose", False)):
            raise ValueError("wrist_pose_provider must explicitly provide verified 6-DoF")
        self.backend = backend
        self.wrist_pose_provider = wrist_pose_provider
        self.planner = planner
        self.arm_token = arm_token
        self.clock = clock

    def step(self, request_id: str) -> TianjiRuntimeStep:
        request = str(request_id).strip()
        if not request:
            raise ValueError("request_id must be non-empty")
        try:
            # This read supplies the IK seed.  submit_target performs its own
            # later feedback read and all final watchdog/safety checks.
            state = self.backend.read_state()
            pose = self.wrist_pose_provider.read_pose()
            now_ns = int(self.clock())
            plan = self.planner.plan(pose, seed_q_rad=state.q_rad, now_ns=now_ns)
        except Exception as exc:
            return TianjiRuntimeStep(
                request_id=request,
                accepted=False,
                reason=f"planning_blocked:{type(exc).__name__}:{exc}",
            )
        receipt = self.backend.submit_target(
            request_id=request,
            q_target_rad=plan.q_rad,
            target_timestamp_ns=plan.target_timestamp_ns,
            arm_token=self.arm_token,
            wrist_pose_valid=True,
            decision_timestamp_ns=int(self.clock()),
        )
        return TianjiRuntimeStep(
            request_id=request,
            accepted=receipt.accepted,
            reason=receipt.reason,
            receipt=receipt,
            plan=plan,
        )


__all__ = ["TianjiRuntimeStep", "TianjiTeleopRuntime"]
