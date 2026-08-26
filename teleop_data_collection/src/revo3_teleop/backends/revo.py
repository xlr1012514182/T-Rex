"""Controller-boundary receipt adapter for the existing Revo safety path."""

from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np

from revo3_v1.revo import (
    JOINT_ORDER_HASH,
    RevoCommandPipeline,
    RevoState,
    SafetyContext,
    assert_joint_vector,
)

from revo3_teleop.contracts import CommandReceipt


class TeleopRevoWriter:
    """Submit teleoperation targets through the sole Revo write path.

    A successful return means ``RevoCommandPipeline`` completed the backend
    write.  Only then is ``exact_sent_target`` populated and eligible as an
    imitation-learning label.  Safety vetoes remain diagnostic receipts and
    can never supervise a training anchor.
    """

    def __init__(
        self,
        pipeline: RevoCommandPipeline,
        *,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.pipeline = pipeline
        self._clock_ns = clock_ns
        self._controller_sequence = 0
        self._last_write_timestamp_ns = -1

    async def submit_target(
        self,
        *,
        request_id: str,
        nominal_q_rad: np.ndarray,
        residual_q_rad: Optional[np.ndarray] = None,
        task_id: str,
        task_version: int,
        safety_context: SafetyContext,
        emg_requests_close: bool = False,
        source_chunk_id: Optional[str] = None,
        decision_timestamp_ns: Optional[int] = None,
        state: Optional[RevoState] = None,
    ) -> CommandReceipt:
        decision_ns = (
            self._clock_ns()
            if decision_timestamp_ns is None
            else int(decision_timestamp_ns)
        )
        nominal = assert_joint_vector(nominal_q_rad, name="nominal_q_rad")
        residual = (
            np.zeros_like(nominal)
            if residual_q_rad is None
            else assert_joint_vector(residual_q_rad, name="residual_q_rad")
        )
        requested = nominal + residual
        result = await self.pipeline.execute(
            nominal_q_rad=nominal,
            residual_q_rad=residual,
            task_id=task_id,
            task_version=task_version,
            source_chunk_id=source_chunk_id,
            safety_context=safety_context,
            emg_requests_close=emg_requests_close,
            now_ns=decision_ns,
            state=state,
        )
        if result.vetoed:
            return CommandReceipt(
                request_id=request_id,
                component="revo_hand",
                accepted=False,
                requested_target=requested,
                authorized_target=result.q_authorized_rad,
                decision_timestamp_ns=decision_ns,
                clipped=result.clipped,
                reason=result.reason,
                unit="rad",
                joint_order_hash=JOINT_ORDER_HASH,
            )

        # ``execute`` awaited the backend write before returning.  This local
        # monotonically increasing sequence is the collector's write-boundary
        # sequence; it is not presented as motor feedback sequence.
        write_ns = max(
            self._clock_ns(),
            decision_ns,
            self._last_write_timestamp_ns + 1,
        )
        self._last_write_timestamp_ns = write_ns
        sequence = self._controller_sequence
        self._controller_sequence += 1
        return CommandReceipt(
            request_id=request_id,
            component="revo_hand",
            accepted=True,
            requested_target=requested,
            authorized_target=result.q_authorized_rad,
            exact_sent_target=result.q_authorized_rad,
            decision_timestamp_ns=decision_ns,
            write_timestamp_ns=write_ns,
            controller_sequence=sequence,
            clipped=result.clipped,
            reason=result.reason,
            unit="rad",
            joint_order_hash=JOINT_ORDER_HASH,
        )
