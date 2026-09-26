from __future__ import annotations

import asyncio
from pathlib import Path
import sys
import time

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from revo3_teleop.backends import TeleopRevoWriter
from revo3_v1.revo import (
    MockRevoBackend,
    RevoCommandPipeline,
    SafetyContext,
    SafetyEnvelope,
    SafetySupervisor,
)


def test_revo_writer_records_only_the_safety_authorized_controller_target() -> None:
    async def scenario():
        backend = MockRevoBackend()
        writer = TeleopRevoWriter(
            RevoCommandPipeline(
                backend,
                SafetySupervisor(SafetyEnvelope.demo(max_step_rad=0.05)),
            )
        )
        state = await backend.read_state()
        now = max(time.monotonic_ns(), state.timestamp_ns)
        receipt = await writer.submit_target(
            request_id="hand-0",
            nominal_q_rad=np.full(21, 0.7, np.float32),
            task_id="bottle",
            task_version=1,
            safety_context=SafetyContext(),
            decision_timestamp_ns=now,
            state=state,
        )
        return backend, receipt

    backend, receipt = asyncio.run(scenario())
    assert receipt.accepted
    assert receipt.clipped
    np.testing.assert_allclose(receipt.requested_target, 0.7)
    np.testing.assert_allclose(receipt.exact_sent_target, 0.05)
    np.testing.assert_allclose(backend.commands[-1].q_target_rad, receipt.exact_sent_target)


def test_revo_writer_keeps_safety_veto_out_of_supervision() -> None:
    async def scenario():
        backend = MockRevoBackend()
        writer = TeleopRevoWriter(
            RevoCommandPipeline(backend, SafetySupervisor(SafetyEnvelope.demo()))
        )
        state = await backend.read_state()
        return await writer.submit_target(
            request_id="hand-veto",
            nominal_q_rad=np.ones(21, np.float32),
            task_id="phone",
            task_version=1,
            safety_context=SafetyContext(emergency_stop=True),
            decision_timestamp_ns=max(time.monotonic_ns(), state.timestamp_ns),
            state=state,
        )

    receipt = asyncio.run(scenario())
    assert not receipt.accepted
    assert receipt.exact_sent_target is None
    assert "emergency_stop" in receipt.reason
