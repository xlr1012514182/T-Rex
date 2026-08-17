import asyncio

import numpy as np

from revo3_v1.revo import (
    MockRevoBackend,
    RevoCommandPipeline,
    RevoState,
    SafetyContext,
    SafetyEnvelope,
    SafetySupervisor,
)


def _state(timestamp_ns=1_000):
    return RevoState(timestamp_ns, np.zeros(21))


def test_safety_clips_joint_range_and_per_tick_step():
    envelope = SafetyEnvelope(
        q_min_rad=np.full(21, -0.5),
        q_max_rad=np.full(21, 0.5),
        max_step_rad=np.full(21, 0.1),
        max_state_age_ns=100,
        max_abs_current_a=1.0,
    )
    result = SafetySupervisor(envelope).authorize(
        np.full(21, 2.0),
        _state(),
        now_ns=1_050,
        context=SafetyContext(),
        emg_requests_close=True,
    )
    assert not result.vetoed
    assert result.clipped
    np.testing.assert_allclose(result.q_authorized_rad, 0.1)


def test_final_safety_vetoes_emg_close_on_collision():
    supervisor = SafetySupervisor(SafetyEnvelope.demo())
    result = supervisor.authorize(
        np.ones(21),
        _state(),
        now_ns=1_001,
        context=SafetyContext(collision_active=True),
        emg_requests_close=True,
    )
    assert result.vetoed
    assert "collision_active" in result.reason
    np.testing.assert_array_equal(result.q_authorized_rad, np.zeros(21))


def test_pipeline_never_writes_a_vetoed_target():
    async def run():
        backend = MockRevoBackend()
        pipeline = RevoCommandPipeline(
            backend, SafetySupervisor(SafetyEnvelope.demo())
        )
        result = await pipeline.execute(
            nominal_q_rad=np.ones(21),
            residual_q_rad=np.full(21, 0.01),
            task_id="task",
            task_version=1,
            source_chunk_id="chunk",
            safety_context=SafetyContext(tactile_overload=True),
            emg_requests_close=True,
            now_ns=10,
            state=RevoState(10, np.zeros(21)),
        )
        assert result.vetoed
        assert not backend.commands

    asyncio.run(run())
