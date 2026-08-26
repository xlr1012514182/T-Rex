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


class _ReadFailsStopWorks(MockRevoBackend):
    async def read_state(self):
        raise RuntimeError("telemetry unavailable")


class _ReadAndStopFail(_ReadFailsStopWorks):
    async def soft_stop(self, reason):
        del reason
        raise RuntimeError("stop transport unavailable")


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


def test_final_safety_vetoes_fresh_firmware_stall_status():
    supervisor = SafetySupervisor(SafetyEnvelope.demo())
    stalled = RevoState(
        timestamp_ns=1_000,
        q_rad=np.zeros(21),
        status=np.array([1 << 8] + [0] * 20),
    )
    result = supervisor.authorize(
        np.ones(21),
        stalled,
        now_ns=1_001,
        context=SafetyContext(),
        emg_requests_close=True,
    )
    assert result.vetoed
    assert "motor_fault_or_stall" in result.reason


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


def test_abort_attempts_and_confirms_soft_stop_even_when_telemetry_read_fails():
    async def run():
        backend = _ReadFailsStopWorks()
        pipeline = RevoCommandPipeline(
            backend, SafetySupervisor(SafetyEnvelope.demo())
        )
        result = await pipeline.abort("watchdog")
        assert result.soft_stop_succeeded
        assert pipeline.soft_stop_confirmed
        assert backend.soft_stop_reasons
        assert "telemetry_read_failed" in result.reason

    asyncio.run(run())


def test_abort_never_claims_stop_when_telemetry_and_soft_stop_both_fail():
    async def run():
        pipeline = RevoCommandPipeline(
            _ReadAndStopFail(), SafetySupervisor(SafetyEnvelope.demo())
        )
        first = await pipeline.abort("watchdog")
        second = await pipeline.abort("shutdown_retry")
        assert not first.soft_stop_succeeded
        assert not second.soft_stop_succeeded
        assert pipeline.hard_fault_latched
        assert not pipeline.soft_stop_confirmed

    asyncio.run(run())


def test_pipeline_polls_backend_collision_before_every_single_write():
    async def run():
        backend = MockRevoBackend()
        backend.set_collision(True)
        pipeline = RevoCommandPipeline(
            backend, SafetySupervisor(SafetyEnvelope.demo())
        )
        result = await pipeline.execute(
            nominal_q_rad=np.zeros(21),
            residual_q_rad=None,
            task_id="task",
            task_version=1,
            source_chunk_id="chunk",
            safety_context=SafetyContext(),
            emg_requests_close=True,
            now_ns=10,
            state=RevoState(10, np.zeros(21)),
        )
        assert result.vetoed
        assert "collision_active" in result.reason
        assert result.hard_fault_latched
        assert result.clear_policy_cache
        assert result.soft_stop_succeeded
        assert backend.soft_stop_reasons
        assert not backend.commands

    asyncio.run(run())


def test_non_hard_fault_holds_without_soft_stop_or_automatic_open():
    async def run():
        backend = MockRevoBackend(np.full(21, 0.4, dtype=np.float32))
        pipeline = RevoCommandPipeline(
            backend, SafetySupervisor(SafetyEnvelope.demo())
        )
        result = await pipeline.execute(
            nominal_q_rad=np.zeros(21),
            residual_q_rad=None,
            task_id="task",
            task_version=1,
            source_chunk_id="chunk",
            safety_context=SafetyContext(
                policy_aborted=True, holding_object=True
            ),
            emg_requests_close=False,
            now_ns=10,
            state=RevoState(10, np.full(21, 0.4, dtype=np.float32)),
        )
        assert result.vetoed
        assert not result.hard_fault_latched
        assert not backend.soft_stop_reasons
        np.testing.assert_allclose(result.q_authorized_rad, 0.4)

    asyncio.run(run())


def test_safety_vetoes_velocity_and_temperature_thresholds():
    envelope = SafetyEnvelope(
        q_min_rad=np.full(21, -1.0),
        q_max_rad=np.full(21, 1.0),
        max_step_rad=np.full(21, 0.1),
        max_abs_velocity_rad_s=1.0,
        max_abs_acceleration_rad_s2=10.0,
        max_abs_current_a=2.0,
        max_temperature_c=45.0,
        require_temperature_telemetry=True,
    )
    state = RevoState(
        timestamp_ns=100,
        q_rad=np.zeros(21),
        dq_rad_s=np.full(21, 2.0),
        temperature_c=np.full(21, 50.0),
    )
    result = SafetySupervisor(envelope).authorize(
        np.zeros(21), state, now_ns=100, context=SafetyContext()
    )
    assert result.vetoed
    assert "over_velocity" in result.reason
    assert "over_temperature" in result.reason


def test_acceleration_is_derived_only_after_two_fresh_samples():
    envelope = SafetyEnvelope(
        q_min_rad=np.full(21, -1.0),
        q_max_rad=np.full(21, 1.0),
        max_step_rad=np.full(21, 0.1),
        max_abs_velocity_rad_s=100.0,
        max_abs_acceleration_rad_s2=5.0,
        max_abs_current_a=2.0,
    )
    supervisor = SafetySupervisor(envelope)
    first = supervisor.authorize(
        np.zeros(21),
        RevoState(1_000_000_000, np.zeros(21), dq_rad_s=np.zeros(21), sequence=1),
        now_ns=1_000_000_000,
        context=SafetyContext(),
    )
    assert not first.vetoed
    second = supervisor.authorize(
        np.zeros(21),
        RevoState(
            1_100_000_000,
            np.zeros(21),
            dq_rad_s=np.ones(21),
            sequence=2,
        ),
        now_ns=1_100_000_000,
        context=SafetyContext(),
    )
    assert second.vetoed
    assert "over_acceleration" in second.reason


def test_command_velocity_and_acceleration_are_checked_on_100hz_writer_grid():
    envelope = SafetyEnvelope(
        q_min_rad=np.full(21, -1.0),
        q_max_rad=np.full(21, 1.0),
        max_step_rad=np.full(21, 1.0),
        max_abs_velocity_rad_s=2.0,
        max_abs_acceleration_rad_s2=100.0,
        max_abs_current_a=2.0,
        command_period_ns=10_000_000,
    )
    supervisor = SafetySupervisor(envelope)
    too_fast = supervisor.authorize(
        np.full(21, 0.1),
        RevoState(100, np.zeros(21), sequence=1),
        now_ns=100,
        context=SafetyContext(),
    )
    assert too_fast.vetoed
    assert too_fast.reason == "command_over_velocity"

    acceleration_envelope = SafetyEnvelope(
        q_min_rad=np.full(21, -1.0),
        q_max_rad=np.full(21, 1.0),
        max_step_rad=np.full(21, 1.0),
        max_abs_velocity_rad_s=100.0,
        max_abs_acceleration_rad_s2=50.0,
        max_abs_current_a=2.0,
        command_period_ns=10_000_000,
    )
    supervisor = SafetySupervisor(acceleration_envelope)
    first = supervisor.authorize(
        np.full(21, 0.001),
        RevoState(100, np.zeros(21), sequence=1),
        now_ns=100,
        context=SafetyContext(),
    )
    assert not first.vetoed
    second = supervisor.authorize(
        np.full(21, 0.02),
        RevoState(10_000_100, np.full(21, 0.001), sequence=2),
        now_ns=10_000_100,
        context=SafetyContext(),
    )
    assert second.vetoed
    assert second.reason == "command_over_acceleration"


def test_state_stale_and_backend_write_failure_latch_soft_stop_and_clear_cache():
    class FailingWriteBackend(MockRevoBackend):
        async def write_command(self, command):
            raise RuntimeError("injected write failure")

    async def run():
        stale_backend = MockRevoBackend()
        stale_pipeline = RevoCommandPipeline(
            stale_backend, SafetySupervisor(SafetyEnvelope.demo())
        )
        stale = await stale_pipeline.execute(
            nominal_q_rad=np.zeros(21),
            residual_q_rad=None,
            task_id="task",
            task_version=1,
            source_chunk_id="chunk",
            safety_context=SafetyContext(),
            emg_requests_close=False,
            now_ns=2_000_000_000,
            state=RevoState(1, np.zeros(21)),
        )
        assert stale.vetoed and stale.soft_stop_succeeded and stale.clear_policy_cache
        assert "state_stale" in stale.reason

        write_backend = FailingWriteBackend()
        write_pipeline = RevoCommandPipeline(
            write_backend, SafetySupervisor(SafetyEnvelope.demo())
        )
        failed = await write_pipeline.execute(
            nominal_q_rad=np.zeros(21),
            residual_q_rad=None,
            task_id="task",
            task_version=1,
            source_chunk_id="chunk",
            safety_context=SafetyContext(),
            emg_requests_close=False,
            now_ns=10,
            state=RevoState(10, np.zeros(21)),
        )
        assert failed.vetoed and failed.soft_stop_succeeded and failed.clear_policy_cache
        assert "backend_write_failed" in failed.reason

    asyncio.run(run())


def test_missing_temperature_fails_closed_when_real_profile_requires_it():
    envelope = SafetyEnvelope(
        q_min_rad=np.full(21, -1.0),
        q_max_rad=np.full(21, 1.0),
        max_step_rad=np.full(21, 0.1),
        max_abs_velocity_rad_s=1.0,
        max_abs_current_a=2.0,
        max_temperature_c=45.0,
        require_temperature_telemetry=True,
    )
    result = SafetySupervisor(envelope).authorize(
        np.zeros(21), RevoState(100, np.zeros(21)),
        now_ns=100, context=SafetyContext(),
    )
    assert result.vetoed
    assert "temperature_telemetry_missing" in result.reason


def test_holding_abort_never_turns_into_an_open_command():
    state = RevoState(100, np.full(21, 0.4))
    result = SafetySupervisor(SafetyEnvelope.demo()).authorize(
        np.zeros(21),
        state,
        now_ns=100,
        context=SafetyContext(policy_aborted=True, holding_object=True),
    )
    assert result.vetoed
    assert "policy_aborted_hold" in result.reason
    np.testing.assert_array_equal(result.q_authorized_rad, state.q_rad)


def test_real_writer_rejects_demo_or_unidentified_safety_profile_before_write():
    class FakeHardwareBackend(MockRevoBackend):
        def __init__(self):
            super().__init__()
            self.is_hardware = True

    async def run():
        backend = FakeHardwareBackend()
        pipeline = RevoCommandPipeline(
            backend, SafetySupervisor(SafetyEnvelope.demo())
        )
        try:
            await pipeline.execute(
                nominal_q_rad=np.zeros(21),
                residual_q_rad=None,
                task_id="task",
                task_version=1,
                source_chunk_id="chunk",
                safety_context=SafetyContext(),
                emg_requests_close=False,
                now_ns=10,
            )
        except RuntimeError as exc:
            assert "explicit, non-demo safety profile" in str(exc)
        else:
            raise AssertionError("real writer accepted a demo safety envelope")
        assert not backend.commands

    asyncio.run(run())
