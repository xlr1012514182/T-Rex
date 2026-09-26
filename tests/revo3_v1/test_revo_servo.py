from __future__ import annotations

import asyncio
from dataclasses import replace
import hashlib
import time

import numpy as np

from revo3_v1.executive import MotionDirective, TaskLease
from revo3_v1.policy import (
    ActionChunk,
    AsyncTReXPolicyRunner,
    InferenceMode,
    MockTReXBackend,
    TReXRevoPolicyAdapter,
    TaskKey,
)
from revo3_v1.revo import (
    MockRevoBackend,
    RevoCommandPipeline,
    RevoState,
    SafetyContext,
    SafetyEnvelope,
    SafetySupervisor,
)
from revo3_v1.revo.servo import RevoServoConfig, RevoServoExecutor
from revo3_v1.tactile import ReflexConfig, ReflexPhase, TactileReflexPlugin


INSTRUCTION = "Grasp the centered bottle using a power grasp."


def lease() -> TaskLease:
    return TaskLease(
        task_id="bottle",
        task_version=1,
        lease_id="lease-a",
        instruction=INSTRUCTION,
        instruction_hash=hashlib.sha256(INSTRUCTION.encode()).hexdigest(),
        version_fingerprint="versions",
        issued_at_ns=0,
        expires_at_ns=2_000_000_000,
        primitive="POWER_GRASP",
    )


def key(value: TaskLease) -> TaskKey:
    return TaskKey(
        value.task_id,
        value.task_version,
        value.instruction_hash,
        value.lease_id,
        value.version_fingerprint,
    )


def state(timestamp_ns: int, q: float = 0.0, sequence: int = 0) -> RevoState:
    return RevoState(
        timestamp_ns,
        np.full(21, q, np.float32),
        sequence=sequence,
        temperature_c=np.full(21, 25.0, np.float32),
    )


def components():
    backend = MockRevoBackend()
    pipeline = RevoCommandPipeline(backend, SafetySupervisor(SafetyEnvelope.demo()))
    policy = AsyncTReXPolicyRunner(TReXRevoPolicyAdapter(MockTReXBackend()))
    cair = TactileReflexPlugin(ReflexConfig(enabled=False))
    servo = RevoServoExecutor(pipeline, policy, cair, RevoServoConfig.demo())
    return backend, policy, servo


def test_servo_linearly_interpolates_30hz_chunk_at_100hz_through_single_writer() -> None:
    async def run() -> None:
        backend, policy, servo = components()
        task_lease = lease()
        actions = np.stack(
            [np.full(21, 0.01 * (index + 1), np.float32) for index in range(16)]
        )
        policy.aggregator.add(
            ActionChunk(
                actions,
                start_step=0,
                generated_ns=0,
                observation_timestamp_ns=0,
                task_key=key(task_lease),
                mode=InferenceMode.SLOW_AND_FAST,
                chunk_id="chunk-a",
            ),
            now_ns=0,
        )
        first = await servo.step(
            now_ns=0,
            directive=MotionDirective.POLICY,
            lease=task_lease,
            action_epoch_ns=0,
            tactile=None,
            reflex_phase=ReflexPhase.PRECONTACT,
            state=state(0, 0.0, 0),
            source_chunk_id="chunk-a",
        )
        second = await servo.step(
            now_ns=10_000_000,
            directive=MotionDirective.POLICY,
            lease=task_lease,
            action_epoch_ns=0,
            tactile=None,
            reflex_phase=ReflexPhase.PRECONTACT,
            state=state(10_000_000, 0.01, 1),
            source_chunk_id="chunk-a",
        )
        assert first.wrote_command and second.wrote_command
        np.testing.assert_allclose(first.nominal_q_rad, 0.01, atol=1e-6)
        # 10 ms is 0.3 of one 33.333 ms action interval.
        np.testing.assert_allclose(second.nominal_q_rad, 0.013, atol=1e-5)
        assert len(backend.commands) == 2
        assert policy.close()

    asyncio.run(run())


def test_hold_is_latched_and_abort_softstops_without_opening() -> None:
    async def run() -> None:
        backend, policy, servo = components()
        task_lease = lease()
        held = await servo.step(
            now_ns=0,
            directive=MotionDirective.HOLD_POSITION,
            lease=task_lease,
            action_epoch_ns=None,
            tactile=None,
            reflex_phase=ReflexPhase.HOLD,
            state=state(0, 0.2, 0),
        )
        assert held.wrote_command
        np.testing.assert_allclose(backend.commands[-1].q_target_rad, 0.2)
        aborted = await servo.step(
            now_ns=10_000_000,
            directive=MotionDirective.SAFE_STOP,
            lease=task_lease,
            action_epoch_ns=None,
            tactile=None,
            reflex_phase=ReflexPhase.HOLD,
            state=state(10_000_000, 0.2, 1),
        )
        assert not aborted.wrote_command
        assert aborted.safety.soft_stop_succeeded
        assert backend.soft_stop_reasons
        assert len(backend.commands) == 1
        assert policy.close()

    asyncio.run(run())


class FreshReadHardwareBackend(MockRevoBackend):
    def __init__(self) -> None:
        super().__init__(np.full(21, 0.2, np.float32))
        self.is_hardware = True
        self.read_calls = 0

    async def read_state(self) -> RevoState:
        self.read_calls += 1
        return await super().read_state()


def hardware_envelope() -> SafetyEnvelope:
    return SafetyEnvelope(
        q_min_rad=np.full(21, -1.0, np.float32),
        q_max_rad=np.full(21, 1.0, np.float32),
        max_step_rad=np.full(21, 0.1, np.float32),
        max_state_age_ns=1_000_000_000,
        max_abs_current_a=np.full(21, 5.0, np.float32),
        max_abs_velocity_rad_s=np.full(21, 100.0, np.float32),
        max_abs_acceleration_rad_s2=np.full(21, 1000.0, np.float32),
        max_temperature_c=np.full(21, 80.0, np.float32),
        require_temperature_telemetry=True,
        hardware_profile_id="fixture-only-hardware-profile",
        simulation_only=False,
    )


def test_hardware_servo_never_passes_aligned_caller_state_into_pipeline() -> None:
    async def run() -> None:
        backend = FreshReadHardwareBackend()
        pipeline = RevoCommandPipeline(
            backend, SafetySupervisor(hardware_envelope())
        )
        policy = AsyncTReXPolicyRunner(TReXRevoPolicyAdapter(MockTReXBackend()))
        servo = RevoServoExecutor(
            pipeline,
            policy,
            TactileReflexPlugin(ReflexConfig(enabled=False)),
            RevoServoConfig(
                np.zeros(21, np.float32),
                hardware_mode=True,
                calibration_id="fixture-only-safe-open",
            ),
        )
        now = time.monotonic_ns()
        caller_state = state(now, 0.9, 999)
        with np.testing.assert_raises_regex(
            RuntimeError, "forbids caller-injected state"
        ):
            await pipeline.execute(
                nominal_q_rad=np.full(21, 0.2, np.float32),
                residual_q_rad=None,
                task_id="fixture",
                task_version=1,
                source_chunk_id=None,
                safety_context=SafetyContext(),
                emg_requests_close=False,
                now_ns=now,
                state=caller_state,
            )
        task_lease = replace(
            lease(), issued_at_ns=now, expires_at_ns=now + 2_000_000_000
        )
        result = await servo.step(
            now_ns=now,
            directive=MotionDirective.HOLD_POSITION,
            lease=task_lease,
            action_epoch_ns=None,
            tactile=None,
            reflex_phase=ReflexPhase.HOLD,
            state=caller_state,
        )
        assert result.wrote_command
        assert backend.read_calls >= 2
        np.testing.assert_allclose(result.nominal_q_rad, 0.2, atol=1e-6)
        np.testing.assert_allclose(backend.commands[-1].q_target_rad, 0.2, atol=1e-6)
        assert policy.close()

    asyncio.run(run())
