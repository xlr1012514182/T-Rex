"""Deterministic end-to-end mock of the Revo3 V1 runtime.

This module exercises real contracts and safety ownership but uses a mock VLM,
mock T-Rex policy and in-memory hand.  It never claims task success and never
writes hardware.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
import json
from pathlib import Path
import time
from typing import Dict, List

import numpy as np

from revo3_v1.emg.streaming import BinaryIntentGate, IntentGateConfig
from revo3_v1.executive import (
    CompletionState,
    EmgEvent,
    EmgIntent,
    ExecutiveTick,
    ModalityTimestamps,
    PolicyResponseEnvelope,
    RuntimeVersions,
    SafetySignal,
    TaskExecutive,
    TaskExecutiveConfig,
)
from revo3_v1.planner import (
    AskToClarifyPlanner,
    MockPlannerBackend,
    PlannerRequest,
    SupportedTask,
    VisualGate,
    VisualGateConfig,
)
from revo3_v1.policy import (
    MAIN_ALIGNED_SCHEDULE,
    MockTReXBackend,
    PolicyObservation,
    TReXPolicyRunner,
    TReXRevoPolicyAdapter,
    TaskKey,
)
from revo3_v1.revo import (
    JOINT_ORDER_HASH,
    MockRevoBackend,
    RevoCommandPipeline,
    SafetyContext,
    SafetyEnvelope,
    SafetySupervisor,
)
from revo3_v1.tactile import (
    DenseTactileBuffer,
    ReflexConfig,
    ReflexPhase,
    TactileFrame,
    TactileReflexPlugin,
)
from revo3_v1.timing import (
    AlignmentMode,
    CausalTimestampAligner,
    StreamConfig,
    TimestampedSample,
)


@dataclass(frozen=True)
class DemoConfig:
    task: SupportedTask = SupportedTask.BOTTLE
    visual_area_threshold: float = 0.15
    emg_close_probability: float = 0.95
    emulate_release: bool = True
    output_trace: str = ""


def _versions() -> RuntimeVersions:
    return RuntimeVersions(
        schema_version="revo3-v1-demo",
        planner_revision="mock-ask-to-clarify-v1",
        policy_revision="mock-trex-21d-v1",
        hardware_manifest_hash="simulation-only",
        joint_order_hash=JOINT_ORDER_HASH,
        tactile_profile_hash="mock-f6-five-finger-v1",
    )


def _tactile_frame(timestamp_ns: int, sequence: int, normal: float) -> TactileFrame:
    values = np.zeros((5, 6), dtype=np.float32)
    values[:, 2] = normal
    return TactileFrame(timestamp_ns, values, sequence)


async def run_mock_demo(config: DemoConfig = DemoConfig()) -> Dict[str, object]:
    """Run CLOSE + visual start, one full chunk, then optional controlled open."""

    base = time.monotonic_ns() + 400_000_000
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    planner = AskToClarifyPlanner(MockPlannerBackend(default_task=config.task))
    visual_gate = VisualGate(
        VisualGateConfig(
            min_area=config.visual_area_threshold,
            min_consecutive_ready=2,
        )
    )
    first = planner.plan(
        PlannerRequest(emg_action="CLOSE", images=(frame,), timestamp_ns=base - 1_000_000)
    )
    visual_gate.update(first, now_ns=base - 1_000_000)
    decision = planner.plan(
        PlannerRequest(emg_action="CLOSE", images=(frame,), timestamp_ns=base)
    )
    visual = visual_gate.update(decision, now_ns=base)
    if not visual.ready:
        raise RuntimeError(f"mock visual context did not become ready: {visual.reason}")

    # Convert sustained model probability into one debounced edge event.  No
    # raw EMG or EMG embedding is ever passed into the VLA policy.
    intent_gate = BinaryIntentGate(
        IntentGateConfig(close_dwell_ms=300, open_dwell_ms=500)
    )
    intent_gate.update(config.emg_close_probability, base - 300_000_000, 1.0)
    start_event = intent_gate.update(config.emg_close_probability, base, 1.0)
    if start_event is None:
        raise RuntimeError("mock EMG close failed to pass the intent dwell")

    touch_buffer = DenseTactileBuffer()
    touch_period = 8_333_333  # mock 120 Hz sensor stream
    for index in range(16):
        touch_buffer.append(
            _tactile_frame(base - (15 - index) * touch_period, index, 0.01)
        )
    touch = touch_buffer.snapshot(now_ns=base, max_age_ns=150_000_000)

    # The trigger grid uses an exact causal GCD (50/30/100/120 -> 10 Hz).
    # The actuator still executes T-Rex actions at 30 Hz below.
    aligner = CausalTimestampAligner(
        (
            StreamConfig("emg", 50, max_age_ns=100_000_000),
            StreamConfig("camera", 30, max_age_ns=100_000_000),
            StreamConfig("state", 100, max_age_ns=50_000_000),
            StreamConfig("touch", 120, max_age_ns=150_000_000),
        ),
        mode=AlignmentMode.GCD,
        epoch_ns=base,
    )
    for name, value, timestamp in (
        ("emg", start_event.to_dict(), start_event.timestamp_ns),
        ("camera", frame, decision.timestamp_ns),
        ("state", np.zeros(21, dtype=np.float32), base),
        ("touch", touch.current, touch.newest_timestamp_ns),
    ):
        aligner.offer(name, TimestampedSample(timestamp, value))
    aligned = aligner.align_at(base)
    if not aligned.ready:
        raise RuntimeError(f"mock trigger alignment failed: {aligned.missing}/{aligned.stale}")

    executive = TaskExecutive(
        TaskExecutiveConfig(
            camera_ttl_ns=150_000_000,
            state_ttl_ns=150_000_000,
            touch_ttl_ns=200_000_000,
        ),
        id_factory=iter((f"demo-id-{i}" for i in range(20))).__next__,
    )
    versions = _versions()
    start = executive.step(
        ExecutiveTick(
            now_ns=base,
            emg=EmgEvent(
                EmgIntent.CLOSE,
                start_event.timestamp_ns,
                start_event.confidence,
                start_event.event_id,
            ),
            visual=visual,
            planner=decision,
            timestamps=ModalityTimestamps(
                camera_ns=decision.timestamp_ns,
                state_ns=base,
                touch_ns=touch.newest_timestamp_ns,
                policy_ns=None,
            ),
            versions=versions,
        )
    )
    if start.output.value != "START" or start.lease is None:
        raise RuntimeError(f"Task Executive did not START: {start.reason}")

    lease = start.lease
    key = TaskKey.from_instruction(
        task_id=lease.task_id,
        task_version=lease.task_version,
        instruction=lease.instruction,
        lease_id=lease.lease_id,
    )
    if key.instruction_hash != lease.instruction_hash:
        raise RuntimeError("planner/task/policy instruction hash disagreement")

    hand = MockRevoBackend()
    demo_envelope = replace(
        SafetyEnvelope.demo(max_step_rad=0.06), max_state_age_ns=10_000_000_000
    )
    command_path = RevoCommandPipeline(
        hand, SafetySupervisor(demo_envelope)
    )
    policy = TReXPolicyRunner(TReXRevoPolicyAdapter(MockTReXBackend()))
    policy.reset("task_start")
    reflex = TactileReflexPlugin(ReflexConfig.demo())
    action_period = MAIN_ALIGNED_SCHEDULE.action_period_ns
    current_chunk = None
    policy_timestamp = base
    trace: List[Dict[str, object]] = []

    for offset in range(MAIN_ALIGNED_SCHEDULE.chunk_size):
        now = base + offset * action_period
        normal = 0.01 if offset < 7 else min(0.55, 0.10 + 0.06 * (offset - 7))
        # Offset zero reuses the newest real trigger sample; policy requests
        # must never fabricate a duplicate sensor frame.
        tactile_sequence = 15 + offset
        if offset > 0:
            touch_buffer.append(_tactile_frame(now, tactile_sequence, normal))
        touch = touch_buffer.snapshot(now_ns=now, max_age_ns=150_000_000)
        state = await hand.read_state()
        mode = MAIN_ALIGNED_SCHEDULE.mode_at_chunk_offset(offset)
        observation = PolicyObservation(
            timestamp_ns=now,
            # The mock backend read completes at this simulated poll tick.
            # Hardware uses the acquisition timestamp returned by its adapter.
            state_timestamp_ns=now,
            rgb_timestamp_ns=now,
            tactile_timestamp_ns=touch.newest_timestamp_ns,
            q_rad=state.q_rad,
            tactile_f6=touch.current,
            tactile_history_f6=touch.f6,
            instruction=lease.instruction,
            task_key=key,
            images={"full": frame},
        )
        generated_chunk = policy.infer_if_due(
            global_step=offset,
            observation=observation,
            now_ns=now,
        )
        if generated_chunk is not None:
            current_chunk = generated_chunk
            policy_timestamp = now
            envelope = PolicyResponseEnvelope(
                task_id=lease.task_id,
                task_version=lease.task_version,
                lease_id=lease.lease_id,
                instruction_hash=lease.instruction_hash,
                version_fingerprint=lease.version_fingerprint,
                observation_timestamp_ns=now,
                produced_at_ns=now,
            )
            accepted, reason = executive.validate_policy_response(envelope, now_ns=now)
            if not accepted:
                raise RuntimeError(f"Task Executive rejected mock policy: {reason}")
        if current_chunk is None:
            raise RuntimeError("no action chunk at execution start")

        nominal_target = policy.target_for_step(
            global_step=offset, task_key=key, now_ns=now
        )

        reflex_result = reflex.update(
            _tactile_frame(touch.newest_timestamp_ns, tactile_sequence, normal),
            phase=ReflexPhase.PRECONTACT if offset < 7 else ReflexPhase.HOLD,
        )
        result = await command_path.execute(
            nominal_q_rad=nominal_target,
            residual_q_rad=reflex_result.residual_q_rad,
            task_id=lease.task_id,
            task_version=lease.task_version,
            source_chunk_id=current_chunk.chunk_id,
            safety_context=SafetyContext(tactile_overload=reflex_result.hard_overload),
            emg_requests_close=True,
            now_ns=now,
            state=state,
        )
        if result.vetoed:
            raise RuntimeError(f"mock command was vetoed: {result.reason}")
        active = executive.step(
            ExecutiveTick(
                now_ns=now,
                emg=EmgEvent(EmgIntent.REST, now, 1.0),
                visual=None,
                planner=None,
                timestamps=ModalityTimestamps(now, now, now, policy_timestamp),
                versions=versions,
                safety=SafetySignal(),
            )
        )
        trace.append(
            {
                "step": offset,
                "mode": mode.value,
                "executive": active.output.value,
                "reflex": reflex_result.reason,
                "command_clipped": result.clipped,
            }
        )

    finish_time = base + MAIN_ALIGNED_SCHEDULE.chunk_size * action_period
    stable = executive.step(
        ExecutiveTick(
            now_ns=finish_time,
            emg=EmgEvent(EmgIntent.REST, finish_time),
            visual=None,
            planner=None,
            timestamps=ModalityTimestamps(finish_time, finish_time, finish_time, policy_timestamp),
            versions=versions,
            completion=CompletionState.GRASP_STABLE,
        )
    )

    terminal = stable
    if config.emulate_release:
        release_time = finish_time + 10_000_000
        release = executive.step(
            ExecutiveTick(
                now_ns=release_time,
                emg=EmgEvent(EmgIntent.OPEN, release_time, 0.95, "mock-open"),
                visual=None,
                planner=None,
                timestamps=ModalityTimestamps(release_time, release_time, release_time, policy_timestamp),
                versions=versions,
            )
        )
        # Controlled opening is deterministic and does not ask the VLA for a
        # contradictory chunk.  Safety remains the final writer authority.
        for index in range(8):
            now = release_time + index * action_period
            state = await hand.read_state()
            fraction = 1.0 - (index + 1) / 8.0
            opening_result = await command_path.execute(
                nominal_q_rad=state.q_rad * fraction,
                residual_q_rad=None,
                task_id=lease.task_id,
                task_version=lease.task_version + 1,
                source_chunk_id=None,
                safety_context=SafetyContext(),
                emg_requests_close=False,
                now_ns=now,
                state=state,
            )
            if opening_result.vetoed:
                raise RuntimeError(f"mock controlled-open command vetoed: {opening_result.reason}")
        done_time = release_time + 8 * action_period
        terminal = executive.step(
            ExecutiveTick(
                now_ns=done_time,
                emg=EmgEvent(EmgIntent.REST, done_time),
                visual=None,
                planner=None,
                timestamps=ModalityTimestamps(done_time, done_time, done_time, policy_timestamp),
                versions=versions,
                completion=CompletionState.RELEASED,
            )
        )
        trace.append({"release_directive": release.directive.value, "terminal": terminal.output.value})

    result = {
        "schema_version": "revo3-v1-mock-trace-v1",
        "verification_scope": "runnable plumbing smoke; no robot/task-success claim",
        "task": config.task.value,
        "instruction": lease.instruction,
        "trigger_grid_hz": str(aligned.grid_hz),
        "policy_hz": MAIN_ALIGNED_SCHEDULE.command_hz,
        "chunk_size": MAIN_ALIGNED_SCHEDULE.chunk_size,
        "refine_offsets": list(MAIN_ALIGNED_SCHEDULE.refine_offsets),
        "commands_written": len(hand.commands),
        "stable_output": stable.output.value,
        "terminal_output": terminal.output.value,
        "trace": trace,
    }
    if config.output_trace:
        path = Path(config.output_trace)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
    return result


def run(config: DemoConfig = DemoConfig()) -> Dict[str, object]:
    return asyncio.run(run_mock_demo(config))
