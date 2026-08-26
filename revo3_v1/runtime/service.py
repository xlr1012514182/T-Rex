"""Executable 30 Hz decision / 100 Hz single-writer runtime boundary."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
import inspect
import time
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from .orchestrator import (
    ClarificationToken,
    OnlineV1Coordinator,
    Runtime30HzResult,
    RuntimeSynchronizedInput,
)


@dataclass(frozen=True)
class EMGPacket:
    samples: np.ndarray
    sample_timestamps_ns: Sequence[int] | np.ndarray
    signal_quality: float = 1.0


@runtime_checkable
class RuntimeIO(Protocol):
    """Hardware acquisition boundary injected by the deployment application.

    Implementations must return already rectified/aligned Profile-A inputs.
    They do not write motors; the coordinator's ``RevoServoExecutor`` remains
    the only writer.  ``servo_input`` and ``close`` are deadline-controlled by
    the service and MUST be cancellation-safe: they must propagate
    ``asyncio.CancelledError`` promptly, finish their own resource cleanup
    before unwinding, and never leave an unowned background read/task.  A
    blocking vendor SDK therefore belongs behind an adapter with its own
    bounded, owned worker; it must not run directly on this event loop.
    """

    async def poll_emg_packet(self, *, now_ns: int) -> EMGPacket | None:
        ...

    async def control_input(self, *, now_ns: int) -> RuntimeSynchronizedInput:
        ...

    async def servo_input(
        self,
        *,
        now_ns: int,
        latest_control: RuntimeSynchronizedInput,
    ) -> RuntimeSynchronizedInput:
        ...

    async def close(self) -> None:
        ...


@dataclass(frozen=True)
class RuntimeShutdown:
    planner_clean: bool
    policy_clean: bool
    io_clean: bool
    io_timed_out: bool
    io_intervention_required: bool
    backend_clean: bool
    backend_timed_out: bool
    backend_intervention_required: bool
    stop_succeeded: bool

    @property
    def clean(self) -> bool:
        return (
            self.planner_clean
            and self.policy_clean
            and self.io_clean
            and not self.io_timed_out
            and not self.io_intervention_required
            and self.backend_clean
            and not self.backend_timed_out
            and not self.backend_intervention_required
            and self.stop_succeeded
        )


@dataclass(frozen=True)
class RuntimeEvidence:
    """Bounded, claim-neutral evidence emitted by the executable smoke path."""

    executive_outputs: tuple[str, ...]
    executive_reasons: tuple[str, ...]
    motion_directives: tuple[str, ...]
    policy_states: tuple[str, ...]
    servo_write_count: int


class DoubleRateRuntimeService:
    """Drive the same coordinator in simulation and production.

    The service does no interpolation or safety work itself.  It schedules the
    30 Hz lifecycle/policy boundary and the independent 100 Hz servo boundary;
    all authorization remains in the coordinator and its sole motor writer.
    """

    control_period_ns = 1_000_000_000 // 30
    servo_period_ns = 1_000_000_000 // 100
    default_servo_io_timeout_ns = 50_000_000
    default_io_close_timeout_ns = 1_000_000_000

    def __init__(
        self,
        coordinator: OnlineV1Coordinator,
        io: RuntimeIO,
        *,
        clock_ns=time.perf_counter_ns,
        sleeper=asyncio.sleep,
        allow_direct_emg_fixtures: bool = False,
        control_watchdog_ns: int | None = None,
        servo_io_timeout_ns: int = default_servo_io_timeout_ns,
        io_close_timeout_ns: int = default_io_close_timeout_ns,
    ) -> None:
        if not isinstance(io, RuntimeIO):
            raise TypeError("io must implement the RuntimeIO acquisition boundary")
        self.coordinator = coordinator
        self.io = io
        self._clock_ns = clock_ns
        self._sleeper = sleeper
        self.allow_direct_emg_fixtures = bool(allow_direct_emg_fixtures)
        configured_watchdog = (
            min(
                3 * self.control_period_ns,
                coordinator.executive.config.camera_stale_abort_ns,
                coordinator.executive.config.policy_stale_abort_ns,
            )
            if control_watchdog_ns is None
            else int(control_watchdog_ns)
        )
        if configured_watchdog <= 0:
            raise ValueError("control_watchdog_ns must be positive")
        if int(servo_io_timeout_ns) <= 0:
            raise ValueError("servo_io_timeout_ns must be positive")
        if int(servo_io_timeout_ns) > configured_watchdog:
            raise ValueError(
                "servo_io_timeout_ns must not exceed control_watchdog_ns"
            )
        if int(io_close_timeout_ns) <= 0:
            raise ValueError("io_close_timeout_ns must be positive")
        if not inspect.iscoroutinefunction(self.io.servo_input):
            raise TypeError("RuntimeIO.servo_input must be an async cancellation-safe method")
        if not inspect.iscoroutinefunction(self.io.close):
            raise TypeError("RuntimeIO.close must be an async cancellation-safe method")
        self.control_watchdog_ns = configured_watchdog
        self.servo_io_timeout_ns = int(servo_io_timeout_ns)
        self.io_close_timeout_ns = int(io_close_timeout_ns)
        self._latest_control: RuntimeSynchronizedInput | None = None
        self._latest_result: Runtime30HzResult | None = None
        self._last_control_success_ns: int | None = None
        self._closed = False
        self._shutdown: RuntimeShutdown | None = None
        self._control_ready = asyncio.Event()
        self._executive_outputs: list[str] = []
        self._executive_reasons: list[str] = []
        self._motion_directives: list[str] = []
        self._policy_states: list[str] = []
        self._servo_write_count = 0
        self._coordinator_lock = asyncio.Lock()
        self._next_servo_deadline_ns: int | None = None
        self._completed_servo_deadline_ns: int | None = None
        self._io_faulted = False
        self._io_timed_out = False
        self._io_intervention_required = False
        self._terminal_ack_pending: tuple[str, bool] | None = None
        self._run_stop: asyncio.Event | None = None

    @property
    def evidence(self) -> RuntimeEvidence:
        return RuntimeEvidence(
            tuple(self._executive_outputs),
            tuple(self._executive_reasons),
            tuple(self._motion_directives),
            tuple(self._policy_states),
            self._servo_write_count,
        )

    @staticmethod
    def _append_once(values: list[str], value: str) -> None:
        if value not in values:
            values.append(value)

    @property
    def clarification_token(self) -> ClarificationToken | None:
        return self.coordinator.clarification_token

    def submit_clarification(self, answer: str, *, token: ClarificationToken) -> None:
        self.coordinator.submit_clarification(
            answer,
            event_id=token.event_id,
            planner_generation=token.planner_generation,
            task_version=token.task_version,
            received_at_ns=int(self._clock_ns()),
        )

    @property
    def terminal_ack_pending(self) -> bool:
        """Whether a supervisor acknowledgement awaits the next 30 Hz commit."""

        return self._terminal_ack_pending is not None

    def request_shutdown(self) -> None:
        """Ask an active continuous run to follow its normal bounded shutdown."""

        stop = self._run_stop
        if stop is not None:
            stop.set()

    async def acknowledge_terminal(
        self,
        *,
        safe_state_confirmed: bool,
        operator_reset_confirmed: bool = False,
    ) -> None:
        """Queue an explicit COMPLETE/ABORT reset at the next control commit.

        COMPLETE is never swallowed automatically: a supervisor must first
        confirm its task-specific safe postcondition.  ABORT additionally
        requires an operator/bench reset and a previously confirmed SoftStop;
        it never opens the hand.  Deferring the actual reset to the next fresh
        control snapshot avoids a gap in which the 100 Hz writer could observe
        a reset coordinator without a corresponding IDLE decision.
        """

        if not safe_state_confirmed:
            raise ValueError("terminal acknowledgement requires safe_state_confirmed")
        async with self._coordinator_lock:
            if self._closed:
                raise RuntimeError("runtime service is closed")
            if self._terminal_ack_pending is not None:
                raise RuntimeError("terminal acknowledgement is already pending")
            if self._latest_result is None:
                raise RuntimeError("no terminal TaskExecutive result is observable")
            output = self._latest_result.executive.output.value
            pipeline = self.coordinator.servo.pipeline
            if output == "COMPLETE":
                if pipeline.hard_fault_latched:
                    raise RuntimeError("COMPLETE cannot reset a latched hardware fault")
            elif output == "ABORT":
                if not operator_reset_confirmed:
                    raise ValueError(
                        "ABORT reset requires explicit operator_reset_confirmed"
                    )
                if not pipeline.soft_stop_confirmed:
                    raise RuntimeError("ABORT reset requires a confirmed SoftStop")
            else:
                raise RuntimeError("terminal acknowledgement requires COMPLETE or ABORT")
            self._terminal_ack_pending = (output, bool(operator_reset_confirmed))

    def _apply_terminal_ack_locked(self) -> None:
        pending = self._terminal_ack_pending
        if pending is None:
            return
        output, operator_reset_confirmed = pending
        pipeline = self.coordinator.servo.pipeline
        if output == "ABORT":
            if not operator_reset_confirmed or not pipeline.soft_stop_confirmed:
                raise RuntimeError("ABORT acknowledgement lost its safety precondition")
        self.coordinator.reset_terminal(
            safe_state_confirmed=True,
            operator_reset_confirmed=operator_reset_confirmed,
        )
        self._terminal_ack_pending = None

    async def _maybe_ingest_emg(self, now_ns: int) -> None:
        packet = await self.io.poll_emg_packet(now_ns=now_ns)
        if packet is not None:
            self.coordinator.ingest_emg_packet(
                packet.samples,
                packet.sample_timestamps_ns,
                signal_quality=packet.signal_quality,
            )

    async def step_control(self, *, now_ns: int) -> Runtime30HzResult:
        await self._maybe_ingest_emg(now_ns)
        value = await self.io.control_input(now_ns=now_ns)
        if value.emg is not None and not self.allow_direct_emg_fixtures:
            raise RuntimeError(
                "RuntimeIO may not inject EMG events; use raw poll_emg_packet -> StreamingEMG bridge"
            )
        # Full-frame luma/focus/hash work is CPU-heavy but independent of the
        # sole writer.  Run it off the event-loop thread so the 30 Hz camera
        # gate cannot serially delay a 100 Hz motor tick.
        health = await asyncio.to_thread(
            self.coordinator.camera_health.evaluate,
            value.views,
            now_ns=value.now_ns,
        )
        await self._yield_to_imminent_servo_deadline()
        async with self._coordinator_lock:
            self._apply_terminal_ack_locked()
            result = self.coordinator.step_30hz(
                value, camera_health_result=health
            )
            self._latest_control = value
            self._latest_result = result
            self._last_control_success_ns = int(now_ns)
            self._append_once(self._executive_outputs, result.executive.output.value)
            self._append_once(self._executive_reasons, result.executive.reason)
            self._append_once(
                self._motion_directives, result.executive.directive.value
            )
            if result.policy is not None:
                self._append_once(self._policy_states, result.policy.state.value)
            self._control_ready.set()
        return result

    async def _yield_to_imminent_servo_deadline(self) -> None:
        """Give the sole writer priority near a shared event-loop deadline."""

        deadline = self._next_servo_deadline_ns
        if deadline is None:
            return
        now = int(self._clock_ns())
        if deadline - now > 4_000_000:
            return
        # The loops remain independent: this delays only the 30 Hz commit,
        # never the writer.  Wait until the servo loop records completion of
        # the deadline that was imminent when this check began.  A bounded
        # run may finish its last permitted servo tick after this deadline was
        # published but before it was executed; the shared run-stop is then
        # authoritative and must release the control loop immediately.
        run_stop = self._run_stop
        while (
            not self._closed
            and not (run_stop is not None and run_stop.is_set())
            and (self._completed_servo_deadline_ns or -1) < deadline
        ):
            await self._sleeper(0)

    async def step_servo(self, *, now_ns: int):
        if self._latest_control is None:
            raise RuntimeError("servo started before the first control snapshot")
        latest_control = self._latest_control
        try:
            value = await asyncio.wait_for(
                self.io.servo_input(
                    now_ns=now_ns,
                    latest_control=latest_control,
                ),
                timeout=self.servo_io_timeout_ns / 1_000_000_000.0,
            )
        except asyncio.TimeoutError as exc:
            # wait_for cancels and joins the cancellation-safe RuntimeIO
            # coroutine before raising, so no read task is orphaned.
            self._io_faulted = True
            self._io_timed_out = True
            self._io_intervention_required = True
            raise RuntimeError("servo_input_timeout") from exc
        except asyncio.CancelledError:
            raise
        except BaseException:
            self._io_faulted = True
            self._io_intervention_required = True
            raise
        if value.emg is not None and not self.allow_direct_emg_fixtures:
            self._io_faulted = True
            self._io_intervention_required = True
            raise RuntimeError("servo RuntimeIO may not inject EMG events")
        async with self._coordinator_lock:
            last_control = self._last_control_success_ns
            if (
                last_control is None
                or int(now_ns) - last_control > self.control_watchdog_ns
            ):
                await self.coordinator.servo.pipeline.abort(
                    "control_heartbeat_timeout"
                )
                raise RuntimeError("control_heartbeat_timeout")
            chunk_id = None
            if (
                self._latest_result is not None
                and self._latest_result.policy is not None
                and self._latest_result.policy.chunk is not None
            ):
                chunk_id = self._latest_result.policy.chunk.chunk_id
            result = await self.coordinator.step_servo(value, source_chunk_id=chunk_id)
            safety = result.safety
            if safety is not None and safety.hard_fault_latched:
                terminal = self.coordinator.latch_servo_fault(safety.reason)
                if self._latest_result is not None:
                    self._latest_result = replace(
                        self._latest_result,
                        executive=terminal,
                    )
                self._append_once(self._executive_outputs, terminal.output.value)
                self._append_once(self._executive_reasons, terminal.reason)
                self._append_once(self._motion_directives, terminal.directive.value)
                if safety.soft_stop_requested and not safety.soft_stop_succeeded:
                    raise RuntimeError(
                        "servo_soft_stop_unconfirmed:" + safety.reason
                    )
                raise RuntimeError("servo_hard_fault_latched:" + safety.reason)
        if result.wrote_command:
            self._servo_write_count += 1
        return result

    async def _wait_until(self, deadline_ns: int, stop: asyncio.Event) -> None:
        """Yield cooperatively until a high-resolution runtime deadline.

        Windows' default positive asyncio timer quantum can exceed the frozen
        ±2 ms writer-jitter envelope.  Zero-delay yields retain cancellation
        and task fairness while the high-resolution performance counter owns
        the actual deadline.  Hardware deployments may inject an equivalent
        RT-aware sleeper without changing coordinator or writer semantics.
        """

        while not stop.is_set() and int(self._clock_ns()) < int(deadline_ns):
            await self._sleeper(0)

    async def _control_loop(self, stop: asyncio.Event) -> None:
        """Run acquisition/planning independently of the 100 Hz writer."""

        next_control_ns = int(self._clock_ns())
        while not stop.is_set():
            now_ns = int(self._clock_ns())
            if now_ns < next_control_ns:
                await self._wait_until(next_control_ns, stop)
                continue
            await self.step_control(now_ns=now_ns)
            # Skip missed acquisitions instead of producing non-causal bursts.
            while next_control_ns <= now_ns:
                next_control_ns += self.control_period_ns

    async def _servo_loop(
        self, stop: asyncio.Event, *, max_servo_ticks: int | None
    ) -> None:
        """Run the sole motor writer on its own fixed-rate schedule."""

        # There is no safe servo input before the first complete acquisition.
        # Starting the 100 Hz clock afterwards prevents initialization latency
        # from being misreported as a writer timing fault.
        await self._control_ready.wait()
        next_servo_ns = int(self._clock_ns())
        self._next_servo_deadline_ns = next_servo_ns
        servo_ticks = 0
        while not stop.is_set() and (
            max_servo_ticks is None or servo_ticks < max_servo_ticks
        ):
            now_ns = int(self._clock_ns())
            if now_ns < next_servo_ns:
                await self._wait_until(next_servo_ns, stop)
                continue
            executing_deadline = next_servo_ns
            await self.step_servo(now_ns=now_ns)
            self._completed_servo_deadline_ns = executing_deadline
            servo_ticks += 1
            # Account for both time spent inside acquisition/safety/write and
            # a late tick start before advancing the phase schedule.  Merely
            # moving past ``completed_ns`` is insufficient: when this tick
            # starts late but completes before the next nominal phase, that
            # phase could still start less than one safety command period
            # after ``now_ns``.  Drop every such phase instead of issuing a
            # catch-up write.  This preserves the production acceleration
            # envelope while making missed host/SDK deadlines non-bursty.
            completed_ns = int(self._clock_ns())
            minimum_next_start_ns = now_ns + max(
                self.servo_period_ns,
                int(
                    self.coordinator.servo.pipeline.safety.envelope.command_period_ns
                ),
            )
            while (
                next_servo_ns <= completed_ns
                or next_servo_ns < minimum_next_start_ns
            ):
                next_servo_ns += self.servo_period_ns
            self._next_servo_deadline_ns = next_servo_ns
        stop.set()

    async def run(self, *, max_servo_ticks: int | None = None) -> RuntimeShutdown:
        if max_servo_ticks is not None and max_servo_ticks <= 0:
            raise ValueError("max_servo_ticks must be positive")
        if self._closed:
            raise RuntimeError("runtime service is closed")
        stop = asyncio.Event()
        self._run_stop = stop
        control_task = asyncio.create_task(
            self._control_loop(stop), name="revo3-control-30hz"
        )
        servo_task = asyncio.create_task(
            self._servo_loop(stop, max_servo_ticks=max_servo_ticks),
            name="revo3-servo-100hz",
        )
        tasks = (control_task, servo_task)
        shutdown: RuntimeShutdown | None = None
        try:
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_EXCEPTION
            )
            # FIRST_EXCEPTION also returns when every task completed normally.
            # If one task failed, cancel its peer before propagating the
            # original exception through the common fail-closed path.
            failure = next(
                (
                    task.exception()
                    for task in done
                    if not task.cancelled() and task.exception() is not None
                ),
                None,
            )
            if failure is not None:
                stop.set()
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                raise failure
            stop.set()
            await asyncio.gather(*pending)
        except BaseException as exc:
            # IO/model/coordinator failures must never leave the last command
            # active.  Route the stop through the same sole-writer pipeline;
            # real hardware re-reads telemetry and invokes its verified
            # SoftStop callback.  Preserve the original exception afterwards.
            # Stop and join both schedulers first so a concurrent regular
            # writer cannot race the final abort during external cancellation.
            stop.set()
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            try:
                await self.coordinator.servo.pipeline.abort(
                    f"runtime_loop_exception:{type(exc).__name__}"
                )
            except BaseException:
                pass
            await self.close()
            raise
        finally:
            if not self._closed:
                shutdown = await self.close()
            self._run_stop = None
        if shutdown is None:
            shutdown = self._shutdown or RuntimeShutdown(
                planner_clean=True,
                policy_clean=True,
                io_clean=True,
                io_timed_out=False,
                io_intervention_required=False,
                backend_clean=True,
                backend_timed_out=False,
                backend_intervention_required=False,
                stop_succeeded=True,
            )
        return shutdown

    async def close(self) -> RuntimeShutdown:
        if self._closed:
            assert self._shutdown is not None
            return self._shutdown
        self._closed = True
        stop_succeeded = False
        try:
            stop = await self.coordinator.servo.pipeline.abort("runtime_shutdown")
            stop_succeeded = bool(
                stop.soft_stop_succeeded
                or self.coordinator.servo.pipeline.soft_stop_confirmed
            )
        except BaseException:
            stop_succeeded = False
        backend_clean = True
        try:
            close_backend = getattr(
                self.coordinator.servo.pipeline.backend, "close", None
            )
            if callable(close_backend):
                value = close_backend()
                if inspect.isawaitable(value):
                    value = await value
                backend_clean = value is not False
        except BaseException:
            backend_clean = False
        backend_timed_out = bool(
            getattr(self.coordinator.servo.pipeline.backend, "timed_out", False)
        )
        backend_intervention_required = bool(
            getattr(
                self.coordinator.servo.pipeline.backend,
                "intervention_required",
                False,
            )
        ) or not backend_clean or not stop_succeeded
        io_clean = not self._io_faulted
        try:
            await asyncio.wait_for(
                self.io.close(),
                timeout=self.io_close_timeout_ns / 1_000_000_000.0,
            )
        except asyncio.TimeoutError:
            io_clean = False
            self._io_timed_out = True
            self._io_intervention_required = True
        except BaseException:
            io_clean = False
            self._io_intervention_required = True
        planner_clean, policy_clean = self.coordinator.close()
        self._shutdown = RuntimeShutdown(
            planner_clean=planner_clean,
            policy_clean=policy_clean,
            io_clean=io_clean,
            io_timed_out=self._io_timed_out,
            io_intervention_required=self._io_intervention_required,
            backend_clean=backend_clean,
            backend_timed_out=backend_timed_out,
            backend_intervention_required=backend_intervention_required,
            stop_succeeded=stop_succeeded,
        )
        return self._shutdown


__all__ = [
    "DoubleRateRuntimeService",
    "EMGPacket",
    "RuntimeIO",
    "RuntimeEvidence",
    "RuntimeShutdown",
]
