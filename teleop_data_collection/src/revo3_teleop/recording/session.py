"""Fail-closed coordination for one multi-rate collection episode.

``CollectionSession`` serializes recorder access from asynchronous source and
controller callbacks, but it never schedules or rate-limits a hardware control
loop.  Native samples and controller receipts therefore retain their actual
rates.  The caller invokes :meth:`record_anchor_if_due` from its 30 Hz policy
loop; the recorder remains the sole implementation of causal
latest-not-after alignment.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
import threading
import time
from typing import Callable, Mapping, Optional

from revo3_teleop.contracts import CausalAnchor, CommandReceipt, NativeSample
from revo3_teleop.recording.recorder import EpisodeRecorder, RecorderState


Clock = Callable[[], int]
StopCallback = Callable[[], None]
SafetyCallback = Callable[[str], None]


class SessionState(str, Enum):
    """Externally visible collection lifecycle."""

    ARMED = "armed"
    RECORDING = "recording"
    STOP = "stop"
    FAULT = "fault"


class CollectionSessionFault(RuntimeError):
    """The episode was stopped safely and quarantined when possible."""

    def __init__(self, reason: str, quarantine_path: Optional[Path] = None) -> None:
        message = reason
        if quarantine_path is not None:
            message = f"{reason} (quarantine={quarantine_path})"
        super().__init__(message)
        self.reason = reason
        self.quarantine_path = quarantine_path


def _callback(value: object, *, name: str) -> Callable[..., None]:
    if not callable(value):
        raise TypeError(f"{name} must be an explicit callable; collection is fail-closed")
    return value


def _stream_name(value: str) -> str:
    name = str(value).strip()
    if not name or not all(character.isalnum() or character in "-_" for character in name):
        raise ValueError("stream names may contain only letters, digits, '-' and '_'")
    return name


class CollectionSession:
    """Coordinate sources, command evidence, anchors, and terminal safety.

    Safety hooks are mandatory rather than defaulting to no-ops.  This makes a
    missing Revo hold or Tianji soft-stop implementation a construction-time
    error, before an episode can enter ``RECORDING``.

    ``required_source_timeouts_ns`` is measured against each sample's host
    receive timestamp.  Call :meth:`poll` from an independent watchdog (30 Hz
    or faster) so a completely silent source is detected even when no other
    callback arrives.
    """

    _SUPPORTED_COMPONENTS = frozenset({"revo_hand", "tianji_arm"})

    def __init__(
        self,
        recorder: EpisodeRecorder,
        *,
        required_source_timeouts_ns: Mapping[str, int],
        anchor_streams: tuple[str, ...],
        stop_targets: StopCallback,
        revo_hold: SafetyCallback,
        tianji_soft_stop: SafetyCallback,
        flush: StopCallback,
        clock: Clock = time.monotonic_ns,
    ) -> None:
        if not isinstance(recorder, EpisodeRecorder):
            raise TypeError("recorder must be an EpisodeRecorder")
        if recorder.state != RecorderState.NEW:
            raise ValueError("session requires a new EpisodeRecorder")
        if not required_source_timeouts_ns:
            raise ValueError("at least one required source is mandatory (fail-closed)")
        timeouts: dict[str, int] = {}
        for raw_name, raw_timeout in required_source_timeouts_ns.items():
            name = _stream_name(raw_name)
            timeout = int(raw_timeout)
            if timeout <= 0:
                raise ValueError(f"required source timeout for {name!r} must be positive")
            timeouts[name] = timeout
        streams = tuple(dict.fromkeys(_stream_name(item) for item in anchor_streams))
        if not streams:
            raise ValueError("anchor_streams must not be empty")
        missing_required = sorted(set(streams) - set(timeouts))
        if missing_required:
            raise ValueError(
                "every anchor stream must be required and have a timeout: "
                + ", ".join(missing_required)
            )
        self.recorder = recorder
        self.required_source_timeouts_ns = timeouts
        self.anchor_streams = streams
        self._stop_targets = _callback(stop_targets, name="stop_targets")
        self._revo_hold = _callback(revo_hold, name="revo_hold")
        self._tianji_soft_stop = _callback(tianji_soft_stop, name="tianji_soft_stop")
        self._flush = _callback(flush, name="flush")
        self._clock = _callback(clock, name="clock")
        self._state = SessionState.ARMED
        self._started_timestamp_ns: Optional[int] = None
        self._last_receive_ns: dict[str, int] = {}
        self._next_anchor_index = 0
        self._fault_reason: Optional[str] = None
        self._quarantine_path: Optional[Path] = None
        self._lock = threading.RLock()

    @property
    def state(self) -> SessionState:
        with self._lock:
            return self._state

    @property
    def fault_reason(self) -> Optional[str]:
        with self._lock:
            return self._fault_reason

    @property
    def quarantine_path(self) -> Optional[Path]:
        with self._lock:
            return self._quarantine_path

    @property
    def next_anchor_timestamp_ns(self) -> int:
        with self._lock:
            return self.recorder.anchor_timestamp_ns(self._next_anchor_index)

    def _now(self) -> int:
        value = int(self._clock())
        if value < 0:
            raise ValueError("clock returned a negative timestamp")
        return value

    def _require_recording(self) -> None:
        if self._state != SessionState.RECORDING:
            raise RuntimeError(f"session is not recording (state={self._state.value})")

    def _run_motion_stop(self, reason: str) -> list[str]:
        """Stop target production before commanding either hardware safe state."""

        failures: list[str] = []
        callbacks: tuple[tuple[str, Callable[..., None], tuple[object, ...]], ...] = (
            ("stop_targets", self._stop_targets, ()),
            ("revo_hold", self._revo_hold, (reason,)),
            ("tianji_soft_stop", self._tianji_soft_stop, (reason,)),
        )
        for name, callback, arguments in callbacks:
            try:
                callback(*arguments)
            except Exception as exc:  # continue so the other actuator is still stopped
                failures.append(f"{name}:{type(exc).__name__}:{exc}")
        return failures

    def _abort_fault(
        self,
        reason: str,
        *,
        cause: Optional[BaseException] = None,
        motion_already_stopped: bool = False,
        prior_failures: tuple[str, ...] = (),
    ) -> None:
        """Apply all safe-state hooks, quarantine, and raise one stable error."""

        failures = list(prior_failures)
        if not motion_already_stopped:
            failures.extend(self._run_motion_stop(reason))
        complete_reason = reason
        if failures:
            complete_reason += "; terminal_failures=" + " | ".join(failures)
        quarantine: Optional[Path] = None
        try:
            if self.recorder.state == RecorderState.RECORDING:
                quarantine = self.recorder.abort(complete_reason)
        except Exception as exc:
            failures.append(f"recorder_abort:{type(exc).__name__}:{exc}")
            complete_reason = reason + "; terminal_failures=" + " | ".join(failures)
        with self._lock:
            self._state = SessionState.FAULT
            self._fault_reason = complete_reason
            self._quarantine_path = quarantine
        error = CollectionSessionFault(complete_reason, quarantine)
        if cause is None:
            raise error
        raise error from cause

    def start(self) -> Path:
        """Publish the in-progress episode and enter ``RECORDING``."""

        try:
            with self._lock:
                if self._state != SessionState.ARMED:
                    raise RuntimeError("start is valid only in ARMED")
                started = self._now()
                path = self.recorder.start()
                self._started_timestamp_ns = started
                self._state = SessionState.RECORDING
                return path
        except Exception as exc:
            with self._lock:
                self._state = SessionState.FAULT
            self._abort_fault("session_start_failed", cause=exc)

    def accept_sample(self, stream: str, sample: NativeSample) -> Path:
        """Persist one native-rate sample without resampling the source."""

        stream_name = _stream_name(stream)
        failure: Optional[tuple[str, BaseException]] = None
        path: Optional[Path] = None
        with self._lock:
            self._require_recording()
            try:
                path = self.recorder.append(stream_name, sample)
                receive_ns = int(sample.header.receive_timestamp_ns)
                previous = self._last_receive_ns.get(stream_name)
                if previous is not None and receive_ns < previous:
                    raise ValueError(f"{stream_name} receive timestamp cannot regress")
                self._last_receive_ns[stream_name] = receive_ns
                if stream_name in self.required_source_timeouts_ns and not sample.header.valid:
                    raise ValueError(f"required source {stream_name!r} produced an invalid sample")
            except Exception as exc:
                self._state = SessionState.FAULT
                failure = (f"source_write_or_validation_failed:{stream_name}", exc)
        if failure is not None:
            self._abort_fault(failure[0], cause=failure[1])
        assert path is not None
        return path

    def accept_command(self, receipt: CommandReceipt) -> None:
        """Persist an exact Revo/Tianji controller receipt.

        A rejected controller transaction is retained diagnostically and then
        faults the episode.  It can never be used as action supervision.
        """

        failure: Optional[tuple[str, BaseException]] = None
        with self._lock:
            self._require_recording()
            try:
                if receipt.component not in self._SUPPORTED_COMPONENTS:
                    raise ValueError(
                        "collection session accepts only revo_hand or tianji_arm receipts"
                    )
                self.recorder.record_command(receipt)
                if not receipt.accepted:
                    raise RuntimeError(
                        f"{receipt.component} rejected command {receipt.request_id}: "
                        f"{receipt.reason or 'unspecified'}"
                    )
            except Exception as exc:
                self._state = SessionState.FAULT
                failure = (f"controller_receipt_failed:{receipt.component}", exc)
        if failure is not None:
            self._abort_fault(failure[0], cause=failure[1])

    def _health_failure(self, now_ns: int) -> Optional[str]:
        assert self._started_timestamp_ns is not None
        if now_ns < self._started_timestamp_ns:
            return "workstation_monotonic_clock_regressed"
        for stream, timeout_ns in self.required_source_timeouts_ns.items():
            last_receive_ns = self._last_receive_ns.get(stream)
            reference_ns = self._started_timestamp_ns if last_receive_ns is None else last_receive_ns
            if reference_ns > now_ns:
                return f"required_source_timestamp_in_future:{stream}"
            if now_ns - reference_ns > timeout_ns:
                state = "never_received" if last_receive_ns is None else "stale"
                return f"required_source_timeout:{stream}:{state}"
        return None

    def poll(self, *, now_ns: Optional[int] = None) -> None:
        """Run the independent required-source watchdog once."""

        failure: Optional[str]
        with self._lock:
            self._require_recording()
            observed_ns = self._now() if now_ns is None else int(now_ns)
            if observed_ns < 0:
                raise ValueError("now_ns must be non-negative")
            failure = self._health_failure(observed_ns)
            if failure is not None:
                self._state = SessionState.FAULT
        if failure is not None:
            self._abort_fault(failure)

    def record_anchor_if_due(
        self,
        *,
        hand_command_request_id: str,
        now_ns: Optional[int] = None,
    ) -> Optional[CausalAnchor]:
        """Record one due 30 Hz anchor, otherwise return ``None``.

        This method does not sleep, send a command, or alter any source/control
        frequency.  It delegates sample selection exclusively to
        :meth:`EpisodeRecorder.record_anchor`, preserving its causal
        latest-not-after rule.
        """

        failure: Optional[tuple[str, BaseException]] = None
        anchor: Optional[CausalAnchor] = None
        with self._lock:
            self._require_recording()
            observed_ns = self._now() if now_ns is None else int(now_ns)
            if observed_ns < 0:
                raise ValueError("now_ns must be non-negative")
            health_failure = self._health_failure(observed_ns)
            if health_failure is not None:
                self._state = SessionState.FAULT
                failure = (health_failure, RuntimeError(health_failure))
            elif observed_ns < self.recorder.anchor_timestamp_ns(self._next_anchor_index):
                return None
            else:
                try:
                    anchor = self.recorder.record_anchor(
                        anchor_index=self._next_anchor_index,
                        streams=self.anchor_streams,
                        hand_command_request_id=hand_command_request_id,
                        max_age_ns=self.required_source_timeouts_ns,
                    )
                    self._next_anchor_index += 1
                except Exception as exc:
                    self._state = SessionState.FAULT
                    failure = ("anchor_record_failed", exc)
        if failure is not None:
            self._abort_fault(failure[0], cause=failure[1])
        return anchor

    def backend_fault(self, component: str, reason: str) -> None:
        """Trip the common terminal path for a hardware/backend watchdog."""

        name = str(component).strip()
        detail = str(reason).strip()
        if not name or not detail:
            raise ValueError("component and reason must be non-empty")
        with self._lock:
            self._require_recording()
            self._state = SessionState.FAULT
        self._abort_fault(f"backend_fault:{name}:{detail}")

    def stop(self) -> Path:
        """Stop targets, hold/soft-stop hardware, flush, then commit atomically."""

        with self._lock:
            self._require_recording()
            observed_ns = self._now()
            health_failure = self._health_failure(observed_ns)
            # Close the callback admission gate before stopping target producers.
            self._state = SessionState.STOP
        motion_failures = tuple(self._run_motion_stop("normal_stop"))
        if health_failure is not None:
            self._abort_fault(
                health_failure,
                motion_already_stopped=True,
                prior_failures=motion_failures,
            )
        if motion_failures:
            self._abort_fault(
                "normal_stop_safety_failed",
                motion_already_stopped=True,
                prior_failures=motion_failures,
            )
        try:
            self._flush()
            committed = self.recorder.commit()
        except Exception as exc:
            self._abort_fault(
                "flush_or_commit_failed",
                cause=exc,
                motion_already_stopped=True,
            )
        with self._lock:
            self._state = SessionState.STOP
        return committed


__all__ = [
    "CollectionSession",
    "CollectionSessionFault",
    "SessionState",
]
