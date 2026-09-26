from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from revo3_teleop import CommandReceipt, NativeSample, SampleHeader
from revo3_teleop.recording import (
    CollectionSession,
    CollectionSessionFault,
    EpisodeRecorder,
    RecorderState,
    SessionState,
)
from revo3_v1.revo.contracts import JOINT_ORDER_HASH


class FakeClock:
    def __init__(self, value: int) -> None:
        self.value = value

    def __call__(self) -> int:
        return self.value


class FakeHooks:
    def __init__(self) -> None:
        self.events: list[str] = []

    def stop_targets(self) -> None:
        self.events.append("stop_targets")

    def revo_hold(self, reason: str) -> None:
        self.events.append(f"revo_hold:{reason}")

    def tianji_soft_stop(self, reason: str) -> None:
        self.events.append(f"tianji_soft_stop:{reason}")

    def flush(self) -> None:
        self.events.append("flush")


def native_sample(
    stream: str,
    sequence: int,
    timestamp_ns: int,
    *,
    valid: bool = True,
) -> NativeSample:
    if stream == "camera":
        payload = {"rgb": np.full((2, 3, 3), sequence, dtype=np.uint8)}
    else:
        payload = {"value": np.asarray([sequence], dtype=np.float32)}
    return NativeSample(
        SampleHeader(
            source_id=f"fake_{stream}",
            sequence=sequence,
            capture_timestamp_ns=timestamp_ns,
            receive_timestamp_ns=timestamp_ns,
            clock_domain="fake_monotonic",
            valid=valid,
        ),
        payload,
    )


def hand_receipt(request_id: str, anchor_ns: int, sequence: int = 0) -> CommandReceipt:
    target = np.full(21, 0.1 + sequence * 0.01, dtype=np.float32)
    return CommandReceipt(
        request_id=request_id,
        component="revo_hand",
        accepted=True,
        requested_target=target,
        authorized_target=target,
        exact_sent_target=target,
        decision_timestamp_ns=anchor_ns + 20,
        write_timestamp_ns=anchor_ns + 21,
        controller_sequence=sequence,
        unit="rad",
        joint_order_hash=JOINT_ORDER_HASH,
    )


def make_session(
    tmp_path: Path,
    *,
    epoch_ns: int,
    clock: FakeClock,
    hooks: FakeHooks,
    episode_id: str = "session_fixture",
    timeout_ns: int = 1_000,
) -> CollectionSession:
    return CollectionSession(
        EpisodeRecorder(tmp_path, episode_id=episode_id, epoch_ns=epoch_ns),
        required_source_timeouts_ns={"camera": timeout_ns, "revo_state": timeout_ns},
        anchor_streams=("camera", "revo_state"),
        stop_targets=hooks.stop_targets,
        revo_hold=hooks.revo_hold,
        tianji_soft_stop=hooks.tianji_soft_stop,
        flush=hooks.flush,
        clock=clock,
    )


def test_happy_session_uses_recorder_causal_anchor_and_ordered_stop(tmp_path: Path) -> None:
    epoch = 1_000_000
    clock = FakeClock(epoch - 100)
    hooks = FakeHooks()
    session = make_session(tmp_path, epoch_ns=epoch, clock=clock, hooks=hooks)
    session.start()

    # Samples and controller callbacks remain native-rate calls.  An anchor
    # poll before the exact 30 Hz timestamp does not sleep or emit a row.
    session.accept_sample("camera", native_sample("camera", 0, epoch - 10))
    session.accept_sample("revo_state", native_sample("revo_state", 0, epoch - 5))
    receipt = hand_receipt("hand-0", epoch)
    session.accept_command(receipt)
    clock.value = epoch - 1
    assert session.record_anchor_if_due(
        hand_command_request_id=receipt.request_id
    ) is None

    # Future samples may already be buffered.  EpisodeRecorder, not this
    # coordinator, selects latest-not-after and must retain the older rows.
    session.accept_sample("camera", native_sample("camera", 1, epoch + 10))
    session.accept_sample("revo_state", native_sample("revo_state", 1, epoch + 15))
    clock.value = epoch + 25
    anchor = session.record_anchor_if_due(hand_command_request_id=receipt.request_id)
    assert anchor is not None
    assert anchor.streams["camera"].sequence == 0
    assert anchor.streams["revo_state"].sequence == 0
    assert session.next_anchor_timestamp_ns == epoch + 1_000_000_000 // 30

    committed = session.stop()
    assert session.state == SessionState.STOP
    assert session.recorder.state == RecorderState.COMMITTED
    assert committed.parent.name == "committed"
    assert hooks.events == [
        "stop_targets",
        "revo_hold:normal_stop",
        "tianji_soft_stop:normal_stop",
        "flush",
    ]


def test_invalid_required_sample_stops_hardware_and_quarantines(tmp_path: Path) -> None:
    epoch = 2_000_000
    clock = FakeClock(epoch)
    hooks = FakeHooks()
    session = make_session(tmp_path, epoch_ns=epoch, clock=clock, hooks=hooks)
    session.start()

    with pytest.raises(CollectionSessionFault, match="source_write_or_validation_failed:camera"):
        session.accept_sample(
            "camera", native_sample("camera", 0, epoch, valid=False)
        )

    assert session.state == SessionState.FAULT
    assert session.recorder.state == RecorderState.ABORTED
    assert session.quarantine_path is not None
    assert session.quarantine_path.parent.name == "quarantine"
    assert hooks.events[:3] == [
        "stop_targets",
        "revo_hold:source_write_or_validation_failed:camera",
        "tianji_soft_stop:source_write_or_validation_failed:camera",
    ]


def test_watchdog_times_out_silent_required_source_fail_closed(tmp_path: Path) -> None:
    epoch = 3_000_000
    clock = FakeClock(epoch)
    hooks = FakeHooks()
    session = make_session(
        tmp_path,
        epoch_ns=epoch,
        clock=clock,
        hooks=hooks,
        timeout_ns=100,
    )
    session.start()
    session.accept_sample("camera", native_sample("camera", 0, epoch))
    session.accept_sample("revo_state", native_sample("revo_state", 0, epoch))
    clock.value = epoch + 101

    with pytest.raises(CollectionSessionFault, match="required_source_timeout:camera:stale"):
        session.poll()

    assert session.state == SessionState.FAULT
    manifest = json.loads(
        (session.quarantine_path / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["lifecycle"] == "aborted"
    assert manifest["abort_reason"] == "required_source_timeout:camera:stale"


def test_recorder_writer_exception_trips_same_terminal_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    epoch = 4_000_000
    clock = FakeClock(epoch)
    hooks = FakeHooks()
    session = make_session(tmp_path, epoch_ns=epoch, clock=clock, hooks=hooks)
    session.start()

    def raise_disk_error(*args: object, **kwargs: object) -> Path:
        raise OSError("disk unavailable")

    monkeypatch.setattr(session.recorder, "append", raise_disk_error)
    with pytest.raises(CollectionSessionFault) as raised:
        session.accept_sample("camera", native_sample("camera", 0, epoch))

    assert isinstance(raised.value.__cause__, OSError)
    assert session.recorder.state == RecorderState.ABORTED
    assert hooks.events[0] == "stop_targets"
    assert hooks.events[1].startswith("revo_hold:")
    assert hooks.events[2].startswith("tianji_soft_stop:")


def test_backend_fault_and_rejected_receipt_are_terminal(tmp_path: Path) -> None:
    epoch = 5_000_000
    clock = FakeClock(epoch)
    hooks = FakeHooks()
    session = make_session(
        tmp_path,
        epoch_ns=epoch,
        clock=clock,
        hooks=hooks,
        episode_id="backend_fault",
    )
    session.start()
    with pytest.raises(CollectionSessionFault, match="backend_fault:tianji_arm:feedback_stale"):
        session.backend_fault("tianji_arm", "feedback_stale")
    assert session.state == SessionState.FAULT

    hooks_2 = FakeHooks()
    session_2 = make_session(
        tmp_path,
        epoch_ns=epoch,
        clock=clock,
        hooks=hooks_2,
        episode_id="rejected_receipt",
    )
    session_2.start()
    target = np.zeros(7, dtype=np.float32)
    rejected = CommandReceipt(
        request_id="arm-rejected",
        component="tianji_arm",
        accepted=False,
        requested_target=target,
        decision_timestamp_ns=epoch,
        reason="not_armed",
        unit="rad",
        joint_order_hash="fixture",
    )
    with pytest.raises(CollectionSessionFault, match="controller_receipt_failed:tianji_arm"):
        session_2.accept_command(rejected)
    assert session_2.recorder.state == RecorderState.ABORTED


def test_flush_failure_never_publishes_episode(tmp_path: Path) -> None:
    epoch = 6_000_000
    clock = FakeClock(epoch)
    hooks = FakeHooks()

    def fail_flush() -> None:
        hooks.events.append("flush")
        raise OSError("flush failed")

    session = CollectionSession(
        EpisodeRecorder(tmp_path, episode_id="flush_failure", epoch_ns=epoch),
        required_source_timeouts_ns={"camera": 1_000},
        anchor_streams=("camera",),
        stop_targets=hooks.stop_targets,
        revo_hold=hooks.revo_hold,
        tianji_soft_stop=hooks.tianji_soft_stop,
        flush=fail_flush,
        clock=clock,
    )
    session.start()
    session.accept_sample("camera", native_sample("camera", 0, epoch))

    with pytest.raises(CollectionSessionFault, match="flush_or_commit_failed"):
        session.stop()

    assert session.state == SessionState.FAULT
    assert session.recorder.state == RecorderState.ABORTED
    assert not (tmp_path / "committed" / "flush_failure").exists()
    # Stop hooks run once and before the failed flush; they are not retriggered.
    assert hooks.events == [
        "stop_targets",
        "revo_hold:normal_stop",
        "tianji_soft_stop:normal_stop",
        "flush",
    ]


def test_one_safety_hook_failure_does_not_skip_the_other_actuator(tmp_path: Path) -> None:
    epoch = 7_000_000
    clock = FakeClock(epoch)
    events: list[str] = []

    def stop_targets() -> None:
        events.append("stop_targets")

    def broken_revo_hold(reason: str) -> None:
        events.append("revo_hold")
        raise OSError("Revo transport unavailable")

    def tianji_soft_stop(reason: str) -> None:
        events.append("tianji_soft_stop")

    session = CollectionSession(
        EpisodeRecorder(tmp_path, episode_id="safety_hook_failure", epoch_ns=epoch),
        required_source_timeouts_ns={"camera": 1_000},
        anchor_streams=("camera",),
        stop_targets=stop_targets,
        revo_hold=broken_revo_hold,
        tianji_soft_stop=tianji_soft_stop,
        flush=lambda: events.append("flush"),
        clock=clock,
    )
    session.start()
    session.accept_sample("camera", native_sample("camera", 0, epoch))

    with pytest.raises(CollectionSessionFault, match="normal_stop_safety_failed"):
        session.stop()

    assert events == ["stop_targets", "revo_hold", "tianji_soft_stop"]
    assert session.recorder.state == RecorderState.ABORTED
    manifest = json.loads(
        (session.quarantine_path / "manifest.json").read_text(encoding="utf-8")
    )
    assert "revo_hold:OSError" in manifest["abort_reason"]


def test_constructor_requires_explicit_safety_and_source_contract(tmp_path: Path) -> None:
    recorder = EpisodeRecorder(tmp_path, episode_id="fail_closed", epoch_ns=1)
    with pytest.raises(ValueError, match="at least one required source"):
        CollectionSession(
            recorder,
            required_source_timeouts_ns={},
            anchor_streams=("camera",),
            stop_targets=lambda: None,
            revo_hold=lambda reason: None,
            tianji_soft_stop=lambda reason: None,
            flush=lambda: None,
        )
