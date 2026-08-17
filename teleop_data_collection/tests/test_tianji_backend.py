from __future__ import annotations

import ctypes
from pathlib import Path
import sys

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from revo3_teleop.backends import (  # noqa: E402
    TianjiMarvinBackend,
    TianjiPhysicalInterventionRequired,
    TianjiSafetyLimits,
    tianji_joint_order_hash,
)


class FakeClock:
    def __init__(self, now_ns: int = 1_000_000_000) -> None:
        self.now_ns = now_ns

    def __call__(self) -> int:
        return self.now_ns

    def advance(self, nanoseconds: int) -> None:
        self.now_ns += nanoseconds


def feedback(
    serial: int,
    *,
    cur_state: int = 1,
    err_code: int = 0,
    a_pos_deg: float = 0.0,
    b_pos_deg: float = 0.0,
) -> dict[str, object]:
    def output(position: float, torque: float) -> dict[str, object]:
        return {
            "frame_serial": serial,
            "fb_joint_pos": [position] * 7,
            "fb_joint_vel": [90.0] * 7,
            "fb_joint_sToq": [torque] * 7,
        }

    state = {"cur_state": cur_state, "cmd_state": cur_state, "err_code": err_code}
    input_state = {"frame_miss_cnt": 0, "max_frame_miss_cnt": 0}
    return {
        "states": [dict(state), dict(state)],
        "outputs": [output(a_pos_deg, 1.25), output(b_pos_deg, 2.5)],
        "inputs": [dict(input_state), dict(input_state)],
    }


class FakeNativeClient:
    def __init__(
        self,
        frames: list[dict[str, object]],
        *,
        getbuf_returns_none: bool = False,
    ) -> None:
        if not frames:
            raise ValueError("fake needs at least one feedback frame")
        self.frames = list(frames)
        self.frame_index = 0
        self.getbuf_returns_none = getbuf_returns_none
        self.calls: list[tuple[str, object]] = []

    def OnLinkTo(self, *octets):
        self.calls.append(("OnLinkTo", tuple(int(item.value) for item in octets)))
        return 1

    def OnRelease(self):
        self.calls.append(("OnRelease", None))
        return 1

    def OnGetBuf(self, *args):
        self.calls.append(("OnGetBuf", args))
        frame = self.frames[min(self.frame_index, len(self.frames) - 1)]
        self.frame_index += 1
        if self.getbuf_returns_none:
            return None
        return frame

    def OnClearSet(self):
        self.calls.append(("OnClearSet", None))
        return 1

    def OnSetSend(self):
        self.calls.append(("OnSetSend", None))
        return 1

    def OnSetJointCmdPos_A(self, values):
        self.calls.append(("OnSetJointCmdPos_A", tuple(float(value) for value in values)))
        return 1

    def OnSetJointCmdPos_B(self, values):
        self.calls.append(("OnSetJointCmdPos_B", tuple(float(value) for value in values)))
        return 1

    def OnSetTargetState_A(self, value):
        self.calls.append(("OnSetTargetState_A", int(value.value)))
        return 1

    def OnSetTargetState_B(self, value):
        self.calls.append(("OnSetTargetState_B", int(value.value)))
        return 1

    def OnEMG_A(self):
        self.calls.append(("OnEMG_A", None))
        return 1

    def OnEMG_B(self):
        self.calls.append(("OnEMG_B", None))
        return 1


def limits(
    *,
    max_delta_rad: float = 2.0,
    max_feedback_age_ns: int = 250_000_000,
    max_target_age_ns: int = 100_000_000,
    require_wrist_pose: bool = True,
) -> TianjiSafetyLimits:
    return TianjiSafetyLimits(
        q_min_rad=np.full(7, -np.pi),
        q_max_rad=np.full(7, np.pi),
        max_delta_rad=max_delta_rad,
        max_feedback_age_ns=max_feedback_age_ns,
        max_target_age_ns=max_target_age_ns,
        require_wrist_pose=require_wrist_pose,
    )


def armed_backend(
    client: FakeNativeClient,
    clock: FakeClock,
    *,
    side: str = "A",
    safety_limits: TianjiSafetyLimits | None = None,
    capability_probe_confirmed: bool = True,
) -> TianjiMarvinBackend:
    backend = TianjiMarvinBackend(
        client,
        side=side,
        safety_limits=safety_limits or limits(),
        allow_hardware_write=True,
        capability_probe_confirmed=capability_probe_confirmed,
        arm_token="bench-confirmed",
        clock=clock,
        sleeper=lambda _: None,
    )
    backend.connect("192.168.1.190")
    return backend


def submit(
    backend: TianjiMarvinBackend,
    clock: FakeClock,
    *,
    request_id: str = "arm-0",
    target: np.ndarray | None = None,
    token: str = "bench-confirmed",
    wrist_pose_valid: bool = True,
):
    return backend.submit_target(
        request_id=request_id,
        q_target_rad=np.zeros(7, np.float32) if target is None else target,
        target_timestamp_ns=clock.now_ns,
        arm_token=token,
        wrist_pose_valid=wrist_pose_valid,
        decision_timestamp_ns=clock.now_ns,
    )


def test_ongetbuf_side_b_converts_degrees_and_degrees_per_second_to_si() -> None:
    clock = FakeClock()
    client = FakeNativeClient([feedback(8, a_pos_deg=90.0, b_pos_deg=180.0)])
    backend = TianjiMarvinBackend(client, side="B", clock=clock)
    backend.connect()

    state = backend.read_state()

    assert client.calls[0] == ("OnLinkTo", (192, 168, 1, 190))
    np.testing.assert_allclose(state.q_rad, np.pi)
    np.testing.assert_allclose(state.dq_rad_s, np.pi / 2)
    np.testing.assert_allclose(state.tau_nm, 2.5)
    assert state.frame_serial == 8
    assert state.side.value == "B"


def test_verified_joint_order_is_configurable_and_hashed_at_backend_boundary() -> None:
    client = FakeNativeClient([feedback(1)])
    order = tuple(f"verified_joint_{index}" for index in range(7))
    backend = TianjiMarvinBackend(client, side="A", joint_order=order)

    assert backend.joint_order == order
    assert backend.joint_order_hash == tianji_joint_order_hash(order)
    with pytest.raises(ValueError, match="unique"):
        TianjiMarvinBackend(client, side="A", joint_order=("same",) * 7)


def test_position_receipt_records_only_exact_native_transaction_target() -> None:
    clock = FakeClock()
    client = FakeNativeClient([feedback(10), feedback(11)])
    backend = armed_backend(client, clock)
    backend.read_state()  # establish a baseline; the write read proves advancement
    target = np.full(7, np.pi / 2, np.float32)

    receipt = submit(backend, clock, target=target)

    assert receipt.accepted
    np.testing.assert_allclose(receipt.exact_sent_target, target)
    transaction = [
        call for call in client.calls if call[0] in {
            "OnClearSet", "OnSetJointCmdPos_A", "OnSetSend"
        }
    ]
    assert [name for name, _ in transaction] == [
        "OnClearSet",
        "OnSetJointCmdPos_A",
        "OnSetSend",
    ]
    np.testing.assert_allclose(transaction[1][1], 90.0, atol=1e-6)


def test_write_is_blocked_without_matching_token_or_confirmed_capability_probe() -> None:
    clock = FakeClock()
    wrong_token_client = FakeNativeClient([feedback(1), feedback(2)])
    wrong_token = armed_backend(wrong_token_client, clock)
    receipt = submit(wrong_token, clock, token="wrong")
    assert not receipt.accepted
    assert "arm_token_mismatch" in receipt.reason
    assert not any(call[0] == "OnClearSet" for call in wrong_token_client.calls)

    unprobed_client = FakeNativeClient([feedback(1), feedback(2)])
    unprobed = armed_backend(
        unprobed_client,
        clock,
        capability_probe_confirmed=False,
    )
    receipt = submit(unprobed, clock)
    assert not receipt.accepted
    assert "capability_probe_unconfirmed" in receipt.reason


def test_missing_verified_wrist_pose_blocks_arm_before_feedback_or_motion() -> None:
    clock = FakeClock()
    client = FakeNativeClient([feedback(1), feedback(2)])
    backend = armed_backend(client, clock)

    receipt = submit(backend, clock, wrist_pose_valid=False)

    assert not receipt.accepted
    assert "verified_wrist_pose_required" in receipt.reason
    assert [name for name, _ in client.calls] == ["OnLinkTo"]


def test_serial_must_advance_and_frozen_serial_eventually_becomes_stale() -> None:
    clock = FakeClock()
    never_advances_client = FakeNativeClient([feedback(4), feedback(4)])
    never_advances = armed_backend(never_advances_client, clock)
    never_advances.read_state()
    receipt = submit(never_advances, clock)
    assert not receipt.accepted
    assert "feedback_serial_not_proven_advancing" in receipt.reason

    stale_client = FakeNativeClient([feedback(4), feedback(5), feedback(5)])
    stale = armed_backend(
        stale_client,
        clock,
        safety_limits=limits(max_feedback_age_ns=100_000_000),
    )
    stale.read_state()
    clock.advance(10_000_000)
    stale.read_state()  # serial 5 advanced at this instant
    clock.advance(101_000_000)
    receipt = submit(stale, clock)
    assert not receipt.accepted
    assert "feedback_stale" in receipt.reason
    assert any(call[0] == "OnEMG_A" for call in stale_client.calls)
    assert not any(call[0] == "OnClearSet" for call in stale_client.calls)


@pytest.mark.parametrize(
    ("cur_state", "err_code", "reason"),
    [
        (100, 0, "arm_state_error_100"),
        (1, 23, "feedback_error:23"),
    ],
)
def test_error_state_or_nonzero_error_code_blocks_and_soft_stops(
    cur_state: int,
    err_code: int,
    reason: str,
) -> None:
    clock = FakeClock()
    client = FakeNativeClient(
        [feedback(1), feedback(2, cur_state=cur_state, err_code=err_code)]
    )
    backend = armed_backend(client, clock)
    backend.read_state()

    receipt = submit(backend, clock)

    assert not receipt.accepted
    assert reason in receipt.reason
    assert any(call[0] == "OnEMG_A" for call in client.calls)
    assert not any(call[0] == "OnClearSet" for call in client.calls)


def test_target_age_joint_limits_delta_shape_and_finite_are_fail_closed() -> None:
    clock = FakeClock()
    client = FakeNativeClient([feedback(1), feedback(2), feedback(3)])
    backend = armed_backend(
        client,
        clock,
        safety_limits=limits(max_delta_rad=0.05, max_target_age_ns=10_000_000),
    )
    with pytest.raises(ValueError, match="shape"):
        backend.submit_target(
            request_id="shape",
            q_target_rad=np.zeros(6),
            target_timestamp_ns=clock.now_ns,
            arm_token="bench-confirmed",
            wrist_pose_valid=True,
        )
    with pytest.raises(ValueError, match="NaN or infinity"):
        backend.submit_target(
            request_id="nan",
            q_target_rad=np.full(7, np.nan),
            target_timestamp_ns=clock.now_ns,
            arm_token="bench-confirmed",
            wrist_pose_valid=True,
        )

    old_target_ns = clock.now_ns
    clock.advance(10_000_001)
    stale = backend.submit_target(
        request_id="stale-target",
        q_target_rad=np.zeros(7),
        target_timestamp_ns=old_target_ns,
        arm_token="bench-confirmed",
        wrist_pose_valid=True,
        decision_timestamp_ns=clock.now_ns,
    )
    assert not stale.accepted
    assert "target_stale" in stale.reason

    outside = submit(backend, clock, target=np.full(7, 4.0, np.float32))
    assert not outside.accepted
    assert "joint_limit_violation" in outside.reason

    backend.read_state()
    clock.advance(1)
    too_large_step = submit(backend, clock, target=np.full(7, 0.1, np.float32))
    assert not too_large_step.accepted
    assert "feedback_delta_limit_violation" in too_large_step.reason


def test_real_pointer_ongetbuf_is_only_enabled_by_injected_buffer_and_decoder() -> None:
    class DummyBuffer(ctypes.Structure):
        _fields_ = [("marker", ctypes.c_int)]

    clock = FakeClock()
    payload = feedback(12)
    client = FakeNativeClient([payload], getbuf_returns_none=True)
    backend = TianjiMarvinBackend(
        client,
        side="A",
        feedback_buffer_factory=DummyBuffer,
        feedback_decoder=lambda _: payload,
        clock=clock,
    )
    backend.connect()

    state = backend.read_state()

    assert state.frame_serial == 12
    getbuf_args = next(args for name, args in client.calls if name == "OnGetBuf")
    assert len(getbuf_args) == 1


def test_close_never_releases_tcp_when_state_zero_cannot_be_confirmed() -> None:
    clock = FakeClock()
    client = FakeNativeClient([feedback(1, cur_state=1)])
    backend = armed_backend(client, clock)

    with pytest.raises(TianjiPhysicalInterventionRequired, match="physical intervention"):
        backend.close(timeout_ns=1, poll_interval_s=1.0)

    assert not any(name == "OnRelease" for name, _ in client.calls)
    assert backend.connected


def test_close_releases_only_after_state_zero_feedback() -> None:
    clock = FakeClock()
    client = FakeNativeClient([feedback(1, cur_state=0)])
    backend = armed_backend(client, clock)

    backend.close()

    assert [name for name, _ in client.calls][-2:] == ["OnGetBuf", "OnRelease"]
    assert not backend.connected


def test_historical_void_soft_stop_return_is_accepted() -> None:
    clock = FakeClock()
    client = FakeNativeClient([feedback(1)])

    def void_soft_stop():
        client.calls.append(("OnEMG_A", None))
        return None

    client.OnEMG_A = void_soft_stop
    backend = TianjiMarvinBackend(client, side="A", clock=clock)
    backend.connect()

    backend.soft_stop()

    assert client.calls[-1] == ("OnEMG_A", None)
