from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import subprocess
import sys
import types
from typing import Any

import numpy as np
import pytest

from revo3_v1.policy import (
    REVO3_FULL_CENTER_PROFILE,
    PolicyObservation,
    TaskKey,
    TReXTransportTimeout,
    TReXServerIdentity,
    TReXWireProtocolError,
    ZmqReqTransport,
    ZmqTReXBackend,
)


class FakeTransport:
    def __init__(self, replies: list[Any]) -> None:
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []
        self.closed = False

    def request(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        self.requests.append(dict(payload))
        if not self.replies:
            raise AssertionError("unexpected transport request")
        reply = self.replies.pop(0)
        if callable(reply):
            reply = reply(dict(payload))
        if isinstance(reply, BaseException):
            raise reply
        return reply

    def close(self) -> None:
        self.closed = True


def _task(version: int = 1, *, lease: str = "lease-a") -> tuple[TaskKey, str]:
    instruction = "Grasp the centered bottle using a power grasp."
    return (
        TaskKey.from_instruction(
            task_id="bottle",
            task_version=version,
            instruction=instruction,
            lease_id=lease,
            version_fingerprint="versions-sha256",
        ),
        instruction,
    )


def _observation(version: int = 1, *, lease: str = "lease-a") -> PolicyObservation:
    task_key, instruction = _task(version, lease=lease)
    return PolicyObservation(
        timestamp_ns=100,
        state_timestamp_ns=99,
        rgb_timestamp_ns=98,
        tactile_timestamp_ns=97,
        q_rad=np.linspace(-0.2, 0.2, 21, dtype=np.float32),
        tactile_f6=np.arange(30, dtype=np.float32).reshape(5, 6),
        tactile_history_f6=np.concatenate(
            [
                np.zeros((15, 5, 6), dtype=np.float32),
                np.arange(30, dtype=np.float32).reshape(1, 5, 6),
            ],
            axis=0,
        ),
        instruction=instruction,
        task_key=task_key,
        images={"head": np.full((12, 16, 3), 127, dtype=np.uint8)},
        tactile_history_timestamps_ns=np.arange(82, 98, dtype=np.int64),
        tactile_history_sequences=np.arange(16, dtype=np.int64),
        lease_expires_at_ns=1_000,
    )


def _dual_view_observation() -> PolicyObservation:
    base = _observation()
    return PolicyObservation(
        **{
            **base.__dict__,
            "images": {
                "full": np.full((288, 384, 3), 80, dtype=np.uint8),
                "fixed_center": np.full((288, 384, 3), 160, dtype=np.uint8),
            },
            "tactile_deform": np.zeros((5, 240, 240), dtype=np.uint8),
            "tactile_deform_timestamp_ns": np.full(5, 97, dtype=np.int64),
            "tactile_profile": "profile_a_force6d_diff",
        }
    )


def _profile_b_observation() -> PolicyObservation:
    base = _dual_view_observation()
    return PolicyObservation(
        **{
            **base.__dict__,
            "tactile_f6": None,
            "tactile_history_f6": None,
            "tactile_history_timestamps_ns": None,
            "tactile_history_sequences": None,
            "tactile_profile": "profile_b_diff_only",
        }
    )


def _success(mode: str, value: float, chunk_id: Any = 4) -> dict[str, Any]:
    return {
        "status": "success",
        "mode": mode,
        "actions": np.full((16, 21), value, dtype=np.float32),
        "chunk_id": chunk_id,
        "latency_ms": 1.5,
    }


def _server_identity(
    tactile_profile: str = "profile_a_force6d_diff",
) -> TReXServerIdentity:
    return TReXServerIdentity(
        checkpoint_sha256="1" * 64,
        model_config_sha256="2" * 64,
        training_args_sha256="3" * 64,
        checkpoint_lineage_sha256="4" * 64,
        normalization_statistics_sha256="5" * 64,
        normalization_artifact_sha256="6" * 64,
        checkpoint_family_id="revo-checkpoint-v1",
        normalization_family_id="revo-normalization-v1",
        tactile_profile_manifest_sha256="0" * 64,
        capability_manifest_sha256="7" * 64,
        split_manifest_sha256="8" * 64,
        tactile_profile=tactile_profile,
        camera_profile=REVO3_FULL_CENTER_PROFILE,
        joint_order_hash="9" * 64,
        training_stage="sft",
    )


def _echo_success(
    observation,
    mode: str,
    value: float,
    chunk_id: Any = 4,
    *,
    server_identity: TReXServerIdentity | None = None,
):
    identity = server_identity or _server_identity(observation.tactile_profile)

    def build(payload):
        return {
            **_success(mode, value, chunk_id),
            "task_id": observation.task_key.task_id,
            "task_version": observation.task_key.task_version,
            "instruction_hash": observation.task_key.instruction_hash,
            "lease_id": observation.task_key.lease_id,
            "version_fingerprint": observation.task_key.version_fingerprint,
            "observation_timestamp_ns": observation.timestamp_ns,
            "produced_at_ns": observation.timestamp_ns + 2,
            "server_identity": identity.as_mapping(),
        }

    return build


def test_module_import_does_not_load_optional_zmq_or_pillow() -> None:
    code = (
        "import sys; import revo3_v1.policy.zmq_backend; "
        "assert 'zmq' not in sys.modules; assert 'PIL.Image' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_revo_backend_requires_a_pinned_server_identity() -> None:
    with pytest.raises(ValueError, match="pinned expected_server_identity"):
        ZmqTReXBackend(
            FakeTransport([]),
            image_profile=REVO3_FULL_CENTER_PROFILE,
            tactile_profile="profile_a_force6d_diff",
        )

    transport = FakeTransport([])
    auto_backend = ZmqTReXBackend(
        transport,
        image_profile="auto",
        tactile_profile="profile_a_force6d_diff",
    )
    with pytest.raises(TReXWireProtocolError, match="before send"):
        auto_backend.slow_and_fast(_dual_view_observation())
    assert transport.requests == []


def test_wrong_server_checkpoint_is_rejected_before_action_cache() -> None:
    observation = _dual_view_observation()
    expected = _server_identity()
    wrong = replace(expected, checkpoint_sha256="a" * 64)
    transport = FakeTransport(
        [
            _echo_success(
                observation,
                "slow_and_fast",
                0.25,
                server_identity=wrong,
            )
        ]
    )
    backend = ZmqTReXBackend(
        transport,
        image_profile=REVO3_FULL_CENTER_PROFILE,
        tactile_profile="profile_a_force6d_diff",
        expected_server_identity=expected,
        clock=iter([101, 103]).__next__,
    )

    with pytest.raises(TReXWireProtocolError, match="checkpoint_sha256"):
        backend.slow_and_fast(observation)
    assert backend.server_chunk_id is None


def test_missing_server_identity_is_rejected_before_action_cache() -> None:
    observation = _dual_view_observation()
    reply = _echo_success(observation, "slow_and_fast", 0.25)({})
    reply.pop("server_identity")
    backend = ZmqTReXBackend(
        FakeTransport([reply]),
        image_profile=REVO3_FULL_CENTER_PROFILE,
        tactile_profile="profile_a_force6d_diff",
        expected_server_identity=_server_identity(),
        clock=iter([101, 103]).__next__,
    )

    with pytest.raises(TReXWireProtocolError, match="missing or has an invalid"):
        backend.slow_and_fast(observation)
    assert backend.server_chunk_id is None


def test_live_identity_probe_and_matching_action_reply_are_accepted() -> None:
    observation = _dual_view_observation()
    expected = _server_identity()
    transport = FakeTransport(
        [
            {
                "status": "success",
                "mode": "identity",
                "server_identity": expected.as_mapping(),
            },
            _echo_success(observation, "slow_and_fast", 0.25),
        ]
    )
    backend = ZmqTReXBackend(
        transport,
        image_profile=REVO3_FULL_CENTER_PROFILE,
        tactile_profile="profile_a_force6d_diff",
        expected_server_identity=expected,
        clock=iter([101, 103]).__next__,
    )

    assert backend.probe_server_identity() == expected
    actions = backend.slow_and_fast(observation)
    assert actions.shape == (16, 21)
    assert backend.server_chunk_id == 4


def test_slow_and_fast_encodes_exact_revo_rep_payload() -> None:
    transport = FakeTransport([_success("slow_and_fast", 0.25, chunk_id=8)])
    backend = ZmqTReXBackend(transport)
    observation = _observation()

    actions = backend.slow_and_fast(observation)

    assert actions.shape == (16, 21)
    assert np.all(actions == np.float32(0.25))
    assert backend.server_chunk_id == 8
    assert len(transport.requests) == 1
    payload = transport.requests[0]
    assert set(payload) == {
        "mode",
        "image_head",
        "task_description",
        "state_fast",
        "tactile_f6",
    }
    assert payload["mode"] == "slow_and_fast"
    assert payload["image_head"].startswith(b"\x89PNG\r\n\x1a\n")
    assert payload["task_description"] == observation.instruction
    np.testing.assert_array_equal(payload["state_fast"], observation.q_rad)
    np.testing.assert_array_equal(payload["tactile_f6"], observation.tactile_f6)


def test_slow_uses_slow_mode_and_same_full_observation_contract() -> None:
    transport = FakeTransport([_success("slow", 0.1, chunk_id="slow-2")])
    backend = ZmqTReXBackend(transport)

    actions = backend.slow(_observation())

    assert actions.shape == (16, 21)
    assert transport.requests[0]["mode"] == "slow"
    assert transport.requests[0]["state_fast"].shape == (21,)
    assert transport.requests[0]["tactile_f6"].shape == (5, 6)
    assert backend.server_chunk_id == "slow-2"


def test_fast_sends_only_current_tactile_and_preserves_server_chunk() -> None:
    transport = FakeTransport(
        [
            _success("slow_and_fast", 0.1, chunk_id=11),
            _success("fast", 0.2, chunk_id=11),
        ]
    )
    backend = ZmqTReXBackend(transport)
    observation = _observation()
    cached = backend.slow_and_fast(observation)

    refined = backend.fast(observation, cached, chunk_offset=4)

    assert np.all(refined == np.float32(0.2))
    assert set(transport.requests[1]) == {
        "mode", "chunk_id", "camera_profile", "tactile_f6"
    }
    assert transport.requests[1]["mode"] == "fast"
    assert transport.requests[1]["chunk_id"] == 11
    assert transport.requests[1]["camera_profile"] == "official_single_view"
    np.testing.assert_array_equal(
        transport.requests[1]["tactile_f6"], observation.tactile_f6
    )

    # The adapter cache is replaced by every fast refinement, so the next
    # request must present the newest reply, not the original slow chunk.
    transport.replies.append(_success("fast", 0.3, chunk_id=11))
    newest = backend.fast(observation, refined, chunk_offset=8)
    assert np.all(newest == np.float32(0.3))


def test_fast_rejects_task_or_local_chunk_mismatch_before_network() -> None:
    transport = FakeTransport([_success("slow_and_fast", 0.1, chunk_id=3)])
    backend = ZmqTReXBackend(transport)
    observation = _observation()
    cached = backend.slow_and_fast(observation)

    with pytest.raises(TReXWireProtocolError, match="task/version"):
        backend.fast(_observation(version=2), cached, chunk_offset=4)
    assert len(transport.requests) == 1

    # A task mismatch invalidates the local identity.  Establish it again,
    # then demonstrate that an altered adapter cache is also rejected.
    transport.replies.append(_success("slow_and_fast", 0.1, chunk_id=4))
    cached = backend.slow_and_fast(observation)
    altered = cached.copy()
    altered[0, 0] += 0.01
    with pytest.raises(TReXWireProtocolError, match="cached_chunk"):
        backend.fast(observation, altered, chunk_offset=4)
    assert len(transport.requests) == 2


def test_fast_rejects_server_chunk_mismatch_and_clears_cache() -> None:
    transport = FakeTransport(
        [
            _success("slow_and_fast", 0.1, chunk_id=20),
            _success("fast", 0.2, chunk_id=21),
        ]
    )
    backend = ZmqTReXBackend(transport)
    observation = _observation()
    cached = backend.slow_and_fast(observation)

    with pytest.raises(TReXWireProtocolError, match="does not match"):
        backend.fast(observation, cached, chunk_offset=4)
    assert backend.server_chunk_id is None
    with pytest.raises(TReXWireProtocolError, match="no matching"):
        backend.fast(observation, cached, chunk_offset=8)
    assert len(transport.requests) == 2


@pytest.mark.parametrize(
    "reply, message",
    [
        ({"status": "error", "message": "bad cache"}, "server returned error"),
        (_success("fast", 0.0), "reply mode"),
        (
            {
                **_success("slow_and_fast", 0.0),
                "actions": np.zeros((15, 21), dtype=np.float32),
            },
            "shape",
        ),
        (
            {
                **_success("slow_and_fast", 0.0),
                "actions": np.full((16, 21), np.nan, dtype=np.float32),
            },
            "NaN",
        ),
        ({**_success("slow_and_fast", 0.0), "chunk_id": None}, "chunk_id"),
    ],
)
def test_malformed_or_error_reply_fails_closed(reply: dict[str, Any], message: str) -> None:
    transport = FakeTransport([reply])
    backend = ZmqTReXBackend(transport)
    with pytest.raises(TReXWireProtocolError, match=message):
        backend.slow_and_fast(_observation())
    assert backend.server_chunk_id is None


def test_transport_timeout_is_typed_and_invalidates_cache() -> None:
    transport = FakeTransport(
        [_success("slow_and_fast", 0.1, chunk_id=2), TimeoutError("late")]
    )
    backend = ZmqTReXBackend(transport)
    observation = _observation()
    cached = backend.slow_and_fast(observation)

    with pytest.raises(TReXTransportTimeout):
        backend.fast(observation, cached, chunk_offset=4)
    assert backend.server_chunk_id is None


def test_slow_requires_unambiguous_uint8_head_image() -> None:
    transport = FakeTransport([])
    backend = ZmqTReXBackend(transport, image_key="main")
    base = _observation()
    with pytest.raises(TReXWireProtocolError, match="silently discard extra images"):
        backend.slow_and_fast(
            PolicyObservation(
                **{
                    **base.__dict__,
                    "images": {
                        "left": np.zeros((4, 4, 3), dtype=np.uint8),
                        "right": np.zeros((4, 4, 3), dtype=np.uint8),
                    },
                }
            )
        )
    assert transport.requests == []


def test_revo_mainline_sends_both_derived_views_without_silent_drop() -> None:
    observation = _dual_view_observation()
    transport = FakeTransport([_echo_success(observation, "slow_and_fast", 0.25, chunk_id=8)])
    backend = ZmqTReXBackend(
        transport,
        image_profile=REVO3_FULL_CENTER_PROFILE,
        tactile_profile="profile_a_force6d_diff",
        expected_server_identity=_server_identity(),
        clock=iter([101, 103]).__next__,
    )

    backend.slow_and_fast(observation)

    payload = transport.requests[0]
    assert payload["camera_profile"] == REVO3_FULL_CENTER_PROFILE
    assert payload["capture_timestamp_ns"] == observation.rgb_timestamp_ns
    assert payload["image_view_names"] == ("full", "fixed_center")
    assert payload["image_head"].startswith(b"\x89PNG\r\n\x1a\n")
    assert payload["image_wrist_right"].startswith(b"\x89PNG\r\n\x1a\n")
    assert payload["task_id"] == observation.task_key.task_id
    assert payload["version_fingerprint"] == "versions-sha256"
    assert payload["tactile_f6_history"].shape == (16, 5, 6)
    assert payload["tactile_f6_history_timestamps_ns"].shape == (16,)
    assert payload["tactile_f6_history_sequences"].shape == (16,)
    assert payload["tactile_deform"].shape == (5, 240, 240)
    assert "tactile_deform_delayed" not in payload
    assert "tactile_deform_delayed_timestamps_ns" not in payload


def test_profile_b_sends_real_diff_without_fabricated_force6d() -> None:
    observation = _profile_b_observation()
    transport = FakeTransport(
        [_echo_success(observation, "slow_and_fast", 0.25, chunk_id=8)]
    )
    backend = ZmqTReXBackend(
        transport,
        image_profile=REVO3_FULL_CENTER_PROFILE,
        tactile_profile="profile_b_diff_only",
        expected_server_identity=_server_identity("profile_b_diff_only"),
        clock=iter([101, 103]).__next__,
    )

    backend.slow_and_fast(observation)

    payload = transport.requests[0]
    assert "tactile_f6" not in payload
    assert "tactile_f6_history" not in payload
    assert "tactile_f6_history_timestamps_ns" not in payload
    assert "tactile_f6_history_sequences" not in payload
    assert payload["tactile_deform"].shape == (5, 240, 240)
    assert "tactile_deform_delayed" not in payload


@pytest.mark.parametrize("profile_factory", [_dual_view_observation, _profile_b_observation])
def test_revo_current_diff_rejects_bad_shape_dtype_future_or_stale(profile_factory) -> None:
    base = profile_factory()
    with pytest.raises(ValueError, match="shape"):
        PolicyObservation(
            **{**base.__dict__, "tactile_deform": np.zeros((4, 240, 240), np.uint8)}
        )
    with pytest.raises(ValueError, match="uint8"):
        PolicyObservation(
            **{**base.__dict__, "tactile_deform": np.zeros((5, 240, 240), np.float32)}
        )
    with pytest.raises(ValueError, match="causal"):
        PolicyObservation(
            **{
                **base.__dict__,
                "tactile_deform_timestamp_ns": np.full(5, base.timestamp_ns + 1, np.int64),
            }
        )

    stale_now = 200_000_000
    stale = PolicyObservation(
        **{
            **base.__dict__,
            "timestamp_ns": stale_now,
            "lease_expires_at_ns": stale_now + 1_000,
        }
    )
    backend = ZmqTReXBackend(
        FakeTransport([]),
        image_profile=REVO3_FULL_CENTER_PROFILE,
        tactile_profile=stale.tactile_profile,
        expected_server_identity=_server_identity(stale.tactile_profile),
        clock=lambda: stale_now + 1,
    )
    with pytest.raises(TReXWireProtocolError, match="current DIFF is stale"):
        backend.slow_and_fast(stale)


def test_revo_fast_sends_each_newest_current_diff_and_ignores_legacy_delayed() -> None:
    slow = _dual_view_observation()
    fast = PolicyObservation(
        **{
            **slow.__dict__,
            "timestamp_ns": 110,
            "state_timestamp_ns": 109,
            "rgb_timestamp_ns": 108,
            "tactile_timestamp_ns": 107,
            "tactile_history_timestamps_ns": np.arange(92, 108, dtype=np.int64),
            "tactile_deform": np.full((5, 240, 240), 7, np.uint8),
            "tactile_deform_timestamp_ns": np.full(5, 109, np.int64),
            "tactile_deform_delayed": np.zeros((4, 5, 240, 240), np.uint8),
            "tactile_deform_delayed_timestamps_ns": np.full((4, 5), 90, np.int64),
        }
    )
    transport = FakeTransport(
        [
            _echo_success(slow, "slow_and_fast", 0.1, chunk_id=13),
            _echo_success(fast, "fast", 0.2, chunk_id=13),
        ]
    )
    backend = ZmqTReXBackend(
        transport,
        image_profile=REVO3_FULL_CENTER_PROFILE,
        tactile_profile="profile_a_force6d_diff",
        expected_server_identity=_server_identity(),
        clock=iter([101, 103, 111, 113]).__next__,
    )
    cached = backend.slow_and_fast(slow)
    backend.fast(fast, cached, chunk_offset=4)
    np.testing.assert_array_equal(
        transport.requests[1]["tactile_deform"], fast.tactile_deform
    )
    assert "tactile_deform_delayed" not in transport.requests[1]


def test_profile_c_runtime_is_explicitly_blocked() -> None:
    base = _observation()
    with pytest.raises(ValueError, match="Profile C is not launchable"):
        PolicyObservation(**{**base.__dict__, "tactile_profile": "profile_c_pressure_matrix"})


def test_revo_reply_with_stale_produced_timestamp_clears_cache() -> None:
    observation = _dual_view_observation()

    def stale(payload):
        return {
            **_echo_success(observation, "slow_and_fast", 0.1, chunk_id=3)(payload),
            "produced_at_ns": observation.timestamp_ns,
        }

    backend = ZmqTReXBackend(
        FakeTransport([stale]),
        image_profile=REVO3_FULL_CENTER_PROFILE,
        tactile_profile="profile_a_force6d_diff",
        expected_server_identity=_server_identity(),
        clock=iter([101, 2_000_000_000]).__next__,
        max_response_age_ns=1_000_000_000,
    )
    with pytest.raises(TReXWireProtocolError, match="stale"):
        backend.slow_and_fast(observation)
    assert backend.server_chunk_id is None


def test_revo_dual_view_profile_fails_closed_on_missing_or_wrong_sized_center() -> None:
    transport = FakeTransport([])
    backend = ZmqTReXBackend(
        transport,
        image_profile=REVO3_FULL_CENTER_PROFILE,
        tactile_profile="profile_a_force6d_diff",
        expected_server_identity=_server_identity(),
        clock=lambda: 101,
    )
    base = _dual_view_observation()
    with pytest.raises(TReXWireProtocolError, match="both full and fixed_center"):
        backend.slow_and_fast(
            PolicyObservation(**{**base.__dict__, "images": {"full": base.images["full"]}})
        )
    with pytest.raises(TReXWireProtocolError, match="384x288"):
        backend.slow_and_fast(
            PolicyObservation(
                **{
                    **base.__dict__,
                    "images": {
                        "full": base.images["full"],
                        "fixed_center": np.zeros((12, 16, 3), dtype=np.uint8),
                    },
                }
            )
        )
    assert transport.requests == []


def test_real_transport_rebuilds_req_socket_after_timeout(monkeypatch: Any) -> None:
    class FakeAgain(Exception):
        pass

    class FakeZmqError(Exception):
        pass

    class Socket:
        def __init__(self, *, fail_recv: bool) -> None:
            self.fail_recv = fail_recv
            self.closed = False

        def setsockopt(self, option: int, value: int) -> None:
            del option, value

        def connect(self, endpoint: str) -> None:
            assert endpoint == "tcp://unit-test:9999"

        def send(self, payload: bytes) -> None:
            assert pickle_load(payload)["mode"] == "fast"

        def recv(self) -> bytes:
            if self.fail_recv:
                raise FakeAgain("deadline")
            return b"unused"

        def close(self, linger: int) -> None:
            assert linger == 0
            self.closed = True

    class Context:
        def __init__(self) -> None:
            self.sockets: list[Socket] = []

        def socket(self, kind: int) -> Socket:
            assert kind == 1
            socket = Socket(fail_recv=not self.sockets)
            self.sockets.append(socket)
            return socket

    fake_zmq = types.ModuleType("zmq")
    fake_zmq.REQ = 1
    fake_zmq.LINGER = 2
    fake_zmq.SNDTIMEO = 3
    fake_zmq.RCVTIMEO = 4
    fake_zmq.Again = FakeAgain
    fake_zmq.ZMQError = FakeZmqError
    monkeypatch.setitem(sys.modules, "zmq", fake_zmq)
    context = Context()
    transport = ZmqReqTransport(
        "tcp://unit-test:9999", timeout_ms=25, context=context
    )

    with pytest.raises(TReXTransportTimeout, match="socket was reset"):
        transport.request({"mode": "fast"})

    assert len(context.sockets) == 2
    assert context.sockets[0].closed
    assert not context.sockets[1].closed
    transport.close()
    assert context.sockets[1].closed


def pickle_load(payload: bytes) -> Any:
    # Keep pickle out of the fake socket's implementation details at module
    # import time; the real transport itself is what defines this wire format.
    import pickle

    return pickle.loads(payload)
