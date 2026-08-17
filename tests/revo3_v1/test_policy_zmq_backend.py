from __future__ import annotations

from collections.abc import Mapping
import subprocess
import sys
import types
from typing import Any

import numpy as np
import pytest

from revo3_v1.policy import (
    PolicyObservation,
    TaskKey,
    TReXTransportTimeout,
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
        tactile_history_f6=np.zeros((16, 5, 6), dtype=np.float32),
        instruction=instruction,
        task_key=task_key,
        images={"head": np.full((12, 16, 3), 127, dtype=np.uint8)},
    )


def _success(mode: str, value: float, chunk_id: Any = 4) -> dict[str, Any]:
    return {
        "status": "success",
        "mode": mode,
        "actions": np.full((16, 21), value, dtype=np.float32),
        "chunk_id": chunk_id,
        "latency_ms": 1.5,
    }


def test_module_import_does_not_load_optional_zmq_or_pillow() -> None:
    code = (
        "import sys; import revo3_v1.policy.zmq_backend; "
        "assert 'zmq' not in sys.modules; assert 'PIL.Image' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


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
    assert set(transport.requests[1]) == {"mode", "chunk_id", "tactile_f6"}
    assert transport.requests[1]["mode"] == "fast"
    assert transport.requests[1]["chunk_id"] == 11
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
    with pytest.raises(TReXWireProtocolError, match="multiple cameras"):
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

