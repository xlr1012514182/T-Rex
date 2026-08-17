"""Strict ZeroMQ client backend for the repository's ``scripts/test.py`` server.

The public backend implements :class:`~revo3_v1.policy.adapter.TReXBackend`
without importing either pyzmq or Pillow at module-import time.  Tests and
offline demos can inject a mapping-level request/reply transport; real use can
construct :class:`ZmqReqTransport`, which speaks the server's pickle-over-ZMQ
REP protocol.

Only RGB, language, Revo state, and F6 tactile observations cross this
boundary.  EMG remains outside the VLA policy by design.
"""

from __future__ import annotations

from collections.abc import Mapping
import io
import pickle
import threading
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .contracts import ACTION_CHUNK, ACTION_DIM, PolicyObservation, TaskKey


class TReXZmqError(RuntimeError):
    """Base class for transport and wire-protocol failures."""


class TReXTransportError(TReXZmqError):
    """The request/reply transport failed before a valid reply was received."""


class TReXTransportTimeout(TReXTransportError):
    """The request exceeded the configured send or receive deadline."""


class TReXWireProtocolError(TReXZmqError):
    """The server reply or local request state violated the REP contract."""


@runtime_checkable
class RequestReplyTransport(Protocol):
    """Mapping-level synchronous transport used by :class:`ZmqTReXBackend`."""

    def request(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        ...

    def close(self) -> None:
        ...


class ZmqReqTransport:
    """Pickle-over-pyzmq REQ transport compatible with ``scripts/test.py``.

    A REQ socket cannot safely continue after a timeout because its send/recv
    state is then ambiguous.  Therefore every timeout or ZMQ error closes the
    socket with ``LINGER=0`` and creates a fresh socket for the next request.
    The shared context is not terminated by :meth:`close`.
    """

    def __init__(
        self,
        endpoint: str = "tcp://127.0.0.1:5555",
        *,
        timeout_ms: int = 5_000,
        context: Any | None = None,
    ) -> None:
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise ValueError("endpoint must be a non-empty ZMQ endpoint.")
        if int(timeout_ms) <= 0:
            raise ValueError("timeout_ms must be positive.")

        # Lazy optional dependency: importing revo3_v1.policy must not require
        # pyzmq on data-preparation or unit-test hosts.
        try:
            import zmq  # type: ignore
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "Real T-Rex transport requires pyzmq; install package 'pyzmq'."
            ) from exc

        self._zmq = zmq
        self.endpoint = endpoint.strip()
        self.timeout_ms = int(timeout_ms)
        self._context = zmq.Context.instance() if context is None else context
        self._socket: Any | None = None
        self._closed = False
        self._lock = threading.Lock()
        self._rebuild_socket()

    def _rebuild_socket(self) -> None:
        self._close_socket()
        if self._closed:
            return
        socket = self._context.socket(self._zmq.REQ)
        socket.setsockopt(self._zmq.LINGER, 0)
        socket.setsockopt(self._zmq.SNDTIMEO, self.timeout_ms)
        socket.setsockopt(self._zmq.RCVTIMEO, self.timeout_ms)
        socket.connect(self.endpoint)
        self._socket = socket

    def _close_socket(self) -> None:
        if self._socket is not None:
            try:
                self._socket.close(linger=0)
            finally:
                self._socket = None

    def request(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        if self._closed:
            raise TReXTransportError("ZMQ transport is closed.")
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping.")

        encoded = pickle.dumps(dict(payload), protocol=pickle.HIGHEST_PROTOCOL)
        with self._lock:
            try:
                assert self._socket is not None
                self._socket.send(encoded)
                reply_bytes = self._socket.recv()
            except self._zmq.Again as exc:
                self._rebuild_socket()
                raise TReXTransportTimeout(
                    f"T-Rex request to {self.endpoint} timed out after "
                    f"{self.timeout_ms} ms; REQ socket was reset."
                ) from exc
            except self._zmq.ZMQError as exc:
                self._rebuild_socket()
                raise TReXTransportError(
                    f"T-Rex ZMQ request to {self.endpoint} failed; REQ socket "
                    "was reset."
                ) from exc

        try:
            reply = pickle.loads(reply_bytes)
        except Exception as exc:
            raise TReXWireProtocolError("T-Rex reply is not valid pickle.") from exc
        if not isinstance(reply, Mapping):
            raise TReXWireProtocolError(
                f"T-Rex reply must be a mapping, got {type(reply).__name__}."
            )
        return reply

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                self._close_socket()

    def __enter__(self) -> "ZmqReqTransport":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        self.close()


def _task_identity(task_key: TaskKey) -> tuple[str, int, str, str]:
    return (
        task_key.task_id,
        task_key.task_version,
        task_key.instruction_hash,
        task_key.lease_id,
    )


def _validate_action_chunk(value: Any) -> np.ndarray:
    try:
        actions = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise TReXWireProtocolError("reply actions cannot be converted to float32.") from exc
    expected = (ACTION_CHUNK, ACTION_DIM)
    if actions.shape != expected:
        raise TReXWireProtocolError(
            f"reply actions must have shape {expected}, got {actions.shape}."
        )
    if not np.isfinite(actions).all():
        raise TReXWireProtocolError("reply actions contain NaN or infinity.")
    return actions.copy()


class ZmqTReXBackend:
    """Real T-Rex backend with strict task and server-chunk cache checks.

    ``fast`` requests intentionally send only the current F6 frame plus the
    cached server chunk identifier.  The current server ignores the request's
    ``chunk_id`` field but echoes its own ID in the reply; including the field
    makes the request self-describing and remains forward-compatible with a
    stricter REP server.
    """

    def __init__(
        self,
        transport: RequestReplyTransport | None = None,
        *,
        endpoint: str = "tcp://127.0.0.1:5555",
        timeout_ms: int = 5_000,
        image_key: str = "head",
    ) -> None:
        if not image_key:
            raise ValueError("image_key must be non-empty.")
        self.transport = (
            ZmqReqTransport(endpoint, timeout_ms=timeout_ms)
            if transport is None
            else transport
        )
        if not isinstance(self.transport, RequestReplyTransport):
            raise TypeError("transport must implement request(payload) and close().")
        self.image_key = image_key
        self._server_chunk_id: Any | None = None
        self._task_identity: tuple[str, int, str, str] | None = None
        self._last_actions: np.ndarray | None = None
        self._lock = threading.Lock()

    @property
    def server_chunk_id(self) -> Any | None:
        return self._server_chunk_id

    def _clear_cache(self) -> None:
        self._server_chunk_id = None
        self._task_identity = None
        self._last_actions = None

    def reset(self) -> None:
        """Drop local server-cache identity without closing the transport."""

        with self._lock:
            self._clear_cache()

    def close(self) -> None:
        with self._lock:
            self._clear_cache()
            self.transport.close()

    def _select_head_image(self, observation: PolicyObservation) -> np.ndarray:
        images = observation.images
        if not images:
            raise TReXWireProtocolError(
                "slow/slow_and_fast request requires observation.images."
            )
        candidates = (
            self.image_key,
            "image_head",
            "head",
            "rgb",
            "camera",
        )
        selected = None
        for key in candidates:
            if key in images:
                selected = images[key]
                break
        if selected is None:
            if len(images) != 1:
                raise TReXWireProtocolError(
                    f"image key {self.image_key!r} was not found and the mapping "
                    "contains multiple cameras."
                )
            selected = next(iter(images.values()))

        image = np.asarray(selected)
        if image.shape[-1:] != (3,) or image.ndim != 3:
            raise TReXWireProtocolError(
                f"head image must have shape HxWx3, got {image.shape}."
            )
        if image.dtype != np.uint8:
            raise TReXWireProtocolError(
                f"head image must be uint8 RGB, got dtype {image.dtype}."
            )
        return np.ascontiguousarray(image)

    @staticmethod
    def _png_bytes(image: np.ndarray) -> bytes:
        # Lazy optional dependency for the same reason as pyzmq above.
        try:
            from PIL import Image  # type: ignore
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "Encoding image_head requires Pillow; install package 'Pillow'."
            ) from exc
        output = io.BytesIO()
        Image.fromarray(image, mode="RGB").save(output, format="PNG")
        return output.getvalue()

    def _slow_payload(self, mode: str, observation: PolicyObservation) -> dict[str, Any]:
        image = self._select_head_image(observation)
        return {
            "mode": mode,
            "image_head": self._png_bytes(image),
            "task_description": observation.instruction,
            "state_fast": observation.q_rad.copy(),
            "tactile_f6": observation.tactile_f6.copy(),
        }

    def _request(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        try:
            response = self.transport.request(payload)
        except TReXZmqError:
            self._clear_cache()
            raise
        except TimeoutError as exc:
            self._clear_cache()
            raise TReXTransportTimeout("Injected T-Rex transport timed out.") from exc
        except Exception as exc:
            self._clear_cache()
            raise TReXTransportError("Injected T-Rex transport failed.") from exc
        if not isinstance(response, Mapping):
            self._clear_cache()
            raise TReXWireProtocolError("transport response must be a mapping.")
        return response

    @staticmethod
    def _validate_common_reply(
        response: Mapping[str, Any], *, expected_mode: str
    ) -> tuple[np.ndarray, Any]:
        if response.get("status") != "success":
            message = response.get("message", "unspecified server error")
            raise TReXWireProtocolError(f"T-Rex server returned error: {message}")
        if response.get("mode") != expected_mode:
            raise TReXWireProtocolError(
                f"reply mode {response.get('mode')!r} does not match "
                f"request mode {expected_mode!r}."
            )
        if "chunk_id" not in response or response["chunk_id"] is None:
            raise TReXWireProtocolError("successful reply is missing chunk_id.")
        chunk_id = response["chunk_id"]
        if isinstance(chunk_id, str) and not chunk_id:
            raise TReXWireProtocolError("reply chunk_id must not be empty.")
        actions = _validate_action_chunk(response.get("actions"))
        return actions, chunk_id

    def _begin(self, mode: str, observation: PolicyObservation) -> np.ndarray:
        response = self._request(self._slow_payload(mode, observation))
        try:
            actions, chunk_id = self._validate_common_reply(
                response, expected_mode=mode
            )
        except Exception:
            self._clear_cache()
            raise
        self._server_chunk_id = chunk_id
        self._task_identity = _task_identity(observation.task_key)
        self._last_actions = actions.copy()
        return actions

    def slow_and_fast(self, observation: PolicyObservation) -> np.ndarray:
        with self._lock:
            return self._begin("slow_and_fast", observation)

    def slow(self, observation: PolicyObservation) -> np.ndarray:
        with self._lock:
            return self._begin("slow", observation)

    def fast(
        self,
        observation: PolicyObservation,
        cached_chunk: np.ndarray,
        chunk_offset: int,
    ) -> np.ndarray:
        del chunk_offset  # cadence is enforced by the policy adapter/schedule.
        with self._lock:
            if (
                self._server_chunk_id is None
                or self._task_identity is None
                or self._last_actions is None
            ):
                raise TReXWireProtocolError(
                    "fast request has no matching server-side slow cache."
                )
            if _task_identity(observation.task_key) != self._task_identity:
                self._clear_cache()
                raise TReXWireProtocolError(
                    "fast request task/version/instruction/lease does not match "
                    "the cached slow request."
                )
            supplied_cache = _validate_action_chunk(cached_chunk)
            if not np.array_equal(supplied_cache, self._last_actions):
                self._clear_cache()
                raise TReXWireProtocolError(
                    "fast request cached_chunk does not match the last server reply."
                )

            expected_chunk_id = self._server_chunk_id
            response = self._request(
                {
                    "mode": "fast",
                    "chunk_id": expected_chunk_id,
                    "tactile_f6": observation.tactile_f6.copy(),
                }
            )
            try:
                actions, reply_chunk_id = self._validate_common_reply(
                    response, expected_mode="fast"
                )
            except Exception:
                self._clear_cache()
                raise
            if reply_chunk_id != expected_chunk_id:
                self._clear_cache()
                raise TReXWireProtocolError(
                    f"fast reply chunk_id {reply_chunk_id!r} does not match "
                    f"cached server chunk_id {expected_chunk_id!r}."
                )
            self._last_actions = actions.copy()
            return actions

    def __enter__(self) -> "ZmqTReXBackend":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        self.close()

