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
import time
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .contracts import ACTION_CHUNK, ACTION_DIM, PolicyObservation, TaskKey
from .server_identity import TReXServerIdentity


OFFICIAL_SINGLE_VIEW_PROFILE = "official_single_view"
REVO3_FULL_CENTER_PROFILE = "revo3_full_center_v1"
_CAMERA_PROFILES = {
    "auto",
    OFFICIAL_SINGLE_VIEW_PROFILE,
    REVO3_FULL_CENTER_PROFILE,
}
_REVO3_IMAGE_SHAPE = (288, 384, 3)


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


def _task_identity(task_key: TaskKey) -> tuple[str, int, str, str, str]:
    return (
        task_key.task_id,
        task_key.task_version,
        task_key.instruction_hash,
        task_key.lease_id,
        task_key.version_fingerprint,
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
        image_profile: str = "auto",
        full_image_key: str = "full",
        center_image_key: str = "fixed_center",
        tactile_profile: str = "legacy_force6d",
        expected_server_identity: TReXServerIdentity | Mapping[str, Any] | None = None,
        clock: Any = time.monotonic_ns,
        max_response_age_ns: int = 1_000_000_000,
        max_tactile_age_ns: int = 150_000_000,
        slow_response_observation_budget_ns: int = 1_500_000_000,
        fast_response_observation_budget_ns: int = 500_000_000,
    ) -> None:
        if not image_key:
            raise ValueError("image_key must be non-empty.")
        if image_profile not in _CAMERA_PROFILES:
            raise ValueError(
                f"image_profile must be one of {sorted(_CAMERA_PROFILES)}."
            )
        if not full_image_key or not center_image_key or full_image_key == center_image_key:
            raise ValueError("full and center image keys must be distinct and non-empty.")
        self.transport = (
            ZmqReqTransport(endpoint, timeout_ms=timeout_ms)
            if transport is None
            else transport
        )
        if not isinstance(self.transport, RequestReplyTransport):
            raise TypeError("transport must implement request(payload) and close().")
        self.image_key = image_key
        self.image_profile = image_profile
        self.full_image_key = full_image_key
        self.center_image_key = center_image_key
        if tactile_profile not in {
            "legacy_force6d",
            "profile_a_force6d_diff",
            "profile_b_diff_only",
            "ablation_force6d_only",
        }:
            raise ValueError("unsupported tactile_profile")
        if any(
            int(value) <= 0
            for value in (
                max_response_age_ns,
                max_tactile_age_ns,
                slow_response_observation_budget_ns,
                fast_response_observation_budget_ns,
            )
        ):
            raise ValueError("response/tactile latency limits must be positive")
        if int(fast_response_observation_budget_ns) > int(
            slow_response_observation_budget_ns
        ):
            raise ValueError("fast response budget cannot exceed slow response budget")
        self.tactile_profile = tactile_profile
        if expected_server_identity is None:
            self.expected_server_identity = None
        elif isinstance(expected_server_identity, TReXServerIdentity):
            self.expected_server_identity = expected_server_identity
        else:
            try:
                self.expected_server_identity = TReXServerIdentity.from_mapping(
                    expected_server_identity
                )
            except (TypeError, ValueError) as exc:
                raise ValueError("invalid expected_server_identity") from exc
        if (
            image_profile == REVO3_FULL_CENTER_PROFILE
            and self.expected_server_identity is None
        ):
            raise ValueError(
                "revo3_full_center_v1 requires a pinned expected_server_identity"
            )
        if self.expected_server_identity is not None:
            if (
                image_profile != "auto"
                and self.expected_server_identity.camera_profile != image_profile
            ):
                raise ValueError(
                    "expected server camera profile does not match backend image_profile"
                )
            if self.expected_server_identity.tactile_profile != tactile_profile:
                raise ValueError(
                    "expected server tactile profile does not match backend tactile_profile"
                )
        self._clock = clock
        self.max_response_age_ns = int(max_response_age_ns)
        self.max_tactile_age_ns = int(max_tactile_age_ns)
        self.slow_response_observation_budget_ns = int(
            slow_response_observation_budget_ns
        )
        self.fast_response_observation_budget_ns = int(
            fast_response_observation_budget_ns
        )
        self._active_image_profile: str | None = None
        self._server_chunk_id: Any | None = None
        self._task_identity: tuple[str, int, str, str, str] | None = None
        self._last_actions: np.ndarray | None = None
        self._verified_server_identity: TReXServerIdentity | None = None
        self._lock = threading.Lock()

    @property
    def server_chunk_id(self) -> Any | None:
        return self._server_chunk_id

    @property
    def verified_server_identity(self) -> TReXServerIdentity | None:
        return self._verified_server_identity

    def _clear_cache(self) -> None:
        self._server_chunk_id = None
        self._task_identity = None
        self._last_actions = None
        self._active_image_profile = None

    def reset(self) -> None:
        """Drop local server-cache identity without closing the transport."""

        with self._lock:
            self._clear_cache()

    def close(self) -> None:
        with self._lock:
            self._clear_cache()
            self.transport.close()

    @staticmethod
    def _validate_image(image: Any, *, name: str, exact_revo_shape: bool) -> np.ndarray:
        image = np.asarray(image)
        if image.shape[-1:] != (3,) or image.ndim != 3:
            raise TReXWireProtocolError(
                f"{name} must have shape HxWx3, got {image.shape}."
            )
        if image.dtype != np.uint8:
            raise TReXWireProtocolError(
                f"{name} must be uint8 RGB, got dtype {image.dtype}."
            )
        if exact_revo_shape and image.shape != _REVO3_IMAGE_SHAPE:
            raise TReXWireProtocolError(
                f"{name} must be the frozen Revo3 384x288 view with array "
                f"shape {_REVO3_IMAGE_SHAPE}, got {image.shape}."
            )
        return np.ascontiguousarray(image)

    def _select_single_image(self, observation: PolicyObservation) -> np.ndarray:
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
        if len(images) != 1:
            raise TReXWireProtocolError(
                "official_single_view cannot silently discard extra images; "
                "select revo3_full_center_v1 for the Revo3 dual derived views."
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

        return self._validate_image(selected, name="image_head", exact_revo_shape=False)

    def _resolve_image_profile(self, observation: PolicyObservation) -> str:
        if self.image_profile != "auto":
            return self.image_profile
        images = observation.images or {}
        has_full = self.full_image_key in images
        has_center = self.center_image_key in images
        if has_full or has_center:
            if not (has_full and has_center):
                raise TReXWireProtocolError(
                    "Revo3 view mapping is partial; both full and fixed_center are required."
                )
            if len(images) != 2:
                raise TReXWireProtocolError(
                    "Revo3 dual-view mapping contains undeclared extra images."
                )
            return REVO3_FULL_CENTER_PROFILE
        return OFFICIAL_SINGLE_VIEW_PROFILE

    def _image_payload(self, observation: PolicyObservation) -> dict[str, Any]:
        profile = self._resolve_image_profile(observation)
        images = observation.images or {}
        if profile == OFFICIAL_SINGLE_VIEW_PROFILE:
            image = self._select_single_image(observation)
            return {"image_head": self._png_bytes(image)}
        if self.full_image_key not in images or self.center_image_key not in images:
            raise TReXWireProtocolError(
                "revo3_full_center_v1 requires both full and fixed_center images."
            )
        if len(images) != 2:
            raise TReXWireProtocolError(
                "revo3_full_center_v1 accepts exactly the two declared derived views."
            )
        full = self._validate_image(
            images[self.full_image_key], name="full", exact_revo_shape=True
        )
        center = self._validate_image(
            images[self.center_image_key], name="fixed_center", exact_revo_shape=True
        )
        return {
            "camera_profile": REVO3_FULL_CENTER_PROFILE,
            "capture_timestamp_ns": int(observation.rgb_timestamp_ns),
            "image_head": self._png_bytes(full),
            # The official server treats wrist images as the fast visual token
            # group.  Revo3 deterministically maps its centre crop to that slot.
            "image_wrist_right": self._png_bytes(center),
            "image_view_names": ("full", "fixed_center"),
        }

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
        image_payload = self._image_payload(observation)
        payload = {
            "mode": mode,
            **image_payload,
            "task_description": observation.instruction,
            "state_fast": observation.q_rad.copy(),
        }
        if observation.tactile_f6 is not None:
            payload["tactile_f6"] = observation.tactile_f6.copy()
        if self._resolve_image_profile(observation) == REVO3_FULL_CENTER_PROFILE:
            payload.update(self._revo_runtime_payload(observation))
        return payload

    def _revo_runtime_payload(self, observation: PolicyObservation) -> dict[str, Any]:
        key = observation.task_key
        if observation.tactile_profile != self.tactile_profile:
            raise TReXWireProtocolError(
                "PolicyObservation tactile_profile does not match backend checkpoint profile."
            )
        if not key.version_fingerprint:
            raise TReXWireProtocolError("Revo3 request requires version_fingerprint.")
        if observation.lease_expires_at_ns is None:
            raise TReXWireProtocolError("Revo3 request requires lease_expires_at_ns.")
        now_ns = int(self._clock())
        age_ns = now_ns - int(observation.timestamp_ns)
        if age_ns < 0 or age_ns > self.max_response_age_ns:
            raise TReXWireProtocolError("Revo3 observation is future-dated or stale before send.")
        if now_ns >= int(observation.lease_expires_at_ns):
            raise TReXWireProtocolError("Revo3 task lease expired before send.")
        requires_diff = self.tactile_profile in {
            "profile_a_force6d_diff",
            "profile_b_diff_only",
        }
        requires_force = self.tactile_profile in {
            "profile_a_force6d_diff",
            "ablation_force6d_only",
        }
        if self.tactile_profile == "legacy_force6d":
            raise TReXWireProtocolError(
                "Revo3 mainline must declare profile_a/profile_b/force6d ablation; legacy tactile is forbidden."
            )
        if requires_diff and observation.tactile_deform is None:
            raise TReXWireProtocolError(f"{self.tactile_profile} requires five DIFF images.")
        if requires_diff:
            diff_timestamps = np.asarray(
                observation.tactile_deform_timestamp_ns, dtype=np.int64
            )
            diff_age_ns = int(observation.timestamp_ns) - diff_timestamps
            if np.any(diff_age_ns < 0):
                raise TReXWireProtocolError("Revo3 current DIFF is future-dated.")
            if np.any(diff_age_ns > self.max_tactile_age_ns):
                raise TReXWireProtocolError("Revo3 current DIFF is stale.")
        if requires_force and (
            observation.tactile_f6 is None
            or observation.tactile_history_f6 is None
            or observation.tactile_history_timestamps_ns is None
            or observation.tactile_history_sequences is None
        ):
            raise TReXWireProtocolError(
                f"{self.tactile_profile} requires complete Force6D history and metadata."
            )
        if not requires_force and (
            observation.tactile_f6 is not None
            or observation.tactile_history_f6 is not None
            or observation.tactile_history_timestamps_ns is not None
            or observation.tactile_history_sequences is not None
        ):
            raise TReXWireProtocolError("profile_b_diff_only forbids Force6D tensors.")
        payload: dict[str, Any] = {
            "task_id": key.task_id,
            "task_version": key.task_version,
            "instruction_hash": key.instruction_hash,
            "lease_id": key.lease_id,
            "lease_expires_at_ns": int(observation.lease_expires_at_ns),
            "version_fingerprint": key.version_fingerprint,
            "observation_timestamp_ns": int(observation.timestamp_ns),
            "state_timestamp_ns": int(observation.state_timestamp_ns),
            "rgb_timestamp_ns": int(observation.rgb_timestamp_ns),
            "tactile_timestamp_ns": int(observation.tactile_timestamp_ns),
            "request_sent_at_ns": now_ns,
            "tactile_profile": self.tactile_profile,
        }
        if requires_force:
            payload.update({
                "tactile_f6": observation.tactile_f6.copy(),
                "tactile_f6_history": observation.tactile_history_f6.copy(),
                "tactile_f6_history_timestamps_ns": observation.tactile_history_timestamps_ns.copy(),
                "tactile_f6_history_sequences": observation.tactile_history_sequences.copy(),
            })
        if requires_diff:
            payload["tactile_deform"] = observation.tactile_deform.copy()
            payload["tactile_deform_timestamp_ns"] = observation.tactile_deform_timestamp_ns.copy()
        return payload

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

    def _validate_server_identity(
        self, response: Mapping[str, Any]
    ) -> TReXServerIdentity:
        if self.expected_server_identity is None:
            raise TReXWireProtocolError(
                "Revo3 response cannot be accepted without a pinned server identity."
            )
        try:
            observed = TReXServerIdentity.from_mapping(response.get("server_identity"))
        except (TypeError, ValueError) as exc:
            raise TReXWireProtocolError(
                "reply is missing or has an invalid server-owned identity"
            ) from exc
        if observed != self.expected_server_identity:
            expected = self.expected_server_identity.as_mapping()
            actual = observed.as_mapping()
            mismatched = sorted(
                name for name, value in expected.items() if actual.get(name) != value
            )
            raise TReXWireProtocolError(
                f"reply server identity mismatch: {mismatched}"
            )
        self._verified_server_identity = observed
        return observed

    def probe_server_identity(self) -> TReXServerIdentity:
        """Fail-closed live handshake before a production runtime is assembled."""

        with self._lock:
            response = self._request({"mode": "identity"})
            try:
                identity = self._validate_server_identity(response)
                if response.get("status") != "success" or response.get("mode") != "identity":
                    raise TReXWireProtocolError(
                        "server identity probe did not return a successful identity reply"
                    )
            except Exception:
                self._clear_cache()
                raise
            return identity

    def _validate_common_reply(
        self,
        response: Mapping[str, Any],
        *,
        expected_mode: str,
        observation: PolicyObservation,
        strict_revo: bool,
    ) -> tuple[np.ndarray, Any]:
        if strict_revo:
            # This check intentionally precedes action parsing and cache
            # assignment.  A task/lease-correct reply from the wrong model is
            # still unusable.
            self._validate_server_identity(response)
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
        if strict_revo:
            expected_identity = {
                "task_id": observation.task_key.task_id,
                "task_version": observation.task_key.task_version,
                "instruction_hash": observation.task_key.instruction_hash,
                "lease_id": observation.task_key.lease_id,
                "version_fingerprint": observation.task_key.version_fingerprint,
                "observation_timestamp_ns": observation.timestamp_ns,
            }
            mismatched = [
                name for name, expected in expected_identity.items()
                if response.get(name) != expected
            ]
            if mismatched:
                raise TReXWireProtocolError(
                    f"reply identity/timestamp echo mismatch: {mismatched}"
                )
            produced_at_ns = response.get("produced_at_ns")
            now_ns = int(self._clock())
            if not isinstance(produced_at_ns, (int, np.integer)):
                raise TReXWireProtocolError("Revo3 reply lacks produced_at_ns.")
            if produced_at_ns < observation.timestamp_ns:
                raise TReXWireProtocolError("reply predates its observation.")
            if produced_at_ns > now_ns or now_ns - produced_at_ns > self.max_response_age_ns:
                raise TReXWireProtocolError("reply produced_at_ns is future-dated or stale.")
            observation_latency_ns = now_ns - observation.timestamp_ns
            latency_budget_ns = (
                self.fast_response_observation_budget_ns
                if expected_mode == "fast"
                else self.slow_response_observation_budget_ns
            )
            if observation_latency_ns > latency_budget_ns:
                raise TReXWireProtocolError(
                    f"{expected_mode} reply exceeded its response-observation latency budget."
                )
            if observation.lease_expires_at_ns is None or now_ns >= observation.lease_expires_at_ns:
                raise TReXWireProtocolError("task lease expired before reply validation.")
        return actions, chunk_id

    def _begin(self, mode: str, observation: PolicyObservation) -> np.ndarray:
        profile = self._resolve_image_profile(observation)
        if (
            profile == REVO3_FULL_CENTER_PROFILE
            and self.expected_server_identity is None
        ):
            self._clear_cache()
            raise TReXWireProtocolError(
                "Revo3 request requires a pinned expected_server_identity before send."
            )
        response = self._request(self._slow_payload(mode, observation))
        try:
            actions, chunk_id = self._validate_common_reply(
                response,
                expected_mode=mode,
                observation=observation,
                strict_revo=profile == REVO3_FULL_CENTER_PROFILE,
            )
        except Exception:
            self._clear_cache()
            raise
        self._server_chunk_id = chunk_id
        self._task_identity = _task_identity(observation.task_key)
        self._last_actions = actions.copy()
        self._active_image_profile = profile
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
            fast_payload = {
                "mode": "fast",
                "chunk_id": expected_chunk_id,
                "camera_profile": self._active_image_profile,
                **(
                    self._revo_runtime_payload(observation)
                    if self._active_image_profile == REVO3_FULL_CENTER_PROFILE
                    else {}
                ),
            }
            if (
                self._active_image_profile != REVO3_FULL_CENTER_PROFILE
                and observation.tactile_f6 is not None
            ):
                fast_payload["tactile_f6"] = observation.tactile_f6.copy()
            response = self._request(fast_payload)
            try:
                actions, reply_chunk_id = self._validate_common_reply(
                    response,
                    expected_mode="fast",
                    observation=observation,
                    strict_revo=self._active_image_profile == REVO3_FULL_CENTER_PROFILE,
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
