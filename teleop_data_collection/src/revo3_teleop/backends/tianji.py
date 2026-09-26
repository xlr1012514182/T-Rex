"""Fail-closed Tianji Marvin 7-DoF SDK boundary.

This module deliberately contains no Tianji/Wuji SDK source, binary, ctypes
structure, or implicit library loader.  The caller injects an already-created
native client and, for the real ``OnGetBuf(pointer)`` ABI, a feedback-buffer
factory and decoder supplied by the locally installed SDK.

The documented Marvin boundary uses degrees and degrees/second.  Everything
exposed by this module uses SI units: radians, radians/second, and Nm.  A
position write is considered successful only after the complete native
transaction ``OnClearSet -> OnSetJointCmdPos_{A|B} -> OnSetSend`` succeeds.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import ipaddress
import math
import numbers
import threading
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from revo3_teleop.contracts import CommandReceipt


TIANJI_JOINT_COUNT = 7
# The public SDK exposes only array positions 0..6 at this boundary.  Do not
# invent anatomical names before the exact Tianji model/joint convention is
# confirmed on the project hardware.
TIANJI_SDK_JOINT_ORDER = tuple(f"sdk_joint_{index}" for index in range(7))


def tianji_joint_order_hash(joint_order: Sequence[str]) -> str:
    """Return the stable controller-label hash for one verified 7-axis order."""

    normalized = tuple(str(name).strip() for name in joint_order)
    if len(normalized) != TIANJI_JOINT_COUNT:
        raise ValueError("Tianji joint_order must contain exactly 7 names")
    if any(not name for name in normalized) or len(set(normalized)) != len(normalized):
        raise ValueError("Tianji joint_order names must be non-empty and unique")
    return hashlib.sha256("\n".join(normalized).encode("utf-8")).hexdigest()


TIANJI_JOINT_ORDER_HASH = tianji_joint_order_hash(TIANJI_SDK_JOINT_ORDER)


class TianjiBackendError(RuntimeError):
    """Base error for the isolated Tianji native boundary."""


class TianjiWriteNotArmed(TianjiBackendError):
    """A motion-producing operation did not pass every arming gate."""


class TianjiFeedbackError(TianjiBackendError):
    """The native feedback payload is missing, malformed, or unsafe."""


class TianjiPhysicalInterventionRequired(TianjiBackendError):
    """Servo-off could not be confirmed; physical intervention is required."""


class TianjiSide(str, Enum):
    A = "A"
    B = "B"

    @property
    def index(self) -> int:
        return 0 if self is TianjiSide.A else 1

    @classmethod
    def parse(cls, value: "TianjiSide | str") -> "TianjiSide":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().upper())
        except ValueError as exc:
            raise ValueError("Tianji side must be 'A' or 'B'") from exc


def _vector7(value: Any, *, name: str, dtype: Any = np.float64) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.shape != (TIANJI_JOINT_COUNT,):
        raise ValueError(
            f"{name} must have shape ({TIANJI_JOINT_COUNT},), got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return array.copy()


def _positive_ns(value: Any, *, name: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _sdk_scalar(value: Any) -> Any:
    return value.value if hasattr(value, "value") else value


def _strict_int(value: Any, *, name: str, nonnegative: bool = False) -> int:
    raw = _sdk_scalar(value)
    if isinstance(raw, bool) or not isinstance(raw, numbers.Integral):
        try:
            converted = int(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TianjiFeedbackError(f"{name} is not an integer") from exc
        try:
            if float(raw) != float(converted):
                raise TianjiFeedbackError(f"{name} is not an integer")
        except (TypeError, ValueError, OverflowError) as exc:
            raise TianjiFeedbackError(f"{name} is not an integer") from exc
        result = converted
    else:
        result = int(raw)
    if nonnegative and result < 0:
        raise TianjiFeedbackError(f"{name} must be non-negative")
    return result


@dataclass(frozen=True)
class TianjiSafetyLimits:
    """Bench-verified limits required before any Tianji motion write.

    There are intentionally no model-specific numeric defaults.  The caller
    must supply limits validated for the exact arm, firmware, payload, tool
    transform, workspace, and physical test fixture.
    """

    q_min_rad: np.ndarray
    q_max_rad: np.ndarray
    max_delta_rad: np.ndarray | float
    max_feedback_age_ns: int = 250_000_000
    max_target_age_ns: int = 100_000_000
    require_wrist_pose: bool = True

    def __post_init__(self) -> None:
        lower = _vector7(self.q_min_rad, name="q_min_rad")
        upper = _vector7(self.q_max_rad, name="q_max_rad")
        if np.any(lower >= upper):
            raise ValueError("every q_min_rad entry must be less than q_max_rad")
        delta_raw = np.asarray(self.max_delta_rad, dtype=np.float64)
        if delta_raw.ndim == 0:
            delta = np.full(TIANJI_JOINT_COUNT, float(delta_raw), dtype=np.float64)
        else:
            delta = _vector7(delta_raw, name="max_delta_rad")
        if not np.isfinite(delta).all() or np.any(delta <= 0.0):
            raise ValueError("max_delta_rad must contain finite positive values")
        lower.setflags(write=False)
        upper.setflags(write=False)
        delta.setflags(write=False)
        object.__setattr__(self, "q_min_rad", lower)
        object.__setattr__(self, "q_max_rad", upper)
        object.__setattr__(self, "max_delta_rad", delta)
        object.__setattr__(
            self,
            "max_feedback_age_ns",
            _positive_ns(self.max_feedback_age_ns, name="max_feedback_age_ns"),
        )
        object.__setattr__(
            self,
            "max_target_age_ns",
            _positive_ns(self.max_target_age_ns, name="max_target_age_ns"),
        )


@dataclass(frozen=True)
class TianjiArmState:
    """One decoded SDK frame in SI units.

    ``timestamp_ns`` is the host-monotonic time at which ``frame_serial`` last
    advanced.  Re-reading a frozen native buffer therefore makes this
    timestamp age instead of falsely refreshing the state.
    """

    side: TianjiSide
    timestamp_ns: int
    receive_timestamp_ns: int
    q_rad: np.ndarray
    dq_rad_s: np.ndarray
    tau_nm: np.ndarray
    cur_state: int
    cmd_state: int
    err_code: int
    frame_serial: int
    frame_miss_count: int = 0
    max_frame_miss_count: int = 0
    read_sequence: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "side", TianjiSide.parse(self.side))
        if self.timestamp_ns < 0 or self.receive_timestamp_ns < self.timestamp_ns:
            raise ValueError("invalid Tianji feedback timestamps")
        if self.read_sequence < 0:
            raise ValueError("read_sequence must be non-negative")
        for name in ("q_rad", "dq_rad_s", "tau_nm"):
            array = _vector7(getattr(self, name), name=name, dtype=np.float32)
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        for name in ("frame_serial", "frame_miss_count", "max_frame_miss_count"):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")

    def age_ns(self, now_ns: int) -> int:
        return max(0, int(now_ns) - self.timestamp_ns)


@dataclass(frozen=True)
class TianjiCapabilityReport:
    side: TianjiSide
    required_methods: tuple[str, ...]
    missing_methods: tuple[str, ...]
    checked_timestamp_ns: int

    @property
    def complete(self) -> bool:
        return not self.missing_methods


@dataclass(frozen=True)
class _ParsedFeedback:
    q_deg: np.ndarray
    dq_deg_s: np.ndarray
    tau_nm: np.ndarray
    cur_state: int
    cmd_state: int
    err_code: int
    frame_serial: int
    frame_miss_count: int
    max_frame_miss_count: int


FeedbackBufferFactory = Callable[[], Any]
FeedbackDecoder = Callable[[Any], Mapping[str, Any]]
FeedbackArgumentAdapter = Callable[[Any], Any]
Clock = Callable[[], int]
Sleeper = Callable[[float], None]


class TianjiMarvinBackend:
    """Narrow, injected adapter for one Tianji Marvin arm side.

    Motion is disabled by default.  A position write needs all three explicit
    conditions: ``allow_hardware_write=True``, a confirmed external capability
    probe, and a per-call token equal to the configured arm token.  Method
    presence is checked again at runtime; a Boolean alone cannot bypass a
    missing native capability.
    """

    hardware_autostart = False

    def __init__(
        self,
        client: Any,
        *,
        side: TianjiSide | str,
        safety_limits: TianjiSafetyLimits | None = None,
        allow_hardware_write: bool = False,
        capability_probe_confirmed: bool = False,
        arm_token: str | None = None,
        feedback_buffer_factory: FeedbackBufferFactory | None = None,
        feedback_decoder: FeedbackDecoder | None = None,
        feedback_argument_adapter: FeedbackArgumentAdapter | None = None,
        joint_order: Sequence[str] = TIANJI_SDK_JOINT_ORDER,
        clock: Clock = time.monotonic_ns,
        sleeper: Sleeper = time.sleep,
    ) -> None:
        if (feedback_buffer_factory is None) != (feedback_decoder is None):
            raise ValueError(
                "feedback_buffer_factory and feedback_decoder must be supplied together"
            )
        token = None if arm_token is None else str(arm_token).strip()
        if arm_token is not None and not token:
            raise ValueError("arm_token must be non-empty when supplied")
        self.client = client
        self.side = TianjiSide.parse(side)
        self.safety_limits = safety_limits
        self.allow_hardware_write = bool(allow_hardware_write)
        self.capability_probe_confirmed = bool(capability_probe_confirmed)
        self._expected_arm_token = token
        self.joint_order = tuple(str(name).strip() for name in joint_order)
        self.joint_order_hash = tianji_joint_order_hash(self.joint_order)
        self._feedback_decoder = feedback_decoder
        self._feedback_buffer = (
            None if feedback_buffer_factory is None else feedback_buffer_factory()
        )
        self._feedback_argument_adapter = feedback_argument_adapter or ctypes.byref
        self._clock = clock
        self._sleeper = sleeper
        self._connected = False
        self._last_frame_serial: int | None = None
        self._last_serial_advance_ns: int | None = None
        self._serial_has_advanced = False
        self._read_sequence = 0
        self._controller_sequence = 0
        self._last_sent_target_rad: np.ndarray | None = None
        self._last_sent_target_timestamp_ns: int | None = None
        self._lock = threading.RLock()

    @property
    def connected(self) -> bool:
        return self._connected

    def _required_motion_methods(self) -> tuple[str, ...]:
        suffix = self.side.value
        return (
            "OnClearSet",
            f"OnSetJointCmdPos_{suffix}",
            "OnSetSend",
            f"OnSetTargetState_{suffix}",
            f"OnEMG_{suffix}",
        )

    def probe_capabilities(self) -> TianjiCapabilityReport:
        required = (
            "OnLinkTo",
            "OnRelease",
            "OnGetBuf",
            *self._required_motion_methods(),
        )
        missing = tuple(
            name for name in required if not callable(getattr(self.client, name, None))
        )
        return TianjiCapabilityReport(
            side=self.side,
            required_methods=required,
            missing_methods=missing,
            checked_timestamp_ns=int(self._clock()),
        )

    def _method(self, name: str) -> Callable[..., Any]:
        method = getattr(self.client, name, None)
        if not callable(method):
            raise TianjiBackendError(f"injected Tianji client has no callable {name}")
        return method

    @staticmethod
    def _result_succeeded(result: Any, *, allow_none: bool = False) -> bool:
        if result is None:
            return allow_none
        raw = _sdk_scalar(result)
        if isinstance(raw, (bool, np.bool_)):
            return bool(raw)
        if isinstance(raw, numbers.Integral):
            return int(raw) == 1
        return False

    def _call_checked(self, name: str, *args: Any, allow_none: bool = False) -> Any:
        result = self._method(name)(*args)
        if not self._result_succeeded(result, allow_none=allow_none):
            raise TianjiBackendError(f"Tianji SDK {name} failed with {result!r}")
        return result

    def connect(self, robot_ip: str = "192.168.1.190") -> None:
        """Open the one SDK TCP session; no servo or motion command is sent."""

        with self._lock:
            if self._connected:
                raise TianjiBackendError("Tianji backend is already connected")
            try:
                address = ipaddress.IPv4Address(str(robot_ip).strip())
            except ipaddress.AddressValueError as exc:
                raise ValueError(f"invalid Tianji IPv4 address: {robot_ip!r}") from exc
            octets = tuple(ctypes.c_ubyte(value) for value in address.packed)
            self._call_checked("OnLinkTo", *octets)
            self._connected = True
            self._last_frame_serial = None
            self._last_serial_advance_ns = None
            self._serial_has_advanced = False
            self._last_sent_target_rad = None
            self._last_sent_target_timestamp_ns = None

    def _require_connected(self) -> None:
        if not self._connected:
            raise TianjiBackendError("Tianji backend is not connected")

    @staticmethod
    def _side_item(container: Any, side: TianjiSide, *, name: str) -> Mapping[str, Any]:
        index = side.index
        try:
            if isinstance(container, Mapping):
                candidates: tuple[Any, ...] = (
                    side.value,
                    side.value.lower(),
                    index,
                    str(index),
                )
                item = next(container[key] for key in candidates if key in container)
            else:
                item = container[index]
        except (IndexError, KeyError, StopIteration, TypeError) as exc:
            raise TianjiFeedbackError(f"feedback {name} has no side {side.value}") from exc
        if not isinstance(item, Mapping):
            raise TianjiFeedbackError(f"feedback {name}[{index}] must be a mapping")
        return item

    def _decode_feedback(self, payload: Mapping[str, Any]) -> _ParsedFeedback:
        if not isinstance(payload, Mapping):
            raise TianjiFeedbackError("OnGetBuf decoder must return a mapping")
        try:
            state = self._side_item(payload["states"], self.side, name="states")
            output = self._side_item(payload["outputs"], self.side, name="outputs")
        except KeyError as exc:
            raise TianjiFeedbackError(
                f"feedback is missing top-level key {exc.args[0]!r}"
            ) from exc
        inputs_container = payload.get("inputs")
        inputs: Mapping[str, Any] = {}
        if inputs_container is not None:
            inputs = self._side_item(inputs_container, self.side, name="inputs")
        try:
            q_deg = _vector7(output["fb_joint_pos"], name="fb_joint_pos")
            dq_deg_s = _vector7(output["fb_joint_vel"], name="fb_joint_vel")
            # The public normalized dictionary exposes sensor torque as
            # fb_joint_sToq.  Do not use the broken fb_joint_tor spelling from
            # TianjiChestDriver.get_current_joint_torques().
            tau_nm = _vector7(output["fb_joint_sToq"], name="fb_joint_sToq")
            frame_serial = _strict_int(
                output["frame_serial"], name="frame_serial", nonnegative=True
            )
            cur_state = _strict_int(state["cur_state"], name="cur_state")
            cmd_state = _strict_int(state["cmd_state"], name="cmd_state")
            err_code = _strict_int(state["err_code"], name="err_code")
        except KeyError as exc:
            raise TianjiFeedbackError(f"feedback is missing field {exc.args[0]!r}") from exc
        return _ParsedFeedback(
            q_deg=q_deg,
            dq_deg_s=dq_deg_s,
            tau_nm=tau_nm,
            cur_state=cur_state,
            cmd_state=cmd_state,
            err_code=err_code,
            frame_serial=frame_serial,
            frame_miss_count=_strict_int(
                inputs.get("frame_miss_cnt", 0),
                name="frame_miss_cnt",
                nonnegative=True,
            ),
            max_frame_miss_count=_strict_int(
                inputs.get("max_frame_miss_cnt", 0),
                name="max_frame_miss_cnt",
                nonnegative=True,
            ),
        )

    def _get_feedback_mapping(self) -> Mapping[str, Any]:
        method = self._method("OnGetBuf")
        if self._feedback_buffer is None:
            payload = method()
            if not isinstance(payload, Mapping):
                raise TianjiFeedbackError(
                    "OnGetBuf did not return a normalized mapping; inject the local "
                    "SDK feedback_buffer_factory and feedback_decoder"
                )
            return payload
        result = method(self._feedback_argument_adapter(self._feedback_buffer))
        if not self._result_succeeded(result, allow_none=True):
            raise TianjiFeedbackError(f"Tianji SDK OnGetBuf failed with {result!r}")
        assert self._feedback_decoder is not None
        payload = self._feedback_decoder(self._feedback_buffer)
        if not isinstance(payload, Mapping):
            raise TianjiFeedbackError("feedback_decoder must return a mapping")
        return payload

    def read_state(self) -> TianjiArmState:
        """Read one coherent native frame and convert deg/deg/s to SI."""

        with self._lock:
            self._require_connected()
            receive_ns = int(self._clock())
            if receive_ns < 0:
                raise TianjiFeedbackError("clock returned a negative timestamp")
            parsed = self._decode_feedback(self._get_feedback_mapping())
            if self._last_frame_serial is None:
                self._last_serial_advance_ns = receive_ns
            elif parsed.frame_serial != self._last_frame_serial:
                self._last_serial_advance_ns = receive_ns
                self._serial_has_advanced = True
            self._last_frame_serial = parsed.frame_serial
            assert self._last_serial_advance_ns is not None
            state = TianjiArmState(
                side=self.side,
                timestamp_ns=self._last_serial_advance_ns,
                receive_timestamp_ns=receive_ns,
                q_rad=np.deg2rad(parsed.q_deg).astype(np.float32),
                dq_rad_s=np.deg2rad(parsed.dq_deg_s).astype(np.float32),
                tau_nm=parsed.tau_nm.astype(np.float32),
                cur_state=parsed.cur_state,
                cmd_state=parsed.cmd_state,
                err_code=parsed.err_code,
                frame_serial=parsed.frame_serial,
                frame_miss_count=parsed.frame_miss_count,
                max_frame_miss_count=parsed.max_frame_miss_count,
                read_sequence=self._read_sequence,
            )
            self._read_sequence += 1
            return state

    def _authorization_rejection(self, supplied_arm_token: str | None) -> str | None:
        if not self.allow_hardware_write:
            return "hardware_write_disabled"
        if not self.capability_probe_confirmed:
            return "capability_probe_unconfirmed"
        if self._expected_arm_token is None:
            return "arm_token_not_configured"
        if supplied_arm_token is None or str(supplied_arm_token) != self._expected_arm_token:
            return "arm_token_mismatch"
        report = self.probe_capabilities()
        if not report.complete:
            return "missing_native_capabilities:" + ",".join(report.missing_methods)
        return None

    def _require_motion_authorization(self, supplied_arm_token: str | None) -> None:
        reason = self._authorization_rejection(supplied_arm_token)
        if reason is not None:
            raise TianjiWriteNotArmed(reason)

    def _reject(
        self,
        *,
        request_id: str,
        requested: np.ndarray,
        decision_ns: int,
        reason: str,
        authorized: np.ndarray | None = None,
    ) -> CommandReceipt:
        return CommandReceipt(
            request_id=request_id,
            component="tianji_arm",
            accepted=False,
            requested_target=requested.astype(np.float32),
            authorized_target=(
                None if authorized is None else authorized.astype(np.float32)
            ),
            decision_timestamp_ns=decision_ns,
            reason=f"side_{self.side.value}:{reason}",
            unit="rad",
            joint_order_hash=self.joint_order_hash,
        )

    def _best_effort_soft_stop(self) -> None:
        if not self._connected:
            return
        try:
            # Historical MarvinSDK.h declares OnEMG_* as void.
            self._call_checked(f"OnEMG_{self.side.value}", allow_none=True)
        except Exception:
            # Preserve the original safety rejection/failure.  A public
            # soft_stop() call remains strict and surfaces its own failure.
            pass

    def _position_transaction(self, target_rad: np.ndarray) -> np.ndarray:
        target_deg = np.rad2deg(target_rad.astype(np.float64))
        native_array = (ctypes.c_double * TIANJI_JOINT_COUNT)(*target_deg.tolist())
        self._call_checked("OnClearSet")
        self._call_checked(f"OnSetJointCmdPos_{self.side.value}", native_array)
        self._call_checked("OnSetSend")
        # This is the exact SI equivalent of the doubles passed to the native
        # setter, not merely the upstream requested array.
        return np.deg2rad(np.asarray(list(native_array), dtype=np.float64)).astype(np.float32)

    def submit_target(
        self,
        *,
        request_id: str,
        q_target_rad: Sequence[float] | np.ndarray,
        target_timestamp_ns: int | None,
        arm_token: str | None,
        wrist_pose_valid: bool = False,
        decision_timestamp_ns: int | None = None,
    ) -> CommandReceipt:
        """Validate and atomically submit one absolute 7-D joint target.

        Safety vetoes return rejected receipts.  Malformed arrays raise
        ``ValueError`` because they are not valid command records at all.
        ``exact_sent_target`` is populated only after ``OnSetSend`` succeeds.
        """

        requested = _vector7(q_target_rad, name="q_target_rad", dtype=np.float32)
        with self._lock:
            observed_now_ns = int(self._clock())
            decision_ns = (
                observed_now_ns
                if decision_timestamp_ns is None
                else int(decision_timestamp_ns)
            )
            if decision_ns < 0:
                raise ValueError("decision_timestamp_ns must be non-negative")
            if decision_ns > observed_now_ns:
                return self._reject(
                    request_id=request_id,
                    requested=requested,
                    decision_ns=decision_ns,
                    reason="decision_timestamp_in_future",
                )
            if not self._connected:
                return self._reject(
                    request_id=request_id,
                    requested=requested,
                    decision_ns=decision_ns,
                    reason="not_connected",
                )
            authorization_reason = self._authorization_rejection(arm_token)
            if authorization_reason is not None:
                return self._reject(
                    request_id=request_id,
                    requested=requested,
                    decision_ns=decision_ns,
                    reason=authorization_reason,
                )
            limits = self.safety_limits
            if limits is None:
                return self._reject(
                    request_id=request_id,
                    requested=requested,
                    decision_ns=decision_ns,
                    reason="joint_limits_unverified",
                )
            if limits.require_wrist_pose and not bool(wrist_pose_valid):
                return self._reject(
                    request_id=request_id,
                    requested=requested,
                    decision_ns=decision_ns,
                    reason="verified_wrist_pose_required",
                )
            if target_timestamp_ns is None:
                return self._reject(
                    request_id=request_id,
                    requested=requested,
                    decision_ns=decision_ns,
                    reason="target_timestamp_required",
                )
            target_ns = int(target_timestamp_ns)
            if target_ns < 0 or target_ns > observed_now_ns:
                return self._reject(
                    request_id=request_id,
                    requested=requested,
                    decision_ns=decision_ns,
                    reason="target_timestamp_invalid_or_future",
                )
            if observed_now_ns - target_ns > limits.max_target_age_ns:
                return self._reject(
                    request_id=request_id,
                    requested=requested,
                    decision_ns=decision_ns,
                    reason="target_stale",
                )
            if (
                self._last_sent_target_timestamp_ns is not None
                and target_ns <= self._last_sent_target_timestamp_ns
            ):
                return self._reject(
                    request_id=request_id,
                    requested=requested,
                    decision_ns=decision_ns,
                    reason="target_timestamp_not_strictly_increasing",
                )
            if np.any(requested < limits.q_min_rad) or np.any(requested > limits.q_max_rad):
                return self._reject(
                    request_id=request_id,
                    requested=requested,
                    decision_ns=decision_ns,
                    reason="joint_limit_violation",
                )

            try:
                state = self.read_state()
            except Exception as exc:
                self._best_effort_soft_stop()
                return self._reject(
                    request_id=request_id,
                    requested=requested,
                    decision_ns=decision_ns,
                    reason=f"feedback_read_failed:{type(exc).__name__}",
                )
            # Re-check freshness at the actual post-read decision boundary;
            # a blocking OnGetBuf call must consume both target and feedback
            # watchdog budgets rather than being hidden by a pre-read clock.
            validation_now_ns = int(self._clock())
            if validation_now_ns < observed_now_ns:
                self._best_effort_soft_stop()
                return self._reject(
                    request_id=request_id,
                    requested=requested,
                    decision_ns=decision_ns,
                    reason="monotonic_clock_regressed",
                )
            if validation_now_ns - target_ns > limits.max_target_age_ns:
                return self._reject(
                    request_id=request_id,
                    requested=requested,
                    decision_ns=decision_ns,
                    reason="target_stale_after_feedback_read",
                )
            if state.err_code != 0:
                self._best_effort_soft_stop()
                return self._reject(
                    request_id=request_id,
                    requested=requested,
                    decision_ns=decision_ns,
                    reason=f"feedback_error:{state.err_code}",
                )
            # Historical MarvinSDK.h explicitly defines 100 as ARM_STATE_ERROR.
            if state.cur_state == 100:
                self._best_effort_soft_stop()
                return self._reject(
                    request_id=request_id,
                    requested=requested,
                    decision_ns=decision_ns,
                    reason="arm_state_error_100",
                )
            if not self._serial_has_advanced:
                return self._reject(
                    request_id=request_id,
                    requested=requested,
                    decision_ns=decision_ns,
                    reason="feedback_serial_not_proven_advancing",
                )
            feedback_age_ns = state.age_ns(validation_now_ns)
            if feedback_age_ns > limits.max_feedback_age_ns:
                self._best_effort_soft_stop()
                return self._reject(
                    request_id=request_id,
                    requested=requested,
                    decision_ns=decision_ns,
                    reason="feedback_stale",
                )
            if state.cur_state != 1:
                return self._reject(
                    request_id=request_id,
                    requested=requested,
                    decision_ns=decision_ns,
                    reason=f"position_state_required:cur_state={state.cur_state}",
                )
            if np.any(np.abs(requested - state.q_rad) > limits.max_delta_rad):
                return self._reject(
                    request_id=request_id,
                    requested=requested,
                    decision_ns=decision_ns,
                    reason="feedback_delta_limit_violation",
                )
            if self._last_sent_target_rad is not None and np.any(
                np.abs(requested - self._last_sent_target_rad) > limits.max_delta_rad
            ):
                return self._reject(
                    request_id=request_id,
                    requested=requested,
                    decision_ns=decision_ns,
                    reason="command_delta_limit_violation",
                )

            try:
                sent_rad = self._position_transaction(requested)
            except Exception as exc:
                self._best_effort_soft_stop()
                return self._reject(
                    request_id=request_id,
                    requested=requested,
                    authorized=requested,
                    decision_ns=decision_ns,
                    reason=f"sdk_transaction_failed:{type(exc).__name__}",
                )
            write_ns = max(int(self._clock()), decision_ns)
            sequence = self._controller_sequence
            self._controller_sequence += 1
            self._last_sent_target_rad = sent_rad.copy()
            self._last_sent_target_timestamp_ns = target_ns
            return CommandReceipt(
                request_id=request_id,
                component="tianji_arm",
                accepted=True,
                requested_target=requested,
                authorized_target=sent_rad,
                exact_sent_target=sent_rad,
                decision_timestamp_ns=decision_ns,
                write_timestamp_ns=write_ns,
                controller_sequence=sequence,
                clipped=False,
                reason=f"side_{self.side.value}:sent",
                unit="rad",
                joint_order_hash=self.joint_order_hash,
            )

    def write_position_rad(
        self,
        q_rad: Sequence[float] | np.ndarray,
        *,
        request_id: str,
        target_timestamp_ns: int | None,
        arm_token: str | None,
        wrist_pose_valid: bool = False,
        decision_timestamp_ns: int | None = None,
    ) -> CommandReceipt:
        """SI-named convenience wrapper for :meth:`submit_target`."""

        return self.submit_target(
            request_id=request_id,
            q_target_rad=q_rad,
            target_timestamp_ns=target_timestamp_ns,
            arm_token=arm_token,
            wrist_pose_valid=wrist_pose_valid,
            decision_timestamp_ns=decision_timestamp_ns,
        )

    def soft_stop(self) -> None:
        """Invoke the side-specific native emergency/soft-stop function.

        Native ``OnEMG_*`` means emergency stop here; it has no relationship
        to surface-electromyography data used elsewhere in this project.
        """

        with self._lock:
            self._require_connected()
            # Historical MarvinSDK.h declares OnEMG_* as void.
            self._call_checked(f"OnEMG_{self.side.value}", allow_none=True)

    def _state_transaction(self, state_code: int) -> None:
        self._call_checked("OnClearSet")
        self._call_checked(
            f"OnSetTargetState_{self.side.value}", ctypes.c_int(int(state_code))
        )
        self._call_checked("OnSetSend")

    def _poll_for_state(
        self,
        desired_state: int,
        *,
        timeout_ns: int,
        poll_interval_s: float,
    ) -> TianjiArmState:
        timeout = _positive_ns(timeout_ns, name="timeout_ns")
        if not math.isfinite(poll_interval_s) or poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be finite and positive")
        deadline = int(self._clock()) + timeout
        max_polls = max(2, int(math.ceil(timeout / (poll_interval_s * 1e9))) + 2)
        last: TianjiArmState | None = None
        for _ in range(max_polls):
            last = self.read_state()
            if last.cur_state == desired_state and last.err_code == 0:
                return last
            if last.cur_state == 100 or last.err_code != 0:
                break
            if int(self._clock()) >= deadline:
                break
            self._sleeper(poll_interval_s)
        state_text = "unknown" if last is None else (
            f"cur_state={last.cur_state},err_code={last.err_code}"
        )
        raise TianjiBackendError(
            f"Tianji side {self.side.value} did not reach state {desired_state}: {state_text}"
        )

    def enable_position(
        self,
        *,
        arm_token: str | None,
        timeout_ns: int = 2_000_000_000,
        poll_interval_s: float = 0.01,
    ) -> TianjiArmState:
        """Request POSITION state 1 and return only after feedback confirms it."""

        with self._lock:
            self._require_connected()
            self._require_motion_authorization(arm_token)
            state = self.read_state()
            if state.err_code != 0 or state.cur_state == 100:
                self._best_effort_soft_stop()
                raise TianjiFeedbackError(
                    f"cannot enable position from cur_state={state.cur_state}, "
                    f"err_code={state.err_code}"
                )
            if not self._serial_has_advanced:
                raise TianjiFeedbackError("feedback serial has not been proven advancing")
            self._state_transaction(1)
            return self._poll_for_state(
                1, timeout_ns=timeout_ns, poll_interval_s=poll_interval_s
            )

    def disable_and_verify(
        self,
        *,
        timeout_ns: int = 2_000_000_000,
        poll_interval_s: float = 0.01,
    ) -> TianjiArmState:
        """Request servo-off state 0 and require feedback confirmation.

        This fail-safe operation does not require the motion arm token.  If
        confirmation is unavailable, callers must use the physical E-stop and
        inspect the robot; the TCP session is intentionally left open.
        """

        with self._lock:
            self._require_connected()
            try:
                initial = self.read_state()
                if initial.cur_state == 0 and initial.err_code == 0:
                    return initial
                self._best_effort_soft_stop()
                self._state_transaction(0)
                return self._poll_for_state(
                    0, timeout_ns=timeout_ns, poll_interval_s=poll_interval_s
                )
            except TianjiPhysicalInterventionRequired:
                raise
            except Exception as exc:
                raise TianjiPhysicalInterventionRequired(
                    "Tianji servo-off state 0 was not confirmed; physical intervention "
                    "and the hardware E-stop are required before disconnecting"
                ) from exc

    def close(
        self,
        *,
        timeout_ns: int = 2_000_000_000,
        poll_interval_s: float = 0.01,
    ) -> None:
        """Confirm state 0, then and only then release the SDK TCP session."""

        with self._lock:
            if not self._connected:
                return
            self.disable_and_verify(
                timeout_ns=timeout_ns, poll_interval_s=poll_interval_s
            )
            self._call_checked("OnRelease")
            self._connected = False


__all__ = [
    "TIANJI_JOINT_COUNT",
    "TIANJI_JOINT_ORDER_HASH",
    "TIANJI_SDK_JOINT_ORDER",
    "tianji_joint_order_hash",
    "TianjiArmState",
    "TianjiBackendError",
    "TianjiCapabilityReport",
    "TianjiFeedbackError",
    "TianjiMarvinBackend",
    "TianjiPhysicalInterventionRequired",
    "TianjiSafetyLimits",
    "TianjiSide",
    "TianjiWriteNotArmed",
]
