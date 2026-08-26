"""Glove-to-Revo3 21-D retargeting with one controller-write authority.

Retargeting outputs are only *requested* hand targets.  They become eligible
imitation labels solely after ``TeleopRevoWriter`` has awaited the existing
``RevoCommandPipeline`` and returned an accepted exact-sent receipt.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Mapping, Protocol

import numpy as np

from revo3_v1.revo import MockRevoBackend, RevoState, SafetyContext, assert_joint_vector

from revo3_teleop.backends.revo import TeleopRevoWriter
from revo3_teleop.backends.tianji_loader import (
    CallableModuleProvenance,
    resolve_hashed_callable,
)
from revo3_teleop.contracts import CommandReceipt, NativeSample


class HandRetargetingBlocked(RuntimeError):
    pass


def _revision(value: object, *, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return result


@dataclass(frozen=True)
class RevoHandRetargetResult:
    requested_q_rad: np.ndarray
    source_id: str
    source_sequence: int
    source_timestamp_ns: int
    input_kind: str
    calibration_revision: str
    model_revision: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "requested_q_rad",
            assert_joint_vector(self.requested_q_rad, name="requested_q_rad"),
        )
        for name in (
            "source_id",
            "input_kind",
            "calibration_revision",
            "model_revision",
        ):
            object.__setattr__(self, name, _revision(getattr(self, name), name=name))
        if self.source_sequence < 0 or self.source_timestamp_ns < 0:
            raise ValueError("source sequence/timestamp must be non-negative")


class RevoHandRetargeter(Protocol):
    def retarget(self, sample: NativeSample) -> RevoHandRetargetResult: ...


@dataclass(frozen=True)
class BrainCoSixFlexCalibration:
    """Explicit, project-calibrated six-flex to canonical Revo 21-D mapping."""

    flex_min: np.ndarray
    flex_max: np.ndarray
    normalized_to_q_matrix: np.ndarray
    q_bias_rad: np.ndarray
    q_min_rad: np.ndarray
    q_max_rad: np.ndarray
    calibration_revision: str
    model_revision: str

    def __post_init__(self) -> None:
        vectors: dict[str, np.ndarray] = {}
        for name, size in (
            ("flex_min", 6),
            ("flex_max", 6),
            ("q_bias_rad", 21),
            ("q_min_rad", 21),
            ("q_max_rad", 21),
        ):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (size,) or not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite with shape ({size},)")
            vectors[name] = value.copy()
        matrix = np.asarray(self.normalized_to_q_matrix, dtype=np.float64)
        if matrix.shape != (21, 6) or not np.isfinite(matrix).all():
            raise ValueError("normalized_to_q_matrix must be finite with shape (21, 6)")
        if np.any(vectors["flex_min"] >= vectors["flex_max"]):
            raise ValueError("every flex_min must be less than flex_max")
        if np.any(vectors["q_min_rad"] >= vectors["q_max_rad"]):
            raise ValueError("every q_min_rad must be less than q_max_rad")
        for name, value in vectors.items():
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        matrix.setflags(write=False)
        object.__setattr__(self, "normalized_to_q_matrix", matrix)
        object.__setattr__(
            self,
            "calibration_revision",
            _revision(self.calibration_revision, name="calibration_revision"),
        )
        object.__setattr__(
            self,
            "model_revision",
            _revision(self.model_revision, name="model_revision"),
        )


class BrainCoSixFlexRetargeter:
    """No calibration means no mapping; six channels are never tiled to 21-D."""

    def __init__(self, calibration: BrainCoSixFlexCalibration | None) -> None:
        if calibration is None:
            raise HandRetargetingBlocked(
                "explicit BrainCo six-flex to Revo3 21-D calibration is required"
            )
        self.calibration = calibration

    def retarget(self, sample: NativeSample) -> RevoHandRetargetResult:
        header = sample.header
        if not header.valid or header.dropped_since_previous:
            raise HandRetargetingBlocked("BrainCo flex sample is invalid or follows a gap")
        if "flex_raw" not in sample.payload:
            raise HandRetargetingBlocked("BrainCo retargeter requires flex_raw[6]")
        flex = np.asarray(sample.payload["flex_raw"], dtype=np.float64)
        if flex.shape != (6,) or not np.isfinite(flex).all():
            raise HandRetargetingBlocked("flex_raw must be finite with shape (6,)")
        calibration = self.calibration
        if np.any(flex < calibration.flex_min) or np.any(flex > calibration.flex_max):
            raise HandRetargetingBlocked("flex sample is outside the calibrated range")
        normalized = (flex - calibration.flex_min) / (
            calibration.flex_max - calibration.flex_min
        )
        requested = (
            calibration.normalized_to_q_matrix @ normalized
            + calibration.q_bias_rad
        )
        if np.any(requested < calibration.q_min_rad) or np.any(
            requested > calibration.q_max_rad
        ):
            raise HandRetargetingBlocked("mapped hand target violates calibration bounds")
        return RevoHandRetargetResult(
            requested_q_rad=requested.astype(np.float32),
            source_id=header.source_id,
            source_sequence=header.sequence,
            source_timestamp_ns=header.capture_timestamp_ns,
            input_kind="brainco_six_flex",
            calibration_revision=calibration.calibration_revision,
            model_revision=calibration.model_revision,
        )


class PluginRetargetModel(Protocol):
    calibration_revision: str
    model_revision: str
    input_kind: str

    def retarget(self, sample: NativeSample) -> np.ndarray | Mapping[str, Any]: ...


class PluginRevoHandRetargeter:
    """Validated wrapper for MANUS/other official retargeting plugins."""

    def __init__(
        self,
        plugin: PluginRetargetModel,
        *,
        module_provenance: CallableModuleProvenance | None = None,
    ) -> None:
        if not callable(getattr(plugin, "retarget", None)):
            raise TypeError("retarget plugin must provide retarget(sample)")
        self.plugin = plugin
        self.module_provenance = module_provenance
        self.module_hash_verified = bool(
            module_provenance is not None and module_provenance.hash_verified
        )
        self.calibration_revision = _revision(
            getattr(plugin, "calibration_revision", ""),
            name="plugin.calibration_revision",
        )
        self.model_revision = _revision(
            getattr(plugin, "model_revision", ""), name="plugin.model_revision"
        )
        self.input_kind = _revision(
            getattr(plugin, "input_kind", ""), name="plugin.input_kind"
        )

    def retarget(self, sample: NativeSample) -> RevoHandRetargetResult:
        if not sample.header.valid or sample.header.dropped_since_previous:
            raise HandRetargetingBlocked("glove sample is invalid or follows a gap")
        output = self.plugin.retarget(sample)
        if isinstance(output, Mapping):
            if "requested_q_rad" not in output:
                raise HandRetargetingBlocked(
                    "plugin mapping output lacks requested_q_rad"
                )
            output = output["requested_q_rad"]
        try:
            q_rad = assert_joint_vector(output, name="plugin requested_q_rad")
        except (TypeError, ValueError) as exc:
            raise HandRetargetingBlocked(str(exc)) from exc
        header = sample.header
        return RevoHandRetargetResult(
            requested_q_rad=q_rad,
            source_id=header.source_id,
            source_sequence=header.sequence,
            source_timestamp_ns=header.capture_timestamp_ns,
            input_kind=self.input_kind,
            calibration_revision=self.calibration_revision,
            model_revision=self.model_revision,
        )


def load_revo_hand_retargeter(
    factory_target: str,
    *,
    expected_module_sha256: str,
    factory_kwargs: Mapping[str, Any] | None = None,
) -> PluginRevoHandRetargeter:
    """Load a retarget factory whose executable module hash is explicit."""

    factory, provenance = resolve_hashed_callable(
        factory_target,
        expected_module_sha256=expected_module_sha256,
        name="Revo hand retarget factory",
    )
    if not provenance.hash_verified:
        raise HandRetargetingBlocked("Revo hand retarget factory hash is unverified")
    plugin = factory(**dict(factory_kwargs or {}))
    return PluginRevoHandRetargeter(plugin, module_provenance=provenance)


@dataclass(frozen=True)
class RevoHandExecution:
    retarget: RevoHandRetargetResult
    receipt: CommandReceipt


class RevoGloveTeleopController:
    """The only provided glove-retarget-to-Revo execution path."""

    def __init__(
        self,
        *,
        retargeter: RevoHandRetargeter,
        writer: TeleopRevoWriter,
    ) -> None:
        self.retargeter = retargeter
        self.writer = writer
        if (
            isinstance(retargeter, PluginRevoHandRetargeter)
            and not retargeter.module_hash_verified
            and not isinstance(getattr(writer.pipeline, "backend", None), MockRevoBackend)
        ):
            raise HandRetargetingBlocked(
                "external hand-retarget plugin hash must be verified before a non-mock write"
            )

    async def execute(
        self,
        sample: NativeSample,
        *,
        request_id: str,
        task_id: str,
        task_version: int,
        safety_context: SafetyContext,
        emg_requests_close: bool,
        state: RevoState | None = None,
        decision_timestamp_ns: int | None = None,
    ) -> RevoHandExecution:
        result = self.retargeter.retarget(sample)
        decision_ns = (
            time.monotonic_ns()
            if decision_timestamp_ns is None
            else int(decision_timestamp_ns)
        )
        receipt = await self.writer.submit_target(
            request_id=request_id,
            nominal_q_rad=result.requested_q_rad,
            task_id=task_id,
            task_version=task_version,
            safety_context=safety_context,
            emg_requests_close=emg_requests_close,
            source_chunk_id=(
                f"glove:{result.source_id}:{result.source_sequence}:"
                f"{result.calibration_revision}:{result.model_revision}"
            ),
            decision_timestamp_ns=decision_ns,
            state=state,
        )
        return RevoHandExecution(retarget=result, receipt=receipt)


__all__ = [
    "BrainCoSixFlexCalibration",
    "BrainCoSixFlexRetargeter",
    "HandRetargetingBlocked",
    "PluginRetargetModel",
    "PluginRevoHandRetargeter",
    "RevoGloveTeleopController",
    "RevoHandExecution",
    "RevoHandRetargetResult",
    "RevoHandRetargeter",
    "load_revo_hand_retargeter",
]
