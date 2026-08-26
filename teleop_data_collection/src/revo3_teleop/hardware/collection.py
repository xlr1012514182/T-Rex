"""Fail-closed orchestration for one real-hardware collection episode.

The checked-in CLI/configuration exercise only the static readiness audit.
No SDK is imported and no device is opened until all identity, calibration,
limit, model and plugin evidence is complete *and* the operator supplies the
two command-line execution confirmations.

This module deliberately does not implement a glove-to-Revo or wrist-to-
Tianji mapping.  A site-specific assembly factory must be loaded through the
existing SHA-256 verified callable boundary and return already composed,
safety-reviewed dependencies.  The common runtime owns recording, native-rate
source draining, 30 Hz causal anchors, terminal holds/stops, and quarantine.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import hashlib
import inspect
import json
from pathlib import Path
import threading
import time
from typing import Any, Awaitable, Callable, Mapping, Protocol, Sequence

import numpy as np

from revo3_v1.data.episode import SUPPORTED_TASKS
from revo3_v1.revo import JOINT_COUNT, JOINT_ORDER_HASH

from revo3_teleop.backends.tianji_loader import resolve_hashed_callable
from revo3_teleop.contracts import CommandReceipt, NativeSample
from revo3_teleop.hardware.tianji import (
    HardwareConfig,
    assess_hardware_config,
)
from revo3_teleop.recording import (
    CollectionSession,
    CollectionSessionFault,
    EpisodeRecorder,
    Revo3ExportConfig,
    SessionState,
    export_revo3_episode,
)


HARDWARE_COLLECTION_SCHEMA_VERSION = "revo3-hardware-collection-v1"
_PLACEHOLDERS = ("REPLACE", "UNVERIFIED", "TODO", "CHANGEME")


def _placeholder(value: object) -> bool:
    if value is None:
        return True
    if not isinstance(value, str):
        return False
    text = value.strip().upper()
    return not text or any(marker in text for marker in _PLACEHOLDERS)


def _sha256(value: object) -> bool:
    if _placeholder(value):
        return False
    text = str(value).strip().lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _positive_number(value: object, *, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _finite_vector(value: object, *, name: str, size: int) -> np.ndarray | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    if len(value) != size or any(item is None or _placeholder(item) for item in value):
        return None
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        return None
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_path(base: Path, value: object) -> Path | None:
    if _placeholder(value):
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


@dataclass(frozen=True)
class HardwareCollectionConfig:
    """Parsed configuration; loading it is side-effect-free with respect to hardware."""

    path: Path
    raw: Mapping[str, Any]

    @classmethod
    def from_json(cls, path: str | Path) -> "HardwareCollectionConfig":
        source = Path(path).resolve()
        value = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError("hardware collection config root must be a JSON object")
        if value.get("schema_version") != HARDWARE_COLLECTION_SCHEMA_VERSION:
            raise ValueError(
                "hardware collection schema_version must be "
                f"{HARDWARE_COLLECTION_SCHEMA_VERSION!r}"
            )
        if value.get("mode") != "hardware_collection":
            raise ValueError("hardware collection mode must be 'hardware_collection'")
        for section in (
            "episode",
            "outputs",
            "runtime",
            "revo",
            "camera",
            "emg",
            "tactile",
            "glove",
            "tianji",
            "assembly_factory",
        ):
            _mapping(value.get(section), name=section)
        return cls(path=source, raw=dict(value))

    def resolve_path(self, value: object) -> Path | None:
        return _relative_path(self.path.parent, value)


@dataclass(frozen=True)
class HardwareCollectionReadiness:
    config_path: str
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    evidence: tuple[str, ...]

    @property
    def execute_ready(self) -> bool:
        return not self.blockers

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "revo3-hardware-collection-readiness-v1",
            "config_path": self.config_path,
            "execute_ready": self.execute_ready,
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "evidence": list(self.evidence),
            "default_action": "audit_only_no_import_no_connect_no_write",
            "action_label_contract": "accepted_revo_hand_exact_sent_target_only",
            "alignment_contract": "30hz_latest_not_after_no_future_samples",
            "unsupported_claims": [
                "real hardware connected or moved",
                "glove-to-Revo or wrist-to-Tianji mapping verified by this audit",
                "task success, clinical benefit, or human-use safety",
            ],
        }


def _check_file_hash(
    config: HardwareCollectionConfig,
    *,
    path_value: object,
    hash_value: object,
    prefix: str,
    blockers: list[str],
    evidence: list[str],
) -> None:
    path = config.resolve_path(path_value)
    if path is None or not _sha256(hash_value):
        blockers.append(f"{prefix}_path_and_sha256_required")
        return
    if not path.is_file():
        blockers.append(f"{prefix}_file_missing")
        return
    observed = _file_sha256(path)
    if observed != str(hash_value).strip().lower():
        blockers.append(f"{prefix}_sha256_mismatch")
        return
    evidence.append(f"{prefix}_sha256_verified")


def assess_hardware_collection_config(
    config: HardwareCollectionConfig,
) -> HardwareCollectionReadiness:
    """Perform all static gates before an assembly plugin can be imported."""

    raw = config.raw
    blockers: list[str] = []
    warnings: list[str] = [
        "execution_requires_execute_flag_plus_connect_and_write_confirmations",
        "EMG_export_requires_separate_human_reviewed_label_intervals",
    ]
    evidence: list[str] = ["readiness_audit_has_no_sdk_import_connect_or_write"]

    if raw.get("allow_hardware_connect") is not True:
        blockers.append("allow_hardware_connect_is_false")
    if raw.get("allow_hardware_write") is not True:
        blockers.append("allow_hardware_write_is_false")

    episode = _mapping(raw["episode"], name="episode")
    if _placeholder(episode.get("episode_id")):
        blockers.append("episode_id_required")
    task = str(episode.get("task", "")).strip()
    if task not in SUPPORTED_TASKS:
        blockers.append("supported_task_required")
    if _placeholder(episode.get("instruction")):
        blockers.append("planner_instruction_required")
    for key in (
        "task_id",
        "object_id",
        "object_instance",
        "operator",
        "collection_day",
        "grasp_primitive",
        "instruction_source",
    ):
        if _placeholder(episode.get(key)):
            blockers.append(f"episode_{key}_required")
    try:
        if int(episode.get("task_version")) <= 0:
            raise ValueError
    except (TypeError, ValueError):
        blockers.append("positive_episode_task_version_required")
    if str(episode.get("instruction_source", "")) == "frozen_planner":
        if _placeholder(episode.get("planner_revision")):
            blockers.append("episode_planner_revision_required")
        if not _sha256(episode.get("planner_output_sha256")):
            blockers.append("episode_planner_output_sha256_required")
    try:
        duration_s = _positive_number(episode.get("duration_s"), name="episode.duration_s")
        warmup_s = _positive_number(
            episode.get("anchor_start_delay_s"), name="episode.anchor_start_delay_s"
        )
        if duration_s <= warmup_s + 2.0 / 30.0:
            blockers.append("duration_must_allow_at_least_two_30hz_anchors")
    except (TypeError, ValueError):
        blockers.append("valid_episode_duration_and_anchor_delay_required")

    outputs = _mapping(raw["outputs"], name="outputs")
    output_paths = {
        name: config.resolve_path(outputs.get(name))
        for name in ("master_root", "vla_root", "emg_review_root")
    }
    if any(value is None for value in output_paths.values()):
        blockers.append("three_distinct_output_roots_required")
    else:
        resolved = [path for path in output_paths.values() if path is not None]
        distinct = len({str(path) for path in resolved}) == 3
        non_nested = not any(
            left in right.parents or right in left.parents
            for index, left in enumerate(resolved)
            for right in resolved[index + 1 :]
        )
        if not distinct:
            blockers.append("master_vla_emg_output_roots_must_be_distinct")
        if not non_nested:
            blockers.append("master_vla_emg_output_roots_must_not_be_nested")
        if distinct and non_nested:
            evidence.append("vla_and_emg_derived_outputs_are_physically_separate")

    runtime = _mapping(raw["runtime"], name="runtime")
    required_streams = runtime.get("required_source_timeouts_ms")
    anchor_streams = runtime.get("anchor_streams")
    parsed_timeouts: dict[str, float] = {}
    if not isinstance(required_streams, Mapping) or not required_streams:
        blockers.append("required_source_timeouts_required")
    else:
        try:
            parsed_timeouts = {
                str(name): _positive_number(value, name=f"timeout:{name}")
                for name, value in required_streams.items()
            }
            if isinstance(anchor_streams, list):
                missing = set(str(value) for value in anchor_streams) - set(parsed_timeouts)
                if missing:
                    blockers.append("every_anchor_stream_requires_timeout")
        except (TypeError, ValueError):
            blockers.append("valid_required_source_timeouts_required")
    anchor_names = (
        [] if not isinstance(anchor_streams, list) else [str(value) for value in anchor_streams]
    )
    if len(anchor_names) < 3:
        blockers.append("camera_state_tactile_anchor_streams_required")
    elif len(set(anchor_names)) != len(anchor_names):
        blockers.append("anchor_streams_must_be_unique")
    if "emg" in set(anchor_names):
        blockers.append("emg_must_not_be_a_vla_anchor_stream")
    core_streams = {
        "camera": str(runtime.get("camera_stream", "")).strip(),
        "state": str(runtime.get("state_stream", "")).strip(),
        "tactile": str(runtime.get("tactile_stream", "")).strip(),
    }
    if any(not name for name in core_streams.values()):
        blockers.append("explicit_camera_state_tactile_runtime_streams_required")
    elif len(set(core_streams.values())) != 3:
        blockers.append("camera_state_tactile_runtime_streams_must_be_unique")
    else:
        if not set(core_streams.values()).issubset(anchor_names):
            blockers.append("camera_state_tactile_streams_must_be_anchor_streams")
        if not set(core_streams.values()).issubset(parsed_timeouts):
            blockers.append("camera_state_tactile_streams_require_timeouts")
    if "emg" not in parsed_timeouts:
        blockers.append("emg_required_source_timeout_required")
    vla_streams = {
        "camera": str(runtime.get("vla_camera_stream", "")).strip(),
        "state": str(runtime.get("vla_state_stream", "")).strip(),
        "tactile": str(runtime.get("vla_tactile_stream", "")).strip(),
    }
    if outputs.get("export_vla_after_commit", True) is True:
        if any(not name for name in vla_streams.values()):
            blockers.append("explicit_vla_camera_state_tactile_streams_required")
        elif len(set(vla_streams.values())) != 3:
            blockers.append("vla_camera_state_tactile_streams_must_be_unique")
        else:
            if vla_streams != core_streams:
                blockers.append("vla_streams_must_match_runtime_anchor_roles")
            if not set(vla_streams.values()).issubset(anchor_names):
                blockers.append("all_vla_streams_must_be_anchor_streams")
            if not set(vla_streams.values()).issubset(parsed_timeouts):
                blockers.append("all_vla_streams_require_timeouts")
    runtime_budgets: dict[str, float] = {}
    for key in (
        "control_step_timeout_ms",
        "max_command_latency_ms",
        "shutdown_timeout_ms",
        "safety_watchdog_budget_ms",
    ):
        try:
            runtime_budgets[key] = _positive_number(
                runtime.get(key), name=f"runtime.{key}"
            )
        except (TypeError, ValueError):
            blockers.append(f"valid_{key}_required")
    if (
        "control_step_timeout_ms" in runtime_budgets
        and "safety_watchdog_budget_ms" in runtime_budgets
        and runtime_budgets["control_step_timeout_ms"]
        > runtime_budgets["safety_watchdog_budget_ms"]
    ):
        blockers.append("control_step_timeout_exceeds_safety_watchdog_budget")
    if (
        "control_step_timeout_ms" in runtime_budgets
        and runtime_budgets["control_step_timeout_ms"] > 1000.0 / 30.0
    ):
        blockers.append("control_step_timeout_exceeds_30hz_anchor_period")
    if (
        "max_command_latency_ms" in runtime_budgets
        and runtime_budgets["max_command_latency_ms"] > 1000.0 / 30.0
    ):
        blockers.append("max_command_latency_exceeds_30hz_anchor_period")

    revo = _mapping(raw["revo"], name="revo")
    if str(revo.get("state_stream", "")).strip() != core_streams["state"]:
        blockers.append("revo_state_stream_must_match_runtime_state_stream")
    if _placeholder(revo.get("expected_serial")):
        blockers.append("revo_expected_serial_required")
    if not _sha256(revo.get("expected_probe_fingerprint")):
        blockers.append("revo_probe_fingerprint_required")
    if revo.get("joint_order_hash") != JOINT_ORDER_HASH:
        blockers.append("revo_canonical_joint_order_hash_required")
    lower = _finite_vector(revo.get("q_min_rad"), name="revo.q_min_rad", size=JOINT_COUNT)
    upper = _finite_vector(revo.get("q_max_rad"), name="revo.q_max_rad", size=JOINT_COUNT)
    delta = _finite_vector(
        revo.get("max_delta_rad"), name="revo.max_delta_rad", size=JOINT_COUNT
    )
    if lower is None or upper is None or delta is None:
        blockers.append("bench_verified_revo_21d_limits_required")
    elif np.any(lower >= upper) or np.any(delta <= 0):
        blockers.append("valid_revo_21d_limits_required")
    for flag in (
        "state_units_bench_verified",
        "u21vt_identity_verified",
        "tactile_zero_and_saturation_verified",
        "physical_estop_verified",
        "hold_path_verified",
        "limits_verified",
    ):
        if revo.get(flag) is not True:
            blockers.append(f"revo_{flag}_required")

    camera = _mapping(raw["camera"], name="camera")
    if str(camera.get("rectified_stream", "")).strip() != core_streams["camera"]:
        blockers.append("camera_rectified_stream_must_match_runtime_camera_stream")
    if not _sha256(camera.get("expected_probe_fingerprint")):
        blockers.append("camera_probe_fingerprint_required")
    if str(camera.get("camera_profile_id", "")).strip() != "revo3_full_center_v1":
        blockers.append("revo3_full_center_camera_profile_required")
    _check_file_hash(
        config,
        path_value=camera.get("calibration_file"),
        hash_value=camera.get("calibration_sha256"),
        prefix="camera_calibration",
        blockers=blockers,
        evidence=evidence,
    )

    emg = _mapping(raw["emg"], name="emg")
    if str(emg.get("stream", "")).strip() != "emg":
        blockers.append("brainco_emg_stream_must_be_emg")
    if _placeholder(emg.get("expected_serial")):
        blockers.append("emg_expected_serial_required")
    if not _sha256(emg.get("expected_discovery_fingerprint")):
        blockers.append("emg_discovery_fingerprint_required")

    tactile = _mapping(raw["tactile"], name="tactile")
    tactile_mode = str(tactile.get("mode", "")).strip().lower()
    tactile_stream = str(tactile.get("stream", "")).strip()
    if tactile_stream != core_streams["tactile"]:
        blockers.append("tactile_stream_must_match_runtime_tactile_stream")
    if tactile_mode not in {"visiontouch_force6d", "u21vt_pressure"}:
        blockers.append("supported_tactile_mode_required")
    elif tactile_mode == "visiontouch_force6d":
        if tactile_stream != "tactile":
            blockers.append("visiontouch_force6d_stream_must_be_tactile")
        capture_profile = str(tactile.get("capture_profile", "")).strip().lower()
        vla_profile = str(tactile.get("vla_tactile_profile", "")).strip()
        profile_pairs = {
            "force6d_diff": "profile_a_force6d_diff",
            "diff_only": "profile_b_diff_only",
            "force6d": "ablation_force6d_only",
        }
        if capture_profile not in profile_pairs:
            blockers.append("explicit_visiontouch_capture_profile_required")
        if vla_profile not in set(profile_pairs.values()):
            blockers.append("supported_vla_tactile_profile_required")
        elif capture_profile in profile_pairs and profile_pairs[capture_profile] != vla_profile:
            blockers.append("visiontouch_capture_and_vla_tactile_profile_mismatch")
        try:
            _positive_number(
                tactile.get("max_inter_finger_skew_ns"),
                name="tactile.max_inter_finger_skew_ns",
            )
        except (TypeError, ValueError):
            blockers.append("approved_visiontouch_max_inter_finger_skew_ns_required")
        for key in ("checkpoint_family_id", "normalization_family_id"):
            if _placeholder(tactile.get(key)):
                blockers.append(f"tactile_{key}_required")
        if not _sha256(tactile.get("capability_manifest_sha256")):
            blockers.append("tactile_capability_manifest_sha256_required")
        serials = tactile.get("finger_serials")
        hashes = tactile.get("expected_model_sha256")
        model_root = config.resolve_path(tactile.get("force_model_dir"))
        finger_order = ("thumb", "index", "middle", "ring", "pinky")
        needs_force = capture_profile in {"force6d", "force6d_diff"}
        if not isinstance(serials, Mapping):
            blockers.append("five_visiontouch_serials_required")
        else:
            observed_serials: list[str] = []
            for finger in finger_order:
                serial = serials.get(finger)
                if _placeholder(serial):
                    blockers.append(f"visiontouch_{finger}_serial_required")
                    continue
                serial_text = str(serial).strip()
                observed_serials.append(serial_text)
            if len(observed_serials) == 5 and len(set(observed_serials)) != 5:
                blockers.append("visiontouch_finger_serials_must_be_unique")
            if needs_force:
                if not isinstance(hashes, Mapping):
                    blockers.append("five_visiontouch_model_hashes_required")
                elif model_root is None or not model_root.is_dir():
                    blockers.append("visiontouch_force_model_dir_required")
                else:
                    force_errors_before = len(blockers)
                    for finger in finger_order:
                        serial = serials.get(finger)
                        expected = hashes.get(finger)
                        if _placeholder(serial) or not _sha256(expected):
                            blockers.append(
                                f"visiontouch_{finger}_serial_and_model_hash_required"
                            )
                            continue
                        serial_text = str(serial).strip()
                        model = model_root / serial_text / f"{serial_text}.onnx.enc"
                        if not model.is_file():
                            blockers.append(f"visiontouch_{finger}_force_model_missing")
                        elif _file_sha256(model) != str(expected).strip().lower():
                            blockers.append(f"visiontouch_{finger}_force_model_hash_mismatch")
                    if len(blockers) == force_errors_before:
                        evidence.append("all_visiontouch_force_model_hashes_verified")
            elif hashes not in (None, {}):
                blockers.append("diff_only_must_not_claim_unused_force_model_hashes")
            elif tactile.get("force_model_dir") not in (None, ""):
                blockers.append("diff_only_must_not_claim_unused_force_model_dir")
    else:
        if tactile_stream != "tactile_pressure":
            blockers.append("u21vt_pressure_stream_must_be_tactile_pressure")
        if _placeholder(tactile.get("pressure_projection_revision")):
            blockers.append("bench_verified_pressure_projection_revision_required")
        if tactile.get("pressure_projection_bench_verified") is not True:
            blockers.append("pressure_projection_bench_verification_required")
        if outputs.get("export_vla_after_commit", True) is True:
            blockers.append("u21vt_pressure_requires_export_vla_after_commit_false")
        warnings.append("u21vt_pressure_is_not_force6d_and_cannot_use_trex_tactile_exporter")

    glove = _mapping(raw["glove"], name="glove")
    if glove.get("enabled") is True:
        # bc-edu-sdk/libedu exposes module-global callbacks.  Registering the
        # glove and armband clients in one process would let one overwrite the
        # other's callback.  The V1 collector therefore accepts glove data
        # only through a separately hosted, timestamp-preserving IPC source.
        if str(glove.get("acquisition_mode", "")).strip().lower() != (
            "external_timestamped_ipc"
        ):
            blockers.append("brainco_glove_and_emg_cannot_share_libedu_process")
        if _placeholder(glove.get("ipc_clock_domain")):
            blockers.append("glove_ipc_clock_domain_required")
        ipc_streams = glove.get("ipc_streams")
        if (
            not isinstance(ipc_streams, list)
            or not ipc_streams
            or any(_placeholder(value) for value in ipc_streams)
            or len({str(value) for value in ipc_streams}) != len(ipc_streams)
        ):
            blockers.append("unique_glove_ipc_streams_required")
        elif glove.get("required_for_episode") is True:
            if not {str(value) for value in ipc_streams}.issubset(parsed_timeouts):
                blockers.append("required_glove_ipc_streams_need_timeouts")
        elif glove.get("required_for_episode") is False:
            warnings.append("glove_ipc_streams_are_diagnostic_not_episode_required")
        else:
            blockers.append("glove_required_for_episode_boolean_required")
        for key in ("source_factory", "retarget_factory"):
            if _placeholder(glove.get(key)):
                blockers.append(f"verified_glove_{key}_required")
        for key in ("expected_source_module_sha256", "expected_retarget_module_sha256"):
            if not _sha256(glove.get(key)):
                blockers.append(f"glove_{key}_required")
        if _placeholder(glove.get("calibration_revision")):
            blockers.append("glove_to_revo_calibration_revision_required")
        _check_file_hash(
            config,
            path_value=glove.get("calibration_file"),
            hash_value=glove.get("calibration_sha256"),
            prefix="glove_to_revo_calibration",
            blockers=blockers,
            evidence=evidence,
        )
        warnings.append(
            "glove_source_must_run_in_a_process_separate_from_brainco_edu_emg"
        )

    tianji = _mapping(raw["tianji"], name="tianji")
    if tianji.get("enabled") is True:
        tianji_path = config.resolve_path(tianji.get("config_path"))
        if tianji_path is None or not tianji_path.is_file():
            blockers.append("tianji_hardware_config_required")
        else:
            try:
                tianji_report = assess_hardware_config(HardwareConfig.from_json(tianji_path))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                blockers.append("valid_tianji_hardware_config_required")
            else:
                blockers.extend(
                    f"tianji:{item}" for item in tianji_report.arm_write_blockers
                )
                if tianji_report.arm_write_ready:
                    evidence.append("tianji_static_write_gates_passed")
    else:
        warnings.append("tianji_disabled_hand_only_collection")

    factory = _mapping(raw["assembly_factory"], name="assembly_factory")
    if _placeholder(factory.get("factory")):
        blockers.append("hardware_assembly_factory_required")
    if not _sha256(factory.get("expected_module_sha256")):
        blockers.append("hardware_assembly_factory_module_hash_required")
    factory_kwargs = factory.get("kwargs", {})
    if not isinstance(factory_kwargs, Mapping) or any(
        not isinstance(key, str) or not key.strip() or key == "config"
        for key in factory_kwargs
    ):
        blockers.append("valid_nonshadowing_assembly_factory_kwargs_required")
    warnings.append(
        "assembly_factory_must_construct_disconnected_dependencies_and_pass_assert_disconnected"
    )

    return HardwareCollectionReadiness(
        config_path=str(config.path),
        blockers=tuple(dict.fromkeys(blockers)),
        warnings=tuple(dict.fromkeys(warnings)),
        evidence=tuple(dict.fromkeys(evidence)),
    )


class SensorRunner(Protocol):
    async def run(self, *, duration_s: float | None = None) -> None: ...

    def request_stop(self) -> None: ...


class CollectionControlDriver(Protocol):
    async def step(self, anchor_timestamp_ns: int) -> "CollectionControlCycle": ...


class QueueNativeSource(Protocol):
    def start(self) -> None: ...

    def drain(self) -> Sequence[object]: ...

    def stop(self) -> None: ...


@dataclass(frozen=True)
class CollectionControlCycle:
    hand_receipt: CommandReceipt
    auxiliary_receipts: tuple[CommandReceipt, ...] = ()
    phase: str = ""
    policy_loss_eligible: bool | None = None

    def __post_init__(self) -> None:
        receipt = self.hand_receipt
        if receipt.component != "revo_hand":
            raise ValueError("control cycle hand_receipt must be a revo_hand receipt")
        # Rejected receipts are allowed through this envelope so the session
        # can persist the veto as diagnostic evidence before quarantining the
        # episode.  Only an accepted receipt can reach record_anchor().
        if receipt.accepted:
            if receipt.exact_sent_target is None:
                raise ValueError("accepted Revo receipt lacks exact_sent_target")
            if receipt.exact_sent_target.shape != (JOINT_COUNT,):
                raise ValueError(
                    f"exact-sent Revo receipt must have shape ({JOINT_COUNT},)"
                )
            if receipt.unit != "rad" or receipt.joint_order_hash != JOINT_ORDER_HASH:
                raise ValueError("exact-sent Revo receipt unit/joint order mismatch")
        if any(item.component != "tianji_arm" for item in self.auxiliary_receipts):
            raise ValueError(
                "auxiliary_receipts may contain only Tianji arm receipts; "
                "the cycle has exactly one Revo hand authority"
            )
        if self.phase not in {"context", "precontact", "contact", "hold", "release"}:
            raise ValueError("control cycle requires one explicit policy phase")
        if not isinstance(self.policy_loss_eligible, bool):
            raise ValueError("control cycle requires an explicit policy_loss_eligible boolean")


def _native_sample(value: object) -> NativeSample:
    if isinstance(value, NativeSample):
        return value
    sample = getattr(value, "sample", None)
    if isinstance(sample, NativeSample):
        return sample
    raise TypeError("auxiliary source drain item must be NativeSample or expose .sample")


@dataclass(frozen=True)
class AuxiliarySourceBinding:
    """Native queue source and an explicit stream router.

    ``stream_for`` must be supplied by the verified assembly.  The common
    runtime never guesses glove identity or maps glove values into hand joints.
    """

    name: str
    source: QueueNativeSource
    stream_for: Callable[[NativeSample], str]
    drain_hz: float = 250.0

    def __post_init__(self) -> None:
        if not self.name.strip() or not callable(self.stream_for):
            raise ValueError("auxiliary source name/router must be explicit")
        _positive_number(self.drain_hz, name="auxiliary drain_hz")


@dataclass(frozen=True)
class HardwareCollectionDependencies:
    """Site-assembled dependencies returned by a SHA-256 verified factory."""

    sensor_runner_factory: Callable[[CollectionSession], SensorRunner]
    control_driver: CollectionControlDriver
    stop_targets: Callable[[], None]
    revo_hold: Callable[[str], None]
    tianji_soft_stop: Callable[[str], None]
    flush: Callable[[], None]
    close: Callable[[], object]
    abort_construction: Callable[[], None]
    assert_disconnected: Callable[[], None]
    auxiliary_sources: tuple[AuxiliarySourceBinding, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "sensor_runner_factory",
            "stop_targets",
            "revo_hold",
            "tianji_soft_stop",
            "flush",
            "close",
            "abort_construction",
            "assert_disconnected",
        ):
            if not callable(getattr(self, name)):
                raise TypeError(f"{name} must be callable")
        if not callable(getattr(self.control_driver, "step", None)):
            raise TypeError("control_driver must expose async step(anchor_timestamp_ns)")
        # Round-trip now so manifest metadata cannot fail after a device starts.
        json.dumps(dict(self.metadata), ensure_ascii=False)


def _assert_disconnected_or_abort(
    dependencies: HardwareCollectionDependencies,
) -> None:
    """Verify construction did not connect; synchronously unwind violations."""

    try:
        dependencies.assert_disconnected()
    except BaseException as assertion_failure:
        try:
            cleanup_result = dependencies.abort_construction()
            if inspect.isawaitable(cleanup_result):
                # Construction cleanup must be usable before any event loop is
                # started.  Close coroutine objects to avoid an un-awaited
                # warning, but still reject this invalid safety contract.
                close_coroutine = getattr(cleanup_result, "close", None)
                if callable(close_coroutine):
                    close_coroutine()
                raise TypeError("abort_construction must be synchronous")
        except BaseException as cleanup_failure:
            raise RuntimeError(
                "assembly was not disconnected and synchronous construction "
                f"cleanup failed: {type(cleanup_failure).__name__}:{cleanup_failure}"
            ) from assertion_failure
        raise RuntimeError(
            "assembly factory returned connected or write-active dependencies; "
            "construction was synchronously aborted"
        ) from assertion_failure


def load_hardware_collection_dependencies(
    config: HardwareCollectionConfig,
    readiness: HardwareCollectionReadiness,
) -> HardwareCollectionDependencies:
    """Load the assembly only after the complete static audit passes."""

    if not readiness.execute_ready:
        raise PermissionError("hardware collection readiness has blockers")
    section = _mapping(config.raw["assembly_factory"], name="assembly_factory")
    factory, provenance = resolve_hashed_callable(
        str(section["factory"]),
        expected_module_sha256=str(section["expected_module_sha256"]),
        name="hardware collection assembly factory",
    )
    if not provenance.hash_verified:
        raise PermissionError("hardware collection assembly factory hash is unverified")
    kwargs = _mapping(section.get("kwargs", {}), name="assembly_factory.kwargs")
    if any(not isinstance(key, str) or not key.strip() for key in kwargs):
        raise ValueError("assembly_factory.kwargs keys must be non-empty strings")
    if "config" in kwargs:
        raise ValueError("assembly_factory.kwargs cannot shadow the config argument")
    result = factory(config, **dict(kwargs))
    if not isinstance(result, HardwareCollectionDependencies):
        raise TypeError("assembly factory must return HardwareCollectionDependencies")
    # The assembly phase may instantiate adapters but may not open/connect or
    # write hardware.  The verified factory must expose a concrete assertion;
    # actual starts are owned by runner.run()/auxiliary source start below.
    _assert_disconnected_or_abort(result)
    return result


@dataclass(frozen=True)
class HardwareCollectionResult:
    master_episode: Path
    vla_episode: Path | None
    emg_review_root: Path
    anchors: int


class HardwareCollectionOrchestrator:
    """Own one episode while leaving actuator authority in verified adapters."""

    def __init__(
        self,
        config: HardwareCollectionConfig,
        dependencies: HardwareCollectionDependencies,
        *,
        readiness: HardwareCollectionReadiness | None = None,
        clock: Callable[[], int] = time.monotonic_ns,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        source_sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        control_wait_for: Callable[[Awaitable[Any], float], Awaitable[Any]] = (
            asyncio.wait_for
        ),
    ) -> None:
        observed_readiness = (
            assess_hardware_collection_config(config)
            if readiness is None
            else readiness
        )
        if not observed_readiness.execute_ready:
            raise PermissionError("hardware collection readiness has blockers")
        if observed_readiness.config_path != str(config.path):
            raise ValueError("readiness report belongs to another collection config")
        self.config = config
        self.dependencies = dependencies
        self.readiness = observed_readiness
        self.clock = clock
        if not callable(sleep):
            raise TypeError("sleep must be an async callable")
        if not callable(source_sleep):
            raise TypeError("source_sleep must be an async callable")
        if not callable(control_wait_for):
            raise TypeError("control_wait_for must be an async callable")
        self._sleep = sleep
        self._source_sleep = source_sleep
        self._control_wait_for = control_wait_for
        # Also enforce the boundary for direct dependency injection in tests
        # and embedding applications, not only the CLI factory loader.
        _assert_disconnected_or_abort(self.dependencies)

    async def _drain_source(
        self,
        binding: AuxiliarySourceBinding,
        session: CollectionSession,
        stop_event: asyncio.Event,
    ) -> None:
        delay = 1.0 / binding.drain_hz
        while not stop_event.is_set():
            for value in binding.source.drain():
                sample = _native_sample(value)
                stream = str(binding.stream_for(sample)).strip()
                if not stream:
                    raise ValueError(f"auxiliary source {binding.name} routed an empty stream")
                session.accept_sample(stream, sample)
            await self._source_sleep(delay)

    def _close_dependencies_blocking(self, timeout_s: float) -> None:
        """Run sync/async close in a bounded daemon thread.

        A timed-out vendor close may leave its daemon thread alive and retain
        process-global ownership.  The caller treats that as an intervention-
        required fault and never reports a clean episode commit.
        """

        completed = threading.Event()
        failure: list[BaseException] = []

        def target() -> None:
            try:
                value = self.dependencies.close()
                if inspect.isawaitable(value):
                    asyncio.run(value)
            except BaseException as exc:
                failure.append(exc)
            finally:
                completed.set()

        thread = threading.Thread(
            target=target,
            name="revo3-collection-close",
            daemon=True,
        )
        thread.start()
        if not completed.wait(timeout_s):
            raise RuntimeError(
                "dependencies_close_timeout_process_or_device_intervention_required"
            )
        if failure:
            raise RuntimeError("dependencies_close_failed") from failure[0]

    @staticmethod
    def _bounded_auxiliary_stop(
        binding: AuxiliarySourceBinding,
        timeout_s: float,
    ) -> None:
        completed = threading.Event()
        failure: list[BaseException] = []

        def target() -> None:
            try:
                binding.source.stop()
            except BaseException as exc:
                failure.append(exc)
            finally:
                completed.set()

        thread = threading.Thread(
            target=target,
            name=f"revo3-stop-{binding.name}",
            daemon=True,
        )
        thread.start()
        if not completed.wait(timeout_s):
            raise RuntimeError(
                f"{binding.name}_stop_timeout_process_or_device_intervention_required"
            )
        if failure:
            raise RuntimeError(f"{binding.name}_stop_failed") from failure[0]

    async def _finish_task_bounded(
        self,
        task: asyncio.Task[None],
        *,
        label: str,
        timeout_s: float,
        cancel_first: bool,
    ) -> BaseException | None:
        if cancel_first:
            task.cancel()
        done, _ = await asyncio.wait({task}, timeout=timeout_s)
        if not done:
            task.cancel()
            # Give a cancellation-cooperative task one more bounded window.
            await asyncio.wait({task}, timeout=timeout_s)
            return RuntimeError(
                f"{label}_shutdown_timeout_process_or_device_intervention_required"
            )
        try:
            task.result()
        except asyncio.CancelledError:
            return None if cancel_first else RuntimeError(f"{label}_cancelled_unexpectedly")
        except BaseException as exc:
            return exc
        return None

    @staticmethod
    def _write_cleanup_failure_evidence(path: Path | None, detail: str) -> None:
        if path is None or not path.is_dir():
            return
        destination = path / "cleanup_failure.json"
        temporary = path / ".cleanup_failure.json.tmp"
        temporary.write_text(
            json.dumps(
                {
                    "schema_version": "revo3-cleanup-failure-v1",
                    "detail": detail,
                    "clean_close": False,
                    "intervention_required": True,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)

    async def run(self) -> HardwareCollectionResult:
        raw = self.config.raw
        episode = _mapping(raw["episode"], name="episode")
        outputs = _mapping(raw["outputs"], name="outputs")
        runtime = _mapping(raw["runtime"], name="runtime")
        duration_s = float(episode["duration_s"])
        warmup_s = float(episode["anchor_start_delay_s"])
        control_step_timeout_s = float(runtime["control_step_timeout_ms"]) / 1000.0
        shutdown_timeout_s = float(runtime["shutdown_timeout_ms"]) / 1000.0
        start_ns = int(self.clock())
        epoch_ns = start_ns + int(round(warmup_s * 1_000_000_000.0))
        master_root = self.config.resolve_path(outputs["master_root"])
        vla_root = self.config.resolve_path(outputs["vla_root"])
        emg_review_root = self.config.resolve_path(outputs["emg_review_root"])
        assert master_root is not None and vla_root is not None and emg_review_root is not None
        # Fail on filesystem/episode collisions before constructing or
        # starting any hardware dependency.  Empty product roots are harmless
        # preparation evidence; an episode is published only by recorder
        # commit/export atomic renames.
        for root in (master_root, vla_root, emg_review_root):
            root.mkdir(parents=True, exist_ok=True)
        episode_id = str(episode["episode_id"])
        occupied = (
            master_root / ".inprogress" / episode_id,
            master_root / "committed" / episode_id,
            master_root / "quarantine" / f"{episode_id}.aborted",
            vla_root / episode_id,
            vla_root / f".{episode_id}.inprogress",
        )
        collision = next((path for path in occupied if path.exists()), None)
        if collision is not None:
            raise FileExistsError(f"collection output already exists: {collision}")
        timeouts = {
            str(name): int(round(float(milliseconds) * 1_000_000.0))
            for name, milliseconds in _mapping(
                runtime["required_source_timeouts_ms"],
                name="runtime.required_source_timeouts_ms",
            ).items()
        }
        anchor_streams = tuple(str(value) for value in runtime["anchor_streams"])
        metadata = {
            "task": str(episode["task"]),
            "instruction": str(episode["instruction"]),
            **{
                key: episode[key]
                for key in (
                    "task_id",
                    "task_version",
                    "object_id",
                    "object_instance",
                    "operator",
                    "collection_day",
                    "grasp_primitive",
                    "instruction_source",
                    "planner_revision",
                    "planner_output_sha256",
                )
                if key in episode
            },
            "hardware_collection": True,
            "synthetic_fixture": False,
            "action_label_source": "accepted_revo_hand_exact_sent_target",
            "emg_in_vla_projection": False,
            "emg_export_requires_human_reviewed_intervals": True,
            "contains_cair_residual": False,
            "control_step_timeout_ms": float(runtime["control_step_timeout_ms"]),
            "max_command_latency_ns": int(
                round(float(runtime["max_command_latency_ms"]) * 1_000_000.0)
            ),
            "shutdown_timeout_ms": float(runtime["shutdown_timeout_ms"]),
            "safety_watchdog_budget_ms": float(runtime["safety_watchdog_budget_ms"]),
            **dict(self.dependencies.metadata),
            "tactile_profile": str(
                _mapping(raw["tactile"], name="tactile")["vla_tactile_profile"]
            ),
            "checkpoint_family_id": str(
                _mapping(raw["tactile"], name="tactile")["checkpoint_family_id"]
            ),
            "normalization_family_id": str(
                _mapping(raw["tactile"], name="tactile")["normalization_family_id"]
            ),
            "capability_manifest_sha256": str(
                _mapping(raw["tactile"], name="tactile")["capability_manifest_sha256"]
            ),
            "max_tactile_inter_finger_skew_ns": int(
                _mapping(raw["tactile"], name="tactile")["max_inter_finger_skew_ns"]
            ),
            "camera_profile_id": str(
                _mapping(raw["camera"], name="camera")["camera_profile_id"]
            ),
            "camera_calibration_sha256": str(
                _mapping(raw["camera"], name="camera")["calibration_sha256"]
            ),
            "policy_annotations": [],
        }
        recorder = EpisodeRecorder(
            master_root,
            episode_id=episode_id,
            epoch_ns=epoch_ns,
            metadata=metadata,
            anchor_hz=30,
        )
        close_attempted = False
        close_succeeded = False
        close_failure_detail: str | None = None

        def flush_and_close_before_commit() -> None:
            nonlocal close_attempted, close_succeeded, close_failure_detail
            failures: list[BaseException] = []
            try:
                self.dependencies.flush()
            except BaseException as exc:
                failures.append(exc)
            close_attempted = True
            try:
                self._close_dependencies_blocking(shutdown_timeout_s)
                close_succeeded = True
            except BaseException as exc:
                close_failure_detail = f"{type(exc).__name__}:{exc}"
                failures.append(exc)
            if failures:
                raise RuntimeError(
                    "flush_or_bounded_close_failed_before_commit"
                ) from failures[0]

        session = CollectionSession(
            recorder,
            required_source_timeouts_ns=timeouts,
            anchor_streams=anchor_streams,
            stop_targets=self.dependencies.stop_targets,
            revo_hold=self.dependencies.revo_hold,
            tianji_soft_stop=self.dependencies.tianji_soft_stop,
            flush=flush_and_close_before_commit,
            clock=self.clock,
        )
        runner: SensorRunner | None = None
        stop_event = asyncio.Event()
        attempted_auxiliary: list[AuxiliarySourceBinding] = []
        sensor_task: asyncio.Task[None] | None = None
        auxiliary_tasks: list[asyncio.Task[None]] = []
        failure: BaseException | None = None
        cleanup_failures: list[str] = []
        committed: Path | None = None
        try:
            session.start()
            runner = self.dependencies.sensor_runner_factory(session)
            if not callable(getattr(runner, "run", None)) or not callable(
                getattr(runner, "request_stop", None)
            ):
                raise TypeError("sensor_runner_factory returned an invalid runner")
            # Start the main sensor runner first so camera/EMG/Revo watchdogs
            # continue to receive samples while optional IPC sources start.
            sensor_task = asyncio.create_task(runner.run(duration_s=duration_s))
            for binding in self.dependencies.auxiliary_sources:
                # Real SDK starts may block while opening serial/IPC resources;
                # keep the event-loop watchdog schedulable.  Each concrete
                # source still owns its bounded fail-closed startup timeout.
                # Register ownership before calling start.  A source may open
                # IPC/vendor state and then raise while intentionally retaining
                # its handle; the finally path must still call bounded stop.
                attempted_auxiliary.append(binding)
                await asyncio.to_thread(binding.source.start)
            auxiliary_tasks = [
                asyncio.create_task(self._drain_source(binding, session, stop_event))
                for binding in attempted_auxiliary
            ]
            end_ns = start_ns + int(round(duration_s * 1_000_000_000.0))
            anchor_period_ns = 1_000_000_000 // 30
            while int(self.clock()) < end_ns:
                now_ns = int(self.clock())
                session.poll(now_ns=now_ns)
                if sensor_task.done():
                    await sensor_task
                    if now_ns + 2_000_000 < end_ns:
                        raise RuntimeError("real sensor runner stopped before episode end")
                for task in auxiliary_tasks:
                    if task.done():
                        await task
                        raise RuntimeError("auxiliary source pump stopped before episode end")
                next_anchor_ns = session.next_anchor_timestamp_ns
                if now_ns >= next_anchor_ns:
                    if now_ns - next_anchor_ns > anchor_period_ns:
                        raise RuntimeError("30 Hz control loop missed an anchor; no catch-up burst")
                    try:
                        cycle = await self._control_wait_for(
                            self.dependencies.control_driver.step(next_anchor_ns),
                            control_step_timeout_s,
                        )
                    except asyncio.TimeoutError as exc:
                        raise RuntimeError(
                            "control_step_timeout_safety_watchdog_triggered"
                        ) from exc
                    for receipt in cycle.auxiliary_receipts:
                        session.accept_command(receipt)
                    session.accept_command(cycle.hand_receipt)
                    decision_now = max(now_ns, cycle.hand_receipt.decision_timestamp_ns)
                    recorded_anchor = session.record_anchor_if_due(
                        hand_command_request_id=cycle.hand_receipt.request_id,
                        now_ns=decision_now,
                    )
                    if recorded_anchor is not None:
                        annotations = recorder.metadata["policy_annotations"]
                        assert isinstance(annotations, list)
                        annotations.append(
                            {
                                "anchor_index": recorded_anchor.anchor_index,
                                "phase": cycle.phase,
                                "policy_loss_eligible": cycle.policy_loss_eligible,
                            }
                        )
                await self._sleep(0.001)
        except BaseException as exc:
            failure = exc
        finally:
            stop_event.set()
            if runner is not None:
                try:
                    runner.request_stop()
                except BaseException as exc:
                    cleanup_failures.append(
                        f"sensor_request_stop:{type(exc).__name__}:{exc}"
                    )
            for index, task in enumerate(auxiliary_tasks):
                result = await self._finish_task_bounded(
                    task,
                    label=f"auxiliary_pump_{index}",
                    timeout_s=shutdown_timeout_s,
                    cancel_first=True,
                )
                if result is not None:
                    cleanup_failures.append(
                        f"auxiliary_pump:{type(result).__name__}:{result}"
                    )
            if sensor_task is not None:
                sensor_result = await self._finish_task_bounded(
                    sensor_task,
                    label="sensor_runner",
                    timeout_s=shutdown_timeout_s,
                    cancel_first=False,
                )
                if sensor_result is not None:
                    if failure is None:
                        failure = sensor_result
                    else:
                        cleanup_failures.append(
                            f"sensor_runner:{type(sensor_result).__name__}:{sensor_result}"
                        )
            # Stop every auxiliary source even if one stop raises.
            for binding in reversed(attempted_auxiliary):
                try:
                    await asyncio.to_thread(
                        self._bounded_auxiliary_stop,
                        binding,
                        shutdown_timeout_s,
                    )
                except BaseException as exc:
                    cleanup_failures.append(
                        f"auxiliary_stop:{binding.name}:{type(exc).__name__}:{exc}"
                    )
        if failure is not None or cleanup_failures:
            detail = (
                f"{type(failure).__name__}:{failure}"
                if failure is not None
                else "cleanup_failure"
            )
            if cleanup_failures:
                detail += "; " + " | ".join(cleanup_failures)
            terminal: BaseException = failure or RuntimeError(detail)
            if session.state == SessionState.RECORDING:
                try:
                    session.backend_fault("hardware_collection", detail)
                except BaseException as exc:
                    terminal = exc
            if not close_attempted:
                close_attempted = True
                try:
                    self._close_dependencies_blocking(shutdown_timeout_s)
                    close_succeeded = True
                except BaseException as exc:
                    close_failure_detail = f"{type(exc).__name__}:{exc}"
                    detail += f"; dependencies_close:{close_failure_detail}"
            if close_failure_detail is not None:
                self._write_cleanup_failure_evidence(
                    session.quarantine_path,
                    close_failure_detail,
                )
                raise CollectionSessionFault(detail, session.quarantine_path) from terminal
            if isinstance(terminal, CollectionSessionFault):
                raise terminal
            raise CollectionSessionFault(detail, session.quarantine_path) from terminal

        # Session.stop owns stop-targets -> Revo hold -> Tianji soft-stop, then
        # calls the bounded flush+close wrapper before atomically committing.
        # A close timeout therefore quarantines rather than publishing data.
        try:
            committed = session.stop()
        except BaseException:
            if not close_attempted:
                close_attempted = True
                try:
                    self._close_dependencies_blocking(shutdown_timeout_s)
                    close_succeeded = True
                except BaseException as close_exc:
                    close_failure_detail = (
                        f"{type(close_exc).__name__}:{close_exc}"
                    )
            if close_failure_detail is not None:
                self._write_cleanup_failure_evidence(
                    session.quarantine_path,
                    close_failure_detail,
                )
            raise
        if not close_attempted or not close_succeeded:
            raise RuntimeError("dependencies were not cleanly closed before commit")
        anchors_path = committed / "anchors_30hz.jsonl"
        anchors = len(anchors_path.read_text(encoding="utf-8").splitlines())
        if anchors < 2:
            raise RuntimeError("committed hardware episode has fewer than two anchors")
        emg_review_root.mkdir(parents=True, exist_ok=True)
        tactile_mode = str(_mapping(raw["tactile"], name="tactile")["mode"])
        vla_episode: Path | None = None
        if bool(outputs.get("export_vla_after_commit", True)):
            if tactile_mode != "visiontouch_force6d":
                raise RuntimeError(
                    "VLA export needs VisionTouch Force6D; U21VT pressure remains master evidence"
                )
            vla_episode = export_revo3_episode(
                committed,
                vla_root,
                Revo3ExportConfig(
                    camera_stream=str(runtime.get("vla_camera_stream", "camera_rectified")),
                    state_stream=str(runtime.get("vla_state_stream", "revo_state")),
                    tactile_stream=str(runtime.get("vla_tactile_stream", "tactile")),
                    tactile_profile=str(
                        _mapping(raw["tactile"], name="tactile")[
                            "vla_tactile_profile"
                        ]
                    ),
                    synthetic_fixture=False,
                ),
            )
        return HardwareCollectionResult(
            master_episode=committed,
            vla_episode=vla_episode,
            emg_review_root=emg_review_root,
            anchors=anchors,
        )


__all__ = [
    "AuxiliarySourceBinding",
    "CollectionControlCycle",
    "CollectionControlDriver",
    "HARDWARE_COLLECTION_SCHEMA_VERSION",
    "HardwareCollectionConfig",
    "HardwareCollectionDependencies",
    "HardwareCollectionOrchestrator",
    "HardwareCollectionReadiness",
    "HardwareCollectionResult",
    "SensorRunner",
    "assess_hardware_collection_config",
    "load_hardware_collection_dependencies",
]
