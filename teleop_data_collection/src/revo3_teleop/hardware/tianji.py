"""Configuration, readiness audit and lazy assembly for Tianji teleoperation.

The checked-in example is deliberately non-armable.  A dry run performs no
imports, opens no device and writes no hardware.  Explicit plugin loading only
constructs injected objects; connection/servo/motion remain separate calls.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import ipaddress
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from revo3_teleop.backends.tianji import (
    TianjiMarvinBackend,
    TianjiSafetyLimits,
)
from revo3_teleop.backends.tianji_loader import (
    CallableModuleProvenance,
    LoadedTianjiSdk,
    TianjiSdkPluginSpec,
    load_tianji_sdk,
    resolve_hashed_callable,
)
from revo3_teleop.retargeting import (
    CartesianPose,
    CartesianSafetyBounds,
    IKSolver,
    RelativeWristRetargeter,
    TianjiJointTargetPlanner,
    WristPoseProvider,
    WristRetargetCalibration,
)
from revo3_teleop.tianji_runtime import TianjiTeleopRuntime


HARDWARE_SCHEMA_VERSION = "revo3-teleop-hardware-v2"
_PLACEHOLDER_MARKERS = ("REPLACE", "UNVERIFIED", "TODO", "CHANGEME")


def _placeholder(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        text = value.strip().upper()
        return not text or any(marker in text for marker in _PLACEHOLDER_MARKERS)
    return False


def _valid_sha256(value: object) -> bool:
    if _placeholder(value):
        return False
    digest = str(value).strip()
    return len(digest) == 64 and all(char in "0123456789abcdef" for char in digest)


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _sequence(value: object, *, name: str, size: int) -> tuple[Any, ...] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{name} must contain exactly {size} entries or be null")
    return tuple(value)


def _numeric_vector_or_none(value: object, *, name: str, size: int) -> np.ndarray | None:
    items = _sequence(value, name=name, size=size)
    if items is None or any(item is None or _placeholder(item) for item in items):
        return None
    result = np.asarray(items, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite values")
    return result


@dataclass(frozen=True)
class HardwareConfig:
    path: Path
    raw: Mapping[str, Any]

    @classmethod
    def from_json(cls, path: str | Path) -> "HardwareConfig":
        config_path = Path(path).resolve()
        with config_path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, Mapping):
            raise ValueError("hardware config root must be a JSON object")
        if value.get("schema_version") != HARDWARE_SCHEMA_VERSION:
            raise ValueError(
                f"hardware config schema_version must be {HARDWARE_SCHEMA_VERSION!r}"
            )
        if value.get("mode") != "hardware":
            raise ValueError("hardware config mode must be 'hardware'")
        # Validate all required sections without importing any plugin.
        for section in (
            "hand_collection",
            "tianji",
            "wrist_pose",
            "retargeting",
            "physical_safety",
        ):
            _mapping(value.get(section), name=section)
        return cls(path=config_path, raw=dict(value))


@dataclass(frozen=True)
class HardwareReadinessReport:
    config_path: str
    hand_collection_allowed: bool
    arm_planning_blockers: tuple[str, ...]
    arm_write_blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    evidence: tuple[str, ...]

    @property
    def arm_planning_ready(self) -> bool:
        return not self.arm_planning_blockers

    @property
    def arm_write_ready(self) -> bool:
        return not self.arm_write_blockers

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_path": self.config_path,
            "verification_level": "component-verified configuration boundary",
            "hand_collection_allowed": self.hand_collection_allowed,
            "arm_planning_ready": self.arm_planning_ready,
            "arm_hardware_write_ready": self.arm_write_ready,
            "arm_planning_blockers": list(self.arm_planning_blockers),
            "arm_write_blockers": list(self.arm_write_blockers),
            "warnings": list(self.warnings),
            "evidence": list(self.evidence),
            "unsupported_claims": [
                "Tianji vendor SDK identity or ABI verified",
                "Tianji hardware connected or moved",
                "glove-to-arm calibration verified",
                "task success or safe human operation",
            ],
        }


def _unique(items: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(items))


def assess_hardware_config(config: HardwareConfig) -> HardwareReadinessReport:
    raw = config.raw
    hand = _mapping(raw["hand_collection"], name="hand_collection")
    tianji = _mapping(raw["tianji"], name="tianji")
    wrist = _mapping(raw["wrist_pose"], name="wrist_pose")
    retarget = _mapping(raw["retargeting"], name="retargeting")
    physical = _mapping(raw["physical_safety"], name="physical_safety")
    planning: list[str] = []
    write: list[str] = []
    warnings: list[str] = []
    evidence: list[str] = ["dry_run_has_no_import_connect_or_write_side_effects"]

    hand_allowed = bool(hand.get("enabled", True))
    if not hand_allowed:
        warnings.append("hand_collection_disabled_by_config")

    if not bool(tianji.get("enabled", False)):
        planning.append("tianji_arm_disabled")
    side = str(tianji.get("side", "")).strip().upper()
    if side not in {"A", "B"}:
        planning.append("tianji_side_A_or_B_required")
    try:
        address = ipaddress.IPv4Address(str(tianji.get("robot_ip", "")))
        documentation_networks = (
            ipaddress.IPv4Network("192.0.2.0/24"),
            ipaddress.IPv4Network("198.51.100.0/24"),
            ipaddress.IPv4Network("203.0.113.0/24"),
        )
        if (
            address.is_unspecified
            or address.is_multicast
            or address.is_reserved
            or any(address in network for network in documentation_networks)
        ):
            write.append("tianji_robot_ip_is_not_a_deployable_device_address")
    except ipaddress.AddressValueError:
        write.append("valid_tianji_ipv4_required")

    joint_order = tianji.get("joint_order")
    if not isinstance(joint_order, list) or len(joint_order) != 7:
        planning.append("verified_seven_axis_joint_order_required")
    else:
        normalized = [str(item).strip() for item in joint_order]
        if any(_placeholder(item) for item in normalized) or len(set(normalized)) != 7:
            planning.append("verified_unique_nonplaceholder_joint_order_required")

    sdk = _mapping(tianji.get("sdk", {}), name="tianji.sdk")
    if _placeholder(sdk.get("client_factory")):
        write.append("local_tianji_sdk_client_factory_required")
    if _placeholder(sdk.get("expected_module_sha256")):
        write.append("local_tianji_sdk_module_hash_required")
    feedback_mode = str(sdk.get("feedback_mode", "normalized_mapping"))
    if feedback_mode == "pointer_decoder":
        for field in (
            "expected_feedback_buffer_factory_sha256",
            "expected_feedback_decoder_sha256",
        ):
            if not _valid_sha256(sdk.get(field)):
                write.append(field + "_required")
    if not _placeholder(sdk.get("feedback_argument_adapter")) and not _valid_sha256(
        sdk.get("expected_feedback_argument_adapter_sha256")
    ):
        write.append("expected_feedback_argument_adapter_sha256_required")
    if _placeholder(sdk.get("native_library_path")) or _placeholder(
        sdk.get("expected_native_library_sha256")
    ):
        write.append("local_tianji_native_library_path_and_hash_required")
    else:
        client_kwargs = _mapping(
            sdk.get("client_kwargs", {}), name="tianji.sdk.client_kwargs"
        )
        if _placeholder(client_kwargs.get("library_path")):
            write.append("sdk_client_kwargs_library_path_binding_required")
        else:
            configured_native = Path(str(sdk["native_library_path"])).expanduser()
            factory_native = Path(str(client_kwargs["library_path"])).expanduser()
            if configured_native.resolve() != factory_native.resolve():
                write.append("sdk_client_and_hashed_native_library_paths_mismatch")
    if _placeholder(sdk.get("sdk_identity")) or _placeholder(sdk.get("sdk_version")):
        write.append("verified_tianji_sdk_identity_and_version_required")
    if not _placeholder(sdk.get("client_factory")):
        try:
            TianjiSdkPluginSpec(
                client_factory=str(sdk["client_factory"]),
                feedback_mode=str(sdk.get("feedback_mode", "normalized_mapping")),
                feedback_buffer_factory=sdk.get("feedback_buffer_factory"),
                feedback_decoder=sdk.get("feedback_decoder"),
                feedback_argument_adapter=sdk.get("feedback_argument_adapter"),
                client_kwargs=_mapping(
                    sdk.get("client_kwargs", {}), name="tianji.sdk.client_kwargs"
                ),
                expected_module_sha256=sdk.get("expected_module_sha256"),
                expected_feedback_buffer_factory_sha256=sdk.get(
                    "expected_feedback_buffer_factory_sha256"
                ),
                expected_feedback_decoder_sha256=sdk.get(
                    "expected_feedback_decoder_sha256"
                ),
                expected_feedback_argument_adapter_sha256=sdk.get(
                    "expected_feedback_argument_adapter_sha256"
                ),
                native_library_path=sdk.get("native_library_path"),
                expected_native_library_sha256=sdk.get(
                    "expected_native_library_sha256"
                ),
                sdk_identity=str(sdk.get("sdk_identity", "UNVERIFIED")),
                sdk_version=str(sdk.get("sdk_version", "UNVERIFIED")),
            )
        except (TypeError, ValueError):
            write.append("valid_tianji_sdk_plugin_spec_required")

    limits = _mapping(tianji.get("safety_limits", {}), name="tianji.safety_limits")
    lower = _numeric_vector_or_none(limits.get("q_min_rad"), name="q_min_rad", size=7)
    upper = _numeric_vector_or_none(limits.get("q_max_rad"), name="q_max_rad", size=7)
    delta = _numeric_vector_or_none(limits.get("max_delta_rad"), name="max_delta_rad", size=7)
    if lower is None or upper is None or delta is None:
        write.append("bench_verified_tianji_joint_limits_required")
    elif np.any(lower >= upper) or np.any(delta <= 0.0):
        write.append("valid_tianji_joint_limits_required")
    else:
        try:
            TianjiSafetyLimits(
                q_min_rad=lower,
                q_max_rad=upper,
                max_delta_rad=delta,
                max_feedback_age_ns=int(limits.get("max_feedback_age_ns", 0)),
                max_target_age_ns=int(limits.get("max_target_age_ns", 0)),
                require_wrist_pose=True,
            )
        except (TypeError, ValueError):
            write.append("valid_tianji_watchdog_limits_required")

    source_kind = str(wrist.get("source_kind", "")).strip().lower()
    declared_pose = bool(wrist.get("provides_6dof", False))
    # This is invariant, not a configurable opinion: BrainCo EDU exposes flex,
    # IMU and magnetometer telemetry, not a calibrated translational 6-DoF pose.
    if source_kind in {"brainco", "brainco_edu", "brainco_glove"}:
        planning.append("brainco_edu_glove_does_not_provide_verified_6dof_wrist_pose")
        if declared_pose:
            planning.append("brainco_edu_provides_6dof_claim_rejected")
    elif not declared_pose:
        planning.append("verified_6dof_wrist_pose_source_required")
    if _placeholder(wrist.get("provider_factory")):
        planning.append("wrist_pose_provider_factory_required")
    if not _valid_sha256(wrist.get("expected_provider_module_sha256")):
        planning.append("wrist_pose_provider_module_hash_required")
    if _placeholder(wrist.get("source_frame")) or _placeholder(
        wrist.get("calibration_revision")
    ):
        planning.append("verified_wrist_frame_and_calibration_revision_required")
    if not bool(wrist.get("mapping_verified", False)):
        planning.append("wrist_pose_mapping_not_verified")

    if _placeholder(retarget.get("ik_solver_factory")):
        planning.append("robot_specific_ik_solver_factory_required")
    if not _valid_sha256(retarget.get("expected_ik_solver_module_sha256")):
        planning.append("ik_solver_module_hash_required")
    if _placeholder(retarget.get("ik_model_revision")):
        planning.append("tianji_kinematic_model_revision_required")
    if not bool(retarget.get("calibration_verified", False)):
        planning.append("wrist_to_tianji_calibration_not_verified")
    if not bool(retarget.get("workspace_verified", False)):
        planning.append("cartesian_workspace_not_verified")
    if bool(retarget.get("calibration_verified", False)) and bool(
        retarget.get("workspace_verified", False)
    ):
        try:
            calibration_section = _mapping(
                retarget.get("calibration"), name="retargeting.calibration"
            )
            bounds_section = _mapping(
                retarget.get("cartesian_bounds"), name="retargeting.cartesian_bounds"
            )
            calibration = WristRetargetCalibration(
                revision=str(calibration_section.get("revision", "")),
                source_frame=str(calibration_section.get("source_frame", "")),
                robot_base_frame=str(
                    calibration_section.get("robot_base_frame", "")
                ),
                tool_frame=str(calibration_section.get("tool_frame", "")),
                source_reference=_cartesian_pose(
                    calibration_section.get("source_reference"),
                    name="source_reference",
                ),
                robot_reference=_cartesian_pose(
                    calibration_section.get("robot_reference"),
                    name="robot_reference",
                ),
                source_axes_to_robot=calibration_section.get(
                    "source_axes_to_robot"
                ),
                translation_gain=calibration_section.get("translation_gain"),
            )
            CartesianSafetyBounds(
                min_position_m=bounds_section.get("min_position_m"),
                max_position_m=bounds_section.get("max_position_m"),
                max_translation_from_reference_m=float(
                    bounds_section.get("max_translation_from_reference_m")
                ),
                max_orientation_from_reference_rad=float(
                    bounds_section.get("max_orientation_from_reference_rad")
                ),
            )
            if calibration.source_frame != str(wrist.get("source_frame", "")):
                planning.append("wrist_and_retarget_source_frames_mismatch")
        except (TypeError, ValueError):
            planning.append("complete_valid_retarget_calibration_and_bounds_required")

    write.extend(planning)
    if not bool(raw.get("allow_hardware_write", False)):
        write.append("allow_hardware_write_is_false")
    if not bool(raw.get("capability_probe_confirmed", False)):
        write.append("tianji_capability_probe_unconfirmed")
    token_env = str(raw.get("arm_token_env", "")).strip()
    if not token_env:
        write.append("arm_token_environment_variable_name_required")
    elif not os.environ.get(token_env):
        write.append("arm_token_environment_variable_not_set")
    for field in (
        "physical_estop_verified",
        "payload_com_verified",
        "tool_transform_verified",
        "watchdog_verified",
        "human_safe_test_fixture_verified",
    ):
        if not bool(physical.get(field, False)):
            write.append(field + "_required")

    warnings.extend(
        (
            "public_GitHub_sources_are_third_party_or_historical_not_current_vendor_authority",
            "hardware_write_requires_separate_explicit_connect_enable_and_runtime_arm_token",
            "hand_only_collection_remains_allowed_when_all_Tianji_arm_targets_are_blocked",
        )
    )
    return HardwareReadinessReport(
        config_path=str(config.path),
        hand_collection_allowed=hand_allowed,
        arm_planning_blockers=_unique(planning),
        arm_write_blockers=_unique(write),
        warnings=_unique(warnings),
        evidence=_unique(evidence),
    )


@dataclass(frozen=True)
class TianjiHardwareAssembly:
    config: HardwareConfig
    report: HardwareReadinessReport
    sdk: LoadedTianjiSdk | None = None
    backend: TianjiMarvinBackend | None = None
    wrist_pose_provider: WristPoseProvider | None = None
    planner: TianjiJointTargetPlanner | None = None
    runtime: TianjiTeleopRuntime | None = None


def _plugin_instance(
    section: Mapping[str, Any],
    key: str,
    kwargs_key: str,
    expected_hash_key: str,
) -> tuple[Any, CallableModuleProvenance | None]:
    target = section.get(key)
    if _placeholder(target):
        return None, None
    factory, provenance = resolve_hashed_callable(
        str(target),
        expected_module_sha256=section.get(expected_hash_key),
        name=key,
    )
    kwargs = section.get(kwargs_key, {})
    if not isinstance(kwargs, Mapping):
        raise ValueError(f"{kwargs_key} must be an object")
    return factory(**dict(kwargs)), provenance


def _safety_limits(tianji: Mapping[str, Any]) -> TianjiSafetyLimits | None:
    limits = _mapping(tianji.get("safety_limits", {}), name="tianji.safety_limits")
    lower = _numeric_vector_or_none(limits.get("q_min_rad"), name="q_min_rad", size=7)
    upper = _numeric_vector_or_none(limits.get("q_max_rad"), name="q_max_rad", size=7)
    delta = _numeric_vector_or_none(limits.get("max_delta_rad"), name="max_delta_rad", size=7)
    if lower is None or upper is None or delta is None:
        return None
    return TianjiSafetyLimits(
        q_min_rad=lower,
        q_max_rad=upper,
        max_delta_rad=delta,
        max_feedback_age_ns=int(limits.get("max_feedback_age_ns", 250_000_000)),
        max_target_age_ns=int(limits.get("max_target_age_ns", 100_000_000)),
        require_wrist_pose=True,
    )


def _cartesian_pose(value: object, *, name: str) -> CartesianPose:
    section = _mapping(value, name=name)
    return CartesianPose(
        frame=str(section.get("frame", "")),
        child_frame=str(section.get("child_frame", "")),
        position_m=section.get("position_m"),
        quaternion_xyzw=section.get("quaternion_xyzw"),
    )


def _planner_from_config(
    config: HardwareConfig,
    ik_solver: IKSolver,
) -> TianjiJointTargetPlanner:
    raw = config.raw
    wrist = _mapping(raw["wrist_pose"], name="wrist_pose")
    retarget = _mapping(raw["retargeting"], name="retargeting")
    calibration = _mapping(retarget.get("calibration"), name="retargeting.calibration")
    bounds = _mapping(retarget.get("cartesian_bounds"), name="retargeting.cartesian_bounds")
    retargeter = RelativeWristRetargeter(
        WristRetargetCalibration(
            revision=str(calibration.get("revision", "")),
            source_frame=str(calibration.get("source_frame", "")),
            robot_base_frame=str(calibration.get("robot_base_frame", "")),
            tool_frame=str(calibration.get("tool_frame", "")),
            source_reference=_cartesian_pose(
                calibration.get("source_reference"), name="source_reference"
            ),
            robot_reference=_cartesian_pose(
                calibration.get("robot_reference"), name="robot_reference"
            ),
            source_axes_to_robot=calibration.get("source_axes_to_robot"),
            translation_gain=calibration.get("translation_gain"),
        ),
        CartesianSafetyBounds(
            min_position_m=bounds.get("min_position_m"),
            max_position_m=bounds.get("max_position_m"),
            max_translation_from_reference_m=float(
                bounds.get("max_translation_from_reference_m")
            ),
            max_orientation_from_reference_rad=float(
                bounds.get("max_orientation_from_reference_rad")
            ),
        ),
    )
    tianji = _mapping(raw["tianji"], name="tianji")
    return TianjiJointTargetPlanner(
        retargeter=retargeter,
        ik_solver=ik_solver,
        expected_joint_order=tuple(str(item) for item in tianji["joint_order"]),
        expected_clock_domain=str(wrist.get("clock_domain", "workstation_monotonic")),
        max_pose_age_ns=int(wrist.get("max_pose_age_ns", 100_000_000)),
        minimum_pose_confidence=float(wrist.get("minimum_confidence", 0.8)),
        max_ik_position_residual_m=float(
            retarget.get("max_ik_position_residual_m", 0.005)
        ),
        max_ik_orientation_residual_rad=float(
            retarget.get("max_ik_orientation_residual_rad", 0.0523598776)
        ),
        max_ik_iterations=int(retarget.get("max_ik_iterations", 200)),
    )


def assemble_tianji_hardware(
    config: HardwareConfig,
    *,
    load_sdk_plugin: bool = False,
    load_retargeting_plugins: bool = False,
) -> TianjiHardwareAssembly:
    """Construct opt-in boundaries, but never connect, enable, or send motion."""

    report = assess_hardware_config(config)
    raw = config.raw
    tianji = _mapping(raw["tianji"], name="tianji")
    sdk_bundle: LoadedTianjiSdk | None = None
    backend: TianjiMarvinBackend | None = None
    provider: WristPoseProvider | None = None
    planner: TianjiJointTargetPlanner | None = None
    runtime: TianjiTeleopRuntime | None = None
    extra_write_blockers: list[str] = []
    evidence = list(report.evidence)

    if load_sdk_plugin:
        sdk_section = _mapping(tianji.get("sdk", {}), name="tianji.sdk")
        if _placeholder(sdk_section.get("client_factory")):
            extra_write_blockers.append("sdk_plugin_load_requested_but_factory_missing")
        else:
            spec = TianjiSdkPluginSpec(
                client_factory=str(sdk_section["client_factory"]),
                feedback_mode=str(sdk_section.get("feedback_mode", "normalized_mapping")),
                feedback_buffer_factory=sdk_section.get("feedback_buffer_factory"),
                feedback_decoder=sdk_section.get("feedback_decoder"),
                feedback_argument_adapter=sdk_section.get("feedback_argument_adapter"),
                client_kwargs=_mapping(
                    sdk_section.get("client_kwargs", {}), name="tianji.sdk.client_kwargs"
                ),
                expected_module_sha256=sdk_section.get("expected_module_sha256"),
                expected_feedback_buffer_factory_sha256=sdk_section.get(
                    "expected_feedback_buffer_factory_sha256"
                ),
                expected_feedback_decoder_sha256=sdk_section.get(
                    "expected_feedback_decoder_sha256"
                ),
                expected_feedback_argument_adapter_sha256=sdk_section.get(
                    "expected_feedback_argument_adapter_sha256"
                ),
                native_library_path=sdk_section.get("native_library_path"),
                expected_native_library_sha256=sdk_section.get(
                    "expected_native_library_sha256"
                ),
                sdk_identity=str(sdk_section.get("sdk_identity", "UNVERIFIED")),
                sdk_version=str(sdk_section.get("sdk_version", "UNVERIFIED")),
            )
            sdk_bundle = load_tianji_sdk(spec)
            # Construct a disconnected, permanently non-writable probe adapter.
            # A separate backend is created below only after *all* static and
            # dynamically verified readiness gates have passed.
            backend = TianjiMarvinBackend(
                sdk_bundle.client,
                side=str(tianji.get("side", "")),
                safety_limits=_safety_limits(tianji),
                allow_hardware_write=False,
                capability_probe_confirmed=False,
                arm_token=None,
                feedback_buffer_factory=sdk_bundle.feedback_buffer_factory,
                feedback_decoder=sdk_bundle.feedback_decoder,
                feedback_argument_adapter=sdk_bundle.feedback_argument_adapter,
                joint_order=tuple(str(item) for item in tianji.get("joint_order", ())),
            )
            capability = backend.probe_capabilities()
            if not capability.complete:
                extra_write_blockers.append(
                    "sdk_missing_capabilities:" + ",".join(capability.missing_methods)
                )
            for executable in sdk_bundle.provenance.executable_plugins:
                evidence.append(
                    "sdk_executable_hash_verified:"
                    + executable.import_target
                    + "="
                    + str(executable.hash_verified).lower()
                )
                if not executable.hash_verified:
                    extra_write_blockers.append(
                        "sdk_executable_module_hash_unverified:" + executable.import_target
                    )
            if not sdk_bundle.provenance.native_library_hash_verified:
                extra_write_blockers.append("sdk_native_library_hash_unverified")
            if not sdk_bundle.provenance.client_native_library_path_verified:
                extra_write_blockers.append("sdk_native_library_binding_unverified")
            evidence.append("sdk_plugin_loaded_without_robot_connection")
            evidence.append(
                "sdk_module_hash_verified=" + str(sdk_bundle.provenance.hash_verified).lower()
            )
            evidence.append(
                "sdk_native_library_hash_verified="
                + str(
                    sdk_bundle.provenance.native_library_hash_verified
                ).lower()
            )
            evidence.append(
                "sdk_client_native_library_binding_verified="
                + str(
                    sdk_bundle.provenance.client_native_library_path_verified
                ).lower()
            )

    if load_retargeting_plugins:
        wrist = _mapping(raw["wrist_pose"], name="wrist_pose")
        source_kind = str(wrist.get("source_kind", "")).strip().lower()
        if source_kind in {"brainco", "brainco_edu", "brainco_glove"}:
            # Never honor a plugin that attempts to relabel BrainCo EDU raw
            # telemetry as translational wrist pose.
            extra_write_blockers.append("brainco_edu_arm_plugin_load_hard_blocked")
        else:
            provider, provider_provenance = _plugin_instance(
                wrist,
                "provider_factory",
                "provider_kwargs",
                "expected_provider_module_sha256",
            )
            if provider is not None and not bool(
                getattr(provider, "provides_wrist_pose", False)
            ):
                raise ValueError("loaded wrist provider does not declare verified 6-DoF")
            if provider_provenance is not None:
                evidence.append(
                    "wrist_provider_module_hash_verified="
                    + str(provider_provenance.hash_verified).lower()
                )
                if not provider_provenance.hash_verified:
                    extra_write_blockers.append("wrist_provider_module_hash_unverified")
        retarget = _mapping(raw["retargeting"], name="retargeting")
        ik_solver, ik_provenance = _plugin_instance(
            retarget,
            "ik_solver_factory",
            "ik_solver_kwargs",
            "expected_ik_solver_module_sha256",
        )
        if ik_provenance is not None:
            evidence.append(
                "ik_solver_module_hash_verified="
                + str(ik_provenance.hash_verified).lower()
            )
            if not ik_provenance.hash_verified:
                extra_write_blockers.append("ik_solver_module_hash_unverified")
        if provider is not None and ik_solver is not None:
            planner = _planner_from_config(config, ik_solver)
            evidence.append("wrist_and_ik_plugins_constructed_without_starting_hardware")

    if bool(raw.get("allow_hardware_write", False)):
        if sdk_bundle is None:
            extra_write_blockers.append("sdk_plugin_must_be_loaded_before_write")
        if provider is None or planner is None:
            extra_write_blockers.append(
                "verified_wrist_and_ik_plugins_must_be_loaded_before_write"
            )

    report = replace(
        report,
        arm_write_blockers=_unique(
            (*report.arm_write_blockers, *extra_write_blockers)
        ),
        evidence=_unique(evidence),
    )

    # Bind the readiness result into the actual write authority.  The earlier
    # adapter, if any, is probe-only.  A writable backend/runtime exists only
    # when the complete frozen report has no blocker.
    if (
        report.arm_write_ready
        and sdk_bundle is not None
        and provider is not None
        and planner is not None
    ):
        token_env = str(raw.get("arm_token_env", "")).strip()
        backend = TianjiMarvinBackend(
            sdk_bundle.client,
            side=str(tianji.get("side", "")),
            safety_limits=_safety_limits(tianji),
            allow_hardware_write=True,
            capability_probe_confirmed=True,
            arm_token=os.environ.get(token_env) if token_env else None,
            feedback_buffer_factory=sdk_bundle.feedback_buffer_factory,
            feedback_decoder=sdk_bundle.feedback_decoder,
            feedback_argument_adapter=sdk_bundle.feedback_argument_adapter,
            joint_order=tuple(str(item) for item in tianji.get("joint_order", ())),
        )
        runtime = TianjiTeleopRuntime(
            backend=backend,
            wrist_pose_provider=provider,
            planner=planner,
            arm_token=os.environ.get(token_env) if token_env else None,
        )
    return TianjiHardwareAssembly(
        config=config,
        report=report,
        sdk=sdk_bundle,
        backend=backend,
        wrist_pose_provider=provider,
        planner=planner,
        runtime=runtime,
    )


__all__ = [
    "HARDWARE_SCHEMA_VERSION",
    "HardwareConfig",
    "HardwareReadinessReport",
    "TianjiHardwareAssembly",
    "assemble_tianji_hardware",
    "assess_hardware_config",
]
