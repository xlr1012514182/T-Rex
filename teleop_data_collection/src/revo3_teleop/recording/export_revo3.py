"""Allowlist projection from a committed master episode to Revo3 VLA data."""

from __future__ import annotations

from dataclasses import dataclass
import json
import hashlib
import os
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image

from revo3_v1.data.episode import (
    ACTION_SEMANTICS,
    POLICY_PHASES,
    RevoEpisode,
    SUPPORTED_TASKS,
)
from revo3_v1.planner.schema import TASK_GRASP_PRIMITIVES, SupportedTask
from revo3_v1.revo.contracts import JOINT_COUNT, JOINT_ORDER_HASH

from revo3_teleop.contracts import CommandReceipt
from revo3_teleop.recording.recorder import load_native_payload


META_ALLOWLIST = frozenset(
    {
        "schema_version",
        "episode_id",
        "task_id",
        "task_version",
        "task",
        "object_id",
        "object_instance",
        "operator",
        "collection_day",
        "grasp_primitive",
        "instruction",
        "instruction_sha256",
        "instruction_source",
        "planner_revision",
        "planner_output_sha256",
        "fps",
        "time_grid_hz",
        "timestamp_tolerance_ns",
        "resampled_30hz",
        "timestamp_alignment_verified",
        "max_command_latency_ns",
        "max_tactile_inter_finger_skew_ns",
        "joint_order_hash",
        "tactile_num_fingers",
        "tactile_profile",
        "checkpoint_family_id",
        "normalization_family_id",
        "capability_manifest_sha256",
        "camera_profile_id",
        "camera_calibration_sha256",
        "action_label_source",
        "action_semantics",
        "contains_cair_residual",
        "image_paths",
        "contains_emg",
        "synthetic_fixture",
    }
)
FRAME_ALLOWLIST = frozenset(
    {
        "timestamp_ns",
        "camera_capture_timestamp_ns",
        "camera_receive_timestamp_ns",
        "state_timestamp_ns",
        "touch_timestamp_ns",
        "tactile_diff_timestamp_ns",
        "action_decision_timestamp_ns",
        "action_write_timestamp_ns",
        "controller_sequence",
        "request_id_hash",
        "policy_loss_eligible",
        "phase",
        "state_rad",
        "action_target_rad",
        "tactile_features",
        "tactile_diff",
        "force6d_finger_timestamp_ns",
        "tactile_history_f6",
        "tactile_history_timestamp_ns",
        "tactile_history_sequence",
    }
)


@dataclass(frozen=True)
class Revo3ExportConfig:
    camera_stream: str = "camera"
    state_stream: str = "revo_state"
    tactile_stream: str = "tactile"
    camera_key: str = "rgb"
    state_key: str = "q_rad"
    tactile_key: str = "features"
    tactile_diff_key: str = "tactile_diff"
    tactile_diff_timestamp_key: str = "tactile_diff_timestamp_ns"
    tactile_profile: str = "profile_a_force6d_diff"
    max_observation_age_ns: int = 150_000_000
    timestamp_tolerance_ns: int = 2_000_000
    synthetic_fixture: bool = False

    def __post_init__(self) -> None:
        streams = (self.camera_stream, self.state_stream, self.tactile_stream)
        if any(not str(value).strip() for value in streams):
            raise ValueError("export stream names must be non-empty")
        if len(set(streams)) != 3:
            raise ValueError("camera/state/tactile streams must be distinct")
        if any(
            not str(value).strip()
            for value in (self.camera_key, self.state_key, self.tactile_key)
        ):
            raise ValueError("export payload keys must be non-empty")
        if self.tactile_profile not in {
            "profile_a_force6d_diff",
            "profile_b_diff_only",
            "ablation_force6d_only",
        }:
            raise ValueError("unsupported export tactile_profile")
        if self.max_observation_age_ns <= 0:
            raise ValueError("max_observation_age_ns must be positive")
        if not 0 <= self.timestamp_tolerance_ns <= 2_000_000:
            raise ValueError("timestamp_tolerance_ns must be within 2 ms")


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"row {line_number} in {path} is not an object")
            rows.append(value)
    return rows


def _receipt(row: Mapping[str, object]) -> CommandReceipt:
    return CommandReceipt(
        request_id=str(row["request_id"]),
        component=str(row["component"]),
        accepted=bool(row["accepted"]),
        requested_target=np.asarray(row["requested_target"], dtype=np.float32),
        authorized_target=(
            None
            if row.get("authorized_target") is None
            else np.asarray(row["authorized_target"], dtype=np.float32)
        ),
        exact_sent_target=(
            None
            if row.get("exact_sent_target") is None
            else np.asarray(row["exact_sent_target"], dtype=np.float32)
        ),
        decision_timestamp_ns=int(row["decision_timestamp_ns"]),
        write_timestamp_ns=(
            None if row.get("write_timestamp_ns") is None else int(row["write_timestamp_ns"])
        ),
        controller_sequence=(
            None if row.get("controller_sequence") is None else int(row["controller_sequence"])
        ),
        clipped=bool(row.get("clipped", False)),
        reason=str(row.get("reason", "")),
        unit=str(row.get("unit", "rad")),
        joint_order_hash=str(row.get("joint_order_hash", "")),
    )


def _stream_reference(anchor: Mapping[str, object], stream: str) -> Mapping[str, object]:
    streams = anchor.get("streams")
    if not isinstance(streams, dict) or stream not in streams:
        raise ValueError(f"anchor is missing required stream {stream!r}")
    reference = streams[stream]
    if not isinstance(reference, dict):
        raise ValueError(f"anchor stream {stream!r} reference is malformed")
    if str(reference.get("stream")) != stream:
        raise ValueError("anchor stream key/reference mismatch")
    return reference


def _stream_index(episode_root: Path, stream: str) -> dict[int, Mapping[str, object]]:
    result: dict[int, Mapping[str, object]] = {}
    for row in _read_jsonl(episode_root / "streams" / stream / "index.jsonl"):
        header = row.get("header")
        if not isinstance(header, dict):
            raise ValueError(f"stream {stream!r} index header is malformed")
        sequence = int(header.get("sequence", -1))
        if sequence in result:
            raise ValueError(f"stream {stream!r} has duplicate sequence {sequence}")
        result[sequence] = row
    return result


def _validate_reference(
    reference: Mapping[str, object],
    *,
    stream: str,
    index: Mapping[int, Mapping[str, object]],
    anchor_timestamp_ns: int,
    decision_timestamp_ns: int,
) -> None:
    sequence = int(reference.get("sequence", -1))
    row = index.get(sequence)
    if row is None:
        raise ValueError(f"anchor references absent {stream!r} sequence {sequence}")
    header = row["header"]
    expected = (
        (str(reference.get("source_id")), str(header.get("source_id")), "source_id"),
        (
            str(reference.get("clock_domain")),
            str(header.get("clock_domain")),
            "clock_domain",
        ),
        (
            int(reference.get("capture_timestamp_ns", -1)),
            int(header.get("capture_timestamp_ns", -2)),
            "capture_timestamp_ns",
        ),
        (str(reference.get("relative_path")), str(row.get("relative_path")), "relative_path"),
        (int(reference.get("row_index", -1)), int(row.get("row_index", -2)), "row_index"),
    )
    for observed, required, name in expected:
        if observed != required:
            raise ValueError(f"anchor/index {stream!r} {name} mismatch")
    capture_ns = int(reference["capture_timestamp_ns"])
    if int(reference.get("age_ns", -1)) != anchor_timestamp_ns - capture_ns:
        raise ValueError(f"anchor {stream!r} age_ns is inconsistent")
    if not bool(header.get("valid", False)):
        raise ValueError(f"anchor references invalid {stream!r} sample")
    receive_ns = int(header.get("receive_timestamp_ns", -1))
    if receive_ns < 0:
        raise ValueError(f"anchor references {stream!r} sample without receive timestamp")
    if receive_ns > decision_timestamp_ns:
        raise ValueError(
            f"anchor references {stream!r} sample received after controller decision"
        )


def export_revo3_episode(
    committed_episode: str | Path,
    output_root: str | Path,
    config: Revo3ExportConfig = Revo3ExportConfig(),
) -> Path:
    """Create the only hand-only view eligible for the current T-Rex loader.

    The projector copies only RGB, Revo state, exact sent Revo controller
    targets, tactile features, instruction, and anchor timestamps.  Tianji,
    glove, and raw EMG streams remain solely in the committed master episode.
    """

    source = Path(committed_episode).resolve()
    manifest = _read_json(source / "manifest.json")
    if manifest.get("schema_version") != "revo3-teleop-master-v1":
        raise ValueError("unsupported master episode schema")
    if manifest.get("lifecycle") != "committed":
        raise ValueError("only committed master episodes may be exported")
    if int(manifest.get("anchor_hz", 0)) != 30:
        raise ValueError("Revo3 VLA projection requires a 30 Hz anchor index")
    episode_id = str(manifest["episode_id"])
    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("master episode metadata is malformed")
    task = str(metadata.get("task", ""))
    instruction = str(metadata.get("instruction", "")).strip()
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"unsupported Revo3 task: {task!r}")
    if not instruction:
        raise ValueError("master episode requires a non-empty instruction")
    required_provenance = (
        "task_id",
        "task_version",
        "object_id",
        "object_instance",
        "operator",
        "collection_day",
        "grasp_primitive",
        "instruction_source",
        "checkpoint_family_id",
        "normalization_family_id",
        "capability_manifest_sha256",
        "camera_profile_id",
        "camera_calibration_sha256",
        "max_command_latency_ns",
        "max_tactile_inter_finger_skew_ns",
    )
    if config.synthetic_fixture:
        synthetic_defaults = {
            "task_id": f"synthetic-{episode_id}",
            "task_version": 1,
            "object_id": task,
            "object_instance": "synthetic-instance",
            "operator": "synthetic-operator",
            "collection_day": "synthetic",
            "grasp_primitive": TASK_GRASP_PRIMITIVES[
                SupportedTask(task)
            ].value,
            "instruction_source": "manual_canonical",
            "checkpoint_family_id": "synthetic-checkpoint",
            "normalization_family_id": "synthetic-normalization",
            "capability_manifest_sha256": "0" * 64,
            "camera_profile_id": "synthetic-camera",
            "camera_calibration_sha256": "0" * 64,
            "max_command_latency_ns": 33_333_333,
            "max_tactile_inter_finger_skew_ns": 50_000_000,
            "contains_cair_residual": False,
        }
        metadata = {**synthetic_defaults, **metadata}
    missing_provenance = [
        name for name in required_provenance
        if metadata.get(name) in (None, "")
    ]
    if missing_provenance:
        raise ValueError(
            f"master episode lacks required policy provenance: {missing_provenance}"
        )
    max_command_latency_ns = int(metadata["max_command_latency_ns"])
    if not 0 < max_command_latency_ns <= 33_333_333:
        raise ValueError("max_command_latency_ns must be in (0,33333333]")
    max_tactile_inter_finger_skew_ns = int(
        metadata["max_tactile_inter_finger_skew_ns"]
    )
    if max_tactile_inter_finger_skew_ns <= 0:
        raise ValueError("max_tactile_inter_finger_skew_ns must be positive")
    if metadata.get("contains_cair_residual") is not False:
        raise ValueError("teleoperation policy export requires contains_cair_residual=false")
    instruction_sha256 = hashlib.sha256(instruction.encode("utf-8")).hexdigest()
    declared_instruction_sha256 = str(
        metadata.get("instruction_sha256", instruction_sha256)
    )
    if declared_instruction_sha256 != instruction_sha256:
        raise ValueError("instruction_sha256 does not match instruction")
    instruction_source = str(metadata["instruction_source"])
    planner_revision = str(metadata.get("planner_revision", ""))
    planner_output_sha256 = str(metadata.get("planner_output_sha256", ""))
    if instruction_source == "frozen_planner" and (
        not planner_revision or len(planner_output_sha256) != 64
    ):
        raise ValueError("frozen_planner instruction lacks revision/hash provenance")

    anchors = _read_jsonl(source / "anchors_30hz.jsonl")
    if len(anchors) < 2:
        raise ValueError("Revo3 export requires at least two 30 Hz anchors")
    annotations = metadata.get("policy_annotations")
    if annotations is None:
        if not config.synthetic_fixture:
            raise ValueError(
                "real export requires policy_annotations for every anchor"
            )
        annotations = [
            {"anchor_index": index, "phase": "precontact", "policy_loss_eligible": True}
            for index in range(len(anchors))
        ]
    if not isinstance(annotations, list) or len(annotations) != len(anchors):
        raise ValueError("policy_annotations must contain one entry per anchor")
    annotation_by_index: dict[int, Mapping[str, object]] = {}
    for annotation in annotations:
        if not isinstance(annotation, dict):
            raise ValueError("policy annotation must be an object")
        annotation_index = int(annotation.get("anchor_index", -1))
        if annotation_index in annotation_by_index:
            raise ValueError("duplicate policy annotation anchor_index")
        phase = str(annotation.get("phase", ""))
        if phase not in POLICY_PHASES:
            raise ValueError(f"invalid policy phase: {phase!r}")
        if not isinstance(annotation.get("policy_loss_eligible"), bool):
            raise ValueError("policy_loss_eligible annotation must be boolean")
        annotation_by_index[annotation_index] = annotation
    if set(annotation_by_index) != set(range(len(anchors))):
        raise ValueError("policy annotations must cover contiguous anchor indices")
    receipts: dict[str, CommandReceipt] = {}
    for row in _read_jsonl(source / "command_receipts.jsonl"):
        receipt = _receipt(row)
        if receipt.request_id in receipts:
            raise ValueError(f"duplicate command receipt: {receipt.request_id}")
        receipts[receipt.request_id] = receipt
    stream_indices = {
        stream: _stream_index(source, stream)
        for stream in (config.camera_stream, config.state_stream, config.tactile_stream)
    }
    tactile_native_rows = sorted(
        stream_indices[config.tactile_stream].values(),
        key=lambda row: int(row["header"]["sequence"]),
    )
    tactile_position_by_sequence = {
        int(row["header"]["sequence"]): index
        for index, row in enumerate(tactile_native_rows)
    }

    timestamps: list[int] = []
    camera_capture_timestamps: list[int] = []
    camera_receive_timestamps: list[int] = []
    state_timestamps: list[int] = []
    touch_timestamps: list[int] = []
    diff_timestamps: list[np.ndarray] = []
    action_decision_timestamps: list[int] = []
    action_write_timestamps: list[int] = []
    controller_sequences: list[int] = []
    request_id_hashes: list[str] = []
    policy_loss_eligible: list[bool] = []
    phases: list[str] = []
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    tactile: list[np.ndarray] = []
    tactile_diff: list[np.ndarray] = []
    force_finger_timestamps: list[np.ndarray] = []
    tactile_histories: list[np.ndarray] = []
    tactile_history_timestamps: list[np.ndarray] = []
    tactile_history_sequences: list[np.ndarray] = []
    images: list[np.ndarray] = []
    used_commands: set[str] = set()
    previous_timestamp = -1

    for expected_index, anchor in enumerate(anchors):
        index = int(anchor.get("anchor_index", -1))
        timestamp_ns = int(anchor.get("timestamp_ns", -1))
        if index != expected_index:
            raise ValueError("anchor indices must be contiguous from zero")
        expected_timestamp_ns = int(manifest["epoch_ns"]) + (1_000_000_000 * index) // 30
        if timestamp_ns != expected_timestamp_ns:
            raise ValueError("anchor timestamp is not on the declared absolute 30 Hz grid")
        if timestamp_ns <= previous_timestamp:
            raise ValueError("anchor timestamps must be strictly increasing")
        previous_timestamp = timestamp_ns

        command_id = str(anchor.get("hand_command_request_id", ""))
        if command_id in used_commands:
            raise ValueError("one controller receipt cannot supervise two anchors")
        used_commands.add(command_id)
        receipt = receipts.get(command_id)
        if receipt is None:
            raise ValueError(f"anchor references unknown command {command_id!r}")
        if receipt.component != "revo_hand" or not receipt.accepted:
            raise ValueError("anchor must reference an accepted Revo hand command")
        if receipt.unit != "rad" or receipt.joint_order_hash != JOINT_ORDER_HASH:
            raise ValueError("Revo command unit or joint order does not match the VLA contract")
        if receipt.exact_sent_target is None or receipt.exact_sent_target.shape != (JOINT_COUNT,):
            raise ValueError(f"exact_sent_target must have shape ({JOINT_COUNT},)")
        if receipt.decision_timestamp_ns < timestamp_ns:
            raise ValueError("controller decision predates its 30 Hz anchor")
        assert receipt.write_timestamp_ns is not None
        assert receipt.controller_sequence is not None
        if receipt.write_timestamp_ns - receipt.decision_timestamp_ns > max_command_latency_ns:
            raise ValueError("controller write latency exceeds max_command_latency_ns")

        camera_ref = _stream_reference(anchor, config.camera_stream)
        state_ref = _stream_reference(anchor, config.state_stream)
        tactile_ref = _stream_reference(anchor, config.tactile_stream)
        requires_force = config.tactile_profile in {
            "profile_a_force6d_diff", "ablation_force6d_only"
        }
        requires_diff = config.tactile_profile in {
            "profile_a_force6d_diff", "profile_b_diff_only"
        }
        history_payloads: list[np.ndarray] = []
        history_timestamps = np.empty(0, dtype=np.int64)
        history_sequences = np.empty(0, dtype=np.int64)
        if requires_force:
            selected_sequence = int(tactile_ref["sequence"])
            selected_position = tactile_position_by_sequence.get(selected_sequence)
            if selected_position is None:
                raise ValueError("tactile anchor sequence is absent from native index")
            if selected_position < 15:
                # No repeated/padded samples: leading warm-up anchors are not
                # exported as policy frames at all.
                continue
            history_rows = tactile_native_rows[selected_position - 15 : selected_position + 1]
            history_timestamps = np.asarray(
                [int(row["header"]["capture_timestamp_ns"]) for row in history_rows],
                dtype=np.int64,
            )
            history_sequences = np.asarray(
                [int(row["header"]["sequence"]) for row in history_rows],
                dtype=np.int64,
            )
            if np.any(np.diff(history_timestamps) <= 0) or np.any(
                np.diff(history_sequences) <= 0
            ):
                raise ValueError("native tactile history must contain 16 distinct ordered samples")
            for row in history_rows:
                payload = load_native_payload(source, config.tactile_stream, row)
                if config.tactile_key not in payload:
                    raise KeyError("native tactile history lacks Force6D features")
                feature = np.asarray(payload[config.tactile_key], dtype=np.float32)
                if feature.shape != (5, 6) or not np.isfinite(feature).all():
                    raise ValueError("native tactile history item must be finite [5,6]")
                history_payloads.append(feature.copy())
        referenced = (
            (config.camera_stream, camera_ref),
            (config.state_stream, state_ref),
            (config.tactile_stream, tactile_ref),
        )
        for stream, reference in referenced:
            _validate_reference(
                reference,
                stream=stream,
                index=stream_indices[stream],
                anchor_timestamp_ns=timestamp_ns,
                decision_timestamp_ns=receipt.decision_timestamp_ns,
            )
            capture_ns = int(reference["capture_timestamp_ns"])
            if capture_ns > timestamp_ns:
                raise ValueError("export refuses future observation samples")
            if capture_ns > receipt.decision_timestamp_ns:
                raise ValueError("controller decision predates a selected observation")

        camera_payload = load_native_payload(source, config.camera_stream, camera_ref)
        state_payload = load_native_payload(source, config.state_stream, state_ref)
        tactile_payload = load_native_payload(source, config.tactile_stream, tactile_ref)
        if config.camera_key not in camera_payload:
            raise KeyError(f"camera payload lacks {config.camera_key!r}")
        if config.state_key not in state_payload:
            raise KeyError(f"state payload lacks {config.state_key!r}")
        if requires_force and config.tactile_key not in tactile_payload:
            raise KeyError(f"tactile payload lacks {config.tactile_key!r}")
        if not requires_force and config.tactile_key in tactile_payload:
            raise ValueError("DIFF-only profile must not carry unused Force6D features")
        if requires_diff and config.tactile_diff_key not in tactile_payload:
            raise KeyError(f"tactile payload lacks {config.tactile_diff_key!r}")
        if requires_diff and config.tactile_diff_timestamp_key not in tactile_payload:
            raise KeyError(
                f"tactile payload lacks {config.tactile_diff_timestamp_key!r}"
            )
        rgb = np.asarray(camera_payload[config.camera_key])
        state = np.asarray(state_payload[config.state_key], dtype=np.float32)
        touch = (
            np.asarray(tactile_payload[config.tactile_key], dtype=np.float32)
            if requires_force else None
        )
        diff = (
            np.asarray(tactile_payload[config.tactile_diff_key])
            if requires_diff else None
        )
        diff_timestamp = (
            np.asarray(
                tactile_payload[config.tactile_diff_timestamp_key], dtype=np.int64
            )
            if requires_diff else None
        )
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
            raise ValueError("RGB payload must be uint8 [H,W,3]")
        if state.shape != (JOINT_COUNT,) or not np.isfinite(state).all():
            raise ValueError(f"Revo state must be finite [{JOINT_COUNT}]")
        if requires_force and (
            touch.shape != (5, 6) or not np.isfinite(touch).all()
        ):
            raise ValueError("Revo tactile feature must be finite [5,6]")
        force_finger_timestamp = None
        if requires_force:
            if "force6d_finger_timestamp_ns" not in tactile_payload:
                if not config.synthetic_fixture:
                    raise KeyError("Force6D payload lacks per-finger timestamps")
                force_finger_timestamp = np.full(5, int(tactile_ref["capture_timestamp_ns"]), np.int64)
            else:
                force_finger_timestamp = np.asarray(
                    tactile_payload["force6d_finger_timestamp_ns"], dtype=np.int64
                )
            if force_finger_timestamp.shape != (5,):
                raise ValueError("force6d_finger_timestamp_ns must have shape [5]")
            if np.any(force_finger_timestamp > receipt.decision_timestamp_ns):
                raise ValueError("Force6D per-finger timestamp occurred after decision")
            if (
                int(np.max(force_finger_timestamp) - np.min(force_finger_timestamp))
                > max_tactile_inter_finger_skew_ns
            ):
                raise ValueError("Force6D per-finger timestamp skew exceeds manifest budget")
        if requires_diff:
            if diff.shape != (5, 240, 240) or diff.dtype != np.uint8:
                raise ValueError("Revo DIFF tactile must be uint8 [5,240,240]")
            if diff_timestamp.shape != (5,):
                raise ValueError("Revo DIFF timestamps must have shape [5]")
            if np.any(diff_timestamp > receipt.decision_timestamp_ns):
                raise ValueError("DIFF observation occurred after controller decision")
            if np.any(receipt.decision_timestamp_ns - diff_timestamp > config.max_observation_age_ns):
                raise ValueError("DIFF observation is stale at controller decision")

        camera_index_row = stream_indices[config.camera_stream][int(camera_ref["sequence"])]
        camera_header = camera_index_row["header"]
        camera_capture_ns = int(camera_ref["capture_timestamp_ns"])
        camera_receive_ns = int(camera_header["receive_timestamp_ns"])
        state_capture_ns = int(state_ref["capture_timestamp_ns"])
        touch_capture_ns = int(tactile_ref["capture_timestamp_ns"])
        for name, observed_ns in (
            ("camera", camera_capture_ns),
            ("state", state_capture_ns),
            ("touch", touch_capture_ns),
        ):
            if receipt.decision_timestamp_ns - observed_ns > config.max_observation_age_ns:
                raise ValueError(f"{name} observation is stale at controller decision")

        timestamps.append(timestamp_ns)
        camera_capture_timestamps.append(camera_capture_ns)
        camera_receive_timestamps.append(camera_receive_ns)
        state_timestamps.append(state_capture_ns)
        if requires_force:
            touch_timestamps.append(touch_capture_ns)
            force_finger_timestamps.append(force_finger_timestamp.copy())
            tactile_histories.append(np.stack(history_payloads).astype(np.float32))
            tactile_history_timestamps.append(history_timestamps)
            tactile_history_sequences.append(history_sequences)
        if requires_diff:
            diff_timestamps.append(diff_timestamp.copy())
        action_decision_timestamps.append(receipt.decision_timestamp_ns)
        action_write_timestamps.append(receipt.write_timestamp_ns)
        controller_sequences.append(receipt.controller_sequence)
        request_id_hashes.append(
            hashlib.sha256(receipt.request_id.encode("utf-8")).hexdigest()
        )
        annotation = annotation_by_index[index]
        policy_loss_eligible.append(bool(annotation["policy_loss_eligible"]))
        phases.append(str(annotation["phase"]))
        states.append(state.copy())
        actions.append(receipt.exact_sent_target.copy())
        if requires_force:
            tactile.append(touch.copy())
        if requires_diff:
            tactile_diff.append(diff.copy())
        images.append(rgb.copy())

    if len(timestamps) < 2:
        raise ValueError(
            "fewer than two exportable anchors remain after the 16-sample "
            "native tactile warm-up gate"
        )

    destination_root = Path(output_root)
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / episode_id
    temporary = destination_root / f".{episode_id}.inprogress"
    if destination.exists() or temporary.exists():
        raise FileExistsError(f"derived episode already exists: {destination}")
    rgb_root = temporary / "rgb"
    rgb_root.mkdir(parents=True)
    image_paths = []
    for index, rgb in enumerate(images):
        relative = Path("rgb") / f"frame_{index:06d}.png"
        Image.fromarray(rgb, mode="RGB").save(temporary / relative, format="PNG")
        image_paths.append(relative.as_posix())

    meta = {
        "schema_version": "revo3-episode-v1",
        "episode_id": episode_id,
        "task_id": str(metadata["task_id"]),
        "task_version": int(metadata["task_version"]),
        "task": task,
        "object_id": str(metadata["object_id"]),
        "object_instance": str(metadata["object_instance"]),
        "operator": str(metadata["operator"]),
        "collection_day": str(metadata["collection_day"]),
        "grasp_primitive": str(metadata["grasp_primitive"]),
        "instruction": instruction,
        "instruction_sha256": declared_instruction_sha256,
        "instruction_source": instruction_source,
        "planner_revision": planner_revision,
        "planner_output_sha256": planner_output_sha256,
        "fps": 30,
        "time_grid_hz": 30,
        "timestamp_tolerance_ns": int(config.timestamp_tolerance_ns),
        "resampled_30hz": True,
        "timestamp_alignment_verified": True,
        "max_command_latency_ns": max_command_latency_ns,
        "max_tactile_inter_finger_skew_ns": max_tactile_inter_finger_skew_ns,
        "joint_order_hash": JOINT_ORDER_HASH,
        "tactile_num_fingers": 5,
        "tactile_profile": config.tactile_profile,
        "checkpoint_family_id": str(metadata["checkpoint_family_id"]),
        "normalization_family_id": str(metadata["normalization_family_id"]),
        "capability_manifest_sha256": str(metadata["capability_manifest_sha256"]),
        "camera_profile_id": str(metadata["camera_profile_id"]),
        "camera_calibration_sha256": str(metadata["camera_calibration_sha256"]),
        "action_label_source": "controller_target",
        "action_semantics": ACTION_SEMANTICS,
        "contains_cair_residual": False,
        "image_paths": image_paths,
        "contains_emg": False,
        "synthetic_fixture": bool(config.synthetic_fixture),
    }
    if set(meta) != META_ALLOWLIST:
        raise AssertionError("internal error: Revo meta allowlist changed")
    with (temporary / "meta.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(meta, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    frames = {
        "timestamp_ns": np.asarray(timestamps, dtype=np.int64),
        "camera_capture_timestamp_ns": np.asarray(
            camera_capture_timestamps, dtype=np.int64
        ),
        "camera_receive_timestamp_ns": np.asarray(
            camera_receive_timestamps, dtype=np.int64
        ),
        "state_timestamp_ns": np.asarray(state_timestamps, dtype=np.int64),
        "action_decision_timestamp_ns": np.asarray(
            action_decision_timestamps, dtype=np.int64
        ),
        "action_write_timestamp_ns": np.asarray(
            action_write_timestamps, dtype=np.int64
        ),
        "controller_sequence": np.asarray(controller_sequences, dtype=np.int64),
        "request_id_hash": np.asarray(request_id_hashes, dtype="U64"),
        "policy_loss_eligible": np.asarray(policy_loss_eligible, dtype=np.bool_),
        "phase": np.asarray(phases, dtype="U16"),
        "state_rad": np.stack(states).astype(np.float32),
        "action_target_rad": np.stack(actions).astype(np.float32),
    }
    if requires_force:
        frames["touch_timestamp_ns"] = np.asarray(touch_timestamps, dtype=np.int64)
        frames["tactile_features"] = np.stack(tactile).astype(np.float32)
        frames["force6d_finger_timestamp_ns"] = np.stack(
            force_finger_timestamps
        ).astype(np.int64)
        frames["tactile_history_f6"] = np.stack(tactile_histories).astype(np.float32)
        frames["tactile_history_timestamp_ns"] = np.stack(
            tactile_history_timestamps
        ).astype(np.int64)
        frames["tactile_history_sequence"] = np.stack(
            tactile_history_sequences
        ).astype(np.int64)
    if requires_diff:
        frames["tactile_diff_timestamp_ns"] = np.stack(diff_timestamps).astype(np.int64)
        frames["tactile_diff"] = np.stack(tactile_diff).astype(np.uint8)
    if not set(frames).issubset(FRAME_ALLOWLIST):
        raise AssertionError("internal error: Revo frame allowlist changed")
    with (temporary / "frames.npz").open("wb") as handle:
        np.savez_compressed(handle, **frames)

    # Reuse the production loader as the compatibility gate before publishing.
    RevoEpisode.load(temporary)
    os.replace(temporary, destination)
    return destination
