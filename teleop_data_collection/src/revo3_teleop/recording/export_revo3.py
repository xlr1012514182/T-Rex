"""Allowlist projection from a committed master episode to Revo3 VLA data."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image

from revo3_v1.data.episode import RevoEpisode, SUPPORTED_TASKS
from revo3_v1.revo.contracts import JOINT_COUNT, JOINT_ORDER_HASH

from revo3_teleop.contracts import CommandReceipt
from revo3_teleop.recording.recorder import load_native_payload


META_ALLOWLIST = frozenset(
    {
        "schema_version",
        "episode_id",
        "task",
        "instruction",
        "fps",
        "joint_order_hash",
        "tactile_num_fingers",
        "action_label_source",
        "image_paths",
        "contains_emg",
        "synthetic_fixture",
    }
)
FRAME_ALLOWLIST = frozenset(
    {"timestamp_ns", "state_rad", "action_target_rad", "tactile_features"}
)


@dataclass(frozen=True)
class Revo3ExportConfig:
    camera_stream: str = "camera"
    state_stream: str = "revo_state"
    tactile_stream: str = "tactile"
    camera_key: str = "rgb"
    state_key: str = "q_rad"
    tactile_key: str = "features"
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

    anchors = _read_jsonl(source / "anchors_30hz.jsonl")
    if len(anchors) < 2:
        raise ValueError("Revo3 export requires at least two 30 Hz anchors")
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

    timestamps: list[int] = []
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    tactile: list[np.ndarray] = []
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

        camera_ref = _stream_reference(anchor, config.camera_stream)
        state_ref = _stream_reference(anchor, config.state_stream)
        tactile_ref = _stream_reference(anchor, config.tactile_stream)
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
        if config.tactile_key not in tactile_payload:
            raise KeyError(f"tactile payload lacks {config.tactile_key!r}")
        rgb = np.asarray(camera_payload[config.camera_key])
        state = np.asarray(state_payload[config.state_key], dtype=np.float32)
        touch = np.asarray(tactile_payload[config.tactile_key], dtype=np.float32)
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
            raise ValueError("RGB payload must be uint8 [H,W,3]")
        if state.shape != (JOINT_COUNT,) or not np.isfinite(state).all():
            raise ValueError(f"Revo state must be finite [{JOINT_COUNT}]")
        if touch.shape != (5, 6) or not np.isfinite(touch).all():
            raise ValueError("Revo tactile feature must be finite [5,6]")

        timestamps.append(timestamp_ns)
        states.append(state.copy())
        actions.append(receipt.exact_sent_target.copy())
        tactile.append(touch.copy())
        images.append(rgb.copy())

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
        "task": task,
        "instruction": instruction,
        "fps": 30,
        "joint_order_hash": JOINT_ORDER_HASH,
        "tactile_num_fingers": 5,
        "action_label_source": "controller_target",
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
        "state_rad": np.stack(states).astype(np.float32),
        "action_target_rad": np.stack(actions).astype(np.float32),
        "tactile_features": np.stack(tactile).astype(np.float32),
    }
    if set(frames) != FRAME_ALLOWLIST:
        raise AssertionError("internal error: Revo frame allowlist changed")
    with (temporary / "frames.npz").open("wb") as handle:
        np.savez_compressed(handle, **frames)

    # Reuse the production loader as the compatibility gate before publishing.
    RevoEpisode.load(temporary)
    os.replace(temporary, destination)
    return destination
