"""Crash-contained native-rate episode recorder.

The recorder deliberately writes every source sample at its native rate.  A
separate 30 Hz index stores causal references into those streams and the exact
hand controller command produced by that control cycle.  It never merges raw
EMG, glove, arm, camera, or hand payloads into a training row.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import asdict
from enum import Enum
import json
import os
from pathlib import Path
from typing import Iterable, Mapping, Optional

import h5py
import numpy as np

from revo3_v1.revo.contracts import JOINT_COUNT, JOINT_ORDER_HASH

from revo3_teleop.contracts import (
    CausalAnchor,
    CommandReceipt,
    NativeSample,
    StreamReference,
)


NANOSECONDS_PER_SECOND = 1_000_000_000


class RecorderState(str, Enum):
    NEW = "new"
    RECORDING = "recording"
    COMMITTED = "committed"
    ABORTED = "aborted"


def _safe_name(value: str, *, kind: str) -> str:
    name = str(value).strip()
    if not name or not all(character.isalnum() or character in "-_" for character in name):
        raise ValueError(f"{kind} must contain only letters, digits, '-' and '_'")
    return name


def _append_jsonl(path: Path, value: Mapping[str, object]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(value), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(dict(value), handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _header_json(sample: NativeSample) -> dict[str, object]:
    return asdict(sample.header)


def _receipt_json(receipt: CommandReceipt) -> dict[str, object]:
    def values(array: Optional[np.ndarray]) -> object:
        return None if array is None else array.tolist()

    return {
        "request_id": receipt.request_id,
        "component": receipt.component,
        "accepted": receipt.accepted,
        "requested_target": values(receipt.requested_target),
        "authorized_target": values(receipt.authorized_target),
        "exact_sent_target": values(receipt.exact_sent_target),
        "decision_timestamp_ns": receipt.decision_timestamp_ns,
        "write_timestamp_ns": receipt.write_timestamp_ns,
        "controller_sequence": receipt.controller_sequence,
        "clipped": receipt.clipped,
        "reason": receipt.reason,
        "unit": receipt.unit,
        "joint_order_hash": receipt.joint_order_hash,
    }


def _anchor_json(anchor: CausalAnchor) -> dict[str, object]:
    return {
        "anchor_index": anchor.anchor_index,
        "timestamp_ns": anchor.timestamp_ns,
        "hand_command_request_id": anchor.hand_command_request_id,
        "streams": {
            name: asdict(reference) for name, reference in sorted(anchor.streams.items())
        },
    }


def _chunk_rows(shape: tuple[int, ...], dtype: np.dtype, shard_rows: int) -> int:
    """Choose roughly 1 MiB payload chunks, bounded by a shard."""

    elements = max(1, int(np.prod(shape, dtype=np.int64)))
    row_bytes = max(1, elements * int(dtype.itemsize))
    return max(1, min(shard_rows, 1_048_576 // row_bytes))


class _StreamShardWriter:
    """Append fixed-schema samples to bounded, crash-auditable HDF5 shards."""

    _HEADER_DTYPES = {
        "sequence": np.dtype("int64"),
        "capture_timestamp_ns": np.dtype("int64"),
        "receive_timestamp_ns": np.dtype("int64"),
        "device_timestamp_ns": np.dtype("int64"),
        "valid": np.dtype("uint8"),
        "dropped_since_previous": np.dtype("int64"),
    }

    def __init__(
        self,
        episode_root: Path,
        *,
        stream: str,
        source_id: str,
        clock_domain: str,
        schema: Mapping[str, tuple[str, tuple[int, ...]]],
        shard_rows: int,
        compression: Optional[str],
    ) -> None:
        self.episode_root = episode_root
        self.stream = stream
        self.source_id = source_id
        self.clock_domain = clock_domain
        self.schema = dict(schema)
        self.shard_rows = int(shard_rows)
        self.compression = compression
        self._shard_index = -1
        self._row_index = 0
        self._file: Optional[h5py.File] = None
        self._relative_path: Optional[Path] = None

    def _open_next(self) -> None:
        self._shard_index += 1
        self._row_index = 0
        relative = (
            Path("streams")
            / self.stream
            / "shards"
            / f"shard_{self._shard_index:06d}.h5"
        )
        absolute = self.episode_root / relative
        absolute.parent.mkdir(parents=True, exist_ok=True)
        if absolute.exists():
            raise FileExistsError(f"stream shard already exists: {absolute}")
        handle = h5py.File(absolute, "x", libver="latest")
        handle.attrs["schema_version"] = "revo3-native-stream-shard-v1"
        handle.attrs["stream"] = self.stream
        handle.attrs["source_id"] = self.source_id
        handle.attrs["clock_domain"] = self.clock_domain
        handle.attrs["compression"] = "none" if self.compression is None else self.compression
        handle.attrs["payload_schema_json"] = json.dumps(
            {
                key: {"dtype": dtype, "shape": list(shape)}
                for key, (dtype, shape) in sorted(self.schema.items())
            },
            sort_keys=True,
        )
        handle.attrs["committed_rows"] = 0
        handle.attrs["closed_cleanly"] = False
        header_group = handle.create_group("header")
        header_chunks = (min(self.shard_rows, 1024),)
        for name, dtype in self._HEADER_DTYPES.items():
            header_group.create_dataset(
                name,
                shape=(0,),
                maxshape=(None,),
                chunks=header_chunks,
                dtype=dtype,
                compression=self.compression,
            )
        payload_group = handle.create_group("payload")
        for name, (dtype_text, shape) in sorted(self.schema.items()):
            dtype = np.dtype(dtype_text)
            rows = _chunk_rows(shape, dtype, self.shard_rows)
            payload_group.create_dataset(
                name,
                shape=(0, *shape),
                maxshape=(None, *shape),
                chunks=(rows, *shape),
                dtype=dtype,
                compression=self.compression,
            )
        handle.flush()
        self._file = handle
        self._relative_path = relative

    def append(self, sample: NativeSample) -> tuple[Path, int]:
        if self._file is None:
            self._open_next()
        elif self._row_index >= self.shard_rows:
            self.close()
            self._open_next()
        assert self._file is not None and self._relative_path is not None
        row = self._row_index
        new_length = row + 1
        header_values = {
            "sequence": sample.header.sequence,
            "capture_timestamp_ns": sample.header.capture_timestamp_ns,
            "receive_timestamp_ns": sample.header.receive_timestamp_ns,
            "device_timestamp_ns": (
                -1 if sample.header.device_timestamp_ns is None else sample.header.device_timestamp_ns
            ),
            "valid": int(sample.header.valid),
            "dropped_since_previous": sample.header.dropped_since_previous,
        }
        for name, value in header_values.items():
            dataset = self._file["header"][name]
            dataset.resize((new_length,))
            dataset[row] = value
        for name, value in sample.payload.items():
            dataset = self._file["payload"][name]
            dataset.resize((new_length, *value.shape))
            dataset[row] = value
        # First make all row bytes durable, then publish the row count.  A
        # crash before the second flush leaves only an uncommitted tail.
        self._file.flush()
        self._file.attrs.modify("committed_rows", new_length)
        self._file.flush()
        self._row_index = new_length
        return self._relative_path, row

    def close(self) -> None:
        if self._file is None:
            return
        self._file.attrs.modify("closed_cleanly", True)
        self._file.flush()
        self._file.close()
        self._file = None
        self._relative_path = None


def load_native_payload(
    episode_root: str | Path,
    stream: str,
    reference_or_index_row: Mapping[str, object],
    *,
    require_clean_close: bool = True,
) -> dict[str, np.ndarray]:
    """Load one committed native-rate payload through the stable row contract.

    ``reference_or_index_row`` may be either a serialized ``StreamReference``
    from ``anchors_30hz.jsonl`` or a row from a stream's ``index.jsonl``.  The
    function owns all HDF5 layout knowledge so downstream EMG exporters and
    audit tools need only the public path+row reference.
    """

    stream_name = _safe_name(stream, kind="stream")
    row = dict(reference_or_index_row)
    nested_header = row.get("header")
    if nested_header is not None:
        if not isinstance(nested_header, Mapping):
            raise ValueError("native index row header is malformed")
        reference = {
            "relative_path": row.get("relative_path"),
            "row_index": row.get("row_index"),
            "source_id": nested_header.get("source_id"),
            "clock_domain": nested_header.get("clock_domain"),
            "sequence": nested_header.get("sequence"),
            "capture_timestamp_ns": nested_header.get("capture_timestamp_ns"),
        }
    else:
        reference = row

    relative = Path(str(reference.get("relative_path", "")))
    if relative.is_absolute():
        raise ValueError("native stream reference must be episode-relative")
    if tuple(relative.parts[:3]) != ("streams", stream_name, "shards"):
        raise ValueError(f"native reference does not belong to stream {stream_name!r}")
    if relative.suffix.lower() != ".h5":
        raise ValueError("native reference must point to an HDF5 shard")
    root = Path(episode_root).resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("native stream reference escapes the episode") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    row_index = int(reference.get("row_index", -1))
    if row_index < 0:
        raise ValueError("native reference requires a non-negative row_index")

    with h5py.File(path, "r", swmr=True) as handle:
        if handle.attrs.get("schema_version") != "revo3-native-stream-shard-v1":
            raise ValueError("unsupported native HDF5 shard schema")
        expected_identity = (
            (str(handle.attrs.get("stream")), stream_name, "stream"),
            (str(handle.attrs.get("source_id")), str(reference.get("source_id")), "source_id"),
            (
                str(handle.attrs.get("clock_domain")),
                str(reference.get("clock_domain")),
                "clock_domain",
            ),
        )
        for observed, expected, name in expected_identity:
            if observed != expected:
                raise ValueError(f"native HDF5 {name} identity mismatch")
        committed_rows = int(handle.attrs.get("committed_rows", -1))
        if row_index >= committed_rows:
            raise ValueError("native reference points to an uncommitted HDF5 tail row")
        if require_clean_close and not bool(handle.attrs.get("closed_cleanly", False)):
            raise ValueError("native shard was not closed cleanly")
        if int(handle["header/sequence"][row_index]) != int(reference.get("sequence", -1)):
            raise ValueError("native HDF5 sequence/reference mismatch")
        if int(handle["header/capture_timestamp_ns"][row_index]) != int(
            reference.get("capture_timestamp_ns", -1)
        ):
            raise ValueError("native HDF5 capture timestamp/reference mismatch")
        schema = json.loads(str(handle.attrs["payload_schema_json"]))
        if set(schema) != set(handle["payload"]):
            raise ValueError("native HDF5 payload schema keys changed")
        result: dict[str, np.ndarray] = {}
        for key, description in schema.items():
            dataset = handle["payload"][key]
            expected_shape = tuple(int(value) for value in description["shape"])
            expected_dtype = np.dtype(description["dtype"])
            if dataset.shape[1:] != expected_shape or dataset.dtype != expected_dtype:
                raise ValueError(f"native HDF5 payload schema changed for {key!r}")
            result[key] = np.asarray(dataset[row_index]).copy()
        return result


class EpisodeRecorder:
    """Persist one collection episode and atomically publish it when complete."""

    def __init__(
        self,
        root: str | Path,
        *,
        episode_id: str,
        epoch_ns: int,
        metadata: Optional[Mapping[str, object]] = None,
        anchor_hz: int = 30,
        alignment_buffer_size: int = 4096,
        stream_shard_rows: int = 2048,
        hdf5_compression: Optional[str] = "lzf",
    ) -> None:
        self.root = Path(root)
        self.episode_id = _safe_name(episode_id, kind="episode_id")
        self.epoch_ns = int(epoch_ns)
        self.anchor_hz = int(anchor_hz)
        if self.epoch_ns < 0:
            raise ValueError("epoch_ns must be non-negative")
        if self.anchor_hz != 30:
            raise ValueError("the Revo3 collection projection fixes anchor_hz=30")
        self.alignment_buffer_size = int(alignment_buffer_size)
        if self.alignment_buffer_size < 32:
            raise ValueError("alignment_buffer_size must be at least 32")
        self.stream_shard_rows = int(stream_shard_rows)
        if self.stream_shard_rows < 2:
            raise ValueError("stream_shard_rows must be at least two")
        if hdf5_compression not in {None, "lzf"}:
            raise ValueError("hdf5_compression must be None or 'lzf'")
        self.hdf5_compression = hdf5_compression
        # Round-trip now so invalid/non-portable manifest metadata fails before
        # any directory is created.
        self.metadata = json.loads(json.dumps(dict(metadata or {}), ensure_ascii=False))
        self.state = RecorderState.NEW
        self._path = self.root / ".inprogress" / self.episode_id
        self._entries: dict[str, list[dict[str, object]]] = {}
        self._capture_by_stream: dict[str, list[int]] = {}
        self._stream_counts: dict[str, int] = {}
        self._last_by_stream: dict[str, tuple[int, int]] = {}
        self._identity_by_stream: dict[str, tuple[str, str]] = {}
        self._schema_by_stream: dict[str, dict[str, tuple[str, tuple[int, ...]]]] = {}
        self._writers: dict[str, _StreamShardWriter] = {}
        self._write_failed = False
        self._receipts: dict[str, CommandReceipt] = {}
        self._last_command_by_component: dict[str, tuple[int, int, int]] = {}
        self._anchored_command_ids: set[str] = set()
        self._next_anchor_index = 0

    @property
    def path(self) -> Path:
        return self._path

    def anchor_timestamp_ns(self, anchor_index: int) -> int:
        index = int(anchor_index)
        if index < 0:
            raise ValueError("anchor_index must be non-negative")
        # Absolute-index calculation prevents accumulated 30 Hz rounding drift.
        return self.epoch_ns + (NANOSECONDS_PER_SECOND * index) // self.anchor_hz

    def _require_recording(self) -> None:
        if self.state != RecorderState.RECORDING:
            raise RuntimeError(f"recorder is not recording (state={self.state.value})")
        if self._write_failed:
            raise RuntimeError("recorder is faulted after a native stream write failure")

    def _manifest(self, lifecycle: str) -> dict[str, object]:
        return {
            "schema_version": "revo3-teleop-master-v1",
            "episode_id": self.episode_id,
            "lifecycle": lifecycle,
            "epoch_ns": self.epoch_ns,
            "anchor_hz": self.anchor_hz,
            "alignment_buffer_size": self.alignment_buffer_size,
            "native_storage": {
                "schema_version": "revo3-native-stream-shard-v1",
                "format": "hdf5",
                "shard_rows": self.stream_shard_rows,
                "compression": (
                    "none" if self.hdf5_compression is None else self.hdf5_compression
                ),
                "crash_boundary": (
                    "row payload flush -> committed_rows flush -> index.jsonl fsync"
                ),
            },
            "metadata": self.metadata,
            "stream_counts": {
                name: count for name, count in sorted(self._stream_counts.items())
            },
            "command_receipts": len(self._receipts),
            "anchors": self._next_anchor_index,
        }

    def _close_writers(self) -> None:
        for writer in self._writers.values():
            writer.close()

    def start(self) -> Path:
        if self.state != RecorderState.NEW:
            raise RuntimeError("start is valid only for a new recorder")
        if self._path.exists() or (self.root / "committed" / self.episode_id).exists():
            raise FileExistsError(f"episode already exists: {self.episode_id}")
        (self._path / "streams").mkdir(parents=True, exist_ok=False)
        _atomic_json(self._path / "manifest.json", self._manifest("recording"))
        self.state = RecorderState.RECORDING
        return self._path

    def append(self, stream: str, sample: NativeSample) -> Path:
        """Write one native sample immediately and enforce per-stream order."""

        self._require_recording()
        stream_name = _safe_name(stream, kind="stream")
        sequence = int(sample.header.sequence)
        capture_ns = int(sample.header.capture_timestamp_ns)
        identity = (sample.header.source_id, sample.header.clock_domain)
        previous_identity = self._identity_by_stream.get(stream_name)
        if previous_identity is not None and identity != previous_identity:
            raise ValueError(f"{stream_name} source_id/clock_domain changed within episode")
        schema = {
            key: (str(value.dtype), tuple(int(dim) for dim in value.shape))
            for key, value in sample.payload.items()
        }
        previous_schema = self._schema_by_stream.get(stream_name)
        if previous_schema is not None and schema != previous_schema:
            raise ValueError(f"{stream_name} payload schema changed within episode")
        previous = self._last_by_stream.get(stream_name)
        if previous is not None:
            previous_sequence, previous_capture = previous
            if sequence <= previous_sequence:
                raise ValueError(f"{stream_name} sequence must be strictly increasing")
            if capture_ns <= previous_capture:
                raise ValueError(f"{stream_name} capture timestamp must be strictly increasing")

        writer = self._writers.get(stream_name)
        if writer is None:
            writer = _StreamShardWriter(
                self._path,
                stream=stream_name,
                source_id=sample.header.source_id,
                clock_domain=sample.header.clock_domain,
                schema=schema,
                shard_rows=self.stream_shard_rows,
                compression=self.hdf5_compression,
            )
            self._writers[stream_name] = writer
        try:
            relative_path, row_index = writer.append(sample)
        except Exception:
            self._write_failed = True
            raise
        absolute_path = self._path / relative_path
        index_row = {
            "header": _header_json(sample),
            "relative_path": relative_path.as_posix(),
            "row_index": row_index,
            "payload": {
                key: {"dtype": str(value.dtype), "shape": list(value.shape)}
                for key, value in sample.payload.items()
            },
        }
        try:
            _append_jsonl(self._path / "streams" / stream_name / "index.jsonl", index_row)
        except Exception:
            # The HDF5 row may be durable, but without its fsynced index it is
            # intentionally an orphan and cannot be referenced or exported.
            self._write_failed = True
            raise
        entries = self._entries.setdefault(stream_name, [])
        captures = self._capture_by_stream.setdefault(stream_name, [])
        entries.append(index_row)
        captures.append(capture_ns)
        # Alignment is an online operation.  Keep a generous bounded history
        # while the complete native stream remains on disk.  Trim in batches
        # to avoid per-sample O(n) list shifts during long collections.
        if len(entries) > 2 * self.alignment_buffer_size:
            self._entries[stream_name] = entries[-self.alignment_buffer_size :]
            self._capture_by_stream[stream_name] = captures[-self.alignment_buffer_size :]
        self._stream_counts[stream_name] = self._stream_counts.get(stream_name, 0) + 1
        self._identity_by_stream.setdefault(stream_name, identity)
        self._schema_by_stream.setdefault(stream_name, schema)
        self._last_by_stream[stream_name] = (sequence, capture_ns)
        return absolute_path

    def record_command(self, receipt: CommandReceipt) -> None:
        """Persist a controller receipt; rejected receipts remain diagnostics."""

        self._require_recording()
        if receipt.request_id in self._receipts:
            raise ValueError(f"duplicate command request_id: {receipt.request_id}")
        if receipt.accepted:
            assert receipt.controller_sequence is not None
            assert receipt.write_timestamp_ns is not None
            previous = self._last_command_by_component.get(receipt.component)
            current = (
                receipt.controller_sequence,
                receipt.decision_timestamp_ns,
                receipt.write_timestamp_ns,
            )
            if previous is not None:
                sequence, decision_ns, write_ns = current
                previous_sequence, previous_decision_ns, previous_write_ns = previous
                if sequence <= previous_sequence or decision_ns <= previous_decision_ns:
                    raise ValueError(
                        f"{receipt.component} controller sequence/decision time must increase"
                    )
                # Some vendor/Windows monotonic clocks expose coarser ticks
                # than their nanosecond integer type.  Sequence disambiguates
                # writes sharing one clock tick; actual reversal is forbidden.
                if write_ns < previous_write_ns:
                    raise ValueError(f"{receipt.component} controller write time reversed")
            self._last_command_by_component[receipt.component] = current
        self._receipts[receipt.request_id] = receipt
        _append_jsonl(self._path / "command_receipts.jsonl", _receipt_json(receipt))

    def record_anchor(
        self,
        *,
        anchor_index: int,
        streams: Iterable[str],
        hand_command_request_id: str,
        max_age_ns: Optional[Mapping[str, int]] = None,
    ) -> CausalAnchor:
        """Record one exact 30 Hz control anchor using latest-not-after samples."""

        self._require_recording()
        index = int(anchor_index)
        if index != self._next_anchor_index:
            raise ValueError(
                f"anchor_index must be contiguous; expected {self._next_anchor_index}, got {index}"
            )
        anchor_ns = self.anchor_timestamp_ns(index)
        receipt = self._receipts.get(hand_command_request_id)
        if receipt is None:
            raise KeyError(f"unknown command receipt: {hand_command_request_id}")
        if receipt.component != "revo_hand":
            raise ValueError("anchor supervision must reference a revo_hand receipt")
        if not receipt.accepted or receipt.exact_sent_target is None:
            raise ValueError("anchor supervision requires an accepted exact_sent_target")
        if hand_command_request_id in self._anchored_command_ids:
            raise ValueError("one controller receipt cannot supervise two anchors")
        if receipt.unit != "rad" or receipt.joint_order_hash != JOINT_ORDER_HASH:
            raise ValueError("anchor Revo command unit/joint order does not match the VLA contract")
        if receipt.exact_sent_target.shape != (JOINT_COUNT,):
            raise ValueError(f"anchor exact_sent_target must have shape ({JOINT_COUNT},)")
        if receipt.decision_timestamp_ns < anchor_ns:
            raise ValueError("hand command decision cannot precede its control anchor")

        selected: dict[str, StreamReference] = {}
        requested_streams = tuple(dict.fromkeys(_safe_name(item, kind="stream") for item in streams))
        if not requested_streams:
            raise ValueError("at least one stream is required for an anchor")
        for stream in requested_streams:
            entries = self._entries.get(stream, [])
            captures = self._capture_by_stream.get(stream, [])
            selected_index = bisect_right(captures, anchor_ns) - 1
            # Capture time alone is insufficient for an online-causal
            # observation.  In particular, a derived camera frame retains the
            # physical exposure timestamp while its receive timestamp records
            # when rectification actually completed.  Walk back past any
            # sample that was not yet available when the controller made this
            # decision instead of leaking post-decision preprocessing into the
            # training observation.
            while selected_index >= 0:
                candidate_header = entries[selected_index]["header"]
                candidate_receive_ns = int(candidate_header["receive_timestamp_ns"])
                if candidate_receive_ns <= receipt.decision_timestamp_ns:
                    break
                selected_index -= 1
            if selected_index < 0:
                raise ValueError(
                    f"no causally available sample for stream {stream!r} at anchor {index}"
                )
            entry = entries[selected_index]
            header = entry["header"]
            if not bool(header["valid"]):
                raise ValueError(f"latest causal sample for stream {stream!r} is invalid")
            capture_ns = int(header["capture_timestamp_ns"])
            age_ns = anchor_ns - capture_ns
            limit = None if max_age_ns is None else max_age_ns.get(stream)
            if limit is not None and age_ns > int(limit):
                raise ValueError(f"causal sample for stream {stream!r} is stale")
            if receipt.decision_timestamp_ns < capture_ns:
                raise ValueError("controller decision predates a selected observation")
            if receipt.decision_timestamp_ns < int(header["receive_timestamp_ns"]):
                raise ValueError(
                    "controller decision predates selected observation availability"
                )
            selected[stream] = StreamReference(
                stream=stream,
                source_id=str(header["source_id"]),
                clock_domain=str(header["clock_domain"]),
                sequence=int(header["sequence"]),
                capture_timestamp_ns=capture_ns,
                age_ns=age_ns,
                relative_path=str(entry["relative_path"]),
                row_index=int(entry["row_index"]),
            )

        anchor = CausalAnchor(
            anchor_index=index,
            timestamp_ns=anchor_ns,
            streams=selected,
            hand_command_request_id=hand_command_request_id,
        )
        _append_jsonl(self._path / "anchors_30hz.jsonl", _anchor_json(anchor))
        self._anchored_command_ids.add(hand_command_request_id)
        self._next_anchor_index += 1
        return anchor

    def commit(self) -> Path:
        """Atomically publish the episode by moving it out of ``.inprogress``."""

        self._require_recording()
        destination = self.root / "committed" / self.episode_id
        if destination.exists():
            raise FileExistsError(f"committed episode already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._close_writers()
        _atomic_json(self._path / "manifest.json", self._manifest("committed"))
        os.replace(self._path, destination)
        self._path = destination
        self.state = RecorderState.COMMITTED
        return destination

    def abort(self, reason: str) -> Path:
        """Quarantine an incomplete episode; it is never visible to exporters."""

        if self.state != RecorderState.RECORDING:
            raise RuntimeError(f"recorder is not recording (state={self.state.value})")
        text = str(reason).strip()
        if not text:
            raise ValueError("abort reason must be non-empty")
        destination = self.root / "quarantine" / f"{self.episode_id}.aborted"
        if destination.exists():
            raise FileExistsError(f"quarantined episode already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._close_writers()
        manifest = self._manifest("aborted")
        manifest["abort_reason"] = text
        _atomic_json(self._path / "manifest.json", manifest)
        os.replace(self._path, destination)
        self._path = destination
        self.state = RecorderState.ABORTED
        return destination
