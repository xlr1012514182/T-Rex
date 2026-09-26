"""Short per-user/day EMG calibration with strict lineage binding."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .data import EMGWindowDataset, load_manifest
from .model import load_emg_checkpoint
from .preprocessing import (
    EmgPreprocessingProfile,
    normalization_sha256,
    validate_npz_profile,
)


CALIBRATION_SCHEMA = "revo3-emg-calibration-v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _channel_order_hash(channel_order: Sequence[str]) -> str:
    payload = json.dumps(list(channel_order), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CalibrationDatasetIdentity:
    subject_id: str
    day_id: str
    session_ids: tuple[str, ...]
    duration_s: float
    label_counts: Mapping[str, int]


def validate_calibration_manifest(
    path: str | Path,
    *,
    label_names: Sequence[str],
    expected_subject_id: str,
    expected_day_id: str,
    base_training_sessions: Sequence[str],
    min_duration_s: float = 300.0,
    max_duration_s: float = 600.0,
) -> CalibrationDatasetIdentity:
    rows = load_manifest(path)
    subjects = {str(row.get("subject_id", "")) for row in rows}
    days = {str(row.get("day_id", "")) for row in rows}
    sessions = {str(row.get("session_id", "")) for row in rows}
    if subjects != {expected_subject_id}:
        raise ValueError("Calibration manifest must contain exactly the requested subject")
    if days != {expected_day_id}:
        raise ValueError("Calibration manifest must contain exactly the requested day")
    if "" in sessions:
        raise ValueError("Calibration session_id cannot be empty")
    overlap = sessions.intersection(str(value) for value in base_training_sessions)
    if overlap:
        raise ValueError(f"Calibration sessions overlap base training lineage: {sorted(overlap)}")
    by_session: dict[str, tuple[int, int]] = {}
    counts = {str(label): 0 for label in label_names}
    for row in rows:
        if "window_start_ns" not in row or "window_end_ns" not in row:
            raise ValueError("Calibration rows require window_start_ns/window_end_ns")
        start, end = int(row["window_start_ns"]), int(row["window_end_ns"])
        if end <= start:
            raise ValueError("Calibration window_end_ns must be after window_start_ns")
        session = str(row["session_id"])
        previous = by_session.get(session, (start, end))
        by_session[session] = (min(previous[0], start), max(previous[1], end))
        label = int(row["label"])
        if not 0 <= label < len(label_names):
            raise ValueError("Calibration label is outside the base checkpoint vocabulary")
        expected_name = str(label_names[label])
        supplied_name = str(row.get("label_name", expected_name)).upper()
        if supplied_name != expected_name:
            raise ValueError("Calibration label_name disagrees with numeric label")
        counts[expected_name] += 1
    if any(count < 2 for count in counts.values()):
        raise ValueError("Calibration requires at least two labeled windows per learned class")
    duration_s = sum(end - start for start, end in by_session.values()) / 1e9
    if not min_duration_s <= duration_s <= max_duration_s:
        raise ValueError(
            f"Calibration session span must be {min_duration_s:.0f}..{max_duration_s:.0f}s, got {duration_s:.1f}s"
        )
    return CalibrationDatasetIdentity(
        subject_id=expected_subject_id,
        day_id=expected_day_id,
        session_ids=tuple(sorted(sessions)),
        duration_s=duration_s,
        label_counts=counts,
    )


def _fit_temperature(logits, labels) -> float:
    import torch

    log_temperature = torch.zeros((), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50)

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits / log_temperature.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0).item())


def calibrate_emg_checkpoint(
    *,
    base_checkpoint: str | Path,
    windows_npz: str | Path,
    manifest: str | Path,
    output: str | Path,
    channel_order: Sequence[str],
    subject_id: str,
    day_id: str,
    method: str = "temperature",
    head_epochs: int = 30,
    head_lr: float = 1e-3,
    device: str = "cpu",
) -> Mapping[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    method = method.strip().lower()
    if method not in {"temperature", "prototype", "head"}:
        raise ValueError("method must be temperature, prototype, or head")
    checkpoint_path = Path(base_checkpoint).resolve()
    model, base_payload = load_emg_checkpoint(checkpoint_path, map_location=device)
    labels_map = base_payload.get("labels", {})
    label_names = tuple(str(labels_map.get(index, labels_map.get(str(index), ""))).upper() for index in range(model.config.output_channels))
    if any(not value for value in label_names):
        raise ValueError("Base checkpoint has an incomplete label vocabulary")
    if len(channel_order) != model.config.input_channels or len(set(channel_order)) != len(channel_order):
        raise ValueError("channel_order must uniquely name every base checkpoint input channel")
    metadata = base_payload.get("training_metadata", {})
    base_sessions = metadata.get("train_session_ids", metadata.get("training_session_ids"))
    if not isinstance(base_sessions, Sequence) or isinstance(base_sessions, (str, bytes)):
        raise ValueError("Base checkpoint lacks train_session_ids required for leakage audit")
    identity = validate_calibration_manifest(
        manifest,
        label_names=label_names,
        expected_subject_id=subject_id,
        expected_day_id=day_id,
        base_training_sessions=tuple(str(value) for value in base_sessions),
    )
    raw_profile = base_payload.get("preprocessing_profile")
    if not isinstance(raw_profile, Mapping):
        raise ValueError(
            "Base checkpoint is not bound to an EMG preprocessing profile; "
            "daily calibration refuses unprofiled fixture checkpoints"
        )
    profile = EmgPreprocessingProfile.from_mapping(raw_profile)
    profile.validate_stream(
        sample_rate_hz=profile.sample_rate_hz,
        channel_order=channel_order,
    )
    normalization = base_payload["normalization"]
    def _numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float32)

    normalized = {
        "mean": _numpy(normalization["mean"]),
        "std": _numpy(normalization["std"]),
    }
    profile.bind_normalization(normalized)
    with np.load(Path(windows_npz), allow_pickle=False) as archive:
        preprocessing_state = validate_npz_profile(archive, profile)
    filter_lineage = (
        preprocessing_state.provenance
        if preprocessing_state.preprocessed
        else "independent_window_zero_state_calibration_fallback"
    )
    dataset = EMGWindowDataset(
        windows_npz,
        manifest,
        normalized,
        channel_rotation=0,
        preprocessing_profile=profile,
        allow_window_reset_fallback=True,
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    model = model.to(device)
    model.eval()

    if method == "head":
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.projection.parameters():
            parameter.requires_grad = True
        optimizer = torch.optim.AdamW(model.projection.parameters(), lr=head_lr, weight_decay=1e-4)
        for _ in range(int(head_epochs)):
            for signal, label in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(model(signal.to(device)), label.to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.projection.parameters(), 0.5)
                optimizer.step()
        model.eval()

    all_logits, all_features, all_labels = [], [], []
    with torch.no_grad():
        for signal, label in loader:
            signal = signal.to(device)
            sequence_features = model.forward_features(signal)
            features = sequence_features.mean(dim=1)
            all_features.append(features.cpu())
            all_logits.append(model.projection(features).cpu())
            all_labels.append(label.cpu())
    logits = torch.cat(all_logits)
    features = torch.cat(all_features)
    labels = torch.cat(all_labels)
    prototypes = None
    if method == "prototype":
        prototypes = torch.stack([
            features[labels == index].mean(dim=0)
            for index in range(model.config.output_channels)
        ])
        logits = torch.nn.functional.normalize(features, dim=-1) @ torch.nn.functional.normalize(prototypes, dim=-1).T
    temperature = _fit_temperature(logits.float(), labels)
    predicted = (logits / temperature).argmax(dim=-1)
    artifact = {
        "schema_version": CALIBRATION_SCHEMA,
        "method": method,
        "base_checkpoint_sha256": _sha256_file(checkpoint_path),
        "base_checkpoint_schema": base_payload.get("schema_version"),
        "normalization_sha256": normalization_sha256(normalization),
        "preprocessing_profile_id": profile.profile_id,
        "preprocessing_profile_fingerprint": profile.fingerprint,
        "channel_order": tuple(str(value) for value in channel_order),
        "channel_order_sha256": _channel_order_hash(channel_order),
        "label_names": label_names,
        "subject_id": identity.subject_id,
        "day_id": identity.day_id,
        "session_ids": identity.session_ids,
        "session_span_s": identity.duration_s,
        "temperature": temperature,
        "prototype": prototypes,
        "head_state_dict": (
            {key: value.detach().cpu() for key, value in model.projection.state_dict().items()}
            if method == "head" else None
        ),
        "calibration_accuracy": float((predicted == labels).float().mean().item()),
        "verification_scope": "offline labeled-window calibration; no clinical-performance claim",
        "filter_lineage": filter_lineage,
    }
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, target)
    return artifact


class LoadedEMGCalibration:
    def __init__(self, payload: Mapping[str, Any]) -> None:
        self.payload = payload

    def apply_head(self, model: Any) -> None:
        state = self.payload.get("head_state_dict")
        if state is not None:
            model.projection.load_state_dict(state, strict=True)

    def logits(self, model: Any, inputs: Any) -> Any:
        import torch

        features = model.forward_features(inputs).mean(dim=1)
        prototype = self.payload.get("prototype")
        if prototype is not None:
            prototype = prototype.to(device=features.device, dtype=features.dtype)
            values = torch.nn.functional.normalize(features, dim=-1) @ torch.nn.functional.normalize(prototype, dim=-1).T
        else:
            values = model.projection(features)
        return values / float(self.payload["temperature"])


def load_emg_calibration(
    path: str | Path,
    *,
    base_checkpoint: str | Path,
    channel_order: Sequence[str],
    expected_subject_id: Optional[str] = None,
    expected_day_id: Optional[str] = None,
) -> LoadedEMGCalibration:
    import torch

    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(Path(path), map_location="cpu")
    if payload.get("schema_version") != CALIBRATION_SCHEMA:
        raise ValueError("Unsupported EMG calibration schema")
    checkpoint = Path(base_checkpoint).resolve()
    if payload.get("base_checkpoint_sha256") != _sha256_file(checkpoint):
        raise ValueError("Calibration artifact does not match the base checkpoint bytes")
    _, base_payload = load_emg_checkpoint(checkpoint)
    if payload.get("normalization_sha256") != normalization_sha256(base_payload["normalization"]):
        raise ValueError("Calibration artifact normalization lineage mismatch")
    raw_profile = base_payload.get("preprocessing_profile")
    if not isinstance(raw_profile, Mapping):
        raise ValueError("Base checkpoint lacks preprocessing profile lineage")
    profile = EmgPreprocessingProfile.from_mapping(raw_profile)
    if payload.get("preprocessing_profile_id") != profile.profile_id:
        raise ValueError("Calibration artifact preprocessing profile ID mismatch")
    if payload.get("preprocessing_profile_fingerprint") != profile.fingerprint:
        raise ValueError("Calibration artifact preprocessing profile fingerprint mismatch")
    if payload.get("channel_order_sha256") != _channel_order_hash(channel_order):
        raise ValueError("Calibration artifact channel order mismatch")
    if tuple(payload.get("channel_order", ())) != tuple(channel_order):
        raise ValueError("Calibration artifact channel names mismatch")
    if expected_subject_id is not None and payload.get("subject_id") != expected_subject_id:
        raise ValueError("Calibration artifact subject mismatch")
    if expected_day_id is not None and payload.get("day_id") != expected_day_id:
        raise ValueError("Calibration artifact day mismatch")
    return LoadedEMGCalibration(payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--channel-order", type=Path, required=True)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--day-id", required=True)
    parser.add_argument("--method", choices=("temperature", "prototype", "head"), default="temperature")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    channel_order = json.loads(args.channel_order.read_text(encoding="utf-8"))
    if not isinstance(channel_order, list):
        raise ValueError("channel-order file must be a JSON list")
    artifact = calibrate_emg_checkpoint(
        base_checkpoint=args.base_checkpoint,
        windows_npz=args.windows,
        manifest=args.manifest,
        output=args.output,
        channel_order=channel_order,
        subject_id=args.subject_id,
        day_id=args.day_id,
        method=args.method,
        device=args.device,
    )
    summary = {key: value for key, value in artifact.items() if key not in {"prototype", "head_state_dict"}}
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
