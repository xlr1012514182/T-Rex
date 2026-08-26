"""Convert reviewed 30 Hz Revo3 episodes to the T-Rex JSON contract.

The converter is deliberately strict: it uses true capture/receive/touch and
controller-receipt timestamps, excludes ineligible pre-roll anchors, keeps
full and fixed-center camera slots atomic, and never pads action, tactile, or
FLARE futures. Statistics are fit only on manifest-declared training splits.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
from PIL import Image

from revo3_v1.revo.contracts import JOINT_COUNT, JOINT_ORDER, JOINT_ORDER_HASH

from .episode import ACTION_SEMANTICS, RevoEpisode, RevoEpisodeMeta
from .native_touch import NativeForceStream, native_force_stream
from .splits import CorpusSplit, RevoCorpusSplitManifest


FORCE_PROFILES = {"profile_a_force6d_diff", "ablation_force6d_only"}
DIFF_PROFILES = {"profile_a_force6d_diff", "profile_b_diff_only"}


@dataclass(frozen=True)
class ConversionConfig:
    action_chunk: int = 16
    tactile_history: int = 16
    action_grid_hz: int = 30
    training_anchor_hz: int = 10
    tactile_delay_offsets: tuple[int, ...] = (0, 4, 8, 12)
    tactile_temporal_jitter: tuple[int, ...] = (-1, 0, 1)
    tactile_temporal_jitter_native_offsets: tuple[int, ...] = (-2, -1, 0)
    tactile_num_fingers: int = 5
    dataset_name: str = "revo3_single_hand"
    dataset_split: str = "synthetic_smoke"
    tactile_profile: str = "ablation_force6d_only"
    checkpoint_family_id: str = "synthetic-ablation_force6d_only-v1"
    normalization_family_id: str = "synthetic-ablation_force6d_only-norm-v1"
    image_width: int = 384
    image_height: int = 288
    center_crop_fraction: float = 0.60
    flare_steps: int = 8
    flare_frame_stride: int = 4
    terminal_padding: str = "forbidden"

    def validate(self) -> None:
        if self.action_chunk != 16 or self.tactile_history != 16:
            raise ValueError("T-Rex Revo3 V1 fixes action/history length to 16")
        if self.action_grid_hz != 30 or self.training_anchor_hz != 10:
            raise ValueError("Revo3 V1 fixes a 30 Hz action grid and 10 Hz training anchors")
        if self.action_grid_hz % self.training_anchor_hz:
            raise ValueError("training anchor rate must divide the action grid")
        if self.tactile_delay_offsets != (0, 4, 8, 12):
            raise ValueError("Revo3 V1 fixes tactile delay offsets to (0,4,8,12)")
        if self.tactile_temporal_jitter != (-1, 0, 1):
            raise ValueError("Revo3 V1 fixes tactile temporal jitter to +/-1 sample")
        if self.tactile_temporal_jitter_native_offsets != (-2, -1, 0):
            raise ValueError(
                "causal jitter maps logical -1/0/+1 to previous/center/latest native sample"
            )
        if self.tactile_num_fingers != 5:
            raise ValueError("Revo3 single-hand V1 requires five tactile streams")
        if self.tactile_profile not in FORCE_PROFILES | DIFF_PROFILES:
            raise ValueError("current trainer supports Profile A/B and Force6D-only ablation")
        if not self.checkpoint_family_id or not self.normalization_family_id:
            raise ValueError("checkpoint and normalization family IDs must be non-empty")
        if self.terminal_padding != "forbidden":
            raise ValueError("terminal action/history/FLARE padding is forbidden")
        if self.image_width != 384 or self.image_height != 288:
            raise ValueError("Revo3 V1 fixes both model views to 384x288")
        if not 0.25 <= self.center_crop_fraction <= 0.9:
            raise ValueError("center_crop_fraction must be in [0.25,0.9]")
        if self.flare_steps != 8 or self.flare_frame_stride != 4:
            raise ValueError("Revo V1 fixes FLARE at 8 future frames with stride 4")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _robust_block(values: np.ndarray) -> Dict[str, object]:
    q01 = np.quantile(values, 0.01, axis=0).astype(np.float32)
    q99 = np.quantile(values, 0.99, axis=0).astype(np.float32)
    mask = np.abs(q99 - q01) > 1e-6
    return {"q01": q01.tolist(), "q99": q99.tolist(), "mask": mask.tolist()}


def _chunk(values: np.ndarray, start: int, length: int) -> np.ndarray:
    selected = values[start : start + length]
    if selected.shape[0] != length:
        raise ValueError("action chunk crosses the episode boundary; padding is forbidden")
    return selected.astype(np.float32, copy=False)


def _dense_history(values: np.ndarray, end: int, length: int) -> np.ndarray:
    start = int(end) - int(length) + 1
    if start < 0:
        raise ValueError("tactile history is not backed by 16 real samples")
    selected = values[start : end + 1]
    if selected.shape[0] != length:
        raise ValueError("tactile history crosses the episode boundary")
    return selected.astype(np.float32, copy=False)


def _eligible_anchors(episode: RevoEpisode, config: ConversionConfig) -> list[int]:
    # Native Force6D rings already include samples before the policy anchor.
    # Keep the 10 Hz anchor phase locked to the 30 Hz grid (0,3,6,...) rather
    # than shifting it to satisfy a policy-frame approximation of jitter.
    first = 0
    future_horizon = max(
        config.action_chunk - 1,
        max(config.tactile_delay_offsets),
        config.flare_steps * config.flare_frame_stride,
    )
    last = episode.num_frames - future_horizon - 1
    if last < first:
        return []
    stride = config.action_grid_hz // config.training_anchor_hz
    result = []
    for frame in range(first, last + 1, stride):
        # The history may intentionally include no-contact pre-roll. Every
        # supervised action and FLARE future, however, must remain inside one
        # explicitly authorized policy-loss segment.
        authorized = episode.policy_loss_eligible[frame : frame + future_horizon + 1]
        if authorized.shape[0] == future_horizon + 1 and bool(np.all(authorized)):
            result.append(frame)
    return result


def _native_force_jitter_options(
    stream: NativeForceStream,
    latest_sequence: int,
    config: ConversionConfig,
) -> tuple[list[object], list[object], list[int], list[object], list[object]]:
    """Build three causal candidates from distinct native Force6D samples."""

    touches: list[object] = []
    histories: list[object] = []
    touch_timestamps: list[int] = []
    history_timestamps: list[object] = []
    history_sequences: list[object] = []
    for native_offset in config.tactile_temporal_jitter_native_offsets:
        values, timestamps, sequences = stream.window_ending_at(
            latest_sequence,
            relative_end=native_offset,
            length=config.tactile_history,
        )
        histories.append(values.tolist())
        touches.append(values[-1].tolist())
        touch_timestamps.append(int(timestamps[-1]))
        history_timestamps.append(timestamps.astype(np.int64).tolist())
        history_sequences.append(sequences.astype(np.int64).tolist())
    return (
        touches,
        histories,
        touch_timestamps,
        history_timestamps,
        history_sequences,
    )


def _causal_diff_jitter_frames(
    episode: RevoEpisode,
    delayed_frame: int,
    *,
    count: int = 3,
) -> list[int]:
    """Select the latest distinct DIFF observations before one decision."""

    if episode.tactile_diff_timestamp_ns is None:
        raise ValueError("DIFF timestamps are required")
    decision = int(episode.action_decision_timestamp_ns[delayed_frame])
    selected_latest_first: list[int] = []
    observed_timestamp_rows: set[tuple[int, ...]] = set()
    for candidate in range(delayed_frame, -1, -1):
        timestamp_row = tuple(
            int(value) for value in episode.tactile_diff_timestamp_ns[candidate]
        )
        if max(timestamp_row) > decision:
            continue
        if timestamp_row in observed_timestamp_rows:
            continue
        observed_timestamp_rows.add(timestamp_row)
        selected_latest_first.append(candidate)
        if len(selected_latest_first) == count:
            break
    if len(selected_latest_first) != count:
        raise ValueError(
            "DIFF temporal jitter requires three distinct causal observations"
        )
    return list(reversed(selected_latest_first))


def _prepare_dual_views(
    source: Path,
    destination_root: Path,
    *,
    episode_id: str,
    frame: int,
    config: ConversionConfig,
) -> tuple[Path, Path]:
    if not source.is_file():
        raise FileNotFoundError(f"source RGB frame does not exist: {source}")
    view_root = destination_root / "revo3_views" / episode_id
    view_root.mkdir(parents=True, exist_ok=True)
    full_path = view_root / f"image{frame:06d}_full.png"
    center_path = view_root / f"image{frame:06d}_center.png"
    if full_path.is_file() and center_path.is_file():
        return full_path, center_path
    with Image.open(source) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        crop_width = max(1, int(round(width * config.center_crop_fraction)))
        crop_height = max(1, int(round(height * config.center_crop_fraction)))
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        center = rgb.crop((left, top, left + crop_width, top + crop_height))
        target = (config.image_width, config.image_height)
        rgb.resize(target, Image.Resampling.LANCZOS).save(full_path, format="PNG")
        center.resize(target, Image.Resampling.LANCZOS).save(center_path, format="PNG")
    return full_path, center_path


def _prepare_diff_images(
    episode: RevoEpisode, destination_root: Path, frame: int
) -> list[Path]:
    if episode.tactile_diff is None:
        raise ValueError(f"{episode.meta.tactile_profile} has no real DIFF tensor")
    diff_root = destination_root / "revo3_tactile_diff" / episode.meta.episode_id
    diff_root.mkdir(parents=True, exist_ok=True)
    paths = []
    for finger in range(episode.meta.tactile_num_fingers):
        path = diff_root / f"frame{frame:06d}_finger{finger}.png"
        if not path.is_file():
            value = np.asarray(episode.tactile_diff[frame, finger], dtype=np.float32)
            if value.shape != (240, 240) or not np.isfinite(value).all():
                raise ValueError("DIFF must be a finite [240,240] image per finger")
            if float(value.max(initial=0.0)) <= 1.0:
                value = value * 255.0
            Image.fromarray(np.clip(value, 0, 255).astype(np.uint8), mode="L").save(
                path, format="PNG"
            )
        paths.append(path)
    return paths


def _relative(paths: Sequence[Path], root: Path) -> list[str]:
    return [os.path.relpath(path, root).replace("\\", "/") for path in paths]


def _fit_train_statistics(
    episodes: Sequence[RevoEpisode], config: ConversionConfig
) -> Mapping[str, object]:
    action_values, state_values, tactile_values, tracking_values = [], [], [], []
    no_contact_values = []
    for episode in episodes:
        for frame in _eligible_anchors(episode, config):
            action_values.append(_chunk(episode.action_target_rad, frame, config.action_chunk))
            state_values.append(episode.state_rad[frame])
            if config.tactile_profile in FORCE_PROFILES:
                if episode.tactile_features is None:
                    raise ValueError("Force6D profile is missing tactile_features")
                tactile_values.append(episode.tactile_features[frame].reshape(-1))
        eligible_pairs = episode.policy_loss_eligible[1:] & episode.policy_loss_eligible[:-1]
        if np.any(eligible_pairs):
            tracking_values.append(
                episode.state_rad[1:][eligible_pairs]
                - episode.action_target_rad[:-1][eligible_pairs]
            )
        if config.tactile_profile in FORCE_PROFILES:
            assert episode.tactile_features is not None
            no_contact = episode.phase == "context"
            if np.any(no_contact):
                no_contact_values.append(episode.tactile_features[no_contact])
    if not action_values or not tracking_values:
        raise ValueError("train split has no eligible anchors/tracking pairs for statistics")
    block: dict[str, object] = {
        "action": _robust_block(np.stack(action_values)),
        "state": _robust_block(np.stack(state_values)),
        "tracking_error": {
            "mean": np.concatenate(tracking_values).mean(axis=0).astype(np.float32).tolist(),
            "std": np.concatenate(tracking_values).std(axis=0).astype(np.float32).tolist(),
        },
    }
    if config.tactile_profile in FORCE_PROFILES:
        block["tactile_f6"] = _robust_block(np.stack(tactile_values))
        if not no_contact_values:
            raise ValueError("Force6D training stats require explicit context/no-contact samples")
        baseline = np.concatenate(no_contact_values, axis=0)
        if baseline.shape[0] < 32:
            raise ValueError("at least 32 no-contact Force6D samples are required for noise stats")
        flat = baseline.reshape(baseline.shape[0], -1)
        center = np.median(baseline, axis=0).astype(np.float32)
        scale = (1.4826 * np.median(np.abs(baseline - center), axis=0)).astype(np.float32)
        block["tactile_no_contact_noise"] = {
            "robust_center": center.tolist(),
            "robust_scale": scale.tolist(),
            "covariance": np.cov(flat, rowvar=False).astype(np.float32).tolist(),
            "source_phase": "context",
            "minimum_samples": 32,
            "observed_samples": int(baseline.shape[0]),
            "normalization_family_id": config.normalization_family_id,
            "checkpoint_family_id": config.checkpoint_family_id,
        }
    return {config.dataset_name: block}


def _select_episodes(
    episodes_root: Path,
    config: ConversionConfig,
    split_manifest: str | Path | None,
) -> tuple[list[RevoEpisode], list[RevoEpisode], Mapping[str, object]]:
    meta_paths = sorted(episodes_root.glob("*/meta.json"))
    if not meta_paths:
        raise FileNotFoundError(f"no Revo episodes found under {episodes_root}")
    metas = {path.parent.name: RevoEpisodeMeta.load(path) for path in meta_paths}
    synthetic = all(meta.synthetic_fixture for meta in metas.values())
    if any(meta.synthetic_fixture != synthetic for meta in metas.values()):
        raise ValueError("real and synthetic episodes cannot share one corpus")
    if synthetic:
        if split_manifest is not None or config.dataset_split != "synthetic_smoke":
            raise ValueError("synthetic conversion uses only the explicit synthetic_smoke split")
        selected_ids = tuple(metas)
        stats_ids = selected_ids
        split_evidence: Mapping[str, object] = {
            "dataset_split": "synthetic_smoke",
            "split_manifest_sha256": "",
            "split_before_statistics": True,
            "stats_episode_ids": list(stats_ids),
            "duration_report": {},
            "duration_targets_enforced": False,
        }
    else:
        if split_manifest is None:
            raise ValueError("real data conversion requires a split manifest before statistics")
        split_path = Path(split_manifest)
        manifest = RevoCorpusSplitManifest.load(split_path)
        manifest.assert_matches_episode_root(episodes_root)
        split = CorpusSplit(config.dataset_split)
        selected_ids = manifest.episode_ids((split,))
        # Normalization is frozen from MIDTRAIN_TRAIN only.  SFT is never a
        # statistics source, even though it is a train-authorized policy split.
        stats_ids = manifest.episode_ids((CorpusSplit.MIDTRAIN_TRAIN,))
        split_evidence = {
            "dataset_split": split.value,
            "split_manifest_sha256": _sha256_file(split_path),
            "split_before_statistics": True,
            "stats_episode_ids": list(stats_ids),
            "duration_report": manifest.duration_report(),
            "duration_targets_enforced": manifest.duration_targets_enforced,
            "locked_test_isolation": list(manifest.locked_test_isolation),
        }
    selected = [RevoEpisode.load(episodes_root / item) for item in selected_ids]
    should_open_stats_source = synthetic or (
        config.dataset_split == CorpusSplit.MIDTRAIN_TRAIN.value
    )
    stats_episodes = (
        [RevoEpisode.load(episodes_root / item) for item in stats_ids]
        if should_open_stats_source
        else []
    )
    if not selected:
        raise ValueError(f"selected split {config.dataset_split!r} contains no episodes")
    return selected, stats_episodes, split_evidence


def _statistics_artifact_path(statistics_path: Path) -> Path:
    return statistics_path.with_name(statistics_path.stem + "_artifact.json")


def _write_frozen_statistics(
    statistics_path: Path,
    statistics: Mapping[str, object],
    *,
    source_split: str,
    split_evidence: Mapping[str, object],
    config: ConversionConfig,
    capability_manifest_sha256: str,
) -> tuple[Path, Mapping[str, object]]:
    with statistics_path.open("w", encoding="utf-8") as handle:
        json.dump(statistics, handle, ensure_ascii=False, indent=2)
    artifact = {
        "schema_version": "revo3-normalization-artifact-v1",
        "statistics_path": str(statistics_path.resolve()),
        "statistics_sha256": _sha256_file(statistics_path),
        "source_split": source_split,
        "split_manifest_sha256": split_evidence["split_manifest_sha256"],
        "stats_episode_ids": list(split_evidence["stats_episode_ids"]),
        "joint_order_hash": JOINT_ORDER_HASH,
        "tactile_profile": config.tactile_profile,
        "checkpoint_family_id": config.checkpoint_family_id,
        "normalization_family_id": config.normalization_family_id,
        "capability_manifest_sha256": capability_manifest_sha256,
    }
    artifact_path = _statistics_artifact_path(statistics_path)
    with artifact_path.open("w", encoding="utf-8") as handle:
        json.dump(artifact, handle, ensure_ascii=False, indent=2)
    return artifact_path, artifact


def _load_frozen_statistics(
    statistics_path: Path,
    artifact_path: Path,
    *,
    split_evidence: Mapping[str, object],
    config: ConversionConfig,
    capability_manifest_sha256: str,
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    if not statistics_path.is_file() or not artifact_path.is_file():
        raise FileNotFoundError(
            "non-midtrain conversion requires the frozen MIDTRAIN_TRAIN "
            "statistics JSON and companion artifact"
        )
    with statistics_path.open("r", encoding="utf-8") as handle:
        statistics = json.load(handle)
    with artifact_path.open("r", encoding="utf-8") as handle:
        artifact = json.load(handle)
    required = {
        "schema_version": "revo3-normalization-artifact-v1",
        "statistics_path": str(statistics_path.resolve()),
        "statistics_sha256": _sha256_file(statistics_path),
        "source_split": CorpusSplit.MIDTRAIN_TRAIN.value,
        "split_manifest_sha256": split_evidence["split_manifest_sha256"],
        "stats_episode_ids": list(split_evidence["stats_episode_ids"]),
        "joint_order_hash": JOINT_ORDER_HASH,
        "tactile_profile": config.tactile_profile,
        "checkpoint_family_id": config.checkpoint_family_id,
        "normalization_family_id": config.normalization_family_id,
        "capability_manifest_sha256": capability_manifest_sha256,
    }
    mismatches = {
        key: (expected, artifact.get(key))
        for key, expected in required.items()
        if artifact.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"frozen normalization artifact mismatch: {mismatches}")
    if not isinstance(statistics, dict) or set(statistics) != {config.dataset_name}:
        raise ValueError("frozen normalization JSON has an unexpected dataset schema")
    return statistics, artifact


def convert_revo_episodes_to_trex_json(
    episodes_root: str | Path,
    output_json: str | Path,
    config: ConversionConfig = ConversionConfig(),
    *,
    split_manifest: str | Path | None = None,
    frozen_statistics_path: str | Path | None = None,
    frozen_statistics_artifact: str | Path | None = None,
) -> Dict[str, object]:
    """Write EMG-free Revo records plus train-only normalization statistics."""

    config.validate()
    source_root = Path(episodes_root)
    episodes, stats_episodes, split_evidence = _select_episodes(
        source_root, config, split_manifest
    )
    for episode in episodes + stats_episodes:
        expected = (
            config.tactile_profile,
            config.checkpoint_family_id,
            config.normalization_family_id,
        )
        observed = (
            episode.meta.tactile_profile,
            episode.meta.checkpoint_family_id,
            episode.meta.normalization_family_id,
        )
        if observed != expected:
            raise ValueError(f"mixed tactile profile/checkpoint families: {observed} != {expected}")
    capability_hashes = {
        episode.meta.capability_manifest_sha256
        for episode in episodes + stats_episodes
    }
    if len(capability_hashes) != 1:
        raise ValueError("one conversion cannot mix tactile capability manifests")
    capability_manifest_sha256 = next(iter(capability_hashes))

    output = Path(output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    instruction_source_counts = {source: 0 for source in ("manual_canonical", "frozen_planner")}
    for episode in episodes:
        force_stream = (
            native_force_stream(episode)
            if config.tactile_profile in FORCE_PROFILES
            else None
        )
        anchors = _eligible_anchors(episode, config)
        if not anchors:
            raise ValueError(
                f"episode {episode.meta.episode_id!r} has no fully eligible, unpadded anchor"
            )
        for frame in anchors:
            action_chunk = _chunk(episode.action_target_rad, frame, config.action_chunk)
            full_absolute, center_absolute = _prepare_dual_views(
                episode.root / episode.meta.image_paths[frame],
                output.parent,
                episode_id=episode.meta.episode_id,
                frame=frame,
                config=config,
            )
            flare_frames = [
                frame + (step + 1) * config.flare_frame_stride
                for step in range(config.flare_steps)
            ]
            flare_paths = []
            for future in flare_frames:
                future_full, _ = _prepare_dual_views(
                    episode.root / episode.meta.image_paths[future],
                    output.parent,
                    episode_id=episode.meta.episode_id,
                    frame=future,
                    config=config,
                )
                flare_paths.append(future_full)

            force_current: list[float] = []
            delayed_touch: list[object] = []
            delayed_histories: list[object] = []
            jittered_touch: list[object] = []
            jittered_histories: list[object] = []
            delayed_touch_ts: list[int] = []
            delayed_history_ts: list[object] = []
            jittered_touch_ts: list[list[int]] = []
            jittered_history_ts: list[object] = []
            delayed_history_sequence: list[object] = []
            jittered_history_sequence: list[object] = []
            if config.tactile_profile in FORCE_PROFILES:
                assert force_stream is not None
                assert episode.tactile_features is not None
                assert episode.touch_timestamp_ns is not None
                assert episode.tactile_history_f6 is not None
                assert episode.tactile_history_timestamp_ns is not None
                assert episode.tactile_history_sequence is not None
                force_current = episode.tactile_features[frame].reshape(-1).tolist()
                delayed_touch = [
                    episode.tactile_features[frame + offset].tolist()
                    for offset in config.tactile_delay_offsets
                ]
                delayed_histories = [
                    episode.tactile_history_f6[frame + offset].tolist()
                    for offset in config.tactile_delay_offsets
                ]
                delayed_touch_ts = [
                    int(episode.touch_timestamp_ns[frame + offset])
                    for offset in config.tactile_delay_offsets
                ]
                delayed_history_ts = [
                    episode.tactile_history_timestamp_ns[frame + offset]
                    .astype(np.int64)
                    .tolist()
                    for offset in config.tactile_delay_offsets
                ]
                delayed_history_sequence = [
                    episode.tactile_history_sequence[frame + offset]
                    .astype(np.int64)
                    .tolist()
                    for offset in config.tactile_delay_offsets
                ]
                for offset in config.tactile_delay_offsets:
                    latest_sequence = int(
                        episode.tactile_history_sequence[frame + offset, -1]
                    )
                    (
                        touch_options,
                        history_options,
                        touch_timestamp_options,
                        history_timestamp_options,
                        history_sequence_options,
                    ) = _native_force_jitter_options(
                        force_stream, latest_sequence, config
                    )
                    jittered_touch.append(touch_options)
                    jittered_histories.append(history_options)
                    jittered_touch_ts.append(touch_timestamp_options)
                    jittered_history_ts.append(history_timestamp_options)
                    jittered_history_sequence.append(history_sequence_options)

            delayed_diff_paths: list[list[str]] = []
            delayed_diff_ts: list[list[int]] = []
            jittered_diff_paths: list[list[list[str]]] = []
            jittered_diff_ts: list[list[list[int]]] = []
            if config.tactile_profile in DIFF_PROFILES:
                assert episode.tactile_diff_timestamp_ns is not None
                for offset in config.tactile_delay_offsets:
                    delayed_diff_paths.append(
                        _relative(
                            _prepare_diff_images(episode, output.parent, frame + offset),
                            output.parent,
                        )
                    )
                    delayed_diff_ts.append(
                        episode.tactile_diff_timestamp_ns[frame + offset]
                        .astype(np.int64)
                        .tolist()
                    )
                    jitter_frames = _causal_diff_jitter_frames(
                        episode, frame + offset
                    )
                    jittered_diff_paths.append(
                        [
                            _relative(
                                _prepare_diff_images(episode, output.parent, candidate),
                                output.parent,
                            )
                            for candidate in jitter_frames
                        ]
                    )
                    jittered_diff_ts.append(
                        [
                            episode.tactile_diff_timestamp_ns[candidate]
                            .astype(np.int64)
                            .tolist()
                            for candidate in jitter_frames
                        ]
                    )

            row = {
                "schema_version": "revo3-trex-json-v1",
                "episode_id": episode.meta.episode_id,
                "task_id": episode.meta.task_id,
                "task_version": episode.meta.task_version,
                "frame_index": frame,
                "phase": str(episode.phase[frame]),
                "policy_loss_eligible": True,
                "timestamp_ns": int(episode.timestamp_ns[frame]),
                "action_decision_timestamp_ns": int(
                    episode.action_decision_timestamp_ns[frame]
                ),
                "rgb_full_timestamp_ns": int(episode.camera_capture_timestamp_ns[frame]),
                "rgb_center_timestamp_ns": int(episode.camera_capture_timestamp_ns[frame]),
                "rgb_receive_timestamp_ns": int(episode.camera_receive_timestamp_ns[frame]),
                "state_timestamp_ns": int(episode.state_timestamp_ns[frame]),
                "input_prompt": episode.meta.instruction,
                "instruction_source": episode.meta.instruction_source,
                "instruction_sha256": episode.meta.instruction_sha256,
                "planner_revision": episode.meta.planner_revision,
                "planner_output_sha256": episode.meta.planner_output_sha256,
                "input_image_slow": _relative([full_absolute], output.parent),
                "input_image_fast": _relative([center_absolute], output.parent),
                "flare_image_full": _relative(flare_paths, output.parent),
                "flare_timestamp_ns": episode.camera_capture_timestamp_ns[flare_frames]
                .astype(np.int64)
                .tolist(),
                "state_fast": episode.state_rad[frame].tolist(),
                "action": action_chunk.reshape(-1).tolist(),
                "action_write_timestamp_ns": episode.action_write_timestamp_ns[
                    frame : frame + config.action_chunk
                ].astype(np.int64).tolist(),
                "action_controller_sequence": episode.controller_sequence[
                    frame : frame + config.action_chunk
                ].astype(np.int64).tolist(),
                "action_request_id_hash": episode.request_id_hash[
                    frame : frame + config.action_chunk
                ].tolist(),
                "tactile_profile": config.tactile_profile,
                "checkpoint_family_id": config.checkpoint_family_id,
                "normalization_family_id": config.normalization_family_id,
                "tactile_f6": force_current,
                "tactile_delay_offsets": list(config.tactile_delay_offsets),
                "tactile_decision_timestamp_ns_delayed": [
                    int(episode.action_decision_timestamp_ns[frame + offset])
                    for offset in config.tactile_delay_offsets
                ],
                "tactile_temporal_jitter_samples": list(config.tactile_temporal_jitter),
                "tactile_temporal_jitter_native_offsets": list(
                    config.tactile_temporal_jitter_native_offsets
                ),
                "tactile_f6_delayed": delayed_touch,
                "tactile_f6_history_delayed": delayed_histories,
                "tactile_f6_delayed_jitter": jittered_touch,
                "tactile_f6_history_delayed_jitter": jittered_histories,
                "touch_timestamp_ns_delayed": delayed_touch_ts,
                "tactile_f6_history_timestamp_ns_delayed": delayed_history_ts,
                "tactile_f6_history_sequence_delayed": delayed_history_sequence,
                "touch_timestamp_ns_delayed_jitter": jittered_touch_ts,
                "tactile_f6_history_timestamp_ns_delayed_jitter": jittered_history_ts,
                "tactile_f6_history_sequence_delayed_jitter": jittered_history_sequence,
                "tactile_image_deform": delayed_diff_paths[0] if delayed_diff_paths else [],
                "tactile_image_deform_delayed": delayed_diff_paths,
                "tactile_deform_timestamp_ns": delayed_diff_ts[0] if delayed_diff_ts else [],
                "tactile_deform_timestamp_ns_delayed": delayed_diff_ts,
                "tactile_image_deform_delayed_jitter": jittered_diff_paths,
                "tactile_deform_timestamp_ns_delayed_jitter": jittered_diff_ts,
                "action_label_source": episode.meta.action_label_source,
                "action_semantics": episode.meta.action_semantics,
                "contains_cair_residual": False,
                "contains_emg": False,
            }
            rows.append(row)
            instruction_source_counts[episode.meta.instruction_source] += 1

    if config.dataset_split == CorpusSplit.SFT_TRAIN.value:
        total = len(rows)
        planner_ratio = instruction_source_counts["frozen_planner"] / max(total, 1)
        if not 0.4 <= planner_ratio <= 0.6:
            raise ValueError(
                "SFT language distribution must be approximately 50% manual canonical / "
                f"50% frozen Planner output; observed frozen_planner={planner_ratio:.3f}"
            )

    with output.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False)
    is_synthetic = config.dataset_split == "synthetic_smoke"
    is_midtrain_source = config.dataset_split == CorpusSplit.MIDTRAIN_TRAIN.value
    supplied_frozen = (
        frozen_statistics_path is not None
        or frozen_statistics_artifact is not None
    )
    if supplied_frozen and (
        frozen_statistics_path is None or frozen_statistics_artifact is None
    ):
        raise ValueError(
            "frozen_statistics_path and frozen_statistics_artifact are one atomic input"
        )
    if is_synthetic and supplied_frozen:
        raise ValueError("synthetic smoke conversion cannot consume a real frozen artifact")
    if is_synthetic or (is_midtrain_source and not supplied_frozen):
        statistics = _fit_train_statistics(stats_episodes, config)
        stats_path = Path(str(output).replace(".json", "_statistics.json"))
        stats_artifact_path, stats_artifact = _write_frozen_statistics(
            stats_path,
            statistics,
            source_split=config.dataset_split,
            split_evidence=split_evidence,
            config=config,
            capability_manifest_sha256=capability_manifest_sha256,
        )
    else:
        if not supplied_frozen:
            raise ValueError(
                "SFT/development/locked conversion must load the frozen "
                "MIDTRAIN_TRAIN normalization artifact"
            )
        stats_path = Path(frozen_statistics_path).resolve()
        stats_artifact_path = Path(frozen_statistics_artifact).resolve()
        statistics, stats_artifact = _load_frozen_statistics(
            stats_path,
            stats_artifact_path,
            split_evidence=split_evidence,
            config=config,
            capability_manifest_sha256=capability_manifest_sha256,
        )
    manifest = {
        "schema_version": "revo3-trex-conversion-v1",
        "config": asdict(config),
        "source_root": str(source_root.resolve()),
        "records": len(rows),
        "episode_ids": [episode.meta.episode_id for episode in episodes],
        "joint_order": list(JOINT_ORDER),
        "joint_order_hash": JOINT_ORDER_HASH,
        "action_shape": [config.action_chunk, JOINT_COUNT],
        "state_shape": [JOINT_COUNT],
        "tactile_shape": [config.tactile_num_fingers, 6]
        if config.tactile_profile in FORCE_PROFILES else None,
        "tactile_history_shape": [
            config.tactile_history, config.tactile_num_fingers, 6
        ] if config.tactile_profile in FORCE_PROFILES else None,
        "tactile_deform_shape": [config.tactile_num_fingers, 1, 240, 240]
        if config.tactile_profile in DIFF_PROFILES else None,
        "tactile_delay_offsets": list(config.tactile_delay_offsets),
        "tactile_temporal_jitter_samples": list(config.tactile_temporal_jitter),
        "tactile_temporal_jitter_native_offsets": list(
            config.tactile_temporal_jitter_native_offsets
        ),
        "action_grid_hz": config.action_grid_hz,
        "training_anchor_hz": config.training_anchor_hz,
        "flare_steps": config.flare_steps,
        "flare_frame_stride": config.flare_frame_stride,
        "flare_padding": "forbidden",
        "view_slots": {"slow": "full", "fast": "fixed_center"},
        "view_shape_hwc": [config.image_height, config.image_width, 3],
        "views_share_timestamp": True,
        "center_crop_fraction": config.center_crop_fraction,
        "terminal_padding": config.terminal_padding,
        "tactile_profile": config.tactile_profile,
        "checkpoint_family_id": config.checkpoint_family_id,
        "normalization_family_id": config.normalization_family_id,
        "capability_manifest_sha256": capability_manifest_sha256,
        "action_label_source": "controller_target",
        "action_semantics": ACTION_SEMANTICS,
        "contains_cair_residual": False,
        "contains_emg": False,
        "instruction_source_counts": instruction_source_counts,
        "stats_path": str(stats_path.resolve()),
        "stats_artifact_path": str(stats_artifact_path.resolve()),
        "statistics_sha256": stats_artifact["statistics_sha256"],
        "statistics_artifact_sha256": _sha256_file(stats_artifact_path),
        "statistics_source_split": stats_artifact["source_split"],
        "normalization_frozen": True,
        **split_evidence,
        "limitations": [
            "Synthetic episodes validate plumbing only and provide no robot-performance evidence.",
            "Pressure/matrix Profile C requires its own native schema and trainer.",
        ],
    }
    manifest_path = output.with_name(output.stem + "_manifest.json")
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest
