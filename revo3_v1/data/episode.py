"""Fail-closed on-disk contract for a time-aligned Revo3 policy episode.

``action_target_rad`` is the *accepted exact-sent teleoperation target* at
the controller write boundary. It is not measured state, a CAIR-adjusted
online command, or an unacknowledged requested target. Raw EMG is physically
excluded from this corpus.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from revo3_v1.revo.contracts import JOINT_COUNT, JOINT_ORDER_HASH
from revo3_v1.planner.schema import TASK_GRASP_PRIMITIVES, SupportedTask


SUPPORTED_TASKS = ("bottle", "phone", "plastic_bag", "refrigerator_door")
SUPPORTED_TACTILE_PROFILES = (
    "profile_a_force6d_diff",
    "profile_b_diff_only",
    "ablation_force6d_only",
)
ACTION_SEMANTICS = "accepted_exact_sent_teleop_target"
INSTRUCTION_SOURCES = ("manual_canonical", "frozen_planner")
POLICY_PHASES = ("context", "precontact", "contact", "hold", "release")
FROZEN_GRID_HZ = 30
DEFAULT_GRID_TOLERANCE_NS = 2_000_000
MAX_OBSERVATION_AGE_NS = 150_000_000


def _optional_array(
    archive: np.lib.npyio.NpzFile, key: str, dtype: object
) -> Optional[np.ndarray]:
    if key not in archive.files:
        return None
    return np.asarray(archive[key], dtype=dtype)


@dataclass(frozen=True)
class RevoEpisodeMeta:
    episode_id: str
    task_id: str
    task_version: int
    task: str
    object_id: str
    object_instance: str
    operator: str
    collection_day: str
    grasp_primitive: str
    instruction: str
    instruction_sha256: str
    instruction_source: str
    planner_revision: str
    planner_output_sha256: str
    fps: int
    time_grid_hz: int
    timestamp_tolerance_ns: int
    max_command_latency_ns: int
    max_tactile_inter_finger_skew_ns: int
    resampled_30hz: bool
    timestamp_alignment_verified: bool
    joint_order_hash: str
    tactile_num_fingers: int
    tactile_profile: str
    checkpoint_family_id: str
    normalization_family_id: str
    capability_manifest_sha256: str
    camera_profile_id: str
    camera_calibration_sha256: str
    action_label_source: str
    action_semantics: str
    contains_cair_residual: bool
    image_paths: Tuple[str, ...]
    contains_emg: bool
    synthetic_fixture: bool

    @classmethod
    def load(cls, path: Path) -> "RevoEpisodeMeta":
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return cls(
            episode_id=str(value["episode_id"]),
            task_id=str(value.get("task_id", "")),
            task_version=int(value.get("task_version", 0)),
            task=str(value["task"]),
            object_id=str(value.get("object_id", "")),
            object_instance=str(value.get("object_instance", "")),
            operator=str(value.get("operator", "")),
            collection_day=str(value.get("collection_day", "")),
            grasp_primitive=str(value.get("grasp_primitive", "")),
            instruction=str(value["instruction"]),
            instruction_sha256=str(value.get("instruction_sha256", "")),
            instruction_source=str(value.get("instruction_source", "")),
            planner_revision=str(value.get("planner_revision", "")),
            planner_output_sha256=str(value.get("planner_output_sha256", "")),
            fps=int(value["fps"]),
            time_grid_hz=int(value.get("time_grid_hz", 0)),
            timestamp_tolerance_ns=int(
                value.get("timestamp_tolerance_ns", DEFAULT_GRID_TOLERANCE_NS)
            ),
            max_command_latency_ns=int(value.get("max_command_latency_ns", 0)),
            max_tactile_inter_finger_skew_ns=int(
                value.get("max_tactile_inter_finger_skew_ns", 0)
            ),
            resampled_30hz=bool(value.get("resampled_30hz", False)),
            timestamp_alignment_verified=bool(
                value.get("timestamp_alignment_verified", False)
            ),
            joint_order_hash=str(value["joint_order_hash"]),
            tactile_num_fingers=int(value["tactile_num_fingers"]),
            tactile_profile=str(value.get("tactile_profile", "")),
            checkpoint_family_id=str(value.get("checkpoint_family_id", "")),
            normalization_family_id=str(value.get("normalization_family_id", "")),
            capability_manifest_sha256=str(value.get("capability_manifest_sha256", "")),
            camera_profile_id=str(value.get("camera_profile_id", "")),
            camera_calibration_sha256=str(value.get("camera_calibration_sha256", "")),
            action_label_source=str(value["action_label_source"]),
            action_semantics=str(value.get("action_semantics", "")),
            contains_cair_residual=bool(value.get("contains_cair_residual", True)),
            image_paths=tuple(str(item) for item in value["image_paths"]),
            contains_emg=bool(value.get("contains_emg", True)),
            synthetic_fixture=bool(value.get("synthetic_fixture", False)),
        )


@dataclass(frozen=True)
class RevoEpisode:
    root: Path
    meta: RevoEpisodeMeta
    timestamp_ns: np.ndarray
    camera_capture_timestamp_ns: np.ndarray
    camera_receive_timestamp_ns: np.ndarray
    state_timestamp_ns: np.ndarray
    touch_timestamp_ns: Optional[np.ndarray]
    force6d_finger_timestamp_ns: Optional[np.ndarray]
    tactile_history_f6: Optional[np.ndarray]
    tactile_history_timestamp_ns: Optional[np.ndarray]
    tactile_history_sequence: Optional[np.ndarray]
    tactile_diff_timestamp_ns: Optional[np.ndarray]
    action_decision_timestamp_ns: np.ndarray
    action_write_timestamp_ns: np.ndarray
    controller_sequence: np.ndarray
    request_id_hash: np.ndarray
    policy_loss_eligible: np.ndarray
    phase: np.ndarray
    state_rad: np.ndarray
    action_target_rad: np.ndarray
    tactile_features: Optional[np.ndarray]
    tactile_diff: Optional[np.ndarray]

    @classmethod
    def load(cls, root: str | Path) -> "RevoEpisode":
        episode_root = Path(root)
        meta = RevoEpisodeMeta.load(episode_root / "meta.json")
        with np.load(episode_root / "frames.npz", allow_pickle=False) as archive:
            required = (
                "timestamp_ns",
                "camera_capture_timestamp_ns",
                "camera_receive_timestamp_ns",
                "state_timestamp_ns",
                "action_decision_timestamp_ns",
                "action_write_timestamp_ns",
                "controller_sequence",
                "request_id_hash",
                "policy_loss_eligible",
                "phase",
                "state_rad",
                "action_target_rad",
            )
            missing = [key for key in required if key not in archive.files]
            if missing:
                raise ValueError(
                    "episode lacks real observation/write-boundary timestamps or receipts: "
                    f"{missing}. Old real episodes must be explicitly re-exported."
                )
            result = cls(
                root=episode_root,
                meta=meta,
                timestamp_ns=np.asarray(archive["timestamp_ns"], dtype=np.int64),
                camera_capture_timestamp_ns=np.asarray(
                    archive["camera_capture_timestamp_ns"], dtype=np.int64
                ),
                camera_receive_timestamp_ns=np.asarray(
                    archive["camera_receive_timestamp_ns"], dtype=np.int64
                ),
                state_timestamp_ns=np.asarray(archive["state_timestamp_ns"], dtype=np.int64),
                touch_timestamp_ns=_optional_array(archive, "touch_timestamp_ns", np.int64),
                force6d_finger_timestamp_ns=_optional_array(
                    archive, "force6d_finger_timestamp_ns", np.int64
                ),
                tactile_history_f6=_optional_array(
                    archive, "tactile_history_f6", np.float32
                ),
                tactile_history_timestamp_ns=_optional_array(
                    archive, "tactile_history_timestamp_ns", np.int64
                ),
                tactile_history_sequence=_optional_array(
                    archive, "tactile_history_sequence", np.int64
                ),
                tactile_diff_timestamp_ns=_optional_array(
                    archive, "tactile_diff_timestamp_ns", np.int64
                ),
                action_decision_timestamp_ns=np.asarray(
                    archive["action_decision_timestamp_ns"], dtype=np.int64
                ),
                action_write_timestamp_ns=np.asarray(
                    archive["action_write_timestamp_ns"], dtype=np.int64
                ),
                controller_sequence=np.asarray(archive["controller_sequence"], dtype=np.int64),
                request_id_hash=np.asarray(archive["request_id_hash"]).astype(str),
                policy_loss_eligible=np.asarray(
                    archive["policy_loss_eligible"], dtype=np.bool_
                ),
                phase=np.asarray(archive["phase"]).astype(str),
                state_rad=np.asarray(archive["state_rad"], dtype=np.float32),
                action_target_rad=np.asarray(archive["action_target_rad"], dtype=np.float32),
                tactile_features=_optional_array(archive, "tactile_features", np.float32),
                tactile_diff=_optional_array(archive, "tactile_diff", np.float32),
            )
        result.validate()
        return result

    @property
    def num_frames(self) -> int:
        return int(self.timestamp_ns.shape[0])

    @staticmethod
    def _require_time_vector(
        name: str, value: np.ndarray, n: int, *, strict: bool = True
    ) -> None:
        if value.shape != (n,):
            raise ValueError(f"{name} must have shape [N]")
        delta = np.diff(value)
        if np.any(delta <= 0) if strict else np.any(delta < 0):
            qualifier = "strictly monotonic" if strict else "monotonic nondecreasing"
            raise ValueError(f"{name} must be {qualifier}")

    def validate(self) -> None:
        n = self.num_frames
        meta = self.meta
        if meta.task not in SUPPORTED_TASKS:
            raise ValueError(f"unsupported Revo task: {meta.task}")
        if not meta.episode_id or not meta.instruction:
            raise ValueError("episode metadata is incomplete")
        real_provenance = (
            meta.task_id,
            meta.object_id,
            meta.object_instance,
            meta.operator,
            meta.collection_day,
            meta.grasp_primitive,
            meta.instruction_sha256,
            meta.capability_manifest_sha256,
            meta.camera_profile_id,
            meta.camera_calibration_sha256,
        )
        if meta.task_version <= 0 or any(not value for value in real_provenance):
            raise ValueError("episode task/object/operator/camera/capability provenance is incomplete")
        expected_instruction_sha = hashlib.sha256(
            meta.instruction.encode("utf-8")
        ).hexdigest()
        if meta.instruction_sha256 != expected_instruction_sha:
            raise ValueError("instruction_sha256 does not match the serialized instruction")
        expected_primitive = TASK_GRASP_PRIMITIVES[SupportedTask(meta.task)].value
        if meta.grasp_primitive != expected_primitive:
            raise ValueError(
                f"task {meta.task!r} requires grasp_primitive {expected_primitive!r}"
            )
        if meta.instruction_source not in INSTRUCTION_SOURCES:
            raise ValueError(f"unsupported instruction_source: {meta.instruction_source!r}")
        if meta.instruction_source == "frozen_planner" and not (
            meta.planner_revision and len(meta.planner_output_sha256) == 64
        ):
            raise ValueError(
                "frozen_planner instructions require planner_revision and SHA-256 provenance"
            )
        if meta.joint_order_hash != JOINT_ORDER_HASH:
            raise ValueError("episode joint order does not match the Revo3 canonical order")
        if meta.fps != FROZEN_GRID_HZ or meta.time_grid_hz != FROZEN_GRID_HZ:
            raise ValueError("Revo policy episodes must be explicitly resampled to the 30 Hz grid")
        if not meta.resampled_30hz or not meta.timestamp_alignment_verified:
            raise ValueError("30 Hz resampling/alignment evidence is required")
        if not 0 <= meta.timestamp_tolerance_ns <= DEFAULT_GRID_TOLERANCE_NS:
            raise ValueError("timestamp_tolerance_ns must be within the frozen 2 ms limit")
        if not 0 < meta.max_command_latency_ns <= int(round(1e9 / FROZEN_GRID_HZ)):
            raise ValueError("max_command_latency_ns must be positive and no more than one 30 Hz step")
        if meta.max_tactile_inter_finger_skew_ns <= 0:
            raise ValueError("max_tactile_inter_finger_skew_ns must be positive")
        if meta.tactile_profile not in SUPPORTED_TACTILE_PROFILES:
            raise ValueError(f"unknown tactile profile: {meta.tactile_profile!r}")
        if meta.tactile_num_fingers != 5:
            raise ValueError("Revo single-hand tactile profiles require exactly five fingers")
        if not meta.checkpoint_family_id or not meta.normalization_family_id:
            raise ValueError("tactile checkpoint/normalization family IDs are required")
        if len(meta.capability_manifest_sha256) != 64 or len(
            meta.camera_calibration_sha256
        ) != 64:
            raise ValueError("capability/camera calibration provenance requires SHA-256")
        if meta.action_label_source != "controller_target":
            raise ValueError("only accepted controller_target labels are trainable")
        if meta.action_semantics != ACTION_SEMANTICS:
            raise ValueError(f"action_semantics must be {ACTION_SEMANTICS!r}")
        if meta.contains_cair_residual:
            raise ValueError("CAIR-adjusted commands cannot be policy nominal-action labels")
        if meta.contains_emg:
            raise ValueError("EMG is physically excluded from every Revo VLA episode")

        for name, value in (
            ("timestamp_ns", self.timestamp_ns),
            ("camera_capture_timestamp_ns", self.camera_capture_timestamp_ns),
            ("camera_receive_timestamp_ns", self.camera_receive_timestamp_ns),
            ("action_decision_timestamp_ns", self.action_decision_timestamp_ns),
            ("action_write_timestamp_ns", self.action_write_timestamp_ns),
        ):
            self._require_time_vector(name, value, n)
        self._require_time_vector(
            "state_timestamp_ns", self.state_timestamp_ns, n, strict=False
        )
        if n < 2:
            raise ValueError("episode timestamps must contain at least two frames")
        expected_period = 1e9 / FROZEN_GRID_HZ
        if np.any(
            np.abs(np.diff(self.timestamp_ns).astype(np.float64) - expected_period)
            > meta.timestamp_tolerance_ns
        ):
            raise ValueError("episode timestamp grid exceeds the declared 30 Hz tolerance")
        if np.any(self.timestamp_ns > self.action_decision_timestamp_ns):
            raise ValueError("30 Hz anchor timestamp must not follow its action decision")
        if np.any(self.camera_capture_timestamp_ns > self.camera_receive_timestamp_ns):
            raise ValueError("camera capture must not occur after camera receive")
        if np.any(self.camera_receive_timestamp_ns > self.action_decision_timestamp_ns):
            raise ValueError("camera observation arrived after the action decision")
        if np.any(self.state_timestamp_ns > self.action_decision_timestamp_ns):
            raise ValueError("state observation occurred after the action decision")
        if np.any(
            self.action_decision_timestamp_ns - self.camera_receive_timestamp_ns
            > MAX_OBSERVATION_AGE_NS
        ):
            raise ValueError("camera observation exceeds the frozen 150 ms age limit")
        if np.any(
            self.action_decision_timestamp_ns - self.state_timestamp_ns
            > MAX_OBSERVATION_AGE_NS
        ):
            raise ValueError("state observation exceeds the frozen 150 ms age limit")
        if np.any(self.action_decision_timestamp_ns > self.action_write_timestamp_ns):
            raise ValueError("action write cannot precede its decision")
        if np.any(
            self.action_write_timestamp_ns - self.action_decision_timestamp_ns
            > meta.max_command_latency_ns
        ):
            raise ValueError("accepted controller receipt exceeded max_command_latency_ns")

        if self.controller_sequence.shape != (n,) or np.any(np.diff(self.controller_sequence) <= 0):
            raise ValueError("controller_sequence must be [N], unique, and strictly increasing")
        if self.request_id_hash.shape != (n,):
            raise ValueError("request_id_hash must have shape [N]")
        request_ids = [str(item).strip() for item in self.request_id_hash.tolist()]
        if (
            any(
                len(value) != 64
                or any(character not in "0123456789abcdef" for character in value.lower())
                for value in request_ids
            )
            or len(set(request_ids)) != n
        ):
            raise ValueError("request_id_hash must contain N unique SHA-256 receipts")
        if self.policy_loss_eligible.shape != (n,):
            raise ValueError("policy_loss_eligible must have shape [N]")
        if self.phase.shape != (n,) or any(value not in POLICY_PHASES for value in self.phase):
            raise ValueError(f"phase must be [N] with values from {POLICY_PHASES}")
        if np.any(self.policy_loss_eligible & (self.phase == "context")):
            raise ValueError("context/pre-roll frames cannot be hand-policy loss anchors")

        if self.state_rad.shape != (n, JOINT_COUNT):
            raise ValueError(f"state_rad must have shape [N,{JOINT_COUNT}]")
        if self.action_target_rad.shape != (n, JOINT_COUNT):
            raise ValueError(f"action_target_rad must have shape [N,{JOINT_COUNT}]")
        for name, values in (
            ("state_rad", self.state_rad),
            ("action_target_rad", self.action_target_rad),
        ):
            if not np.isfinite(values).all():
                raise ValueError(f"{name} contains NaN or infinity")

        requires_force = meta.tactile_profile in (
            "profile_a_force6d_diff",
            "ablation_force6d_only",
        )
        requires_diff = meta.tactile_profile in (
            "profile_a_force6d_diff",
            "profile_b_diff_only",
        )
        if requires_force:
            expected = (n, meta.tactile_num_fingers, 6)
            if self.tactile_features is None or self.tactile_features.shape != expected:
                raise ValueError(f"{meta.tactile_profile} requires tactile_features {expected}")
            if self.touch_timestamp_ns is None or self.touch_timestamp_ns.shape != (n,):
                raise ValueError(f"{meta.tactile_profile} requires touch_timestamp_ns [N]")
            if (
                self.force6d_finger_timestamp_ns is None
                or self.force6d_finger_timestamp_ns.shape != (n, meta.tactile_num_fingers)
            ):
                raise ValueError("Force6D profile requires per-finger timestamps [N,5]")
            if np.any(np.diff(self.force6d_finger_timestamp_ns, axis=0) < 0):
                raise ValueError("each Force6D finger timestamp must be nondecreasing")
            if not np.array_equal(
                self.force6d_finger_timestamp_ns.max(axis=1), self.touch_timestamp_ns
            ):
                raise ValueError("touch_timestamp_ns must equal max per-finger Force6D timestamp")
            finger_skew = (
                self.force6d_finger_timestamp_ns.max(axis=1)
                - self.force6d_finger_timestamp_ns.min(axis=1)
            )
            if np.any(finger_skew > meta.max_tactile_inter_finger_skew_ns):
                raise ValueError("Force6D inter-finger skew exceeds the capability limit")
            if (
                self.tactile_history_f6 is None
                or self.tactile_history_f6.shape != (n, 16, meta.tactile_num_fingers, 6)
                or self.tactile_history_timestamp_ns is None
                or self.tactile_history_timestamp_ns.shape != (n, 16)
                or self.tactile_history_sequence is None
                or self.tactile_history_sequence.shape != (n, 16)
            ):
                raise ValueError(
                    "Force6D profile requires native [N,16,5,6] history plus timestamp/sequence"
                )
            if np.any(np.diff(self.tactile_history_timestamp_ns, axis=1) <= 0):
                raise ValueError("each Force6D history must contain 16 distinct timestamps")
            if np.any(np.diff(self.tactile_history_sequence, axis=1) <= 0):
                raise ValueError("each Force6D history must contain 16 distinct native sequences")
            if not np.array_equal(
                self.tactile_history_timestamp_ns[:, -1], self.touch_timestamp_ns
            ):
                raise ValueError("Force6D history must end at the current native touch sample")
            if not np.allclose(
                self.tactile_history_f6[:, -1], self.tactile_features, rtol=0.0, atol=1e-6
            ):
                raise ValueError("Force6D history tail must equal current tactile_features")
            self._require_time_vector(
                "touch_timestamp_ns", self.touch_timestamp_ns, n, strict=False
            )
            if np.any(self.touch_timestamp_ns > self.action_decision_timestamp_ns):
                raise ValueError("Force6D observation occurred after the action decision")
            if np.any(
                self.action_decision_timestamp_ns - self.touch_timestamp_ns
                > MAX_OBSERVATION_AGE_NS
            ):
                raise ValueError("Force6D observation exceeds the frozen 150 ms age limit")
            if not np.isfinite(self.tactile_features).all():
                raise ValueError("tactile_features contains NaN or infinity")
        elif any(
            value is not None
            for value in (
                self.tactile_features,
                self.touch_timestamp_ns,
                self.force6d_finger_timestamp_ns,
                self.tactile_history_f6,
                self.tactile_history_timestamp_ns,
                self.tactile_history_sequence,
            )
        ):
            raise ValueError(
                f"{meta.tactile_profile} must not carry unused/fake Force6D observations"
            )

        if requires_diff:
            expected_diff = (n, meta.tactile_num_fingers, 240, 240)
            expected_diff_ts = (n, meta.tactile_num_fingers)
            if self.tactile_diff is None or self.tactile_diff.shape != expected_diff:
                raise ValueError(f"{meta.tactile_profile} requires tactile_diff {expected_diff}")
            if (
                self.tactile_diff_timestamp_ns is None
                or self.tactile_diff_timestamp_ns.shape != expected_diff_ts
            ):
                raise ValueError(
                    f"{meta.tactile_profile} requires tactile_diff_timestamp_ns {expected_diff_ts}"
                )
            if np.any(
                self.tactile_diff_timestamp_ns
                > self.action_decision_timestamp_ns[:, None]
            ):
                raise ValueError("DIFF observation occurred after the action decision")
            if np.any(
                self.tactile_diff_timestamp_ns.max(axis=1)
                - self.tactile_diff_timestamp_ns.min(axis=1)
                > meta.max_tactile_inter_finger_skew_ns
            ):
                raise ValueError("DIFF inter-finger skew exceeds the capability limit")
            if np.any(np.diff(self.tactile_diff_timestamp_ns, axis=0) < 0):
                raise ValueError("each DIFF finger timestamp must be monotonic nondecreasing")
            if np.any(
                self.action_decision_timestamp_ns[:, None]
                - self.tactile_diff_timestamp_ns
                > MAX_OBSERVATION_AGE_NS
            ):
                raise ValueError("DIFF observation exceeds the frozen 150 ms age limit")
            if not np.isfinite(self.tactile_diff).all():
                raise ValueError("tactile_diff contains NaN or infinity")
        elif self.tactile_diff is not None or self.tactile_diff_timestamp_ns is not None:
            raise ValueError(f"{meta.tactile_profile} must not carry unused DIFF observations")

        if len(meta.image_paths) != n:
            raise ValueError("image path count does not match timestamps")
        missing = [path for path in meta.image_paths if not (self.root / path).is_file()]
        if missing:
            raise FileNotFoundError(f"episode has missing RGB frames, first={missing[0]}")
