import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "revo3_v1_trex.py"
SPEC = importlib.util.spec_from_file_location("revo3_v1_trex_launcher", SCRIPT)
LAUNCHER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(LAUNCHER)

SERVER_SCRIPT = REPO_ROOT / "scripts" / "test.py"
SERVER_SPEC = importlib.util.spec_from_file_location("revo3_v1_trex_server", SERVER_SCRIPT)
SERVER = importlib.util.module_from_spec(SERVER_SPEC)
assert SERVER_SPEC.loader is not None
with mock.patch.dict(sys.modules, {"zmq": types.SimpleNamespace()}):
    SERVER_SPEC.loader.exec_module(SERVER)


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)


class RevoTRexLauncherTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.base = self.root / "qwen"
        self.checkpoint = self.root / "pretrain"
        self.output = self.root / "out"
        self.base.mkdir()
        self.checkpoint.mkdir()
        for name in ("config.json", "tokenizer_config.json", "preprocessor_config.json"):
            _write(self.base / name, {})
        (self.base / "model-00001-of-00001.safetensors").write_bytes(b"test-only")
        (self.checkpoint / "model.pt").write_bytes(b"test-only")
        _write(self.checkpoint / "training_args.json", {"use_tactile_vqvae": 0})

        self.split_hash = "c" * 64
        self.capability_hash = "a" * 64
        self.profile = self.root / "tactile_profile.json"
        _write(self.profile, {
            "schema_version": "revo3-tactile-profile-v1",
            "profile": "profile_a_force6d_diff",
            "approved_for_training": True,
            "capability_manifest_sha256": self.capability_hash,
            "checkpoint_family_id": "revo-profile-a-v1",
            "normalization_family_id": "revo-profile-a-norm-v1",
        })

        self.vqvae = self.root / "revo_vq.pt"
        self.vqvae.write_bytes(b"test-only-vq")
        self.vq_artifact = self.root / "revo_vq_artifact.json"
        vq_train_episode_ids = ["ep0", "ep1"]
        vq_validation_episode_ids = ["dev0"]
        _write(self.vq_artifact, {
            "schema_version": "revo3-force6d-vqvae-artifact-v1",
            "checkpoint_sha256": LAUNCHER._sha256_file(self.vqvae),
            "trained_from_scratch": True,
            "source_sensor_family": "revo3_u21vt_force6d",
            "tactile_profile": "profile_a_force6d_diff",
            "checkpoint_family_id": "revo-profile-a-v1",
            "normalization_family_id": "revo-profile-a-norm-v1",
            "capability_manifest_sha256": self.capability_hash,
            "split_manifest_sha256": self.split_hash,
            "locked_test_opened": False,
            "num_fingers": 5, "window": 16, "stride": 4,
            "codebook_size": 64, "embed_dim": 256,
            "ema_decay": 0.99, "commitment_beta": 0.25,
            "granularity": "finger",
            "source_splits": ["midtrain_train"],
            "train_episode_ids": vq_train_episode_ids,
            "train_episode_ids_sha256": hashlib.sha256(
                json.dumps(vq_train_episode_ids, separators=(",", ":")).encode()
            ).hexdigest(),
            "validation_split": "development",
            "validation_episode_ids": vq_validation_episode_ids,
            "validation_episode_ids_sha256": hashlib.sha256(
                json.dumps(vq_validation_episode_ids, separators=(",", ":")).encode()
            ).hexdigest(),
        })
        self.deform = self.root / "revo_deform.pt"
        self.deform.write_bytes(b"test-only-deform")
        self.deform_artifact = self.root / "revo_deform_artifact.json"
        train_episode_ids = ["ep0", "ep1"]
        validation_episode_ids = ["dev0"]
        _write(self.deform_artifact, {
            "schema_version": "revo3-deform-encoder-artifact-v1",
            "checkpoint_sha256": LAUNCHER._sha256_file(self.deform),
            "trained_from_scratch": True,
            "source_sensor_family": "revo3_visiontouch_diff",
            "tactile_profile": "profile_a_force6d_diff",
            "checkpoint_family_id": "revo-profile-a-v1",
            "normalization_family_id": "revo-profile-a-norm-v1",
            "capability_manifest_sha256": self.capability_hash,
            "split_manifest_sha256": self.split_hash,
            "locked_test_opened": False,
            "input_shape": [5, 1, 240, 240],
            "num_fingers": 5,
            "encoder_state_complete": True,
            "source_splits": ["midtrain_train"],
            "train_episode_ids": train_episode_ids,
            "train_episode_ids_sha256": hashlib.sha256(
                json.dumps(train_episode_ids, separators=(",", ":")).encode()
            ).hexdigest(),
            "validation_split": "development",
            "validation_episode_ids": validation_episode_ids,
            "validation_episode_ids_sha256": hashlib.sha256(
                json.dumps(validation_episode_ids, separators=(",", ":")).encode()
            ).hexdigest(),
        })

        self.episodes = self.root / "episodes"
        instruction = "Grasp the centered bottle using a power grasp."
        for episode_id in ("ep0", "ep1"):
            _write(self.episodes / episode_id / "meta.json", {
                "episode_id": episode_id,
                "task_id": f"task-{episode_id}",
                "object_id": "bottle",
                "object_instance": f"bottle-{episode_id}",
                "operator": "operator", "collection_day": "day-1",
                "grasp_primitive": "POWER_GRASP",
                "instruction_sha256": hashlib.sha256(instruction.encode()).hexdigest(),
                "camera_profile_id": "revo3_full_center_v1",
                "camera_calibration_sha256": "b" * 64,
                "capability_manifest_sha256": self.capability_hash,
                "synthetic_fixture": False, "contains_emg": False,
                "action_label_source": "controller_target",
                "action_semantics": "accepted_exact_sent_teleop_target",
                "contains_cair_residual": False,
            })

        image_names = ["rgb_full.png", "rgb_center.png"] + [
            f"flare_{index}.png" for index in range(8)
        ] + [f"diff_{finger}.png" for finger in range(5)]
        for name in image_names:
            (self.root / name).write_bytes(b"fixture")
        diff_paths = [f"diff_{finger}.png" for finger in range(5)]
        history = [[[[0.0] * 6 for _ in range(5)] for _ in range(16)] for _ in range(4)]
        history_jitter = [[
            [[[0.0] * 6 for _ in range(5)] for _ in range(16)] for _ in range(3)
        ] for _ in range(4)]
        history_ts = [[[100 + sample for sample in range(16)] for _ in range(3)] for _ in range(4)]
        history_sequence = [[[sample for sample in range(16)] for _ in range(3)] for _ in range(4)]
        row = {
            "schema_version": "revo3-trex-json-v1",
            "episode_id": "ep0", "frame_index": 15,
            "input_prompt": instruction,
            "input_image_slow": ["rgb_full.png"],
            "input_image_fast": ["rgb_center.png"],
            "flare_image_full": [f"flare_{index}.png" for index in range(8)],
            "flare_timestamp_ns": list(range(8)),
            "state_fast": [0.0] * 21,
            "action": [0.0] * (16 * 21),
            "policy_loss_eligible": True,
            "action_write_timestamp_ns": list(range(2000, 2016)),
            "action_controller_sequence": list(range(16)),
            "action_request_id_hash": [f"{index:064x}" for index in range(16)],
            "tactile_profile": "profile_a_force6d_diff",
            "checkpoint_family_id": "revo-profile-a-v1",
            "normalization_family_id": "revo-profile-a-norm-v1",
            "tactile_f6": [0.0] * 30,
            "tactile_delay_offsets": [0, 4, 8, 12],
            "tactile_temporal_jitter_samples": [-1, 0, 1],
            "tactile_temporal_jitter_native_offsets": [-2, -1, 0],
            "tactile_decision_timestamp_ns_delayed": [1000] * 4,
            "tactile_f6_delayed": [[[0.0] * 6 for _ in range(5)] for _ in range(4)],
            "tactile_f6_history_delayed": history,
            "tactile_f6_delayed_jitter": [[[[0.0] * 6 for _ in range(5)] for _ in range(3)] for _ in range(4)],
            "tactile_f6_history_delayed_jitter": history_jitter,
            "touch_timestamp_ns_delayed_jitter": [[700, 800, 900] for _ in range(4)],
            "tactile_f6_history_timestamp_ns_delayed_jitter": history_ts,
            "tactile_f6_history_sequence_delayed_jitter": history_sequence,
            "tactile_image_deform": diff_paths,
            "tactile_image_deform_delayed": [diff_paths for _ in range(4)],
            "tactile_image_deform_delayed_jitter": [[diff_paths for _ in range(3)] for _ in range(4)],
            "tactile_deform_timestamp_ns_delayed_jitter": [[[700] * 5, [800] * 5, [900] * 5] for _ in range(4)],
            "rgb_full_timestamp_ns": 500, "rgb_center_timestamp_ns": 500,
            "rgb_receive_timestamp_ns": 600, "action_decision_timestamp_ns": 1000,
            "action_label_source": "controller_target",
            "action_semantics": "accepted_exact_sent_teleop_target",
            "contains_cair_residual": False, "contains_emg": False,
        }
        self.data = self.root / "revo3.json"
        self.dev_data = self.root / "revo3_dev.json"
        _write(self.data, [row]); _write(self.dev_data, [row])

        stats = {"revo3_single_hand": {
            "action": {"q01": [[0.0] * 21 for _ in range(16)], "q99": [[1.0] * 21 for _ in range(16)], "mask": [[True] * 21 for _ in range(16)]},
            "state": {"q01": [0.0] * 21, "q99": [1.0] * 21, "mask": [True] * 21},
            "tactile_f6": {"q01": [0.0] * 30, "q99": [1.0] * 30, "mask": [True] * 30},
            "tracking_error": {"mean": [0.0] * 21, "std": [0.01] * 21},
            "tactile_no_contact_noise": {
                "robust_center": [[0.0] * 6 for _ in range(5)],
                "robust_scale": [[0.01] * 6 for _ in range(5)],
                "covariance": [[0.0] * 30 for _ in range(30)],
                "normalization_family_id": "revo-profile-a-norm-v1",
                "checkpoint_family_id": "revo-profile-a-v1",
            },
        }}
        self.stats = self.root / "revo3_statistics.json"
        _write(self.stats, stats)
        self.stats_artifact = self.root / "revo3_statistics_artifact.json"
        _write(self.stats_artifact, {
            "schema_version": "revo3-normalization-artifact-v1",
            "statistics_path": str(self.stats.resolve()),
            "statistics_sha256": LAUNCHER._sha256_file(self.stats),
            "source_split": "midtrain_train",
            "split_manifest_sha256": self.split_hash,
            "stats_episode_ids": ["ep0", "ep1"],
            "joint_order_hash": LAUNCHER.JOINT_ORDER_HASH,
            "tactile_profile": "profile_a_force6d_diff",
            "checkpoint_family_id": "revo-profile-a-v1",
            "normalization_family_id": "revo-profile-a-norm-v1",
            "capability_manifest_sha256": self.capability_hash,
        })

        base_conversion = {
            "schema_version": "revo3-trex-conversion-v1",
            "action_shape": [16, 21], "state_shape": [21],
            "tactile_shape": [5, 6], "tactile_history_shape": [16, 5, 6],
            "tactile_deform_shape": [5, 1, 240, 240],
            "tactile_delay_offsets": [0, 4, 8, 12],
            "tactile_temporal_jitter_samples": [-1, 0, 1],
            "tactile_temporal_jitter_native_offsets": [-2, -1, 0],
            "action_grid_hz": 30, "training_anchor_hz": 10,
            "flare_steps": 8, "flare_frame_stride": 4, "flare_padding": "forbidden",
            "view_slots": {"slow": "full", "fast": "fixed_center"},
            "view_shape_hwc": [288, 384, 3], "views_share_timestamp": True,
            "terminal_padding": "forbidden",
            "tactile_profile": "profile_a_force6d_diff",
            "checkpoint_family_id": "revo-profile-a-v1",
            "normalization_family_id": "revo-profile-a-norm-v1",
            "capability_manifest_sha256": self.capability_hash,
            "action_label_source": "controller_target",
            "action_semantics": "accepted_exact_sent_teleop_target",
            "contains_cair_residual": False, "contains_emg": False,
            "split_before_statistics": True, "duration_targets_enforced": True,
            "normalization_frozen": True,
            "statistics_source_split": "midtrain_train",
            "split_manifest_sha256": self.split_hash,
            "episode_ids": ["ep0", "ep1"], "source_root": str(self.episodes),
            "stats_episode_ids": ["ep0", "ep1"],
            "stats_path": str(self.stats.resolve()),
            "stats_artifact_path": str(self.stats_artifact.resolve()),
            "statistics_sha256": LAUNCHER._sha256_file(self.stats),
            "statistics_artifact_sha256": LAUNCHER._sha256_file(self.stats_artifact),
        }
        self.conversion = self.root / "conversion.json"
        self.dev_conversion = self.root / "dev_conversion.json"
        _write(self.conversion, {**base_conversion, "dataset_split": "midtrain_train"})
        _write(self.dev_conversion, {**base_conversion, "dataset_split": "development"})

        readiness = {
            "schema_version": "revo3-vla-readiness-v1", "dataset_kind": "real_robot",
            "ready_for_training": True, "synthetic_fixture": False,
            "contains_emg": False, "action_label_source": "controller_target",
            "action_semantics": "accepted_exact_sent_teleop_target",
            "contains_cair_residual": False, "timestamp_alignment_verified": True,
            "split_before_statistics_verified": True, "duration_targets_reviewed": True,
            "replay_gate_passed": True,
        }
        self.readiness = self.root / "readiness.json"
        self.dev_readiness = self.root / "dev_readiness.json"
        _write(self.readiness, readiness); _write(self.dev_readiness, readiness)

    def tearDown(self):
        self.temp.cleanup()

    def _train_args(self, **updates):
        values = dict(
            base_model=self.base, checkpoint=self.checkpoint,
            checkpoint_id="miniFranka/T-Rex_pretrain_mecka22k_epoch1",
            resume_kind="official_pretrain", mode="main",
            ack_heterogeneous_midtrain_ablation=False,
            data_json=self.data, conversion_manifest=self.conversion,
            readiness_manifest=self.readiness,
            development_data_json=self.dev_data,
            development_conversion_manifest=self.dev_conversion,
            development_readiness_manifest=self.dev_readiness,
            output_dir=self.output, num_processes=1,
            accelerate_config=REPO_ROOT / "config" / "sft_qwen.yaml",
            run_name="unit_test", stage="w0",
            tactile_profile="profile_a_force6d_diff",
            tactile_profile_manifest=self.profile,
            vqvae_checkpoint=self.vqvae, vqvae_artifact=self.vq_artifact,
            deform_encoder_checkpoint=self.deform,
            deform_encoder_artifact=self.deform_artifact,
            ack_tactile_ablation=False,
        )
        values.update(updates)
        return type("Args", (), values)()

    def _midtrain_ablation_args(self, **updates):
        values = dict(stage="midtrain", mode="midtrain_ablation",
            resume_kind="official_midtrain_ablation",
            checkpoint_id="miniFranka/T-Rex_midtrain_mecka23k_ucb100_vqvae_epoch6")
        values.update(updates)
        return self._train_args(**values)

    def test_primary_command_freezes_revo_contract_and_pretrain_route(self):
        config = LAUNCHER.load_launch_config(REPO_ROOT / "config" / "revo3_v1_trex.json")
        joined = " ".join(LAUNCHER.build_train_command(self._train_args(), config))
        for fragment in ("--action_dim 21", "--action_chunk 16", "--tactile_num_fingers 5",
                         "--use_robot_state 1", "--use_tactile_vec 0",
                         "--use_tactile_deform 0", "--use_tactile_vqvae 0",
                         "--revo_training_stage w0", "--max_steps 1000",
                         "--flare_loss_weight 0.0", "--resume_source official_pretrain"):
            self.assertIn(fragment, joined)
        self.assertNotIn("emg", joined.lower())

    def test_synthetic_or_unapproved_data_is_rejected(self):
        payload = json.loads(self.readiness.read_text())
        payload.update(synthetic_fixture=True, dataset_kind="synthetic")
        _write(self.readiness, payload)
        config = LAUNCHER.load_launch_config(REPO_ROOT / "config" / "revo3_v1_trex.json")
        with self.assertRaisesRegex(LAUNCHER.LaunchContractError, "readiness gate"):
            LAUNCHER.build_train_command(self._train_args(), config)

    def test_mock_corpus_is_rejected_even_with_forged_ready_flag(self):
        _write(self.episodes / "corpus_meta.json", {"synthetic_fixture": True})
        config = LAUNCHER.load_launch_config(REPO_ROOT / "config" / "revo3_v1_trex.json")
        with self.assertRaisesRegex(LAUNCHER.LaunchContractError, "smoke-only"):
            LAUNCHER.build_train_command(self._train_args(), config)

    def test_midtrain_checkpoint_requires_explicit_ablation_ack(self):
        config = LAUNCHER.load_launch_config(REPO_ROOT / "config" / "revo3_v1_trex.json")
        with self.assertRaisesRegex(LAUNCHER.LaunchContractError, "ablation only"):
            LAUNCHER.build_train_command(self._midtrain_ablation_args(), config)

    def test_raw_midtrain_is_rejected_even_after_ablation_ack(self):
        _write(self.checkpoint / "training_args.json", {
            "action_dim": 62, "action_chunk": 16, "tactile_num_fingers": 10,
            "use_robot_state": 0, "use_tactile_vec": 1, "use_tactile_deform": 1,
            "use_tactile_vqvae": 1, "use_tactile_code": 1,
        })
        config = LAUNCHER.load_launch_config(REPO_ROOT / "config" / "revo3_v1_trex.json")
        with self.assertRaisesRegex(LAUNCHER.LaunchContractError, "not launchable"):
            LAUNCHER.build_train_command(self._midtrain_ablation_args(
                ack_heterogeneous_midtrain_ablation=True), config)

    def _write_migrated_midtrain(self):
        _write(self.checkpoint / "training_args.json", {
            "action_dim": 21, "action_chunk": 16, "tactile_num_fingers": 5,
            "use_robot_state": 1, "use_tactile_vec": 1, "use_tactile_deform": 0,
            "use_tactile_vqvae": 0, "use_tactile_code": 0,
        })
        _write(self.checkpoint / "revo3_migration.json", {
            "schema_version": "revo3-trex-midtrain-migration-v1",
            "source_checkpoint_id": "miniFranka/T-Rex_midtrain_mecka23k_ucb100_vqvae_epoch6",
            "source_tactile_num_fingers": 10, "target_tactile_num_fingers": 5,
            "target_action_dim": 21, "removed_embedded_vqvae": True,
            "reinitialized_shape_mismatched_heads": True, "migration_test_passed": True,
        })

    def _write_serve_checkpoint(
        self,
        *,
        tactile_profile="profile_a_force6d_diff",
        stats_path=None,
        stats_artifact_path=None,
    ):
        stats_path = Path(stats_path or self.stats)
        stats_artifact_path = Path(stats_artifact_path or self.stats_artifact)
        is_diff_only = tactile_profile == "profile_b_diff_only"
        checkpoint_family = "revo-profile-b-v1" if is_diff_only else "revo-profile-a-v1"
        normalization_family = (
            "revo-profile-b-norm-v1" if is_diff_only else "revo-profile-a-norm-v1"
        )
        profile = self.root / f"{tactile_profile}.json"
        _write(profile, {
            "schema_version": "revo3-tactile-profile-v1",
            "profile": tactile_profile,
            "approved_for_training": True,
            "capability_manifest_sha256": self.capability_hash,
            "checkpoint_family_id": checkpoint_family,
            "normalization_family_id": normalization_family,
        })
        profile_hash = LAUNCHER._sha256_file(profile)
        (self.checkpoint / "processor").mkdir(exist_ok=True)
        training_args = {
            "action_dim": 21, "action_chunk": 16, "tactile_num_fingers": 5,
            "use_robot_state": 1,
            "use_tactile_vec": 0 if is_diff_only else 1,
            "use_tactile_deform": 1,
            "use_tactile_vqvae": 0 if is_diff_only else 1,
            "use_tactile_code": 0 if is_diff_only else 1,
            "tactile_profile": tactile_profile,
            "checkpoint_family_id": checkpoint_family,
            "normalization_family_id": normalization_family,
            "normalization_statistics_sha256": LAUNCHER._sha256_file(stats_path),
            "normalization_artifact_sha256": LAUNCHER._sha256_file(stats_artifact_path),
            "split_manifest_sha256": self.split_hash,
            "tactile_profile_manifest_sha256": profile_hash,
            "capability_manifest_sha256": self.capability_hash,
            "revo_training_stage": "sft",
            "camera_profile": "revo3_full_center_v1",
            "view_slots": {"slow": "full", "fast": "fixed_center"},
        }
        _write(self.checkpoint / "training_args.json", training_args)
        _write(self.checkpoint / "checkpoint_lineage.json", {
            "schema_version": "revo3-checkpoint-lineage-v1",
            "checkpoint_sha256": LAUNCHER._sha256_file(self.checkpoint / "model.pt"),
            "stage": "sft",
            "split_manifest_sha256": self.split_hash,
            "normalization_statistics_sha256": training_args["normalization_statistics_sha256"],
            "normalization_artifact_sha256": training_args["normalization_artifact_sha256"],
            "capability_manifest_sha256": self.capability_hash,
            "tactile_profile": tactile_profile,
            "checkpoint_family_id": checkpoint_family,
            "normalization_family_id": normalization_family,
            "tactile_profile_manifest_sha256": profile_hash,
            "joint_order_hash": LAUNCHER.JOINT_ORDER_HASH,
            "camera_profile": "revo3_full_center_v1",
            "view_slots": {"slow": "full", "fast": "fixed_center"},
        })
        return profile, training_args

    def test_verified_revo_migration_can_build_midtrain_ablation(self):
        self._write_migrated_midtrain()
        config = LAUNCHER.load_launch_config(REPO_ROOT / "config" / "revo3_v1_trex.json")
        joined = " ".join(LAUNCHER.build_train_command(self._midtrain_ablation_args(
            ack_heterogeneous_midtrain_ablation=True), config))
        self.assertIn("--resume_source official_midtrain_ablation", joined)
        self.assertIn("--vqvae_artifact", joined)
        self.assertIn("--deform_encoder_artifact", joined)

    def test_tactile_artifact_hash_mismatch_is_rejected(self):
        self._write_migrated_midtrain()
        payload = json.loads(self.vq_artifact.read_text())
        payload["checkpoint_sha256"] = "0" * 64
        _write(self.vq_artifact, payload)
        config = LAUNCHER.load_launch_config(REPO_ROOT / "config" / "revo3_v1_trex.json")
        with self.assertRaisesRegex(LAUNCHER.LaunchContractError, "provenance mismatch"):
            LAUNCHER.build_train_command(self._midtrain_ablation_args(
                ack_heterogeneous_midtrain_ablation=True), config)

    def test_serve_rejects_non_revo_checkpoint(self):
        (self.checkpoint / "processor").mkdir()
        _write(self.checkpoint / "training_args.json", {
            "action_dim": 62, "action_chunk": 16, "tactile_num_fingers": 10,
            "use_robot_state": 0, "use_tactile_vec": 1, "use_tactile_deform": 1,
            "use_tactile_vqvae": 1, "use_tactile_code": 1,
        })
        config = LAUNCHER.load_launch_config(REPO_ROOT / "config" / "revo3_v1_trex.json")
        with self.assertRaisesRegex(LAUNCHER.LaunchContractError, "not the reviewed"):
            LAUNCHER.validate_serve_checkpoint(self.checkpoint, config)

    def test_serve_command_is_fixed_to_revo_contract(self):
        self._write_serve_checkpoint()
        config = LAUNCHER.load_launch_config(REPO_ROOT / "config" / "revo3_v1_trex.json")
        identity_manifest = self.root / "server_identity.json"
        args = type("Args", (), {"base_model": self.base, "checkpoint": self.checkpoint,
            "stats_path": self.stats, "stats_artifact_path": self.stats_artifact,
            "identity_manifest_out": identity_manifest,
            "cuda": "0", "port": 5555})()
        joined = " ".join(LAUNCHER.build_serve_command(args, config))
        for fragment in ("--action_dim 21", "--tactile_num_fingers 5",
                         "--use_robot_state 1", "--camera_profile revo3_full_center_v1",
                         "--tactile_profile profile_a_force6d_diff", "--image_size 384 288"):
            self.assertIn(fragment, joined)
        self.assertNotIn("emg", joined.lower())
        identity = json.loads(identity_manifest.read_text(encoding="utf-8"))
        self.assertEqual(identity["checkpoint_sha256"], LAUNCHER._sha256_file(self.checkpoint / "model.pt"))
        self.assertEqual(identity["tactile_profile"], "profile_a_force6d_diff")
        self.assertEqual(len(identity["identity_sha256"]), 64)

    def test_profile_b_diff_only_command_parses_and_model_loads_without_f6_stats(self):
        stats = self.root / "profile_b_statistics.json"
        _write(stats, {"revo3_single_hand": {
            "action": {"q01": [[0.0] * 21 for _ in range(16)], "q99": [[1.0] * 21 for _ in range(16)], "mask": [[True] * 21 for _ in range(16)]},
            "state": {"q01": [0.0] * 21, "q99": [1.0] * 21, "mask": [True] * 21},
            "tracking_error": {"mean": [0.0] * 21, "std": [0.01] * 21},
        }})
        artifact = self.root / "profile_b_statistics_artifact.json"
        _write(artifact, {
            "schema_version": "revo3-normalization-artifact-v1",
            "statistics_path": str(stats.resolve()),
            "statistics_sha256": LAUNCHER._sha256_file(stats),
            "source_split": "midtrain_train",
            "split_manifest_sha256": self.split_hash,
            "stats_episode_ids": ["ep0", "ep1"],
            "joint_order_hash": LAUNCHER.JOINT_ORDER_HASH,
            "tactile_profile": "profile_b_diff_only",
            "checkpoint_family_id": "revo-profile-b-v1",
            "normalization_family_id": "revo-profile-b-norm-v1",
            "capability_manifest_sha256": self.capability_hash,
        })
        self._write_serve_checkpoint(
            tactile_profile="profile_b_diff_only",
            stats_path=stats,
            stats_artifact_path=artifact,
        )
        config = LAUNCHER.load_launch_config(REPO_ROOT / "config" / "revo3_v1_trex.json")
        args = type("Args", (), {"base_model": self.base, "checkpoint": self.checkpoint,
            "stats_path": stats, "stats_artifact_path": artifact,
            "cuda": "0", "port": 5555})()
        command = LAUNCHER.build_serve_command(args, config)
        parsed = SERVER.build_server_argument_parser().parse_args(command[2:])
        fake_model = mock.Mock()
        fake_model.load_state_dict.return_value = ([], [])
        fake_model.to.return_value = fake_model
        fake_model.tactile_vqvae = None
        with mock.patch.object(SERVER.AutoProcessor, "from_pretrained", return_value=object()), \
             mock.patch.object(SERVER.Qwen3VLVLAModel, "from_pretrained_qwen3vl", return_value=fake_model), \
             mock.patch.object(SERVER.torch, "load", return_value={}):
            _, _, loaded = SERVER.model_load(parsed)
        self.assertNotIn("tacf6_mask", loaded)
        self.assertEqual(loaded["state_min"].shape, (21,))
        identity = parsed._server_identity
        self.assertEqual(identity["checkpoint_sha256"], LAUNCHER._sha256_file(self.checkpoint / "model.pt"))
        self.assertEqual(identity["normalization_statistics_sha256"], LAUNCHER._sha256_file(stats))
        self.assertEqual(identity["normalization_artifact_sha256"], LAUNCHER._sha256_file(artifact))
        server = SERVER.CascadedServer(parsed, fake_model, object(), loaded)
        reply = server.predict("identity", {"server_identity": {"checkpoint_sha256": "evil"}})
        self.assertEqual(reply["server_identity"], identity)
        self.assertNotEqual(reply["server_identity"], {"checkpoint_sha256": "evil"})

    def test_serve_rejects_same_shape_wrong_statistics(self):
        self._write_serve_checkpoint()
        wrong_stats = self.root / "wrong_statistics.json"
        payload = json.loads(self.stats.read_text())
        payload["revo3_single_hand"]["action"]["q01"][0][0] = 0.123
        _write(wrong_stats, payload)
        args = type("Args", (), {"base_model": self.base, "checkpoint": self.checkpoint,
            "stats_path": wrong_stats, "stats_artifact_path": self.stats_artifact,
            "cuda": "0", "port": 5555})()
        config = LAUNCHER.load_launch_config(REPO_ROOT / "config" / "revo3_v1_trex.json")
        with self.assertRaisesRegex(LAUNCHER.LaunchContractError, "normalization lineage"):
            LAUNCHER.build_serve_command(args, config)


if __name__ == "__main__":
    unittest.main()
