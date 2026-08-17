import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "revo3_v1_trex.py"
SPEC = importlib.util.spec_from_file_location("revo3_v1_trex_launcher", SCRIPT)
LAUNCHER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(LAUNCHER)


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

        self.data = self.root / "revo3.json"
        row = {
            "schema_version": "revo3-trex-json-v1",
            "episode_id": "ep0",
            "frame_index": 0,
            "input_prompt": "Grasp the bottle.",
            "input_image_slow": ["rgb.png"],
            "state_fast": [0.0] * 21,
            "action": [0.0] * (16 * 21),
            "tactile_f6": [0.0] * (5 * 6),
            "action_label_source": "controller_target",
            "contains_emg": False,
        }
        _write(self.data, [row])
        stats = {
            "revo3_single_hand": {
                "action": {
                    "q01": [[0.0] * 21 for _ in range(16)],
                    "q99": [[1.0] * 21 for _ in range(16)],
                    "mask": [[True] * 21 for _ in range(16)],
                },
                "state": {"q01": [0.0] * 21, "q99": [1.0] * 21, "mask": [True] * 21},
                "tactile_f6": {
                    "q01": [0.0] * 30,
                    "q99": [1.0] * 30,
                    "mask": [True] * 30,
                },
            }
        }
        self.stats = self.root / "revo3_statistics.json"
        _write(self.stats, stats)
        self.conversion = self.root / "conversion.json"
        self.episodes = self.root / "episodes"
        for episode_id in ("ep0", "ep1"):
            _write(
                self.episodes / episode_id / "meta.json",
                {
                    "episode_id": episode_id,
                    "synthetic_fixture": False,
                    "contains_emg": False,
                    "action_label_source": "controller_target",
                },
            )
        _write(
            self.conversion,
            {
                "schema_version": "revo3-trex-conversion-v1",
                "action_shape": [16, 21],
                "state_shape": [21],
                "tactile_shape": [5, 6],
                "action_label_source": "controller_target",
                "contains_emg": False,
                "episode_ids": ["ep0", "ep1"],
                "source_root": str(self.episodes),
                "stats_path": str(self.stats),
            },
        )
        self.readiness = self.root / "readiness.json"
        _write(
            self.readiness,
            {
                "schema_version": "revo3-vla-readiness-v1",
                "dataset_kind": "real_robot",
                "ready_for_training": True,
                "synthetic_fixture": False,
                "contains_emg": False,
                "action_label_source": "controller_target",
                "timestamp_alignment_verified": True,
                "replay_gate_passed": True,
            },
        )

    def tearDown(self):
        self.temp.cleanup()

    def _train_args(self, **updates):
        values = dict(
            base_model=self.base,
            checkpoint=self.checkpoint,
            checkpoint_id="miniFranka/T-Rex_pretrain_mecka22k_epoch1",
            mode="main",
            ack_heterogeneous_midtrain_ablation=False,
            data_json=self.data,
            conversion_manifest=self.conversion,
            readiness_manifest=self.readiness,
            output_dir=self.output,
            num_processes=1,
            accelerate_config=REPO_ROOT / "config" / "sft_qwen.yaml",
            run_name="unit_test",
        )
        values.update(updates)
        return type("Args", (), values)()

    def test_primary_command_freezes_revo_contract_and_pretrain_route(self):
        config = LAUNCHER.load_launch_config(REPO_ROOT / "config" / "revo3_v1_trex.json")
        command = LAUNCHER.build_train_command(self._train_args(), config)
        joined = " ".join(command)
        for fragment in (
            "--action_dim 21",
            "--action_chunk 16",
            "--tactile_num_fingers 5",
            "--use_robot_state 1",
            "--use_tactile_vec 1",
            "--use_tactile_deform 0",
            "--use_tactile_vqvae 0",
            "--resume_source pretrain",
        ):
            self.assertIn(fragment, joined)
        self.assertNotIn("emg", joined.lower())

    def test_synthetic_or_unapproved_data_is_rejected(self):
        payload = json.loads(self.readiness.read_text(encoding="utf-8"))
        payload["synthetic_fixture"] = True
        payload["dataset_kind"] = "synthetic"
        _write(self.readiness, payload)
        config = LAUNCHER.load_launch_config(REPO_ROOT / "config" / "revo3_v1_trex.json")
        with self.assertRaisesRegex(LAUNCHER.LaunchContractError, "readiness gate"):
            LAUNCHER.build_train_command(self._train_args(), config)

    def test_mock_corpus_is_rejected_even_with_forged_ready_flag(self):
        _write(
            self.episodes / "corpus_meta.json",
            {"schema_version": "revo3-synthetic-corpus-v1", "synthetic_fixture": True},
        )
        config = LAUNCHER.load_launch_config(REPO_ROOT / "config" / "revo3_v1_trex.json")
        with self.assertRaisesRegex(LAUNCHER.LaunchContractError, "smoke-only"):
            LAUNCHER.build_train_command(self._train_args(), config)

    def test_midtrain_checkpoint_requires_explicit_ablation_ack(self):
        config = LAUNCHER.load_launch_config(REPO_ROOT / "config" / "revo3_v1_trex.json")
        args = self._train_args(
            mode="midtrain_ablation",
            checkpoint_id="miniFranka/T-Rex_midtrain_mecka23k_ucb100_vqvae_epoch6",
        )
        with self.assertRaisesRegex(LAUNCHER.LaunchContractError, "ablation only"):
            LAUNCHER.build_train_command(args, config)

    def test_raw_midtrain_is_rejected_even_after_ablation_ack(self):
        _write(
            self.checkpoint / "training_args.json",
            {
                "action_dim": 62,
                "action_chunk": 16,
                "tactile_num_fingers": 10,
                "use_robot_state": 0,
                "use_tactile_vec": 1,
                "use_tactile_deform": 1,
                "use_tactile_vqvae": 1,
                "use_tactile_code": 1,
            },
        )
        config = LAUNCHER.load_launch_config(REPO_ROOT / "config" / "revo3_v1_trex.json")
        args = self._train_args(
            mode="midtrain_ablation",
            checkpoint_id="miniFranka/T-Rex_midtrain_mecka23k_ucb100_vqvae_epoch6",
            ack_heterogeneous_midtrain_ablation=True,
        )
        with self.assertRaisesRegex(LAUNCHER.LaunchContractError, "not launchable"):
            LAUNCHER.build_train_command(args, config)

    def test_verified_revo_migration_can_build_midtrain_ablation(self):
        _write(
            self.checkpoint / "training_args.json",
            {
                "action_dim": 21,
                "action_chunk": 16,
                "tactile_num_fingers": 5,
                "use_robot_state": 1,
                "use_tactile_vec": 1,
                "use_tactile_deform": 0,
                "use_tactile_vqvae": 0,
                "use_tactile_code": 0,
            },
        )
        _write(
            self.checkpoint / "revo3_migration.json",
            {
                "schema_version": "revo3-trex-midtrain-migration-v1",
                "source_checkpoint_id": "miniFranka/T-Rex_midtrain_mecka23k_ucb100_vqvae_epoch6",
                "source_tactile_num_fingers": 10,
                "target_tactile_num_fingers": 5,
                "target_action_dim": 21,
                "removed_embedded_vqvae": True,
                "reinitialized_shape_mismatched_heads": True,
                "migration_test_passed": True,
            },
        )
        config = LAUNCHER.load_launch_config(REPO_ROOT / "config" / "revo3_v1_trex.json")
        args = self._train_args(
            mode="midtrain_ablation",
            checkpoint_id="miniFranka/T-Rex_midtrain_mecka23k_ucb100_vqvae_epoch6",
            ack_heterogeneous_midtrain_ablation=True,
        )
        command = LAUNCHER.build_train_command(args, config)
        self.assertIn("--resume_source midtrain", " ".join(command))

    def test_serve_rejects_non_revo_checkpoint(self):
        (self.checkpoint / "processor").mkdir()
        _write(
            self.checkpoint / "training_args.json",
            {
                "action_dim": 62,
                "action_chunk": 16,
                "tactile_num_fingers": 10,
                "use_robot_state": 0,
                "use_tactile_vec": 1,
                "use_tactile_deform": 1,
                "use_tactile_vqvae": 1,
                "use_tactile_code": 1,
            },
        )
        with self.assertRaisesRegex(LAUNCHER.LaunchContractError, "not the reviewed"):
            LAUNCHER.validate_serve_checkpoint(self.checkpoint)

    def test_serve_command_is_fixed_to_revo_contract(self):
        (self.checkpoint / "processor").mkdir()
        _write(
            self.checkpoint / "training_args.json",
            {
                "action_dim": 21,
                "action_chunk": 16,
                "tactile_num_fingers": 5,
                "use_robot_state": 1,
                "use_tactile_vec": 1,
                "use_tactile_deform": 0,
                "use_tactile_vqvae": 0,
                "use_tactile_code": 0,
            },
        )
        config = LAUNCHER.load_launch_config(REPO_ROOT / "config" / "revo3_v1_trex.json")
        args = type(
            "Args",
            (),
            {
                "base_model": self.base,
                "checkpoint": self.checkpoint,
                "stats_path": self.stats,
                "cuda": "0",
                "port": 5555,
            },
        )()
        joined = " ".join(LAUNCHER.build_serve_command(args, config))
        self.assertIn("--action_dim 21", joined)
        self.assertIn("--tactile_num_fingers 5", joined)
        self.assertIn("--use_robot_state 1", joined)
        self.assertNotIn("emg", joined.lower())


if __name__ == "__main__":
    unittest.main()
