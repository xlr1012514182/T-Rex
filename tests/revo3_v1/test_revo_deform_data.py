import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from revo3_v1.data import SyntheticRevoConfig, generate_synthetic_revo_episodes


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "train_revo_deform_ae", REPO_ROOT / "scripts" / "train_revo_deform_ae.py"
)
DEFORM = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(DEFORM)


def test_deform_training_opens_only_midtrain_and_development(tmp_path):
    episodes = tmp_path / "episodes"
    summary = generate_synthetic_revo_episodes(
        episodes,
        SyntheticRevoConfig(
            episodes_per_task=1,
            frames_per_episode=64,
            tactile_profile="profile_b_diff_only",
        ),
    )
    ids = summary["episode_ids"]
    splits = ("midtrain_train", "sft_train", "development", "locked_test")
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps({
        "schema_version": "revo3-corpus-split-v1",
        "episodes": [
            {
                "episode_id": episode_id,
                "split": split,
                "day": f"day-{index}",
                "object_instance": f"object-{index}",
                "operator": f"operator-{index}",
                "duration_seconds": 64 / 30,
            }
            for index, (episode_id, split) in enumerate(zip(ids, splits))
        ],
        "evaluation_protocol": {
            "locked_test_isolation": ["day", "object_instance"]
        },
        "duration_targets_hours": {
            "midtrain_train": 12.0,
            "sft_train": 4.0,
            "development": 2.0,
            "locked_test": 2.0,
        },
        "duration_targets_enforced": False,
    }), encoding="utf-8")
    args = SimpleNamespace(
        data_root=episodes,
        split_manifest=split_path,
        tactile_profile="profile_b_diff_only",
        checkpoint_family_id="synthetic-profile_b_diff_only-v1",
        normalization_family_id="synthetic-profile_b_diff_only-norm-v1",
        capability_manifest_sha256="a" * 64,
    )
    opened = []
    original_load = DEFORM.RevoEpisode.load

    def tracked_load(path):
        opened.append(Path(path).name)
        return original_load(path)

    with mock.patch.object(DEFORM.RevoEpisode, "load", side_effect=tracked_load):
        _, train, development = DEFORM.load_revo_deform_splits(args)

    assert [episode.meta.episode_id for episode in train] == [ids[0]]
    assert [episode.meta.episode_id for episode in development] == [ids[2]]
    assert opened == [ids[0], ids[2]]
    assert ids[1] not in opened  # SFT is downstream of the frozen DIFF encoder.
    assert ids[3] not in opened  # locked_test is never available for fitting/selection.
