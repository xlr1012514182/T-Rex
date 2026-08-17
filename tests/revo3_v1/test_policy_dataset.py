import numpy as np
import pytest

from revo3_v1.policy import (
    RevoEpisode,
    RevoEpisodeAdapter,
    RevoNormStats,
    build_revo_feature_schema,
)


def _episode(length=40, metadata=None):
    tactile = np.zeros((length, 5, 6), dtype=np.float32)
    history = np.repeat(tactile[:, None], 16, axis=1)
    return RevoEpisode(
        episode_id="ep0",
        timestamp_ns=np.arange(length, dtype=np.int64) * 33_333_333,
        q_state_rad=np.zeros((length, 21), dtype=np.float32),
        q_action_abs_rad=np.arange(length, dtype=np.float32)[:, None]
        * np.ones((1, 21), dtype=np.float32),
        tactile_f6=tactile,
        tactile_history_f6=history,
        instruction="Grasp the bottle.",
        full_rgb=np.zeros((length, 8, 8, 3), dtype=np.uint8),
        center_rgb=np.zeros((length, 8, 8, 3), dtype=np.uint8),
        metadata={} if metadata is None else metadata,
    )


def test_revo_feature_schema_is_21d_single_hand():
    schema = build_revo_feature_schema()
    assert schema["observation.state"]["shape"] == (21,)
    assert schema["action"]["shape"] == (16, 21)
    assert schema["observation.tactile_f6"]["shape"] == (5, 6)
    assert schema["observation.tactile_history_f6"]["shape"] == (16, 5, 6)


def test_episode_adapter_forms_causal_history_and_future_chunk():
    sample = RevoEpisodeAdapter(_episode()).sample(15)
    assert sample.state_q_rad.shape == (21,)
    assert sample.action_chunk_abs_rad.shape == (16, 21)
    assert sample.tactile_history_f6.shape == (16, 5, 6)
    np.testing.assert_allclose(sample.action_chunk_abs_rad[:, 0], np.arange(15, 31))
    item = sample.as_trex_item()
    assert item["action"].shape == (16, 21)
    assert item["observation.tactile_f6"].shape == (5, 6)
    assert item["observation.tactile_history_f6"].shape == (16, 5, 6)


def test_episode_never_accepts_emg_as_vla_feature():
    with pytest.raises(ValueError, match="EMG"):
        _episode(metadata={"raw_emg": "forbidden"})


def test_episode_rejects_nested_emg_and_missing_center_view():
    with pytest.raises(ValueError, match="EMG"):
        _episode(metadata={"diagnostics": {"EMG_quality": 1.0}})
    episode = _episode()
    with pytest.raises(ValueError, match="atomic two-view"):
        RevoEpisode(
            episode_id=episode.episode_id,
            timestamp_ns=episode.timestamp_ns,
            q_state_rad=episode.q_state_rad,
            q_action_abs_rad=episode.q_action_abs_rad,
            tactile_f6=episode.tactile_f6,
            tactile_history_f6=episode.tactile_history_f6,
            instruction=episode.instruction,
            full_rgb=episode.full_rgb,
            center_rgb=None,
        )


def test_norm_stats_have_revo_dimensions():
    stats = RevoNormStats.fit_train_episodes([_episode()])
    assert stats.state_q01.shape == (21,)
    assert stats.action_q99.shape == (16, 21)
    assert stats.tactile_q01.shape == (30,)
    assert stats.tracking_error_mean.shape == (21,)
