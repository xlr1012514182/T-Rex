import numpy as np
import pytest

from revo3_v1.policy import (
    ActionChunk,
    ActionTemporalAggregator,
    InferenceMode,
    TaskKey,
    TemporalAggregationError,
)


def _key(version=0):
    return TaskKey.from_instruction(
        task_id="task", task_version=version,
        instruction="Grasp the bottle.", lease_id="lease",
    )


def _chunk(value, generated, *, key=None, chunk_id="stable", start=0):
    return ActionChunk(
        q_target_rad=np.full((16, 21), value, dtype=np.float32),
        start_step=start,
        generated_ns=generated,
        observation_timestamp_ns=generated,
        task_key=_key() if key is None else key,
        mode=InferenceMode.SLOW_AND_FAST if generated == 100 else InferenceMode.FAST,
        chunk_id=chunk_id,
    )


def test_main_default_uniformly_aggregates_slow_and_fast_revisions():
    agg = ActionTemporalAggregator(k=0.0)
    agg.add(_chunk(1.0, 100), now_ns=100)
    agg.add(_chunk(3.0, 110), now_ns=110)
    np.testing.assert_allclose(agg.target_at(4, task_key=_key(), now_ns=120), 2.0)


def test_positive_k_weights_newest_revision_more_heavily():
    agg = ActionTemporalAggregator(k=1.0)
    agg.add(_chunk(1.0, 100), now_ns=100)
    agg.add(_chunk(3.0, 110), now_ns=110)
    target = agg.target_at(4, task_key=_key(), now_ns=120)
    assert np.all(target > 2.0)
    assert np.all(target < 3.0)


def test_aggregator_rejects_cross_task_and_stale_chunks():
    agg = ActionTemporalAggregator(max_chunk_age_ns=100)
    agg.add(_chunk(1.0, 100), now_ns=100)
    with pytest.raises(TemporalAggregationError, match="mismatch"):
        agg.target_at(0, task_key=_key(version=1), now_ns=110)
    with pytest.raises(TemporalAggregationError, match="non-stale"):
        agg.target_at(0, task_key=_key(), now_ns=1_000)


def test_new_slow_chunk_resets_main_style_revision_buffer():
    agg = ActionTemporalAggregator(k=0.0)
    agg.add(_chunk(1.0, 100), now_ns=100)
    agg.add(_chunk(5.0, 200, chunk_id="next", start=16), now_ns=200)
    np.testing.assert_allclose(agg.target_at(16, task_key=_key(), now_ns=200), 5.0)
