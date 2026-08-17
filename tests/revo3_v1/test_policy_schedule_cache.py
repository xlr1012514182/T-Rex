import numpy as np
import pytest

from revo3_v1.policy import (
    CacheProtocolError,
    InferenceMode,
    MAIN_ALIGNED_SCHEDULE,
    MockTReXBackend,
    PolicyObservation,
    PolicyRequest,
    TReXRevoPolicyAdapter,
    TReXPolicyRunner,
    TaskKey,
)


def _request(mode, offset, timestamp, *, task_version=0):
    instruction = "Grasp the centered bottle."
    key = TaskKey.from_instruction(
        task_id="task", task_version=task_version,
        instruction=instruction, lease_id="lease",
    )
    obs = PolicyObservation(
        timestamp_ns=timestamp,
        state_timestamp_ns=timestamp,
        rgb_timestamp_ns=timestamp,
        tactile_timestamp_ns=timestamp,
        q_rad=np.zeros(21),
        tactile_f6=np.zeros((5, 6)),
        tactile_history_f6=np.zeros((16, 5, 6)),
        instruction=instruction,
        task_key=key,
    )
    return PolicyRequest(mode, offset, obs)


def test_main_schedule_matches_public_client_defaults():
    schedule = MAIN_ALIGNED_SCHEDULE
    assert schedule.command_hz == 30.0
    assert schedule.chunk_size == 16
    assert schedule.execute_steps_per_chunk == 16
    assert schedule.refine_offsets == (4, 8, 12)
    assert [schedule.mode_at_chunk_offset(i) for i in (0, 4, 8, 12)] == [
        InferenceMode.SLOW_AND_FAST,
        InferenceMode.FAST,
        InferenceMode.FAST,
        InferenceMode.FAST,
    ]


def test_slow_fast_cache_keeps_stable_chunk_identity_and_rejects_replay():
    adapter = TReXRevoPolicyAdapter(MockTReXBackend())
    slow = adapter.infer(
        _request(InferenceMode.SLOW_AND_FAST, 0, 100),
        global_start_step=0,
        now_ns=100,
    )
    fast = adapter.infer(
        _request(InferenceMode.FAST, 4, 110),
        global_start_step=999,
        now_ns=110,
    )
    assert fast.chunk_id == slow.chunk_id
    assert fast.start_step == slow.start_step == 0
    with pytest.raises(CacheProtocolError, match="strictly increasing"):
        adapter.infer(
            _request(InferenceMode.FAST, 4, 120),
            global_start_step=0,
            now_ns=120,
        )


def test_cache_rejects_task_version_change_and_clear_rejects_fast():
    adapter = TReXRevoPolicyAdapter(MockTReXBackend())
    adapter.infer(
        _request(InferenceMode.SLOW_AND_FAST, 0, 100), global_start_step=0, now_ns=100
    )
    with pytest.raises(CacheProtocolError, match="mismatch"):
        adapter.infer(
            _request(InferenceMode.FAST, 4, 110, task_version=1),
            global_start_step=0,
            now_ns=110,
        )
    adapter.reset("REPLAN")
    assert adapter.cache.last_clear_reason == "REPLAN"
    with pytest.raises(CacheProtocolError, match="no slow cache"):
        adapter.infer(
            _request(InferenceMode.FAST, 4, 120), global_start_step=0, now_ns=120
        )


def test_cache_rejects_stale_touch():
    adapter = TReXRevoPolicyAdapter(MockTReXBackend())
    with pytest.raises(CacheProtocolError, match="stale"):
        adapter.infer(
            _request(InferenceMode.SLOW_AND_FAST, 0, 100),
            global_start_step=0,
            now_ns=200_000_000,
        )


def test_policy_runner_issues_main_offsets_and_exposes_aggregated_target():
    runner = TReXPolicyRunner(TReXRevoPolicyAdapter(MockTReXBackend()))
    slow_request = _request(InferenceMode.SLOW_AND_FAST, 0, 100)
    slow = runner.infer_if_due(
        global_step=0, observation=slow_request.observation, now_ns=100
    )
    assert slow is not None and slow.mode is InferenceMode.SLOW_AND_FAST
    assert runner.infer_if_due(
        global_step=1, observation=slow_request.observation, now_ns=101
    ) is None
    fast_request = _request(InferenceMode.FAST, 4, 110)
    fast = runner.infer_if_due(
        global_step=4, observation=fast_request.observation, now_ns=110
    )
    assert fast is not None and fast.mode is InferenceMode.FAST
    target = runner.target_for_step(
        global_step=4, task_key=fast.task_key, now_ns=110
    )
    assert target.shape == (21,)
