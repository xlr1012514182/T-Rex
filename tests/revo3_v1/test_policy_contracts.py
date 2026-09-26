import numpy as np
import pytest

from revo3_v1.policy import (
    ActionChunk,
    InferenceMode,
    PolicyObservation,
    TaskKey,
)


def _observation(timestamp=100):
    instruction = "Grasp the bottle and hold it securely."
    key = TaskKey.from_instruction(
        task_id="t1", task_version=2, instruction=instruction, lease_id="lease"
    )
    return PolicyObservation(
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


def test_observation_and_absolute_chunk_shapes():
    obs = _observation()
    chunk = ActionChunk(
        np.zeros((16, 21)), 0, 100, 100, obs.task_key,
        InferenceMode.SLOW_AND_FAST, "chunk",
    )
    assert obs.q_rad.shape == (21,)
    assert obs.tactile_f6.shape == (5, 6)
    assert obs.tactile_history_f6.shape == (16, 5, 6)
    assert chunk.q_target_rad.shape == (16, 21)


@pytest.mark.parametrize(
    "current,history",
    [
        (np.zeros((10, 6)), np.zeros((16, 5, 6))),
        (np.zeros((5, 6)), np.zeros((16, 10, 6))),
    ],
)
def test_never_silently_accepts_or_pads_a_second_hand(current, history):
    obs = _observation()
    with pytest.raises(ValueError):
        PolicyObservation(
            timestamp_ns=100,
            state_timestamp_ns=100,
            rgb_timestamp_ns=100,
            tactile_timestamp_ns=100,
            q_rad=np.zeros(21),
            tactile_f6=current,
            tactile_history_f6=history,
            instruction=obs.instruction,
            task_key=obs.task_key,
        )


def test_instruction_hash_mismatch_is_rejected():
    obs = _observation()
    with pytest.raises(ValueError, match="instruction_hash"):
        PolicyObservation(
            timestamp_ns=100,
            state_timestamp_ns=100,
            rgb_timestamp_ns=100,
            tactile_timestamp_ns=100,
            q_rad=np.zeros(21),
            tactile_f6=np.zeros((5, 6)),
            tactile_history_f6=np.zeros((16, 5, 6)),
            instruction="Release the bottle.",
            task_key=obs.task_key,
        )
