import numpy as np
import pytest

from revo3_v1.tactile import (
    DenseTactileBuffer,
    TactileBufferError,
    TactileFrame,
    TactileNotReady,
)


def _frame(i):
    return TactileFrame(
        timestamp_ns=i * 10,
        sequence=i,
        f6=np.full((5, 6), i, dtype=np.float32),
    )


def test_dense_buffer_uses_16_real_samples_not_request_duplicates():
    buffer = DenseTactileBuffer()
    for i in range(16):
        assert buffer.append(_frame(i))
    assert not buffer.append(_frame(15))
    window = buffer.snapshot(now_ns=151, max_age_ns=10)
    assert window.f6.shape == (16, 5, 6)
    assert window.current.shape == (5, 6)
    assert window.span_ns == 150
    np.testing.assert_allclose(window.f6[:, 0, 0], np.arange(16))


def test_insufficient_stale_and_out_of_order_history_are_rejected():
    buffer = DenseTactileBuffer()
    for i in range(15):
        buffer.append(_frame(i))
    with pytest.raises(TactileNotReady):
        buffer.snapshot(now_ns=140, max_age_ns=10)
    buffer.append(_frame(15))
    with pytest.raises(TactileNotReady, match="stale"):
        buffer.snapshot(now_ns=1_000, max_age_ns=10)
    with pytest.raises(TactileBufferError, match="out-of-order"):
        buffer.append(_frame(14))
