from fractions import Fraction

from revo3_v1.timing import (
    AlignmentMode,
    CausalTimestampAligner,
    StreamConfig,
    TimestampedSample,
)


def configs():
    return [
        StreamConfig("emg", 2000, max_age_ns=1_000_000_000),
        StreamConfig("hand", 100, max_age_ns=1_000_000_000),
        StreamConfig("camera", 30, max_age_ns=1_000_000_000),
        StreamConfig("policy", 30, max_age_ns=1_000_000_000),
    ]


def test_lowest_and_gcd_modes_are_distinct():
    lowest = CausalTimestampAligner(configs(), mode=AlignmentMode.LOWEST)
    common = CausalTimestampAligner(configs(), mode=AlignmentMode.GCD)
    assert lowest.grid_hz == Fraction(30, 1)
    assert common.grid_hz == Fraction(10, 1)


def test_alignment_never_selects_future_sample():
    aligner = CausalTimestampAligner(
        [StreamConfig("camera", 30, max_age_ns=100)],
        mode=AlignmentMode.LOWEST,
    )
    aligner.offer("camera", TimestampedSample(90, "past"))
    aligner.offer("camera", TimestampedSample(101, "future"))
    frame = aligner.align_at(100)
    assert frame.ready
    assert frame.value("camera") == "past"
    assert frame.samples["camera"].sample.timestamp_ns <= frame.anchor_ns


def test_no_past_sample_is_missing_even_if_future_sample_exists():
    aligner = CausalTimestampAligner([StreamConfig("emg", 1000)])
    aligner.offer("emg", TimestampedSample(11, "future"))
    frame = aligner.align_at(10)
    assert not frame.ready
    assert frame.missing == ("emg",)


def test_stale_sample_is_reported_separately_from_missing():
    aligner = CausalTimestampAligner(
        [StreamConfig("state", 100, max_age_ns=5)]
    )
    aligner.offer("state", TimestampedSample(10, "q"))
    frame = aligner.align_at(20)
    assert frame.missing == ()
    assert frame.stale == ("state",)


def test_out_of_order_delivery_and_causal_history_are_timestamp_sorted():
    aligner = CausalTimestampAligner([StreamConfig("touch", 30)])
    aligner.offer("touch", TimestampedSample(30, 3))
    aligner.offer("touch", TimestampedSample(10, 1))
    aligner.offer("touch", TimestampedSample(20, 2))
    assert [sample.value for sample in aligner.causal_window("touch", end_ns=25, count=16)] == [1, 2]
    assert aligner.align_at(25).value("touch") == 2


def test_absolute_anchor_calculation_does_not_accumulate_rounding_error():
    aligner = CausalTimestampAligner(
        [StreamConfig("camera", 30)], epoch_ns=1_000_000_000
    )
    assert aligner.anchor_at_index(0) == 1_000_000_000
    assert aligner.anchor_at_index(30) == 2_000_000_000


def test_default_grid_epoch_is_lazily_set_by_first_sample():
    aligner = CausalTimestampAligner([StreamConfig("camera", 10)])
    assert aligner.drain_until(5_000_000_000) == ()
    aligner.offer("camera", TimestampedSample(5_000_000_000, "frame"))
    frames = aligner.drain_until(5_000_000_000)
    assert len(frames) == 1
    assert frames[0].anchor_ns == 5_000_000_000
