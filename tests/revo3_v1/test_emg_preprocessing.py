from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from revo3_v1.emg.model import GNIModelConfig
from revo3_v1.emg.preprocessing import (
    BRAINCO_EDU_8CH_250HZ,
    CausalEMGPreprocessor,
    validate_npz_profile,
)
from revo3_v1.emg.streaming import IntentGateConfig, MulticlassIntentGate, StreamingEMGClassifier


class FixedPowerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(
            input_channels=8,
            output_channels=5,
            resolved_labels=lambda: (
                "POWER_GRASP",
                "PRECISION_GRASP",
                "LATERAL_GRASP",
                "RELEASE",
                "REST",
            ),
        )

    def forward(self, values):
        logits = torch.tensor([8.0, 0.0, 0.0, 0.0, 0.0], device=values.device)
        return logits[None].repeat(values.shape[0], 1) + self.anchor * 0


def test_brainco_profile_and_explicit_gni_temporal_rescaling():
    profile = BRAINCO_EDU_8CH_250HZ
    assert profile.channel_count == 8
    assert profile.window_samples == 500
    assert profile.stride_samples_pattern == (12, 13)
    config = GNIModelConfig.brainco_edu()
    assert (config.input_channels, config.kernel_width, config.stride) == (8, 3, 1)


def test_causal_filter_is_identical_across_packet_boundaries_and_state_is_bound():
    profile = BRAINCO_EDU_8CH_250HZ
    rng = np.random.RandomState(7)
    signal = rng.randn(8, 500).astype(np.float32)
    whole = CausalEMGPreprocessor(profile).process_window(signal)
    packetized = CausalEMGPreprocessor(profile)
    chunks = [
        packetized.process_chunk(
            signal[:, :200], sample_rate_hz=250, channel_order=profile.channel_order
        ),
        packetized.process_chunk(
            signal[:, 200:], sample_rate_hz=250, channel_order=profile.channel_order
        ),
    ]
    np.testing.assert_allclose(whole, np.concatenate(chunks, axis=1), atol=1e-6)
    state = packetized.state_dict()
    with pytest.raises(ValueError, match="profile"):
        CausalEMGPreprocessor(
            profile.with_normalization("0" * 64)
        ).load_state_dict(state)


def test_npz_profile_rejects_mismatch_and_accepts_continuous_filter_lineage(tmp_path):
    profile = BRAINCO_EDU_8CH_250HZ
    path = tmp_path / "windows.npz"
    np.savez(
        path,
        signal=np.zeros((1, 8, 500), dtype=np.float32),
        label=np.zeros(1, dtype=np.int64),
        sample_rate_hz=np.asarray(250),
        channel_order=np.asarray(profile.channel_order),
        preprocessing_profile_id=np.asarray(profile.profile_id),
        preprocessed=np.asarray(True),
        filter_state_provenance=np.asarray("session_continuous_causal_sos_before_windowing"),
        preprocessing_profile_fingerprint=np.asarray(profile.acquisition_fingerprint),
    )
    with np.load(path, allow_pickle=False) as archive:
        assert validate_npz_profile(archive, profile).preprocessed
    with np.load(path, allow_pickle=False) as archive:
        with pytest.raises(ValueError, match="sample rate"):
            validate_npz_profile(
                archive,
                type(profile)(
                    **{**profile.to_mapping(), "sample_rate_hz": 200, "highpass_hz": 30.0}
                ),
            )


def test_timestamped_20_sample_packets_preserve_roughly_20hz_and_sample_time_dwell():
    profile = BRAINCO_EDU_8CH_250HZ
    gate = MulticlassIntentGate(
        IntentGateConfig(close_dwell_ms=300, max_update_gap_ms=100)
    )
    classifier = StreamingEMGClassifier(
        FixedPowerModel(),
        {"mean": np.zeros(8, np.float32), "std": np.ones(8, np.float32)},
        sample_rate_hz=250,
        window_samples=500,
        stride_samples=12,
        gate=gate,
        preprocessing_profile=profile,
        channel_order=profile.channel_order,
    )
    outputs = []
    period_ns = 4_000_000
    for packet in range(40):
        first = packet * 20
        timestamps = (np.arange(first, first + 20, dtype=np.int64) * period_ns)
        outputs.extend(
            classifier.push_many(
                np.zeros((8, 20), dtype=np.float32), timestamps, signal_quality=1.0
            )
        )
    output_times = np.asarray([row["timestamp_ns"] for row in outputs], dtype=np.int64)
    assert len(outputs) > 20
    assert np.all(np.diff(output_times) > 0)
    assert np.mean(np.diff(output_times)) / 1e6 == pytest.approx(50.0, abs=0.2)
    events = [row["event"] for row in outputs if row["event"] is not None]
    assert len(events) == 1
    assert events[0]["type"] == "StartIntentEvent"
    # First inference occurs at sample 499 (1.996 s); 300 ms dwell therefore
    # cannot fire before 2.296 s regardless of packet callback boundaries.
    assert events[0]["timestamp_ns"] >= 2_296_000_000
