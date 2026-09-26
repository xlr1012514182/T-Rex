from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from revo3_teleop.backends import TeleopRevoWriter  # noqa: E402
from revo3_teleop.contracts import NativeSample, SampleHeader  # noqa: E402
from revo3_teleop.hand_retargeting import (  # noqa: E402
    BrainCoSixFlexCalibration,
    BrainCoSixFlexRetargeter,
    HandRetargetingBlocked,
    PluginRevoHandRetargeter,
    RevoGloveTeleopController,
    load_revo_hand_retargeter,
)
from revo3_v1.revo import (  # noqa: E402
    MockRevoBackend,
    RevoCommandPipeline,
    SafetyContext,
    SafetyEnvelope,
    SafetySupervisor,
)


def flex_sample(values=None) -> NativeSample:
    return NativeSample(
        SampleHeader(
            source_id="brainco_glove_flex",
            sequence=4,
            capture_timestamp_ns=100,
            receive_timestamp_ns=100,
        ),
        {
            "flex_raw": np.asarray(
                [0.5] * 6 if values is None else values, dtype=np.float32
            )
        },
    )


class FixtureHandRetargetPlugin:
    calibration_revision = "fixture-cal-v1"
    model_revision = "fixture-model-v1"
    input_kind = "fixture"

    def retarget(self, sample):
        return np.zeros(21, dtype=np.float32)


def make_fixture_hand_retarget_plugin():
    return FixtureHandRetargetPlugin()


def test_brainco_six_flex_cannot_be_silently_tiled_to_21d() -> None:
    with pytest.raises(HandRetargetingBlocked, match="explicit"):
        BrainCoSixFlexRetargeter(None)

    matrix = np.zeros((21, 6), dtype=np.float64)
    for joint in range(21):
        matrix[joint, joint % 6] = 0.2
    retargeter = BrainCoSixFlexRetargeter(
        BrainCoSixFlexCalibration(
            flex_min=np.zeros(6),
            flex_max=np.ones(6),
            normalized_to_q_matrix=matrix,
            q_bias_rad=np.zeros(21),
            q_min_rad=np.full(21, -1.0),
            q_max_rad=np.full(21, 1.0),
            calibration_revision="subject-day-cal-v1",
            model_revision="explicit-linear-map-v1",
        )
    )

    result = retargeter.retarget(flex_sample())

    assert result.requested_q_rad.shape == (21,)
    np.testing.assert_allclose(result.requested_q_rad, 0.1)
    with pytest.raises(HandRetargetingBlocked, match="outside"):
        retargeter.retarget(flex_sample([2.0] * 6))


def test_plugin_request_only_becomes_label_after_revo_pipeline_write() -> None:
    class FakeManusRetarget:
        calibration_revision = "manus-revo-cal-v1"
        model_revision = "fake-retarget-model-v1"
        input_kind = "manus_raw_nodes"

        def retarget(self, sample):
            assert sample.header.source_id == "manus_glove"
            return {"requested_q_rad": np.full(21, 0.7, np.float32)}

    async def scenario():
        backend = MockRevoBackend()
        writer = TeleopRevoWriter(
            RevoCommandPipeline(
                backend,
                SafetySupervisor(SafetyEnvelope.demo(max_step_rad=0.05)),
            )
        )
        controller = RevoGloveTeleopController(
            retargeter=PluginRevoHandRetargeter(FakeManusRetarget()),
            writer=writer,
        )
        sample = NativeSample(
            SampleHeader(
                source_id="manus_glove",
                sequence=8,
                capture_timestamp_ns=200,
                receive_timestamp_ns=200,
            ),
            {"raw_node_positions": np.zeros((1, 3), dtype=np.float32)},
        )
        state = await backend.read_state()
        result = await controller.execute(
            sample,
            request_id="glove-hand-0",
            task_id="bottle",
            task_version=1,
            safety_context=SafetyContext(),
            emg_requests_close=True,
            state=state,
            decision_timestamp_ns=max(time.monotonic_ns(), state.timestamp_ns),
        )
        return backend, result

    backend, result = asyncio.run(scenario())

    np.testing.assert_allclose(result.retarget.requested_q_rad, 0.7)
    assert result.receipt.accepted and result.receipt.clipped
    np.testing.assert_allclose(result.receipt.exact_sent_target, 0.05)
    np.testing.assert_allclose(
        backend.commands[-1].q_target_rad,
        result.receipt.exact_sent_target,
    )


def test_external_hand_retarget_plugin_hash_is_required_for_non_mock_write() -> None:
    unverified = PluginRevoHandRetargeter(FixtureHandRetargetPlugin())
    non_mock_writer = SimpleNamespace(
        pipeline=SimpleNamespace(backend=object())
    )
    with pytest.raises(HandRetargetingBlocked, match="hash must be verified"):
        RevoGloveTeleopController(
            retargeter=unverified,
            writer=non_mock_writer,
        )

    this_file = Path(__file__).resolve()
    digest = hashlib.sha256(this_file.read_bytes()).hexdigest()
    verified = load_revo_hand_retargeter(
        f"{__name__}:make_fixture_hand_retarget_plugin",
        expected_module_sha256=digest,
    )
    assert verified.module_hash_verified
