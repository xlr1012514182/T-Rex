"""Deterministic multi-rate mock collection for interface verification.

The generated trajectories are synthetic fixtures.  They prove serialization,
causal alignment, physical modality separation, and controller-label lineage;
they do not prove task success or hardware readiness.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Iterable, Mapping

import numpy as np

from revo3_v1.revo import (
    MockRevoBackend,
    RevoCommandPipeline,
    RevoState,
    SafetyContext,
    SafetyEnvelope,
    SafetySupervisor,
)

from revo3_teleop.backends import (
    TeleopRevoWriter,
    TianjiMarvinBackend,
    TianjiSafetyLimits,
)
from revo3_teleop.contracts import NativeSample, SampleHeader
from revo3_teleop.recording import (
    EpisodeRecorder,
    Revo3ExportConfig,
    export_revo3_episode,
)
from revo3_teleop.sources import BrainCoEduEMGParser


TASK_INSTRUCTIONS = {
    "bottle": "Grasp the centered bottle with a power grasp and hold it securely.",
    "phone": "Grasp the centered phone with a precision grasp and hold it securely.",
    "plastic_bag": "Grasp the handles of the centered plastic bag and lift it.",
    "refrigerator_door": "Grasp the refrigerator door handle and pull it open.",
}


_SYNTHETIC_TIANJI_ARM_TOKEN = "synthetic-fixture-arm-token"
_TACTILE_HISTORY_LENGTH = 16


class _SyntheticMonotonicClock:
    """Controllable monotonic clock used only by the synthetic SDK fixture."""

    def __init__(self, now_ns: int) -> None:
        self._now_ns = int(now_ns)

    def __call__(self) -> int:
        return self._now_ns

    def advance_to(self, timestamp_ns: int) -> None:
        candidate = int(timestamp_ns)
        if candidate < self._now_ns:
            raise ValueError("synthetic monotonic clock cannot move backwards")
        self._now_ns = candidate


class SyntheticTianjiNativeClient:
    """Normalized fake of the narrow Marvin SDK boundary used by this demo.

    This class is deliberately implemented independently of the unit-test fake
    and is never a hardware simulator or a claim of Tianji task success.  It
    exists so the four-task synthetic fixture traverses the same capability,
    connection, feedback and native write transaction checks as a real injected
    SDK client.
    """

    synthetic_fixture = True

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self._connected = False
        self._frame_serial = 0
        self._state = {"A": 1, "B": 1}
        self._q_deg = {
            "A": np.zeros(7, dtype=np.float64),
            "B": np.zeros(7, dtype=np.float64),
        }
        self._pending_positions: dict[str, np.ndarray] = {}
        self._pending_states: dict[str, int] = {}

    @property
    def frame_serial(self) -> int:
        return self._frame_serial

    @staticmethod
    def _native_scalar(value: object) -> object:
        return value.value if hasattr(value, "value") else value

    def OnLinkTo(self, *octets: object) -> int:
        address = tuple(int(self._native_scalar(item)) for item in octets)
        self.calls.append(("OnLinkTo", address))
        if self._connected or len(address) != 4:
            return 0
        self._connected = True
        return 1

    def OnRelease(self) -> int:
        self.calls.append(("OnRelease", None))
        if not self._connected:
            return 0
        self._connected = False
        return 1

    def OnGetBuf(self) -> Mapping[str, object]:
        self.calls.append(("OnGetBuf", None))
        if not self._connected:
            raise RuntimeError("synthetic Tianji client is not connected")
        self._frame_serial += 1

        def state(side: str) -> dict[str, int]:
            current = self._state[side]
            return {"cur_state": current, "cmd_state": current, "err_code": 0}

        def output(side: str) -> dict[str, object]:
            return {
                "frame_serial": self._frame_serial,
                "fb_joint_pos": self._q_deg[side].tolist(),
                "fb_joint_vel": [0.0] * 7,
                "fb_joint_sToq": [0.0] * 7,
            }

        inputs = {"frame_miss_cnt": 0, "max_frame_miss_cnt": 0}
        return {
            "states": [state("A"), state("B")],
            "outputs": [output("A"), output("B")],
            "inputs": [dict(inputs), dict(inputs)],
        }

    def OnClearSet(self) -> int:
        self.calls.append(("OnClearSet", None))
        self._pending_positions.clear()
        self._pending_states.clear()
        return 1

    def _set_position(self, side: str, values: object) -> int:
        target = np.asarray(list(values), dtype=np.float64)
        if target.shape != (7,) or not np.isfinite(target).all():
            return 0
        self.calls.append((f"OnSetJointCmdPos_{side}", tuple(target.tolist())))
        self._pending_positions[side] = target.copy()
        return 1

    def OnSetJointCmdPos_A(self, values: object) -> int:
        return self._set_position("A", values)

    def OnSetJointCmdPos_B(self, values: object) -> int:
        return self._set_position("B", values)

    def _set_state(self, side: str, value: object) -> int:
        state = int(self._native_scalar(value))
        self.calls.append((f"OnSetTargetState_{side}", state))
        self._pending_states[side] = state
        return 1

    def OnSetTargetState_A(self, value: object) -> int:
        return self._set_state("A", value)

    def OnSetTargetState_B(self, value: object) -> int:
        return self._set_state("B", value)

    def OnSetSend(self) -> int:
        self.calls.append(("OnSetSend", None))
        for side, target in self._pending_positions.items():
            self._q_deg[side] = target.copy()
        for side, state in self._pending_states.items():
            self._state[side] = state
        self._pending_positions.clear()
        self._pending_states.clear()
        return 1

    def OnEMG_A(self) -> int:
        self.calls.append(("OnEMG_A", None))
        return 1

    def OnEMG_B(self) -> int:
        self.calls.append(("OnEMG_B", None))
        return 1


def _build_synthetic_tianji_backend(
    *,
    epoch_ns: int,
    side: str = "A",
) -> tuple[TianjiMarvinBackend, SyntheticTianjiNativeClient, _SyntheticMonotonicClock]:
    """Create a fully gated Tianji backend without bypassing its SDK boundary."""

    clock = _SyntheticMonotonicClock(epoch_ns)
    client = SyntheticTianjiNativeClient()

    # Capability confirmation is derived from a real method-presence probe,
    # rather than being asserted without evidence by the synthetic runner.
    probe = TianjiMarvinBackend(client, side=side, clock=clock)
    report = probe.probe_capabilities()
    if not report.complete:
        raise RuntimeError(
            "synthetic Tianji client is missing required capabilities: "
            + ",".join(report.missing_methods)
        )

    backend = TianjiMarvinBackend(
        client,
        side=side,
        safety_limits=TianjiSafetyLimits(
            q_min_rad=np.full(7, -0.5, dtype=np.float64),
            q_max_rad=np.full(7, 0.5, dtype=np.float64),
            max_delta_rad=0.15,
            max_feedback_age_ns=20_000_000,
            max_target_age_ns=20_000_000,
            require_wrist_pose=True,
        ),
        allow_hardware_write=True,
        capability_probe_confirmed=report.complete,
        arm_token=_SYNTHETIC_TIANJI_ARM_TOKEN,
        clock=clock,
        sleeper=lambda _: None,
    )
    backend.connect("192.168.1.190")
    backend.read_state()  # baseline frame; a later frame must advance before write
    return backend, client, clock


@dataclass(frozen=True)
class MockCollectionConfig:
    output_root: Path
    duration_s: float = 0.5
    tasks: tuple[str, ...] = tuple(TASK_INSTRUCTIONS)
    camera_hz: int = 30
    revo_state_hz: int = 100
    tactile_hz: int = 120
    glove_hz: int = 120
    tianji_state_hz: int = 100
    emg_packet_hz: float = 12.5  # 20 samples/packet at 250 Hz
    anchor_hz: int = 30

    def __post_init__(self) -> None:
        if self.duration_s < 0.1:
            raise ValueError("duration_s must be at least 0.1")
        if self.anchor_hz != 30:
            raise ValueError("the T-Rex/Revo3 projection is fixed at 30 Hz")
        unknown = sorted(set(self.tasks) - set(TASK_INSTRUCTIONS))
        if unknown:
            raise ValueError(f"unsupported mock tasks: {unknown}")
        if not self.tasks:
            raise ValueError("at least one mock task is required")
        rates = (
            self.camera_hz,
            self.revo_state_hz,
            self.tactile_hz,
            self.glove_hz,
            self.tianji_state_hz,
            self.emg_packet_hz,
        )
        if any(float(rate) <= 0 for rate in rates):
            raise ValueError("all mock stream rates must be positive")
        if not np.isclose(self.emg_packet_hz, 12.5):
            raise ValueError(
                "EMG packet_hz must be 12.5 because each packet contains "
                "20 samples/channel at 250 Hz"
            )

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
        *,
        base_directory: Path | None = None,
    ) -> "MockCollectionConfig":
        """Parse the checked-in JSON schema without enabling hardware writes."""

        if value.get("schema_version") != "revo3-teleop-config-v1":
            raise ValueError("unsupported mock config schema_version")
        if value.get("mode") != "mock":
            raise ValueError("mock runner accepts only mode='mock'")
        if bool(value.get("allow_hardware_write", False)):
            raise ValueError("mock config must not allow hardware writes")
        rates = value.get("rates_hz")
        if not isinstance(rates, Mapping):
            raise ValueError("mock config requires rates_hz")
        required_rates = {
            "camera",
            "revo_state",
            "tactile",
            "glove",
            "emg_packets",
            "tianji_state",
            "revo_command",
        }
        missing = sorted(required_rates - set(rates))
        if missing:
            raise ValueError(f"mock rates_hz is missing {missing}")
        anchor_hz = int(value.get("anchor_hz", 0))
        if float(rates["revo_command"]) != float(anchor_hz):
            raise ValueError("mock revo_command rate must equal anchor_hz")
        output_value = str(value.get("output_root", "")).strip()
        if not output_value:
            raise ValueError("mock config requires output_root")
        output_root = Path(output_value)
        if not output_root.is_absolute() and base_directory is not None:
            output_root = base_directory / output_root
        tasks_value = value.get("tasks")
        if not isinstance(tasks_value, list) or not all(
            isinstance(task, str) for task in tasks_value
        ):
            raise ValueError("mock config tasks must be a list of strings")
        return cls(
            output_root=output_root,
            duration_s=float(value.get("duration_s", 0.0)),
            tasks=tuple(tasks_value),
            camera_hz=int(rates["camera"]),
            revo_state_hz=int(rates["revo_state"]),
            tactile_hz=int(rates["tactile"]),
            glove_hz=int(rates["glove"]),
            tianji_state_hz=int(rates["tianji_state"]),
            emg_packet_hz=float(rates["emg_packets"]),
            anchor_hz=anchor_hz,
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "MockCollectionConfig":
        config_path = Path(path).resolve()
        with config_path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise ValueError("mock config root must be a JSON object")
        # Keep relative output paths relative to the process working directory,
        # matching ordinary CLI path semantics and the checked-in example.
        return cls.from_mapping(value)


def _header(source: str, sequence: int, capture_ns: int) -> SampleHeader:
    return SampleHeader(
        source_id=source,
        sequence=sequence,
        capture_timestamp_ns=capture_ns,
        receive_timestamp_ns=capture_ns + 100_000,
        clock_domain="mock_shared_monotonic",
    )


def _timestamps(epoch_ns: int, duration_s: float, rate_hz: float) -> Iterable[tuple[int, int]]:
    count = max(2, int(np.floor(duration_s * rate_hz)) + 1)
    for sequence in range(count):
        yield sequence, epoch_ns + int(sequence * 1_000_000_000 / rate_hz)


def _append_native_streams(
    recorder: EpisodeRecorder,
    *,
    config: MockCollectionConfig,
) -> None:
    epoch = recorder.epoch_ns
    for sequence, timestamp_ns in _timestamps(epoch, config.duration_s, config.camera_hz):
        height, width = 48, 64
        rgb = np.empty((height, width, 3), dtype=np.uint8)
        rgb[..., 0] = (sequence * 11) % 255
        rgb[..., 1] = np.arange(width, dtype=np.uint8)[None, :]
        rgb[..., 2] = np.arange(height, dtype=np.uint8)[:, None]
        recorder.append(
            "camera",
            NativeSample(_header("mock_wrist_camera", sequence, timestamp_ns), {"rgb": rgb}),
        )
    for sequence, timestamp_ns in _timestamps(
        epoch, config.duration_s, config.revo_state_hz
    ):
        phase = sequence / 100.0
        recorder.append(
            "revo_state",
            NativeSample(
                _header("mock_revo3", sequence, timestamp_ns),
                {
                    "q_rad": np.full(21, 0.2 * phase, np.float32),
                    "dq_rad_s": np.full(21, 0.2, np.float32),
                    "current_a": np.full(21, 0.05, np.float32),
                    "status": np.zeros(21, np.int64),
                },
            ),
        )
    # A policy frame consumes 16 *distinct native* Force6D samples.  Seed a
    # genuine pre-roll before the first 30 Hz anchor so even a short plumbing
    # fixture can satisfy that contract without padding or 30 Hz ZOH reuse.
    tactile_period_ns = int(1_000_000_000 / config.tactile_hz)
    tactile_preroll = _TACTILE_HISTORY_LENGTH - 1
    tactile_timestamps = list(
        _timestamps(epoch, config.duration_s, config.tactile_hz)
    )
    tactile_timestamps = [
        (
            sequence,
            epoch - (tactile_preroll - sequence) * tactile_period_ns,
        )
        for sequence in range(tactile_preroll)
    ] + [
        (sequence + tactile_preroll, timestamp_ns)
        for sequence, timestamp_ns in tactile_timestamps
    ]
    for sequence, timestamp_ns in tactile_timestamps:
        features = np.zeros((5, 6), np.float32)
        features[:, 0] = min(sequence / 60.0, 1.0)
        tactile_diff = np.full(
            (5, 240, 240),
            sequence % 255,
            dtype=np.uint8,
        )
        finger_timestamp_ns = np.full(5, timestamp_ns, dtype=np.int64)
        recorder.append(
            "tactile",
            NativeSample(
                _header("mock_u21vt", sequence, timestamp_ns),
                {
                    "features": features,
                    "force6d_finger_timestamp_ns": finger_timestamp_ns,
                    "tactile_diff": tactile_diff,
                    "tactile_diff_timestamp_ns": finger_timestamp_ns.copy(),
                },
            ),
        )
    for sequence, timestamp_ns in _timestamps(epoch, config.duration_s, config.glove_hz):
        recorder.append(
            "glove",
            NativeSample(
                _header("mock_manus", sequence, timestamp_ns),
                {
                    "joint_proxy_rad": np.linspace(0, 0.5, 24, dtype=np.float32)
                    * min(sequence / max(config.glove_hz / 2.0, 1.0), 1.0),
                    "provides_wrist_pose": np.asarray([1], dtype=np.uint8),
                },
            ),
        )
    for sequence, timestamp_ns in _timestamps(
        epoch, config.duration_s, config.tianji_state_hz
    ):
        recorder.append(
            "tianji_state",
            NativeSample(
                _header("mock_tianji", sequence, timestamp_ns),
                {
                    "q_rad": np.linspace(-0.2, 0.2, 7, dtype=np.float32),
                    "dq_rad_s": np.zeros(7, np.float32),
                },
            ),
        )
    emg_parser = BrainCoEduEMGParser(source_id="mock_brainco_emg")
    for sequence, timestamp_ns in _timestamps(
        epoch, config.duration_s, config.emg_packet_hz
    ):
        sample_indices = np.arange(20, dtype=np.float32)
        signal = np.stack(
            [
                np.sin(0.25 * sample_indices + channel * 0.3 + sequence)
                for channel in range(8)
            ]
        ).astype(np.float32)
        row = np.concatenate(
            (
                np.asarray([sequence, 0], dtype=np.float32),
                signal.reshape(-1),
            )
        )
        recorder.append(
            "emg",
            emg_parser.parse_row(row, callback_timestamp_ns=timestamp_ns),
        )


async def _record_commands_and_anchors(
    recorder: EpisodeRecorder,
    *,
    task: str,
    duration_s: float,
) -> None:
    backend = MockRevoBackend()
    revo_clock = _SyntheticMonotonicClock(recorder.epoch_ns)
    writer = TeleopRevoWriter(
        RevoCommandPipeline(
            backend,
            SafetySupervisor(SafetyEnvelope.demo(max_step_rad=0.08)),
        ),
        clock_ns=revo_clock,
    )
    anchor_count = max(2, int(np.floor(duration_s * 30)) + 1)
    previous_q = np.zeros(21, np.float32)
    tianji, _, tianji_clock = _build_synthetic_tianji_backend(
        epoch_ns=recorder.epoch_ns,
        side="A",
    )
    try:
        for index in range(anchor_count):
            anchor_ns = recorder.anchor_timestamp_ns(index)
            state = RevoState(
                timestamp_ns=anchor_ns,
                q_rad=previous_q,
                sequence=index,
            )
            nominal = np.full(21, min(0.02 * (index + 1), 0.7), np.float32)
            decision_ns = anchor_ns + 500_000
            # The synthetic receipt represents a completed controller write,
            # not host wall-clock latency spent constructing fixture streams.
            # Keep its explicit causal latency inside one 30 Hz control period.
            revo_clock.advance_to(decision_ns + 20_000)
            receipt = await writer.submit_target(
                request_id=f"{task}-hand-{index:06d}",
                nominal_q_rad=nominal,
                task_id=task,
                task_version=1,
                safety_context=SafetyContext(),
                emg_requests_close=True,
                decision_timestamp_ns=decision_ns,
                state=state,
            )
            if not receipt.accepted or receipt.exact_sent_target is None:
                raise RuntimeError(
                    f"mock Revo command unexpectedly rejected: {receipt.reason}"
                )
            recorder.record_command(receipt)

            # Traverse the actual fail-closed Tianji backend.  These accepted
            # receipts prove a complete synthetic native SDK transaction; they
            # remain master-only and never become the 21-D hand action label.
            phase = (index + 1) / anchor_count
            arm_target = np.linspace(-0.1, 0.1, 7, dtype=np.float32) * phase
            target_ns = anchor_ns + 300_000
            decision_ns = anchor_ns + 400_000
            tianji_clock.advance_to(anchor_ns + 450_000)
            arm_receipt = tianji.submit_target(
                request_id=f"{task}-arm-{index:06d}",
                q_target_rad=arm_target,
                target_timestamp_ns=target_ns,
                arm_token=_SYNTHETIC_TIANJI_ARM_TOKEN,
                wrist_pose_valid=True,
                decision_timestamp_ns=decision_ns,
            )
            if not arm_receipt.accepted or arm_receipt.exact_sent_target is None:
                raise RuntimeError(
                    "synthetic Tianji command unexpectedly rejected: "
                    f"{arm_receipt.reason}"
                )
            recorder.record_command(arm_receipt)
            recorder.record_anchor(
                anchor_index=index,
                streams=("camera", "revo_state", "tactile"),
                hand_command_request_id=receipt.request_id,
                max_age_ns={
                    "camera": 40_000_000,
                    "revo_state": 20_000_000,
                    "tactile": 20_000_000,
                },
            )
            previous_q = receipt.exact_sent_target.copy()
    finally:
        # The fake applies a state-0 transaction and advancing feedback frame,
        # allowing the production backend to verify servo-off before release.
        tianji.close(timeout_ns=20_000_000, poll_interval_s=0.001)


def run_mock_collection(config: MockCollectionConfig) -> list[dict[str, str]]:
    """Generate master and hand-only derived episodes for all four tasks."""

    root = Path(config.output_root)
    master_root = root / "master"
    derived_root = root / "derived" / "revo3_vla"
    results: list[dict[str, str]] = []
    base_epoch = time.monotonic_ns()
    for task_index, task in enumerate(config.tasks):
        episode_id = f"mock_{task}_{task_index:02d}"
        epoch_ns = base_epoch + task_index * 10_000_000_000
        recorder = EpisodeRecorder(
            master_root,
            episode_id=episode_id,
            epoch_ns=epoch_ns,
            metadata={
                "task": task,
                "instruction": TASK_INSTRUCTIONS[task],
                "synthetic_fixture": True,
                "contains_native_emg": True,
                "arm_target_source": "mock_wrist_pose",
            },
        )
        recorder.start()
        try:
            _append_native_streams(recorder, config=config)
            asyncio.run(
                _record_commands_and_anchors(
                    recorder,
                    task=task,
                    duration_s=config.duration_s,
                )
            )
            committed = recorder.commit()
        except Exception:
            if recorder.state.value == "recording":
                recorder.abort("mock_generation_failed")
            raise
        derived = export_revo3_episode(
            committed,
            derived_root,
            Revo3ExportConfig(synthetic_fixture=True),
        )
        results.append({"task": task, "master": str(committed), "revo3_vla": str(derived)})
    return results
