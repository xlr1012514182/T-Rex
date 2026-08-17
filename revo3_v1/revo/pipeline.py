"""Safety-owned composition of VLA nominal action and tactile residual."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .backend import RevoBackend
from .contracts import RevoCommand, RevoState, assert_joint_vector
from .safety import SafetyContext, SafetyResult, SafetySupervisor


class RevoCommandPipeline:
    """The only path that should call ``backend.write_command``."""

    def __init__(self, backend: RevoBackend, safety: SafetySupervisor) -> None:
        self.backend = backend
        self.safety = safety

    async def execute(
        self,
        *,
        nominal_q_rad: np.ndarray,
        residual_q_rad: Optional[np.ndarray],
        task_id: str,
        task_version: int,
        source_chunk_id: Optional[str],
        safety_context: SafetyContext,
        emg_requests_close: bool,
        now_ns: Optional[int] = None,
        state: Optional[RevoState] = None,
    ) -> SafetyResult:
        now = time.monotonic_ns() if now_ns is None else int(now_ns)
        observed = await self.backend.read_state() if state is None else state
        nominal = assert_joint_vector(nominal_q_rad, name="nominal_q_rad")
        residual = (
            np.zeros_like(nominal)
            if residual_q_rad is None
            else assert_joint_vector(residual_q_rad, name="residual_q_rad")
        )
        result = self.safety.authorize(
            nominal + residual,
            observed,
            now_ns=now,
            context=safety_context,
            emg_requests_close=emg_requests_close,
        )
        if result.vetoed:
            return result
        command = RevoCommand(
            timestamp_ns=now,
            q_target_rad=result.q_authorized_rad,
            task_id=task_id,
            task_version=task_version,
            source_chunk_id=source_chunk_id,
        )
        await self.backend.write_command(command)
        return result
