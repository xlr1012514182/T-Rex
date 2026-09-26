"""Process-global ownership guard for ``bc_edu_sdk.main_mod`` callbacks.

The pinned SDK examples register sensor callbacks through module-level
``set_*_data_callback`` functions.  Until device-ID routing has been verified
on the real multi-device setup, the EMG armband and EDU glove must not stream
from the same Python process.  This registry makes that policy executable and
observable by higher-level orchestration code.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading


BRAINCO_EDU_CALLBACK_NAMESPACE = "bc_edu_sdk.main_mod:module_global_callbacks"


@dataclass(frozen=True)
class BrainCoEduCallbackNamespaceStatus:
    namespace: str
    busy: bool
    owner_kind: str | None


_OWNER_LOCK = threading.Lock()
_OWNER: object | None = None
_OWNER_KIND: str | None = None


def claim_brainco_edu_callback_namespace(owner: object, *, owner_kind: str) -> None:
    if not str(owner_kind).strip():
        raise ValueError("owner_kind must be non-empty")
    global _OWNER, _OWNER_KIND
    with _OWNER_LOCK:
        if _OWNER is not None and _OWNER is not owner:
            raise RuntimeError(
                "bc-edu-sdk callbacks are module-global and already owned by "
                f"{_OWNER_KIND}; run BrainCo EDU EMG and glove acquisition in "
                "separate processes with timestamped IPC"
            )
        _OWNER = owner
        _OWNER_KIND = str(owner_kind)


def release_brainco_edu_callback_namespace(owner: object) -> None:
    global _OWNER, _OWNER_KIND
    with _OWNER_LOCK:
        if _OWNER is owner:
            _OWNER = None
            _OWNER_KIND = None


def brainco_edu_callback_namespace_status() -> BrainCoEduCallbackNamespaceStatus:
    with _OWNER_LOCK:
        return BrainCoEduCallbackNamespaceStatus(
            namespace=BRAINCO_EDU_CALLBACK_NAMESPACE,
            busy=_OWNER is not None,
            owner_kind=_OWNER_KIND,
        )


__all__ = [
    "BRAINCO_EDU_CALLBACK_NAMESPACE",
    "BrainCoEduCallbackNamespaceStatus",
    "brainco_edu_callback_namespace_status",
    "claim_brainco_edu_callback_namespace",
    "release_brainco_edu_callback_namespace",
]
