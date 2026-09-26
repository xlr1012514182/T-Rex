"""Opt-in ctypes client for the *historical* Marvin C function boundary.

The function signatures are cross-checked against the Apache-licensed header
at commit 747f5d0279a91d85e32d06008665d96886eff438, whose author/vendor identity
is unverified.  Therefore construction requires an explicit acknowledgement.
The DCSS feedback structure is intentionally not reproduced here: the actual
installed SDK must supply a matching buffer factory and decoder plugin.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any, Callable


class HistoricalMarvinAbiNotAcknowledged(RuntimeError):
    pass


class CtypesMarvinClient:
    """Disconnected dynamic-library proxy with explicit historical prototypes."""

    def __init__(
        self,
        library_path: str,
        *,
        calling_convention: str = "cdecl",
        acknowledge_historical_abi: bool = False,
        library_loader: Callable[[str], Any] | None = None,
    ) -> None:
        if not acknowledge_historical_abi:
            raise HistoricalMarvinAbiNotAcknowledged(
                "the public Marvin header is historical/unverified; confirm the installed "
                "vendor ABI before loading it"
            )
        path = Path(library_path).expanduser().resolve()
        if library_loader is None and not path.is_file():
            raise FileNotFoundError(f"Tianji native library does not exist: {path}")
        convention = str(calling_convention).strip().lower()
        if convention not in {"cdecl", "stdcall"}:
            raise ValueError("calling_convention must be cdecl or stdcall")
        if library_loader is None:
            if convention == "stdcall":
                loader = getattr(ctypes, "WinDLL", None)
                if loader is None:
                    raise OSError("stdcall/WinDLL is unavailable on this platform")
            else:
                loader = ctypes.CDLL
        else:
            loader = library_loader
        self.library_path = str(path)
        self.calling_convention = convention
        self._library = loader(str(path))
        self._configure_required_functions()

    def _function(self, name: str, argtypes: list[Any], restype: Any) -> Any:
        try:
            function = getattr(self._library, name)
        except AttributeError as exc:
            raise AttributeError(f"Tianji library is missing required symbol {name}") from exc
        function.argtypes = argtypes
        function.restype = restype
        return function

    def _configure_required_functions(self) -> None:
        self.OnLinkTo = self._function(
            "OnLinkTo", [ctypes.c_ubyte] * 4, ctypes.c_bool
        )
        self.OnRelease = self._function("OnRelease", [], ctypes.c_bool)
        self.OnGetBuf = self._function("OnGetBuf", [ctypes.c_void_p], ctypes.c_bool)
        self.OnClearSet = self._function("OnClearSet", [], ctypes.c_bool)
        self.OnSetSend = self._function("OnSetSend", [], ctypes.c_bool)
        double7 = ctypes.POINTER(ctypes.c_double)
        self.OnSetJointCmdPos_A = self._function(
            "OnSetJointCmdPos_A", [double7], ctypes.c_bool
        )
        self.OnSetJointCmdPos_B = self._function(
            "OnSetJointCmdPos_B", [double7], ctypes.c_bool
        )
        self.OnSetTargetState_A = self._function(
            "OnSetTargetState_A", [ctypes.c_int], ctypes.c_bool
        )
        self.OnSetTargetState_B = self._function(
            "OnSetTargetState_B", [ctypes.c_int], ctypes.c_bool
        )
        # The historical header declares emergency-stop calls as void.
        self.OnEMG_A = self._function("OnEMG_A", [], None)
        self.OnEMG_B = self._function("OnEMG_B", [], None)


def create_ctypes_marvin_client(
    *,
    library_path: str,
    calling_convention: str = "cdecl",
    acknowledge_historical_abi: bool = False,
) -> CtypesMarvinClient:
    """Factory target usable from ``TianjiSdkPluginSpec.client_factory``."""

    return CtypesMarvinClient(
        library_path,
        calling_convention=calling_convention,
        acknowledge_historical_abi=acknowledge_historical_abi,
    )


__all__ = [
    "CtypesMarvinClient",
    "HistoricalMarvinAbiNotAcknowledged",
    "create_ctypes_marvin_client",
]
