"""Lazy loader for a locally supplied Tianji/Marvin Python SDK bridge.

No public repository located by this project can be authenticated as the
current Tianji vendor SDK.  Consequently this loader imports only an explicit
local plugin.  The plugin owns native ABI declarations and may either return
normalized feedback mappings or provide a buffer factory and decoder for
``OnGetBuf(pointer)``.  Merely loading a plugin never connects to the robot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import importlib
import inspect
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping, Protocol


class TianjiSdkLoadError(RuntimeError):
    """The explicit local SDK plugin cannot be imported or audited."""


class TianjiFeedbackDecoderPlugin(Protocol):
    """Normalize one local SDK feedback buffer into backend field names."""

    def __call__(self, native_buffer: Any) -> Mapping[str, Any]: ...


class TianjiClientFactoryPlugin(Protocol):
    """Construct a disconnected local SDK client; must not open hardware."""

    def __call__(self, **kwargs: Any) -> Any: ...


def _target(value: object, *, name: str) -> str:
    text = str(value).strip()
    if text.count(":") != 1:
        raise ValueError(f"{name} must use 'python.module:attribute' syntax")
    module, attribute = text.split(":", 1)
    if not module or not attribute or any(not part for part in module.split(".")):
        raise ValueError(f"{name} contains an invalid import target")
    return text


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_sha256(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def resolve_callable(import_target: str) -> tuple[Callable[..., Any], ModuleType, str | None]:
    target = _target(import_target, name="import_target")
    module_name, attribute_name = target.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise TianjiSdkLoadError(
            f"could not import Tianji plugin module {module_name!r}: {type(exc).__name__}"
        ) from exc
    value: Any = module
    try:
        for component in attribute_name.split("."):
            value = getattr(value, component)
    except AttributeError as exc:
        raise TianjiSdkLoadError(f"Tianji plugin target {target!r} does not exist") from exc
    if not callable(value):
        raise TianjiSdkLoadError(f"Tianji plugin target {target!r} is not callable")

    # Bind provenance to the module that actually defines the callable, not
    # merely to a re-exporting module named by the import target.  Otherwise a
    # tiny, hash-pinned shim could re-export an unpinned implementation from a
    # different module while still appearing verified.
    defining_module = inspect.getmodule(value)
    if defining_module is None and not isinstance(value, type):
        defining_module = inspect.getmodule(type(value))
    if not isinstance(defining_module, ModuleType):
        defining_module = module
    try:
        module_path = inspect.getsourcefile(defining_module)
    except TypeError:
        module_path = None
    module_path = module_path or getattr(defining_module, "__file__", None)
    return value, defining_module, module_path


@dataclass(frozen=True)
class CallableModuleProvenance:
    import_target: str
    module_name: str
    module_path: str | None
    module_sha256: str | None
    hash_verified: bool


def resolve_hashed_callable(
    import_target: str,
    *,
    expected_module_sha256: str | None,
    name: str,
) -> tuple[Callable[..., Any], CallableModuleProvenance]:
    """Resolve one executable plugin and bind it to its source/binary hash.

    A missing expected hash is allowed for side-effect-free dry-run assembly,
    but ``hash_verified`` remains false and callers must not grant hardware
    write authority.  A supplied mismatch always raises before construction.
    """

    expected = _optional_sha256(expected_module_sha256, name=f"expected_{name}_sha256")
    value, module, module_path_text = resolve_callable(import_target)
    path: Path | None = None
    digest: str | None = None
    if module_path_text is not None:
        candidate = Path(module_path_text).resolve()
        if candidate.is_file():
            path = candidate
            digest = _sha256(candidate)
    if expected is not None:
        if digest is None:
            raise TianjiSdkLoadError(f"{name} module has no hashable source/binary file")
        if digest != expected:
            raise TianjiSdkLoadError(f"{name} module SHA-256 mismatch")
    return value, CallableModuleProvenance(
        import_target=import_target,
        module_name=module.__name__,
        module_path=None if path is None else str(path),
        module_sha256=digest,
        hash_verified=expected is not None and digest == expected,
    )


@dataclass(frozen=True)
class TianjiSdkPluginSpec:
    client_factory: str
    feedback_mode: str = "normalized_mapping"
    feedback_buffer_factory: str | None = None
    feedback_decoder: str | None = None
    feedback_argument_adapter: str | None = None
    client_kwargs: Mapping[str, Any] = field(default_factory=dict)
    expected_module_sha256: str | None = None
    expected_feedback_buffer_factory_sha256: str | None = None
    expected_feedback_decoder_sha256: str | None = None
    expected_feedback_argument_adapter_sha256: str | None = None
    native_library_path: str | None = None
    expected_native_library_sha256: str | None = None
    sdk_identity: str = "UNVERIFIED_LOCAL_TIANJI_SDK"
    sdk_version: str = "UNVERIFIED"

    def __post_init__(self) -> None:
        object.__setattr__(self, "client_factory", _target(self.client_factory, name="client_factory"))
        mode = str(self.feedback_mode).strip()
        if mode not in {"normalized_mapping", "pointer_decoder"}:
            raise ValueError("feedback_mode must be normalized_mapping or pointer_decoder")
        object.__setattr__(self, "feedback_mode", mode)
        if mode == "normalized_mapping":
            if (
                self.feedback_buffer_factory is not None
                or self.feedback_decoder is not None
                or self.expected_feedback_buffer_factory_sha256 is not None
                or self.expected_feedback_decoder_sha256 is not None
            ):
                raise ValueError(
                    "normalized_mapping mode cannot configure a feedback buffer/decoder"
                )
        else:
            if self.feedback_buffer_factory is None or self.feedback_decoder is None:
                raise ValueError(
                    "pointer_decoder mode requires feedback_buffer_factory and feedback_decoder"
                )
            object.__setattr__(
                self,
                "feedback_buffer_factory",
                _target(self.feedback_buffer_factory, name="feedback_buffer_factory"),
            )
            object.__setattr__(
                self,
                "feedback_decoder",
                _target(self.feedback_decoder, name="feedback_decoder"),
            )
        if self.feedback_argument_adapter is not None:
            object.__setattr__(
                self,
                "feedback_argument_adapter",
                _target(self.feedback_argument_adapter, name="feedback_argument_adapter"),
            )
        elif self.expected_feedback_argument_adapter_sha256 is not None:
            raise ValueError(
                "expected_feedback_argument_adapter_sha256 requires feedback_argument_adapter"
            )
        if not isinstance(self.client_kwargs, Mapping):
            raise TypeError("client_kwargs must be a mapping")
        object.__setattr__(self, "client_kwargs", dict(self.client_kwargs))
        for field_name in (
            "expected_module_sha256",
            "expected_feedback_buffer_factory_sha256",
            "expected_feedback_decoder_sha256",
            "expected_feedback_argument_adapter_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_sha256(getattr(self, field_name), name=field_name),
            )
        if (self.native_library_path is None) != (
            self.expected_native_library_sha256 is None
        ):
            raise ValueError(
                "native_library_path and expected_native_library_sha256 are required together"
            )
        if self.native_library_path is not None:
            native_path = str(self.native_library_path).strip()
            if not native_path:
                raise ValueError("native_library_path must be non-empty")
            object.__setattr__(self, "native_library_path", native_path)
            digest = str(self.expected_native_library_sha256).strip().lower()
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError(
                    "expected_native_library_sha256 must be a lowercase SHA-256 digest"
                )
            object.__setattr__(self, "expected_native_library_sha256", digest)
        identity = str(self.sdk_identity).strip()
        version = str(self.sdk_version).strip()
        if not identity or not version:
            raise ValueError("sdk_identity and sdk_version must be non-empty")
        object.__setattr__(self, "sdk_identity", identity)
        object.__setattr__(self, "sdk_version", version)


@dataclass(frozen=True)
class TianjiSdkProvenance:
    sdk_identity: str
    sdk_version: str
    client_factory: str
    module_name: str
    module_path: str | None
    module_sha256: str | None
    hash_verified: bool
    native_library_path: str | None
    native_library_sha256: str | None
    native_library_hash_verified: bool
    client_native_library_path_verified: bool
    executable_plugins: tuple[CallableModuleProvenance, ...]


@dataclass(frozen=True)
class LoadedTianjiSdk:
    client: Any
    feedback_buffer_factory: Callable[[], Any] | None
    feedback_decoder: Callable[[Any], Mapping[str, Any]] | None
    feedback_argument_adapter: Callable[[Any], Any] | None
    provenance: TianjiSdkProvenance


def load_tianji_sdk(spec: TianjiSdkPluginSpec) -> LoadedTianjiSdk:
    """Load an explicit plugin without connecting or sending any command."""

    factory, factory_provenance = resolve_hashed_callable(
        spec.client_factory,
        expected_module_sha256=spec.expected_module_sha256,
        name="Tianji client factory",
    )
    native_path: Path | None = None
    native_digest: str | None = None
    if spec.native_library_path is not None:
        native_path = Path(spec.native_library_path).expanduser().resolve()
        if not native_path.is_file():
            raise TianjiSdkLoadError("configured Tianji native library does not exist")
        native_digest = _sha256(native_path)
        if native_digest != spec.expected_native_library_sha256:
            raise TianjiSdkLoadError("Tianji native library SHA-256 mismatch")
    try:
        client = factory(**dict(spec.client_kwargs))
    except Exception as exc:
        raise TianjiSdkLoadError(
            f"Tianji client factory failed: {type(exc).__name__}"
        ) from exc
    if client is None:
        raise TianjiSdkLoadError("Tianji client factory returned None")
    client_native_path_verified = False
    if native_path is not None:
        reported_path = getattr(client, "library_path", None)
        if reported_path is None:
            raise TianjiSdkLoadError(
                "Tianji client must expose library_path so the hashed artifact can be bound"
            )
        try:
            client_native_path = Path(str(reported_path)).expanduser().resolve()
        except (TypeError, ValueError, OSError) as exc:
            raise TianjiSdkLoadError("Tianji client reported an invalid library_path") from exc
        if client_native_path != native_path:
            raise TianjiSdkLoadError(
                "Tianji client loaded a different native library than the hashed artifact"
            )
        client_native_path_verified = True

    buffer_factory = None
    decoder = None
    argument_adapter = None
    executable_plugins = [factory_provenance]
    if spec.feedback_mode == "pointer_decoder":
        assert spec.feedback_buffer_factory is not None
        assert spec.feedback_decoder is not None
        buffer_factory, provenance = resolve_hashed_callable(
            spec.feedback_buffer_factory,
            expected_module_sha256=spec.expected_feedback_buffer_factory_sha256,
            name="Tianji feedback buffer factory",
        )
        executable_plugins.append(provenance)
        decoder, provenance = resolve_hashed_callable(
            spec.feedback_decoder,
            expected_module_sha256=spec.expected_feedback_decoder_sha256,
            name="Tianji feedback decoder",
        )
        executable_plugins.append(provenance)
    if spec.feedback_argument_adapter is not None:
        argument_adapter, provenance = resolve_hashed_callable(
            spec.feedback_argument_adapter,
            expected_module_sha256=spec.expected_feedback_argument_adapter_sha256,
            name="Tianji feedback argument adapter",
        )
        executable_plugins.append(provenance)

    return LoadedTianjiSdk(
        client=client,
        feedback_buffer_factory=buffer_factory,
        feedback_decoder=decoder,
        feedback_argument_adapter=argument_adapter,
        provenance=TianjiSdkProvenance(
            sdk_identity=str(spec.sdk_identity).strip(),
            sdk_version=str(spec.sdk_version).strip(),
            client_factory=spec.client_factory,
            module_name=factory_provenance.module_name,
            module_path=factory_provenance.module_path,
            module_sha256=factory_provenance.module_sha256,
            hash_verified=factory_provenance.hash_verified,
            native_library_path=None if native_path is None else str(native_path),
            native_library_sha256=native_digest,
            native_library_hash_verified=(
                spec.expected_native_library_sha256 is not None
                and native_digest == spec.expected_native_library_sha256
            ),
            client_native_library_path_verified=client_native_path_verified,
            executable_plugins=tuple(executable_plugins),
        ),
    )


__all__ = [
    "LoadedTianjiSdk",
    "CallableModuleProvenance",
    "TianjiSdkLoadError",
    "TianjiSdkPluginSpec",
    "TianjiSdkProvenance",
    "TianjiClientFactoryPlugin",
    "TianjiFeedbackDecoderPlugin",
    "load_tianji_sdk",
    "resolve_callable",
    "resolve_hashed_callable",
]
