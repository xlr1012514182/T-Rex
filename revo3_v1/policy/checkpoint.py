"""Conservative official-checkpoint migration into a Revo 21-D model.

This module never pretends that a 31/62-D official action head is compatible
with Revo.  It loads only exact-shape tensors and reports every skipped item;
the dimension-bound action/state/output layers remain newly initialized and
must be trained on Revo data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


REVO_DIMENSION_BOUND_SUFFIXES = (
    "x_embedder.mlp.fc1.weight",
    "state_embedder.mlp.fc1.weight",
    "final_layer.mlp.fc2.weight",
    "final_layer.mlp.fc2.bias",
    "final_layer_tactile.mlp.fc2.weight",
    "final_layer_tactile.mlp.fc2.bias",
)

# These parameters do not have a semantically compatible value in the
# tactile-free, heterogeneous-hand source checkpoint.  They are the *only*
# missing target tensors accepted by the default Revo migration policy.  The
# list is deliberately expressed as model parameter prefixes/tokens rather
# than as a blanket ``strict=False`` escape hatch.
REVO_NEW_PARAMETER_PREFIXES = (
    "state_embedder.",
    "tacf6_embedder.",
    "tactile_code_embedder.",
    "deform_proj.",
    "deform_encoder.",
    "tactile_vqvae.",
    "flare_proj.",
)
REVO_NEW_PARAMETER_NAMES = (
    "flare_queries",
    "tacf6_vqvae_min",
    "tacf6_vqvae_max",
    "tacf6_vqvae_mask",
)

MIGRATION_KINDS = (
    "official_pretrain",
    "official_midtrain_ablation",
    "revo_w0",
    "revo_w1",
    "revo_midtrain",
    "revo_sft",
)


@dataclass(frozen=True)
class CheckpointMigrationReport:
    loaded: tuple[str, ...]
    reinitialized_dimension_bound: tuple[str, ...]
    declared_missing_reinitialized: tuple[str, ...]
    unexpected_shape_mismatch: tuple[str, ...]
    unexpected_source: tuple[str, ...]
    unexpected_missing_target: tuple[str, ...]

    @property
    def skipped_shape_mismatch(self) -> tuple[str, ...]:
        """Backward-compatible name; every entry is an actual migration error."""

        return self.unexpected_shape_mismatch

    @property
    def missing_target(self) -> tuple[str, ...]:
        """Backward-compatible name for undeclared missing target tensors."""

        return self.unexpected_missing_target

    @property
    def clean(self) -> bool:
        return not (
            self.unexpected_shape_mismatch
            or self.unexpected_source
            or self.unexpected_missing_target
        )

    def assert_clean(self) -> None:
        if not self.clean:
            raise RuntimeError(
                "unsafe Revo checkpoint migration: "
                f"unexpected_shape_mismatch={self.unexpected_shape_mismatch[:8]}, "
                f"unexpected_source={self.unexpected_source[:8]}, "
                f"unexpected_missing_target={self.unexpected_missing_target[:8]}"
            )


def build_revo_trex_model(
    text_config: Any,
    *,
    image_token_id: int = 151655,
    tactile_intermediate_size: int | None = 1536,
    use_tactile_deform: bool = False,
    use_tactile_vqvae: bool = False,
    vqvae_config: dict[str, Any] | None = None,
) -> Any:
    """Build the existing T-Rex architecture with Revo-safe public kwargs.

    Raw single-hand tactile remains ``[B,5,6]`` and is embedded as five 6-D
    tokens.  The local model now parameterizes ``tactile_num_fingers``; a
    30-channel Revo VQ-VAE may therefore be enabled only with a separately
    trained Revo checkpoint/config.  Upstream 60-channel/two-hand VQ weights
    are never padded, truncated, or silently reused.
    """

    from qwen_vla.modeling_vla import Qwen3VLVLAModel

    return Qwen3VLVLAModel(
        config=text_config,
        action_dim=21,
        action_chunk=16,
        tacf6_dim=6,
        tactile_num_fingers=5,
        use_tactile_deform=use_tactile_deform,
        use_robot_state=True,
        image_token_id=image_token_id,
        tactile_intermediate_size=tactile_intermediate_size,
        use_tactile_code=use_tactile_vqvae,
        use_tactile_vqvae=use_tactile_vqvae,
        vqvae_config=vqvae_config,
    )


def _dimension_bound(key: str) -> bool:
    return any(key.endswith(suffix) for suffix in REVO_DIMENSION_BOUND_SUFFIXES)


def _declared_revo_reinitialization(key: str) -> bool:
    """Return whether ``key`` is explicitly rebuilt for the Revo family."""

    if _dimension_bound(key):
        return True
    if key in REVO_NEW_PARAMETER_NAMES:
        return True
    if key.startswith(REVO_NEW_PARAMETER_PREFIXES):
        return True
    # Tactile expert copies live inside every MoT block and therefore cannot
    # be represented by one top-level prefix.
    return "_tactile" in key or key.startswith("final_layer_tactile.")


def _allowed_reinitialization(key: str, migration_kind: str) -> bool:
    if migration_kind in {"official_pretrain", "official_midtrain_ablation"}:
        return _declared_revo_reinitialization(key)
    if migration_kind == "revo_w0":
        return key.startswith(
            (
                "tactile_vqvae.",
                "tactile_code_embedder.",
                "deform_encoder.",
                "deform_proj.",
            )
        ) or key in {
            "tacf6_vqvae_min", "tacf6_vqvae_max", "tacf6_vqvae_mask"
        }
    if migration_kind == "revo_w1":
        return key == "flare_queries" or key.startswith("flare_proj.")
    # A Revo midtrain -> SFT (or same-stage resume) is an exact graph load.
    return False


def _force_reinitialization(key: str, migration_kind: str) -> bool:
    """Skip semantically embodiment-bound tensors even when shapes coincide."""

    return migration_kind in {
        "official_pretrain",
        "official_midtrain_ablation",
    } and _declared_revo_reinitialization(key)


def load_revo_compatible_state_dict(
    model: Any,
    source_state_dict: Mapping[str, Any],
    *,
    migration_kind: str = "official_pretrain",
    raise_on_error: bool = True,
) -> CheckpointMigrationReport:
    """Load an explicitly allowlisted source checkpoint into a Revo model.

    A declared Revo-specific tensor may be absent or shape-incompatible and
    is then rebuilt.  Every other missing, unexpected, or shape-mismatched
    tensor is a hard error by default.  This prevents ``strict=False`` from
    silently turning an incomplete checkpoint into a purportedly valid SFT
    starting point.
    """

    if migration_kind not in MIGRATION_KINDS:
        raise ValueError(f"unknown Revo checkpoint migration kind: {migration_kind}")
    target = model.state_dict()
    compatible = {}
    reinitialized = []
    mismatched = []
    unexpected = []
    for key, value in source_state_dict.items():
        if key not in target:
            unexpected.append(key)
            continue
        if _force_reinitialization(key, migration_kind):
            reinitialized.append(key)
            continue
        if tuple(value.shape) != tuple(target[key].shape):
            if _allowed_reinitialization(key, migration_kind):
                reinitialized.append(key)
            else:
                mismatched.append(key)
            continue
        compatible[key] = value

    load_result = model.load_state_dict(compatible, strict=False)
    declared_missing = tuple(
        sorted(
            key
            for key in load_result.missing_keys
            if _allowed_reinitialization(key, migration_kind)
        )
    )
    unexpected_missing = tuple(
        sorted(
            key
            for key in load_result.missing_keys
            if not _allowed_reinitialization(key, migration_kind)
        )
    )
    report = CheckpointMigrationReport(
        loaded=tuple(sorted(compatible)),
        reinitialized_dimension_bound=tuple(sorted(set(reinitialized))),
        declared_missing_reinitialized=declared_missing,
        unexpected_shape_mismatch=tuple(sorted(mismatched)),
        unexpected_source=tuple(sorted(unexpected)),
        unexpected_missing_target=unexpected_missing,
    )
    if raise_on_error:
        report.assert_clean()
    return report
