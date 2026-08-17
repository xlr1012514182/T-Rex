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


@dataclass(frozen=True)
class CheckpointMigrationReport:
    loaded: tuple[str, ...]
    reinitialized_dimension_bound: tuple[str, ...]
    skipped_shape_mismatch: tuple[str, ...]
    unexpected_source: tuple[str, ...]
    missing_target: tuple[str, ...]

    @property
    def clean(self) -> bool:
        return not self.skipped_shape_mismatch and not self.unexpected_source


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


def load_revo_compatible_state_dict(
    model: Any,
    source_state_dict: Mapping[str, Any],
) -> CheckpointMigrationReport:
    """Load exact-shape tensors and return an auditable migration report."""

    target = model.state_dict()
    compatible = {}
    reinitialized = []
    mismatched = []
    unexpected = []
    for key, value in source_state_dict.items():
        if key not in target:
            unexpected.append(key)
            continue
        if _dimension_bound(key):
            reinitialized.append(key)
            continue
        if tuple(value.shape) != tuple(target[key].shape):
            mismatched.append(key)
            continue
        compatible[key] = value

    load_result = model.load_state_dict(compatible, strict=False)
    missing = tuple(
        sorted(key for key in load_result.missing_keys if not _dimension_bound(key))
    )
    return CheckpointMigrationReport(
        loaded=tuple(sorted(compatible)),
        reinitialized_dimension_bound=tuple(sorted(set(reinitialized))),
        skipped_shape_mismatch=tuple(sorted(mismatched)),
        unexpected_source=tuple(sorted(unexpected)),
        missing_target=missing,
    )
