"""Frozen Revo3 V1 training stages and parameter-selection policy.

The upstream trainer historically used one global learning rate and left most
of the Qwen action expert trainable.  That is not the reviewed Revo route.  In
this module parameter names are classified into explicit groups, all other
parameters are frozen, and a stage fails closed if one of its required groups
cannot be found in the instantiated model.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Any, Iterable, Mapping


class RevoTrainingStage(str, Enum):
    W0 = "w0"
    W1 = "w1"
    MIDTRAIN = "midtrain"
    SFT = "sft"


@dataclass(frozen=True)
class RevoStageSpec:
    stage: RevoTrainingStage
    max_steps: int
    max_epochs: int
    effective_batch: int
    tactile_dropout: float
    state_dropout: float
    flare_enabled: bool
    flare_loss_weight: float
    learning_rates: Mapping[str, float]
    train_groups: tuple[str, ...]


STAGE_SPECS: Mapping[RevoTrainingStage, RevoStageSpec] = {
    RevoTrainingStage.W0: RevoStageSpec(
        stage=RevoTrainingStage.W0,
        max_steps=1_000,
        max_epochs=3,
        effective_batch=64,
        tactile_dropout=0.0,
        state_dropout=0.05,
        flare_enabled=False,
        flare_loss_weight=0.0,
        learning_rates={"new_action_state": 1e-4, "action_last4": 1e-5},
        train_groups=("new_action_state", "action_last4"),
    ),
    RevoTrainingStage.W1: RevoStageSpec(
        stage=RevoTrainingStage.W1,
        max_steps=1_500,
        max_epochs=3,
        effective_batch=64,
        tactile_dropout=0.10,
        state_dropout=0.05,
        flare_enabled=False,
        flare_loss_weight=0.0,
        learning_rates={
            "new_tactile": 1e-4,
            "tactile_expert": 3e-5,
            "action_last4": 1e-5,
        },
        train_groups=("new_tactile", "tactile_expert", "action_last4"),
    ),
    RevoTrainingStage.MIDTRAIN: RevoStageSpec(
        stage=RevoTrainingStage.MIDTRAIN,
        max_steps=22_000,
        max_epochs=3,
        effective_batch=64,
        tactile_dropout=0.10,
        state_dropout=0.05,
        flare_enabled=True,
        flare_loss_weight=0.5,
        learning_rates={
            "new_action_state": 1e-4,
            "new_tactile": 1e-4,
            "tactile_expert": 3e-5,
            "action_last4": 1e-5,
            "inherited_boundary": 1e-5,
            "flare": 5e-6,
        },
        train_groups=(
            "new_action_state",
            "new_tactile",
            "tactile_expert",
            "action_last4",
            "inherited_boundary",
            "flare",
        ),
    ),
    RevoTrainingStage.SFT: RevoStageSpec(
        stage=RevoTrainingStage.SFT,
        max_steps=6_500,
        max_epochs=3,
        effective_batch=64,
        tactile_dropout=0.10,
        state_dropout=0.05,
        flare_enabled=True,
        flare_loss_weight=0.5,
        learning_rates={
            "new_action_state": 5e-5,
            "new_tactile": 5e-5,
            "tactile_expert": 5e-6,
            "action_last4": 5e-6,
            "inherited_boundary": 5e-6,
            "flare": 1e-6,
        },
        train_groups=(
            "new_action_state",
            "new_tactile",
            "tactile_expert",
            "action_last4",
            "inherited_boundary",
            "flare",
        ),
    ),
}


_LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_NEW_ACTION_STATE_PREFIXES = (
    "x_embedder.",
    "state_embedder.",
    "final_layer.",
)
_NEW_TACTILE_PREFIXES = (
    "tacf6_embedder.",
    "tactile_code_embedder.",
    "deform_proj.",
    "final_layer_tactile.",
)
_FROZEN_PREFIXES = ("visual.", "tactile_vqvae.", "deform_encoder.")


def _layer_index(name: str) -> int | None:
    match = _LAYER_PATTERN.search(name)
    return None if match is None else int(match.group(1))


def classify_revo_parameter(name: str, *, num_hidden_layers: int) -> str | None:
    """Classify one parameter into a reviewed optimizer group.

    ``None`` means frozen.  Classification order matters: tactile copies are
    checked before generic action/boundary rules.
    """

    if name.startswith(_FROZEN_PREFIXES):
        return None
    if name.startswith(_NEW_TACTILE_PREFIXES):
        return "new_tactile"
    if "_tactile" in name:
        return "tactile_expert"
    if name.startswith(_NEW_ACTION_STATE_PREFIXES):
        return "new_action_state"
    if name.startswith(("t_embedder.", "model.norm_action.", "model.norm_tactile.")):
        return "inherited_boundary"
    if "_action" in name:
        index = _layer_index(name)
        if index is not None and index >= max(0, num_hidden_layers - 4):
            return "action_last4"
        return None
    if name == "flare_queries" or name.startswith("flare_proj."):
        return "flare"
    return None


def infer_num_hidden_layers(model: Any) -> int:
    config = getattr(model, "config", None)
    configured = getattr(config, "num_hidden_layers", None)
    if configured is not None and int(configured) > 0:
        return int(configured)
    indexes = [
        index
        for name, _ in model.named_parameters()
        if (index := _layer_index(name)) is not None
    ]
    if not indexes:
        raise ValueError("cannot infer transformer layer count for Revo freeze policy")
    return max(indexes) + 1


def configure_revo_trainable_parameters(
    model: Any,
    stage: RevoTrainingStage | str,
) -> dict[str, list[tuple[str, Any]]]:
    """Apply the stage freeze policy and return named trainable groups."""

    selected_stage = RevoTrainingStage(stage)
    spec = STAGE_SPECS[selected_stage]
    layer_count = infer_num_hidden_layers(model)
    groups: dict[str, list[tuple[str, Any]]] = {name: [] for name in spec.train_groups}
    for name, parameter in model.named_parameters():
        group = classify_revo_parameter(name, num_hidden_layers=layer_count)
        trainable = group in groups
        parameter.requires_grad = trainable
        if trainable:
            groups[group].append((name, parameter))
    empty = [name for name, values in groups.items() if not values]
    if empty:
        raise ValueError(
            f"Revo stage {selected_stage.value} found no parameters for required groups {empty}"
        )
    return groups


def build_revo_optimizer_groups(
    groups: Mapping[str, Iterable[tuple[str, Any]]],
    stage: RevoTrainingStage | str,
    *,
    weight_decay: float,
) -> list[dict[str, object]]:
    """Create per-LR/per-decay AdamW groups with no duplicate parameters."""

    spec = STAGE_SPECS[RevoTrainingStage(stage)]
    no_decay_tokens = ("bias", "norm.weight", "q_norm.weight", "k_norm.weight")
    result: list[dict[str, object]] = []
    identities: set[int] = set()
    for group_name in spec.train_groups:
        values = list(groups[group_name])
        for use_decay in (True, False):
            params = []
            names = []
            for name, parameter in values:
                is_no_decay = any(token in name for token in no_decay_tokens)
                if use_decay == is_no_decay:
                    continue
                identity = id(parameter)
                if identity in identities:
                    raise ValueError(f"duplicate optimizer parameter: {name}")
                identities.add(identity)
                params.append(parameter)
                names.append(name)
            if params:
                result.append(
                    {
                        "params": params,
                        "lr": float(spec.learning_rates[group_name]),
                        "weight_decay": float(weight_decay if use_decay else 0.0),
                        "revo_group": group_name,
                        "parameter_names": tuple(names),
                    }
                )
    return result


def stage_spec(stage: RevoTrainingStage | str) -> RevoStageSpec:
    return STAGE_SPECS[RevoTrainingStage(stage)]
