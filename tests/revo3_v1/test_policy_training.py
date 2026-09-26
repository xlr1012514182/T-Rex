from types import SimpleNamespace

import torch

from revo3_v1.policy.training import (
    RevoTrainingStage,
    build_revo_optimizer_groups,
    classify_revo_parameter,
    configure_revo_trainable_parameters,
    stage_spec,
)


class _NamedModel:
    def __init__(self):
        self.config = SimpleNamespace(num_hidden_layers=8)
        names = (
            "visual.patch.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.3.mlp_action.gate_proj.weight",
            "model.layers.4.mlp_action.gate_proj.weight",
            "model.layers.7.self_attn.q_proj_action.weight",
            "model.layers.0.mlp_tactile.gate_proj.weight",
            "model.layers.7.self_attn.q_proj_tactile.weight",
            "model.norm_action.weight",
            "x_embedder.mlp.fc1.weight",
            "state_embedder.mlp.fc1.weight",
            "final_layer.mlp.fc2.weight",
            "tacf6_embedder.mlp.fc1.weight",
            "tactile_code_embedder.weight",
            "deform_proj.mlp.fc1.weight",
            "final_layer_tactile.mlp.fc2.weight",
            "tactile_vqvae.encoder.weight",
            "deform_encoder.stem.weight",
            "flare_queries",
            "flare_proj.0.weight",
        )
        self._params = [(name, torch.nn.Parameter(torch.zeros(1))) for name in names]

    def named_parameters(self):
        return iter(self._params)


def test_frozen_stage_specs_match_reviewed_limits():
    mid = stage_spec("midtrain")
    sft = stage_spec("sft")
    assert (mid.max_steps, mid.max_epochs, mid.effective_batch) == (22_000, 3, 64)
    assert (sft.max_steps, sft.max_epochs, sft.effective_batch) == (6_500, 3, 64)
    assert mid.flare_enabled and mid.flare_loss_weight == 0.5
    assert mid.tactile_dropout == 0.10 and mid.state_dropout == 0.05


def test_only_last_four_action_blocks_are_classified():
    assert classify_revo_parameter(
        "model.layers.3.mlp_action.gate_proj.weight", num_hidden_layers=8
    ) is None
    assert classify_revo_parameter(
        "model.layers.4.mlp_action.gate_proj.weight", num_hidden_layers=8
    ) == "action_last4"
    assert classify_revo_parameter(
        "model.layers.0.mlp_tactile.gate_proj.weight", num_hidden_layers=8
    ) == "tactile_expert"


def test_midtrain_freezes_vlm_visual_vq_and_deform_and_builds_distinct_lrs():
    model = _NamedModel()
    groups = configure_revo_trainable_parameters(model, RevoTrainingStage.MIDTRAIN)
    named = dict(model.named_parameters())
    assert not named["visual.patch.weight"].requires_grad
    assert not named["model.layers.0.mlp.gate_proj.weight"].requires_grad
    assert not named["model.layers.3.mlp_action.gate_proj.weight"].requires_grad
    assert not named["tactile_vqvae.encoder.weight"].requires_grad
    assert not named["deform_encoder.stem.weight"].requires_grad
    assert named["model.layers.4.mlp_action.gate_proj.weight"].requires_grad
    assert named["model.layers.0.mlp_tactile.gate_proj.weight"].requires_grad

    optimizer_groups = build_revo_optimizer_groups(
        groups, "midtrain", weight_decay=0.01
    )
    observed = {entry["revo_group"]: entry["lr"] for entry in optimizer_groups}
    assert observed["new_action_state"] == 1e-4
    assert observed["tactile_expert"] == 3e-5
    assert observed["action_last4"] == 1e-5
    assert observed["flare"] == 5e-6


def test_w0_excludes_every_tactile_parameter():
    model = _NamedModel()
    configure_revo_trainable_parameters(model, "w0")
    for name, parameter in model.named_parameters():
        if "tactile" in name or name.startswith(("tacf6", "deform_")):
            assert not parameter.requires_grad, name
