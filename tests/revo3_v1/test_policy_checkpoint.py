from types import ModuleType, SimpleNamespace
import sys

import numpy as np

from revo3_v1.policy import (
    build_revo_trex_model,
    load_revo_compatible_state_dict,
)


def test_model_factory_declares_five_fingers_without_padding(monkeypatch):
    captured = {}

    class FakeVLA:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    module = ModuleType("qwen_vla.modeling_vla")
    module.Qwen3VLVLAModel = FakeVLA
    monkeypatch.setitem(sys.modules, "qwen_vla.modeling_vla", module)
    build_revo_trex_model(object())
    assert captured["action_dim"] == 21
    assert captured["action_chunk"] == 16
    assert captured["tacf6_dim"] == 6
    assert captured["tactile_num_fingers"] == 5


def test_checkpoint_loader_reinitializes_dimension_bound_heads():
    class FakeModel:
        def __init__(self):
            self.loaded = None

        def state_dict(self):
            return {
                "shared.weight": np.zeros((2, 2)),
                "x_embedder.mlp.fc1.weight": np.zeros((2, 21)),
            }

        def load_state_dict(self, state, strict=False):
            assert not strict
            self.loaded = state
            missing = [key for key in self.state_dict() if key not in state]
            return SimpleNamespace(missing_keys=missing, unexpected_keys=[])

    model = FakeModel()
    report = load_revo_compatible_state_dict(
        model,
        {
            "shared.weight": np.ones((2, 2)),
            "x_embedder.mlp.fc1.weight": np.ones((2, 62)),
        },
    )
    assert report.loaded == ("shared.weight",)
    assert report.reinitialized_dimension_bound == (
        "x_embedder.mlp.fc1.weight",
    )
    assert "x_embedder.mlp.fc1.weight" not in model.loaded
