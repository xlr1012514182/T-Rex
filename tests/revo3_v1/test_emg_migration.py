import pytest

torch = pytest.importorskip("torch")

from revo3_v1.emg.migration import GNI_SOURCE_COMMIT, migrate_gni_encoder
from revo3_v1.emg.model import GNIClassifier, GNIModelConfig


def config(*, channels, kernel, stride, outputs):
    return GNIModelConfig(
        input_channels=channels,
        conv_output_channels=16,
        kernel_width=kernel,
        stride=stride,
        lstm_hidden_size=12,
        lstm_num_layers=2,
        output_channels=outputs,
        label_names=tuple(f"LABEL_{index}" for index in range(outputs)),
        dropout=0.0,
    )


def official_style_checkpoint(tmp_path, *, omit=""):
    source = GNIClassifier(config(channels=16, kernel=21, stride=10, outputs=9))
    state = {}
    for key, value in source.state_dict().items():
        if key == omit:
            continue
        fill = 0.25 if key.startswith(("lstm.", "post_")) else 0.75
        state["network." + key] = torch.full_like(value, fill)
    path = tmp_path / "official.ckpt"
    torch.save({"state_dict": state, "epoch": 1}, path)
    return path


def test_migration_inherits_only_shape_compatible_temporal_encoder(tmp_path):
    target = GNIClassifier(config(channels=8, kernel=3, stride=1, outputs=5))
    original_conv = target.conv_layer.weight.detach().clone()
    original_head = target.projection.weight.detach().clone()
    report = migrate_gni_encoder(
        target,
        official_style_checkpoint(tmp_path),
        source_commit=GNI_SOURCE_COMMIT,
    )
    assert report.source_commit == GNI_SOURCE_COMMIT
    assert report.inherited_keys
    assert "conv_layer.weight" in report.reinitialized_keys
    assert "projection.weight" in report.reinitialized_keys
    torch.testing.assert_close(target.conv_layer.weight, original_conv)
    torch.testing.assert_close(target.projection.weight, original_head)
    assert torch.allclose(target.lstm.weight_ih_l0, torch.full_like(target.lstm.weight_ih_l0, 0.25))


def test_migration_fails_on_missing_encoder_or_unknown_tensor(tmp_path):
    target = GNIClassifier(config(channels=8, kernel=3, stride=1, outputs=5))
    missing = official_style_checkpoint(tmp_path, omit="lstm.weight_ih_l0")
    with pytest.raises(ValueError, match="lacks required"):
        migrate_gni_encoder(target, missing, source_commit=GNI_SOURCE_COMMIT)
    payload = torch.load(missing, weights_only=False)
    payload["state_dict"]["network.unexpected.weight"] = torch.ones(1)
    torch.save(payload, missing)
    with pytest.raises(ValueError, match="Unexpected"):
        migrate_gni_encoder(target, missing, source_commit=GNI_SOURCE_COMMIT)
