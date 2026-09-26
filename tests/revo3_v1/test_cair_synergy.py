import json

import numpy as np
import pytest

from revo3_v1.tactile import (
    ClosingSynergyArtifact,
    ReflexConfig,
    fit_closing_synergy,
)


def _consistent_deltas(count=25):
    base = np.linspace(0.1, 1.0, 21, dtype=np.float32)
    return np.stack([base * (0.8 + 0.01 * i) for i in range(count)])


def test_fit_synergy_uses_authorized_deltas_and_roundtrips_hash(tmp_path):
    artifact = fit_closing_synergy(
        _consistent_deltas(), source_manifest_sha256="a" * 64
    )
    assert artifact.sample_count == 25
    assert np.isclose(np.max(np.abs(artifact.vector)), 1.0)
    path = artifact.save(tmp_path / "synergy.json")
    loaded = ClosingSynergyArtifact.load(path)
    assert loaded.artifact_sha256 == artifact.artifact_sha256
    config = ReflexConfig.from_artifact(loaded)
    assert config.hardware_mode and config.enabled
    assert config.synergy_artifact_sha256 == artifact.artifact_sha256


def test_synergy_fit_rejects_unstable_or_insufficient_demonstrations():
    with pytest.raises(ValueError, match="insufficient"):
        fit_closing_synergy(
            _consistent_deltas(3), source_manifest_sha256="b" * 64
        )
    inconsistent = _consistent_deltas()
    inconsistent[-5:] *= -1
    with pytest.raises(ValueError, match="held-out"):
        fit_closing_synergy(inconsistent, source_manifest_sha256="c" * 64)


def test_artifact_tamper_and_unversioned_hardware_config_fail_closed(tmp_path):
    artifact = fit_closing_synergy(
        _consistent_deltas(), source_manifest_sha256="d" * 64
    )
    path = artifact.save(tmp_path / "synergy.json")
    value = json.loads(path.read_text(encoding="utf-8"))
    value["closing_synergy"][0] *= -1
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        ClosingSynergyArtifact.load(path)
    with pytest.raises(ValueError, match="versioned synergy artifact"):
        ReflexConfig(
            enabled=True,
            closing_synergy=np.ones(21),
            hardware_mode=True,
        )
