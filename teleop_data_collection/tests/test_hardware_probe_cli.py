from __future__ import annotations

import json

import pytest

from revo3_teleop.cli.hardware_probe import load_probe_config, main


def test_probe_config_requires_file_and_cli_level_opt_in(tmp_path) -> None:
    path = tmp_path / "probe.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "revo3-hardware-probe-config-v1",
                "allow_hardware_probe": False,
                "allow_hardware_write": False,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(PermissionError, match="keeps hardware probe disabled"):
        load_probe_config(path)
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "--config",
                str(path),
                "--output",
                str(tmp_path / "must-not-exist.json"),
                "--allow-hardware-probe",
            ]
        )
    assert exc.value.code == 2
    assert not (tmp_path / "must-not-exist.json").exists()

    data = json.loads(path.read_text(encoding="utf-8"))
    data["allow_hardware_probe"] = True
    path.write_text(json.dumps(data), encoding="utf-8")
    assert load_probe_config(path)["allow_hardware_write"] is False
