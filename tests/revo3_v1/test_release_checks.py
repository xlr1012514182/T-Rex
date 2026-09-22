from pathlib import Path
import os
import runpy
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
CHECKS = runpy.run_path(str(ROOT / "scripts/check_release.py"))


def test_release_content_checks_pass():
    assert CHECKS["collect_issues"](ROOT) == []


def test_link_checker_finds_missing_local_target(tmp_path):
    path = tmp_path / "README.md"
    path.write_text("[guide](missing.md)\n[remote](https://example.org/guide)", encoding="utf-8")
    issues = CHECKS["local_link_issues"](path, tmp_path)
    assert len(issues) == 1
    assert "missing.md" in issues[0]


def test_link_checker_rejects_outside_repository(tmp_path):
    path = tmp_path / "README.md"
    path.write_text("[outside](../README.md)", encoding="utf-8")
    assert "leaves repository" in CHECKS["local_link_issues"](path, tmp_path)[0]


def test_link_checker_ignores_code_examples(tmp_path):
    path = tmp_path / "README.md"
    path.write_text("```python\nvalues[index](argument)\n```", encoding="utf-8")
    assert CHECKS["local_link_issues"](path, tmp_path) == []


def test_nested_example_write_permissions_are_closed():
    check = CHECKS["unsafe_example_flags"]
    assert check({"controller": {"allow_hardware_write": False}}) == []
    assert check({"controller": {"allow_hardware_write": True}}) == ["controller.allow_hardware_write"]
    assert check({"sources": [{"allow_hardware_connect": "false"}]}) == ["sources[0].allow_hardware_connect"]


def test_shared_dependency_pins_match():
    assert CHECKS["dependency_issues"](ROOT) == []


def test_diff_training_help_resolves_project_without_pythonpath(tmp_path):
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/train_revo_deform_ae.py"), "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "--data-root" in result.stdout
    assert "--split-manifest" in result.stdout
