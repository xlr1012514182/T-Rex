from pathlib import Path
import re

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("name", ["README.md", "README_ZH.md"])
def test_root_readmes_publish_local_runnable_commands(name):
    markdown = (REPO_ROOT / name).read_text(encoding="utf-8")
    blocks = re.findall(r"```(?:powershell|bash)\n(.*?)```", markdown, flags=re.DOTALL)
    assert blocks, "README must expose copy-and-run PowerShell/Bash blocks"
    commands = "\n".join(blocks)
    for fragment in (
        "requirements-dev.txt",
        "scripts/revo3_v1_runtime.py --mode simulation --task all --servo-ticks 120",
        "scripts/revo3_v1_trex.py train --help",
        "scripts/revo3_v1_trex.py serve --help",
        "-m pytest -q",
    ):
        assert fragment in commands
    for placeholder in ("/checkpoints/", "/data/", "checkpoint-X-Y", "hf download"):
        assert placeholder not in commands
    for link in ("docs/DEVELOPMENT.md", "docs/revo3_v1/TRAINING.md", "NOTICE.md"):
        assert link in markdown
        assert (REPO_ROOT / link).is_file()
    assert "deterministic simulation" in markdown or "确定性模拟" in markdown
    assert "[中文](README_ZH.md)" in markdown or "[English](README.md)" in markdown


def test_synthetic_examples_are_kept_in_development_guide():
    guide = (REPO_ROOT / "docs/DEVELOPMENT.md").read_text(encoding="utf-8")
    for script in ("revo3_v1_generate_emg.py", "revo3_v1_train_emg.py", "revo3_v1_generate_robot_demo.py"):
        assert script in guide
        assert (REPO_ROOT / "scripts" / script).is_file()
    assert "--from-scratch-ablation" in guide
    assert "--allow-window-reset-fallback" in guide
    assert "合成" in guide
