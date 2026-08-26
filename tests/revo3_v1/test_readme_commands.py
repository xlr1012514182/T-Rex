from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[2]


def _executable_blocks(markdown: str) -> str:
    blocks = re.findall(r"```(?:powershell|bash)\n(.*?)```", markdown, flags=re.DOTALL)
    assert blocks, "README must expose copy-and-run PowerShell/Bash blocks"
    return "\n".join(blocks)


def test_root_readmes_only_publish_clone_runnable_commands():
    english_path = REPO_ROOT / "README.md"
    chinese_path = REPO_ROOT / "README_ZH.md"
    requirements_path = REPO_ROOT / "requirements-demo.txt"

    assert english_path.is_file()
    assert chinese_path.is_file()
    assert requirements_path.is_file()
    assert not (REPO_ROOT / "README_EN.md").exists()

    english = english_path.read_text(encoding="utf-8")
    chinese = chinese_path.read_text(encoding="utf-8")
    assert "[中文](README_ZH.md)" in english
    assert "[English](README.md)" in chinese

    required_fragments = (
        "git clone --branch agent/revo3-v1-demo --single-branch",
        "requirements-demo.txt",
        "scripts/revo3_v1_runtime.py --mode simulation --task all --servo-ticks 120",
        "scripts/revo3_v1_generate_emg.py",
        "scripts/revo3_v1_train_emg.py",
        "scripts/revo3_v1_trex.py train --help",
        "scripts/revo3_v1_trex.py serve --help",
        "-m pytest -q",
    )
    forbidden_placeholders = (
        "/checkpoints/",
        "/data/",
        "checkpoint-X-Y",
        "my_hardware.bindings",
        "hf download",
    )

    for markdown in (english, chinese):
        commands = _executable_blocks(markdown)
        for fragment in required_fragments:
            assert fragment in commands
        for placeholder in forbidden_placeholders:
            assert placeholder not in commands

    for relative_path in (
        "scripts/revo3_v1_runtime.py",
        "scripts/revo3_v1_generate_emg.py",
        "scripts/revo3_v1_train_emg.py",
        "scripts/revo3_v1_trex.py",
        "tests/revo3_v1",
        "teleop_data_collection/tests",
    ):
        assert (REPO_ROOT / relative_path).exists(), relative_path
