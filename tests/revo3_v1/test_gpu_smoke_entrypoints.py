from __future__ import annotations

import ast
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]


def test_qwen_smoke_entrypoint_is_parseable_and_bounded() -> None:
    path = PROJECT / "scripts" / "revo3_v1_qwen_smoke.py"
    source = path.read_text(encoding="utf-8")
    ast.parse(source)
    assert "--require-schema" in source
    assert "task_success_defined" in source
    assert "claims_excluded" in source
    assert "DEFAULT_REVISION" in source


def test_trex_server_has_bounded_smoke_and_request_modes() -> None:
    path = PROJECT / "scripts" / "test.py"
    source = path.read_text(encoding="utf-8")
    ast.parse(source)
    assert '"--smoke_only"' in source
    assert '"--max_requests"' in source
    assert "ZMQ listener was not started" in source
    assert '"--camera_profile"' in source
    assert "REVO3_FULL_CENTER_PROFILE" in source
    assert "image_wrist_right=fixed_center" in source
