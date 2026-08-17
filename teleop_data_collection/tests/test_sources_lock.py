from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys


PROJECT = Path(__file__).resolve().parents[1]


def test_sources_are_commit_pinned_and_vendor_is_ignored() -> None:
    document = json.loads((PROJECT / "sources.lock.json").read_text(encoding="utf-8"))
    assert document["schema_version"] == "revo3-teleop-sources-v1"
    names = set()
    for source in document["sources"]:
        assert source["name"] not in names
        names.add(source["name"])
        assert re.fullmatch(r"[0-9a-f]{40}", source["commit"])
        assert source["url"].startswith("https://github.com/")
        assert source["license"]
        assert source["redistribution"] in {
            "external_checkout",
            "external_only_ack_required",
        }
    ignored = (PROJECT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "/vendor/" in ignored


def test_source_bootstrap_list_is_read_only_and_complete() -> None:
    completed = subprocess.run(
        [sys.executable, str(PROJECT / "scripts" / "bootstrap_sources.py"), "--list"],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [line for line in completed.stdout.splitlines() if line.strip()]
    locked = json.loads((PROJECT / "sources.lock.json").read_text(encoding="utf-8"))["sources"]
    assert len(rows) == len(locked)
    assert all(source["name"] in completed.stdout for source in locked)
