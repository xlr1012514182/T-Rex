"""Fetch audited SDKs into an ignored, commit-pinned vendor directory.

This script never vendors third-party code into the repository history.  It
creates detached external checkouts under ``teleop_data_collection/vendor``
and verifies every resulting HEAD against ``sources.lock.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = PROJECT_ROOT / "sources.lock.json"
VENDOR_ROOT = PROJECT_ROOT / "vendor"


def _run(arguments: Iterable[str], *, cwd: Path) -> str:
    completed = subprocess.run(
        list(arguments),
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def _load_sources() -> dict[str, dict[str, object]]:
    with LOCK_PATH.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if document.get("schema_version") != "revo3-teleop-sources-v1":
        raise RuntimeError("unsupported sources.lock.json schema")
    rows = document.get("sources")
    if not isinstance(rows, list):
        raise RuntimeError("sources.lock.json has no sources list")
    result = {}
    for row in rows:
        if not isinstance(row, dict) or not row.get("name"):
            raise RuntimeError("malformed source entry")
        result[str(row["name"])] = row
    return result


def _checkout(source: dict[str, object], *, acknowledge_external_license: bool) -> Path:
    name = str(source["name"])
    commit = str(source["commit"])
    policy = str(source.get("redistribution", ""))
    if policy == "external_only_ack_required" and not acknowledge_external_license:
        raise RuntimeError(
            f"{name} has an unverified/mixed external license boundary; "
            "inspect sources.lock.json and pass --ack-external-license to fetch it"
        )
    destination = (VENDOR_ROOT / name).resolve()
    destination.relative_to(VENDOR_ROOT.resolve())
    if destination.exists():
        if not (destination / ".git").exists():
            raise RuntimeError(f"refusing to overwrite non-git directory: {destination}")
        observed = _run(("git", "rev-parse", "HEAD"), cwd=destination)
        if observed != commit:
            raise RuntimeError(
                f"existing {name} checkout is {observed}; expected {commit}. "
                "Remove or relocate it manually before retrying."
            )
        return destination

    temporary = VENDOR_ROOT / f".{name}.inprogress"
    if temporary.exists():
        raise RuntimeError(f"incomplete checkout already exists: {temporary}")
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        _run(("git", "init"), cwd=temporary)
        _run(("git", "remote", "add", "origin", str(source["url"])), cwd=temporary)
        _run(
            ("git", "fetch", "--filter=blob:none", "--depth", "1", "origin", commit),
            cwd=temporary,
        )
        _run(("git", "checkout", "--detach", "FETCH_HEAD"), cwd=temporary)
        observed = _run(("git", "rev-parse", "HEAD"), cwd=temporary)
        if observed != commit:
            raise RuntimeError(f"commit verification failed for {name}: {observed}")
        temporary.replace(destination)
    except Exception:
        # Only the script-owned, explicitly resolved temporary path is removed.
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help="source names; defaults to license-clear sources")
    parser.add_argument(
        "--ack-external-license",
        action="store_true",
        help="acknowledge mixed/unverified license notes before external-only checkout",
    )
    parser.add_argument("--list", action="store_true", help="list locked sources without fetching")
    arguments = parser.parse_args(argv)
    if shutil.which("git") is None:
        raise RuntimeError("git executable not found")
    sources = _load_sources()
    if arguments.list:
        for name, source in sources.items():
            print(f"{name}\t{source['commit']}\t{source['license']}")
        return 0
    names = arguments.names or [
        name
        for name, source in sources.items()
        if source.get("redistribution") != "external_only_ack_required"
    ]
    unknown = sorted(set(names) - set(sources))
    if unknown:
        parser.error(f"unknown source name(s): {', '.join(unknown)}")
    VENDOR_ROOT.mkdir(parents=True, exist_ok=True)
    for name in names:
        path = _checkout(
            sources[name],
            acknowledge_external_license=arguments.ack_external_license,
        )
        print(f"verified {name}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
