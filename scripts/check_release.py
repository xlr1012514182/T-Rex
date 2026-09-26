"""Check source-distribution documentation and configuration (no devices/network)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "README.md", "README_ZH.md", "LICENSE", "NOTICE.md", "CONTRIBUTING.md",
    "docs/DEVELOPMENT.md", "docs/revo3_v1/README.md", "docs/revo3_v1/TRAINING.md",
    "requirements.txt", "requirements-runtime.txt", "requirements-dev.txt",
)
EXCLUDED = (
    "hardware_code/third_party/", "hardware_code/vive_tracker/",
    "dataset_quickstart/third_party/", "teleop_data_collection/vendor/",
)
LOCAL_GENERATED = {
    ".git", ".venv", "__pycache__", ".pytest_cache", "outputs", "logs", "vendor",
    "checkpoints", "data", ".local", "node_modules",
}


def source_files(root: Path, suffix: str):
    for path in sorted(root.rglob("*" + suffix)):
        rel = path.relative_to(root)
        if any(part in LOCAL_GENERATED or part.startswith(".venv-") for part in rel.parts):
            continue
        if rel.as_posix().startswith(EXCLUDED):
            continue
        yield path


def local_link_issues(path: Path, root: Path) -> list[str]:
    """Validate local file/directory targets; URL fragments are not checked."""
    markdown = path.read_text(encoding="utf-8-sig")
    markdown = re.sub(r"```.*?```", "", markdown, flags=re.DOTALL)
    links = re.findall(r"!?\[[^\]\n]*\]\(([^\s)]+)(?:\s+[^)]*)?\)", markdown)
    links += re.findall(r'<(?:img|a)\b[^>]*(?:src|href)=[\"\']([^\"\']+)', markdown)
    issues = []
    for link in links:
        parsed = urlsplit(link.strip("<>"))
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        target = (path.parent / unquote(parsed.path)).resolve()
        if not target.is_relative_to(root.resolve()):
            issues.append(f"{path.relative_to(root)}: local link leaves repository")
        elif not target.exists():
            issues.append(f"{path.relative_to(root)}: missing target {parsed.path}")
    return issues


def pinned_versions(text: str) -> dict[str, str]:
    return {
        name.lower().replace("_", "-"): version
        for name, version in re.findall(r"[\"\']?([A-Za-z0-9_-]+)==([^\s\"\',]+)", text)
    }


def dependency_issues(root: Path) -> list[str]:
    issues = []
    manifest = root / "pyproject.toml"
    runtime = root / "requirements-runtime.txt"
    if not manifest.is_file() or not runtime.is_file():
        return ["dependency manifest missing"]
    project_pins = pinned_versions(manifest.read_text(encoding="utf-8"))
    runtime_pins = pinned_versions(runtime.read_text(encoding="utf-8"))
    for name, version in runtime_pins.items():
        if project_pins.get(name) != version:
            issues.append(f"requirements-runtime.txt: inconsistent pin for {name}")
    for rel, expected in (
        ("requirements.txt", "-e ."),
        ("requirements-dev.txt", "-r requirements-runtime.txt"),
        ("requirements-demo.txt", "-r requirements-dev.txt"),
    ):
        path = root / rel
        if not path.is_file() or expected not in path.read_text(encoding="utf-8").splitlines():
            issues.append(f"{rel}: missing dependency entry point")
    return issues


def unsafe_example_flags(value, prefix="") -> list[str]:
    issues = []
    if isinstance(value, dict):
        for key, child in value.items():
            location = f"{prefix}.{key}" if prefix else key
            if key in {"allow_hardware_write", "allow_hardware_connect"} and child is not False:
                issues.append(location)
            issues.extend(unsafe_example_flags(child, location))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            issues.extend(unsafe_example_flags(child, f"{prefix}[{index}]"))
    return issues


def collect_issues(root: Path) -> list[str]:
    issues = [f"missing release file: {rel}" for rel in REQUIRED if not (root / rel).is_file()]
    if (root / "audit").exists():
        issues.append("audit/: run-history directory is not part of the source distribution")
    issues.extend(dependency_issues(root))
    for path in source_files(root, ".md"):
        issues.extend(local_link_issues(path, root))
    for path in source_files(root, ".ipynb"):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") == "code" and (
                cell.get("outputs") or cell.get("execution_count") is not None
            ):
                issues.append(f"{path.relative_to(root)}: cell {index} contains execution output")
    for parent in (root / "config", root / "teleop_data_collection/configs"):
        for path in sorted(parent.glob("*.example.json")):
            value = json.loads(path.read_text(encoding="utf-8"))
            for location in unsafe_example_flags(value):
                issues.append(f"{path.relative_to(root)}: {location} must default to false")
    for parent in (root / "scripts", root / "utils"):
        for path in sorted(parent.glob("*.sh")):
            content = path.read_text(encoding="utf-8")
            if "/mnt/amlfs-" in content or "/shared/human_egocentric/" in content:
                issues.append(f"{path.relative_to(root)}: machine-specific path")
            if b"\r\n" in path.read_bytes():
                issues.append(f"{path.relative_to(root)}: shell wrapper must use LF")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    issues = collect_issues(args.root.resolve())
    if issues:
        for issue in issues:
            print(f"FAIL {issue}")
        print(f"{len(issues)} release check(s) failed")
        return 1
    print("Release checks passed: documents, dependencies, notebooks, example flags and wrappers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
