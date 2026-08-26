"""Run and audit the synthetic collector without making task-success claims."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


TELEOP_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = TELEOP_ROOT.parent
for path in (REPOSITORY_ROOT, TELEOP_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from revo3_teleop.mock import MockCollectionConfig, run_mock_collection
from revo3_v1.data import RevoEpisode


def _json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"expected JSONL objects: {path}")
                rows.append(value)
    return rows


def run_audit(output: Path, *, duration_s: float) -> dict[str, object]:
    results = run_mock_collection(
        MockCollectionConfig(output_root=output, duration_s=duration_s)
    )
    episode_reports = []
    matched = 0
    compared = 0
    for result in results:
        master = Path(result["master"])
        derived = Path(result["revo3_vla"])
        manifest = _json(master / "manifest.json")
        anchors = _jsonl(master / "anchors_30hz.jsonl")
        receipts = {
            str(row["request_id"]): row
            for row in _jsonl(master / "command_receipts.jsonl")
        }
        episode = RevoEpisode.load(derived)
        if episode.num_frames != len(anchors):
            raise AssertionError("derived frame count differs from 30 Hz anchor count")
        for frame_index, anchor in enumerate(anchors):
            receipt = receipts[str(anchor["hand_command_request_id"])]
            exact = np.asarray(receipt["exact_sent_target"], dtype=np.float32)
            compared += 1
            if np.array_equal(exact, episode.action_target_rad[frame_index]):
                matched += 1
            else:
                raise AssertionError("derived action differs from controller exact_sent_target")
        meta = _json(derived / "meta.json")
        if meta.get("contains_emg") is not False or (derived / "streams").exists():
            raise AssertionError("derived Revo episode violated modality separation")
        counts = manifest.get("stream_counts")
        required_streams = {
            "camera",
            "revo_state",
            "tactile",
            "glove",
            "emg",
            "tianji_state",
        }
        if not isinstance(counts, dict) or not required_streams.issubset(counts):
            raise AssertionError("master episode lacks required native streams")
        episode_reports.append(
            {
                "task": result["task"],
                "anchors": len(anchors),
                "stream_counts": counts,
                "master_lifecycle": manifest.get("lifecycle"),
                "revo_exact_sent_matches": len(anchors),
            }
        )
    return {
        "schema_version": "revo3-teleop-mock-audit-v1",
        "verification_scope": "synthetic serialization and causal-plumbing only",
        "task_success_defined": False,
        "training_approved": False,
        "episodes": episode_reports,
        "revo_controller_targets_compared": compared,
        "revo_controller_targets_exact_match": matched,
        "exact_match_rate": matched / compared if compared else 0.0,
        "claims_excluded": [
            "physical Tianji or Revo execution",
            "glove retargeting quality",
            "real EMG classification accuracy",
            "task success or clinical benefit",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--duration-s", type=float, default=0.1)
    arguments = parser.parse_args(argv)
    report = run_audit(arguments.output, duration_s=arguments.duration_s)
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    with arguments.report.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
