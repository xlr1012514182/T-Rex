"""Leakage-resistant episode split manifest for Revo policy/tokenizer data."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Iterable, Mapping


DEFAULT_DURATION_TARGETS_HOURS = {
    "midtrain_train": 12.0,
    "sft_train": 4.0,
    "development": 2.0,
    "locked_test": 2.0,
}


class CorpusSplit(str, Enum):
    MIDTRAIN_TRAIN = "midtrain_train"
    SFT_TRAIN = "sft_train"
    DEVELOPMENT = "development"
    LOCKED_TEST = "locked_test"


@dataclass(frozen=True)
class SplitEntry:
    episode_id: str
    split: CorpusSplit
    day: str
    object_instance: str
    operator: str
    duration_seconds: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "SplitEntry":
        entry = cls(
            episode_id=str(value.get("episode_id", "")).strip(),
            split=CorpusSplit(str(value.get("split", ""))),
            day=str(value.get("day", "")).strip(),
            object_instance=str(value.get("object_instance", "")).strip(),
            operator=str(value.get("operator", "")).strip(),
            duration_seconds=float(value.get("duration_seconds", 0.0)),
        )
        if not all((entry.episode_id, entry.day, entry.object_instance, entry.operator)):
            raise ValueError("split entries require episode_id/day/object_instance/operator")
        if entry.duration_seconds <= 0:
            raise ValueError("split entries require a positive duration_seconds")
        return entry

    @property
    def group_key(self) -> tuple[str, str, str]:
        return (self.day, self.object_instance, self.operator)


@dataclass(frozen=True)
class RevoCorpusSplitManifest:
    entries: tuple[SplitEntry, ...]
    locked_test_isolation: tuple[str, ...]
    duration_targets_hours: Mapping[str, float]
    duration_tolerance_hours: float
    duration_targets_enforced: bool

    @classmethod
    def load(cls, path: str | Path) -> "RevoCorpusSplitManifest":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict) or payload.get("schema_version") != "revo3-corpus-split-v1":
            raise ValueError("unsupported Revo corpus split manifest")
        raw_entries = payload.get("episodes")
        if not isinstance(raw_entries, list) or not raw_entries:
            raise ValueError("split manifest must contain a non-empty episodes list")
        protocol = payload.get("evaluation_protocol")
        if not isinstance(protocol, dict):
            raise ValueError("split manifest requires evaluation_protocol")
        isolation = protocol.get("locked_test_isolation")
        if not isinstance(isolation, list) or not isolation:
            raise ValueError("evaluation_protocol.locked_test_isolation must be non-empty")
        targets = payload.get("duration_targets_hours")
        if not isinstance(targets, dict):
            raise ValueError("split manifest requires duration_targets_hours")
        result = cls(
            tuple(SplitEntry.from_mapping(item) for item in raw_entries),
            tuple(str(item) for item in isolation),
            {str(key): float(value) for key, value in targets.items()},
            float(payload.get("duration_tolerance_hours", 0.5)),
            bool(payload.get("duration_targets_enforced", True)),
        )
        result.validate()
        return result

    def validate(self) -> None:
        episode_ids = [entry.episode_id for entry in self.entries]
        if len(set(episode_ids)) != len(episode_ids):
            raise ValueError("an episode may appear in exactly one split")
        group_splits: dict[tuple[str, str, str], CorpusSplit] = {}
        for entry in self.entries:
            previous = group_splits.setdefault(entry.group_key, entry.split)
            if previous != entry.split:
                raise ValueError(
                    "day/object-instance/operator group leaks across splits: "
                    f"{entry.group_key}"
                )
        present = {entry.split for entry in self.entries}
        required = set(CorpusSplit)
        missing = required - present
        if missing:
            raise ValueError(f"split manifest is incomplete; missing {sorted(x.value for x in missing)}")
        allowed_dimensions = {"day", "object_instance", "operator"}
        unknown = set(self.locked_test_isolation) - allowed_dimensions
        if unknown:
            raise ValueError(f"unknown locked-test isolation dimensions: {sorted(unknown)}")
        required_isolation = {"day", "object_instance"}
        missing_isolation = required_isolation - set(self.locked_test_isolation)
        if missing_isolation:
            raise ValueError(
                "locked_test_isolation must freeze both day and object_instance; "
                f"missing={sorted(missing_isolation)}"
            )
        locked = [entry for entry in self.entries if entry.split == CorpusSplit.LOCKED_TEST]
        non_locked = [entry for entry in self.entries if entry.split != CorpusSplit.LOCKED_TEST]
        for dimension in self.locked_test_isolation:
            locked_values = {getattr(entry, dimension) for entry in locked}
            train_values = {getattr(entry, dimension) for entry in non_locked}
            overlap = locked_values & train_values
            if overlap:
                raise ValueError(
                    f"locked_test {dimension} is not OOD-isolated; overlap={sorted(overlap)}"
                )
        if set(self.duration_targets_hours) != {item.value for item in CorpusSplit}:
            raise ValueError(
                "duration_targets_hours must declare midtrain/SFT/development/locked_test"
            )
        if any(value <= 0 for value in self.duration_targets_hours.values()):
            raise ValueError("all duration targets must be positive")
        if self.duration_tolerance_hours < 0:
            raise ValueError("duration_tolerance_hours must be non-negative")
        if self.duration_targets_enforced:
            deviations = {
                name: row
                for name, row in self.duration_report().items()
                if not row["within_tolerance"]
            }
            if deviations:
                raise ValueError(
                    "corpus duration targets were not met; either collect the frozen corpus "
                    "or record an explicit reviewed waiver: "
                    f"{deviations}"
                )

    def duration_report(self) -> dict[str, dict[str, float | bool]]:
        actual_seconds = {item.value: 0.0 for item in CorpusSplit}
        for entry in self.entries:
            actual_seconds[entry.split.value] += float(entry.duration_seconds)
        report: dict[str, dict[str, float | bool]] = {}
        for split, target in self.duration_targets_hours.items():
            actual = actual_seconds[split] / 3600.0
            deviation = actual - float(target)
            report[split] = {
                "target_hours": float(target),
                "actual_hours": actual,
                "deviation_hours": deviation,
                "within_tolerance": abs(deviation) <= self.duration_tolerance_hours,
            }
        return report

    def episode_ids(self, splits: Iterable[CorpusSplit | str]) -> tuple[str, ...]:
        wanted = {CorpusSplit(item) for item in splits}
        return tuple(entry.episode_id for entry in self.entries if entry.split in wanted)

    def assert_matches_episode_root(self, root: str | Path) -> None:
        observed = {path.parent.name for path in Path(root).glob("*/meta.json")}
        declared = {entry.episode_id for entry in self.entries}
        missing = declared - observed
        unassigned = observed - declared
        if missing or unassigned:
            raise ValueError(
                f"split/source mismatch: missing={sorted(missing)}, unassigned={sorted(unassigned)}"
            )
