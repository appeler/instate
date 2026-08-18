"""Deterministic split and manifest utilities for model evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

SplitName = Literal["train", "validation", "test"]

SPLIT_SEED = 0
TRAIN_FRACTION = 0.8
VALIDATION_FRACTION = 0.1


@dataclass(frozen=True)
class EvaluationSplits:
    """Disjoint surname membership for training, selection, and final evaluation."""

    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]

    def members(self, split: SplitName) -> tuple[str, ...]:
        """Return the members assigned to ``split``."""
        return getattr(self, split)


def split_surnames(
    surnames: list[str], seed: int = SPLIT_SEED
) -> EvaluationSplits:
    """Assign unique surnames with a stable hash-based 80/10/10 split.

    Hash assignment makes membership independent of input row order. The validation
    split is available for epoch-level model selection; the test split is reserved
    for explicit evaluation of a saved checkpoint.
    """
    members: dict[SplitName, list[str]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    train_cut = 8 * (1 << 64) // 10
    validation_cut = 9 * (1 << 64) // 10
    for surname in sorted(set(surnames)):
        digest = hashlib.sha256(f"{seed}\0{surname}".encode()).digest()
        assignment = int.from_bytes(digest[:8], "big")
        if assignment < train_cut:
            split: SplitName = "train"
        elif assignment < validation_cut:
            split = "validation"
        else:
            split = "test"
        members[split].append(surname)
    return EvaluationSplits(**{key: tuple(value) for key, value in members.items()})


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_members(members: tuple[str, ...] | list[str]) -> str:
    """Hash sorted membership with unambiguous length-prefixed UTF-8 records."""
    digest = hashlib.sha256()
    for member in sorted(members):
        encoded = member.encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def split_manifest(splits: EvaluationSplits) -> dict[str, dict[str, str | int]]:
    """Describe split membership without copying surnames into the manifest."""
    return {
        name: {
            "count": len(members),
            "membership_sha256": sha256_members(members),
        }
        for name, members in (
            ("train", splits.train),
            ("validation", splits.validation),
            ("test", splits.test),
        )
    }


def write_run_manifest(
    path: str | Path,
    *,
    task: str,
    data_path: str | Path,
    checkpoint_path: str | Path,
    labels: list[str],
    splits: EvaluationSplits,
    evaluated_split: SplitName,
    evaluated_members: list[str],
    metrics: dict[str, float],
    seed: int,
    source_selection: dict[str, int | None] | None = None,
) -> None:
    """Write the complete contract for one training or checkpoint-evaluation run."""
    data_path = Path(data_path)
    checkpoint_path = Path(checkpoint_path)
    manifest = {
        "schema_version": 1,
        "task": task,
        "assignment": {
            "unit": "normalized_surname",
            "algorithm": "sha256_first_64_bits",
            "seed": seed,
            "fractions": {"train": 0.8, "validation": 0.1, "test": 0.1},
        },
        "data": {
            "filename": data_path.name,
            "sha256": sha256_file(data_path),
            "selection": source_selection or {},
        },
        "model": {
            "filename": checkpoint_path.name,
            "sha256": sha256_file(checkpoint_path),
        },
        "labels": labels,
        "splits": split_manifest(splits),
        "evaluation": {
            "split": evaluated_split,
            "count": len(evaluated_members),
            "membership_sha256": sha256_members(evaluated_members),
            "metrics": metrics,
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
