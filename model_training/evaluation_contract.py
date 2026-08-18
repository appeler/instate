"""Deterministic split and manifest utilities for model evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import torch

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


@dataclass
class BestValidationCheckpoint:
    """Keep and restore the earliest epoch with the best validation metric."""

    metric: str = "mass_top3"
    best_epoch: int | None = None
    best_score: float = float("-inf")
    best_metrics: dict[str, float] = field(default_factory=dict)
    _state: dict[str, torch.Tensor] = field(default_factory=dict, repr=False)

    def consider(
        self, model: torch.nn.Module, epoch: int, metrics: dict[str, float]
    ) -> bool:
        """Save ``model`` when ``metrics`` strictly improve the selection score."""
        score = metrics[self.metric]
        if score <= self.best_score:
            return False
        self.best_epoch = epoch
        self.best_score = score
        self.best_metrics = metrics.copy()
        self._state = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
        return True

    def restore(self, model: torch.nn.Module) -> None:
        """Restore the selected model state."""
        if self.best_epoch is None:
            raise RuntimeError("no validation checkpoint was selected")
        model.load_state_dict(self._state)

    def manifest(self, total_epochs: int) -> dict[str, str | int | float | bool]:
        """Describe the selected epoch for a training manifest."""
        if self.best_epoch is None:
            raise RuntimeError("no validation checkpoint was selected")
        return {
            "metric": self.metric,
            "mode": "max",
            "best_epoch": self.best_epoch,
            "best_score": self.best_score,
            "total_epochs": total_epochs,
            "restored_before_save": True,
        }


class EvaluationContractError(ValueError):
    """Raised when a checkpoint is ineligible for untouched-test evaluation."""


def split_surnames(surnames: list[str], seed: int = SPLIT_SEED) -> EvaluationSplits:
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
    from instate.nnets import canonicalize_name

    canonical_surnames = {
        canonical for surname in surnames if (canonical := canonicalize_name(surname))
    }
    for surname in sorted(canonical_surnames):
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
    run_kind: Literal["training", "evaluation"],
    test_eligibility: dict[str, Any],
    source_selection: dict[str, int | None] | None = None,
    model_selection: dict[str, str | int | float | bool] | None = None,
    provenance: dict[str, str] | None = None,
) -> None:
    """Write the complete contract for one training or checkpoint-evaluation run."""
    data_path = Path(data_path)
    checkpoint_path = Path(checkpoint_path)
    manifest = {
        "schema_version": 1,
        "run_kind": run_kind,
        "task": task,
        "assignment": {
            "unit": "canonical_model_input",
            "canonicalization": "lowercase_then_keep_ascii_a_to_z",
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
        "model_selection": model_selection,
        "test_eligibility": test_eligibility,
        "provenance": provenance or {},
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


def validate_test_eligibility(
    manifest_path: str | Path,
    *,
    task: str,
    data_path: str | Path,
    checkpoint_path: str | Path,
    labels: list[str],
    splits: EvaluationSplits,
    seed: int,
    source_selection: dict[str, int | None] | None = None,
) -> dict[str, str]:
    """Validate that a checkpoint may be labeled as untouched-test evaluated.

    The training manifest must bind the same task, data bytes, checkpoint bytes,
    seed, canonical split membership, label order, and source selection. Legacy
    and arbitrary checkpoints fail closed.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise EvaluationContractError(
            "untouched-test evaluation requires an eligible training manifest"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise EvaluationContractError("training manifest is not valid JSON") from error
    if not isinstance(manifest, dict):
        raise EvaluationContractError("training manifest must be a JSON object")
    if manifest.get("run_kind") != "training":
        raise EvaluationContractError(
            "training manifest run_kind does not match this evaluation"
        )

    data_section = manifest.get("data")
    model_section = manifest.get("model")
    assignment_section = manifest.get("assignment")
    evaluation_section = manifest.get("evaluation")
    eligibility_section = manifest.get("test_eligibility")
    selection = manifest.get("model_selection")
    if not all(
        isinstance(section, dict)
        for section in (
            data_section,
            model_section,
            assignment_section,
            evaluation_section,
            eligibility_section,
            selection,
        )
    ):
        raise EvaluationContractError("training manifest is missing contract sections")

    expected = {
        "run_kind": "training",
        "task": task,
        "data_sha256": sha256_file(data_path),
        "model_sha256": sha256_file(checkpoint_path),
        "labels": labels,
        "seed": seed,
        "assignment_algorithm": "sha256_first_64_bits",
        "canonicalization": "lowercase_then_keep_ascii_a_to_z",
        "fractions": {"train": 0.8, "validation": 0.1, "test": 0.1},
        "splits": split_manifest(splits),
        "source_selection": source_selection or {},
    }
    actual = {
        "run_kind": manifest.get("run_kind"),
        "task": manifest.get("task"),
        "data_sha256": data_section.get("sha256"),
        "model_sha256": model_section.get("sha256"),
        "labels": manifest.get("labels"),
        "seed": assignment_section.get("seed"),
        "assignment_algorithm": assignment_section.get("algorithm"),
        "canonicalization": assignment_section.get("canonicalization"),
        "fractions": assignment_section.get("fractions"),
        "splits": manifest.get("splits"),
        "source_selection": data_section.get("selection"),
    }
    for field_name, expected_value in expected.items():
        if actual[field_name] != expected_value:
            raise EvaluationContractError(
                f"training manifest {field_name} does not match this evaluation"
            )
    if assignment_section.get("unit") != "canonical_model_input":
        raise EvaluationContractError("training manifest uses an ineligible split unit")
    if evaluation_section.get("split") != "validation":
        raise EvaluationContractError(
            "training manifest did not reserve test for checkpoint evaluation"
        )
    validation_count = evaluation_section.get("count")
    if (
        isinstance(validation_count, bool)
        or not isinstance(validation_count, int)
        or validation_count <= 0
    ):
        raise EvaluationContractError(
            "training manifest does not record positive validation evidence"
        )
    if validation_count > len(splits.validation):
        raise EvaluationContractError(
            "training manifest validation count exceeds the validation split"
        )
    expected_validation_members = list(splits.validation[:validation_count])
    if evaluation_section.get("membership_sha256") != sha256_members(
        expected_validation_members
    ):
        raise EvaluationContractError(
            "training manifest validation membership does not match this evaluation"
        )
    if not selection.get("restored_before_save") or not isinstance(
        selection.get("best_epoch"), int
    ):
        raise EvaluationContractError(
            "training manifest does not record a restored validation checkpoint"
        )
    if eligibility_section.get("eligible") is not True:
        raise EvaluationContractError("training manifest marks checkpoint ineligible")
    return {
        "filename": manifest_path.name,
        "sha256": sha256_file(manifest_path),
    }
