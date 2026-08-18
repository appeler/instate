"""Tests for the metrics reported by the model-training programs."""

import json
from pathlib import Path

import pytest
import torch

from instate.constants import NUM_LANGUAGES
from model_training.evaluation_contract import (
    sha256_file,
    sha256_members,
    split_manifest,
    split_surnames,
    write_run_manifest,
)
from model_training.train_lang_lstm import evaluate as evaluate_language
from model_training.train_state_lstm import evaluate as evaluate_state


class FixedRankModel(torch.nn.Module):
    """Return fixed rankings selected by the first encoded character."""

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Rank 0/1/2 for token 1 and 1/0/2 for token 2."""
        logits = torch.full((len(x), NUM_LANGUAGES), -100.0)
        for row, token in enumerate(x[:, 0].tolist()):
            order = (0, 1, 2) if token == 1 else (1, 0, 2)
            for score, label in enumerate(reversed(order), start=1):
                logits[row, label] = float(score)
        return logits


def test_evaluation_split_is_stable_disjoint_and_order_independent() -> None:
    """Surname hashing fixes membership without depending on source row order."""
    names = [f"surname-{index}" for index in range(1_000)]

    first = split_surnames(names, seed=7)
    second = split_surnames(list(reversed(names)), seed=7)

    assert first == second
    assert set(first.train).isdisjoint(first.validation)
    assert set(first.train).isdisjoint(first.test)
    assert set(first.validation).isdisjoint(first.test)
    assert set(first.train) | set(first.validation) | set(first.test) == set(names)


def test_evaluation_manifest_hashes_data_and_membership(tmp_path: Path) -> None:
    """Manifest primitives detect changes to data and split membership."""
    data = tmp_path / "data.csv"
    data.write_text("surname,count\npatel,2\n", encoding="utf-8")
    splits = split_surnames(["patel", "singh", "sood"], seed=0)

    first_data_hash = sha256_file(data)
    first_splits = split_manifest(splits)
    data.write_text("surname,count\npatel,3\n", encoding="utf-8")

    assert sha256_file(data) != first_data_hash
    assert first_splits["train"]["membership_sha256"] == sha256_members(splits.train)
    assert sha256_members(["patel"]) != sha256_members(["singh"])


def test_run_manifest_binds_labels_artifacts_and_evaluated_members(
    tmp_path: Path,
) -> None:
    """A run manifest records enough identity to audit reported metrics."""
    data = tmp_path / "data.csv"
    checkpoint = tmp_path / "model.pt"
    output = tmp_path / "evaluation.json"
    data.write_text("surname,count\npatel,2\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    splits = split_surnames(["patel", "singh", "sood"], seed=0)
    evaluated = list(splits.validation)

    write_run_manifest(
        output,
        task="state",
        data_path=data,
        checkpoint_path=checkpoint,
        labels=["Delhi", "Punjab"],
        splits=splits,
        evaluated_split="validation",
        evaluated_members=evaluated,
        metrics={"modal_top1": 0.5},
        seed=0,
    )

    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["data"]["sha256"] == sha256_file(data)
    assert manifest["model"]["sha256"] == sha256_file(checkpoint)
    assert manifest["labels"] == ["Delhi", "Punjab"]
    assert manifest["evaluation"] == {
        "split": "validation",
        "count": len(evaluated),
        "membership_sha256": sha256_members(evaluated),
        "metrics": {"modal_top1": 0.5},
    }


def test_state_evaluation_reports_modal_and_distribution_metrics() -> None:
    """State evaluation distinguishes modal hits from covered voter mass."""
    test = [([1], {0: 8, 3: 2}), ([2], {2: 6, 1: 4})]

    metrics = evaluate_state(FixedRankModel(), test, "cpu", batch_size=2)

    assert metrics == pytest.approx(
        {
            "modal_top1": 0.5,
            "modal_top3": 1.0,
            "mass_top1": 0.6,
            "mass_top3": 0.9,
        }
    )


def test_language_evaluation_reports_modal_and_distribution_metrics() -> None:
    """Language evaluation weights probability coverage by surname mass."""
    first = [0.0] * NUM_LANGUAGES
    first[0], first[3] = 0.8, 0.2
    second = [0.0] * NUM_LANGUAGES
    second[1], second[2] = 0.4, 0.6

    metrics = evaluate_language(
        FixedRankModel(), [[1], [2]], [first, second], [10.0, 10.0], "cpu"
    )

    assert metrics == pytest.approx(
        {
            "modal_top1": 0.5,
            "modal_top3": 1.0,
            "mass_top1": 0.6,
            "mass_top3": 0.9,
        }
    )
