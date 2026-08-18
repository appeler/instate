"""Tests for the metrics reported by the model-training programs."""

import csv
import gzip
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from instate.constants import GT_KEYS, LANGUAGES, NUM_LANGUAGES
from model_training.evaluation_contract import (
    BestValidationCheckpoint,
    EvaluationContractError,
    EvaluationSplits,
    sha256_file,
    sha256_members,
    split_manifest,
    split_surnames,
    validate_test_eligibility,
    write_run_manifest,
)
from model_training.train_lang_lstm import evaluate as evaluate_language
from model_training.train_lang_lstm import load_lang_data
from model_training.train_state_lstm import evaluate as evaluate_state
from model_training.train_state_lstm import load_surnames

PROJECT_ROOT = Path(__file__).parents[1]


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
    names = [
        f"surname{chr(97 + first)}{chr(97 + second)}"
        for first in range(26)
        for second in range(26)
    ]

    first = split_surnames(names, seed=7)
    second = split_surnames(list(reversed(names)), seed=7)

    assert first == second
    assert set(first.train).isdisjoint(first.validation)
    assert set(first.train).isdisjoint(first.test)
    assert set(first.validation).isdisjoint(first.test)
    assert set(first.train) | set(first.validation) | set(first.test) == set(names)


def test_representation_equivalent_surnames_share_one_partition() -> None:
    """Spacing, punctuation, digits, and case cannot leak one encoding across splits."""
    variants = ["Patel", " pa-tel ", "pa2tel", "p a t e l"]

    splits = split_surnames(variants, seed=3)

    assigned = [*splits.train, *splits.validation, *splits.test]
    assert assigned == ["patel"]


def test_state_loader_aggregates_representation_equivalent_surnames(
    tmp_path: Path,
) -> None:
    """State targets use the same canonical key that defines split membership."""
    data = tmp_path / "state.csv.gz"
    with gzip.open(data, "wt", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["last_name", "state", "n_times"])
        writer.writerows(
            [
                ["Patel", "Delhi", 2],
                ["pa-tel", "Delhi", 3],
                ["pa2tel", "Punjab", 1],
            ]
        )

    by_name = load_surnames(data)

    assert by_name == {"patel": {GT_KEYS.index("Delhi"): 5, GT_KEYS.index("Punjab"): 1}}


def test_language_loader_aggregates_representation_equivalent_surnames(
    tmp_path: Path,
) -> None:
    """Synthetic-language targets aggregate before canonical split assignment."""
    data = tmp_path / "language.csv.gz"
    with gzip.open(data, "wt", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["last_name", *LANGUAGES])
        writer.writeheader()
        for name, hindi_mass in (("Patel", 2.0), ("pa-tel", 3.0)):
            row: dict[str, str | float] = dict.fromkeys(LANGUAGES, 0.0)
            row.update({"last_name": name, "hindi": hindi_mass})
            writer.writerow(row)

    _, targets, weights, names = load_lang_data(data)

    assert names == ["patel"]
    assert weights == [5.0]
    assert targets[0][LANGUAGES.index("hindi")] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "script",
    ["train_state_lstm.py", "train_lang_lstm.py"],
)
def test_documented_training_script_entrypoints_show_help(
    script: str, tmp_path: Path
) -> None:
    """Direct script execution resolves project imports outside the repository."""
    completed = subprocess.run(  # noqa: S603
        [sys.executable, str(PROJECT_ROOT / "model_training" / script), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--training-manifest" in completed.stdout


@pytest.mark.parametrize(
    "script",
    ["train_state_lstm.py", "train_lang_lstm.py"],
)
def test_training_script_entrypoints_reject_negative_eval_n(
    script: str, tmp_path: Path
) -> None:
    """Evaluation limits cannot use Python's negative-slice semantics."""
    completed = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(PROJECT_ROOT / "model_training" / script),
            "--data",
            str(tmp_path / "missing.csv.gz"),
            "--out",
            str(tmp_path / "model.pt"),
            "--eval-n",
            "-1",
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "--eval-n must be non-negative" in completed.stderr


def _write_cli_data(path: Path, task: str, names: list[str]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as file:
        if task == "state":
            writer = csv.writer(file)
            writer.writerow(["last_name", "state", "n_times"])
            writer.writerows((name, "Delhi", 1) for name in names)
        else:
            writer = csv.DictWriter(file, fieldnames=["last_name", *LANGUAGES])
            writer.writeheader()
            for name in names:
                row: dict[str, str | float] = dict.fromkeys(LANGUAGES, 0.0)
                row.update({"last_name": name, "hindi": 1.0})
                writer.writerow(row)


@pytest.mark.parametrize(
    ("task", "script"),
    [
        ("state", "train_state_lstm.py"),
        ("language", "train_lang_lstm.py"),
    ],
)
@pytest.mark.parametrize(
    ("names", "message"),
    [
        (["aag", "aak"], "training split is empty"),
        (["aaa", "aak"], "validation split is empty"),
    ],
)
def test_training_script_entrypoints_reject_empty_required_partitions(
    task: str, script: str, names: list[str], message: str, tmp_path: Path
) -> None:
    """A training manifest requires actual train and validation evidence."""
    data = tmp_path / f"{task}.csv.gz"
    _write_cli_data(data, task, names)

    completed = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(PROJECT_ROOT / "model_training" / script),
            "--data",
            str(data),
            "--out",
            str(tmp_path / "model.pt"),
            "--epochs",
            "1",
            "--samples-per-epoch",
            "1",
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert message in completed.stderr


@pytest.mark.parametrize(
    ("task", "script"),
    [
        ("state", "train_state_lstm.py"),
        ("language", "train_lang_lstm.py"),
    ],
)
def test_checkpoint_entrypoints_reject_empty_test_partition(
    task: str, script: str, tmp_path: Path
) -> None:
    """Untouched-test labeling requires at least one selected test surname."""
    data = tmp_path / f"{task}.csv.gz"
    _write_cli_data(data, task, ["aaa", "aag"])

    completed = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(PROJECT_ROOT / "model_training" / script),
            "--data",
            str(data),
            "--checkpoint",
            str(tmp_path / "missing.pt"),
            "--evaluation-split",
            "test",
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "test split is empty" in completed.stderr


@pytest.mark.parametrize(
    ("use_legacy_manifest", "message"),
    [
        (False, "requires an eligible training manifest"),
        (True, "run_kind does not match"),
    ],
)
def test_state_checkpoint_cli_refuses_ineligible_untouched_test_label(
    tmp_path: Path, use_legacy_manifest: bool, message: str
) -> None:
    """The checkpoint entrypoint fails closed before loading arbitrary weights."""
    data = tmp_path / "state.csv.gz"
    checkpoint = tmp_path / "random.pt"
    with gzip.open(data, "wt", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["last_name", "state", "n_times"])
        writer.writerow(["aak", "Delhi", 2])
    checkpoint.write_bytes(b"not a checkpoint")
    command = [
        sys.executable,
        str(PROJECT_ROOT / "model_training" / "train_state_lstm.py"),
        "--data",
        str(data),
        "--checkpoint",
        str(checkpoint),
        "--evaluation-split",
        "test",
    ]
    if use_legacy_manifest:
        command.extend(
            [
                "--training-manifest",
                str(PROJECT_ROOT / "model_training" / "evaluation_manifest.json"),
            ]
        )

    completed = subprocess.run(  # noqa: S603
        command,
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert message in completed.stderr


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
        run_kind="training",
        test_eligibility={"eligible": True},
        model_selection={
            "metric": "mass_top3",
            "mode": "max",
            "best_epoch": 2,
            "best_score": 0.7,
            "total_epochs": 3,
            "restored_before_save": True,
        },
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


def test_best_validation_checkpoint_restores_selected_epoch() -> None:
    """The saved state can be restored after a later epoch performs worse."""
    model = torch.nn.Linear(1, 1, bias=False)
    selector = BestValidationCheckpoint()
    with torch.no_grad():
        model.weight.fill_(1.0)
    assert selector.consider(model, 1, {"mass_top3": 0.7})
    with torch.no_grad():
        model.weight.fill_(2.0)
    assert not selector.consider(model, 2, {"mass_top3": 0.6})

    selector.restore(model)

    assert model.weight.item() == pytest.approx(1.0)
    assert selector.manifest(2)["best_epoch"] == 1


def _write_eligible_training_manifest(
    tmp_path: Path,
) -> tuple[Path, Path, Path, EvaluationSplits]:
    data = tmp_path / "data.csv"
    checkpoint = tmp_path / "model.pt"
    manifest = tmp_path / "model.pt.training.json"
    data.write_text("surname,count\npatel,2\n", encoding="utf-8")
    checkpoint.write_bytes(b"trained checkpoint")
    splits = split_surnames(["patel", "singh", "sood"], seed=0)
    write_run_manifest(
        manifest,
        task="state",
        data_path=data,
        checkpoint_path=checkpoint,
        labels=["Delhi", "Punjab"],
        splits=splits,
        evaluated_split="validation",
        evaluated_members=list(splits.validation),
        metrics={"mass_top3": 0.7},
        seed=0,
        run_kind="training",
        test_eligibility={"eligible": True},
        model_selection={
            "metric": "mass_top3",
            "mode": "max",
            "best_epoch": 2,
            "best_score": 0.7,
            "total_epochs": 3,
            "restored_before_save": True,
        },
    )
    return data, checkpoint, manifest, splits


def test_random_checkpoint_is_rejected_for_untouched_test_label(
    tmp_path: Path,
) -> None:
    """A manifest for another checkpoint cannot authorize test evaluation."""
    data, _, manifest, splits = _write_eligible_training_manifest(tmp_path)
    random_checkpoint = tmp_path / "random.pt"
    random_checkpoint.write_bytes(b"random checkpoint")

    with pytest.raises(EvaluationContractError, match="model_sha256"):
        validate_test_eligibility(
            manifest,
            task="state",
            data_path=data,
            checkpoint_path=random_checkpoint,
            labels=["Delhi", "Punjab"],
            splits=splits,
            seed=0,
        )


def test_matching_training_manifest_authorizes_untouched_test_label(
    tmp_path: Path,
) -> None:
    """A fully matching selected checkpoint passes the fail-closed contract."""
    data, checkpoint, manifest, splits = _write_eligible_training_manifest(tmp_path)

    provenance = validate_test_eligibility(
        manifest,
        task="state",
        data_path=data,
        checkpoint_path=checkpoint,
        labels=["Delhi", "Punjab"],
        splits=splits,
        seed=0,
    )

    assert provenance == {
        "filename": manifest.name,
        "sha256": sha256_file(manifest),
    }


@pytest.mark.parametrize(
    ("count", "membership", "message"),
    [
        (0, [], "positive validation evidence"),
        (1, ["not-the-evaluated-surname"], "validation membership"),
    ],
)
def test_training_manifest_requires_matching_positive_validation_evidence(
    count: int, membership: list[str], message: str, tmp_path: Path
) -> None:
    """Only a nonempty, matching validation prefix can authorize test labeling."""
    data, checkpoint, manifest, splits = _write_eligible_training_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["evaluation"]["count"] = count
    payload["evaluation"]["membership_sha256"] = sha256_members(membership)
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(EvaluationContractError, match=message):
        validate_test_eligibility(
            manifest,
            task="state",
            data_path=data,
            checkpoint_path=checkpoint,
            labels=["Delhi", "Punjab"],
            splits=splits,
            seed=0,
        )


def test_legacy_manifest_is_rejected_for_untouched_test_label(
    tmp_path: Path,
) -> None:
    """The pre-contract artifact inventory cannot authorize test evaluation."""
    data, checkpoint, _, splits = _write_eligible_training_manifest(tmp_path)
    legacy = PROJECT_ROOT / "model_training" / "evaluation_manifest.json"

    with pytest.raises(EvaluationContractError, match="run_kind"):
        validate_test_eligibility(
            legacy,
            task="state",
            data_path=data,
            checkpoint_path=checkpoint,
            labels=["Delhi", "Punjab"],
            splits=splits,
            seed=0,
        )


@pytest.mark.parametrize("field", ["data_sha256", "seed", "splits"])
def test_test_eligibility_validates_data_seed_and_membership(
    tmp_path: Path, field: str
) -> None:
    """Every split-defining input must match the eligible training run."""
    data, checkpoint, manifest, splits = _write_eligible_training_manifest(tmp_path)
    kwargs = {
        "manifest_path": manifest,
        "task": "state",
        "data_path": data,
        "checkpoint_path": checkpoint,
        "labels": ["Delhi", "Punjab"],
        "splits": splits,
        "seed": 0,
    }
    if field == "data_sha256":
        data.write_text("surname,count\npatel,3\n", encoding="utf-8")
    elif field == "seed":
        kwargs["seed"] = 1
    else:
        kwargs["splits"] = split_surnames(["patel", "singh", "sood", "roy"])

    with pytest.raises(EvaluationContractError, match=field):
        validate_test_eligibility(**kwargs)


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
