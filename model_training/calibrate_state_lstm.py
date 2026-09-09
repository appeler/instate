"""Fit and evaluate temperature scaling for the state char-BiLSTM.

The model's softmax targets the record-weighted state distribution of a
surname's electoral-roll occurrences, so calibration is scored directly
against those empirical distributions: the temperature minimizes
record-weighted cross-entropy on validation names excluded from checkpoint
selection. Test scoring requires an explicit --evaluate-test flag and an
eligible training manifest; developmental runs produce calibration metrics only. The script writes ``instate_state_lstm_calibration.json``
beside the checkpoint.

Run:
    .venv/bin/python model_training/calibrate_state_lstm.py \
        --data model_training/data/instate_processed_v2.csv.gz \
        --checkpoint <path>/instate_state_lstm.safetensors
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

INSTATE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(INSTATE_ROOT))
sys.path.insert(0, str(INSTATE_ROOT / "src"))

from instate.constants import (  # noqa: E402
    GT_KEYS,
    STATE_LSTM_DROPOUT,
    STATE_LSTM_EMB,
    STATE_LSTM_HIDDEN,
    STATE_LSTM_LAYERS,
    VOCAB_SIZE,
)
from instate.nnets import StateLSTM, encode_name, pad_encoded  # noqa: E402
from model_training.evaluation_contract import (  # noqa: E402
    EvaluationContractError,
    sha256_file,
    sha256_members,
    split_surnames,
    validate_training_manifest,
)
from model_training.train_state_lstm import load_surnames  # noqa: E402


def collect_logits(
    model: torch.nn.Module, names: list[str], batch_size: int = 1024
) -> np.ndarray:
    """Return raw logits for every name, in order."""
    rows = np.zeros((len(names), len(GT_KEYS)))
    with torch.no_grad():
        for start in range(0, len(names), batch_size):
            chunk = names[start : start + batch_size]
            x, lengths = pad_encoded([encode_name(name) for name in chunk])
            rows[start : start + len(chunk)] = model(x, lengths).numpy()
    return rows


def empirical_targets(
    names: list[str], by_name: dict[str, dict[int, int]]
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-surname empirical state distributions and record weights."""
    targets = np.zeros((len(names), len(GT_KEYS)))
    weights = np.zeros(len(names))
    for row, name in enumerate(names):
        counts = by_name[name]
        weight = sum(counts.values())
        weights[row] = weight
        for state, count in counts.items():
            targets[row, state] = count / weight
    return targets, weights


def weighted_metrics(
    logits: np.ndarray, targets: np.ndarray, weights: np.ndarray, temperature: float
) -> dict[str, float]:
    """Record-weighted log loss, Brier score, and top-1 reliability."""
    scaled = logits / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    log_probabilities = scaled - np.log(np.exp(scaled).sum(axis=1, keepdims=True))
    probabilities = np.exp(log_probabilities)
    share = weights / weights.sum()
    log_loss = float(-(share * (targets * log_probabilities).sum(axis=1)).sum())
    brier = float((share * ((probabilities - targets) ** 2).sum(axis=1)).sum())
    top = probabilities.argmax(axis=1)
    confidence = float((share * probabilities.max(axis=1)).sum())
    accuracy = float((share * targets[np.arange(len(top)), top]).sum())
    return {
        "log_loss": log_loss,
        "brier": brier,
        "top1_confidence": confidence,
        "top1_mass_covered": accuracy,
        "top1_gap": confidence - accuracy,
    }


def fit_temperature(
    logits: np.ndarray, targets: np.ndarray, weights: np.ndarray
) -> float:
    """Minimize record-weighted cross-entropy over the temperature."""

    def loss(temperature: float) -> float:
        return weighted_metrics(logits, targets, weights, temperature)["log_loss"]

    grid = np.exp(np.linspace(np.log(0.25), np.log(4.0), 33))
    best = min(grid, key=loss)
    low, high = best / 1.2, best * 1.2
    for _ in range(40):
        third = (high - low) / 3
        left, right = low + third, high - third
        if loss(left) < loss(right):
            high = right
        else:
            low = left
    return float((low + high) / 2)


def main() -> None:
    """Fit the temperature and write the calibration artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--eval-n", type=int, default=0, help="Cap per split; 0 uses every member."
    )
    parser.add_argument("--out", default=None)
    parser.add_argument("--training-manifest", default=None)
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Also score test; requires a training manifest eligible for untouched-test evaluation.",
    )
    args = parser.parse_args()
    if args.eval_n < 0:
        parser.error("--eval-n must be non-negative")

    by_name = load_surnames(args.data)
    names = sorted(name for name in by_name if by_name[name] and encode_name(name))
    splits = split_surnames(names, args.seed)
    training_manifest_path = (
        Path(args.training_manifest)
        if args.training_manifest
        else Path(args.checkpoint).with_name(
            Path(args.checkpoint).name + ".training.json"
        )
    )
    try:
        validate_training_manifest(
            training_manifest_path,
            task="state",
            data_path=args.data,
            checkpoint_path=args.checkpoint,
            labels=GT_KEYS,
            splits=splits,
            seed=args.seed,
            source_selection={"max_surnames": None},
            require_untouched_test=args.evaluate_test,
        )
    except EvaluationContractError as error:
        parser.error(str(error))
    training_manifest = json.loads(training_manifest_path.read_text())
    selection = training_manifest["evaluation"]
    calibration_names = list(splits.validation[selection["count"] :])
    if not calibration_names:
        parser.error("checkpoint selection left no independent calibration names")

    model = StateLSTM(
        VOCAB_SIZE,
        len(GT_KEYS),
        STATE_LSTM_EMB,
        STATE_LSTM_HIDDEN,
        STATE_LSTM_LAYERS,
        STATE_LSTM_DROPOUT,
    )
    model.load_state_dict(load_file(args.checkpoint))
    model.eval()

    report: dict[str, dict[str, object]] = {}
    temperature = 1.0
    partitions = [("calibration", calibration_names)]
    if args.evaluate_test:
        partitions.append(("test", list(splits.test)))
    for split_name, members in partitions:
        if args.eval_n:
            members = members[: args.eval_n]
        logits = collect_logits(model, members)
        targets, weights = empirical_targets(members, by_name)
        if split_name == "calibration":
            temperature = fit_temperature(logits, targets, weights)
        report[split_name] = {
            "surnames": len(members),
            "membership_sha256": sha256_members(members),
            "records": float(weights.sum()),
            "uncalibrated": weighted_metrics(logits, targets, weights, 1.0),
            "calibrated": weighted_metrics(logits, targets, weights, temperature),
        }
        print(f"{split_name}: {json.dumps(report[split_name], indent=2)}", flush=True)

    artifact = {
        "schema_version": 2,
        "method": "temperature-scaling",
        "temperature": temperature,
        "fit_split": "calibration",
        "calibration_partition": {
            "parent_split": "validation",
            "rule": "exclude the selection-domain hash-ranked prefix used for checkpoint selection",
            "selection_count": selection["count"],
            "selection_membership_sha256": selection["membership_sha256"],
        },
        "training_manifest_sha256": sha256_file(training_manifest_path),
        "objective": "record-weighted cross-entropy against empirical state shares",
        "evaluation_unit": "surname, record-weighted",
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "data_sha256": sha256_file(args.data),
        "seed": args.seed,
        "metrics": report,
    }
    out = (
        Path(args.out)
        if args.out
        else Path(args.checkpoint).with_name("instate_state_lstm_calibration.json")
    )
    out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"temperature {temperature:.4f} -> {out}", flush=True)


if __name__ == "__main__":
    main()
