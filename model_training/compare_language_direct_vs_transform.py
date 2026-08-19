"""Compare direct language estimation against the state-composition transform.

The language target is defined as census mother-tongue shares mixed by a
surname's electoral-roll state shares. Because the mixing matrix is linear,
a calibrated state model composed with the matrix targets the same quantity
as a model trained directly on the mixed labels. This experiment measures
whether direct optimization buys anything in practice: it trains a
character-BiLSTM on the mixed soft targets for training surnames, then
scores both approaches on the untouched test split with record-weighted
language log loss and Brier score.

Run:
    .venv/bin/python model_training/compare_language_direct_vs_transform.py \
        --data model_training/data/instate_processed_v2.csv.gz \
        --state-checkpoint <path>/instate_state_lstm.pt \
        --state-calibration <path>/instate_state_lstm_calibration.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


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
from instate.nnets import CharBiLSTM, StateLSTM, encode_name, pad_encoded  # noqa: E402
from model_training.evaluation_contract import split_surnames  # noqa: E402
from model_training.train_state_lstm import load_surnames  # noqa: E402

LANGUAGE_SHARES_PATH = (
    INSTATE_ROOT / "src" / "instate" / "data" / "state_language_shares.parquet"
)


def language_matrix() -> tuple[np.ndarray, list[str]]:
    """Return the state-by-language share matrix in GT_KEYS row order."""
    table = pd.read_parquet(LANGUAGE_SHARES_PATH)
    matrix = table.pivot(index="state", columns="language", values="share")
    ordered = sorted(column for column in matrix.columns if column != "other")
    matrix = matrix.loc[GT_KEYS, [*ordered, "other"]]
    return matrix.to_numpy(), list(matrix.columns)


def state_targets(
    names: list[str], by_name: dict[str, dict[int, int]]
) -> tuple[np.ndarray, np.ndarray]:
    """Per-surname empirical state distributions and record weights."""
    targets = np.zeros((len(names), len(GT_KEYS)))
    weights = np.zeros(len(names))
    for row, name in enumerate(names):
        counts = by_name[name]
        weight = sum(counts.values())
        weights[row] = weight
        for state, count in counts.items():
            targets[row, state] = count / weight
    return targets, weights


def batched_probabilities(
    model: torch.nn.Module,
    names: list[str],
    classes: int,
    temperature: float = 1.0,
    batch_size: int = 1024,
) -> np.ndarray:
    """Softmax outputs for every name, in order."""
    rows = np.zeros((len(names), classes))
    with torch.no_grad():
        for start in range(0, len(names), batch_size):
            chunk = names[start : start + batch_size]
            x, lengths = pad_encoded([encode_name(name) for name in chunk])
            logits = model(x, lengths) / temperature
            rows[start : start + len(chunk)] = torch.softmax(logits, dim=1).numpy()
    return rows


def weighted_scores(
    probabilities: np.ndarray, targets: np.ndarray, weights: np.ndarray
) -> dict[str, float]:
    """Record-weighted log loss and Brier score against soft targets."""
    share = weights / weights.sum()
    log_loss = float(
        -(share * (targets * np.log(probabilities + 1e-12)).sum(axis=1)).sum()
    )
    brier = float((share * ((probabilities - targets) ** 2).sum(axis=1)).sum())
    return {"log_loss": log_loss, "brier": brier}


def main() -> None:
    """Train the direct model and score both approaches on the test split."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--state-checkpoint", required=True)
    parser.add_argument("--state-calibration", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--samples-per-epoch", type=int, default=400_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    mixing, languages = language_matrix()
    by_name = load_surnames(args.data)
    names = sorted(name for name in by_name if by_name[name] and encode_name(name))
    splits = split_surnames(names, args.seed)
    train_names = list(splits.train)
    test_names = list(splits.test)

    test_state_targets, test_weights = state_targets(test_names, by_name)
    test_language_targets = test_state_targets @ mixing

    # Direct model: cross-entropy against mixed soft language targets, with
    # surnames sampled by record weight, mirroring the state trainer.
    direct = CharBiLSTM(VOCAB_SIZE, len(languages), 64, 256, 1, 0.0)
    optimizer = torch.optim.Adam(direct.parameters(), lr=args.lr)
    train_targets, train_weights = state_targets(train_names, by_name)
    train_language_targets = torch.tensor(
        train_targets @ mixing, dtype=torch.float32
    )
    encoded = [encode_name(name) for name in train_names]
    population = list(range(len(train_names)))
    weights_list = train_weights.tolist()

    for epoch in range(1, args.epochs + 1):
        direct.train()
        sample = random.choices(
            population, weights=weights_list, k=args.samples_per_epoch
        )
        running = 0.0
        for start in range(0, len(sample), args.batch_size):
            rows = sample[start : start + args.batch_size]
            x, lengths = pad_encoded([encoded[row] for row in rows])
            targets = train_language_targets[rows]
            log_probabilities = torch.log_softmax(direct(x, lengths), dim=1)
            loss = -(targets * log_probabilities).sum(dim=1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * len(rows)
        print(f"epoch {epoch:2d}  loss {running / len(sample):.4f}", flush=True)

    direct.eval()
    direct_scores = weighted_scores(
        batched_probabilities(direct, test_names, len(languages)),
        test_language_targets,
        test_weights,
    )

    state_model = StateLSTM(
        VOCAB_SIZE,
        len(GT_KEYS),
        STATE_LSTM_EMB,
        STATE_LSTM_HIDDEN,
        STATE_LSTM_LAYERS,
        STATE_LSTM_DROPOUT,
    )
    state_model.load_state_dict(
        torch.load(args.state_checkpoint, map_location="cpu", weights_only=True)
    )
    state_model.eval()
    calibration = json.loads(Path(args.state_calibration).read_text("utf-8"))
    state_probabilities = batched_probabilities(
        state_model, test_names, len(GT_KEYS), float(calibration["temperature"])
    )
    transform_scores = weighted_scores(
        state_probabilities @ mixing, test_language_targets, test_weights
    )

    report = {
        "test_surnames": len(test_names),
        "test_records": float(test_weights.sum()),
        "languages": len(languages),
        "direct_lstm": direct_scores,
        "state_transform": transform_scores,
    }
    print(json.dumps(report, indent=2), flush=True)
    if args.out:
        Path(args.out).write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
