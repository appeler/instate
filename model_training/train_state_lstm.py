"""Train the instate state-prediction char-BiLSTM on the v2 last-name data (34 states).

Replaces the legacy batch-1 GRU with a batched bidirectional LSTM (``instate.nnets.StateLSTM``).
Run in instate's own venv (has torch) so it imports the real model + constants, guaranteeing the
saved ``state_dict`` loads back into the package.

    instate/.venv/bin/python model_training/train_state_lstm.py \
        --data model_training/data/instate_processed_v2.csv.gz \
        --out src/instate/data/instate_state_lstm.pt --epochs 8

Smoke test (tiny):
    ... --max-surnames 400 --epochs 1 --samples-per-epoch 2000 --eval-n 100
"""

import argparse
import csv
import gzip
import random
import sys
from pathlib import Path

INSTATE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(INSTATE_ROOT))
sys.path.insert(0, str(INSTATE_ROOT / "src"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from instate.constants import (  # noqa: E402
    GT_KEYS,
    STATE_LSTM_DROPOUT,
    STATE_LSTM_EMB,
    STATE_LSTM_HIDDEN,
    STATE_LSTM_LAYERS,
    VOCAB_SIZE,
)
from instate.nnets import StateLSTM, canonicalize_name, encode_name  # noqa: E402
from model_training.evaluation_contract import (  # noqa: E402
    BestValidationCheckpoint,
    EvaluationContractError,
    SplitName,
    split_surnames,
    validate_test_eligibility,
    write_run_manifest,
)


def load_surnames(path, max_surnames=None):
    """Read (last_name,state,n_times) -> {last_name: {state_idx: n}} over GT_KEYS states."""
    state_idx = {s: i for i, s in enumerate(GT_KEYS)}
    by_name: dict[str, dict[int, int]] = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for raw_name, state, n in reader:
            ln = canonicalize_name(raw_name)
            if not ln:
                continue
            if ln not in by_name:
                if max_surnames and len(by_name) >= max_surnames:
                    break
                by_name[ln] = {}
            si = state_idx.get(state)
            if si is not None:
                by_name[ln][si] = by_name[ln].get(si, 0) + int(n)
    return by_name


def pad_batch(encoded, device):
    """Pad a list of index-lists -> (padded LongTensor [B,T], lengths LongTensor [B])."""
    lengths = torch.tensor([len(e) for e in encoded], dtype=torch.long)
    maxlen = int(lengths.max())
    x = torch.zeros(len(encoded), maxlen, dtype=torch.long)  # 0 == <PAD>
    for i, e in enumerate(encoded):
        x[i, : len(e)] = torch.tensor(e, dtype=torch.long)
    return x.to(device), lengths


@torch.no_grad()
def evaluate(model, test, device, batch_size=512):
    """Report surname-level accuracy and voter-distribution coverage."""
    model.eval()
    modal_top1 = modal_top3 = 0
    mass_top1 = mass_top3 = total_mass = 0.0
    for i in range(0, len(test), batch_size):
        chunk = test[i : i + batch_size]
        x, lengths = pad_batch([e for e, _ in chunk], device)
        predictions = model(x, lengths).topk(3, dim=1).indices.tolist()
        for predicted, (_, counts) in zip(predictions, chunk, strict=True):
            modal = max(counts, key=counts.get)
            weight = sum(counts.values())
            modal_top1 += predicted[0] == modal
            modal_top3 += modal in predicted
            mass_top1 += counts.get(predicted[0], 0)
            mass_top3 += sum(counts.get(label, 0) for label in predicted)
            total_mass += weight
    n = max(1, len(test))
    total_mass = max(1.0, total_mass)
    return {
        "modal_top1": modal_top1 / n,
        "modal_top3": modal_top3 / n,
        "mass_top1": mass_top1 / total_mass,
        "mass_top3": mass_top3 / total_mass,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    destination = ap.add_mutually_exclusive_group(required=True)
    destination.add_argument("--out", help="Train and write a checkpoint.")
    destination.add_argument("--checkpoint", help="Evaluate an existing checkpoint.")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--samples-per-epoch", type=int, default=400_000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument(
        "--eval-n",
        type=int,
        default=20_000,
        help="Members from the selected evaluation split; 0 evaluates all.",
    )
    ap.add_argument(
        "--evaluation-split",
        choices=("validation", "test"),
        default=None,
        help="Default: validation while training, test for a saved checkpoint.",
    )
    ap.add_argument(
        "--manifest-out",
        default=None,
        help="Run manifest path (default derived from the checkpoint and run kind).",
    )
    ap.add_argument(
        "--training-manifest",
        default=None,
        help="Eligible training manifest required for untouched-test evaluation.",
    )
    ap.add_argument("--max-surnames", type=int, default=None, help="cap (smoke test)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.out and args.epochs < 1:
        ap.error("--epochs must be at least 1 when training")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )

    by_name = load_surnames(args.data, args.max_surnames)
    # keep only surnames with >=1 in-vocab char and >=1 state
    enc = {nm: encode_name(nm) for nm in by_name}
    names = sorted(nm for nm in by_name if by_name[nm] and enc[nm])
    splits = split_surnames(names, args.seed)
    evaluation_split: SplitName = args.evaluation_split or (
        "test" if args.checkpoint else "validation"
    )
    if args.out and evaluation_split == "test":
        ap.error("training may evaluate validation only; test is checkpoint-only")
    train_names = list(splits.train)
    evaluation_names = list(splits.members(evaluation_split))
    if args.eval_n:
        evaluation_names = evaluation_names[: args.eval_n]
    evaluation_rows = [(enc[nm], by_name[nm]) for nm in evaluation_names]
    print(
        f"surnames {len(names):,} (train {len(train_names):,}/"
        f"validation {len(splits.validation):,}/test {len(splits.test):,}) "
        f"| evaluating {evaluation_split} {len(evaluation_rows):,} "
        f"| states {len(GT_KEYS)} | device {device}",
        flush=True,
    )

    model = StateLSTM(
        VOCAB_SIZE,
        len(GT_KEYS),
        STATE_LSTM_EMB,
        STATE_LSTM_HIDDEN,
        STATE_LSTM_LAYERS,
        STATE_LSTM_DROPOUT,
    ).to(device)
    checkpoint_path = Path(args.checkpoint or args.out)
    if args.manifest_out:
        manifest_path = Path(args.manifest_out)
    elif args.out:
        manifest_path = checkpoint_path.with_name(
            checkpoint_path.name + ".training.json"
        )
    else:
        manifest_path = checkpoint_path.with_name(
            checkpoint_path.name + f".{evaluation_split}-evaluation.json"
        )
    if args.checkpoint:
        provenance: dict[str, str] = {}
        test_eligibility: dict[str, object] = {
            "eligible": False,
            "reason": "validation evaluation does not assert untouched-test status",
        }
        if evaluation_split == "test":
            training_manifest_path = (
                Path(args.training_manifest)
                if args.training_manifest
                else checkpoint_path.with_name(
                    checkpoint_path.name + ".training.json"
                )
            )
            try:
                provenance = validate_test_eligibility(
                    training_manifest_path,
                    task="state",
                    data_path=args.data,
                    checkpoint_path=checkpoint_path,
                    labels=GT_KEYS,
                    splits=splits,
                    seed=args.seed,
                    source_selection={"max_surnames": args.max_surnames},
                )
            except EvaluationContractError as error:
                ap.error(str(error))
            test_eligibility = {
                "eligible": True,
                "basis": "matching eligible training manifest",
            }
        model.load_state_dict(
            torch.load(args.checkpoint, map_location=device, weights_only=True)
        )
        metrics = evaluate(model, evaluation_rows, device)
        print(
            f"{evaluation_split} modal top1/top3 "
            f"{metrics['modal_top1']:.3f}/{metrics['modal_top3']:.3f}  "
            f"mass top1/top3 {metrics['mass_top1']:.3f}/{metrics['mass_top3']:.3f}",
            flush=True,
        )
        write_run_manifest(
            manifest_path,
            task="state",
            data_path=args.data,
            checkpoint_path=checkpoint_path,
            labels=GT_KEYS,
            splits=splits,
            evaluated_split=evaluation_split,
            evaluated_members=evaluation_names,
            metrics=metrics,
            seed=args.seed,
            run_kind="evaluation",
            test_eligibility=test_eligibility,
            source_selection={"max_surnames": args.max_surnames},
            provenance={"training_manifest_" + key: value for key, value in provenance.items()},
        )
        print(f"manifest -> {manifest_path}", flush=True)
        return

    pool, weights = [], []
    for nm in train_names:
        for si, n in by_name[nm].items():
            pool.append((enc[nm], si))
            weights.append(n)
    print(f"training pool {len(pool):,}", flush=True)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    selector = BestValidationCheckpoint()
    bs = args.batch_size

    for epoch in range(1, args.epochs + 1):
        model.train()
        sample = random.choices(pool, weights=weights, k=args.samples_per_epoch)
        running = 0.0
        for i in range(0, len(sample), bs):
            chunk = sample[i : i + bs]
            x, lengths = pad_batch([e for e, _ in chunk], device)
            targets = torch.tensor([si for _, si in chunk], device=device)
            logits = model(x, lengths)
            loss = criterion(logits, targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * len(chunk)
        metrics = evaluate(model, evaluation_rows, device)
        selected = selector.consider(model, epoch, metrics)
        print(
            f"epoch {epoch:2d}  loss {running / len(sample):.4f}  "
            f"validation modal top1/top3 "
            f"{metrics['modal_top1']:.3f}/{metrics['modal_top3']:.3f}  "
            f"mass top1/top3 {metrics['mass_top1']:.3f}/{metrics['mass_top3']:.3f}"
            f"{'  selected' if selected else ''}",
            flush=True,
        )

    selector.restore(model)
    metrics = selector.best_metrics
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    write_run_manifest(
        manifest_path,
        task="state",
        data_path=args.data,
        checkpoint_path=checkpoint_path,
        labels=GT_KEYS,
        splits=splits,
        evaluated_split="validation",
        evaluated_members=evaluation_names,
        metrics=metrics,
        seed=args.seed,
        run_kind="training",
        test_eligibility={
            "eligible": True,
            "basis": "test split unused during training and validation selection",
        },
        source_selection={"max_surnames": args.max_surnames},
        model_selection=selector.manifest(args.epochs),
    )
    print(
        f"restored validation epoch {selector.best_epoch} "
        f"({selector.metric}={selector.best_score:.3f})",
        flush=True,
    )
    print(f"saved -> {checkpoint_path}", flush=True)
    print(f"manifest -> {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
