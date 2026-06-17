"""Train the instate state-prediction char-BiLSTM on the v2 last-name data (34 states).

Replaces the legacy batch-1 GRU with a batched bidirectional LSTM (``instate.nnets.StateLSTM``).
Run in instate's own venv (has torch) so it imports the real model + constants, guaranteeing the
saved ``state_dict`` loads back into the package.

    instate/.venv/bin/python model_training/train_state_lstm.py \
        --data model_training/data/instate_processed_v2.csv.gz \
        --out instate/instate/data/instate_state_lstm.pt --epochs 8

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
from instate.nnets import StateLSTM, encode_name  # noqa: E402


def load_surnames(path, max_surnames=None):
    """Read (last_name,state,n_times) -> {last_name: {state_idx: n}} over GT_KEYS states."""
    state_idx = {s: i for i, s in enumerate(GT_KEYS)}
    by_name: dict[str, dict[int, int]] = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for ln, state, n in reader:
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
def evaluate(model, test, device, k=3, batch_size=512):
    """Top-k accuracy: is a surname's modal state among the predicted top-k?"""
    model.eval()
    correct = 0
    for i in range(0, len(test), batch_size):
        chunk = test[i : i + batch_size]
        x, lengths = pad_batch([e for e, _ in chunk], device)
        top = model(x, lengths).topk(k, dim=1).indices.tolist()
        correct += sum(modal in top[j] for j, (_, modal) in enumerate(chunk))
    return correct / max(1, len(test))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--samples-per-epoch", type=int, default=400_000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-n", type=int, default=20_000)
    ap.add_argument("--max-surnames", type=int, default=None, help="cap (smoke test)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

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
    random.shuffle(names)
    cut = int(0.8 * len(names))
    train_names, test_names = names[:cut], names[cut:]

    pool, weights = [], []
    for nm in train_names:
        for si, n in by_name[nm].items():
            pool.append((enc[nm], si))
            weights.append(n)
    test = [
        (enc[nm], max(by_name[nm], key=lambda k: by_name[nm][k])) for nm in test_names
    ]
    if args.eval_n:
        test = test[: args.eval_n]
    print(
        f"surnames {len(names):,} (train {len(train_names):,}/test {len(test_names):,}) "
        f"| pool {len(pool):,} | states {len(GT_KEYS)} | device {device}",
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
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
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
        acc = evaluate(model, test, device, k=3)
        print(
            f"epoch {epoch:2d}  loss {running / len(sample):.4f}  top3 {acc:.4f}",
            flush=True,
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
