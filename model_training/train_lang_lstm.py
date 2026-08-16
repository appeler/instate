"""Train the instate language-prediction char-BiLSTM on the v2-derived language table.

Replaces the legacy 3-head ``LanguagePredictor`` with the shared ``CharBiLSTM`` (single softmax
over 37 languages). Trains on ``lang_props_v2.csv.gz`` (built by ``name_tables.py lang-prop`` =
surname-state footprint x state->official-language weights). Run in instate's venv.

    instate/.venv/bin/python model_training/train_lang_lstm.py \
        --data model_training/data/lang_props_v2.csv.gz \
        --out src/instate/data/instate_lang_lstm.pt --epochs 10

Smoke test:
    ... --max-rows 4000 --epochs 1 --samples-per-epoch 2000 --eval-n 200
"""

import argparse
import gzip
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

INSTATE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(INSTATE_ROOT / "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from instate.constants import (  # noqa: E402
    LANG_LSTM_DROPOUT,
    LANG_LSTM_EMB,
    LANG_LSTM_HIDDEN,
    LANG_LSTM_LAYERS,
    LANGUAGES,
    NUM_LANGUAGES,
    VOCAB_SIZE,
)
from instate.nnets import CharBiLSTM, encode_name, pad_encoded  # noqa: E402


def load_lang_data(path, max_rows=None):
    """Read (last_name + 37 lang scores) -> encoded names, normalized targets, weights."""
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        df = pd.read_csv(fh, nrows=max_rows)
    scores = df[list(LANGUAGES)].to_numpy(dtype="float64")
    rowsum = scores.sum(axis=1)
    keep = rowsum > 0
    df, scores, rowsum = df[keep], scores[keep], rowsum[keep]
    targets = scores / rowsum[:, None]
    enc, tgt, wt = [], [], []
    for name, t, w in zip(df["last_name"].astype(str), targets, rowsum, strict=True):
        e = encode_name(name)
        if e:
            enc.append(e)
            tgt.append(t)
            wt.append(float(w))
    return enc, tgt, wt


@torch.no_grad()
def evaluate(model, enc, tgt, wt, device):
    """Report surname-level accuracy and language-distribution coverage."""
    model.eval()
    gold = [int(max(range(NUM_LANGUAGES), key=lambda j: t[j])) for t in tgt]
    modal_top1 = modal_top3 = 0
    mass_top1 = mass_top3 = total_mass = 0.0
    for i in range(0, len(enc), 512):
        x, lengths = pad_encoded(enc[i : i + 512])
        top = model(x.to(device), lengths).topk(3, dim=1).indices.tolist()
        for j, predicted in enumerate(top):
            target = tgt[i + j]
            weight = wt[i + j]
            modal = gold[i + j]
            modal_top1 += predicted[0] == modal
            modal_top3 += modal in predicted
            mass_top1 += weight * target[predicted[0]]
            mass_top3 += weight * sum(target[label] for label in predicted)
            total_mass += weight
    n = max(1, len(enc))
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
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--samples-per-epoch", type=int, default=400_000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument(
        "--eval-n", type=int, default=20_000, help="Held-out rows; 0 evaluates all."
    )
    ap.add_argument("--max-rows", type=int, default=None, help="cap (smoke test)")
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

    enc, tgt, wt = load_lang_data(args.data, args.max_rows)
    idx = list(range(len(enc)))
    random.shuffle(idx)
    cut = int(0.8 * len(idx))
    train_idx, test_idx = idx[:cut], idx[cut:]
    if args.eval_n:
        test_idx = test_idx[: args.eval_n]
    te = [enc[i] for i in test_idx]
    tt = [tgt[i] for i in test_idx]
    tw = [wt[i] for i in test_idx]
    print(
        f"surnames {len(enc):,} (train {len(train_idx):,}/test {len(test_idx):,}) "
        f"| langs {NUM_LANGUAGES} | device {device}",
        flush=True,
    )

    model = CharBiLSTM(
        VOCAB_SIZE,
        NUM_LANGUAGES,
        LANG_LSTM_EMB,
        LANG_LSTM_HIDDEN,
        LANG_LSTM_LAYERS,
        LANG_LSTM_DROPOUT,
    ).to(device)
    if args.checkpoint:
        model.load_state_dict(
            torch.load(args.checkpoint, map_location=device, weights_only=True)
        )
        metrics = evaluate(model, te, tt, tw, device)
        print(
            f"modal top1/top3 {metrics['modal_top1']:.3f}/{metrics['modal_top3']:.3f}  "
            f"mass top1/top3 {metrics['mass_top1']:.3f}/{metrics['mass_top3']:.3f}",
            flush=True,
        )
        return

    train_w = [wt[i] for i in train_idx]
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    bs = args.batch_size

    for epoch in range(1, args.epochs + 1):
        model.train()
        sample = random.choices(train_idx, weights=train_w, k=args.samples_per_epoch)
        running = 0.0
        for i in range(0, len(sample), bs):
            chunk = sample[i : i + bs]
            x, lengths = pad_encoded([enc[j] for j in chunk])
            target = torch.from_numpy(np.stack([tgt[j] for j in chunk])).float()
            logits = model(x.to(device), lengths)
            loss = -(target.to(device) * F.log_softmax(logits, dim=1)).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * len(chunk)
        metrics = evaluate(model, te, tt, tw, device)
        print(
            f"epoch {epoch:2d}  loss {running / len(sample):.4f}  "
            f"modal top1/top3 {metrics['modal_top1']:.3f}/{metrics['modal_top3']:.3f}  "
            f"mass top1/top3 {metrics['mass_top1']:.3f}/{metrics['mass_top3']:.3f}",
            flush=True,
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
