"""
Sentiment model training script.

Usage:
    python -m sentiment.train --data path/to/data.csv [--epochs 10] [--batch-size 32]

CSV format (two required columns):
    text    — the message text
    label   — integer label: 0=negative, 1=neutral, 2=positive
              OR string label: "negative"/"neg" → 0, "neutral"/"neu" → 1, "positive"/"pos" → 2

Public datasets to get started:
    - Twitter Sentiment140: https://www.kaggle.com/datasets/kazanova/sentiment140
      (binary pos/neg — map to 0 and 2, skip neutral or label manually)
    - SST-2 (Stanford): pip install datasets → load_dataset("sst2")
    - Amazon reviews: https://www.kaggle.com/datasets/bittlingmayer/amazonreviews
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Ensure the package root is on the path when run as a module
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sentiment.model import SentimentLSTM

_SENTIMENT_DIR = Path(__file__).parent
MODEL_PATH = _SENTIMENT_DIR / "sentiment_model.pt"
VOCAB_PATH = _SENTIMENT_DIR / "vocab.json"

_LABEL_MAP = {
    "0": 0, "negative": 0, "neg": 0,
    "1": 1, "neutral": 1, "neu": 1,
    "2": 2, "positive": 2, "pos": 2,
}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def _parse_label(raw: str) -> int:
    raw = str(raw).strip().lower()
    if raw not in _LABEL_MAP:
        raise ValueError(f"Unknown label '{raw}'. Use 0/1/2 or negative/neutral/positive.")
    return _LABEL_MAP[raw]


def load_csv(path: str) -> tuple[list[str], list[int]]:
    import csv
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Training data file not found: '{path}'\n\n"
            "Create a CSV file with two columns:\n"
            "  text   — the message text\n"
            "  label  — 0 (negative), 1 (neutral), or 2 (positive)\n\n"
            "Free public datasets:\n"
            "  Sentiment140 (Twitter): https://www.kaggle.com/datasets/kazanova/sentiment140\n"
            "  SST-2:  pip install datasets  →  load_dataset('sst2')\n"
        )
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["text"])
            labels.append(_parse_label(row["label"]))
    return texts, labels


def build_vocab(texts: list[str], min_freq: int = 2) -> dict[str, int]:
    counter: Counter = Counter()
    for text in texts:
        counter.update(_tokenize(text))
    # 0=<pad>, 1=<unk>
    vocab = {"<pad>": 0, "<unk>": 1}
    for word, freq in counter.most_common():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab


class SentimentDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], vocab: dict[str, int], max_len: int) -> None:
        self.data = []
        for text, label in zip(texts, labels):
            tokens = _tokenize(text)[:max_len]
            ids = [vocab.get(t, 1) for t in tokens]
            ids += [0] * (max_len - len(ids))
            self.data.append((torch.tensor(ids, dtype=torch.long), label))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


def train(args: argparse.Namespace) -> None:
    print(f"Loading data from {args.data} ...")
    texts, labels = load_csv(args.data)
    print(f"  {len(texts)} samples loaded. Label distribution: {Counter(labels)}")

    # 90/10 train/val split
    n = len(texts)
    idx = np.random.permutation(n)
    split = int(n * 0.9)
    train_idx, val_idx = idx[:split], idx[split:]

    train_texts = [texts[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    print("Building vocabulary...")
    vocab = build_vocab(train_texts, min_freq=args.min_freq)
    print(f"  Vocabulary size: {len(vocab)}")

    max_len = args.max_len
    train_ds = SentimentDataset(train_texts, train_labels, vocab, max_len)
    val_ds = SentimentDataset(val_texts, val_labels, vocab, max_len)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}...")

    model = SentimentLSTM(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, factor=0.5, min_lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_dl:
            x = x.to(device)
            y = y.clone().detach().long().to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_dl:
                x = x.to(device)
                y = y.clone().detach().long().to(device)
                preds = model(x).argmax(dim=-1)
                correct += (preds == y).sum().item()
                total += len(y)
        val_acc = correct / total
        scheduler.step(1 - val_acc)
        print(f"  Epoch {epoch}/{args.epochs}  loss={total_loss/len(train_dl):.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "vocab_size": len(vocab),
                "embed_dim": args.embed_dim,
                "hidden_dim": args.hidden_dim,
                "max_len": max_len,
            }, MODEL_PATH)
            with open(VOCAB_PATH, "w", encoding="utf-8") as f:
                json.dump(vocab, f, ensure_ascii=False)
            print(f"    -> Saved best model (val_acc={val_acc:.4f})")

    print(f"\nDone. Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Vocab  saved to: {VOCAB_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train  SentimentLSTM model")
    parser.add_argument("--data", required=True, help="Path to CSV file (columns: text, label)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--max-len", type=int, default=64, help="Max token length per message")
    parser.add_argument("--min-freq", type=int, default=2, help="Min word frequency for vocab")
    parser.add_argument("--lr", type=float, default=3e-3)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
