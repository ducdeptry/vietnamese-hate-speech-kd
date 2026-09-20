"""
Step 4: Train the TextCNN student ALONE (no distillation yet) -- this is a
baseline to compare against later: "does knowledge distillation from the
teacher actually help, compared to just training the small model directly
on the labels?" That comparison is the paper's core claim, so we need this
number regardless.

Runs fully on CPU in well under a minute -- no GPU needed.

Usage:
    python 04_train_student.py --epochs 15 --bs 32 --lr 1e-3
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score

from data_utils import Vocab, TextCNNDataset
from model_textcnn import TextCNN

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "checkpoints" / "student_out"
LABEL_NAMES = ["CLEAN", "OFFENSIVE", "HATE"]


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(y.tolist())
    f1 = f1_score(all_labels, all_preds, average="macro")
    return f1, all_labels, all_preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--min_freq", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    train_df = pd.read_csv(DATA_DIR / "train_seg.csv")
    val_df = pd.read_csv(DATA_DIR / "val_seg.csv")
    test_df = pd.read_csv(DATA_DIR / "test_seg.csv")

    vocab = Vocab.build(train_df["text_seg"], min_freq=args.min_freq)
    vocab.save(OUT_DIR / "vocab.json")
    print(f"Vocab size: {len(vocab)}")

    train_ds = TextCNNDataset(train_df, vocab, args.max_len)
    val_ds = TextCNNDataset(val_df, vocab, args.max_len)
    test_ds = TextCNNDataset(test_df, vocab, args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.bs)
    test_loader = DataLoader(test_ds, batch_size=args.bs)

    model = TextCNN(vocab_size=len(vocab), num_classes=3, max_len=args.max_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_f1, best_state = -1, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        val_f1, _, _ = evaluate(model, val_loader, device)
        print(f"epoch {epoch:2d} | train loss {total_loss / len(train_ds):.4f} | val macro-F1 {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1, best_state = val_f1, {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    torch.save(best_state, OUT_DIR / "textcnn_student.pt")

    test_f1, test_labels, test_preds = evaluate(model, test_loader, device)
    report = classification_report(test_labels, test_preds, target_names=LABEL_NAMES, digits=4)
    print(f"\nBest val macro-F1: {best_f1:.4f}")
    print(f"Test macro-F1: {test_f1:.4f}\n")
    print(report)

    (OUT_DIR / "report.txt").write_text(
        f"Best val macro-F1: {best_f1:.4f}\nTest macro-F1: {test_f1:.4f}\n\n{report}"
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Student param count: {n_params:,}")


if __name__ == "__main__":
    main()
