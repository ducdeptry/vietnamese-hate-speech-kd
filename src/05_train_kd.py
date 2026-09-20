"""
Step 5: Knowledge distillation (KD) training -- the paper's actual method.

*** IMPORTANT: read the "Our interpretation of the student replica trick"
*** comment below before using this for the paper's methodology section.
*** The paper's description of this part is somewhat ambiguous; this is a
*** reasonable implementation choice, not a certainty -- verify it against
*** the paper (and consider asking your advisor) before citing it as a
*** faithful reproduction.

Requires teacher logits produced by 03_train_teacher.py on Colab, copied
back into results/checkpoints/teacher_out/ (logits_train.npy, meta_train.csv).

Base KD loss (paper eq. 3-4), applied to every student(-like) network:
    soft_targets  = softmax(teacher_logits / T)
    student_probs = log_softmax(student_logits / T)
    L_KD = alpha * T^2 * KLDiv(student_probs, soft_targets) + (1 - alpha) * CE(y, student_logits)
with T (temperature/rho) = 3, alpha swept in the paper across 0..0.9.

Our interpretation of the "student replica" trick
---------------------------------------------------
The paper states a second, untrained copy S' is trained alongside the
student S, that S' also learns from the teacher, and that S' stabilizes
training before its parameters are "transferred to S". The exact update
rule isn't fully spelled out. We implement this as mutual learning
(closely related to "Deep Mutual Learning"): S and S' are two
independently-initialized TextCNNs, each trained with the KD loss above
AND an extra term pulling their predictions toward each other:

    L_S  = L_KD(S)  + beta * KL(softmax(S_logits)  || softmax(S'_logits).detach())
    L_S' = L_KD(S') + beta * KL(softmax(S'_logits) || softmax(S_logits).detach())

We keep S (not S') for the final model, since S is what's deployed. Use
--use_replica to toggle this on/off, reproducing the paper's ablation:
    Student (Ours)            = 04_train_student.py            (no KD)
    Student + KD              = this script, --use_replica false
    Student + KD (Distil-TextCNN) = this script, --use_replica true

Usage:
    python 05_train_kd.py --alpha 0.5 --use_replica
    python 05_train_kd.py --alpha_sweep          # reproduces paper Figure 2
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score

from data_utils import Vocab
from model_textcnn import TextCNN

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
TEACHER_DIR = Path(__file__).resolve().parent.parent / "results" / "checkpoints" / "teacher_out"
STUDENT_DIR = Path(__file__).resolve().parent.parent / "results" / "checkpoints" / "student_out"
OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "checkpoints" / "kd_out"
LABEL_NAMES = ["CLEAN", "OFFENSIVE", "HATE"]


class KDDataset(Dataset):
    """Like TextCNNDataset, but also returns the teacher's logits for each row."""

    def __init__(self, df, vocab: Vocab, max_len: int, teacher_logits: np.ndarray):
        self.input_ids = [vocab.encode(t, max_len) for t in df["text_seg"].astype(str)]
        self.labels = df["label"].tolist()
        self.teacher_logits = teacher_logits
        assert len(self.teacher_logits) == len(self.labels), (
            "Row-count mismatch between processed text and teacher logits -- "
            "meta_train.csv and train_seg.csv must be in the same order."
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.input_ids[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
            torch.tensor(self.teacher_logits[idx], dtype=torch.float),
        )


class PlainDataset(Dataset):
    """For val/test, where we only need student predictions, no teacher logits."""

    def __init__(self, df, vocab: Vocab, max_len: int):
        self.input_ids = [vocab.encode(t, max_len) for t in df["text_seg"].astype(str)]
        self.labels = df["label"].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.input_ids[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


def kd_loss(student_logits, teacher_logits, labels, alpha, T):
    soft_targets = F.softmax(teacher_logits / T, dim=1)
    student_log_probs = F.log_softmax(student_logits / T, dim=1)
    kd = F.kl_div(student_log_probs, soft_targets, reduction="batchmean") * (T ** 2)
    ce = F.cross_entropy(student_logits, labels)
    return alpha * kd + (1 - alpha) * ce


def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            preds.extend(logits.argmax(1).cpu().tolist())
            labels.extend(y.tolist())
    return f1_score(labels, preds, average="macro"), labels, preds


def train_one_config(train_ds, val_loader, test_loader, vocab, args, device, tag):
    torch.manual_seed(args.seed)
    student = TextCNN(vocab_size=len(vocab), num_classes=3, max_len=args.max_len).to(device)
    opt_s = torch.optim.AdamW(student.parameters(), lr=args.lr)

    replica, opt_r = None, None
    if args.use_replica:
        torch.manual_seed(args.seed + 1)  # different init from the student
        replica = TextCNN(vocab_size=len(vocab), num_classes=3, max_len=args.max_len).to(device)
        opt_r = torch.optim.AdamW(replica.parameters(), lr=args.lr)

    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True)

    best_f1, best_state = -1, None
    for epoch in range(1, args.epochs + 1):
        student.train()
        if replica is not None:
            replica.train()
        total_loss = 0.0
        for x, y, teacher_logits in train_loader:
            x, y, teacher_logits = x.to(device), y.to(device), teacher_logits.to(device)

            s_logits = student(x)
            loss_s = kd_loss(s_logits, teacher_logits, y, args.alpha, args.temperature)

            if replica is not None:
                r_logits = replica(x)
                loss_r = kd_loss(r_logits, teacher_logits, y, args.alpha, args.temperature)
                # mutual-consistency term (see module docstring)
                loss_s = loss_s + args.beta * F.kl_div(
                    F.log_softmax(s_logits, dim=1), F.softmax(r_logits.detach(), dim=1),
                    reduction="batchmean",
                )
                loss_r = loss_r + args.beta * F.kl_div(
                    F.log_softmax(r_logits, dim=1), F.softmax(s_logits.detach(), dim=1),
                    reduction="batchmean",
                )
                opt_r.zero_grad()
                loss_r.backward(retain_graph=True)
                opt_r.step()

            opt_s.zero_grad()
            loss_s.backward()
            opt_s.step()
            total_loss += loss_s.item() * x.size(0)

        val_f1, _, _ = evaluate(student, val_loader, device)
        if val_f1 > best_f1:
            best_f1, best_state = val_f1, {k: v.clone() for k, v in student.state_dict().items()}

        if args.verbose:
            print(f"  [{tag}] epoch {epoch:2d} | train loss {total_loss / len(train_ds):.4f} | val macro-F1 {val_f1:.4f}")

    student.load_state_dict(best_state)
    test_f1, test_labels, test_preds = evaluate(student, test_loader, device)
    return student, best_f1, test_f1, test_labels, test_preds


def load_teacher_logits(split):
    path = TEACHER_DIR / f"logits_{split}.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"\n{path} not found.\n\n"
            "The teacher hasn't been trained yet (that step runs on Colab, "
            "since it needs a GPU). Run 03_train_teacher.py on Colab, then "
            f"copy the whole 'teacher_out' folder back into "
            f"{TEACHER_DIR.parent}/ before running this script.\n"
        )
    return np.load(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)   # paper Table 2: student epochs = 8
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=3.0, help="rho in the paper")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--alpha_sweep", action="store_true",
                     help="sweep alpha in 0.0..0.9 step 0.1, reproducing paper Figure 2")
    ap.add_argument("--use_replica", action="store_true")
    ap.add_argument("--beta", type=float, default=0.5, help="weight of the replica consistency term")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    train_df = pd.read_csv(DATA_DIR / "train_seg.csv")
    val_df = pd.read_csv(DATA_DIR / "val_seg.csv")
    test_df = pd.read_csv(DATA_DIR / "test_seg.csv")
    train_logits = load_teacher_logits("train")

    vocab_path = STUDENT_DIR / "vocab.json"
    vocab = Vocab.load(vocab_path) if vocab_path.exists() else Vocab.build(train_df["text_seg"])

    train_ds = KDDataset(train_df, vocab, args.max_len, train_logits)
    val_loader = DataLoader(PlainDataset(val_df, vocab, args.max_len), batch_size=args.bs)
    test_loader = DataLoader(PlainDataset(test_df, vocab, args.max_len), batch_size=args.bs)

    alphas = [round(a * 0.1, 1) for a in range(10)] if args.alpha_sweep else [args.alpha]
    results = []
    for a in alphas:
        run_args = argparse.Namespace(**{**vars(args), "alpha": a})
        student, best_val_f1, test_f1, test_labels, test_preds = train_one_config(
            train_ds, val_loader, test_loader, vocab, run_args, device, tag=f"alpha={a}"
        )
        print(f"alpha={a:.1f} | val macro-F1 {best_val_f1:.4f} | test macro-F1 {test_f1:.4f}")
        results.append({"alpha": a, "val_f1": best_val_f1, "test_f1": test_f1})

        if not args.alpha_sweep:
            torch.save(student.state_dict(), OUT_DIR / "textcnn_kd.pt")
            report = classification_report(test_labels, test_preds, target_names=LABEL_NAMES, digits=4)
            print(report)
            (OUT_DIR / "report.txt").write_text(report)

    pd.DataFrame(results).to_csv(OUT_DIR / "alpha_sweep.csv", index=False)
    print(f"\nResults written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
