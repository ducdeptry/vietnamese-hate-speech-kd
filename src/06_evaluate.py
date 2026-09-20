"""
Step 6: Final comparison table -- accuracy/F1, model size, and CPU inference
latency across whichever models have been trained so far (teacher / student
alone / student+KD). Mirrors the paper's Table 7-9 style.

Run this any time -- it automatically includes whichever checkpoints exist
in results/checkpoints/ and skips the rest, so you can run it after each
stage to see progress build up.

Note: the paper measured latency on an NVIDIA V100 GPU; this machine has no
GPU, so these latency numbers are CPU-only and NOT directly comparable to
the paper's numbers -- still valid for comparing OUR models to each other.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, classification_report

from data_utils import Vocab, TextCNNDataset
from model_textcnn import TextCNN

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
CKPT_DIR = Path(__file__).resolve().parent.parent / "results" / "checkpoints"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
LABEL_NAMES = ["CLEAN", "OFFENSIVE", "HATE"]
MAX_LEN = 128


def model_size_mb(model) -> float:
    n_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return n_bytes / (1024 ** 2)


def measure_latency_ms(predict_fn, sample, n_runs=30) -> float:
    # warm-up
    for _ in range(3):
        predict_fn(sample)
    start = time.perf_counter()
    for _ in range(n_runs):
        predict_fn(sample)
    return (time.perf_counter() - start) / n_runs * 1000


def eval_textcnn(name, ckpt_path, vocab_path, test_df):
    if not ckpt_path.exists():
        return None
    vocab = Vocab.load(vocab_path)
    model = TextCNN(vocab_size=len(vocab), num_classes=3, max_len=MAX_LEN)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()

    ds = TextCNNDataset(test_df, vocab, MAX_LEN)
    x_all = torch.stack([ds[i][0] for i in range(len(ds))])
    y_all = [ds[i][1].item() for i in range(len(ds))]

    with torch.no_grad():
        logits = model(x_all)
    preds = logits.argmax(1).tolist()
    f1 = f1_score(y_all, preds, average="macro")

    single = x_all[:1]
    with torch.no_grad():
        latency = measure_latency_ms(lambda b: model(b), single)

    return {
        "model": name,
        "params_M": sum(p.numel() for p in model.parameters()) / 1e6,
        "size_MB": round(model_size_mb(model), 2),
        "test_macro_F1": round(f1, 4),
        "latency_ms_per_sample": round(latency, 3),
    }, (y_all, preds)


def eval_teacher(test_df):
    model_dir = CKPT_DIR / "teacher_out" / "phobert_teacher"
    if not model_dir.exists():
        return None, None
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()

    enc = tokenizer(list(test_df["text_seg"]), truncation=True, max_length=MAX_LEN,
                     padding=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits
    preds = logits.argmax(1).tolist()
    y_all = test_df["label"].tolist()
    f1 = f1_score(y_all, preds, average="macro")

    single = {k: v[:1] for k, v in enc.items()}
    with torch.no_grad():
        latency = measure_latency_ms(lambda b: model(**b), single)

    return {
        "model": "Teacher (PhoBERT)",
        "params_M": sum(p.numel() for p in model.parameters()) / 1e6,
        "size_MB": round(model_size_mb(model), 2),
        "test_macro_F1": round(f1, 4),
        "latency_ms_per_sample": round(latency, 3),
    }, (y_all, preds)


def main():
    test_df = pd.read_csv(DATA_DIR / "test_seg.csv")
    rows = []

    teacher_row, _ = eval_teacher(test_df)
    if teacher_row:
        rows.append(teacher_row)
    else:
        print("(skipping teacher -- no results/checkpoints/teacher_out/phobert_teacher/ found yet)")

    student_result = eval_textcnn(
        "Student alone (no KD)",
        CKPT_DIR / "student_out" / "textcnn_student.pt",
        CKPT_DIR / "student_out" / "vocab.json",
        test_df,
    )
    if student_result:
        rows.append(student_result[0])
    else:
        print("(skipping student-alone -- run 04_train_student.py first)")

    kd_result = eval_textcnn(
        "Student + KD (Distil-TextCNN)",
        CKPT_DIR / "kd_out" / "textcnn_kd.pt",
        CKPT_DIR / "student_out" / "vocab.json",
        test_df,
    )
    if kd_result:
        rows.append(kd_result[0])
    else:
        print("(skipping student+KD -- run 05_train_kd.py first)")

    if not rows:
        print("\nNo trained models found yet -- nothing to compare.")
        return

    df = pd.DataFrame(rows)
    if teacher_row:
        bert_size = teacher_row["size_MB"]
        df["compression_x"] = (bert_size / df["size_MB"]).round(1)

    print("\n" + df.to_string(index=False))
    df.to_csv(RESULTS_DIR / "comparison_table.csv", index=False)
    print(f"\nWritten to {RESULTS_DIR / 'comparison_table.csv'}")


if __name__ == "__main__":
    main()
