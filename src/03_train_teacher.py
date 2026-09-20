"""
Step 3: Fine-tune PhoBERT as the "teacher" model.

Run this on Google Colab (needs a GPU -- this machine doesn't have one).
See colab_teacher.ipynb / the README for the exact upload + run steps.

What this produces (in results/checkpoints/teacher_out/), which the KD
(knowledge distillation) stage in step 5 needs:
  - phobert_teacher/            fine-tuned model + tokenizer
  - logits_{train,val,test}.npy raw teacher logits (BEFORE softmax) --
                                 the "soft targets" the student learns from
  - meta_{train,val,test}.csv   text_seg,label in the SAME ROW ORDER as the
                                 .npy files above (so they can be zipped
                                 together later)
  - report.txt                  precision/recall/F1 per class
  - confusion_matrix.png

Usage:
    python 03_train_teacher.py --epochs 5 --bs 32 --lr 2e-5 --max_len 128
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "checkpoints" / "teacher_out"
MODEL_NAME = "vinai/phobert-base-v2"
LABEL_NAMES = ["CLEAN", "OFFENSIVE", "HATE"]


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.enc = tokenizer(
            list(texts), truncation=True, max_length=max_len, padding=False
        )
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.enc.items()}
        item["labels"] = self.labels[idx]
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"f1_macro": f1_score(labels, preds, average="macro")}


def get_logits(trainer, dataset):
    output = trainer.predict(dataset)
    return output.predictions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke_test", action="store_true",
                     help="run on a tiny slice for 1 step, to sanity-check the code")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    splits = {}
    for name in ["train", "val", "test"]:
        df = pd.read_csv(DATA_DIR / f"{name}_seg.csv")
        if args.smoke_test:
            df = df.sample(n=min(8, len(df)), random_state=args.seed)
        splits[name] = df

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABEL_NAMES)
    )

    datasets = {
        name: TextDataset(df["text_seg"], df["label"], tokenizer, args.max_len)
        for name, df in splits.items()
    }

    training_args = TrainingArguments(
        output_dir=str(OUT_DIR / "run"),
        num_train_epochs=1 if args.smoke_test else args.epochs,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=not args.smoke_test,
        metric_for_best_model="f1_macro",
        seed=args.seed,
        report_to=[],
        logging_steps=1 if args.smoke_test else 50,
        max_steps=1 if args.smoke_test else -1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["val"],
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()

    if args.smoke_test:
        print("\nSMOKE TEST PASSED: tokenizer, model, and training loop all run.")
        return

    # Save the fine-tuned model + tokenizer
    model_dir = OUT_DIR / "phobert_teacher"
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    # Produce logits + metadata for every split (train logits are the "soft
    # targets" the student learns from in the KD stage)
    all_preds = {}
    for name, ds in datasets.items():
        logits = get_logits(trainer, ds)
        np.save(OUT_DIR / f"logits_{name}.npy", logits)
        splits[name][["text_seg", "label"]].to_csv(
            OUT_DIR / f"meta_{name}.csv", index=False
        )
        all_preds[name] = logits

    # Evaluation report on the test set
    test_labels = splits["test"]["label"].values
    test_preds = np.argmax(all_preds["test"], axis=-1)
    report = classification_report(
        test_labels, test_preds, target_names=LABEL_NAMES, digits=4
    )
    (OUT_DIR / "report.txt").write_text(report)
    print(report)

    cm = confusion_matrix(test_labels, test_preds)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(LABEL_NAMES)), LABEL_NAMES)
    ax.set_yticks(range(len(LABEL_NAMES)), LABEL_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(len(LABEL_NAMES)):
        for j in range(len(LABEL_NAMES)):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=150)

    print(f"\nAll outputs written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
