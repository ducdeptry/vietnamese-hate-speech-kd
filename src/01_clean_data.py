"""
Step 1: Load the raw Vietnamese hate-speech CSVs, check data quality, and
produce a clean, leakage-free train/val/test split.

What is "leakage" and why does it matter?
------------------------------------------
The whole point of a val/test set is to measure how well the model does on
text it has NEVER seen during training. If the exact same comment appears in
both `train` and `test`, the model can effectively "memorize" it during
training and then get it right on the test set for free -- making the
reported accuracy look better than it really is. That's data leakage, and a
paper reviewer who catches it would (rightly) reject the results.

This script:
  1. Loads train_df.csv / val_df.csv / test_df.csv
  2. Renames columns to clear names (text, label) and checks class balance
  3. Detects exact-duplicate text rows, and rows where the SAME text got
     DIFFERENT labels (a labeling conflict)
  4. Detects leakage: the same text appearing in more than one split
  5. Resolves both issues (documented below) and writes clean splits to
     data/processed/
"""

import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_NAMES = {0: "CLEAN", 1: "OFFENSIVE", 2: "HATE"}


def load(split: str) -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / f"{split}_df.csv")
    df = df.rename(columns={"cmt_col": "text", "labels": "label"})
    df["label"] = df["label"].astype(int)
    df["text"] = df["text"].astype(str).str.strip()
    df["split"] = split
    return df


def report_balance(df: pd.DataFrame, name: str) -> None:
    counts = df["label"].value_counts().sort_index()
    total = len(df)
    print(f"\n{name}: {total} rows")
    for label, count in counts.items():
        pct = 100 * count / total
        print(f"  {LABEL_NAMES[label]:>10} (label={label}): {count:5d} ({pct:5.1f}%)")


def main():
    train = load("train")
    val = load("val")
    test = load("test")

    print("=" * 60)
    print("RAW DATA — class balance per split")
    print("=" * 60)
    for name, df in [("train", train), ("val", val), ("test", test)]:
        report_balance(df, name)

    # --- 0. Measure leakage on the RAW, untouched data first (for the paper's
    #        data section -- report this number honestly before any cleanup) ---
    print("\n" + "=" * 60)
    print("RAW DATA — leakage check (same text in more than one split)")
    print("=" * 60)
    train_texts, val_texts, test_texts = (
        set(train["text"]), set(val["text"]), set(test["text"])
    )
    print(f"  train ∩ val : {len(train_texts & val_texts)} rows")
    print(f"  train ∩ test: {len(train_texts & test_texts)} rows")
    print(f"  val ∩ test  : {len(val_texts & test_texts)} rows")

    all_df = pd.concat([train, val, test], ignore_index=True)

    # --- 1. Exact duplicate (text, label) rows: keep first occurrence only ---
    before = len(all_df)
    all_df = all_df.drop_duplicates(subset=["text", "label"], keep="first")
    print(f"\nDropped {before - len(all_df)} exact duplicate (text,label) rows.")

    # --- 2. Labeling conflicts: same text, different labels across rows ---
    label_counts_per_text = all_df.groupby("text")["label"].nunique()
    conflicting_texts = label_counts_per_text[label_counts_per_text > 1].index
    conflicts = all_df[all_df["text"].isin(conflicting_texts)].sort_values("text")
    conflicts.to_csv(OUT_DIR / "removed_conflicts.csv", index=False)
    print(f"Found {len(conflicting_texts)} texts with conflicting labels "
          f"({len(conflicts)} rows) -> removing all of them (saved to "
          f"data/processed/removed_conflicts.csv for the paper's appendix).")
    all_df = all_df[~all_df["text"].isin(conflicting_texts)]

    # --- 3. Cross-split leakage: same text appearing in more than one split ---
    # Priority: keep in train > val > test, since train is allowed to be large
    # and test/val MUST stay unseen. A text seen in train is dropped from
    # val/test; a text seen in val (not train) is dropped from test.
    split_priority = {"train": 0, "val": 1, "test": 2}
    all_df["_prio"] = all_df["split"].map(split_priority)
    before = len(all_df)
    all_df = (
        all_df.sort_values("_prio")
        .drop_duplicates(subset=["text"], keep="first")
        .drop(columns="_prio")
    )
    leaked = before - len(all_df)
    print(f"Removed {leaked} rows that leaked across splits "
          f"(same text seen in an earlier-priority split).")

    print("\n" + "=" * 60)
    print("CLEAN DATA — class balance per split")
    print("=" * 60)
    for split in ["train", "val", "test"]:
        df = all_df[all_df["split"] == split]
        report_balance(df, split)
        df[["text", "label"]].to_csv(OUT_DIR / f"{split}.csv", index=False)

    print(f"\nClean splits written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
