"""
Step 2: Text preprocessing.

Turns the clean raw comments into a form suitable for both models:
  - PhoBERT (teacher) requires Vietnamese text to be WORD-SEGMENTED first
    (e.g. "Việt Nam" -> "Việt_Nam") because that's how it was pretrained.
  - The student (TextCNN) will build its own vocabulary from this same
    segmented text, so both models see identical tokenization -- keeping
    the comparison between them fair.

Design decisions (documented here so they can be justified in the paper):
  - No lowercasing: Vietnamese diacritics carry meaning, and PhoBERT was
    pretrained on cased text, so casing is preserved.
  - No profanity masking: the model's whole job is to recognize offensive
    content, so we must not hide the signal it needs to learn from.
  - Elongated character runs (e.g. "đẹppppp" -> "đẹpp") are collapsed to at
    most 2 repeats -- this keeps the "emphasis" signal (repeated letters
    often signal strong emotion) while reducing vocabulary sparsity from
    near-infinite spelling variants of the same word.
  - Word segmentation via `underthesea` (the standard tool PhoBERT itself
    was trained with).
"""

import re
import pandas as pd
from pathlib import Path
from underthesea import word_tokenize

IN_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUT_DIR = IN_DIR  # write alongside the cleaned files

ELONGATION_RE = re.compile(r"(.)\1{2,}")  # any char repeated 3+ times


def collapse_elongation(text: str) -> str:
    return ELONGATION_RE.sub(r"\1\1", text)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def preprocess_text(text: str) -> str:
    text = normalize_whitespace(text)
    text = collapse_elongation(text)
    text = word_tokenize(text, format="text")
    return text


def main():
    for split in ["train", "val", "test"]:
        df = pd.read_csv(IN_DIR / f"{split}.csv")
        df["text_seg"] = df["text"].astype(str).apply(preprocess_text)
        df.to_csv(OUT_DIR / f"{split}_seg.csv", index=False)

        lengths = df["text_seg"].str.split().apply(len)
        print(f"{split}: {len(df)} rows | tokens/row -> "
              f"mean={lengths.mean():.1f}, p50={lengths.median():.0f}, "
              f"p95={lengths.quantile(0.95):.0f}, p99={lengths.quantile(0.99):.0f}, "
              f"max={lengths.max()}")

    print(f"\nSegmented files written to {OUT_DIR}/ as *_seg.csv "
          f"(new column: text_seg)")


if __name__ == "__main__":
    main()
