"""
Shared vocabulary + dataset code for the TextCNN student model.

Unlike the teacher (PhoBERT, which comes with its own pretrained
sub-word tokenizer), the student builds its own word-level vocabulary
from scratch, from the same word-segmented training text produced in
02_preprocess.py. This keeps the two models' inputs comparable (same
word segmentation) while letting the student stay lightweight.
"""

import json
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset

PAD, UNK = "<pad>", "<unk>"


class Vocab:
    def __init__(self, token_to_id: dict):
        self.token_to_id = token_to_id
        self.id_to_token = {i: t for t, i in token_to_id.items()}

    @classmethod
    def build(cls, texts, min_freq: int = 2):
        counter = Counter()
        for text in texts:
            counter.update(text.split())
        token_to_id = {PAD: 0, UNK: 1}
        for token, freq in counter.most_common():
            if freq >= min_freq:
                token_to_id[token] = len(token_to_id)
        return cls(token_to_id)

    def encode(self, text: str, max_len: int) -> list:
        ids = [self.token_to_id.get(tok, self.token_to_id[UNK]) for tok in text.split()]
        ids = ids[:max_len]
        ids += [self.token_to_id[PAD]] * (max_len - len(ids))
        return ids

    def __len__(self):
        return len(self.token_to_id)

    def save(self, path: Path):
        Path(path).write_text(json.dumps(self.token_to_id, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path):
        return cls(json.loads(Path(path).read_text()))


class TextCNNDataset(Dataset):
    """Wraps a dataframe with columns text_seg,label into (input_ids, label) tensors."""

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
