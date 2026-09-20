"""
The "student" model: an enhanced TextCNN, following the reference paper's
section 3.3.

Architecture (paper's description, Figure 1's "Student classifier S"):
  input token ids
    -> word embedding + position embedding (summed)
    -> several parallel 1D convolutions with different kernel sizes
       (each kernel size looks at a different "n-gram window" of words:
       a kernel size of 2 looks at word pairs, 5 looks at 5-word phrases)
    -> BatchNorm -> ReLU on each conv's output
    -> k-max pooling (k=2): keep each kernel's top-2 strongest activations
       instead of just the single max (plain "max pooling"), which keeps
       a bit more information about the text
    -> concatenate all kernels' pooled features into one vector
    -> dropout (regularization, reduces overfitting)
    -> fully-connected layer -> softmax -> class probabilities
"""

import torch
import torch.nn as nn


def k_max_pooling(x: torch.Tensor, k: int) -> torch.Tensor:
    """x: (batch, channels, seq_len) -> (batch, channels, k), keeping the
    top-k activations per channel IN THEIR ORIGINAL ORDER along the sequence."""
    topk = x.topk(k, dim=2).indices
    topk, _ = topk.sort(dim=2)
    return x.gather(2, topk)


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int = 3,
        embed_dim: int = 256,
        kernel_sizes=(2, 3, 4, 5),
        num_filters: int = 64,
        max_len: int = 128,
        k: int = 2,
        dropout: float = 0.5,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.word_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        self.k = k

        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, num_filters, kernel_size=ks, padding=ks // 2)
             for ks in kernel_sizes]
        )
        self.bns = nn.ModuleList([nn.BatchNorm1d(num_filters) for _ in kernel_sizes])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes) * k, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (batch, seq_len)
        positions = torch.arange(input_ids.size(1), device=input_ids.device)
        x = self.word_embed(input_ids) + self.pos_embed(positions).unsqueeze(0)
        x = x.transpose(1, 2)  # (batch, embed_dim, seq_len) for Conv1d

        pooled = []
        for conv, bn in zip(self.convs, self.bns):
            h = torch.relu(bn(conv(x)))          # (batch, num_filters, seq_len)
            h = k_max_pooling(h, self.k)          # (batch, num_filters, k)
            pooled.append(h.flatten(1))           # (batch, num_filters * k)

        v = torch.cat(pooled, dim=1)
        v = self.dropout(v)
        logits = self.fc(v)                       # (batch, num_classes) -- raw scores
        return logits
