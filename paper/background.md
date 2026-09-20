# Background

## Knowledge Distillation

Deploying large pretrained language models for real-time content moderation is
often impractical: models such as BERT [1] achieve high accuracy on text
classification tasks but require substantial memory and inference time, making
them poorly suited to resource-constrained or high-throughput settings such as
live social media moderation. Knowledge distillation (KD) [2] addresses this by
transferring the predictive behavior of a large, accurate "teacher" network into
a much smaller "student" network. Rather than training the student only on the
ground-truth (hard) labels, KD additionally trains it to match the teacher's
predicted class probabilities (soft labels). Because these soft probabilities
encode relative similarities between classes that a one-hot label cannot (e.g.,
a comment that is ambiguously "offensive" versus "hate speech" rather than
clearly one or the other), they provide a richer training signal than hard
labels alone, often described as "dark knowledge" [2]. The student is typically
trained with a combined objective,

  L = alpha * L_KD(student_logits, teacher_logits) + (1 - alpha) * L_CE(student_logits, y),

where L_KD is a distillation loss (commonly KL-divergence between softened
teacher and student output distributions) and L_CE is the standard
cross-entropy loss against the true label y. This work follows the
Distil-TextCNN framework of [3], which distills a fine-tuned BERT-family
teacher into a lightweight TextCNN [4] student augmented with multi-kernel
convolutions, and additionally incorporates an auxiliary, untrained "student
replica" during training to provide a supplementary learning signal. [3]
reports that the resulting student matches teacher-level accuracy on Chinese
offensive-language benchmarks while being roughly 70x smaller. We adapt this
framework to Vietnamese, where, to our knowledge, teacher-student distillation
for hate-speech detection has not been previously evaluated.

## PhoBERT

BERT-family models are pretrained on large text corpora using a masked
language modeling objective, producing contextualized representations of
words that can subsequently be fine-tuned for downstream classification tasks
with comparatively little labeled data. Because these representations are
learned from the statistical structure of the specific language they are
pretrained on, a model pretrained on English or multilingual data transfers
imperfectly to Vietnamese, which has distinct orthography, diacritics, and word
segmentation conventions (Vietnamese text is not reliably delimited into
semantic word units by whitespace alone). PhoBERT [5] addresses this by
pretraining a BERT-family model directly on a large Vietnamese corpus, using
Vietnamese-specific word segmentation as a preprocessing step. We use PhoBERT,
fine-tuned on our labeled dataset, as the teacher model in the distillation
pipeline described above, and apply the same Vietnamese word-segmentation
preprocessing (via underthesea) to text before it is passed to PhoBERT.

## References

[1] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova. "BERT: Pre-training of
Deep Bidirectional Transformers for Language Understanding." NAACL-HLT, 2019.

[2] G. Hinton, O. Vinyals, and J. Dean. "Distilling the Knowledge in a Neural
Network." NeurIPS Deep Learning and Representation Learning Workshop, 2015.

[3] A. Fan, X. Sun, Y. Liu, et al. "One BERT, Two TextCNNs: Exploring
Multi-Student Knowledge Distillation." Data Intelligence, vol. 8, Art. 20250349,
2026. https://doi.org/10.3724/2096-7004.di.2025.0349

[4] Y. Kim. "Convolutional Neural Networks for Sentence Classification." EMNLP,
2014.

[5] D. Q. Nguyen and A. T. Nguyen. "PhoBERT: Pre-trained Language Models for
Vietnamese." Findings of EMNLP, 2020.
