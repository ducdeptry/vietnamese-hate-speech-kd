# DPL Project — Vietnamese Hate Speech Detection via Knowledge Distillation

## Goal
Adapt the "Distil-TextCNN" knowledge-distillation approach from the reference paper below
from Chinese offensive-speech detection to **Vietnamese hate speech detection on social media**.

## Reference paper
Fan, A.J., Sun, X.H., Liu, Y., et al. (2026). One BERT, Two TextCNNs: Exploring
Multi-Student Knowledge Distillation. *Data Intelligence*, 8, Art. 20250349.
https://doi.org/10.3724/2096-7004.di.2025.0349 (CC BY 4.0)

Original paper's idea: a large, accurate "teacher" model (BERT) trains a small, fast
"student" model (TextCNN) to mimic its behavior, so the deployed model stays nearly as
accurate as BERT while being ~70x smaller and much faster. Reproduce this recipe on
Vietnamese data as a novel contribution.

## Plan
1. **Data** — acquire and clean a labeled Vietnamese hate-speech dataset.
2. **Teacher** — fine-tune a Vietnamese pretrained language model (e.g. PhoBERT) as the
   hate-speech classifier teacher.
3. **Student** — implement the TextCNN student architecture (multi-kernel convolutions,
   k-max pooling, batch norm).
4. **Distillation** — implement the teacher→student training loop (KL divergence + cross-entropy
   loss, "student replica" stabilization trick from the paper).
5. **Evaluation** — accuracy/precision/recall/F1, model size, inference latency;
   cross-domain generalization if a second dataset is available; ablation study
   (teacher-only vs. student-only vs. distilled student).
6. **Write-up** — results tables/figures feeding into the paper draft.

## Folder structure
- `data/raw/` — original, untouched dataset files
- `data/processed/` — cleaned/tokenized data ready for training
- `src/` — all source code (data loading, models, training, evaluation)
- `notebooks/` — exploratory / experiment notebooks
- `results/checkpoints/` — saved model weights
- `results/figures/` — plots and tables for the paper
- `paper/references/` — reference paper(s) and related material

## Status
- [x] Local environment: `.venv/` (Python virtual env) with torch, transformers,
      pandas, scikit-learn, underthesea installed (`requirements.txt`).
- [x] Dataset located: `data/raw/{train,val,test}_df.csv` — Vietnamese social media
      comments, 3 classes (CLEAN=0, OFFENSIVE=1, HATE=2), ~7,380 rows total.
- [x] `src/01_clean_data.py` — confirmed a data-leakage bug in the raw split
      (85 train∩val, 173 train∩test, 31 val∩test rows — same comment text
      appearing in more than one split), removed exact duplicates and
      label conflicts, and wrote a clean, leakage-free split to
      `data/processed/{train,val,test}.csv` (4861 / 646 / 1238 rows).
- [x] `src/02_preprocess.py` — Vietnamese word segmentation (underthesea) +
      elongation collapsing → `data/processed/{split}_seg.csv`.
- [x] `src/03_train_teacher.py` — PhoBERT fine-tuning script, smoke-tested
      locally, **not yet actually run** (needs a GPU → run on
      `notebooks/colab_teacher.ipynb`, then copy `teacher_out/` back into
      `results/checkpoints/`).
- [x] `src/model_textcnn.py` + `src/data_utils.py` — TextCNN student
      architecture + vocab/dataset code.
- [x] `src/04_train_student.py` — student trained alone (no distillation),
      **real result: test macro-F1 = 0.7084** (baseline to beat).
- [x] `src/05_train_kd.py` — full distillation training (KD loss +
      student-replica trick), smoke-tested with fake teacher logits;
      **needs real teacher logits from the Colab step to produce real
      results.** ⚠️ the "replica" mechanism is our interpretation of an
      ambiguous part of the paper — see the big comment at the top of the
      file before using it in the paper's methodology section.
- [x] `src/06_evaluate.py` — final comparison table (size/F1/latency),
      auto-includes whichever models exist yet.
- [ ] **Next manual step (you): run `notebooks/colab_teacher.ipynb` on
      Google Colab**, then copy the resulting `teacher_out/` folder into
      `results/checkpoints/` here.
- [ ] Cross-domain generalization (2nd Vietnamese dataset) — not started
- [ ] Write-up

## Compute plan
No local GPU. Data prep, TextCNN student, and the distillation training loop
run locally on CPU (small/fast model). The teacher (PhoBERT fine-tuning) runs
on Google Colab's free GPU, then the resulting logits/checkpoint get pulled
back down for the local distillation step.
