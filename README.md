# Emotion Recognition in Conversations (ERC) using LLMs & Explainability

This repository contains experiments for **Emotion Recognition in Conversations (ERC)** with **Transformer encoders** and a **multi-level explainability pipeline**.  
We benchmark **single-utterance** fine-tuned models (BERT / DistilBERT / RoBERTa) on **MELD** and **IEMOCAP**, and analyze their behavior with **local**, **global**, and **representation-level** explainability methods.

> **Why your README was “crossed out” on GitHub:** the token `<s>` is interpreted as an HTML *strikethrough* tag.  
> In this README, RoBERTa special tokens are written as code: `<s>` → `` `<s>` `` and `</s>` → `` `</s>` `` to avoid HTML rendering.

---

## Contents

### Models
- **Single-utterance baselines:** DistilBERT-base, BERT-base, RoBERTa-base (full fine-tuning).
- **(Optional) Context-aware variant:** EmoBERTa-style input construction (speaker + dialogue context) for RoBERTa.

### Datasets
- **MELD** (train/val/test in CSV format)
- **IEMOCAP** (train/val/test in CSV format)

### Evaluation
- Primary metric: **Weighted F1** (robust under class imbalance).
- Diagnostics: **confusion matrices** (and per-epoch tracking where applicable).
- Reporting: results averaged over **5 random seeds** (mean).

### Explainability
- **Utterance-level (local):** LIME, (Kernel)SHAP/SHAP-style, GradSHAP.
- **Corpus-level (global):** aggregated token importance per emotion.
- **Layer-wise diagnostics:** LIG + LGXA relevance profiles; Logit Lens trajectories.
- **Representation geometry:** `[CLS]` / embedding t-SNE visualizations + clustering metrics.

---

## Repository structure

```text
datasets/
├── meld/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
└── iemocap/
    ├── train.csv
    ├── val.csv
    └── test.csv

Explainability/
├── Utterance_explanation_github_clean.ipynb
├── Corpus_level_explanation_github_clean.ipynb
├── Optimus_Emoberta_Utterance_Level_github_clean_headers_no_gini.ipynb
└── Optimus_Global_analysis_github_clean_no_gini.ipynb

Models/
├── Emoberta_meld/
├── Emoberta_iemocap.ipynb
├── fine_tuned_meld_distilbert_bert_roberta.ipynb
└── fine_tuned_iemocap_distilbert_bert_roberta.ipynb

README.md
```

---

## About the dataset CSVs

All datasets are stored under `datasets/` in **`.csv`** format:
- `datasets/meld/` contains the **train/val/test** splits for MELD.
- `datasets/iemocap/` contains the **train/val/test** splits for IEMOCAP.

> If you add additional preprocessing outputs (e.g., constructed context inputs), place them under a separate folder such as `datasets/derived/` to keep raw splits intact.

---

## Setup

Create an environment (recommended) and install dependencies:

```bash
pip install -U pip
pip install torch transformers pandas numpy scikit-learn matplotlib shap lime captum
```

Notes:
- Several notebooks were developed in **Google Colab**. If you see Drive paths (e.g., `/content/drive/...`), replace them with your local paths.
- A GPU is recommended for training and for some explainability runs.

---

## How to run (quick guide)

### 1) Fine-tune single-utterance baselines
Use the notebooks under `Models/`:
- `fine_tuned_meld_distilbert_bert_roberta.ipynb`
- `fine_tuned_iemocap_distilbert_bert_roberta.ipynb`

These notebooks train models on the **target utterance only** and evaluate with **Weighted F1** on the test split (mean over 5 seeds).

### 2) Explainability (local + global)
Use the notebooks under `Explainability/`:

- **Utterance-level explanations**  
  `Explainability/Utterance_explanation_github_clean.ipynb`  
  Generates local explanations (LIME / SHAP / GradSHAP) for selected examples.

- **Corpus-level explanations**  
  `Explainability/Corpus_level_explanation_github_clean.ipynb`  
  Aggregates token importance across the test set per emotion.

- **Optimus analyses**  
  - `Explainability/Optimus_Emoberta_Utterance_Level_github_clean_headers_no_gini.ipynb` (utterance-level)  
  - `Explainability/Optimus_Global_analysis_github_clean_no_gini.ipynb` (global statistics)  
  Produces cumulative contribution curves and coverage-style diagnostics (Baseline A).

---

## Adding plots to the README (recommended approach)

Yes, you can include plots in the README—just keep it **selective**.

A practical layout is:
- Put all generated figures in a `results/` folder (tracked in Git).
- In the README, show only a few representative plots and link the rest.

Example (after adding `results/`):
```md
![Confusion matrix (MELD)](results/meld_confusion_matrix.png)
![t-SNE (IEMOCAP)](results/iemocap_tsne_gold.png)
```

Suggested “minimal set” for the README:
- 1 confusion matrix (best epoch / final checkpoint)
- 1 t-SNE (gold labels) + 1 t-SNE (correct vs misclassified)
- 1 global token-importance plot (GradSHAP/KernelSHAP)
- 1 Optimus cumulative curve comparison

---

## Reproducibility notes

- Reported values are **Weighted F1 on the test split**.
- Training is stochastic → report **mean over 5 random seeds**.
- Keep the dataset splits fixed for fair comparisons.

---

## Acknowledgements

- HuggingFace Transformers & PyTorch
- Captum (Integrated Gradients family)
- SHAP / LIME
- Optimus (attention-derived interpretability utilities)

