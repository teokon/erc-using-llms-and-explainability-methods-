
### MELD — single fine-tuned (utterances) (n=2610)

**Spearman rank correlation (per-example mean)**

| | GradSHAP | LIME | Opt-base | Opt-Prime | Opt-Batch | Random |
|---|---|---|---|---|---|---|
| GradSHAP | — | +0.08 | +0.02 | +0.02 |  | -0.00 |
| LIME | +0.08 | — | +0.09 | +0.08 |  | +0.00 |
| Opt-base | +0.02 | +0.09 | — | +0.50 |  | +0.01 |
| Opt-Prime | +0.02 | +0.08 | +0.50 | — |  | +0.00 |
| Opt-Batch |  |  |  |  | — |  |
| Random | -0.00 | +0.00 | +0.01 | +0.00 |  | — |

**Top-20% token overlap (Jaccard)**

| | GradSHAP | LIME | Opt-base | Opt-Prime | Opt-Batch | Random |
|---|---|---|---|---|---|---|
| GradSHAP | — | +0.23 | +0.20 | +0.19 |  | +0.17 |
| LIME | +0.23 | — | +0.28 | +0.24 |  | +0.18 |
| Opt-base | +0.20 | +0.28 | — | +0.46 |  | +0.19 |
| Opt-Prime | +0.19 | +0.24 | +0.46 | — |  | +0.18 |
| Opt-Batch |  |  |  |  | — |  |
| Random | +0.17 | +0.18 | +0.19 | +0.18 |  | — |

### MELD — context-aware (full context) (n=2610)

**Spearman rank correlation (per-example mean)**

| | GradSHAP | LIME | Opt-base | Opt-Batch | Random |
|---|---|---|---|---|---|
| GradSHAP | — | +0.00 | +0.01 | +0.01 | -0.00 |
| LIME | +0.00 | — | +0.01 | -0.00 | +0.00 |
| Opt-base | +0.01 | +0.01 | — | -0.03 | +0.00 |
| Opt-Batch | +0.01 | -0.00 | -0.03 | — | +0.00 |
| Random | -0.00 | +0.00 | +0.00 | +0.00 | — |

**Top-20% token overlap (Jaccard)**

| | GradSHAP | LIME | Opt-base | Opt-Batch | Random |
|---|---|---|---|---|---|
| GradSHAP | — | +0.11 | +0.15 | +0.11 | +0.11 |
| LIME | +0.11 | — | +0.12 | +0.15 | +0.12 |
| Opt-base | +0.15 | +0.12 | — | +0.14 | +0.11 |
| Opt-Batch | +0.11 | +0.15 | +0.14 | — | +0.11 |
| Random | +0.11 | +0.12 | +0.11 | +0.11 | — |

### IEMOCAP — single fine-tuned (utterances) (n=1622)

**Spearman rank correlation (per-example mean)**

| | GradSHAP | LIME | Opt-base | Opt-Prime | Opt-Batch | Random |
|---|---|---|---|---|---|---|
| GradSHAP | — | +0.06 | +0.00 | +0.00 | +0.10 | +0.01 |
| LIME | +0.06 | — | +0.05 | +0.05 | -0.13 | +0.00 |
| Opt-base | +0.00 | +0.05 | — | +0.69 | +0.18 | +0.02 |
| Opt-Prime | +0.00 | +0.05 | +0.69 | — | +0.07 | +0.00 |
| Opt-Batch | +0.10 | -0.13 | +0.18 | +0.07 | — | +0.15 |
| Random | +0.01 | +0.00 | +0.02 | +0.00 | +0.15 | — |

**Top-20% token overlap (Jaccard)**

| | GradSHAP | LIME | Opt-base | Opt-Prime | Opt-Batch | Random |
|---|---|---|---|---|---|---|
| GradSHAP | — | +0.19 | +0.20 | +0.20 | +0.20 | +0.18 |
| LIME | +0.19 | — | +0.24 | +0.24 | +0.37 | +0.17 |
| Opt-base | +0.20 | +0.24 | — | +0.71 | +0.21 | +0.18 |
| Opt-Prime | +0.20 | +0.24 | +0.71 | — | +0.21 | +0.17 |
| Opt-Batch | +0.20 | +0.37 | +0.21 | +0.21 | — | +0.15 |
| Random | +0.18 | +0.17 | +0.18 | +0.17 | +0.15 | — |

### IEMOCAP — context-aware (full context) (n=1622)

**Spearman rank correlation (per-example mean)**

| | GradSHAP | LIME | Opt-base | Opt-Batch | Random |
|---|---|---|---|---|---|
| GradSHAP | — | +0.00 | +0.05 | +0.00 | -0.00 |
| LIME | +0.00 | — | +0.02 | +0.01 | +0.00 |
| Opt-base | +0.05 | +0.02 | — | -0.00 | +0.00 |
| Opt-Batch | +0.00 | +0.01 | -0.00 | — | +0.00 |
| Random | -0.00 | +0.00 | +0.00 | +0.00 | — |

**Top-20% token overlap (Jaccard)**

| | GradSHAP | LIME | Opt-base | Opt-Batch | Random |
|---|---|---|---|---|---|
| GradSHAP | — | +0.12 | +0.17 | +0.12 | +0.11 |
| LIME | +0.12 | — | +0.14 | +0.21 | +0.11 |
| Opt-base | +0.17 | +0.14 | — | +0.16 | +0.11 |
| Opt-Batch | +0.12 | +0.21 | +0.16 | — | +0.11 |
| Random | +0.11 | +0.11 | +0.11 | +0.11 | — |