# ERC revision — figures & metrics

## 01_embedding_geometry/   (reviewer note 10 — t-SNE / representation geometry)
- `model_comparison/`  — 18 standalone plots: {pretrained, single_ft, context_aware} × {pca, tsne, umap} × {meld, iemocap}
- `summary/`           — 3-model grids, 5-seed t-SNE panels, original-space metric bars, per-epoch summary curves
- `per_epoch/epoch_0..7/` — per epoch (0 = before fine-tuning): tsne/pca/umap coloured by gold emotion + correct-vs-misclassified + metrics.{json,txt}
- `metrics/`           — geometry3_*.{csv,json} (3-model table), geometry_epochs_*.{csv,json} (per-epoch: silhouette, Davies-Bouldin, Calinski-Harabasz, Dunn, k-NN probe)

## 02_faithfulness/   (reviewer note 5)
- `figures/`  — 6-metric bars vs Random + deletion/insertion curves, per dataset
- `tables/`   — faithfulness_*_{single_ft,context}_summary.csv (all explainers, all metrics) + deletion/insertion curve CSVs

## 03_agreement/   (reviewer note 7)
- agreement_*.png              — Spearman distribution, top-k overlap, cross-seed stability
- agreement_matrix.md          — full 6×6 cross-method Spearman + top-20% Jaccard matrices
- agreement_*_summary.json     — LIME↔GradSHAP agreement + per-method cross-seed stability

## 04_optimus_corpus_plots/   (the paper's Optimus figures, RoBERTa-large, full corpus)
- paper_{meld,iemocap}_{cumulative,cov10_by_class,sparsity}.png — FT vs Base, 3 Optimus variants each

NOTE: Optimus-Prime on the CONTEXT model is still computing; those numbers are not yet
in 02_faithfulness/tables (context Table F2). All other cells are final.
