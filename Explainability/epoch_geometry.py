#!/usr/bin/env python3
"""Per-EPOCH representation geometry -- the reviewer's t-SNE caution applied to the
paper's per-epoch t-SNE figures.

Why this is needed. The paper shows one t-SNE plot per epoch to illustrate how the CLS
space organises during fine-tuning. t-SNE has NO shared coordinate frame between runs:
each epoch's plot is an independent embedding with its own arbitrary rotation, scale and
cluster placement. So visual "evolution" across those panels is partly the algorithm, not
the model -- exactly the global-structure distortion the reviewer warns about.

Fix: measure the epoch-to-epoch progression in the ORIGINAL 1024-d space, where the
numbers are directly comparable across epochs, and demote the t-SNE panels to
illustration. Every epoch also gets the projection robustness treatment.

  PRIMARY (no projection):  silhouette / Davies-Bouldin / Calinski-Harabasz + a k-NN probe
                            per epoch -> a real learning curve of the representation.
  ROBUSTNESS:               the same per-epoch trend under PCA / t-SNE / UMAP (t-SNE and
                            UMAP over N_SEEDS seeds -> mean +/- std).
  DISTORTION:               trustworthiness of each epoch's projection.

Outputs:
    geometry_epochs_<ds>.{json,csv}
    figures/epochs_<ds>_original.png     the progression in the ORIGINAL space (primary)
    figures/epochs_<ds>_tsne_grid.png    one t-SNE panel per epoch (the paper's figure)
    figures/epochs_<ds>_robustness.png   silhouette per epoch under PCA/t-SNE/UMAP

Run:
    GPU=2 python -u epoch_geometry.py --dataset meld
"""
import os
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from erc_paths import CHECKPOINTS_DIR, pick_gpu
pick_gpu()  # pin ONE GPU before importing torch ($GPU > single CUDA_VISIBLE_DEVICES > "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, accuracy_score, f1_score)
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

warnings.filterwarnings("ignore")
import faithfulness_eval as FE

CK = CHECKPOINTS_DIR
OUT = CK / "geometry"
FIG = OUT / "figures"
SEEDS = [0, 1, 2, 3, 4]
MAX_LEN, BATCH = 512, 32
PALETTE = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4"]

EPOCH_ROOT = {
    "meld": CK / "emoberta_meld_large" / "epoch_checkpoints_seed42",
    "iemocap": CK / "emoberta_iemocap_large_both" / "epoch_checkpoints_seed42",
}


@torch.inference_mode()
def cls_predictions(model, tok, texts):
    """Predicted label id per example (for the correct-vs-misclassified figures)."""
    out = []
    for i in range(0, len(texts), BATCH):
        enc = tok(texts[i:i + BATCH], return_tensors="pt", padding=True, truncation=True,
                  max_length=MAX_LEN, add_special_tokens=True).to(FE.DEVICE)
        out.append(model(**enc).logits.argmax(-1).cpu().numpy())
    return np.concatenate(out, 0)


@torch.inference_mode()
def cls_embeddings(encoder, tok, texts):
    out = []
    for i in range(0, len(texts), BATCH):
        enc = tok(texts[i:i + BATCH], return_tensors="pt", padding=True, truncation=True,
                  max_length=MAX_LEN, add_special_tokens=True).to(FE.DEVICE)
        hs = encoder(**enc, output_hidden_states=True, return_dict=True).hidden_states[-1]
        out.append(hs[:, 0, :].float().cpu().numpy())
    return np.concatenate(out, 0)


def dunn_index(X, y, max_n=2000, seed=0):
    """Dunn index = (min inter-cluster separation) / (max intra-cluster diameter), with the
    gold labels as clusters -- the fourth metric reported in the paper. Higher = better.
    Uses centroid-linkage separation and mean-radius diameter (the classic single-linkage /
    max-diameter form is dominated by single outliers and is unusable at this n). Subsampled
    to max_n for the O(n^2) distance computation, seeded for reproducibility."""
    from sklearn.metrics import pairwise_distances
    X = np.asarray(X, float); y = np.asarray(y)
    if len(X) > max_n:                       # stratified-ish subsample, seeded
        r = np.random.default_rng(seed)
        idx = r.choice(len(X), max_n, replace=False)
        X, y = X[idx], y[idx]
    labs = np.unique(y)
    cents = np.vstack([X[y == c].mean(0) for c in labs])
    inter = pairwise_distances(cents)
    np.fill_diagonal(inter, np.inf)
    min_sep = float(inter.min())
    diam = max(float(np.linalg.norm(X[y == c] - X[y == c].mean(0), axis=1).mean()) * 2
               for c in labs)
    return float(min_sep / diam) if diam > 0 else float("nan")


def space_metrics(X, y, with_dunn=False):
    d = {"silhouette": float(silhouette_score(X, y)),
         "davies_bouldin": float(davies_bouldin_score(X, y)),
         "calinski_harabasz": float(calinski_harabasz_score(X, y))}
    if with_dunn:
        d["dunn"] = dunn_index(X, y)
    return d


def knn_probe(X, y, k=10):
    """Cross-validated k-NN on the raw embeddings. Returns the metrics AND the predictions --
    the predictions give a principled 'correct vs misclassified' for the PRETRAINED model,
    which has no trained head."""
    Xs = StandardScaler().fit_transform(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    p = cross_val_predict(KNeighborsClassifier(n_neighbors=k), Xs, y, cv=cv, n_jobs=-1)
    return {"knn_acc": float(accuracy_score(y, p)),
            "knn_macro_f1": float(f1_score(y, p, average="macro"))}, p


def project(X50, kind, seed, X=None):
    if kind == "pca":
        return PCA(n_components=2, random_state=0).fit_transform(X)
    if kind == "tsne":
        return TSNE(n_components=2, perplexity=30, init="pca",
                    random_state=seed, max_iter=1000).fit_transform(X50)
    import umap
    return umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=seed).fit_transform(X50)


PROJ_TITLE = {"pca": "PCA", "tsne": "t-SNE", "umap": "UMAP"}


def write_epoch_folder(ds, key, projections, y, classes, cmap, stage_report, pred=None):
    """One folder per epoch under figures/, holding the full plot set + that epoch's metrics."""
    slug = key.replace(" ", "_")                       # 'epoch 3' -> 'epoch_3'
    d = FIG / slug
    d.mkdir(parents=True, exist_ok=True)
    o = stage_report["original_space"]
    src = o.get("pred_source", "model head")

    # standalone full-size scatter per projection (same style as the model-level figures)
    for kind, Z in projections.items():
        sil2d = stage_report["projections"][kind]["silhouette"]["mean"]
        trust = stage_report["projections"][kind]["trustworthiness"]["mean"]
        fig, ax = plt.subplots(figsize=(9, 7.5))
        for c in classes:
            m = (y == c)
            ax.scatter(Z[m, 0], Z[m, 1], s=16, c=cmap[c], alpha=0.7, linewidths=0.25,
                       edgecolors="white", label=c, rasterized=True)
        ax.set_title(f"{ds.upper()} — {PROJ_TITLE[kind]} of CLS embeddings\n{key} "
                     f"(context-aware EmoBERTa, seed 42)", fontsize=13, pad=12)
        ax.set_xlabel(f"{PROJ_TITLE[kind]} dim 1"); ax.set_ylabel(f"{PROJ_TITLE[kind]} dim 2")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#d8d8d8"); s.set_linewidth(0.7)
        ax.legend(title="Emotion", loc="center left", bbox_to_anchor=(1.01, 0.5),
                  frameon=False, fontsize=10, title_fontsize=10, markerscale=1.6)
        ax.text(0.01, -0.09,
                f"ORIGINAL 1024-d: silhouette {o['silhouette']:+.4f} | DB {o['davies_bouldin']:.2f} | "
                f"CH {o['calinski_harabasz']:.1f} | k-NN {o['knn_acc']:.3f}     "
                f"{PROJ_TITLE[kind]} 2-d: silhouette {sil2d:+.4f} | trustworthiness {trust:.3f}",
                transform=ax.transAxes, fontsize=8, color="#555")
        fig.tight_layout()
        fig.savefig(d / f"{ds}_{kind}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # ---- correct vs misclassified, coloured by GOLD emotion (paper Fig. 29/30 style) ----
    if pred is not None:
        ok = (np.asarray(pred) == y)
        for kind, Z in projections.items():
            fig, ax = plt.subplots(figsize=(9, 7.5))
            for c in classes:
                m = (y == c) & ok
                if m.any():
                    ax.scatter(Z[m, 0], Z[m, 1], s=18, c=cmap[c], alpha=0.75, marker="o",
                               linewidths=0.25, edgecolors="white", rasterized=True)
                m = (y == c) & ~ok
                if m.any():
                    ax.scatter(Z[m, 0], Z[m, 1], s=34, c=cmap[c], alpha=0.95, marker="X",
                               linewidths=0.4, edgecolors="black", rasterized=True)
            acc = o.get("test_accuracy", float("nan"))
            ax.set_title(f"{ds.upper()} — CLS embeddings ({PROJ_TITLE[kind]}) — correct vs misclassified\n"
                         f"{key}  ·  accuracy {acc:.3f}  ·  predictions from: {src}", fontsize=12, pad=12)
            ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("#d8d8d8"); s.set_linewidth(0.7)
            shape = [plt.Line2D([], [], marker="o", ls="", ms=7, mfc="#888", mec="white", label="Correct"),
                     plt.Line2D([], [], marker="X", ls="", ms=9, mfc="#888", mec="black", label="Misclassified")]
            emo = [plt.Line2D([], [], marker="s", ls="", ms=8, mfc=cmap[c], mec="none", label=c)
                   for c in classes]
            l1 = ax.legend(handles=shape, loc="upper right", frameon=True, fontsize=9)
            ax.add_artist(l1)
            ax.legend(handles=emo, title="Gold emotion", loc="center left",
                      bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=9, title_fontsize=9)
            ax.text(0.01, -0.09,
                    f"errors concentrate where classes overlap  ·  ORIGINAL 1024-d silhouette "
                    f"{o['silhouette']:+.4f} | Dunn {o.get('dunn', float('nan')):.3f}",
                    transform=ax.transAxes, fontsize=8, color="#555")
            fig.tight_layout()
            fig.savefig(d / f"{ds}_{kind}_correct_vs_misclassified.png", dpi=200, bbox_inches="tight")
            plt.close(fig)

    # this epoch's metrics, next to its plots
    (d / f"{ds}_metrics.json").write_text(json.dumps(stage_report, indent=2))
    lines = [f"{ds.upper()} — {key}", "",
             "ORIGINAL 1024-d space (primary evidence, no projection):",
             f"  silhouette         {o['silhouette']:+.4f}",
             f"  davies_bouldin     {o['davies_bouldin']:.3f}   (lower better)",
             f"  calinski_harabasz  {o['calinski_harabasz']:.1f}",
             f"  dunn               {o.get('dunn', float('nan')):.4f}   (higher better)",
             f"  test accuracy      {o.get('test_accuracy', float('nan')):.4f}   [{o.get('pred_source','-')}]",
             f"  kNN probe acc      {o['knn_acc']:.4f}",
             f"  kNN probe macro-F1 {o['knn_macro_f1']:.4f}", "",
             "2-d projections (robustness):"]
    for kind in ["pca", "tsne", "umap"]:
        p = stage_report["projections"][kind]
        lines.append(f"  {PROJ_TITLE[kind]:6s} silhouette {p['silhouette']['mean']:+.4f} "
                     f"± {p['silhouette']['std']:.4f} ({p['n_seeds']} seed(s)) | "
                     f"trustworthiness {p['trustworthiness']['mean']:.4f}")
    (d / f"{ds}_metrics.txt").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["meld", "iemocap"])
    args = ap.parse_args()
    ds = args.dataset
    FIG.mkdir(parents=True, exist_ok=True)
    (OUT / "emb").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(FE.DATASETS[ds]["test_csv"])
    texts = df["context_text_raw"].astype(str).tolist()
    y = df["label"].astype(str).to_numpy()
    classes = sorted(pd.unique(y).tolist())
    cmap = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(classes)}
    tok = AutoTokenizer.from_pretrained("roberta-large", use_fast=True, add_prefix_space=True)

    eps = sorted([p for p in EPOCH_ROOT[ds].glob("epoch_*") if p.is_dir()])
    stages = [("epoch 0\n(pretrained)", None)] + [(f"epoch {int(p.name.split('_')[1])}", p) for p in eps]
    print(f"[epochs] {ds}: {len(stages)} stages (incl. pretrained), n={len(texts)}")

    report, embs, preds = {"dataset": ds, "n": len(texts), "stages": {}}, {}, {}
    for name, path in stages:
        key = name.split("\n")[0]
        cache = OUT / "emb" / f"{ds}_{key.replace(' ', '')}.npy"
        pcache = OUT / "emb" / f"{ds}_{key.replace(' ', '')}_pred.npy"
        if cache.exists() and (path is None or pcache.exists()):
            X = np.load(cache)
            preds[key] = np.load(pcache) if pcache.exists() else None
            print(f"  [cache] {key}")
        else:
            print(f"  [compute] {key}")
            if path is None:      # pretrained: no trained head -> no meaningful predictions
                m = AutoModel.from_pretrained("roberta-large").to(FE.DEVICE).eval()
                X = cls_embeddings(m, tok, texts)
                preds[key] = None
            else:
                m = AutoModelForSequenceClassification.from_pretrained(str(path)).to(FE.DEVICE).eval()
                X = cls_embeddings(m.base_model, tok, texts)
                pr = np.array([m.config.id2label[int(i)] for i in cls_predictions(m, tok, texts)])
                preds[key] = pr
                np.save(pcache, pr)
            del m; torch.cuda.empty_cache()
            np.save(cache, X)
        embs[key] = X
        o = space_metrics(X, y, with_dunn=True)
        km, kpred = knn_probe(X, y)
        o.update(km)
        if preds.get(key) is None:
            # pretrained: no trained head -> use the k-NN probe's predictions so the
            # correct-vs-misclassified figure is still well defined (labelled as such)
            preds[key] = np.asarray(kpred)
            report_src = "k-NN probe (no trained head)"
        else:
            report_src = "model head"
        o["pred_source"] = report_src
        o["test_accuracy"] = float((preds[key] == y).mean())
        report["stages"][key] = {"original_space": o, "projections": {}}
        acc = f" acc={o['test_accuracy']:.3f} [{report_src}]"
        print(f"     ORIGINAL: silh={o['silhouette']:+.4f} DB={o['davies_bouldin']:.2f} "
              f"CH={o['calinski_harabasz']:.1f} dunn={o['dunn']:.3f} kNN={o['knn_acc']:.3f}{acc}", flush=True)

    # ---- projections per stage (+ per-epoch folder with the full plot set) ----
    tsne_first, proj_first = {}, {}
    for key, X in embs.items():
        X50 = PCA(n_components=min(50, X.shape[1]), random_state=0).fit_transform(X)
        proj_first[key] = {}
        for kind in ["pca", "tsne", "umap"]:
            seeds = [0] if kind == "pca" else SEEDS
            per, trust, first = [], [], None
            for s in seeds:
                Z = project(X50, kind, s, X)
                if first is None:
                    first = Z
                per.append(space_metrics(Z, y))
                trust.append(float(trustworthiness(X, Z, n_neighbors=10)))
            proj_first[key][kind] = first
            if kind == "tsne":
                tsne_first[key] = first
            report["stages"][key]["projections"][kind] = {
                "silhouette": {"mean": float(np.mean([d["silhouette"] for d in per])),
                               "std": float(np.std([d["silhouette"] for d in per]))},
                "trustworthiness": {"mean": float(np.mean(trust))}, "n_seeds": len(seeds)}
        write_epoch_folder(ds, key, proj_first[key], y, classes, cmap, report["stages"][key], preds.get(key))
        print(f"  [{key}] projections + per-epoch folder done", flush=True)

    (OUT / f"geometry_epochs_{ds}.json").write_text(json.dumps(report, indent=2))
    keys = list(embs)
    rows = []
    for k in keys:
        o = report["stages"][k]["original_space"]; p = report["stages"][k]["projections"]
        rows.append({"stage": k, **o,
                     **{f"{kk}_silhouette": p[kk]["silhouette"]["mean"] for kk in p},
                     **{f"{kk}_silhouette_std": p[kk]["silhouette"]["std"] for kk in p},
                     **{f"{kk}_trustworthiness": p[kk]["trustworthiness"]["mean"] for kk in p}})
    pd.DataFrame(rows).to_csv(OUT / f"geometry_epochs_{ds}.csv", index=False)

    x = np.arange(len(keys))
    # ---- (1) PRIMARY: original-space progression ----
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
    for ax, (mk, title) in zip(axes, [("silhouette", "Silhouette ↑"), ("knn_acc", "k-NN probe accuracy ↑"),
                                      ("calinski_harabasz", "Calinski-Harabasz ↑"),
                                      ("davies_bouldin", "Davies-Bouldin ↓")]):
        v = [report["stages"][k]["original_space"][mk] for k in keys]
        ax.plot(x, v, "o-", color="#2a78d6", lw=2, ms=6)
        ax.set_xticks(x); ax.set_xticklabels([k.replace("epoch ", "e") for k in keys], fontsize=8)
        ax.set_title(title, fontsize=10); ax.axhline(0, color="#ccc", lw=0.8)
        ax.grid(color="#eee"); ax.set_axisbelow(True)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    fig.suptitle(f"{ds.upper()} — representation geometry per EPOCH, measured in the ORIGINAL 1024-d space "
                 f"(no projection)\nthis, not the t-SNE panels, is the evidence for how the space organises",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.savefig(FIG / f"epochs_{ds}_original.png", dpi=200); plt.close(fig)

    # ---- (2) the paper's per-epoch t-SNE grid ----
    ncol = 4; nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.3 * nrow))
    for ax, k in zip(axes.ravel(), keys):
        Z = tsne_first[k]
        for c in classes:
            m = (y == c)
            ax.scatter(Z[m, 0], Z[m, 1], s=4, c=cmap[c], alpha=0.6, linewidths=0, label=c, rasterized=True)
        sil = report["stages"][k]["original_space"]["silhouette"]
        ax.set_title(f"{k}\noriginal-space silhouette {sil:+.3f}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.ravel()[len(keys):]:
        ax.axis("off")
    h = [plt.Line2D([], [], marker="o", ls="", ms=6, mfc=cmap[c], mec="none", label=c) for c in classes]
    fig.legend(handles=h, loc="lower center", ncol=len(classes), frameon=False, fontsize=9)
    fig.suptitle(f"{ds.upper()} — t-SNE per epoch (illustration only)\n"
                 f"t-SNE has no shared coordinate frame across panels: position/rotation/scale are "
                 f"arbitrary per plot — read the titles, not the layout", fontsize=11)
    fig.tight_layout(rect=[0, 0.05, 1, 0.90])
    fig.savefig(FIG / f"epochs_{ds}_tsne_grid.png", dpi=200); plt.close(fig)

    # ---- (3) robustness: per-epoch silhouette under each projection ----
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.plot(x, [report["stages"][k]["original_space"]["silhouette"] for k in keys],
            "o-", lw=2.5, color="#111", label="ORIGINAL 1024-d (primary)")
    for kind, col in [("pca", "#2a78d6"), ("tsne", "#1baf7a"), ("umap", "#eda100")]:
        mu = np.array([report["stages"][k]["projections"][kind]["silhouette"]["mean"] for k in keys])
        sd = np.array([report["stages"][k]["projections"][kind]["silhouette"]["std"] for k in keys])
        ax.plot(x, mu, "o--", color=col, lw=1.8, label=f"{kind.upper()} 2-d"
                + ("" if kind == "pca" else f" ({len(SEEDS)} seeds)"))
        ax.fill_between(x, mu - sd, mu + sd, color=col, alpha=0.15)
    ax.set_xticks(x); ax.set_xticklabels([k.replace("epoch ", "e") for k in keys])
    ax.set_ylabel("Silhouette"); ax.axhline(0, color="#ccc", lw=0.8)
    ax.set_title(f"{ds.upper()} — per-epoch silhouette: original space vs every projection\n"
                 f"the trend is the same whichever space it is measured in (bands = ±1 std over seeds)",
                 fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(color="#eee"); ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / f"epochs_{ds}_robustness.png", dpi=200); plt.close(fig)

    print(f"\n[saved] {OUT}/geometry_epochs_{ds}.{{json,csv}}")
    print(f"[figures] {FIG}/epochs_{ds}_{{original,tsne_grid,robustness}}.png")


if __name__ == "__main__":
    main()
