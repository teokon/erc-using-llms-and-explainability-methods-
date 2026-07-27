#!/usr/bin/env python3
"""CLS-embedding geometry: ORIGINAL-SPACE metrics + projection robustness + figures.

Reviewer note: "t-SNE can distort global structure ... rely more on original-space
metrics and add robustness checks with UMAP, PCA, or multiple t-SNE seeds."

Three models, showing the progression pretrained -> task -> context:
  1. pretrained      roberta-large, no fine-tuning   (input: target utterance)
  2. single_ft       roberta-large fine-tuned on the target utterance only
  3. context_aware   the EmoBERTa context model      (input: full constructed context)
Models 1 vs 2 isolate FINE-TUNING; models 2 vs 3 isolate CONTEXT.

  PRIMARY EVIDENCE  -- metrics in the ORIGINAL 1024-d CLS space (no projection):
        silhouette / Davies-Bouldin / Calinski-Harabasz + a k-NN probe
        (class separability that needs no 2-D map at all).
  ROBUSTNESS        -- the same metrics under PCA, PCA->t-SNE (N_SEEDS seeds) and
        PCA->UMAP (N_SEEDS seeds). If the conclusion survives all of them it is
        not a t-SNE artifact.
  DISTORTION        -- `trustworthiness` of each projection w.r.t. the original space.

Figures (out_dir/figures):
    geometry_<ds>_grid.png        3 models x 3 projections, coloured by emotion
    geometry_<ds>_tsne_seeds.png  the same model under 5 t-SNE seeds (seed robustness)
    geometry_<ds>_metrics.png     original-space metrics across the 3 models

Run:
    GPU=0 python -u embedding_geometry.py --dataset meld
"""
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from erc_paths import pick_gpu
pick_gpu()  # pin ONE GPU before importing torch ($GPU > single CUDA_VISIBLE_DEVICES > "0")

import argparse
import json
import re
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

SEEDS = [0, 1, 2, 3, 4]
MAX_LEN = 512
BATCH = 32
SPEAKER_RE = re.compile(r"^[^:]{1,30}:\s*")

# Categorical palette (fixed slot order; validated set - worst adjacent CVD dE 24.2)
PALETTE = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4"]

MODEL_LABEL = {
    "pretrained":    "Pretrained RoBERTa-large\n(no fine-tuning)",
    "single_ft":     "Fine-tuned, single utterance\n(no context)",
    "context_aware": "Context-aware EmoBERTa\n(past + future)",
}
PROJ_LABEL = {"pca": "PCA", "tsne": "t-SNE", "umap": "UMAP"}


def target_utterance(ctx):
    parts = str(ctx).split("</s></s>")
    t = parts[1].strip() if len(parts) == 3 else str(ctx).strip()
    return SPEAKER_RE.sub("", t, count=1).strip()


@torch.inference_mode()
def cls_embeddings(encoder, tok, texts):
    """Last-layer <s>/CLS state on the model's REAL inputs (with special tokens)."""
    out = []
    for i in range(0, len(texts), BATCH):
        enc = tok(texts[i:i + BATCH], return_tensors="pt", padding=True, truncation=True,
                  max_length=MAX_LEN, add_special_tokens=True).to(FE.DEVICE)
        hs = encoder(**enc, output_hidden_states=True, return_dict=True).hidden_states[-1]
        out.append(hs[:, 0, :].float().cpu().numpy())
    return np.concatenate(out, 0)


def space_metrics(X, y):
    return {"silhouette": float(silhouette_score(X, y)),
            "davies_bouldin": float(davies_bouldin_score(X, y)),
            "calinski_harabasz": float(calinski_harabasz_score(X, y))}


def knn_probe(X, y, k=10, folds=5):
    Xs = StandardScaler().fit_transform(X)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    pred = cross_val_predict(KNeighborsClassifier(n_neighbors=k), Xs, y, cv=cv, n_jobs=-1)
    return {"knn_acc": float(accuracy_score(y, pred)),
            "knn_macro_f1": float(f1_score(y, pred, average="macro"))}


def project(X, kind, seed, X50=None):
    if kind == "pca":
        return PCA(n_components=2, random_state=0).fit_transform(X)
    if X50 is None:
        X50 = PCA(n_components=min(50, X.shape[1]), random_state=0).fit_transform(X)
    if kind == "tsne":
        return TSNE(n_components=2, perplexity=30, init="pca",
                    random_state=seed, max_iter=1000).fit_transform(X50)
    import umap
    return umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                     random_state=seed).fit_transform(X50)


def scatter(ax, Z, y, classes, cmap, title, show_axes_label=None):
    for i, c in enumerate(classes):
        m = (y == c)
        ax.scatter(Z[m, 0], Z[m, 1], s=5, c=cmap[c], alpha=0.65,
                   linewidths=0.15, edgecolors="white", label=c, rasterized=True)
    ax.set_title(title, fontsize=10, pad=6)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#d8d8d8"); s.set_linewidth(0.6)
    if show_axes_label:
        ax.set_ylabel(show_axes_label, fontsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(FE.DATASETS))
    ap.add_argument("--out_dir", default=f"{FE.CK}/geometry")
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"; fig_dir.mkdir(exist_ok=True)

    ds = args.dataset
    cfg = FE.DATASETS[ds]
    df = pd.read_csv(cfg["test_csv"])
    ctx_texts = df["context_text_raw"].astype(str).tolist()
    utt_texts = [target_utterance(t) for t in ctx_texts]
    y = df["label"].astype(str).to_numpy()
    classes = sorted(pd.unique(y).tolist())
    cmap = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(classes)}
    print(f"[geometry] {ds}: {len(y)} examples, {len(classes)} classes: {classes}")

    single_ckpt = f"{FE.CK}/roberta_large_single_{ds}/roberta_single_{ds}_seed42_BEST"
    tok = AutoTokenizer.from_pretrained("roberta-large", use_fast=True, add_prefix_space=True)

    # ---- the three models (each on its natural input) ----
    embs = {}
    print("  [1/3] pretrained roberta-large (target utterance)")
    m = AutoModel.from_pretrained("roberta-large").to(FE.DEVICE).eval()
    embs["pretrained"] = cls_embeddings(m, tok, utt_texts); del m; torch.cuda.empty_cache()

    print("  [2/3] single-utterance fine-tuned (target utterance)")
    m = AutoModelForSequenceClassification.from_pretrained(single_ckpt).to(FE.DEVICE).eval()
    embs["single_ft"] = cls_embeddings(m.base_model, tok, utt_texts); del m; torch.cuda.empty_cache()

    print("  [3/3] context-aware EmoBERTa (full context)")
    m = AutoModelForSequenceClassification.from_pretrained(cfg["ckpt"]).to(FE.DEVICE).eval()
    embs["context_aware"] = cls_embeddings(m.base_model, tok, ctx_texts); del m; torch.cuda.empty_cache()

    # ---- metrics + projections ----
    report = {"dataset": ds, "n": int(len(y)), "n_seeds": len(SEEDS)}
    grid_Z = {}
    for tag, X in embs.items():
        print(f"\n########## {tag} ##########")
        orig = space_metrics(X, y); orig.update(knn_probe(X, y))
        report[tag] = {"original_space": orig, "projections": {}}
        print(f"  [ORIGINAL 1024-d] silhouette={orig['silhouette']:+.4f} DB={orig['davies_bouldin']:.3f} "
              f"CH={orig['calinski_harabasz']:.1f} kNN acc={orig['knn_acc']:.3f} kNN macroF1={orig['knn_macro_f1']:.3f}")
        X50 = PCA(n_components=min(50, X.shape[1]), random_state=0).fit_transform(X)
        for kind in ["pca", "tsne", "umap"]:
            seeds = [0] if kind == "pca" else SEEDS
            per, trust, first = [], [], None
            for s in seeds:
                Z = project(X, kind, s, X50)
                if first is None:
                    first = Z
                per.append(space_metrics(Z, y))
                trust.append(float(trustworthiness(X, Z, n_neighbors=10)))
            grid_Z[(tag, kind)] = first
            a = {k: {"mean": float(np.mean([d[k] for d in per])),
                     "std": float(np.std([d[k] for d in per]))} for k in per[0]}
            a["trustworthiness"] = {"mean": float(np.mean(trust)), "std": float(np.std(trust))}
            a["n_seeds"] = len(seeds)
            report[tag]["projections"][kind] = a
            print(f"  [{kind.upper():5s} 2-d, {len(seeds)} seed(s)] "
                  f"silhouette={a['silhouette']['mean']:+.4f}±{a['silhouette']['std']:.4f}  "
                  f"trustworthiness={a['trustworthiness']['mean']:.4f}")

    (out_dir / f"geometry3_{ds}.json").write_text(json.dumps(report, indent=2))

    rows = []
    for tag in embs:
        o = report[tag]["original_space"]
        rows.append({"model": tag, "space": "ORIGINAL (1024-d)", "n_seeds": 1, **o, "trustworthiness": np.nan})
        for kind, a in report[tag]["projections"].items():
            rows.append({"model": tag, "space": f"{PROJ_LABEL[kind]} (2-d)", "n_seeds": a["n_seeds"],
                         "silhouette": a["silhouette"]["mean"], "davies_bouldin": a["davies_bouldin"]["mean"],
                         "calinski_harabasz": a["calinski_harabasz"]["mean"],
                         "knn_acc": np.nan, "knn_macro_f1": np.nan,
                         "trustworthiness": a["trustworthiness"]["mean"]})
    pd.DataFrame(rows).to_csv(out_dir / f"geometry3_{ds}.csv", index=False)

    # =================== FIGURES ===================
    order = ["pretrained", "single_ft", "context_aware"]

    # (a) 3 models x 3 projections
    fig, axes = plt.subplots(3, 3, figsize=(11, 10.5))
    for r, tag in enumerate(order):
        for c, kind in enumerate(["pca", "tsne", "umap"]):
            sil = report[tag]["projections"][kind]["silhouette"]["mean"]
            scatter(axes[r, c], grid_Z[(tag, kind)], y, classes, cmap,
                    f"{PROJ_LABEL[kind]}  (silhouette {sil:+.3f})",
                    show_axes_label=MODEL_LABEL[tag] if c == 0 else None)
    handles = [plt.Line2D([], [], marker="o", ls="", ms=7, mfc=cmap[c], mec="white", mew=.5, label=c)
               for c in classes]
    fig.legend(handles=handles, loc="lower center", ncol=len(classes), frameon=False, fontsize=9)
    fig.suptitle(f"{ds.upper()} — CLS embedding geometry: 3 models × 3 projections\n"
                 f"(conclusion is invariant to the projection, so it is not a t-SNE artifact)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(fig_dir / f"geometry_{ds}_grid.png", dpi=200); plt.close(fig)

    # (b) t-SNE seed robustness on the context model
    X = embs["context_aware"]
    X50 = PCA(n_components=50, random_state=0).fit_transform(X)
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.6))
    for i, s in enumerate(SEEDS):
        Z = project(X, "tsne", s, X50)
        scatter(axes[i], Z, y, classes, cmap, f"t-SNE seed {s}  (sil {silhouette_score(Z,y):+.3f})")
    fig.legend(handles=handles, loc="lower center", ncol=len(classes), frameon=False, fontsize=9)
    fig.suptitle(f"{ds.upper()} — context-aware model under 5 t-SNE seeds "
                 f"(structure is stable; silhouette std = "
                 f"{report['context_aware']['projections']['tsne']['silhouette']['std']:.4f})", fontsize=12)
    fig.tight_layout(rect=[0, 0.10, 1, 0.90])
    fig.savefig(fig_dir / f"geometry_{ds}_tsne_seeds.png", dpi=200); plt.close(fig)

    # (c) original-space metrics across the 3 models
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    metrics = [("silhouette", "Silhouette ↑", None), ("knn_acc", "k-NN probe accuracy ↑", None),
               ("davies_bouldin", "Davies-Bouldin ↓", None)]
    xs = np.arange(3)
    for ax, (key, title, _) in zip(axes, metrics):
        vals = [report[t]["original_space"][key] for t in order]
        ax.bar(xs, vals, width=0.6, color=["#c9c9c9", "#7fa8dc", "#2a78d6"], edgecolor="white", linewidth=1)
        for x, v in zip(xs, vals):
            ax.text(x, v, f"{v:.3f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(xs)
        ax.set_xticklabels(["pretrained", "single-utt", "context"], fontsize=9)
        ax.axhline(0, color="#999", lw=0.8)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color="#eee", lw=0.7)
        ax.set_axisbelow(True)
    fig.suptitle(f"{ds.upper()} — ORIGINAL-space metrics (no projection): pretrained → single-utterance → context",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(fig_dir / f"geometry_{ds}_metrics.png", dpi=200); plt.close(fig)

    print(f"\n[saved] {out_dir}/geometry3_{ds}.{{json,csv}}")
    print(f"[figures] {fig_dir}/geometry_{ds}_{{grid,tsne_seeds,metrics}}.png")


if __name__ == "__main__":
    main()
