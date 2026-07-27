#!/usr/bin/env python3
"""Standalone, full-size CLS-embedding visualisations -- one figure per
(model x projection), in the same style as the original t-SNE figure.

For each of the three models
    pretrained  /  single-utterance fine-tuned  /  context-aware EmoBERTa
and each projection
    PCA  /  t-SNE  /  UMAP
it writes one large scatter coloured by emotion, titled with the original-space
and 2-D silhouette so the figure carries its own evidence.

Embeddings are cached to <out_dir>/emb/<ds>_<model>.npy, so re-plotting (restyling,
different projections) costs no GPU time.

Run:
    GPU=0 python -u embedding_plots.py --dataset meld
"""
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from erc_paths import pick_gpu
pick_gpu()  # pin ONE GPU before importing torch ($GPU > single CUDA_VISIBLE_DEVICES > "0")

import argparse
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
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

warnings.filterwarnings("ignore")
import faithfulness_eval as FE

MAX_LEN, BATCH, TSNE_SEED = 512, 32, 0
SPEAKER_RE = re.compile(r"^[^:]{1,30}:\s*")
PALETTE = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4"]

MODELS = ["pretrained", "single_ft", "context_aware"]
MODEL_TITLE = {
    "pretrained":    "Pretrained RoBERTa-large (no fine-tuning)",
    "single_ft":     "Fine-tuned RoBERTa-large — single utterance (no context)",
    "context_aware": "Context-aware EmoBERTa (past + future)",
}
PROJ_TITLE = {"pca": "PCA", "tsne": "t-SNE", "umap": "UMAP"}


def target_utterance(ctx):
    p = str(ctx).split("</s></s>")
    t = p[1].strip() if len(p) == 3 else str(ctx).strip()
    return SPEAKER_RE.sub("", t, count=1).strip()


@torch.inference_mode()
def cls_embeddings(encoder, tok, texts):
    out = []
    for i in range(0, len(texts), BATCH):
        enc = tok(texts[i:i + BATCH], return_tensors="pt", padding=True, truncation=True,
                  max_length=MAX_LEN, add_special_tokens=True).to(FE.DEVICE)
        hs = encoder(**enc, output_hidden_states=True, return_dict=True).hidden_states[-1]
        out.append(hs[:, 0, :].float().cpu().numpy())
    return np.concatenate(out, 0)


def get_embeddings(ds, cfg, emb_dir, tok, utt_texts, ctx_texts):
    """Compute once, cache to .npy."""
    embs = {}
    single_ckpt = f"{FE.CK}/roberta_large_single_{ds}/roberta_single_{ds}_seed42_BEST"
    for tag in MODELS:
        f = emb_dir / f"{ds}_{tag}.npy"
        if f.exists():
            embs[tag] = np.load(f); print(f"  [cache] {tag}"); continue
        print(f"  [compute] {tag}")
        if tag == "pretrained":
            m = AutoModel.from_pretrained("roberta-large").to(FE.DEVICE).eval()
            X = cls_embeddings(m, tok, utt_texts)
        elif tag == "single_ft":
            m = AutoModelForSequenceClassification.from_pretrained(single_ckpt).to(FE.DEVICE).eval()
            X = cls_embeddings(m.base_model, tok, utt_texts)
        else:
            m = AutoModelForSequenceClassification.from_pretrained(cfg["ckpt"]).to(FE.DEVICE).eval()
            X = cls_embeddings(m.base_model, tok, ctx_texts)
        del m; torch.cuda.empty_cache()
        np.save(f, X); embs[tag] = X
    return embs


def project(X, kind, X50):
    if kind == "pca":
        return PCA(n_components=2, random_state=0).fit_transform(X)
    if kind == "tsne":
        return TSNE(n_components=2, perplexity=30, init="pca",
                    random_state=TSNE_SEED, max_iter=1000).fit_transform(X50)
    import umap
    return umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(X50)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(FE.DATASETS))
    ap.add_argument("--out_dir", default=f"{FE.CK}/geometry")
    args = ap.parse_args()

    ds = args.dataset
    out = Path(args.out_dir); fig_dir = out / "figures_single"; fig_dir.mkdir(parents=True, exist_ok=True)
    emb_dir = out / "emb"; emb_dir.mkdir(parents=True, exist_ok=True)

    cfg = FE.DATASETS[ds]
    df = pd.read_csv(cfg["test_csv"])
    ctx_texts = df["context_text_raw"].astype(str).tolist()
    utt_texts = [target_utterance(t) for t in ctx_texts]
    y = df["label"].astype(str).to_numpy()
    classes = sorted(pd.unique(y).tolist())
    cmap = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(classes)}
    print(f"[plots] {ds}: {len(y)} examples, {len(classes)} classes")

    tok = AutoTokenizer.from_pretrained("roberta-large", use_fast=True, add_prefix_space=True)
    embs = get_embeddings(ds, cfg, emb_dir, tok, utt_texts, ctx_texts)

    for tag in MODELS:
        X = embs[tag]
        sil_orig = silhouette_score(X, y)
        X50 = PCA(n_components=min(50, X.shape[1]), random_state=0).fit_transform(X)
        for kind in ["pca", "tsne", "umap"]:
            Z = project(X, kind, X50)
            sil2d = silhouette_score(Z, y)

            fig, ax = plt.subplots(figsize=(9, 7.5))
            for c in classes:
                m = (y == c)
                ax.scatter(Z[m, 0], Z[m, 1], s=16, c=cmap[c], alpha=0.7,
                           linewidths=0.25, edgecolors="white", label=c, rasterized=True)
            ax.set_title(f"{ds.upper()} — {PROJ_TITLE[kind]} of CLS embeddings\n{MODEL_TITLE[tag]}",
                         fontsize=13, pad=12)
            ax.set_xlabel(f"{PROJ_TITLE[kind]} dim 1", fontsize=10)
            ax.set_ylabel(f"{PROJ_TITLE[kind]} dim 2", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("#d8d8d8"); s.set_linewidth(0.7)
            ax.legend(title="Emotion", loc="center left", bbox_to_anchor=(1.01, 0.5),
                      frameon=False, fontsize=10, title_fontsize=10, markerscale=1.6)
            ax.text(0.01, -0.09,
                    f"silhouette: original 1024-d = {sil_orig:+.4f}   |   {PROJ_TITLE[kind]} 2-d = {sil2d:+.4f}",
                    transform=ax.transAxes, fontsize=9, color="#555")
            fig.tight_layout()
            f = fig_dir / f"{ds}_{tag}_{kind}.png"
            fig.savefig(f, dpi=200, bbox_inches="tight"); plt.close(fig)
            print(f"  [saved] {f.name}   (2-d silhouette {sil2d:+.4f})")

    print(f"\n[figures] {fig_dir}")


if __name__ == "__main__":
    main()
