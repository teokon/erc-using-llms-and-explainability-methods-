#!/usr/bin/env python3
"""The paper's Optimus corpus-level figures, rebuilt from the saved attributions.

Reproduces Optimus_Corpus_Level.ipynb (cumulative contribution curves + Coverage@10%
per predicted emotion, FT vs Base) on RoBERTa-large, extended to all THREE Optimus
variants, using the attributions saved by faithfulness_eval.py --save_scores.

No GPU and no re-explaining: Optimus Prime costs ~1768 config evaluations per example,
so the scores are reused rather than recomputed.

The metrics are the notebook's, unchanged:
    coverage_curve(s)            share of total (positive) attribution held by the top x%
                                 of tokens, on a 100-step grid
    coverage_at_fraction(s, .1)  Coverage@10%

NOTE the pretrained model carries a randomly-initialised classification head. That makes
its FAITHFULNESS metrics meaningless (they are ~0 for every explainer, including Random)
and those are reported nowhere. Concentration, however, is well defined regardless: it
describes how peaked the attention-derived attribution is, not how well it predicts. The
Baseline (A) variant uses no FTP at all, so the FT-vs-Base Baseline comparison is clean;
Batch/Prime on the pretrained side select their config via FTP querying the random head
and are therefore marked with * and shown for completeness only.

Run:
    python -u paper_plots_from_scores.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from erc_paths import CHECKPOINTS_DIR, RESULTS_DIR

CK = CHECKPOINTS_DIR
SRC = CK / "faith_final"
OUT = RESULTS_DIR / "figures_optimus"

N_STEPS = 100
GRID = np.linspace(1 / N_STEPS, 1.0, N_STEPS)

CLASS_ORDER = {
    "meld": ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"],
    "iemocap": ["neutral", "frustration", "sadness", "anger", "excited", "happiness"],
}
# (model, explainer, label, colour, linestyle) -- paper colours: FT blue/orange, Base green/red
SERIES = [
    ("single_ft",  "optimus",       "FT – Baseline (A)",      "#1f77b4", "-"),
    ("single_ft",  "optimus_batch", "FT – Optimus Batch",     "#17becf", "-"),
    ("single_ft",  "optimus_prime", "FT – Optimus Prime",     "#ff7f0e", "-"),
    ("pretrained", "optimus",       "Base – Baseline (A)",    "#2ca02c", "--"),
    ("pretrained", "optimus_batch", "Base – Optimus Batch*",  "#8c564b", "--"),
    ("pretrained", "optimus_prime", "Base – Optimus Prime*",  "#d62728", "--"),
]


def coverage_curve(s):
    s = np.maximum(np.asarray(s, float), 0)
    tot = s.sum()
    if tot <= 0:
        return None
    p = np.sort(s / tot)[::-1]
    c = np.cumsum(p)
    T = len(p)
    return np.array([c[min(max(1, int(np.ceil(g * T))), T) - 1] for g in GRID])


def coverage_at_fraction(s, frac=0.1):
    s = np.maximum(np.asarray(s, float), 0)
    tot = s.sum()
    if tot <= 0:
        return 0.0
    p = np.sort(s / tot)[::-1]
    k = max(1, int(np.ceil(frac * len(p))))
    return float(p[:k].sum())


def load(ds, model, expl):
    f = SRC / f"scores_{ds}_{model}_{expl}.npz"
    if not f.exists():
        return None
    d = np.load(f, allow_pickle=True)
    return d["scores"], d["labels"]


def _clean(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(True, ls="--", alpha=0.3)
    ax.set_axisbelow(True)


def build(ds):
    OUT.mkdir(parents=True, exist_ok=True)
    curves, cov10, byclass, n = {}, {}, {}, 0
    for model, expl, lab, _c, _ls in SERIES:
        got = load(ds, model, expl)
        if got is None:
            print(f"  [{ds}] missing: {model}/{expl}")
            continue
        scores, labels = got
        n = len(scores)
        cs, c10, bc = [], [], {}
        for s, gold in zip(scores, labels):
            c = coverage_curve(s)
            if c is None:
                continue
            cs.append(c)
            v = coverage_at_fraction(s, 0.1)
            c10.append(v)
            bc.setdefault(str(gold), []).append(v)
        key = f"{model}/{expl}"
        curves[key] = np.mean(np.vstack(cs), 0)
        cov10[key] = np.array(c10)
        byclass[key] = bc

    # ---- (1) cumulative contribution curves ----
    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    for model, expl, lab, col, ls in SERIES:
        k = f"{model}/{expl}"
        if k in curves:
            ax.plot(GRID * 100, curves[k], label=lab, color=col, ls=ls, lw=2)
    ax.set_xlabel("Top x% tokens (sorted by importance)")
    ax.set_ylabel("Cumulative share of total attribution")
    ax.set_title(f"Cumulative curves — FT vs Base ({ds.upper()}, single utterances, n={n})")
    ax.legend(frameon=False, fontsize=9)
    _clean(ax)
    fig.text(0.01, 0.005, "*On the pretrained model, Batch/Prime select their config via FTP, which "
                          "queries a randomly-initialised head — shown for completeness only.",
             fontsize=7, color="#777")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(OUT / f"paper_{ds}_cumulative.png", dpi=200)
    plt.close(fig)

    # ---- (2) Coverage@10% per predicted emotion ----
    present = {c for bc in byclass.values() for c in bc}
    classes = [c for c in CLASS_ORDER[ds] if c in present] + sorted(present - set(CLASS_ORDER[ds]))
    x = np.arange(len(classes))
    w = 0.8 / len(SERIES)
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    for i, (model, expl, lab, col, _ls) in enumerate(SERIES):
        k = f"{model}/{expl}"
        if k not in byclass:
            continue
        vals = [np.mean(byclass[k].get(c, [0.0])) for c in classes]
        ax.bar(x + (i - (len(SERIES) - 1) / 2) * w, vals, width=w, label=lab,
               color=col, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylabel("Coverage@10% (mean over utterances)")
    ax.set_title(f"Coverage@10% per predicted emotion — FT vs Base ({ds.upper()}, n={n})")
    ax.legend(frameon=False, ncol=2, fontsize=8.5)
    _clean(ax)
    fig.tight_layout()
    fig.savefig(OUT / f"paper_{ds}_cov10_by_class.png", dpi=200)
    plt.close(fig)

    # ---- (3) sparsity distribution ----
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    for model, expl, lab, col, _ls in SERIES:
        k = f"{model}/{expl}"
        if k in cov10:
            ax.hist(cov10[k], bins=30, histtype="step", lw=2, label=lab, color=col)
    ax.set_xlabel("Coverage@10%")
    ax.set_ylabel("Number of utterances")
    ax.set_title(f"Sparsity distribution — FT vs Base ({ds.upper()}, n={n})")
    ax.legend(frameon=False, fontsize=8.5)
    _clean(ax)
    fig.tight_layout()
    fig.savefig(OUT / f"paper_{ds}_sparsity.png", dpi=200)
    plt.close(fig)

    print(f"[{ds}] n={n} | Coverage@10%: " +
          "  ".join(f"{k.split('/')[0][:4]}-{k.split('/')[1].replace('optimus','opt')}"
                    f"={cov10[k].mean():.3f}" for k in cov10))
    print(f"[{ds}] -> {OUT}/paper_{ds}_{{cumulative,cov10_by_class,sparsity}}.png")


if __name__ == "__main__":
    for ds in ["meld", "iemocap"]:
        build(ds)
