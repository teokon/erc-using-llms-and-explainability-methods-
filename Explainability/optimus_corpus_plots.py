#!/usr/bin/env python3
"""Optimus corpus-level plots -- the figures from Optimus_Corpus_Level.ipynb, ported to
the RoBERTa-large context-aware EmoBERTa model and extended to all THREE Optimus variants.

The notebook measured attribution CONCENTRATION (how few tokens carry the explanation),
which is a complexity/sparsity diagnostic -- distinct from the faithfulness metrics:

    coverage_curve(s)          normalise positive scores to a distribution, sort desc,
                               cumulate -> "share of total attribution held by the top x%
                               of tokens", evaluated on a 100-step grid.
    coverage_at_fraction(s,.1) Coverage@10%: the share held by the top 10% of tokens.

Both are reproduced exactly as in the notebook (same formulas, same GRID).

Figures written to <out>/figures_optimus/:
    optimus_<ds>_cumulative.png        cumulative contribution curves, one line per variant
    optimus_<ds>_cov10_by_class.png    Coverage@10% per predicted emotion, grouped bars
    optimus_<ds>_sparsity.png          distribution of Coverage@10% over utterances

Reads the raw attributions saved by faithfulness_eval.py --save_scores, so it needs no GPU
and no re-explaining (Optimus Prime costs ~113 s/example, so recomputing is not an option).

Run:
    python -u optimus_corpus_plots.py
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
SUB = CK / "faith_subset"
OUT = RESULTS_DIR / "figures_optimus"

N_STEPS = 100
GRID = np.linspace(1 / N_STEPS, 1.0, N_STEPS)

# the three Optimus variants + the two reference explainers, in fixed colour order
SERIES = [
    ("optimus",       "Optimus – Baseline (A)",       "#2a78d6"),
    ("optimus_batch", "Optimus – Batch (max_across)", "#1baf7a"),
    ("optimus_prime", "Optimus – Prime (per instance)", "#eda100"),
    ("gradshap",      "GradSHAP",                     "#4a3aa7"),
    ("lime",          "LIME",                         "#e34948"),
]


def coverage_curve(s):
    """Notebook-identical: share of total (positive) attribution in the top x% of tokens."""
    s = np.maximum(np.asarray(s, dtype=float), 0)
    tot = s.sum()
    if tot <= 0:
        return None
    p = np.sort(s / tot)[::-1]
    c = np.cumsum(p)
    T = len(p)
    return np.array([c[min(max(1, int(np.ceil(g * T))), T) - 1] for g in GRID])


def coverage_at_fraction(s, frac=0.1):
    s = np.maximum(np.asarray(s, dtype=float), 0)
    tot = s.sum()
    if tot <= 0:
        return 0.0
    p = np.sort(s / tot)[::-1]
    k = max(1, int(np.ceil(frac * len(p))))
    return float(p[:k].sum())


def load(ds, name):
    f = SUB / f"scores_{ds}_{name}.npz"
    if not f.exists():
        return None
    d = np.load(f, allow_pickle=True)
    return d["scores"], d["labels"], d["preds"]


def _clean(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(True, ls="--", alpha=0.3)
    ax.set_axisbelow(True)


def build(ds):
    have = {n: load(ds, n) for n, _l, _c in SERIES}
    have = {n: v for n, v in have.items() if v is not None}
    if not have:
        print(f"[{ds}] no saved scores yet -- run faithfulness_eval.py --save_scores first")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    n_ex = len(next(iter(have.values()))[0])
    print(f"[{ds}] {len(have)} explainers, {n_ex} examples: {list(have)}")

    curves, cov10, cov10_by_class = {}, {}, {}
    for name, (scores, labels, preds) in have.items():
        cs, c10 = [], []
        byc = {}
        for s, lab in zip(scores, labels):
            c = coverage_curve(s)
            if c is None:
                continue
            cs.append(c)
            v = coverage_at_fraction(s, 0.1)
            c10.append(v)
            byc.setdefault(str(lab), []).append(v)
        curves[name] = np.mean(np.vstack(cs), axis=0)
        cov10[name] = np.array(c10)
        cov10_by_class[name] = byc

    # ---- (1) cumulative contribution curves ----
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    for name, lab, col in SERIES:
        if name in curves:
            ax.plot(GRID * 100, curves[name], label=lab, color=col, lw=2)
    ax.set_xlabel("Top x% tokens (sorted by importance)")
    ax.set_ylabel("Cumulative share of total attribution")
    ax.set_title(f"{ds.upper()} — cumulative contribution curves (predicted label)\n"
                 f"steeper = the explanation is concentrated in fewer tokens  (n={n_ex})")
    ax.legend(frameon=False, fontsize=9)
    _clean(ax)
    fig.tight_layout()
    fig.savefig(OUT / f"optimus_{ds}_cumulative.png", dpi=200)
    plt.close(fig)

    # ---- (2) Coverage@10% per predicted emotion ----
    classes = sorted({c for byc in cov10_by_class.values() for c in byc})
    names = [n for n, _l, _c in SERIES if n in cov10_by_class]
    x = np.arange(len(classes))
    w = 0.8 / max(1, len(names))
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    for i, name in enumerate(names):
        lab = dict((n, l) for n, l, _ in SERIES)[name]
        col = dict((n, c) for n, _, c in SERIES)[name]
        vals = [np.mean(cov10_by_class[name].get(c, [0.0])) for c in classes]
        ax.bar(x + (i - (len(names) - 1) / 2) * w, vals, width=w, label=lab,
               color=col, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylabel("Coverage@10% (mean over utterances)")
    ax.set_title(f"{ds.upper()} — Coverage@10% per emotion  (n={n_ex})")
    ax.legend(frameon=False, ncol=2, fontsize=9)
    _clean(ax)
    fig.tight_layout()
    fig.savefig(OUT / f"optimus_{ds}_cov10_by_class.png", dpi=200)
    plt.close(fig)

    # ---- (3) sparsity distribution ----
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for name, lab, col in SERIES:
        if name in cov10:
            ax.hist(cov10[name], bins=30, histtype="step", lw=2, label=lab, color=col)
    ax.set_xlabel("Coverage@10% (share of attribution in the top 10% of tokens)")
    ax.set_ylabel("Number of utterances")
    ax.set_title(f"{ds.upper()} — sparsity distribution of the explanations  (n={n_ex})")
    ax.legend(frameon=False, fontsize=9)
    _clean(ax)
    fig.tight_layout()
    fig.savefig(OUT / f"optimus_{ds}_sparsity.png", dpi=200)
    plt.close(fig)

    print(f"[{ds}] Coverage@10% (mean): " +
          "  ".join(f"{n}={cov10[n].mean():.3f}" for n in names))
    print(f"[{ds}] figures -> {OUT}/optimus_{ds}_{{cumulative,cov10_by_class,sparsity}}.png")


if __name__ == "__main__":
    for ds in ["meld", "iemocap"]:
        build(ds)
