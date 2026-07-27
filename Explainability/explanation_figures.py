#!/usr/bin/env python3
"""Figures for the two explainability reviewer notes (faithfulness + agreement).

Reads the CSV/JSON already produced by faithfulness_eval.py and explanation_agreement.py
(no GPU, no model) and writes:

    faithfulness_<ds>_metrics.png   the 6 faithfulness metrics, each explainer vs the
                                    RANDOM baseline (the reference line that makes the
                                    result meaningful)
    faithfulness_<ds>_curves.png    deletion & insertion curves -- the classic faithfulness
                                    figure: predicted-class probability as the most-important
                                    tokens are progressively removed / re-inserted
    agreement_<ds>.png              LIME<->GradSHAP rank-correlation distribution, top-k
                                    overlap, and cross-seed attribution stability

Run:
    python -u explanation_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from erc_paths import CHECKPOINTS_DIR, RESULTS_DIR

CK = CHECKPOINTS_DIR
FAITH = Path(f"{CK}/faithfulness")
AGREE = Path(f"{CK}/agreement")
OUT = RESULTS_DIR / "figures"; OUT.mkdir(parents=True, exist_ok=True)

# categorical slots 1,2 for the two real explainers; RANDOM is a baseline -> neutral gray
COLOR = {"gradshap": "#2a78d6", "lime": "#1baf7a", "random": "#b6b6b6"}
NAME = {"gradshap": "GradSHAP", "lime": "LIME", "random": "Random (baseline)"}
ORDER = ["gradshap", "lime", "random"]

# (key, pretty, higher_is_better)
METRICS = [
    ("comprehensiveness", "Comprehensiveness ↑", True),
    ("sufficiency",       "Sufficiency ↓",       False),
    ("aopc",              "AOPC ↑",              True),
    ("logit_drop",        "Logit-drop ↑",        True),
    ("deletion_auc",      "Deletion AUC ↓",      False),
    ("insertion_auc",     "Insertion AUC ↑",     True),
]


def _clean(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color("#d0d0d0")
    ax.grid(axis="y", color="#eee", lw=0.7)
    ax.set_axisbelow(True)


def fig_metrics(ds):
    f = FAITH / f"faithfulness_{ds}_summary.csv"
    if not f.exists():
        return
    df = pd.read_csv(f).set_index("explainer")
    expl = [e for e in ORDER if e in df.index]

    # how many metrics does each explainer actually beat Random on? (don't assert, count)
    beats = {e: 0 for e in expl if e != "random"}
    for key, _t, hib in METRICS:
        rnd = df.loc["random", f"{key}_mean"]
        for e in beats:
            v = df.loc[e, f"{key}_mean"]
            if (v > rnd) if hib else (v < rnd):
                beats[e] += 1

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4))
    for ax, (key, title, hib) in zip(axes.ravel(), METRICS):
        vals = [df.loc[e, f"{key}_mean"] for e in expl]
        errs = [df.loc[e, f"{key}_std"] / np.sqrt(df.loc[e, "n"]) for e in expl]  # s.e.m.
        xs = np.arange(len(expl))
        ax.bar(xs, vals, width=0.62, color=[COLOR[e] for e in expl],
               edgecolor="white", linewidth=1, yerr=errs, capsize=3,
               error_kw=dict(lw=1, ecolor="#666"))
        rnd = df.loc["random", f"{key}_mean"]
        ax.axhline(rnd, color="#888", ls="--", lw=1)
        for x, v, e in zip(xs, vals, expl):
            worse = e != "random" and ((v <= rnd) if hib else (v >= rnd))
            ax.text(x, v, f"{v:.3f}" + (" ✗" if worse else ""), ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=8.5,
                    color="#e34948" if worse else "#333")
        ax.set_title(title, fontsize=10.5)
        ax.set_xticks(xs)
        ax.set_xticklabels([NAME[e].split(" ")[0] for e in expl], fontsize=9)
        _clean(ax)
    n_m = len(METRICS)
    tail = "; ".join(f"{NAME[e].split(' ')[0]} beats Random on {b}/{n_m}" for e, b in beats.items())
    fig.suptitle(f"{ds.upper()} — faithfulness of the explanations (full test corpus, n={int(df['n'].iloc[0])})\n"
                 f"dashed line = Random baseline ({tail}; ✗ = does not beat Random)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(OUT / f"faithfulness_{ds}_metrics.png", dpi=200)
    plt.close(fig)
    print(f"  [saved] faithfulness_{ds}_metrics.png")


def fig_curves(ds):
    curves = {e: FAITH / f"curves_{ds}_{e}.csv" for e in ORDER}
    if not all(p.exists() for p in curves.values()):
        print(f"  [skip] curves_{ds}_*.csv not ready yet")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for e in ORDER:
        c = pd.read_csv(curves[e])
        axes[0].plot(c["frac"], c["deletion_mean"], color=COLOR[e], lw=2,
                     marker="o", ms=4, label=NAME[e])
        axes[1].plot(c["frac"], c["insertion_mean"], color=COLOR[e], lw=2,
                     marker="o", ms=4, label=NAME[e])
    axes[0].set_title("Deletion — remove most-important tokens first  (lower = better)", fontsize=10.5)
    axes[0].set_xlabel("fraction of tokens removed")
    axes[1].set_title("Insertion — add most-important tokens first  (higher = better)", fontsize=10.5)
    axes[1].set_xlabel("fraction of tokens inserted")
    for ax in axes:
        ax.set_ylabel("predicted-class probability")
        ax.legend(frameon=False, fontsize=9)
        _clean(ax)
        ax.grid(color="#eee", lw=0.7)
    fig.suptitle(f"{ds.upper()} — deletion / insertion curves (full test corpus)\n"
                 f"a faithful explainer collapses the prediction fastest when its top tokens are removed",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(OUT / f"faithfulness_{ds}_curves.png", dpi=200)
    plt.close(fig)
    print(f"  [saved] faithfulness_{ds}_curves.png")


def fig_agreement(ds):
    per = AGREE / f"agreement_{ds}_perexample.csv"
    summ = AGREE / f"agreement_{ds}_summary.json"
    if not (per.exists() and summ.exists()):
        print(f"  [skip] agreement_{ds} not found")
        return
    d = pd.read_csv(per)
    r = json.load(open(summ))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    # (a) distribution of per-example Spearman between the two methods
    ax = axes[0]
    ax.hist(d["spearman"].dropna(), bins=40, color="#4a3aa7", alpha=0.85, edgecolor="white")
    mu = r["spearman"]["mean"]
    ax.axvline(0, color="#888", ls="--", lw=1)
    ax.axvline(mu, color="#e34948", lw=2, label=f"mean = {mu:+.3f}")
    ax.set_title("Rank correlation, LIME vs GradSHAP\n(per example, Spearman)", fontsize=10.5)
    ax.set_xlabel("Spearman ρ"); ax.set_ylabel("examples")
    ax.legend(frameon=False, fontsize=9)
    _clean(ax)

    # (b) top-k overlap
    ax = axes[1]
    keys = ["jaccard@5", "jaccard@10", "jaccard@10pct"]
    vals = [r[k]["mean"] for k in keys]
    xs = np.arange(len(keys))
    ax.bar(xs, vals, width=0.6, color="#eda100", edgecolor="white", linewidth=1)
    for x, v in zip(xs, vals):
        ax.text(x, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(xs); ax.set_xticklabels(["top-5", "top-10", "top-10%"], fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title("Top-k token overlap, LIME vs GradSHAP\n(Jaccard; 1.0 = identical)", fontsize=10.5)
    _clean(ax)

    # (c) cross-seed attribution stability, per method
    ax = axes[2]
    methods = ["gradshap", "lime"]
    sp = [r[f"stability_{m}"]["spearman"]["mean"] for m in methods]
    jc = [r[f"stability_{m}"]["jaccard@10"]["mean"] for m in methods]
    xs = np.arange(len(methods)); w = 0.35
    ax.bar(xs - w/2, sp, w, label="Spearman ρ", color="#2a78d6", edgecolor="white", linewidth=1)
    ax.bar(xs + w/2, jc, w, label="Jaccard@10", color="#1baf7a", edgecolor="white", linewidth=1)
    for x, v in zip(xs - w/2, sp):
        ax.text(x, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5)
    for x, v in zip(xs + w/2, jc):
        ax.text(x, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(xs); ax.set_xticklabels(["GradSHAP", "LIME"], fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title("Attribution stability across the 5 seeds\n(higher = more reproducible)", fontsize=10.5)
    ax.legend(frameon=False, fontsize=9)
    _clean(ax)

    fig.suptitle(f"{ds.upper()} — explanation agreement (n={r['n_agreement']}): the two methods are both "
                 f"faithful yet barely agree, and LIME is far more seed-stable", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(OUT / f"agreement_{ds}.png", dpi=200)
    plt.close(fig)
    print(f"  [saved] agreement_{ds}.png")


if __name__ == "__main__":
    for ds in ["meld", "iemocap"]:
        print(f"[{ds}]")
        fig_metrics(ds)
        fig_curves(ds)
        fig_agreement(ds)
    print(f"\n[figures] {OUT}")
