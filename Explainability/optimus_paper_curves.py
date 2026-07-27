#!/usr/bin/env python3
"""Reproduce the paper's Optimus corpus-level curves on RoBERTa-large.

This is the port of Optimus_Corpus_Level.ipynb: the comparison in the paper is
    pretrained roberta-large   vs   the fine-tuned SINGLE-UTTERANCE model
on SINGLE UTTERANCES (the notebook's TEXT_COLUMN="Utterance"), NOT on the context model.

Metrics are the notebook's, unchanged:
    coverage_curve(s)           share of total (positive) attribution held by the top x%
                                of tokens, on a 100-step grid
    coverage_at_fraction(s,0.1) Coverage@10%
Base-model scores are indexed with the FINE-TUNED model's predicted label, exactly as the
notebook did (`sB_base = scores_base_B_all[pred_idx]`).

Produced with the ORIGINAL Optimus FTP (no vectorised fast path).

CAVEAT (worth a sentence in the paper): the pretrained model has a randomly-initialised
classification head. That is harmless for Baseline (A), which uses no FTP and is therefore
pure pretrained attention -- the quantity being compared. It is NOT harmless for Prime,
whose config search is driven by FTP, i.e. by that random head. We therefore report
Base-Prime for completeness but do not draw conclusions from it.

Run:
    GPU=0 python -u optimus_paper_curves.py --dataset meld
"""
import os
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from erc_paths import CHECKPOINTS_DIR, RESULTS_DIR, OPTIMUS_REPO, pick_gpu
pick_gpu()  # pin ONE GPU before importing torch ($GPU > single CUDA_VISIBLE_DEVICES > "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import re
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.insert(0, str(OPTIMUS_REPO))
import faithfulness_eval as FE

# These figures go in the paper, so they are produced with the ORIGINAL, untouched Optimus
# implementation -- no vectorised FTP. Single utterances are short (~14 tokens), so the
# stock code is affordable here. (The vectorised FTP was verified to select the identical
# config, but for the published figures we do not rely on that.)

CK = CHECKPOINTS_DIR
OUT = RESULTS_DIR / "figures_optimus"
N_STEPS = 100
GRID = np.linspace(1 / N_STEPS, 1.0, N_STEPS)
SPEAKER_RE = re.compile(r"^[^:]{1,30}:\s*")

# Class order as printed in the paper's figures (IEMOCAP is the IEMO6 order, not alphabetical)
CLASS_ORDER = {
    "meld": ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"],
    "iemocap": ["neutral", "frustration", "sadness", "anger", "excited", "happiness"],
}
# Series styling matched to the paper's figures (FT-Baseline blue, FT-Prime orange,
# Base-Baseline green, Base-Prime red), with the Batch variant added in adjacent hues.
STYLE = [
    ("ft_B", "FT – Baseline (A)",     "#1f77b4", "-"),
    ("ft_X", "FT – Optimus Batch",    "#17becf", "-"),
    ("ft_P", "FT – Optimus Prime",    "#ff7f0e", "-"),
    ("bs_B", "Base – Baseline (A)",   "#2ca02c", "--"),
    ("bs_X", "Base – Optimus Batch*", "#8c564b", "--"),
    ("bs_P", "Base – Optimus Prime*", "#d62728", "--"),
]


def target_utterance(ctx):
    p = str(ctx).split("</s></s>")
    t = p[1].strip() if len(p) == 3 else str(ctx).strip()
    return SPEAKER_RE.sub("", t, count=1).strip()


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


def strip_specials(scores_2d, ids, special):
    arr = np.asarray(scores_2d, float)
    T = min(arr.shape[1], len(ids))
    keep = [i for i in range(T) if ids[i] not in special]
    return arr[:, keep] if keep else arr[:, :T]


def build_optimus(ckpt, num_labels, tok, id2label=None, calib=None, tag=""):
    """calib -> runs Optimus' max_across calibration (Batch); None -> baseline/Prime only."""
    kw = {}
    if id2label is not None:
        kw = dict(num_labels=num_labels, id2label=id2label,
                  label2id={v: k for k, v in id2label.items()})
    m = AutoModelForSequenceClassification.from_pretrained(ckpt, **kw).to(FE.DEVICE).eval()
    w = FE._OptimusWrapper(m, tok)
    from optimus import Optimus   # stock FTP (no install_fast_ftp) -- see module docstring
    if calib:
        t0 = time.time()
        print(f"    [{tag}] Batch calibration on {len(calib)} utterances ...", flush=True)
        ion = Optimus(w, tok, w.label_names, task="single_label", set_of_instance=calib)
        print(f"    [{tag}] calibrated in {(time.time()-t0)/60:.1f} min -> config {ion.max_across_a}", flush=True)
    else:
        ion = Optimus(w, tok, w.label_names, task="single_label", set_of_instance=None)
    return ion, w


def build_calib_utts(ds, per_class, tok):
    """Stratified calibration utterances from TRAIN (never the test set)."""
    d = pd.read_csv(FE.TRAIN_CSV[ds])
    r = np.random.default_rng(42)
    picks = []
    for _c, g in d.groupby("label"):
        idx = g.index.to_numpy().copy()
        r.shuffle(idx)
        picks.extend(idx[:per_class].tolist())
    utts = [target_utterance(t) for t in d.loc[picks, "context_text_raw"].astype(str)]
    return [u for u in utts if len(tok(u, add_special_tokens=False)["input_ids"]) >= 3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["meld", "iemocap"])
    ap.add_argument("--per_class", type=int, default=25, help="stratified test examples per class")
    ap.add_argument("--calib_per_class", type=int, default=10, help="Batch calibration utterances per class (from TRAIN)")
    args = ap.parse_args()
    ds = args.dataset
    OUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(FE.DATASETS[ds]["test_csv"])
    r = np.random.default_rng(0)
    keep = []
    for _c, g in df.groupby("label"):
        idx = g.index.to_numpy().copy()
        r.shuffle(idx)
        keep.extend(idx[:args.per_class].tolist())
    df = df.loc[sorted(keep)].reset_index(drop=True)
    texts = [target_utterance(t) for t in df["context_text_raw"].astype(str)]
    print(f"[paper-curves] {ds}: {len(texts)} single utterances ({args.per_class}/class)")

    tok = AutoTokenizer.from_pretrained("roberta-large", use_fast=True)
    special = set(tok.all_special_ids)
    ft_ckpt = str(CK / f"roberta_large_single_{ds}" / f"roberta_single_{ds}_seed42_BEST")
    calib = build_calib_utts(ds, args.calib_per_class, tok)

    print(f"  loading FINE-TUNED single-utterance model (+Batch calib on {len(calib)} train utts) ...")
    ion_ft, w_ft = build_optimus(ft_ckpt, None, tok, calib=calib, tag="FT")
    labels = w_ft.label_names
    print("  loading PRETRAINED roberta-large (random head; see caveat) ...")
    ion_bs, w_bs = build_optimus("roberta-large", len(labels), tok,
                                 id2label={i: l for i, l in enumerate(labels)},
                                 calib=calib, tag="base")

    series = {k: [] for k in ["ft_B", "ft_X", "ft_P", "bs_B", "bs_X", "bs_P"]}
    byclass = {k: {} for k in series}
    t0 = time.time()
    for i, text in enumerate(texts):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) < 3:
            continue
        try:
            pred = int(np.argmax(w_ft.predict_proba(text)))     # align everything to the FT prediction
            got = {}
            for tag, ion in [("ft", ion_ft), ("bs", ion_bs)]:
                for mode, mk in [("baseline", "B"), ("max_across", "X"), ("max_per_instance", "P")]:
                    sc, _t = ion.explain(text, mode=mode, level="token", raw_attention="A")
                    sc = strip_specials(np.asarray(sc, float), ids, special)
                    got[f"{tag}_{mk}"] = sc[pred]
        except Exception as e:
            print(f"   ex{i} failed ({e}); skipping")
            continue
        for k, s in got.items():
            c = coverage_curve(s)
            if c is None:
                continue
            series[k].append(c)
            byclass[k].setdefault(labels[pred], []).append(coverage_at_fraction(s, 0.1))
        if (i + 1) % 25 == 0:
            print(f"   {i+1}/{len(texts)}  ({(time.time()-t0)/(i+1):.1f}s/ex)", flush=True)

    n = len(series["ft_B"])
    # cache the computed curves so the figures can be restyled without re-explaining
    np.savez_compressed(OUT / f"paper_{ds}_data.npz",
                        curves={k: np.mean(np.vstack(v), 0) for k, v in series.items() if v},
                        byclass=byclass, grid=GRID, n=n, allow_pickle=True)
    plot_all(ds, {k: (np.mean(np.vstack(v), 0) if v else None) for k, v in series.items()},
             byclass, n)


def plot_all(ds, curves, byclass, n):

    # ---- cumulative contribution curves (the paper's figure) ----
    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    for k, lab, col, ls in STYLE:
        if curves.get(k) is not None:
            ax.plot(GRID * 100, curves[k], label=lab, color=col, ls=ls, lw=2)
    ax.set_xlabel("Top x% tokens (sorted by importance)")
    ax.set_ylabel("Cumulative share of total attribution")
    ax.set_title(f"Cumulative curves — FT vs Base ({ds.upper()}, single utterances, n={n})")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, ls="--", alpha=0.3); ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.text(0.01, 0.005, "*On the pretrained model, Batch/Prime select their config via FTP, which queries a "
                          "randomly-initialised head — shown for completeness only.", fontsize=7, color="#777")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(OUT / f"paper_{ds}_cumulative.png", dpi=200); plt.close(fig)

    # ---- Coverage@10% per predicted emotion (paper's class order) ----
    present = {c for d in byclass.values() for c in d}
    classes = [c for c in CLASS_ORDER[ds] if c in present] + \
              sorted(present - set(CLASS_ORDER[ds]))
    x = np.arange(len(classes)); w = 0.8 / len(STYLE)
    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    for i, (k, lab, col, _ls) in enumerate(STYLE):
        vals = [np.mean(byclass[k].get(c, [0.0])) for c in classes]
        ax.bar(x + (i - (len(STYLE)-1)/2) * w, vals, width=w, label=lab, color=col, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylabel("Coverage@10% (mean over utterances)")
    ax.set_title(f"Coverage@10% per predicted emotion — FT vs Base ({ds.upper()}, n={n})")
    ax.legend(frameon=False, ncol=2, fontsize=8.5)
    ax.grid(axis="y", ls="--", alpha=0.3); ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / f"paper_{ds}_cov10_by_class.png", dpi=200); plt.close(fig)

    print(f"\n[paper-curves] n={n} | Coverage@10%: " +
          "  ".join(f"{k}={np.mean([v for vs in byclass[k].values() for v in vs]):.3f}" for k, _l, _c, _s in STYLE))
    print(f"[paper-curves] -> {OUT}/paper_{ds}_{{cumulative,cov10_by_class}}.png")


if __name__ == "__main__":
    main()
