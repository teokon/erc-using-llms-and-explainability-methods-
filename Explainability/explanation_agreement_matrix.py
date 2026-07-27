#!/usr/bin/env python3
"""Cross-method explanation agreement, from the saved attributions (no GPU, no re-explaining).

Reviewer note 7: quantify agreement between the explanation methods. Using the raw
per-token attributions saved by faithfulness_eval.py --save_scores, this computes, for
every PAIR of explainers on the SAME examples:
    - rank correlation  (Spearman rho, per example -> mean +/- std)
    - top-k overlap     (Jaccard of the top-10 tokens)
over the real WORD tokens (embedded <s>/</s></s> segment markers excluded).

Emits a Spearman agreement matrix + a top-10 Jaccard matrix per (dataset, model),
and writes markdown tables for the README.

    python -u explanation_agreement_matrix.py
"""
import itertools
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from erc_paths import CHECKPOINTS_DIR, RESULTS_DIR

CK = CHECKPOINTS_DIR
SPECIAL = {0, 1, 2, 3, 50264}      # roberta-large <s> <pad> </s> <unk> <mask>

METHODS = ["gradshap", "lime", "optimus", "optimus_prime", "optimus_batch", "random"]
SHORT = {"gradshap": "GradSHAP", "lime": "LIME", "optimus": "Opt-base",
         "optimus_prime": "Opt-Prime", "optimus_batch": "Opt-Batch", "random": "Random"}
SOURCES = [("single_ft", CK / "faith_final", "single fine-tuned (utterances)"),
           ("context",   CK / "faith_context", "context-aware (full context)")]


def load(src, ds, model, m):
    f = src / f"scores_{ds}_{model}_{m}.npz"
    if not f.exists():
        return None
    d = np.load(f, allow_pickle=True)
    return d["scores"], d["ids"]


def topk_jaccard(a, b, frac=0.20):
    """Top-k overlap with k = ceil(frac * n_tokens). A FRACTION, not a fixed k, so it is
    comparable across the ~14-token utterances and the ~214-498-token contexts -- a fixed
    k=10 would be ~70% of a short utterance but ~2% of a long context, which is not
    a like-for-like overlap."""
    k = max(1, int(np.ceil(frac * len(a))))
    ta = set(np.argsort(a)[::-1][:k].tolist())
    tb = set(np.argsort(b)[::-1][:k].tolist())
    u = ta | tb
    return len(ta & tb) / len(u) if u else np.nan


def pair_agreement(A, B, ids):
    sp, jac = [], []
    for sa, sb, idl in zip(A[0], B[0], ids):
        keep = [i for i, t in enumerate(idl) if int(t) not in SPECIAL]
        a = np.asarray(sa, float)[keep]
        b = np.asarray(sb, float)[keep]
        if len(a) < 3 or np.all(a == a[0]) or np.all(b == b[0]):
            continue
        sp.append(spearmanr(a, b).correlation)
        jac.append(topk_jaccard(a, b, 0.20))
    sp = np.array([x for x in sp if not np.isnan(x)])
    jac = np.array([x for x in jac if not np.isnan(x)])
    return (float(sp.mean()), float(sp.std()), float(jac.mean()), len(sp)) if len(sp) else None


def md_matrix(mat, methods, present, title):
    out = [f"\n**{title}**\n"]
    hdr = "| | " + " | ".join(SHORT[m] for m in present) + " |"
    out.append(hdr)
    out.append("|" + "---|" * (len(present) + 1))
    for i, mi in enumerate(present):
        row = [SHORT[mi]]
        for j, mj in enumerate(present):
            if i == j:
                row.append("—")
            elif (mi, mj) in mat:
                row.append(f"{mat[(mi, mj)]:+.2f}")
            elif (mj, mi) in mat:
                row.append(f"{mat[(mj, mi)]:+.2f}")
            else:
                row.append("")
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


if __name__ == "__main__":
    lines = []
    for ds in ["meld", "iemocap"]:
        for model, src, tag in SOURCES:
            loaded = {m: load(src, ds, model, m) for m in METHODS}
            present = [m for m in METHODS if loaded[m] is not None]
            if len(present) < 2:
                continue
            n = len(loaded[present[0]][0])
            ids = loaded[present[0]][1]
            sp_mat, jac_mat = {}, {}
            for a, b in itertools.combinations(present, 2):
                r = pair_agreement(loaded[a], loaded[b], ids)
                if r:
                    sp_mat[(a, b)] = r[0]
                    jac_mat[(a, b)] = r[2]
            lines.append(f"\n### {ds.upper()} — {tag} (n={n})")
            lines.append(md_matrix(sp_mat, METHODS, present,
                                   "Spearman rank correlation (per-example mean)"))
            lines.append(md_matrix(jac_mat, METHODS, present,
                                   "Top-20% token overlap (Jaccard)"))
            # console preview of the two headline numbers
            for pair in [("gradshap", "lime"), ("gradshap", "optimus"), ("lime", "optimus")]:
                k = pair if pair in sp_mat else pair[::-1]
                if k in sp_mat:
                    print(f"  [{ds}/{model}] {pair[0]}~{pair[1]}: "
                          f"Spearman {sp_mat[k]:+.3f} | Jaccard@20% {jac_mat[k]:.3f}")
    out = RESULTS_DIR / "agreement" / "agreement_matrix.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"\n[saved] {out}")
