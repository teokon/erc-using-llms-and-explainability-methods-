#!/usr/bin/env python3
"""Explanation-agreement analysis (reviewer request) for LIME vs GradSHAP on the
context-aware RoBERTa-large EmoBERTa models (MELD / IEMOCAP).

Quantifies what the reviewer asked for:
  1. Rank correlation      -- Spearman rho / Kendall tau between the two methods'
                              per-token attributions (toward the predicted class).
  2. Top-k overlap         -- Jaccard overlap of each method's top-k tokens (k=5,10,10%).
  3. Stability across seeds-- per method, mean pairwise agreement of attributions
                              across the 5 fine-tuning seeds (42-46).
  4. Explanation variance  -- per-example attribution concentration (Gini of |score|)
     across examples/classes  and its spread, reported overall and per predicted class.

Reuses the (special-token-correct) explainers from faithfulness_eval.py.

Run (any environment with captum + lime installed):
    GPU=0 python -u explanation_agreement.py \
        --dataset meld --stability_n 500
"""
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from erc_paths import pick_gpu
pick_gpu()  # pin ONE GPU before importing torch ($GPU > single CUDA_VISIBLE_DEVICES > "0")

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, kendalltau
from transformers import AutoModelForSequenceClassification

import faithfulness_eval as FE   # scores_gradshap, scores_lime, load_model_tok, DATASETS, CK, DEVICE

SEED_CKPT = {
    "meld":    lambda s: f"{FE.CK}/emoberta_meld_large/roberta_meld_final_seed{s}_BEST",
    "iemocap": lambda s: f"{FE.CK}/emoberta_iemocap_large_both/roberta_iemocap_both_seed{s}_BEST",
}
SEEDS = [42, 43, 44, 45, 46]


def load_model(ckpt):
    return AutoModelForSequenceClassification.from_pretrained(ckpt).to(FE.DEVICE).eval()


# ---------- pairwise agreement helpers ----------
def rank_corr(a, b):
    """Spearman & Kendall between two attribution vectors; nan if degenerate."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return np.nan, np.nan
    rho = spearmanr(a, b).correlation
    tau = kendalltau(a, b).correlation
    return rho, tau


def topk_jaccard(a, b, k):
    """Jaccard overlap of the top-k tokens (by signed attribution, i.e. most
    supportive of the predicted class) of two attribution vectors."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    k = max(1, min(k, len(a)))
    ta = set(np.argsort(a)[::-1][:k].tolist())
    tb = set(np.argsort(b)[::-1][:k].tolist())
    u = ta | tb
    return (len(ta & tb) / len(u)) if u else np.nan


def gini(x):
    """Gini concentration of |attribution| in [0,1]; higher => a few tokens dominate."""
    x = np.abs(np.asarray(x, float))
    s = x.sum()
    if s == 0 or len(x) < 2:
        return np.nan
    x = np.sort(x)
    n = len(x)
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * x) / (n * s)) - (n + 1) / n)


def content_mask(ids, special_ids):
    """Positions of real WORD tokens, excluding the embedded <s>/</s></s> segment
    markers that the EmoBERTa 3-segment text contains (they tokenize to special IDs)."""
    return [p for p, tid in enumerate(ids) if tid not in special_ids]


def summ(vals):
    v = np.asarray([x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))], float)
    return (float(v.mean()), float(v.std()), int(len(v))) if len(v) else (float("nan"), float("nan"), 0)


RNG_SEED = 1234   # fixed explainer RNG so cross-seed differences reflect the MODEL, not sampling


def compute_attrib(model, tok, texts, ids_list):
    """Return dict method -> list of per-example content-token attribution arrays.
    Explainer randomness (GradShap sampling, LIME perturbations) is pinned to a fixed
    seed so that, across model seeds, the only source of variation is the model itself."""
    torch.manual_seed(RNG_SEED); np.random.seed(RNG_SEED)
    t0 = time.time()
    gs = FE.scores_gradshap(model, tok, texts, ids_list)
    t1 = time.time()
    torch.manual_seed(RNG_SEED); np.random.seed(RNG_SEED)
    lm = FE.scores_lime(model, tok, texts, ids_list, num_samples=100)
    t2 = time.time()
    print(f"    gradshap {t1-t0:.0f}s | lime {t2-t1:.0f}s", flush=True)
    return {"gradshap": gs, "lime": lm}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(FE.DATASETS))
    ap.add_argument("--agree_n", type=int, default=0, help="0=full corpus for LIME<->GradSHAP agreement & variance")
    ap.add_argument("--stability_n", type=int, default=500, help="subset size for cross-seed stability")
    ap.add_argument("--out_dir", default=f"{FE.CK}/agreement")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cfg = FE.DATASETS[args.dataset]
    tok = FE.load_model_tok(cfg["ckpt"])[1]  # tokenizer identical across seeds
    SPECIAL = set(tok.all_special_ids)

    df = pd.read_csv(cfg["test_csv"])
    texts_all = df["context_text_raw"].astype(str).tolist()
    labels_all = df["label"].astype(str).tolist()
    ids_all = [tok(t, add_special_tokens=False, truncation=True, max_length=510)["input_ids"] for t in texts_all]
    N = len(ids_all)
    print(f"[agreement] {args.dataset}: {N} test examples")

    # ===== seed-42 attributions on the agreement set (full corpus unless --agree_n) =====
    a_idx = list(range(N)) if args.agree_n == 0 else list(range(min(args.agree_n, N)))
    print(f"[phase 1] LIME<->GradSHAP agreement + variance on {len(a_idx)} examples (seed 42)")
    m42 = load_model(SEED_CKPT[args.dataset](42))
    attr42 = compute_attrib(m42, tok, [texts_all[i] for i in a_idx], [ids_all[i] for i in a_idx])

    rows_ex = []
    for j, i in enumerate(a_idx):
        keep = content_mask(ids_all[i], SPECIAL)
        gs = np.asarray(attr42["gradshap"][j], float)[keep]
        lm = np.asarray(attr42["lime"][j], float)[keep]
        if len(gs) < 3:
            continue
        rho, tau = rank_corr(gs, lm)
        rows_ex.append({
            "label": labels_all[i],
            "spearman": rho, "kendall": tau,
            "jaccard@5": topk_jaccard(gs, lm, 5),
            "jaccard@10": topk_jaccard(gs, lm, 10),
            "jaccard@10pct": topk_jaccard(gs, lm, max(1, round(0.10 * len(gs)))),
            "gini_gradshap": gini(gs), "gini_lime": gini(lm),
        })
    ex = pd.DataFrame(rows_ex)
    ex.to_csv(out_dir / f"agreement_{args.dataset}_perexample.csv", index=False)

    report = {"dataset": args.dataset, "n_agreement": len(ex)}
    print(f"\n===== LIME <-> GradSHAP agreement ({args.dataset}, n={len(ex)}) =====")
    for m in ["spearman", "kendall", "jaccard@5", "jaccard@10", "jaccard@10pct"]:
        mu, sd, n = summ(ex[m])
        report[m] = {"mean": mu, "std": sd, "n": n}
        print(f"  {m:14s}: {mu:.3f} ± {sd:.3f}")

    # variance across examples & classes: concentration (Gini) per method, per class
    print(f"\n===== Explanation concentration (Gini of |attr|), per predicted class =====")
    var_rows = []
    for meth in ["gradshap", "lime"]:
        col = f"gini_{meth}"
        mu, sd, n = summ(ex[col])
        print(f"  [{meth}] overall Gini: {mu:.3f} ± {sd:.3f}  (std across examples = explanation variance)")
        var_rows.append({"method": meth, "class": "ALL", "gini_mean": mu, "gini_std": sd, "n": n})
        for c, g in ex.groupby("label"):
            cmu, csd, cn = summ(g[col])
            var_rows.append({"method": meth, "class": c, "gini_mean": cmu, "gini_std": csd, "n": cn})
    pd.DataFrame(var_rows).to_csv(out_dir / f"agreement_{args.dataset}_variance_byclass.csv", index=False)
    report["variance_byclass_csv"] = f"agreement_{args.dataset}_variance_byclass.csv"

    del m42; torch.cuda.empty_cache()

    # ===== cross-seed stability on a subset =====
    s_idx = list(range(min(args.stability_n, N)))
    print(f"\n[phase 2] cross-seed stability on {len(s_idx)} examples x {len(SEEDS)} seeds")
    per_seed = {}   # seed -> {method -> [attr arrays]}
    for s in SEEDS:
        if s == 42:  # reuse where possible (subset of the agreement set if it covers s_idx)
            if args.agree_n == 0 or args.agree_n >= len(s_idx):
                per_seed[s] = {mth: [attr42[mth][k] for k in s_idx] for mth in ["gradshap", "lime"]}
                print(f"    seed 42: reused")
                continue
        print(f"    seed {s}: computing ...")
        ms = load_model(SEED_CKPT[args.dataset](s))
        per_seed[s] = compute_attrib(ms, tok, [texts_all[i] for i in s_idx], [ids_all[i] for i in s_idx])
        del ms; torch.cuda.empty_cache()

    print(f"\n===== Attribution stability across seeds ({args.dataset}, n={len(s_idx)}) =====")
    for meth in ["gradshap", "lime"]:
        sp_ex, jac_ex = [], []
        for k in range(len(s_idx)):
            keep = content_mask(ids_all[s_idx[k]], SPECIAL)
            arrs = [np.asarray(per_seed[s][meth][k], float)[keep] for s in SEEDS]
            if len(arrs[0]) < 3:
                continue
            sps, jcs = [], []
            for a in range(len(SEEDS)):
                for b in range(a + 1, len(SEEDS)):
                    r, _ = rank_corr(arrs[a], arrs[b]); sps.append(r)
                    jcs.append(topk_jaccard(arrs[a], arrs[b], 10))
            sp_ex.append(np.nanmean(sps)); jac_ex.append(np.nanmean(jcs))
        smu, ssd, sn = summ(sp_ex)
        jmu, jsd, jn = summ(jac_ex)
        report[f"stability_{meth}"] = {"spearman": {"mean": smu, "std": ssd},
                                       "jaccard@10": {"mean": jmu, "std": jsd}, "n": sn}
        print(f"  [{meth}] cross-seed Spearman: {smu:.3f} ± {ssd:.3f} | Jaccard@10: {jmu:.3f} ± {jsd:.3f}")

    (out_dir / f"agreement_{args.dataset}_summary.json").write_text(json.dumps(report, indent=2))
    print(f"\n[saved] {out_dir}/agreement_{args.dataset}_summary.json")


if __name__ == "__main__":
    main()
