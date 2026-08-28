#!/usr/bin/env python3
"""Cross-corpus zero-shot metrics from SAVED predictions (reviewer note f) -- no models needed.

Companion to cross_dataset_eval.py. That script loads the trained models and runs inference (needs
the ~2.7 GB checkpoints + a GPU); it can also dump the per-example (gold, pred) with --save_preds.
This script just recomputes accuracy / weighted-F1 / macro-F1 from that tiny predictions CSV, so the
cross-dataset result reproduces in the lightweight, CPU-only Code Ocean run WITHOUT the models --
exactly like the figure scripts replay saved attributions.

    python Explainability/cross_dataset_from_preds.py
"""
import argparse
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from erc_paths import CHECKPOINTS_DIR, RESULTS_DIR

import numpy as np
import pandas as pd

SHARED = ["anger", "happy", "neutral", "sadness"]


def _f1_scores(gold, pred, labels):
    """Accuracy, weighted-F1, macro-F1 without sklearn (keeps the lightweight capsule dep-free)."""
    gold = np.asarray(gold); pred = np.asarray(pred)
    acc = float((gold == pred).mean())
    f1s, support = [], []
    for c in labels:
        tp = int(((pred == c) & (gold == c)).sum())
        fp = int(((pred == c) & (gold != c)).sum())
        fn = int(((pred != c) & (gold == c)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1); support.append(int((gold == c).sum()))
    f1s = np.array(f1s); support = np.array(support)
    macro = float(f1s.mean())
    weighted = float((f1s * support).sum() / support.sum()) if support.sum() else 0.0
    return acc, weighted, macro


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default=str(RESULTS_DIR / "cross_dataset"))
    args = ap.parse_args()

    src = CHECKPOINTS_DIR / "cross_dataset_preds" / f"cross_dataset_preds_seed{args.seed}.csv"
    if not src.exists():
        print(f"[cross-dataset] no saved predictions at {src} -- run cross_dataset_eval.py --save_preds "
              "first (needs the models), or add that file to /data.")
        return

    df = pd.read_csv(src)
    print(f"Shared classes: {SHARED}  (joy<->happiness merged)\n")
    rows = []
    for (target, role, model), g in df.groupby(["target", "role", "model"], sort=False):
        acc, wf1, mf1 = _f1_scores(g["gold"].tolist(), g["pred"].tolist(), SHARED)
        rows.append({"target": target, "n": len(g), "role": role, "model": model,
                     "accuracy": acc, "weighted_f1": wf1, "macro_f1": mf1})
        print(f"  target={target:8s} n={len(g):4d} | {role:9s} model={model:8s}"
              f" -> acc={acc*100:5.1f}  weightedF1={wf1*100:5.1f}  macroF1={mf1*100:5.1f}")

    out_dir = pathlib.Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"cross_dataset_seed{args.seed}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
