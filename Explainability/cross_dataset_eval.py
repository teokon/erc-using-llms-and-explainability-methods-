#!/usr/bin/env python3
"""Cross-corpus zero-shot generalization eval (reviewer note f).

Evaluates each trained context-aware model on the OTHER corpus with no fine-tuning on the target.
MELD and IEMOCAP share four emotions -- anger, happiness/joy, neutral, sadness (MELD's `joy` and
IEMOCAP's `happiness` are merged). For a target corpus we keep the test rows whose gold label is one
of these, restrict the source model's logits to the shared classes, and report accuracy + macro-F1.
The target's own model on the same subset is the in-domain upper bound.

No new training: reuses the seed-42 BEST checkpoints under $ERC_CHECKPOINTS. Runs on GPU if available.

    GPU=0 python Explainability/cross_dataset_eval.py
    GPU=0 python Explainability/cross_dataset_eval.py --seed 43   # use a different seed's checkpoints
"""
import argparse
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from erc_paths import CHECKPOINTS_DIR, RESULTS_DIR, pick_gpu
pick_gpu()

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# gold label (either corpus) -> shared canonical class; labels not present are dropped
GOLD_TO_CANON = {"anger": "anger", "neutral": "neutral", "sadness": "sadness",
                 "joy": "happy", "happiness": "happy"}
SHARED = ["anger", "happy", "neutral", "sadness"]


def model_paths(seed):
    """(checkpoint dir, target test CSV) for each corpus at a given seed."""
    return {
        "meld": (CHECKPOINTS_DIR / "emoberta_meld_large" / f"roberta_meld_final_seed{seed}_BEST",
                 CHECKPOINTS_DIR / "emoberta_meld_large" / "test_constructed_context_targetSpeaker.csv"),
        "iemocap": (CHECKPOINTS_DIR / "emoberta_iemocap_large_both" / f"roberta_iemocap_both_seed{seed}_BEST",
                    CHECKPOINTS_DIR / "emoberta_iemocap_large_both" / "test_constructed_both.csv"),
    }


def load_model(ckpt):
    """Model + a map {output_index -> canonical class or None} for restricting to shared classes."""
    model = AutoModelForSequenceClassification.from_pretrained(str(ckpt)).to(DEVICE).eval()
    id2canon = {int(i): GOLD_TO_CANON.get(name) for i, name in model.config.id2label.items()}
    return model, id2canon


@torch.inference_mode()
def predict_canonical(model, id2canon, texts, batch=32, max_len=512):
    """Predict a shared canonical label per text by taking argmax over ONLY the shared-class logits."""
    tok = predict_canonical.tok
    idxs = [i for i, c in id2canon.items() if c is not None]     # output positions that are shared
    canon = [id2canon[i] for i in idxs]
    preds = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_len).to(DEVICE)
        logits = model(**enc).logits[:, idxs]
        preds += [canon[j] for j in logits.argmax(-1).cpu().numpy()]
    return preds


def target_subset(test_csv):
    """Target test rows whose gold maps to a shared class: (context texts, canonical gold labels)."""
    df = pd.read_csv(test_csv)
    df["canon"] = df["label"].astype(str).map(GOLD_TO_CANON)
    df = df[df["canon"].isin(SHARED)]
    return df["context_text_raw"].astype(str).tolist(), df["canon"].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42, help="which seed's BEST checkpoints to use")
    ap.add_argument("--out_dir", default=str(RESULTS_DIR / "cross_dataset"))
    ap.add_argument("--save_preds", action="store_true",
                    help="also save per-example (gold, pred) so the metrics can be recomputed later "
                         "WITHOUT the models (a tiny artifact for the Code Ocean lightweight run)")
    args = ap.parse_args()
    pred_rows = []

    predict_canonical.tok = AutoTokenizer.from_pretrained(
        "roberta-large", use_fast=True, add_prefix_space=True)
    paths = model_paths(args.seed)
    models = {name: load_model(p[0]) for name, p in paths.items()}

    rows = []
    print(f"Shared classes: {SHARED}  (joy<->happiness merged)\n")
    for target in ["meld", "iemocap"]:
        texts, gold = target_subset(paths[target][1])
        source = "iemocap" if target == "meld" else "meld"
        n = len(gold)
        for role, which in [("in_domain", target), ("zero_shot", source)]:
            model, id2canon = models[which]
            pred = predict_canonical(model, id2canon, texts)
            acc = accuracy_score(gold, pred)
            wf1 = f1_score(gold, pred, average="weighted", labels=SHARED)
            mf1 = f1_score(gold, pred, average="macro", labels=SHARED)
            rows.append({"target": target, "n": n, "role": role, "model": which,
                         "accuracy": acc, "weighted_f1": wf1, "macro_f1": mf1})
            if args.save_preds:
                pred_rows += [{"target": target, "role": role, "model": which, "gold": g, "pred": p}
                              for g, p in zip(gold, pred)]
            print(f"  target={target:8s} n={n:4d} | {role:9s} model={which:8s}"
                  f" -> acc={acc*100:5.1f}  weightedF1={wf1*100:5.1f}  macroF1={mf1*100:5.1f}")
        print()

    out_dir = pathlib.Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"cross_dataset_seed{args.seed}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[saved] {out}")

    if args.save_preds:
        # tiny artifact (predictions only) so the metrics can be recomputed model-free
        pred_dir = CHECKPOINTS_DIR / "cross_dataset_preds"; pred_dir.mkdir(parents=True, exist_ok=True)
        pp = pred_dir / f"cross_dataset_preds_seed{args.seed}.csv"
        pd.DataFrame(pred_rows).to_csv(pp, index=False)
        print(f"[saved] {pp}  ({len(pred_rows)} rows) -- feeds cross_dataset_from_preds.py")


if __name__ == "__main__":
    main()
