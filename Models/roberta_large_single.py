#!/usr/bin/env python3
"""Single-utterance fine-tuned RoBERTa-large (NO context, NO speaker prefix).

The missing middle model for the 3-way representation-geometry comparison:
    pretrained roberta-large  ->  single-utterance fine-tuned  ->  context-aware (EmoBERTa)

Recipe matches the existing single-utterance baselines
(Models/fine_tuned_{meld,iemocap}_distilbert_bert_roberta.ipynb):
    target utterance ONLY, no speaker prefix, 5 epochs, label smoothing 0.1,
    batch 16, MAX_LEN 256, weight_decay 0.01, warmup 0.06, best-on-weighted-F1,
    DataCollatorWithPadding.
The only change is the backbone: roberta-LARGE (so it is comparable to the
context-aware roberta-large model). LR defaults to 1e-5 -- roberta-large collapses at
the base model's 3e-5 -- override with LR=...

Data is derived from the already-built context CSVs (target = the segment between the
'</s></s>' markers, speaker prefix stripped), so splits/rows/labels are IDENTICAL to the
context-aware model; the ONLY difference is that context is removed.

Inputs:  the constructed context CSVs under checkpoints/emoberta_{meld,iemocap}_large*/
         (the target utterance is extracted from them, so splits/rows/labels match exactly).
Outputs: checkpoints/roberta_large_single_{dataset}/  (BEST checkpoint + test_results_seed*.json).

Run (choose dataset + GPU + seed via env vars):
    GPU=0 DATASET=meld    python Models/roberta_large_single.py
    GPU=0 DATASET=iemocap python Models/roberta_large_single.py
"""
import os
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from erc_paths import CHECKPOINTS_DIR, pick_gpu
pick_gpu()  # pin ONE GPU before importing torch ($GPU > single CUDA_VISIBLE_DEVICES > "0")

import json
import re
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, set_seed, DataCollatorWithPadding)

DATASET = os.environ.get("DATASET", "meld").lower()
SEED = int(os.environ.get("SEED", "42"))
CK = str(CHECKPOINTS_DIR)

SRC = {
    "meld": {"dir": f"{CK}/emoberta_meld_large",
             "train": "train_constructed_context_targetSpeaker.csv",
             "val": "val_constructed_context_targetSpeaker.csv",
             "test": "test_constructed_context_targetSpeaker.csv"},
    "iemocap": {"dir": f"{CK}/emoberta_iemocap_large_both",
                "train": "train_constructed_both.csv",
                "val": "val_constructed_both.csv",
                "test": "test_constructed_both.csv"},
}[DATASET]

OUTPUT_DIR = Path(f"{CK}/roberta_large_single_{DATASET}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_BASE = "roberta-large"
EPOCHS = 5                    # as in the single-utterance baseline notebooks
LR = float(os.environ.get("LR", "1e-5"))   # roberta-large: 3e-5 (the base LR) collapses
LABEL_SMOOTHING = 0.1
BATCH_TRAIN, BATCH_EVAL = 16, 32
MAX_LEN = 256
WEIGHT_DECAY, WARMUP_RATIO = 0.01, 0.06

SPEAKER_RE = re.compile(r"^[^:]{1,30}:\s*")


def target_utterance(ctx: str) -> str:
    """Target utterance between the two '</s></s>' markers, with the speaker prefix removed."""
    parts = str(ctx).split("</s></s>")
    t = parts[1].strip() if len(parts) == 3 else str(ctx).strip()
    return SPEAKER_RE.sub("", t, count=1).strip()


def load_split(name):
    df = pd.read_csv(Path(SRC["dir"]) / SRC[name])
    df["text"] = df["context_text_raw"].map(target_utterance)
    return df[df["text"].str.len() > 0].reset_index(drop=True)


class DS(torch.utils.data.Dataset):
    """Tokenized WITHOUT padding; DataCollatorWithPadding pads each batch."""
    def __init__(self, df, tok, label2id):
        self.enc = tok(df["text"].tolist(), truncation=True, max_length=MAX_LEN, padding=False)
        self.y = [label2id[l] for l in df["label"].astype(str)]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        d = {k: v[i] for k, v in self.enc.items()}
        d["labels"] = self.y[i]
        return d


def metrics(p):
    preds = p.predictions.argmax(-1)
    return {"accuracy": accuracy_score(p.label_ids, preds),
            "weighted_f1": f1_score(p.label_ids, preds, average="weighted"),
            "macro_f1": f1_score(p.label_ids, preds, average="macro")}


def main():
    print(f"[single] dataset={DATASET} seed={SEED} model={MODEL_BASE} LR={LR:.1e} epochs={EPOCHS}")
    tr, va, te = load_split("train"), load_split("val"), load_split("test")
    labels = sorted(pd.unique(pd.concat([tr["label"], va["label"], te["label"]]).astype(str)))
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    print(f"[single] {len(tr)} train / {len(va)} val / {len(te)} test | {len(labels)} classes: {labels}")
    print(f"[single] example (no speaker prefix): {tr['text'].iloc[0]!r}")

    tok = AutoTokenizer.from_pretrained(MODEL_BASE, use_fast=True)
    set_seed(SEED)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_BASE, num_labels=len(labels), id2label=id2label, label2id=label2id)

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / f"hf_seed{SEED}"),
        seed=SEED,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        per_device_train_batch_size=BATCH_TRAIN,
        per_device_eval_batch_size=BATCH_EVAL,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        label_smoothing_factor=LABEL_SMOOTHING,
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="weighted_f1",
        greater_is_better=True,
        fp16=True,
        logging_steps=100,
        report_to=[],
    )
    trainer = Trainer(
        model=model, args=args,
        train_dataset=DS(tr, tok, label2id),
        eval_dataset=DS(va, tok, label2id),
        compute_metrics=metrics,
        data_collator=DataCollatorWithPadding(tokenizer=tok, padding="longest"),
    )
    trainer.train()

    res = trainer.evaluate(DS(te, tok, label2id), metric_key_prefix="test")
    print(f"\n[single] TEST  weighted_f1={res['test_weighted_f1']:.4f}  "
          f"macro_f1={res['test_macro_f1']:.4f}  acc={res['test_accuracy']:.4f}")

    best = OUTPUT_DIR / f"roberta_single_{DATASET}_seed{SEED}_BEST"
    trainer.save_model(str(best))
    tok.save_pretrained(str(best))
    (OUTPUT_DIR / f"test_results_seed{SEED}.json").write_text(json.dumps(
        {k: float(v) for k, v in res.items() if isinstance(v, (int, float))}, indent=2))
    print(f"[single] saved BEST -> {best}")


if __name__ == "__main__":
    main()
