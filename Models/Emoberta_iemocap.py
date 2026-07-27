#!/usr/bin/env python3
"""
IEMOCAP (6-way) / RoBERTa-large — BOTH-CONTEXT (past + future) baseline.

EmoBERTa-faithful rebuild from RAW IEMOCAP CSVs:
  - </s></s> (double [SEP]) bracketing the target, 3-segment
        <s> [past] </s></s> current </s></s> [future] </s>
  - [CLS] + sequence + [EOS]  (fixes the earlier single-</s>, no-EOS version that
    loaded pre-built CSVs; that older run lives in emoberta_iemocap_large/)
  - EmoBERTa NAME_MAP speaker names, 6-class filter
  - context builder shared via iemocap_context.py (mode="both")

Run (inside tmux, single idle/assigned GPU set below):
    cd Models
    python Emoberta_iemocap.py \
        2>&1 | tee checkpoints/emoberta_iemocap_large_both_run.log
"""

import os
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from erc_paths import DATA_DIR, CHECKPOINTS_DIR, pick_gpu
pick_gpu()  # pin ONE GPU before importing torch ($GPU > single CUDA_VISIBLE_DEVICES > "0")

import time
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import optuna

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, TrainerCallback, set_seed, DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, f1_score

import iemocap_context as ic
from iemocap_context import (
    LABELS, label2id, id2label,
    SPEAKER_COL, TEXT_COL, LABEL_COL,
)
import repro_utils

try:
    display  # type: ignore[name-defined]
except NameError:
    def display(obj):
        print(obj)

_run_t0 = time.time()

# =====================
# CONFIG (IEMOCAP 6-way, BOTH context)
# =====================
CONTEXT_MODE = "both"

_DATA = DATA_DIR / "IEMOCAP"
TRAIN_CSV = f"{_DATA}/iemocap_train.csv"
VAL_CSV   = f"{_DATA}/iemocap_val.csv"
TEST_CSV  = f"{_DATA}/iemocap_test.csv"

OUTPUT_DIR = CHECKPOINTS_DIR / "emoberta_iemocap_large_both"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_BASE = "roberta-large"

WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.20
LR_SCHED = "linear"
EPOCHS = 5           # Optuna search epochs (matches original IEMOCAP notebook)
FINAL_EPOCHS = 7     # final per-seed training epochs

N_TRIALS = 5
LR_LOW, LR_HIGH = 1e-6, 1e-4

MAX_LEN = 512
BATCH_TRAIN = 8
BATCH_EVAL  = 16
GRAD_ACCUM  = 1

SEED = 42
SEEDS_FINAL = [42, 43, 44, 45, 46]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", DEVICE, "| MODEL_BASE:", MODEL_BASE, "| CONTEXT_MODE:", CONTEXT_MODE)
print("visible GPU count:", torch.cuda.device_count())
for p in [TRAIN_CSV, VAL_CSV, TEST_CSV]:
    if not Path(p).exists():
        print(f"⚠️ Missing file: {p}")

# =====================
# Tokenizer, collator, metrics
# =====================
tok = AutoTokenizer.from_pretrained(MODEL_BASE, use_fast=True)
collator = DataCollatorWithPadding(tokenizer=tok)

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    if isinstance(preds, (tuple, list)):
        preds = preds[0]
    y_pred = np.argmax(preds, axis=1)
    return {
        "acc": accuracy_score(labels, y_pred),
        "weighted_f1": f1_score(labels, y_pred, average="weighted"),
        "macro_f1": f1_score(labels, y_pred, average="macro"),
    }

# =====================
# Build datasets from raw IEMOCAP (mode="both")
# =====================
train_df = pd.read_csv(TRAIN_CSV)
val_df   = pd.read_csv(VAL_CSV)
test_df  = pd.read_csv(TEST_CSV)
print("Raw rows:", len(train_df), len(val_df), len(test_df))

train_ds_full = ic.build_iemocap_context(train_df, tok, mode=CONTEXT_MODE, max_length=MAX_LEN, debug_n=3)
val_ds_full   = ic.build_iemocap_context(val_df,   tok, mode=CONTEXT_MODE, max_length=MAX_LEN, debug_n=1)
test_ds_full  = ic.build_iemocap_context(test_df,  tok, mode=CONTEXT_MODE, max_length=MAX_LEN, debug_n=1)
print("Sizes:", len(train_ds_full), len(val_ds_full), len(test_ds_full))

# ---- save constructed context CSVs (inspectable) ----
ic.save_constructed_csv(train_ds_full, OUTPUT_DIR / f"train_constructed_{CONTEXT_MODE}.csv", id2label=id2label)
ic.save_constructed_csv(val_ds_full,   OUTPUT_DIR / f"val_constructed_{CONTEXT_MODE}.csv",   id2label=id2label)
ic.save_constructed_csv(test_ds_full,  OUTPUT_DIR / f"test_constructed_{CONTEXT_MODE}.csv",  id2label=id2label)

# =====================
# Optuna LR search
# =====================
def objective(trial):
    set_seed(SEED)
    lr = trial.suggest_float("lr", LR_LOW, LR_HIGH, log=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_BASE, num_labels=len(LABELS), label2id=label2id, id2label=id2label,
    ).to(DEVICE)
    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / f"optuna_lr_trial_{trial.number}"),
        eval_strategy="epoch", save_strategy="no",
        learning_rate=lr, num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_TRAIN, per_device_eval_batch_size=BATCH_EVAL,
        gradient_accumulation_steps=GRAD_ACCUM,
        weight_decay=WEIGHT_DECAY, warmup_ratio=WARMUP_RATIO, lr_scheduler_type=LR_SCHED,
        fp16=torch.cuda.is_available(), report_to="none", seed=SEED, logging_steps=200,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds_full, eval_dataset=val_ds_full,
                      data_collator=collator, tokenizer=tok, compute_metrics=compute_metrics)
    trainer.train()
    return trainer.evaluate(val_ds_full)["eval_loss"]

# Resume-aware: reuse a previously found best LR (survives reboots), else search.
_bestlr_file = OUTPUT_DIR / "best_lr.txt"
if _bestlr_file.exists():
    best_lr = float(_bestlr_file.read_text().strip())
    _optuna_search_sec = 0.0
    print(f"[resume] reusing best_lr={best_lr:g} from {_bestlr_file} (skipping Optuna search)")
else:
    _t_search0 = time.time()
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS)
    _optuna_search_sec = time.time() - _t_search0
    best_lr = study.best_params["lr"]
    _bestlr_file.write_text(str(best_lr))
    print("Best lr:", best_lr, "| Best val loss:", study.best_value)

# =====================
# Final training across seeds
# =====================
class SaveByEpochCallback(TrainerCallback):
    def __init__(self, out_root, tokenizer):
        self.out_root = str(out_root); self.tokenizer = tokenizer
        os.makedirs(self.out_root, exist_ok=True)
    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        ep_i = int(round(state.epoch)) if state.epoch is not None else 0
        save_dir = os.path.join(self.out_root, f"epoch_{ep_i:02d}")
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir); self.tokenizer.save_pretrained(save_dir)
        print(f"✅ Saved epoch checkpoint to: {save_dir}")
        return control

rows = []
_t_train0 = time.time()
for seed in SEEDS_FINAL:
    print("\n" + "=" * 20, "SEED", seed, f"[{CONTEXT_MODE.upper()}]", "=" * 20)
    out_dir = str(OUTPUT_DIR / f"roberta_iemocap_{CONTEXT_MODE}_seed{seed}")
    best_dir = f"{out_dir}_BEST"

    # Resume: if this seed's BEST already exists, re-evaluate it on test and skip training.
    if os.path.exists(os.path.join(best_dir, "model.safetensors")):
        print(f"[resume] seed {seed} already complete -> re-evaluating {best_dir} (skip training)")
        model = AutoModelForSequenceClassification.from_pretrained(best_dir).to(DEVICE)
        ev_trainer = Trainer(
            model=model,
            args=TrainingArguments(output_dir=str(OUTPUT_DIR / "_tmp_eval"),
                                   per_device_eval_batch_size=BATCH_EVAL,
                                   fp16=torch.cuda.is_available(), report_to="none"),
            data_collator=collator, tokenizer=tok, compute_metrics=compute_metrics)
        test_metrics = ev_trainer.evaluate(test_ds_full)
        print("TEST:", test_metrics)
        rows.append({
            "seed": seed, "best_dir": best_dir,
            "test_acc": float(test_metrics["eval_acc"]),
            "test_weighted_f1": float(test_metrics["eval_weighted_f1"]),
            "test_macro_f1": float(test_metrics["eval_macro_f1"]),
        })
        continue

    set_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_BASE, num_labels=len(LABELS), label2id=label2id, id2label=id2label,
    ).to(DEVICE)
    epoch_root = str(OUTPUT_DIR / f"epoch_checkpoints_seed{seed}")
    if os.path.exists(epoch_root):
        shutil.rmtree(epoch_root)
    os.makedirs(epoch_root, exist_ok=True)
    args = TrainingArguments(
        output_dir=out_dir, eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="weighted_f1", greater_is_better=True,
        learning_rate=best_lr, num_train_epochs=FINAL_EPOCHS,
        per_device_train_batch_size=BATCH_TRAIN, per_device_eval_batch_size=BATCH_EVAL,
        gradient_accumulation_steps=GRAD_ACCUM,
        weight_decay=WEIGHT_DECAY, warmup_ratio=WARMUP_RATIO, lr_scheduler_type=LR_SCHED,
        fp16=torch.cuda.is_available(), report_to="none", seed=seed, logging_steps=200,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds_full, eval_dataset=val_ds_full,
                      data_collator=collator, tokenizer=tok, compute_metrics=compute_metrics,
                      callbacks=[SaveByEpochCallback(epoch_root, tok)])
    trainer.train()
    best_ckpt = trainer.state.best_model_checkpoint
    print("Best checkpoint:", best_ckpt)
    best_dir = f"{out_dir}_BEST"
    if os.path.exists(best_dir):
        shutil.rmtree(best_dir)
    shutil.copytree(best_ckpt, best_dir)
    tok.save_pretrained(best_dir)
    print("✅ Saved BEST folder to:", best_dir)
    test_metrics = trainer.evaluate(test_ds_full)
    print("TEST:", test_metrics)
    rows.append({
        "seed": seed, "best_dir": best_dir,
        "test_acc": float(test_metrics["eval_acc"]),
        "test_weighted_f1": float(test_metrics["eval_weighted_f1"]),
        "test_macro_f1": float(test_metrics["eval_macro_f1"]),
    })

df = pd.DataFrame(rows)
display(df)
mean_df = df[["test_acc", "test_weighted_f1", "test_macro_f1"]].mean().to_frame("mean")
std_df  = df[["test_acc", "test_weighted_f1", "test_macro_f1"]].std().to_frame("std")
print("\nMEAN:"); display(mean_df)
print("\nSTD:"); display(std_df)
df.to_csv(OUTPUT_DIR / "results_per_seed.csv", index=False)
mean_df.join(std_df).to_csv(OUTPUT_DIR / "results_mean_std.csv")
print("\n[saved]", OUTPUT_DIR / "results_per_seed.csv")

# =====================
# Reproducibility report
# =====================
_final_training_sec = time.time() - _t_train0
_total_sec = time.time() - _run_t0
repro_utils.save_repro_report(
    OUTPUT_DIR,
    config={
        "run_name": f"IEMOCAP roberta-large {CONTEXT_MODE.upper()}",
        "dataset": "IEMOCAP (6-way)", "context_mode": CONTEXT_MODE, "model": MODEL_BASE,
        "target_separator": "</s></s> (double, EmoBERTa [SEP])",
        "max_len": MAX_LEN, "optuna_epochs": EPOCHS, "final_epochs": FINAL_EPOCHS,
        "batch_train": BATCH_TRAIN, "batch_eval": BATCH_EVAL, "grad_accum": GRAD_ACCUM,
        "weight_decay": WEIGHT_DECAY, "warmup_ratio": WARMUP_RATIO, "lr_scheduler": LR_SCHED,
        "optuna_n_trials": N_TRIALS, "optuna_lr_range": f"{LR_LOW:g} .. {LR_HIGH:g}",
        "best_lr": best_lr, "seeds": str(SEEDS_FINAL), "fp16": bool(torch.cuda.is_available()),
    },
    splits={
        "train": {"dataset": train_ds_full, "df": train_df},
        "val":   {"dataset": val_ds_full,   "df": val_df},
        "test":  {"dataset": test_ds_full,  "df": test_df},
    },
    tokenizer=tok, id2label=id2label, max_len=MAX_LEN,
    speaker_col=SPEAKER_COL, text_col=TEXT_COL, label_col=LABEL_COL, labels=LABELS,
    timings={"optuna_search_sec": _optuna_search_sec,
             "final_training_sec": _final_training_sec, "total_sec": _total_sec},
    sep_tokens_reserved=4,
)
