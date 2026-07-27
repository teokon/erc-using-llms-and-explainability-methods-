#!/usr/bin/env python3
"""
MELD / RoBERTa-large — FUTURE-ONLY context ablation.

Same EmoBERTa-style pipeline as Emoberta_meld.ipynb (both-context baseline),
but the context window is built from FUTURE utterances only:

    <s></s>  TARGET  </s>  [FUTURE context...]  </s>

i.e. only utterances *after* the target are appended (t+1, t+2, ...); no past
(left-side) utterances are prepended. The two </s> still bracket the target so
the model knows which utterance to classify. This is the surgical counterpart
of the past-only script — context DIRECTION is the only variable.

Run (inside tmux, single idle/assigned GPU set below):
    cd Models
    python Emoberta_meld_future_only.py \
        2>&1 | tee checkpoints/emoberta_meld_large_future_only_run.log

is required (loads transformers 4.57.1 from the env, not the
broken 5.12.1 in ~/.local).
"""

import os

# Pin to a single GPU BEFORE importing torch. SET to your idle/assigned index;
# do NOT reuse a GPU another job (e.g. the IEMOCAP run) is already using.
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

from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, TrainerCallback, set_seed, DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, f1_score

import repro_utils


# Plain-Python fallback for the notebook's IPython display(...)
try:
    display  # type: ignore[name-defined]
except NameError:
    def display(obj):
        print(obj)

_run_t0 = time.time()


# =====================
# CONFIG (MELD Ekman-7, FUTURE-ONLY)
# =====================
CONTEXT_MODE = "future_only"

_DATA = DATA_DIR / "Meld"
TRAIN_CSV = f"{_DATA}/train_sent_emo.csv"
VAL_CSV   = f"{_DATA}/dev_sent_emo.csv"
TEST_CSV  = f"{_DATA}/test_sent_emo.csv"

OUTPUT_DIR = CHECKPOINTS_DIR / "emoberta_meld_large_future_only"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DIALOG_COL  = "Dialogue_ID"
UTTID_COL   = "Utterance_ID"
SPEAKER_COL = "Speaker"
TEXT_COL    = "Utterance"
LABEL_COL   = "Emotion"

LABELS = ["neutral", "joy", "sadness", "anger", "surprise", "fear", "disgust"]
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for l, i in label2id.items()}

MODEL_BASE = "roberta-large"

WEIGHT_DECAY = 0.01
EPOCHS = 7
WARMUP_RATIO = 0.20
LR_SCHED = "linear"

# Optuna LR search — kept at 1e-6..1e-4 for consistency with the baseline/IEMOCAP.
# roberta-large can collapse to a single class above ~3e-5; if a seed collapses,
# narrow LR_HIGH to ~3e-5 in all MELD/IEMOCAP scripts together.
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
tok = AutoTokenizer.from_pretrained(MODEL_BASE, use_fast=True, add_prefix_space=True)
collator = DataCollatorWithPadding(tokenizer=tok)

def compute_metrics(eval_pred):
    logits, y_true = eval_pred
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    y_pred = np.argmax(logits, axis=1)
    return {
        "acc": accuracy_score(y_true, y_pred),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }


# =====================
# FUTURE-ONLY context builder
# =====================
# Identical to the baseline builder EXCEPT the left-side (past) expansion is
# removed: only future utterances (t+1, t+2, ...) are appended into the window.
def build_context_dataset_future_only(df, tokenizer, max_length=512, speaker_caps=True, debug_n=3):
    df = df.copy()
    df[TEXT_COL] = df[TEXT_COL].astype(str)
    df[SPEAKER_COL] = df[SPEAKER_COL].astype(str)
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip().str.lower()

    df[UTTID_COL] = pd.to_numeric(df[UTTID_COL], errors="coerce")
    df = df.dropna(subset=[UTTID_COL]).copy()
    df[UTTID_COL] = df[UTTID_COL].astype(int)

    df = df[df[LABEL_COL].isin(LABELS)].copy()
    df = df.sort_values([DIALOG_COL, UTTID_COL]).reset_index(drop=True)

    cls_id = tokenizer.cls_token_id  # <s>
    sep_id = tokenizer.sep_token_id  # </s>
    max_tokens = max_length - 2      # reserve CLS + final outer </s>

    space_ids = tokenizer.encode(" ", add_special_tokens=False)

    all_input_ids, all_attn, all_labels = [], [], []
    all_texts, all_dialog, all_turn = [], [], []
    dbg_printed = 0
    lengths = []

    for d_id, g in df.groupby(DIALOG_COL, sort=False):
        speakers = g[SPEAKER_COL].tolist()
        utts     = g[TEXT_COL].tolist()
        labs     = g[LABEL_COL].tolist()
        turns    = g[UTTID_COL].tolist()

        if speaker_caps:
            speakers = [s.upper() for s in speakers]

        seg_text = [f"{s}: {u}" for s, u in zip(speakers, utts)]
        seg_ids  = [tokenizer.encode(x, add_special_tokens=False) for x in seg_text]
        n = len(seg_ids)

        for t in range(n):
            target_ids = seg_ids[t][:]
            # EmoBERTa uses RoBERTa's [SEP] = two consecutive </s></s> to delimit
            # segments; the current utterance is bracketed by </s></s> on each side
            # (4 separator tokens total), generalizing RoBERTa to 3 segments.
            if len(target_ids) + 4 > max_tokens:
                target_ids = target_ids[: max(0, max_tokens - 4)]

            seq_ids  = [sep_id, sep_id] + target_ids + [sep_id, sep_id]
            seq_text = " </s></s> " + seg_text[t] + " </s></s> "

            # ----- expand FUTURE (right) only -----
            right = t + 1
            blocked_right = False
            while True:
                changed = False
                if right < n and not blocked_right:
                    cand = seg_ids[right]
                    need = len(cand) + (len(space_ids) if len(seq_ids) > 0 else 0)
                    if len(seq_ids) + need <= max_tokens:
                        seq_ids = seq_ids + space_ids + cand if space_ids else seq_ids + cand
                        seq_text = seq_text + " " + seg_text[right]
                        right += 1
                        changed = True
                    else:
                        blocked_right = True
                if not changed:
                    break

            input_ids = [cls_id] + seq_ids + [sep_id]
            input_ids = input_ids[:max_length]

            all_input_ids.append(input_ids)
            all_attn.append([1] * len(input_ids))
            all_labels.append(label2id[labs[t]])
            all_texts.append("<s> " + seq_text.strip() + " </s>")
            all_dialog.append(d_id)
            all_turn.append(turns[t])
            lengths.append(len(input_ids))

            if dbg_printed < debug_n:
                print("=" * 80)
                print(f"DEBUG {dbg_printed+1} [FUTURE-ONLY] | dialog={d_id} | uttid={turns[t]} | label={labs[t]}")
                print("RAW (repr):", repr(all_texts[-1][:600]))
                print("DECODED:", tokenizer.decode(input_ids[:120], skip_special_tokens=False))
                dbg_printed += 1

    print("\nToken length stats:",
          f"min={int(np.min(lengths))}, mean={float(np.mean(lengths)):.1f}, max={int(np.max(lengths))}, n={len(lengths)}")

    return Dataset.from_dict({
        "dialogue_id": all_dialog,
        "utterance_id": all_turn,
        "context_text_raw": all_texts,
        "input_ids": all_input_ids,
        "attention_mask": all_attn,
        "labels": all_labels,
    })


def save_constructed_csv(ds, out_csv, id2label=None):
    d = ds.to_dict()
    df_out = pd.DataFrame({
        "dialogue_id": d["dialogue_id"],
        "utterance_id": d["utterance_id"],
        "label_id": d["labels"],
        "label": [id2label.get(int(x), str(x)) if isinstance(id2label, dict) else str(x) for x in d["labels"]],
        "context_text_raw": d["context_text_raw"],
    })
    df_out.to_csv(out_csv, index=False)
    print("✅ Saved:", out_csv, "| rows:", len(df_out))


# =====================
# Load + build datasets
# =====================
train_df = pd.read_csv(TRAIN_CSV)
val_df   = pd.read_csv(VAL_CSV)
test_df  = pd.read_csv(TEST_CSV)
print("Raw rows:", len(train_df), len(val_df), len(test_df))

train_ds_full = build_context_dataset_future_only(train_df, tok, max_length=MAX_LEN, debug_n=3)
val_ds_full   = build_context_dataset_future_only(val_df,   tok, max_length=MAX_LEN, debug_n=1)
test_ds_full  = build_context_dataset_future_only(test_df,  tok, max_length=MAX_LEN, debug_n=1)
print("Sizes:", len(train_ds_full), len(val_ds_full), len(test_ds_full))

save_constructed_csv(train_ds_full, OUTPUT_DIR / "train_constructed_future_only.csv", id2label=id2label)
save_constructed_csv(val_ds_full,   OUTPUT_DIR / "val_constructed_future_only.csv",   id2label=id2label)
save_constructed_csv(test_ds_full,  OUTPUT_DIR / "test_constructed_future_only.csv",  id2label=id2label)


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
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=lr,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_TRAIN,
        per_device_eval_batch_size=BATCH_EVAL,
        gradient_accumulation_steps=GRAD_ACCUM,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type=LR_SCHED,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=SEED,
        logging_steps=200,
    )
    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds_full, eval_dataset=val_ds_full,
        data_collator=collator, tokenizer=tok, compute_metrics=compute_metrics,
    )
    trainer.train()
    out = trainer.evaluate(val_ds_full)
    return out["eval_loss"]


_t_search0 = time.time()
study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=SEED),
)
study.optimize(objective, n_trials=N_TRIALS)
_optuna_search_sec = time.time() - _t_search0
best_lr = study.best_params["lr"]
print("Best lr:", best_lr, "| Best val loss:", study.best_value)


# =====================
# Final training across seeds
# =====================
class SaveByEpochCallback(TrainerCallback):
    def __init__(self, out_root, tokenizer):
        self.out_root = str(out_root)
        self.tokenizer = tokenizer
        os.makedirs(self.out_root, exist_ok=True)

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        ep_i = int(round(state.epoch)) if state.epoch is not None else 0
        save_dir = os.path.join(self.out_root, f"epoch_{ep_i:02d}")
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        print(f"✅ Saved epoch checkpoint to: {save_dir}")
        return control


rows = []
_t_train0 = time.time()
for seed in SEEDS_FINAL:
    print("\n" + "=" * 20, "SEED", seed, "[FUTURE-ONLY]", "=" * 20)
    set_seed(seed)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_BASE, num_labels=len(LABELS), label2id=label2id, id2label=id2label,
    ).to(DEVICE)

    out_dir = str(OUTPUT_DIR / f"roberta_meld_future_only_seed{seed}")
    epoch_root = str(OUTPUT_DIR / f"epoch_checkpoints_seed{seed}")
    if os.path.exists(epoch_root):
        shutil.rmtree(epoch_root)
    os.makedirs(epoch_root, exist_ok=True)

    args = TrainingArguments(
        output_dir=out_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="weighted_f1",
        greater_is_better=True,
        learning_rate=best_lr,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_TRAIN,
        per_device_eval_batch_size=BATCH_EVAL,
        gradient_accumulation_steps=GRAD_ACCUM,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type=LR_SCHED,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=seed,
        logging_steps=200,
    )
    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds_full, eval_dataset=val_ds_full,
        data_collator=collator, tokenizer=tok, compute_metrics=compute_metrics,
        callbacks=[SaveByEpochCallback(epoch_root, tok)],
    )
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
        "seed": seed,
        "best_dir": best_dir,
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
print("[saved]", OUTPUT_DIR / "results_mean_std.csv")


# =====================
# Reproducibility report (preprocessing / context length / truncation /
# hardware / runtime) -> OUTPUT_DIR/repro_report.{json,md}
# =====================
_final_training_sec = time.time() - _t_train0
_total_sec = time.time() - _run_t0
repro_utils.save_repro_report(
    OUTPUT_DIR,
    config={
        "run_name": "MELD roberta-large FUTURE-ONLY",
        "dataset": "MELD (Ekman-7)",
        "context_mode": CONTEXT_MODE,
        "model": MODEL_BASE,
        "target_separator": "</s></s> (double, EmoBERTa [SEP])",
        "max_len": MAX_LEN,
        "epochs": EPOCHS,
        "batch_train": BATCH_TRAIN,
        "batch_eval": BATCH_EVAL,
        "grad_accum": GRAD_ACCUM,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "lr_scheduler": LR_SCHED,
        "optuna_n_trials": N_TRIALS,
        "optuna_lr_range": f"{LR_LOW:g} .. {LR_HIGH:g}",
        "best_lr": best_lr,
        "seeds": str(SEEDS_FINAL),
        "fp16": bool(torch.cuda.is_available()),
    },
    splits={
        "train": {"dataset": train_ds_full, "df": train_df},
        "val":   {"dataset": val_ds_full,   "df": val_df},
        "test":  {"dataset": test_ds_full,  "df": test_df},
    },
    tokenizer=tok, id2label=id2label, max_len=MAX_LEN,
    speaker_col=SPEAKER_COL, text_col=TEXT_COL, label_col=LABEL_COL, labels=LABELS,
    timings={
        "optuna_search_sec": _optuna_search_sec,
        "final_training_sec": _final_training_sec,
        "total_sec": _total_sec,
    },
    sep_tokens_reserved=4,
)
