#!/usr/bin/env python3
"""Train the context-aware EmoBERTa model on MELD (RoBERTa-large, both-context).

This is the main MELD context-aware training script. It fine-tunes RoBERTa-large on
speaker-aware, context-constructed inputs and is the "EmoBERTa-style" model used throughout the
paper. Executable, top-to-bottom (run it directly; it is not meant to be imported).

What it does:
  - RoBERTa-large backbone.
  - EmoBERTa-faithful `</s></s>` (double [SEP]) bracketing of the target utterance
    (3-segment: past </s></s> current </s></s> future), with speaker-prefixed utterances.
  - Optuna LR search (seeded) + final 5-seed training, saving per-epoch and BEST checkpoints.
  - Writes results_per_seed.csv / results_mean_std.csv and the constructed context CSVs.
  - Emits repro_report.{json,md} (preprocessing / context-length / truncation / hardware /
    runtime) via repro_utils.py.

Inputs:  Datasets/Meld/{train,dev,test}_sent_emo.csv  (raw MELD).
Outputs: checkpoints/emoberta_meld_large/  (checkpoints, constructed CSVs, results, repro_report).

Run:
    GPU=0 python Models/Emoberta_meld.py
"""

# Plain-Python fallback for the notebook's IPython display(...)
try:
    display  # type: ignore[name-defined]
except NameError:
    def display(obj):
        print(obj)

# ==========================
# RoBERTa-large (MELD) fine-tuning — both-context baseline
# Tip: change MODEL_BASE to switch backbone/checkpoint
# ==========================

import os
import sys
import pathlib

# Repo root on sys.path so the shared path/device config is importable, then pin ONE GPU
# BEFORE importing torch. Choose the device with GPU=<idx> (default 0).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from erc_paths import DATA_DIR, CHECKPOINTS_DIR, pick_gpu
pick_gpu()

# On a clean env / Colab, install deps first:
# !pip -q install -U transformers datasets accelerate scikit-learn pandas optuna

import time
import shutil
import numpy as np
import pandas as pd
import torch
import optuna

from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, set_seed, DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, f1_score

import repro_utils  # reproducibility-report helper (same folder)

_run_t0 = time.time()
print("transformers loaded, visible GPU count:", torch.cuda.device_count())


# ==========================
# Config
# ==========================
from pathlib import Path

# Raw MELD CSVs (Datasets/Meld/).
_DATA = DATA_DIR / "Meld"
TRAIN_CSV = f"{_DATA}/train_sent_emo.csv"
VAL_CSV   = f"{_DATA}/dev_sent_emo.csv"
TEST_CSV  = f"{_DATA}/test_sent_emo.csv"

# Speaker tags on/off (reviewer note a: disentangle context from speaker tags).
# SPEAKER_TAGS=0 builds the SAME context windows but WITHOUT the "Speaker: " prefix, so the
# effect of context can be measured independently of speaker identity. Output goes to a separate
# "_nospeaker" folder so it never clobbers the main model.
USE_SPEAKER = os.environ.get("SPEAKER_TAGS", "1") == "1"

# Where checkpoints / constructed CSVs / results go.
OUTPUT_DIR = CHECKPOINTS_DIR / ("emoberta_meld_large" if USE_SPEAKER else "emoberta_meld_large_nospeaker")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Columns
DIALOG_COL  = "Dialogue_ID"
UTTID_COL   = "Utterance_ID"
SPEAKER_COL = "Speaker"
TEXT_COL    = "Utterance"
LABEL_COL   = "Emotion"

# MELD Ekman-7
LABELS = ["neutral", "joy", "sadness", "anger", "surprise", "fear", "disgust"]
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for l, i in label2id.items()}

# Model
MODEL_BASE = "roberta-large"

# Paper constants
WEIGHT_DECAY = 0.01         # L2 regularization rate λ
EPOCHS = 7                  # epochs
WARMUP_RATIO = 0.20
LR_SCHED = "linear"

# Optuna (LR search).
# NOTE: kept at 1e-6..1e-4 for consistency with the IEMOCAP run. roberta-large
# can diverge (collapse to a single class) at the high end (>~3e-5); if any seed
# collapses, narrow LR_HIGH to ~3e-5 here and in the ablation scripts together.
N_TRIALS = 5
LR_LOW, LR_HIGH = 1e-6, 1e-4

# Training defaults
MAX_LEN = 512
BATCH_TRAIN = 8
BATCH_EVAL  = 16
GRAD_ACCUM  = 1

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", DEVICE, "| MODEL_BASE:", MODEL_BASE)
for p in [TRAIN_CSV, VAL_CSV, TEST_CSV]:
    if not Path(p).exists():
        print(f"⚠️ Missing file: {p}")


# ==========================
#  Load MELD CSVs into pandas DataFrames
# ==========================

train_df = pd.read_csv(TRAIN_CSV)
val_df   = pd.read_csv(VAL_CSV)
test_df  = pd.read_csv(TEST_CSV)

print("Rows:", len(train_df), len(val_df), len(test_df))


# ==========================
# 
#  Load tokenizer/model checkpoint and metrics
#
# ==========================

tok = AutoTokenizer.from_pretrained(MODEL_BASE, use_fast=True, add_prefix_space=True)
collator = DataCollatorWithPadding(tokenizer=tok)

def compute_metrics(eval_pred):
    logits, y_true = eval_pred
    y_pred = np.argmax(logits, axis=1)
    return {
        "acc": accuracy_score(y_true, y_pred),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }


# ==========================
#  Build context-augmented dataset (speaker tags + target-aware formatting)
# ==========================

import pandas as pd
import numpy as np
from datasets import Dataset

def build_context_dataset_with_text_target_has_speaker(
    df, tokenizer, max_length=512, speaker_caps=True, use_speaker=USE_SPEAKER, debug_n=3
):
    df = df.copy()

    # normalize
    df[TEXT_COL] = df[TEXT_COL].astype(str)
    df[SPEAKER_COL] = df[SPEAKER_COL].astype(str)
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip().str.lower()

    # ordering
    df[UTTID_COL] = pd.to_numeric(df[UTTID_COL], errors="coerce")
    df = df.dropna(subset=[UTTID_COL]).copy()
    df[UTTID_COL] = df[UTTID_COL].astype(int)

    df = df[df[LABEL_COL].isin(LABELS)].copy()
    df = df.sort_values([DIALOG_COL, UTTID_COL]).reset_index(drop=True)

    cls_id = tokenizer.cls_token_id  # <s>
    sep_id = tokenizer.sep_token_id  # </s>

    #  reserve CLS + final outer </s>
    max_tokens = max_length - 2

    #  "space token" to avoid BPE glue between separately-encoded utterances

    space_ids = tokenizer.encode(" ", add_special_tokens=False)
    if len(space_ids) == 0:

        space_ids = []

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

        # segment text: WITH speaker prefix (default), or utterance-only when use_speaker=False
        seg_text = ([f"{s}: {u}" for s, u in zip(speakers, utts)] if use_speaker
                    else [str(u) for u in utts])

        # IMPORTANT: encode each segment WITHOUT specials
        seg_ids  = [tokenizer.encode(x, add_special_tokens=False) for x in seg_text]
        n = len(seg_ids)

        for t in range(n):
            # target ids (WITH speaker)
            target_ids = seg_ids[t][:]

            # EmoBERTa uses RoBERTa's [SEP] = two consecutive </s></s> to delimit
            # segments; the current utterance is bracketed by </s></s> on each side
            # (4 separator tokens total), generalizing RoBERTa to 3 segments.
            if len(target_ids) + 4 > max_tokens:
                target_ids = target_ids[: max(0, max_tokens - 4)]


            seq_ids  = [sep_id, sep_id] + target_ids + [sep_id, sep_id]
            #  spaced separators for raw text
            seq_text = " </s></s> " + seg_text[t] + " </s></s> "

            left, right = t - 1, t + 1
            blocked_left = blocked_right = False

            while True:
                changed = False

                # ---- prepend left  ----
                if left >= 0 and not blocked_left:
                    cand = seg_ids[left]
                    #  add a space between utterances to avoid BPE glue
                    need = len(cand) + (len(space_ids) if len(seq_ids) > 0 else 0)

                    if len(seq_ids) + need <= max_tokens:
                        # cand + space + current
                        if space_ids:
                            seq_ids  = cand + space_ids + seq_ids
                        else:
                            seq_ids  = cand + seq_ids
                        seq_text = seg_text[left] + " " + seq_text
                        left -= 1
                        changed = True
                    else:
                        blocked_left = True

                # ---- append right WITHOUT adding SEP per utterance ----
                if right < n and not blocked_right:
                    cand = seg_ids[right]
                    need = len(cand) + (len(space_ids) if len(seq_ids) > 0 else 0)

                    if len(seq_ids) + need <= max_tokens:
                        if space_ids:
                            seq_ids  = seq_ids + space_ids + cand
                        else:
                            seq_ids  = seq_ids + cand
                        seq_text = seq_text + " " + seg_text[right]
                        right += 1
                        changed = True
                    else:
                        blocked_right = True

                if not changed:
                    break

            # outer roberta: <s> ... </s>
            input_ids = [cls_id] + seq_ids + [sep_id]
            input_ids = input_ids[:max_length]

            all_input_ids.append(input_ids)
            all_attn.append([1]*len(input_ids))
            all_labels.append(label2id[labs[t]])

            #  raw text stored WITHOUT outer <s> ... </s> 

            all_texts.append("<s> " + seq_text.strip() + " </s>")
            all_dialog.append(d_id)
            all_turn.append(turns[t])
            lengths.append(len(input_ids))

            if dbg_printed < debug_n:
                print("="*80)
                print(f"DEBUG {dbg_printed+1} | dialog={d_id} | uttid={turns[t]} | label={labs[t]}")
                print("RAW strict (repr so you see </s>):")
                print(repr(all_texts[-1][:1200]))
                print("\nDECODED (first 120 tokens):")
                print(tokenizer.decode(input_ids[:120], skip_special_tokens=False))
                dbg_printed += 1

    print("\nToken length stats:",
          f"min={int(np.min(lengths))}, mean={float(np.mean(lengths)):.1f}, max={int(np.max(lengths))}, n={len(lengths)}")

    return Dataset.from_dict({
        "dialogue_id": all_dialog,
        "utterance_id": all_turn,
        "context_text_raw": all_texts,
        "input_ids": all_input_ids,
        "attention_mask": all_attn,
        "labels": all_labels
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


# ----------- BUILD (prints debug examples) -----------
train_ds_full = build_context_dataset_with_text_target_has_speaker(train_df, tok, max_length=MAX_LEN, speaker_caps=True, debug_n=3)
val_ds_full   = build_context_dataset_with_text_target_has_speaker(val_df,   tok, max_length=MAX_LEN, speaker_caps=True, debug_n=1)
test_ds_full  = build_context_dataset_with_text_target_has_speaker(test_df,  tok, max_length=MAX_LEN, speaker_caps=True, debug_n=1)

print("Sizes:", len(train_ds_full), len(val_ds_full), len(test_ds_full))

# ----------- SAVE constructed CSVs under OUTPUT_DIR -----------
save_constructed_csv(train_ds_full, OUTPUT_DIR / "train_constructed_context_targetSpeaker.csv", id2label=id2label)
save_constructed_csv(val_ds_full,   OUTPUT_DIR / "val_constructed_context_targetSpeaker.csv",   id2label=id2label)
save_constructed_csv(test_ds_full,  OUTPUT_DIR / "test_constructed_context_targetSpeaker.csv",  id2label=id2label)


# ==========================
#  Quick sanity check: inspect one processed example
# ==========================

# pick one example from your built dataset
ex = train_ds_full[0]
ids = ex["input_ids"]

print(tok.decode(ids[:120], skip_special_tokens=False))

print("len(input_ids):", len(ex["input_ids"]))
print("len(attn):", len(ex["attention_mask"]))


# ==========================
#  Dataset sanity checks: fingerprints, decode sample, label distribution
# ==========================

import hashlib
import numpy as np

def ds_fingerprint(ds, n=50):
    m = hashlib.md5()
    for i in range(min(n, len(ds))):
        m.update((",".join(map(str, ds[i]["input_ids"]))).encode())
        m.update(str(ds[i]["labels"]).encode())
    return m.hexdigest()

print("train size:", len(train_ds_full), "val size:", len(val_ds_full))
print("fingerprints:")
print(" train:", ds_fingerprint(train_ds_full))
print(" val  :", ds_fingerprint(val_ds_full))

# quick decode sanity
print("\nDECODE sample 0 (first 200 tokens):")
print(tok.decode(train_ds_full[0]["input_ids"][:200], skip_special_tokens=False))

# label distribution sanity (first 5k for speed)
y = [train_ds_full[i]["labels"] for i in range(min(len(train_ds_full), 5000))]
print("\nLabel id dist (sample):", dict(zip(*np.unique(y, return_counts=True))))


# ==========================
# train/evaluate
# ==========================

def objective(trial):
    set_seed(SEED)

    lr = trial.suggest_float("lr", LR_LOW, LR_HIGH, log=True)

    train_ds = build_context_dataset_with_text_target_has_speaker(train_df, tok, max_length=MAX_LEN, speaker_caps=True)
    val_ds   = build_context_dataset_with_text_target_has_speaker(val_df,   tok, max_length=MAX_LEN, speaker_caps=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_BASE,
        num_labels=len(LABELS),
        label2id=label2id,
        id2label=id2label
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
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tok,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    out = trainer.evaluate(val_ds)

    #  minimize cross-entropy loss on validation
    return out["eval_loss"]


# ==========================
# Run Optuna study and select best hyperparameters
# ==========================

_t_search0 = time.time()
study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=SEED),
)
study.optimize(objective, n_trials=N_TRIALS)
_optuna_search_sec = time.time() - _t_search0

print("Best lr:", study.best_params["lr"])
print("Best val loss:", study.best_value)


# ==========================
#   Save model+tokenizer per EPOCH and per SEED (under OUTPUT_DIR)
# - Saves: OUTPUT_DIR/epoch_checkpoints_seed{seed}/epoch_01, epoch_02, ...
# - Also keeps the Trainer's "best checkpoint" and copies it to *_BEST
# ==========================

import os, shutil
import pandas as pd
from transformers import TrainerCallback, TrainingArguments, Trainer

best_lr = study.best_params["lr"]

SEEDS = [42, 43, 44, 45, 46]

# Build datasets ONCE (same for all seeds)
train_ds = build_context_dataset_with_text_target_has_speaker(train_df, tok, max_length=MAX_LEN, speaker_caps=True)
val_ds   = build_context_dataset_with_text_target_has_speaker(val_df,   tok, max_length=MAX_LEN, speaker_caps=True)
test_ds  = build_context_dataset_with_text_target_has_speaker(test_df,  tok, max_length=MAX_LEN, speaker_caps=True)

rows = []
_t_train0 = time.time()

# ---------- callback: save at end of each epoch ----------
class SaveByEpochCallback(TrainerCallback):
    def __init__(self, out_root, tokenizer):
        self.out_root = str(out_root)
        self.tokenizer = tokenizer
        os.makedirs(self.out_root, exist_ok=True)

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        ep = state.epoch
        ep_i = int(round(ep)) if ep is not None else 0

        save_dir = os.path.join(self.out_root, f"epoch_{ep_i:02d}")
        os.makedirs(save_dir, exist_ok=True)

        model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        print(f"✅ Saved epoch checkpoint to: {save_dir}")
        return control


for seed in SEEDS:
    print("\n" + "="*20, "SEED", seed, "="*20)
    set_seed(seed)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_BASE,
        num_labels=len(LABELS),
        label2id=label2id,
        id2label=id2label
    ).to(DEVICE)

    out_dir = str(OUTPUT_DIR / f"roberta_meld_final_seed{seed}")

    #  where we save epoch checkpoints for this seed
    epoch_root = str(OUTPUT_DIR / f"epoch_checkpoints_seed{seed}")
    if os.path.exists(epoch_root):
        shutil.rmtree(epoch_root)
    os.makedirs(epoch_root, exist_ok=True)

    epoch_saver = SaveByEpochCallback(epoch_root, tok)

    args = TrainingArguments(
        output_dir=out_dir,

        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,  # keeps only 2 trainer checkpoints (we keep all epochs separately)

        load_best_model_at_end=True,
        metric_for_best_model="weighted_f1",
        greater_is_better=True,

        learning_rate=best_lr,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_TRAIN,
        per_device_eval_batch_size=BATCH_EVAL,

        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type=LR_SCHED,

        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=seed,
        logging_steps=200,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tok,
        compute_metrics=compute_metrics,
        callbacks=[epoch_saver],   #  save model+tokenizer per epoch
    )

    trainer.train()

    best_ckpt = trainer.state.best_model_checkpoint
    print("Best checkpoint (trainer):", best_ckpt)

    # ===== Save BEST model folder  =====
    best_dir = f"{out_dir}_BEST"
    if os.path.exists(best_dir):
        shutil.rmtree(best_dir)
    shutil.copytree(best_ckpt, best_dir)
    tok.save_pretrained(best_dir)
    print("✅ Saved BEST folder to:", best_dir)
    print("✅ Epoch checkpoints saved in:", epoch_root)

    # ===== Test (only after training) =====
    test_metrics = trainer.evaluate(test_ds)
    print("TEST:", test_metrics)

    rows.append({
        "seed": seed,
        "best_ckpt": best_ckpt,
        "best_dir": best_dir,
        "epoch_root": epoch_root,
        "test_acc": float(test_metrics["eval_acc"]),
        "test_weighted_f1": float(test_metrics["eval_weighted_f1"]),
        "test_macro_f1": float(test_metrics["eval_macro_f1"]),
    })

df = pd.DataFrame(rows)
display(df)

mean_df = df[["test_acc","test_weighted_f1","test_macro_f1"]].mean().to_frame("mean")
std_df  = df[["test_acc","test_weighted_f1","test_macro_f1"]].std().to_frame("std")

print("\nMEAN:")
display(mean_df)

print("\nSTD:")
display(std_df)

# Persist results
df.to_csv(OUTPUT_DIR / "results_per_seed.csv", index=False)
mean_df.join(std_df).to_csv(OUTPUT_DIR / "results_mean_std.csv")
print("\n[saved]", OUTPUT_DIR / "results_per_seed.csv")
print("[saved]", OUTPUT_DIR / "results_mean_std.csv")


# =====================
# Reproducibility report -> OUTPUT_DIR/repro_report.{json,md}
# =====================
_final_training_sec = time.time() - _t_train0
_total_sec = time.time() - _run_t0
repro_utils.save_repro_report(
    OUTPUT_DIR,
    config={
        "run_name": "MELD roberta-large BOTH-CONTEXT (baseline)",
        "dataset": "MELD (Ekman-7)",
        "context_mode": "past+future",
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
        "seeds": str(SEEDS),
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

