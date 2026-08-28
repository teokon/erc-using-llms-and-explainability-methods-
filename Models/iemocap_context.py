#!/usr/bin/env python3
"""
Shared EmoBERTa-style context builder for IEMOCAP (6-way), used by the
both / past-only / future-only training scripts.

Paper-faithful to EmoBERTa (arXiv:2108.12009):
  - RoBERTa [SEP] = two consecutive </s></s>; the current utterance is bracketed
    by </s></s> on each side, generalizing RoBERTa to 3 segments:
        <s>  [past]  </s></s>  current  </s></s>  [future]  </s>
  - outer format [CLS] + sequence + [EOS]  (the original notebook omitted the
    trailing [EOS]; it is restored here)
  - speaker names prepended (IEMOCAP EmoBERTa NAME_MAP)

`mode` selects the ablation:
  - "both"   : expand past (left) and future (right)   -> baseline
  - "past"   : expand past (left) only
  - "future" : expand future (right) only
"""

from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset

# ---- IEMOCAP 6-way label set + variant normalization ----
LABELS = ["neutral", "frustration", "sadness", "anger", "excited", "happiness"]
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for l, i in label2id.items()}

LABEL_MAP = {
    "neu": "neutral", "neutral": "neutral",
    "fru": "frustration", "frustrated": "frustration", "frustration": "frustration",
    "sad": "sadness", "sadness": "sadness",
    "ang": "anger", "anger": "anger",
    "exc": "excited", "excited": "excited",
    "hap": "happiness", "happy": "happiness", "happiness": "happiness",
}

# EmoBERTa-style speaker names for IEMOCAP (actor id = SesXX + F/M -> name)
NAME_MAP = {
    "Ses01F": "MARY",  "Ses02F": "PATRICIA", "Ses03F": "JENNIFER", "Ses04F": "LINDA",   "Ses05F": "ELIZABETH",
    "Ses01M": "JAMES", "Ses02M": "JOHN",     "Ses03M": "ROBERT",   "Ses04M": "MICHAEL", "Ses05M": "WILLIAM",
}

DIALOG_COL  = "Dialogue_ID"
UTTID_COL   = "Utterance_ID"
SPEAKER_COL = "Speaker"
TEXT_COL    = "Utterance"
LABEL_COL   = "Emotion"


def build_iemocap_context(df, tokenizer, mode="both", max_length=512,
                          speaker_caps=True, use_speaker=True,
                          insert_space_between_utts=True,
                          include_raw_text=True, debug_n=3):
    """Return an HF Dataset of tokenized EmoBERTa-style context windows.

    mode: "both" | "past" | "future" (which side(s) of the target to expand).
    use_speaker: if False, drop the "Name: " speaker prefix from every utterance (context is
                 kept, speaker tags removed) -- the context-without-speaker cell of the
                 context/speaker 2x2 ablation (reviewer note a).
    """
    assert mode in ("both", "past", "future"), f"bad mode: {mode}"
    import pandas as pd

    df = df.copy()
    df[TEXT_COL] = df[TEXT_COL].astype(str)
    df[SPEAKER_COL] = df[SPEAKER_COL].astype(str).str.strip().str.upper()
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip().str.lower().replace(LABEL_MAP)
    df = df[df[LABEL_COL].isin(LABELS)].copy()

    # ---- turn ordering (IEMOCAP-style _F003 / _M011 if possible) ----
    turn_ex = df[UTTID_COL].astype(str).str.extract(r"_[FM](\d+)$")[0]
    if turn_ex.notna().all():
        df["_turn"] = turn_ex.astype(int)
        df["_starter"] = df[DIALOG_COL].astype(str).str.extract(r"^Ses\d{2}([FM])")[0].fillna("F").str.upper()
        df["_prio"] = (df[SPEAKER_COL] != df["_starter"]).astype(int)
        df = df.sort_values([DIALOG_COL, "_turn", "_prio"]).reset_index(drop=True)
    else:
        df[UTTID_COL] = pd.to_numeric(df[UTTID_COL], errors="coerce")
        df = df.dropna(subset=[DIALOG_COL, UTTID_COL]).copy()
        df[UTTID_COL] = df[UTTID_COL].astype(int)
        df = df.sort_values([DIALOG_COL, UTTID_COL]).reset_index(drop=True)

    # ---- speaker names (EmoBERTa NAME_MAP; fall back to raw speaker) ----
    df["_session"] = df[DIALOG_COL].astype(str).str.extract(r"^(Ses\d{2})")[0]
    df["_actor"] = (df["_session"].fillna("UNK") + df[SPEAKER_COL])
    df["_name"] = df["_actor"].map(NAME_MAP).fillna(df[SPEAKER_COL])
    if speaker_caps:
        df["_name"] = df["_name"].astype(str).str.upper()

    cls_id = tokenizer.cls_token_id  # <s>
    sep_id = tokenizer.sep_token_id  # </s>

    # reserve CLS + final EOS; the target is bracketed by </s></s> on each side (4 seps)
    max_tokens = max_length - 2
    N_TARGET_SEP = 4

    all_input_ids, all_attn, all_labels = [], [], []
    all_texts, all_dialog, all_turn = [], [], []
    dbg_printed, lengths, sep_counts = 0, [], []

    def enc_no_space(x):   return tokenizer.encode(x, add_special_tokens=False)
    def enc_with_space(x): return tokenizer.encode(" " + x, add_special_tokens=False)

    expand_left  = mode in ("both", "past")
    expand_right = mode in ("both", "future")

    for d_id, g in df.groupby(DIALOG_COL, sort=False):
        names = g["_name"].tolist()
        utts  = g[TEXT_COL].tolist()
        labs  = g[LABEL_COL].tolist()
        turns = g[UTTID_COL].tolist()

        seg_text = ([f"{nm}: {u}" for nm, u in zip(names, utts)] if use_speaker
                    else [str(u) for u in utts])
        seg_ids0 = [enc_no_space(x) for x in seg_text]
        seg_ids1 = [enc_with_space(x) for x in seg_text] if insert_space_between_utts else seg_ids0
        n = len(seg_text)

        for t in range(n):
            target_text = seg_text[t]
            target_ids = seg_ids0[t][:]

            # must fit: [sep,sep] + target + [sep,sep]
            base = N_TARGET_SEP + len(target_ids)
            if base > max_tokens:
                keep = max(0, max_tokens - N_TARGET_SEP)
                target_ids = target_ids[:keep]
                base = N_TARGET_SEP + len(target_ids)

            left_idxs, right_idxs = [], []
            left_len = right_len = 0

            i = 0
            while True:
                changed = False
                i += 1

                li = t - i
                if expand_left and li >= 0:
                    add_len = len(seg_ids0[li]) if len(left_idxs) == 0 else len(seg_ids1[li])
                    if base + left_len + add_len + right_len <= max_tokens:
                        left_idxs.insert(0, li)
                        left_len += add_len
                        changed = True

                ri = t + i
                if expand_right and ri < n:
                    add_len = len(seg_ids0[ri]) if len(right_idxs) == 0 else len(seg_ids1[ri])
                    if base + left_len + right_len + add_len <= max_tokens:
                        right_idxs.append(ri)
                        right_len += add_len
                        changed = True

                if not changed:
                    break
                # stop once we've run past both ends
                if (not expand_left or li < 0) and (not expand_right or ri >= n):
                    break

            left_ids = []
            for k, idx in enumerate(left_idxs):
                left_ids += (seg_ids0[idx] if k == 0 else seg_ids1[idx])
            right_ids = []
            for k, idx in enumerate(right_idxs):
                right_ids += (seg_ids0[idx] if k == 0 else seg_ids1[idx])

            # EmoBERTa 3-segment format with </s></s> [SEP]
            seq_ids = left_ids + [sep_id, sep_id] + target_ids + [sep_id, sep_id] + right_ids
            seq_ids = seq_ids[:max_tokens]
            input_ids = [cls_id] + seq_ids + [sep_id]      # [CLS] + seq + [EOS]
            input_ids = input_ids[:max_length]

            all_input_ids.append(input_ids)
            all_attn.append([1] * len(input_ids))
            all_labels.append(label2id[labs[t]])
            all_dialog.append(d_id)
            all_turn.append(turns[t])

            if include_raw_text:
                left_raw  = " ".join(seg_text[i] for i in left_idxs).strip()
                right_raw = " ".join(seg_text[i] for i in right_idxs).strip()
                raw = f"<s> {left_raw} </s></s> {target_text} </s></s> {right_raw} </s>".strip()
                all_texts.append(" ".join(raw.split()))

            lengths.append(len(input_ids))
            sep_counts.append(int(np.sum(np.array(input_ids) == sep_id)))

            if dbg_printed < debug_n:
                print("=" * 80)
                print(f"DEBUG {dbg_printed+1} [{mode.upper()}] dialog={d_id} turn={turns[t]} label={labs[t]}"
                      f" | left={len(left_idxs)} right={len(right_idxs)} seps={sep_counts[-1]}")
                print("DECODED:", tokenizer.decode(input_ids[:140], skip_special_tokens=False))
                dbg_printed += 1

    print(f"[{mode}] token len: min={int(np.min(lengths))} mean={float(np.mean(lengths)):.1f} "
          f"max={int(np.max(lengths))} n={len(lengths)} | sep/example mean={float(np.mean(sep_counts)):.2f}")

    data = {
        "dialogue_id": all_dialog,
        "utterance_id": all_turn,
        "input_ids": all_input_ids,
        "attention_mask": all_attn,
        "labels": all_labels,
    }
    if include_raw_text:
        data["context_text_raw"] = all_texts
    return Dataset.from_dict(data)


def save_constructed_csv(ds, out_csv, id2label=None):
    """Save a human-inspectable CSV of the constructed context strings
    (dialogue_id, utterance_id, label_id, label, context_text_raw)."""
    d = ds.to_dict()
    df_out = pd.DataFrame({
        "dialogue_id": d["dialogue_id"],
        "utterance_id": d["utterance_id"],
        "label_id": d["labels"],
        "label": [id2label.get(int(x), str(x)) if isinstance(id2label, dict) else str(x)
                  for x in d["labels"]],
        "context_text_raw": d.get("context_text_raw", [""] * len(d["labels"])),
    })
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)
    print("✅ Saved:", out_csv, "| rows:", len(df_out))
