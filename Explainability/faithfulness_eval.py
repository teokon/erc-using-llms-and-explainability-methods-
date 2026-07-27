#!/usr/bin/env python3
"""
Quantitative faithfulness evaluation of explanation methods over the FULL test
corpus, for the EmoBERTa-style RoBERTa-large ERC models (MELD / IEMOCAP).

Reviewer-requested metrics (ERASER-style perturbation tests):
  - Comprehensiveness (↑): p(ŷ|x) − p(ŷ|x with top-k important tokens REMOVED),
    averaged over k. Big drop ⇒ the highlighted tokens truly drive the prediction.
  - Sufficiency (↓):       p(ŷ|x) − p(ŷ|x KEEPING ONLY top-k tokens), averaged over k.
    Small drop ⇒ the top tokens alone are enough.
  - AOPC (↑):              area over the comprehensiveness perturbation curve (mean over k).
  - Deletion AUC (↓):      p(ŷ) vs fraction of tokens removed in importance order (AUC).
  - Insertion AUC (↑):     p(ŷ) vs fraction of tokens added in importance order (AUC).
  - Logit-drop (↑):        logit(ŷ|x) − logit(ŷ|x with top-k removed), averaged over k.

Explainers: GradSHAP (Captum), LIME, Optimus (attention-derived), + a RANDOM
baseline (a faithful explainer must beat random).

Tokens are perturbed by replacing them with <mask> (ERASER convention); special
tokens (<s>, </s>, <pad>) are never perturbed. Metrics are computed on the
PREDICTED class and averaged over the corpus (mean ± std).

Run in the OPTIMUS env (has real tensorflow + transformers-interpret + the
version-matched stack for the Optimus library; also has captum/lime for the
other explainers), inside tmux, one GPU:
    cd Explainability
    GPU=0 python -u faithfulness_eval.py \
        --dataset meld --explainers gradshap,lime,optimus,random \
        2>&1 | tee checkpoints/faithfulness_meld.log
"""

import os
# Force-pin to ONE GPU. The shell often presets CUDA_VISIBLE_DEVICES="0,1,2,3", which makes
# the process init every GPU and hang, so a bare setdefault is not enough.
# Precedence: $GPU  >  an already-single-GPU CUDA_VISIBLE_DEVICES  >  "0".
# (Honouring a single-GPU CUDA_VISIBLE_DEVICES matters when this module is imported by
# another script that pinned a device itself -- previously we silently stole it back to 0.)
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from erc_paths import CHECKPOINTS_DIR, OPTIMUS_REPO, pick_gpu
pick_gpu()  # pin ONE GPU before importing torch ($GPU > single CUDA_VISIBLE_DEVICES > "0")

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

CK = CHECKPOINTS_DIR
# OPTIMUS_REPO comes from erc_paths (vendored: third_party/optimus)
# Optimus config: "baseline" (mean-mean attention, ~4s/instance, full-corpus
# tractable) or "max_per_instance" (Optimus Prime, ~80s/instance).
OPTIMUS_MODE = os.environ.get("OPTIMUS_MODE", "baseline")
DATASETS = {
    "meld": {
        "ckpt": f"{CK}/emoberta_meld_large/roberta_meld_final_seed42_BEST",
        "test_csv": f"{CK}/emoberta_meld_large/test_constructed_context_targetSpeaker.csv",
    },
    "iemocap": {
        "ckpt": f"{CK}/emoberta_iemocap_large_both/roberta_iemocap_both_seed42_BEST",
        "test_csv": f"{CK}/emoberta_iemocap_large_both/test_constructed_both.csv",
    },
}

# --model selects WHICH model is explained and, with it, WHICH input it sees:
#   context    : the context-aware EmoBERTa on the full constructed context
#   single_ft  : the single-utterance fine-tuned model on the target utterance only
#   pretrained : plain roberta-large (randomly-initialised head) on the target utterance
# single_ft/pretrained reproduce the paper's Optimus corpus analysis, which is on utterances.
import re as _re
_SPEAKER_RE = _re.compile(r"^[^:]{1,30}:\s*")


def target_utterance(ctx):
    """Target utterance (between the '</s></s>' markers), speaker prefix stripped -- exactly
    the input the single-utterance model was trained on."""
    p = str(ctx).split("</s></s>")
    t = p[1].strip() if len(p) == 3 else str(ctx).strip()
    return _SPEAKER_RE.sub("", t, count=1).strip()


def model_ckpt(dataset, model):
    if model == "context":
        return DATASETS[dataset]["ckpt"]
    if model == "single_ft":
        return f"{CK}/roberta_large_single_{dataset}/roberta_single_{dataset}_seed42_BEST"
    if model == "pretrained":
        return "roberta-large"
    raise ValueError(model)


def texts_for(df, model):
    col = df["context_text_raw"].astype(str)
    return col.tolist() if model == "context" else [target_utterance(t) for t in col]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
KS = [0.01, 0.05, 0.10, 0.20, 0.50]     # perturbation fractions for comp/suff/AOPC
N_BINS = 10                              # bins for deletion/insertion curves


# =====================================================================
# Model I/O
# =====================================================================
def load_model_tok(ckpt, labels=None):
    tok = AutoTokenizer.from_pretrained("roberta-large", use_fast=True, add_prefix_space=True)
    kw = {}
    if labels is not None:      # pretrained roberta-large: attach a head with the right arity
        kw = dict(num_labels=len(labels), id2label={i: l for i, l in enumerate(labels)},
                  label2id={l: i for i, l in enumerate(labels)})
    model = AutoModelForSequenceClassification.from_pretrained(ckpt, **kw).to(DEVICE).eval()
    return model, tok


@torch.inference_mode()
def batch_probs_logits(model, ids_list, max_len=512, add_specials=True):
    """Pad a list of CONTENT id-lists, forward once, return (probs, logits) as numpy (B, C).

    RoBERTa's classification head reads position 0 (the <s>/CLS slot), so the model
    MUST see <s> ... </s> to reproduce its real predictions. ids_list holds content
    tokens only; we wrap each with <s>/</s> here so every forward matches the deployed
    model while the explainers stay aligned to the content tokens they score."""
    bos = model.config.bos_token_id
    eos = model.config.eos_token_id
    if add_specials and bos is not None and eos is not None:
        ids_list = [[bos] + list(ids)[:max_len - 2] + [eos] for ids in ids_list]
    L = min(max(len(x) for x in ids_list), max_len)
    pad = model.config.pad_token_id if model.config.pad_token_id is not None else 1
    inp = np.full((len(ids_list), L), pad, dtype=np.int64)
    att = np.zeros((len(ids_list), L), dtype=np.int64)
    for i, ids in enumerate(ids_list):
        ids = ids[:L]
        inp[i, :len(ids)] = ids
        att[i, :len(ids)] = 1
    inp = torch.tensor(inp, device=DEVICE)
    att = torch.tensor(att, device=DEVICE)
    out = model(input_ids=inp, attention_mask=att)
    logits = out.logits.float()
    probs = F.softmax(logits, dim=-1)
    return probs.cpu().numpy(), logits.cpu().numpy()


# =====================================================================
# Faithfulness metrics for a single example
# =====================================================================
def faithfulness_example(model, input_ids, pred_class, scores, mask_id, special_ids):
    """input_ids: list[int]; scores: per-position importance (same length); returns metric dict."""
    L = len(input_ids)
    content = [i for i in range(L) if input_ids[i] not in special_ids]
    if len(content) < 2:
        return None
    # rank content positions by importance toward the predicted class (desc)
    ranked = sorted(content, key=lambda i: scores[i], reverse=True)
    nc = len(content)

    # Build all perturbed variants, evaluate in ONE batch.
    variants = [list(input_ids)]     # 0 = original (full)
    tags = [("full", 0)]

    def mask_topk(nk):
        top = set(ranked[:nk])
        return [mask_id if i in top else input_ids[i] for i in range(L)]

    def keep_topk(nk):
        top = set(ranked[:nk])
        return [input_ids[i] if (i in top or input_ids[i] in special_ids) else mask_id for i in range(L)]

    # comprehensiveness / sufficiency at each k
    for k in KS:
        nk = max(1, int(round(k * nc)))
        variants.append(mask_topk(nk));  tags.append(("comp", k))
        variants.append(keep_topk(nk));  tags.append(("suff", k))
    # deletion / insertion curves over bins
    for b in range(1, N_BINS + 1):
        nb = max(1, int(round(b / N_BINS * nc)))
        variants.append(mask_topk(nb));  tags.append(("del", b))
        variants.append(keep_topk(nb));  tags.append(("ins", b))

    probs, logits = batch_probs_logits(model, variants)
    p = probs[:, pred_class]
    lg = logits[:, pred_class]
    p_full, lg_full = p[0], lg[0]

    comp_vals, suff_vals, logit_vals = [], [], []
    for idx, (kind, kv) in enumerate(tags):
        if kind == "comp":
            comp_vals.append(p_full - p[idx])
            logit_vals.append(lg_full - lg[idx])
        elif kind == "suff":
            suff_vals.append(p_full - p[idx])
    # curves in bin order
    del_by_bin = {kv: p[idx] for idx, (kind, kv) in enumerate(tags) if kind == "del"}
    ins_by_bin = {kv: p[idx] for idx, (kind, kv) in enumerate(tags) if kind == "ins"}
    del_curve = [p_full] + [del_by_bin[b] for b in range(1, N_BINS + 1)]     # frac removed 0..1
    # insertion at 0 kept == everything masked == last deletion point (all removed).
    ins_curve = [del_curve[-1]] + [ins_by_bin[b] for b in range(1, N_BINS + 1)]  # frac kept 0..1
    xs = np.linspace(0, 1, N_BINS + 1)
    del_auc = float(np.trapz(del_curve, xs))
    ins_auc = float(np.trapz(ins_curve, xs))
    # AOPC: mean predicted-prob drop as the most-important tokens are progressively
    # removed in importance order (area over the deletion curve) -- distinct from
    # comprehensiveness, which averages over the [1,5,10,20,50]% bins.
    aopc = float(np.mean([p_full - del_curve[b] for b in range(1, N_BINS + 1)]))

    return {
        "comprehensiveness": float(np.mean(comp_vals)),
        "sufficiency": float(np.mean(suff_vals)),
        "aopc": aopc,
        "logit_drop": float(np.mean(logit_vals)),
        "deletion_auc": del_auc,
        "insertion_auc": ins_auc,
        # per-bin curves (kept so the deletion/insertion curves can be plotted)
        "_del_curve": [float(v) for v in del_curve],
        "_ins_curve": [float(v) for v in ins_curve],
    }


# =====================================================================
# Explainers -> per-token importance scores aligned to input_ids
# =====================================================================
def scores_random(model, tok, texts, ids_list, rng):
    return [rng.standard_normal(len(ids)) for ids in ids_list]


def scores_gradshap(model, tok, texts, ids_list, n_samples=20):
    """Captum GradientShap on the input embeddings; per-token score = attribution summed over dim."""
    from captum.attr import GradientShap
    emb_layer = model.get_input_embeddings()

    def fwd(embs, mask):
        return model(inputs_embeds=embs, attention_mask=mask).logits   # full logits (B, C)

    bos, eos = model.config.bos_token_id, model.config.eos_token_id
    gs = GradientShap(fwd)
    out = []
    for ids in ids_list:
        # wrap content with <s>/</s> so the head reads the real CLS slot; strip them after.
        full = [bos] + list(ids) + [eos]
        t = torch.tensor([full], device=DEVICE)
        am = torch.ones_like(t)
        with torch.inference_mode():
            pred = int(model(input_ids=t, attention_mask=am).logits.argmax(-1))
        emb = emb_layer(t).detach().requires_grad_(True)
        base = torch.zeros_like(emb)
        attr = gs.attribute(emb, baselines=base, target=pred,
                            additional_forward_args=(am,), n_samples=n_samples, stdevs=0.0)
        a = attr.sum(-1)[0].detach().cpu().numpy()   # len = len(ids)+2
        out.append(a[1:-1])                           # drop <s>/</s> -> aligns to content ids
        model.zero_grad(set_to_none=True)
    return out


def _word_scores_to_tokens(text, tok, word2score, ids):
    """Map whitespace-word LIME scores onto token positions via char offsets."""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True, truncation=True, max_length=510)
    offsets = enc["offset_mapping"]
    # word char-spans
    words, spans, c = [], [], 0
    for w in text.split(" "):
        if w == "":
            c += 1; continue
        start = text.index(w, c); end = start + len(w); c = end
        words.append(w); spans.append((start, end))
    scores = np.zeros(len(offsets))
    for ti, (s, e) in enumerate(offsets):
        for wi, (ws, we) in enumerate(spans):
            if s >= ws and s < we:
                scores[ti] = word2score.get(wi, 0.0); break
    return scores[:len(ids)]


def scores_lime(model, tok, texts, ids_list, num_samples=1000):
    from lime.lime_text import LimeTextExplainer
    n_labels = model.config.num_labels
    class_names = [model.config.id2label[i] for i in range(n_labels)]

    @torch.inference_mode()
    def predict_proba(batch_texts):
        # add_special_tokens=True so LIME's model queries match the deployed model
        # (RoBERTa head reads the <s> slot).
        enc = tok(list(batch_texts), add_special_tokens=True, return_tensors="pt",
                  padding=True, truncation=True, max_length=512).to(DEVICE)
        return F.softmax(model(**enc).logits, dim=-1).cpu().numpy()

    expl = LimeTextExplainer(class_names=class_names,
                             mask_string=(tok.mask_token or "<mask>"),
                             split_expression=lambda s: s.split(" "))
    out = []
    for text, ids in zip(texts, ids_list):
        pred = int(np.argmax(predict_proba([text])[0]))
        e = expl.explain_instance(text, predict_proba, num_features=len(text.split(" ")),
                                  num_samples=num_samples, labels=(pred,))
        word2score = {wi: w for wi, w in e.as_map()[pred]}   # {word_index: weight}
        out.append(_word_scores_to_tokens(text, tok, word2score, ids))
    return out


# ---- Real Optimus library (intelligence-csd-auth-gr/Optimus) integration ----
# Needs numpy<2 + transformers-interpret alongside the vendored, patched Optimus
# (third_party/optimus, roberta-large dim support; resolved via OPTIMUS_REPO). The
# Optimus wrapper reconstructs attention from hidden states, so we feed it a thin trainer shim.
class _DummyTrainer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    @torch.inference_mode()
    def predict(self, dataset):
        # Forward the dataset's own input_ids directly (robust — no decode round-trip).
        logits_list, hidden_list, attn_list = [], [], []
        for i in range(len(dataset)):
            item = dataset[i]
            ids = torch.as_tensor(item["input_ids"]).view(1, -1).to(self.device)
            am = item.get("attention_mask", None)
            am = torch.as_tensor(am).view(1, -1).to(self.device) if am is not None else torch.ones_like(ids)
            out = self.model(input_ids=ids, attention_mask=am,
                             output_attentions=True, output_hidden_states=True, return_dict=True)
            logits_list.append(out.logits.detach().cpu().numpy())
            hidden_list.append(np.stack([h.detach().cpu().numpy() for h in out.hidden_states], axis=0))
            attn_list.append(np.stack([a.detach().cpu().numpy() for a in out.attentions], axis=0))
        logits = np.concatenate(logits_list, axis=0)
        hidden = np.concatenate(hidden_list, axis=1) if len(hidden_list) > 1 else hidden_list[0]
        attns = np.concatenate(attn_list, axis=1) if len(attn_list) > 1 else attn_list[0]
        return SimpleNamespace(predictions=(logits, hidden, attns))


class _OptimusWrapper:
    """Adapts an already-loaded HF model+tokenizer to what Optimus expects."""
    def __init__(self, hf_model, tokenizer, task="single_label"):
        self._hf_model = hf_model
        self.tokenizer = tokenizer
        self.task = task
        self.trainer = _DummyTrainer(hf_model, tokenizer)
        self.num_labels = hf_model.config.num_labels
        self.label_names = [hf_model.config.id2label[i] for i in range(self.num_labels)]
        self.bos_token = tokenizer.bos_token or "<s>"
        self.eos_token = tokenizer.eos_token or "</s>"
        self.pad_tokens = {"<pad>", "[PAD]"}

    @torch.inference_mode()
    def predict_proba(self, text):
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512,
                             add_special_tokens=True).to(self._hf_model.device)
        return F.softmax(self._hf_model(**enc).logits, dim=-1).cpu().numpy()[0]


_OPTIMUS = {}


def _get_optimus(model, tok):
    if "obj" not in _OPTIMUS:
        if str(OPTIMUS_REPO) not in sys.path:
            sys.path.insert(0, str(OPTIMUS_REPO))
        from optimus import Optimus
        wrapper = _OptimusWrapper(model, tok)
        # set_of_instance=None -> skip the (very slow) max_across calibration.
        _OPTIMUS["obj"] = Optimus(wrapper, tok, wrapper.label_names,
                                  task="single_label", set_of_instance=None)
        _OPTIMUS["labels"] = wrapper.label_names
    return _OPTIMUS["obj"]


def scores_optimus_prime(model, tok, texts, ids_list):
    """Optimus Prime (max_per_instance): the best attention config is chosen PER EXAMPLE
    by Optimus' own FTP criterion. No calibration set. This is the expensive variant
    (~113 s/example on MELD, ~519 s on IEMOCAP), hence used with --per_class."""
    return _scores_optimus_mode(model, tok, texts, ids_list, "max_per_instance", "optimus-prime")


def scores_optimus(model, tok, texts, ids_list):
    """Optimus baseline (fixed config Mean-layers, Mean-heads, 'From'). Uses no FTP."""
    return _scores_optimus_mode(model, tok, texts, ids_list, OPTIMUS_MODE, "optimus")


def _scores_optimus_mode(model, tok, texts, ids_list, mode, tag):
    """Real Optimus token importance (attention interpretation) toward the predicted
    class. Optimus returns per-content-token scores (specials stripped); we align
    them to ids_list (which is add_special_tokens=False, i.e. content tokens).

    Prime is very slow and long runs get killed on the shared box, so each example's
    attribution is CHECKPOINTED to OPTIMUS_CKPT (an .npz). On restart, already-computed
    examples are loaded and skipped -- a kill costs one example, not the whole run."""
    ionbot = _get_optimus(model, tok)
    OPTIMUS_MODE_LOCAL = mode
    ckpt_path = os.environ.get("OPTIMUS_CKPT")
    done = {}
    if ckpt_path and Path(ckpt_path).exists():
        try:
            z = np.load(ckpt_path, allow_pickle=True)
            done = {int(k): v for k, v in zip(z["idx"], z["vecs"])}
            print(f"    [{tag}] resuming from checkpoint: {len(done)} examples already done", flush=True)
        except Exception as e:
            print(f"    [{tag}] checkpoint unreadable ({e}); starting fresh")
    out, n_fail = [], 0
    idx_done, vecs_done = list(done.keys()), list(done.values())
    last_save = time.time()
    for j, (text, ids) in enumerate(zip(texts, ids_list)):
        if j in done:
            out.append(np.asarray(done[j], float))
            continue
        v = np.zeros(len(ids))
        try:
            # predicted class from the real (with-specials) model input, matching the metric
            pr, _ = batch_probs_logits(model, [ids])
            pred = int(pr.argmax(-1))
            sc, _toks = ionbot.explain(text, mode=OPTIMUS_MODE_LOCAL, level="token", raw_attention="A")
            sc = np.asarray(sc, dtype=float)          # (C, Tc) content-token scores
            row = sc[pred]
            n = min(len(ids), len(row))
            v[:n] = row[:n]
        except Exception as e:
            n_fail += 1
            if n_fail <= 3:
                print(f"    [{tag}] example {j} failed ({e}); using zeros")
        out.append(v)
        if ckpt_path:                                  # checkpoint EVERY example (write is
            idx_done.append(j); vecs_done.append(v.astype(np.float32))   # tiny vs Prime cost)
            np.savez(ckpt_path + ".tmp",
                     idx=np.asarray(idx_done, np.int32),
                     vecs=np.array(vecs_done, dtype=object), allow_pickle=True)
            os.replace(ckpt_path + ".tmp.npz", ckpt_path)   # atomic replace
        if (j + 1) % 25 == 0:
            print(f"    {tag} attributions: {j+1}/{len(ids_list)}", flush=True)
    if ckpt_path and idx_done:                          # final flush
        np.savez(ckpt_path + ".tmp", idx=np.asarray(idx_done, np.int32),
                 vecs=np.array(vecs_done, dtype=object), allow_pickle=True)
        os.replace(ckpt_path + ".tmp.npz", ckpt_path)
    if n_fail:
        print(f"    [{tag}] {n_fail}/{len(ids_list)} examples failed -> zero scores")
    return out


# ---- Optimus Batch (max_across): one config chosen on a stratified calibration set ----
_STATE = {}
TRAIN_CSV = {
    "meld": f"{CK}/emoberta_meld_large/train_constructed_context_targetSpeaker.csv",
    "iemocap": f"{CK}/emoberta_iemocap_large_both/train_constructed_both.csv",
}
OPTIMUS_CALIB_PER_CLASS = int(os.environ.get("OPTIMUS_CALIB_PER_CLASS", "10"))
CALIB_DIR = Path(f"{CK}/faithfulness_optimus")


def _build_calib(dataset, per_class, seed=42):
    """Stratified per-class sample of TRAIN contexts (calibrate on train, evaluate on test)."""
    df = pd.read_csv(TRAIN_CSV[dataset])
    rng = np.random.default_rng(seed)
    picks = []
    for _c, g in df.groupby("label"):
        idx = g.index.to_numpy().copy()
        rng.shuffle(idx)
        picks.extend(idx[:per_class].tolist())
    return texts_for(df.loc[picks], _STATE.get("model", "context"))


def _get_optimus_batch(model, tok, dataset):
    key = f"batch_{dataset}"
    if key not in _OPTIMUS:
        if str(OPTIMUS_REPO) not in sys.path:
            sys.path.insert(0, str(OPTIMUS_REPO))
        from optimus import Optimus
        wrapper = _OptimusWrapper(model, tok)
        # DEFAULT: the ORIGINAL, untouched Optimus FTP. Reported results must come from the
        # published implementation, so the vectorised FTP is strictly opt-in and is never
        # used unless OPTIMUS_FAST_FTP=1 is set explicitly.
        if os.environ.get("OPTIMUS_FAST_FTP"):
            from optimus_fast_ftp import install_fast_ftp
            install_fast_ftp(wrapper)
            print("    [optimus-batch] WARNING: using the VECTORISED FTP (opt-in). "
                  "Unset OPTIMUS_FAST_FTP for the published implementation.")
        CALIB_DIR.mkdir(parents=True, exist_ok=True)
        cfile = CALIB_DIR / f"optimus_batch_calib_{dataset}_{_STATE.get('model','context')}_{OPTIMUS_CALIB_PER_CLASS}pc.json"
        ionbot = Optimus(wrapper, tok, wrapper.label_names, task="single_label", set_of_instance=None)
        if cfile.exists():   # resume: reuse the (expensive) chosen config, skip calibration
            saved = json.loads(cfile.read_text())
            ionbot.max_across_a = saved["max_across_a"]
            ionbot.max_across_a_star = saved["max_across_a_star"]
            print(f"    [optimus-batch] reused cached calibration {saved['max_across_a']} "
                  f"({saved['n_calib']} calib examples)")
        else:
            calib = _build_calib(dataset, OPTIMUS_CALIB_PER_CLASS)
            print(f"    [optimus-batch] CALIBRATING on {len(calib)} stratified train examples "
                  f"({OPTIMUS_CALIB_PER_CLASS}/class); one-time, slow ...", flush=True)
            t0 = time.time()
            cal = Optimus(wrapper, tok, wrapper.label_names, task="single_label", set_of_instance=calib)
            ionbot.max_across_a = cal.max_across_a
            ionbot.max_across_a_star = cal.max_across_a_star
            dt = (time.time() - t0) / 3600
            print(f"    [optimus-batch] calibration done in {dt:.2f} h -> config {cal.max_across_a}", flush=True)
            cfile.write_text(json.dumps({
                "dataset": dataset, "per_class": OPTIMUS_CALIB_PER_CLASS, "n_calib": len(calib),
                "calib_hours": round(dt, 3),
                "max_across_a": {k: (int(v) if not isinstance(v, str) else v)
                                 for k, v in cal.max_across_a.items()},
                "max_across_a_star": {k: (int(v) if not isinstance(v, str) else v)
                                      for k, v in cal.max_across_a_star.items()},
            }, indent=2))
        _OPTIMUS[key] = ionbot
    return _OPTIMUS[key]


def scores_optimus_batch(model, tok, texts, ids_list):
    """Optimus Batch (max_across): a single faithfulness-selected attention config,
    chosen once on the calibration set, applied to every example."""
    ionbot = _get_optimus_batch(model, tok, _STATE["dataset"])
    out, n_fail = [], 0
    for j, (text, ids) in enumerate(zip(texts, ids_list)):
        v = np.zeros(len(ids))
        try:
            pr, _ = batch_probs_logits(model, [ids])
            pred = int(pr.argmax(-1))
            sc, _t = ionbot.explain(text, mode="max_across", level="token", raw_attention="A")
            sc = np.asarray(sc, float)
            row = sc[pred]
            n = min(len(ids), len(row))
            v[:n] = row[:n]
        except Exception as e:
            n_fail += 1
            if n_fail <= 3:
                print(f"    [optimus-batch] example {j} failed ({e}); using zeros")
            # Zero-filling a handful of examples is tolerable; zero-filling EVERYTHING means
            # the explainer is broken and would silently be scored as an all-zero attribution.
            if n_fail >= 20 and n_fail == j + 1:
                raise RuntimeError(
                    f"optimus-batch failed on all {n_fail} examples so far ({e}). Refusing to "
                    f"report metrics computed on all-zero attributions.")
        out.append(v)
        if (j + 1) % 200 == 0:
            print(f"    optimus-batch attributions: {j+1}/{len(ids_list)}")
    if n_fail:
        print(f"    [optimus-batch] {n_fail}/{len(ids_list)} examples failed -> zero scores")
    return out


EXPLAINERS = {
    "gradshap": scores_gradshap,
    "lime": scores_lime,
    "optimus_batch": scores_optimus_batch,
    "optimus_prime": scores_optimus_prime,
    "optimus": scores_optimus,
    "random": scores_random,
}


# =====================================================================
# Driver
# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--explainers", default="gradshap,lime,optimus,random")
    ap.add_argument("--model", default="context", choices=["context","single_ft","pretrained"],
                    help="which model to explain; single_ft/pretrained run on the TARGET UTTERANCE (the paper's Optimus setting), context runs on the full constructed context")
    ap.add_argument("--limit", type=int, default=0, help="0 = full test corpus; else first N examples")
    ap.add_argument("--per_class", type=int, default=0,
                    help="0 = full test corpus; else a STRATIFIED subsample of K examples per "
                         "class (used for Optimus Prime, which is inherently per-example and "
                         "cannot cover the full corpus)")
    ap.add_argument("--lime_samples", type=int, default=100)
    ap.add_argument("--save_scores", action="store_true",
                    help="also save the raw per-token attributions (.npz). Needed for the "
                         "Optimus coverage/cumulative plots and the cross-method agreement "
                         "matrix -- otherwise the expensive attributions are discarded after "
                         "the metrics are computed.")
    ap.add_argument("--out_dir", default=f"{CK}/faithfulness")
    args = ap.parse_args()
    _STATE["dataset"] = args.dataset
    _STATE["model"] = args.model

    cfg = DATASETS[args.dataset]
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    _labels = sorted(pd.read_csv(cfg["test_csv"])["label"].astype(str).unique().tolist())
    _ck = model_ckpt(args.dataset, args.model)
    print(f"[faithfulness] dataset={args.dataset} | model={args.model} | ckpt={_ck}")
    model, tok = load_model_tok(_ck, labels=_labels if args.model == "pretrained" else None)
    special_ids = set(tok.all_special_ids)
    mask_id = tok.mask_token_id

    df = pd.read_csv(cfg["test_csv"])
    if args.per_class:
        # class-stratified subsample, seeded -> the same examples for every explainer,
        # so subset results stay directly comparable with each other
        r = np.random.default_rng(0)
        keep = []
        for _c, g in df.groupby("label"):
            idx = g.index.to_numpy().copy()
            r.shuffle(idx)
            keep.extend(idx[:args.per_class].tolist())
        df = df.loc[sorted(keep)].reset_index(drop=True)
        print(f"[faithfulness] STRATIFIED subset: {args.per_class}/class -> {len(df)} examples")
    elif args.limit:
        df = df.iloc[:args.limit]
    texts = texts_for(df, args.model)
    # content tokens only (max 510 so content + <s>/</s> fits the 512 budget)
    ids_list = [tok(t, add_special_tokens=False, truncation=True, max_length=510)["input_ids"] for t in texts]
    # predicted class per example (fixed reference for all explainers)
    preds = []
    for i in range(0, len(ids_list), 64):
        p, _ = batch_probs_logits(model, ids_list[i:i+64])
        preds.extend(p.argmax(-1).tolist())
    print(f"[faithfulness] {len(ids_list)} test examples loaded")

    rows = []
    for name in [e.strip() for e in args.explainers.split(",") if e.strip()]:
        fn = EXPLAINERS[name]
        print(f"\n[explainer] {name} — computing attributions ...")
        t0 = time.time()
        if name == "random":
            scores = fn(model, tok, texts, ids_list, rng)
        elif name == "lime":
            scores = fn(model, tok, texts, ids_list, num_samples=args.lime_samples)
        else:
            scores = fn(model, tok, texts, ids_list)
        attr_sec = time.time() - t0
        print(f"[explainer] {name} — attributions done in {attr_sec/60:.1f} min; scoring metrics ...")

        if args.save_scores:
            # ragged (per-example token counts differ) -> object array, plus the ids/labels
            # needed to align them later.
            np.savez_compressed(
                out_dir / f"scores_{args.dataset}_{args.model}_{name}.npz",
                scores=np.array([np.asarray(s, dtype=np.float32) for s in scores], dtype=object),
                ids=np.array([np.asarray(i, dtype=np.int32) for i in ids_list], dtype=object),
                preds=np.asarray(preds, dtype=np.int32),
                labels=np.asarray(df["label"].astype(str).tolist()),
                allow_pickle=True)
            print(f"[explainer] {name} — raw attributions saved -> scores_{args.dataset}_{args.model}_{name}.npz")

        per_ex = []
        for j, (ids, pc, sc) in enumerate(zip(ids_list, preds, scores)):
            m = faithfulness_example(model, ids, int(pc), np.asarray(sc, float), mask_id, special_ids)
            if m is not None:
                per_ex.append(m)
            if (j + 1) % 200 == 0:
                print(f"    {name}: {j+1}/{len(ids_list)}")
        # mean deletion / insertion curve over the corpus (for the curve figure)
        del_mat = np.array([m.pop("_del_curve") for m in per_ex], float)
        ins_mat = np.array([m.pop("_ins_curve") for m in per_ex], float)
        pd.DataFrame({
            "frac": np.linspace(0, 1, N_BINS + 1),
            "deletion_mean": del_mat.mean(0), "deletion_std": del_mat.std(0),
            "insertion_mean": ins_mat.mean(0), "insertion_std": ins_mat.std(0),
        }).to_csv(out_dir / f"curves_{args.dataset}_{args.model}_{name}.csv", index=False)

        agg = pd.DataFrame(per_ex)
        row = {"explainer": name, "n": len(per_ex), "attr_minutes": round(attr_sec / 60, 1)}
        for c in ["comprehensiveness", "sufficiency", "aopc", "logit_drop", "deletion_auc", "insertion_auc"]:
            row[f"{c}_mean"] = float(agg[c].mean())
            row[f"{c}_std"] = float(agg[c].std())
        rows.append(row)
        agg.to_csv(out_dir / f"faithfulness_{args.dataset}_{args.model}_{name}_perexample.csv", index=False)
        print(f"[explainer] {name}: comprehensiveness={row['comprehensiveness_mean']:.4f} "
              f"sufficiency={row['sufficiency_mean']:.4f} del_auc={row['deletion_auc_mean']:.4f} "
              f"ins_auc={row['insertion_auc_mean']:.4f}")

    # MERGE into any existing summary rather than overwriting it: a partial re-run (e.g. a
    # single explainer) must not wipe the other explainers' rows from a previous full run.
    # Rows for explainers computed in THIS run replace their old entry; others are kept.
    summary_csv = out_dir / f"faithfulness_{args.dataset}_{args.model}_summary.csv"
    new_names = {r["explainer"] for r in rows}
    merged = list(rows)
    if summary_csv.exists():
        try:
            old = pd.read_csv(summary_csv)
            for _, r in old.iterrows():
                if r["explainer"] not in new_names:
                    merged.append(r.to_dict())
        except Exception as e:
            print(f"[summary] could not merge existing summary ({e}); writing fresh")
    ORDER = ["gradshap", "lime", "optimus", "optimus_prime", "optimus_batch", "random"]
    merged.sort(key=lambda r: ORDER.index(r["explainer"]) if r["explainer"] in ORDER else 99)
    summary = pd.DataFrame(merged)
    summary.to_csv(summary_csv, index=False)
    (out_dir / f"faithfulness_{args.dataset}_{args.model}_summary.json").write_text(
        json.dumps(merged, indent=2, default=float))
    print("\n===== SUMMARY (" + args.dataset + ") =====")
    cols = ["explainer", "n", "comprehensiveness_mean", "sufficiency_mean", "aopc_mean",
            "deletion_auc_mean", "insertion_auc_mean", "logit_drop_mean"]
    print(summary[cols].to_string(index=False))
    print("\n[saved]", out_dir / f"faithfulness_{args.dataset}_{args.model}_summary.csv")


if __name__ == "__main__":
    main()
