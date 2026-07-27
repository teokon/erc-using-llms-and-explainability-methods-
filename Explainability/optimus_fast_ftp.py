#!/usr/bin/env python3
"""Vectorised drop-in for Optimus' faithfulness metric (FTP / RFT).

WHY. Optimus' `faithful_truthfulness_penalty` is a pure-Python triple loop over
(configs x labels x tokens). Measured on real MELD contexts it costs ~72.8 min per
calibration example, which makes Optimus Batch (max_across) and Optimus Prime
(max_per_instance) infeasible at any serious scale (~10 days for a 10-per-class
calibration set). The model forwards inside that loop are already cached, so the cost
is arithmetic, not GPU.

WHAT. This module installs a mathematically equivalent vectorised implementation:

  * The masked-token predictions depend only on WHICH token is masked -- not on the
    configuration or the label. So we build, ONCE per instance, the matrix
        D[token, label] = sigmoid(logits_original)[label] - sigmoid(logits_masked_at_token)[label]
    which costs exactly the same T model forwards the original already performed
    (and reuses the same `saved_state` cache), then reuse it for all ~1768 configs.

  * With raw_attention='A' every interpretation is min-max scaled to (1e-7, 1), i.e.
    strictly positive, so `_find_sign` always returns 'positive' and the per-label
    score collapses to
        FTP[label] = sum_token  D[token, label] / value_order[|interp[label, token]|]
    The value_order ranks are reproduced exactly (smallest |attr| -> rank = size,
    largest -> rank = 1; ties take the rank of their first occurrence in the ascending
    order, matching the original dict-update logic).

  * Anything this cannot reproduce exactly -- sentence level, or any interpretation
    containing a negative value (the A* path, where signs really vary) -- is delegated
    to the ORIGINAL implementation. So the fast path is only ever taken where it is
    provably identical.

Use `install_fast_ftp()` before constructing Optimus, and `verify_fast_ftp()` to assert
on real data that the fast and original implementations agree.
"""
import numpy as np

_WRAPPER = {}      # holds the HF model+tokenizer so we can batch the masked forwards


def _batch_logits(insts, batch=64, max_len=512):
    """Logits for many masked variants in a few batched forwards.

    The original calls Optimus' `self.predict` once PER MASKED TOKEN, and each of those
    runs the full `__inference__` -- reconstructing all 24x16 attention matrices and
    copying them to the CPU -- only for FTP to use the logits and discard the attention.
    Tokenisation matches `__inference__` (add_special_tokens=True, truncation), and the
    attention mask makes padding irrelevant, so the logits are identical.
    """
    import torch
    w = _WRAPPER["w"]
    model, tok = w._hf_model, w.tokenizer
    out = []
    with torch.inference_mode():
        for i in range(0, len(insts), batch):
            enc = tok(insts[i:i + batch], return_tensors="pt", padding=True,
                      truncation=True, max_length=max_len, add_special_tokens=True).to(model.device)
            out.append(model(**enc).logits.float().cpu().numpy())
    return np.concatenate(out, 0)


def _sigmoid32(x):
    """float32 sigmoid, matching tf.keras.activations.sigmoid(tf.constant(x, tf.float32))."""
    x = np.asarray(x, dtype=np.float32)
    return (1.0 / (1.0 + np.exp(-x, dtype=np.float32))).astype(np.float32)


def _value_order_vec(a):
    """Vectorised replica of Optimus' `value_order` dict.

    Original: iterate ascending |a|; the rank stored for a value is (size - count) of its
    FIRST occurrence (later, smaller ranks never overwrite it). Returns per-token ranks.
    """
    absa = np.abs(np.asarray(a, dtype=np.float64))
    size = absa.shape[0]
    order = np.argsort(absa)
    sorted_abs = absa[order]
    uniq, first = np.unique(sorted_abs, return_index=True)   # first occurrence, ascending
    rank_of_uniq = size - first                              # == size - count
    return rank_of_uniq[np.searchsorted(uniq, absa)].astype(np.float64)


def install_fast_ftp(wrapper):
    """Monkeypatch MyEvaluation.faithful_truthfulness_penalty with the fast version.

    `wrapper` is the Optimus model wrapper (needs ._hf_model and .tokenizer) so the
    masked-variant logits can be produced in batched, logits-only forwards.
    """
    from myEvaluation import MyEvaluation

    _WRAPPER["w"] = wrapper
    if getattr(MyEvaluation, "_fast_ftp_installed", False):
        return
    original = MyEvaluation.faithful_truthfulness_penalty
    MyEvaluation._original_ftp = original

    def fast_ftp(self, interpretation, tweaked_interpretation, instance, prediction, tokens,
                 hidden_states, t_hidden_states, rationales):
        interp = np.asarray(interpretation, dtype=np.float64)      # (L, T)

        # --- only take the fast path where it is provably identical ---
        if self.sentence_level or interp.ndim != 2 or (interp < 0).any():
            return original(self, interpretation, tweaked_interpretation, instance, prediction,
                            tokens, hidden_states, t_hidden_states, rationales)

        L = len(self.label_names)
        my_range = len(tokens) - 2
        if my_range < 1 or interp.shape[0] != L:
            return original(self, interpretation, tweaked_interpretation, instance, prediction,
                            tokens, hidden_states, t_hidden_states, rationales)

        # --- D[token, label], computed once per instance, reused across all configs ---
        cache = getattr(self, "_fast_D_cache", None)
        if cache is None:
            cache = self._fast_D_cache = {}
        key = (tuple(tokens), my_range)
        if key not in cache:
            P0 = _sigmoid32(prediction)                             # (L,)
            insts = []
            for t in range(my_range):
                tt = list(tokens)
                tt[t + 1] = "[UNK]"
                insts.append(self.fix_instance(" ".join(tt[1:-1])))
            logits = _batch_logits(insts)                           # (T, L), few forwards
            Pm = _sigmoid32(logits)                                 # (T, L)
            for ti, lg in zip(insts, logits):                       # keep the cache coherent
                self.saved_state[ti] = lg
            cache[key] = (P0[None, :] - Pm).astype(np.float64)      # (T, L)
        D = cache[key]

        # --- all signs are 'positive' on the A path -> score = sum D / value_order ---
        predicted_labels = _sigmoid32(prediction)
        out = []
        for lab in range(L):
            if not (predicted_labels[lab] >= 0.5 or self.evaluation_level_all):
                out.append(np.average([]))                          # nan, as in the original
                continue
            a = interp[lab][:my_range]
            vo = _value_order_vec(a)
            out.append(float(np.sum(D[:my_range, lab] / vo)))
        return out

    MyEvaluation.faithful_truthfulness_penalty = fast_ftp
    MyEvaluation._fast_ftp_installed = True
    print("[fast-ftp] vectorised FTP installed (A path only; A*/sentence fall back to original)")


def verify_fast_ftp(evaluator, interpretation, instance, prediction, tokens, atol=1e-6):
    """Assert the fast and original FTP agree on a real (interpretation, instance)."""
    from myEvaluation import MyEvaluation
    fast = MyEvaluation.faithful_truthfulness_penalty(
        evaluator, interpretation, None, instance, prediction, tokens, None, None, None)
    evaluator.saved_state = dict(evaluator.saved_state)   # keep the cache, avoid recompute
    slow = MyEvaluation._original_ftp(
        evaluator, interpretation, None, instance, prediction, tokens, None, None, None)
    f, s = np.asarray(fast, float), np.asarray(slow, float)
    ok = np.allclose(f, s, atol=atol, equal_nan=True)
    return ok, f, s
