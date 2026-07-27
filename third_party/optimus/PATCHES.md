# Vendored Optimus — local patches

This directory is a **vendored copy** of the Optimus attention-explanation library
(intelligence-csd-auth-gr / Optimus), included so the paper's Optimus experiments are self-contained
and reproducible. Upstream ships as a research repository, not a pip package, which is why it is
vendored rather than listed in `requirements.txt`. The original license is kept verbatim in
[`LICENCE`](LICENCE).

Only the runtime modules are vendored (`optimus.py`, `myExplainers.py`, `myEvaluation.py`,
`myModel.py`, `myTransformer.py`, `helper.py`, `dataset.py`); the upstream `datasets/`, `notebooks/`,
and `olderVersion/` trees are omitted.

## Changes we made to upstream

All patches are backward-compatible (default behaviour is unchanged) and were needed to run Optimus on
a **RoBERTa-large** classifier over the MELD / IEMOCAP corpora at a tractable cost.

1. **Model-agnostic dimensions** (`optimus.py`, `__export_model_information__`). Upstream hard-codes
   BERT-base dimensions. We read `num_hidden_layers`, `num_attention_heads`, and `hidden_size` from the
   model's `config`, so the library works with RoBERTa-large (24 layers × 16 heads × 1024 dim). The
   per-layer/head loops in `myExplainers` receive these values instead of the base-model constants.

2. **`set_of_instance=None` handling.** Constructing `Optimus(..., set_of_instance=None)` skips the very
   slow Optimus-Batch calibration (`__identify_max_across__`) instead of failing, so the baseline and
   per-instance (Prime) variants can be used without a calibration set.

3. **Baseline short-circuit** (`explain`, `conf = conf[:1]`). In `baseline` mode only configuration 0
   (mean-over-layers, mean-over-heads, "From") is computed, rather than the full configuration grid —
   a large speedup with identical output.

4. **Optimus-Batch (`max_across`) short-circuit.** The winning configuration is chosen once during
   calibration; at explain time we compute only that single configuration (`_across_shortcut`), and the
   `explain()` guard accepts `max_across` when the winning config was restored from a cached calibration
   (`_restored` / `max_across_a`). This makes Batch as cheap as baseline at inference.

5. **Calibration-cost knobs** (`__identify_max_across__`). Only token-level `max_across` is consumed
   downstream, so by default we calibrate token-level only; set `OPTIMUS_CALIB_LEVELS=token,sentence` to
   restore both. An A* calibration pass can be skipped with `OPTIMUS_CALIB_ASTAR=0`. Both default to the
   cheaper behaviour and do not change the reported (token-level, A) results.

## Runtime acceleration (not a patch to these files)

The faithfulness driver additionally installs an optional **vectorised FTP** (Faithful-Truthfulness
Penalty) implementation at runtime via `Explainability/optimus_fast_ftp.py` (opt-in with
`OPTIMUS_FAST_FTP=1`). It monkey-patches the evaluation at import time and does **not** modify the files
in this directory. It was validated to select the identical configurations as the stock FTP; the paper
figures use stock FTP unless stated otherwise.
