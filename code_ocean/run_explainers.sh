#!/usr/bin/env bash
# Code Ocean — LIVE explainer run: actually compute LIME + GradSHAP (+ Random baseline)
# attributions on the context-aware model and evaluate their ERASER faithfulness. Unlike
# code_ocean/run (which only replays saved artifacts), this loads the trained model and runs
# the explainers from scratch.
#
# Requirements (see code_ocean/REPRODUCING.md):
#   - the trained model checkpoints staged in /data:
#       /data/emoberta_meld_large/roberta_meld_final_seed42_BEST      (+ test_constructed_context_targetSpeaker.csv)
#       /data/emoberta_iemocap_large_both/roberta_iemocap_both_seed42_BEST (+ test_constructed_both.csv)
#   - transformers + captum + lime installed (environment/postInstall)
#   - a GPU is strongly recommended; on CPU keep ERC_LIMIT small.
#
# Tunables:
#   ERC_LIMIT      number of test examples to explain (default 100; 0 = full corpus)
#   ERC_LIME_SAMPLES  LIME perturbations per example (default 100)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ERC_CHECKPOINTS="${ERC_CHECKPOINTS:-/data}"
export ERC_RESULTS="${ERC_RESULTS:-/results}"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
LIMIT="${ERC_LIMIT:-100}"
LIME_SAMPLES="${ERC_LIME_SAMPLES:-100}"
OUT="$ERC_RESULTS/faithfulness_live"
mkdir -p "$OUT"
cd "$REPO"
echo "[explainers] repo=$REPO  inputs=$ERC_CHECKPOINTS  outputs=$OUT  limit=$LIMIT"

for ds in meld iemocap; do
  echo "== faithfulness_eval.py ($ds, context, gradshap+lime+random, limit=$LIMIT) =="
  python Explainability/faithfulness_eval.py \
      --dataset "$ds" --model context \
      --explainers gradshap,lime,random \
      --limit "$LIMIT" --lime_samples "$LIME_SAMPLES" \
      --save_scores --out_dir "$OUT" \
    || echo "  [warn] $ds explainer run failed (missing checkpoint in /data?)"
done

echo "[explainers] done -> $OUT  (per-explainer faithfulness summaries + saved attributions)"

# --- Cross-corpus zero-shot generalization (reviewer note f) ---
# Uses the SAME trained checkpoints staged in /data (no extra inputs). Evaluates each model on the
# other corpus over the shared emotions {anger, happy, neutral, sadness}. Writes to $ERC_RESULTS.
echo "== cross_dataset_eval.py (zero-shot MELD<->IEMOCAP) =="
python Explainability/cross_dataset_eval.py \
  || echo "  [warn] cross_dataset_eval failed (need both models' *_BEST checkpoints in /data)"
echo "[cross-dataset] done -> $ERC_RESULTS/cross_dataset/"
