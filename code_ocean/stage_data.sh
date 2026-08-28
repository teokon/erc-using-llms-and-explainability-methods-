#!/usr/bin/env bash
# Stage the exact saved artifacts the Code Ocean capsule reads into ./capsule_data, which you
# then attach as the capsule's /data. Run from the repo root:
#
#     bash code_ocean/stage_data.sh [SOURCE_CHECKPOINTS_DIR]
#
# SOURCE_CHECKPOINTS_DIR defaults to $ERC_CHECKPOINTS, else ./checkpoints.
set -euo pipefail

SRC="${1:-${ERC_CHECKPOINTS:-checkpoints}}"
DST="capsule_data"
mkdir -p "$DST"
echo "Staging capsule inputs from: $SRC"

# Only the artifacts the capsule's run script consumes (keeps /data small):
#   faith_final    saved attributions (single-FT + pretrained)   -> paper_plots, agreement matrix
#   faith_context  saved attributions (context-aware model)      -> agreement matrix, note-7
#   faithfulness   per-explainer summary/curve CSVs              -> explanation_figures
#   agreement      per-example CSV + summary JSON                -> explanation_figures, note-7
for d in faith_final faith_context faithfulness agreement cross_dataset_preds; do
  if [ -d "$SRC/$d" ]; then cp -r "$SRC/$d" "$DST/"; echo "  staged $d"; else echo "  [skip] $SRC/$d not found"; fi
done

# Optional: also stage the trained model checkpoints + test CSVs needed by the LIVE explainer run
# (code_ocean/run_explainers.sh). ~8 GB. Enable with:  WITH_MODELS=1 bash code_ocean/stage_data.sh
if [ "${WITH_MODELS:-0}" = "1" ]; then
  echo "Staging model checkpoints for the live explainer run (~8 GB)..."
  for m in "emoberta_meld_large/roberta_meld_final_seed42_BEST" \
           "emoberta_meld_large/test_constructed_context_targetSpeaker.csv" \
           "emoberta_iemocap_large_both/roberta_iemocap_both_seed42_BEST" \
           "emoberta_iemocap_large_both/test_constructed_both.csv"; do
    if [ -e "$SRC/$m" ]; then mkdir -p "$DST/$(dirname "$m")"; cp -r "$SRC/$m" "$DST/$m"; echo "  staged $m";
    else echo "  [skip] $SRC/$m not found"; fi
  done
fi

echo "Staged into $DST/  ($(du -sh "$DST" 2>/dev/null | cut -f1)). Attach it as the capsule /data."
