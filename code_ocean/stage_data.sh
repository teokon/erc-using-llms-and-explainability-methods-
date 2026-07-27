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
for d in faith_final faith_context faithfulness agreement; do
  if [ -d "$SRC/$d" ]; then cp -r "$SRC/$d" "$DST/"; echo "  staged $d"; else echo "  [skip] $SRC/$d not found"; fi
done

echo "Staged into $DST/  ($(du -sh "$DST" 2>/dev/null | cut -f1)). Attach it as the capsule /data."
