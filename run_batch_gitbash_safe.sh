#!/usr/bin/env bash
# SAFE runner: never auto-closes, never exits the whole shell, logs per-file.

set +e  # do NOT exit on errors
export PYTHONUNBUFFERED=1

INDIR="/c/data/tiff"
OUTDIR="/c/projects/fiji-stardist-pipeline/outputs/growth"
CONFIG_BASE="configs/sox2_analysis.yaml"
LOGDIR="/c/projects/fiji-stardist-pipeline/logs"

cd /c/projects/fiji-stardist-pipeline || { echo "Cannot cd to project"; read -rp "Press Enter..."; exit 0; }

mkdir -p "$OUTDIR" "$LOGDIR"
rm -f configs/_tmp_*.yaml 2>/dev/null || true

# Collect files
mapfile -d '' FILES < <(find "$INDIR" -maxdepth 1 -type f \( -iname "*.tif" -o -iname "*.tiff" \) -print0)
COUNT=${#FILES[@]}
if (( COUNT == 0 )); then
  echo "No .tif/.tiff files in: $INDIR"
  read -rp "Press Enter to close..."
  exit 0
fi

# Limit to 10 first files
LIMIT=10
(( COUNT < LIMIT )) && LIMIT=$COUNT
echo "Processing $LIMIT of $COUNT images from: $INDIR"
echo

for ((i=0; i<LIMIT; i++)); do
  f="${FILES[$i]}"
  stem="$(basename "$f")"; stem="${stem%.*}"
  yaml="configs/_tmp_${stem}.yaml"
  in_win=$(cygpath -m "$f")
  out_dir_img="${OUTDIR}/${stem}"
  out_win=$(cygpath -m "$out_dir_img")
  log="${LOGDIR}/${stem}.log"

  mkdir -p "$out_dir_img"

  # Per-image config
  {
    cat "$CONFIG_BASE"
    echo
    echo "input_path: \"${in_win}\""
    echo "save_prefix: \"${out_win}/${stem}\""
  } > "$yaml"

  echo ">>> Running ${stem} ..."
  python -m src.nuclei_pipeline \
      --config "$yaml" \
      --plugins stardist_segmentation organelle_puncta timeseries_tracking \
      2>&1 | tee "$log"

  # Check labels exist; if not, warn and continue (do NOT exit)
  lbl_path="${out_dir_img}/${stem}_stardist_labels.tif"
  if [[ ! -f "$lbl_path" ]]; then
    echo "!!! WARNING: Missing labels for ${stem}"
    echo "    Expected: $lbl_path"
    echo "    See log: $log"
    echo
    continue
  fi

  echo "✓ Done: $stem"
  echo
done

# Merge CSVs if any exist
merge_one () {
  local name="$1"
  local dest="${OUTDIR}/${name}"
  rm -f "$dest"
  local first=1
  find "$OUTDIR" -type f -name "$name" -print0 | sort -z | \
  while IFS= read -r -d '' csv; do
    if (( first )); then
      cat "$csv" >> "$dest"
      first=0
    else
      tail -n +2 "$csv" >> "$dest"
    fi
  done
  [[ -f "$dest" ]] && echo "→ Merged: $dest"
}

merge_one "summary.csv"
merge_one "counts_overall.csv"
merge_one "counts_by_region.csv"
merge_one "per_cell_table.csv"

echo
echo "ALL DONE. (This window stays open.)"
read -rp "Press Enter to close..."
