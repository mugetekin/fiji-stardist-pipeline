#!/usr/bin/env bash
set -euo pipefail

# paths
INDIR="/c/data/tiff"
OUTDIR="/c/projects/fiji-stardist-pipeline/outputs/growth"
CONFIG_BASE="configs/sox2_single.yaml"

# ensure
mkdir -p "$OUTDIR"
rm -f configs/_tmp_*.yaml 2>/dev/null || true

# loop every tif/tiff (handles spaces safely)
while IFS= read -r -d '' f; do
  stem="$(basename "$f")"; stem="${stem%.*}"

  # prefixes: nested save_prefix = outputs/growth/<stem>/<stem>
  in_win=$(cygpath -m "$f")
  nested_dir="${OUTDIR}/${stem}"
  nested_prefix="${nested_dir}/${stem}"
  flat_label="${OUTDIR}/${stem}_stardist_labels.tif"
  nested_label="${nested_prefix}_stardist_labels.tif"

  mkdir -p "$nested_dir"

# keep TF quiet and on CPU (harmless if also set in Python)
export CUDA_VISIBLE_DEVICES=-1
export TF_CPP_MIN_LOG_LEVEL=2

  
  # if you already created flat labels earlier, mirror them to nested
  if [[ -f "$flat_label" && ! -f "$nested_label" ]]; then
    cp -f "$flat_label" "$nested_label"
  fi

  # 1) Generate StarDist labels if missing (to the nested path)
  if [[ ! -f "$nested_label" ]]; then
    echo "[stardist] creating labels for ${stem} ..."
    python -m src.make_stardist_labels \
      --input "$in_win" \
      --output "$(cygpath -m "$nested_label")" \
      --channel 0 \
      --mode mip
  fi

  # 2) Compose per-image config (pointing save_prefix to the nested prefix)
  yaml="configs/_tmp_${stem}.yaml"
  {
    cat "$CONFIG_BASE"
    echo
    echo "input_path: \"${in_win}\""
    echo "save_prefix: \"$(cygpath -m "$nested_prefix")\""
    echo "mode: mip"
    echo "rolling_radius: 30"
  } > "$yaml"

  echo ">>> Running ${stem} ..."
  python -m src.nuclei_pipeline --config "$yaml"

done < <(find "$INDIR" -maxdepth 1 -type f \( -iname "*.tif" -o -iname "*.tiff" \) -print0)

echo "DONE."
