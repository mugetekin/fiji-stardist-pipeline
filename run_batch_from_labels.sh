#!/usr/bin/env bash
set -euo pipefail

# paths
INDIR="/c/data/tiff"
OUTDIR="/c/projects/fiji-stardist-pipeline/outputs/growth"
CONFIG_BASE="configs/sox2_single.yaml"

# ensure
mkdir -p "$OUTDIR"
rm -f configs/_tmp_*.yaml 2>/dev/null || true

# loop every tif/tiff
find "$INDIR" -maxdepth 1 -type f \( -iname "*.tif" -o -iname "*.tiff" \) -print0 |
while IFS= read -r -d '' f; do
  stem="$(basename "$f")"; stem="${stem%.*}"

  # prefixes: use the SAME nested scheme that worked for the single image
  #   nested prefix  = outputs/growth/<stem>/<stem>
  #   flat   labels  = outputs/growth/<stem>_stardist_labels.tif  (you moved them here)
  in_win=$(cygpath -m "$f")
  nested_dir="${OUTDIR}/${stem}"
  nested_prefix="${nested_dir}/${stem}"
  flat_label="${OUTDIR}/${stem}_stardist_labels.tif"
  nested_label="${nested_prefix}_stardist_labels.tif"

  mkdir -p "$nested_dir"

  # make sure the nested label exists where post_analysis expects it
  if [[ -f "$flat_label" && ! -f "$nested_label" ]]; then
    cp -f "$flat_label" "$nested_label"
  fi

  # compose per-image config
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
done

echo "DONE."
