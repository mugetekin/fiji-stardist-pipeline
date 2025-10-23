# Fiji–StarDist Pipeline

A modular **Python pipeline** for quantitative fluorescence microscopy image analysis, built around **StarDist** segmentation and extendable with plugins for organelle analysis, tracking, and custom metrics.

---

## What It Does
1. **Segment nuclei** using StarDist (2D or 3D)
2. **Preprocess channels** (illumination + crosstalk correction)
3. **Measure features** per cell (intensity, area, shape)
4. **Assign regions** (e.g., anterior vs. marginal)
5. **Run plugins** for extra analyses (puncta detection, tracking, etc.)
6. **Export results** as clean CSVs, overlay images, and optional HTML reports

---

## Installation

**Requirements**
- Python ≥ 3.9 (tested on 3.10–3.12)
- Git
- (optional) CUDA for deep learning
- (optional) Git LFS for large images

```bash
# Clone the repo
git clone https://github.com/mugetekin/fiji-stardist-pipeline.git
cd fiji-stardist-pipeline

# Example environment
# For Windows:
py -3.10 -m venv .venv310
. .venv310/Scripts/activate

# on macOS/Linux:
python3 -m venv .venv310
source .venv310/bin/activate


pip install --upgrade pip
pip install -r requirements.txt

# or manually:
pip install "numpy<2" "scikit-image==0.21.0" "csbdeep==0.7.4" "stardist==0.8.5"             tensorflow-cpu==2.10.1 tifffile matplotlib pyyaml pandas

# (Optional) Enable GPU Support
pip install tensorflow==2.10.1

# Test the installation
python -m src.nuclei_pipeline --help

```

---

## Run the Pipeline (Single Sample)

Example:
```bash
python -m src.nuclei_pipeline --config configs/example.yaml
```

Run specific plugins:
```bash
python -m src.nuclei_pipeline \
  --config configs/example.yaml \
  --plugins organelle_puncta timeseries_tracking
```

Open interactive review (Napari):
```bash
python -m src.nuclei_pipeline --config configs/example.yaml --review
```

---

## Batch Processing (Multiple Images)

You can process entire folders of `.tif` or `.tiff` images automatically.  
1. Detects all .tif files in your input folder.
2. Creates StarDist segmentation labels (if missing).
3. Runs analysis and plugin processing for each sample.
4. Saves results (CSVs, overlays, previews) in nested folders.

### **Step 1 — Prepare Input Folder**
Place all your TIFFs in a folder, e.g.:
```
C:\data\tiff\
├── sox2 cy3_growth_AS124_1.tif
├── sox2 cy3_growth_AS124_2.tif
├── sox2 cy3_growth_GP125_1.tif
...
```

### **Step 2 — Run the Automated Batch Script**

Simply execute the provided bash script:

```bash
bash run_batch_from_labels.sh
```

This command will:
- Create StarDist label images for each .tif (using CPU, quiet mode, and tiling).
- Run the analysis pipeline (src.nuclei_pipeline) for each sample.
-Save all outputs to:

```bash
C:/projects/fiji-stardist-pipeline/outputs/growth/<sample_name>/
```

Each folder will contain:
```bash
<sample>_stardist_labels.tif
<sample>_per_cell.csv
<sample>_summary.csv
<sample>_counts_overall.csv
<sample>_counts_by_region.csv
<sample>_overlay.png
<sample>_DAPIcolor.jpg
<sample>_Alexacolor.jpg
<sample>_Cy3color.jpg
<sample>_RGBmerge.jpg
```
---

### **Step 3 — (Optional) Clean or Re-run**

If you want to reprocess from scratch:

```bash
rm -rf outputs/growth/*
bash run_batch_from_labels.sh
```

If you just want to regenerate StarDist labels manually for one file:
```bash
python -m src.make_stardist_labels \
  --input "C:/data/tiff/AP231_1.tif" \
  --output "C:/projects/fiji-stardist-pipeline/outputs/growth/AP231_1/AP231_1_stardist_labels.tif" \
  --channel 0 --mode mip --n_tiles 2,2
```
---

### **Step 4 — Combine or Summarize Results (Optional)**

If you want a merged report across all analyzed samples:

```bash
python -m src.merge_reports --outputs_dir outputs/growth
```

---

## Outputs

| File | Description |
|------|--------------|
| `*_per_cell.csv` | Per-cell morphology and intensity data |
| `*_summary.csv` | Region-level summary |
| `*_counts_overall.csv`, `*_counts_by_region.csv` | Compact summaries |
| `*_overlay.png` | Color-coded overlay (Alexa+, Cy3+, co-expressing) |
| `_stardist_labels.tif` | Segmentation masks (StarDist output) |

---

## Example Layout (After Batch Run)

```
outputs/
├── AP231_1_*                          # Classic single-sample example
└── growth/
    ├── sox2 cy3_growth_AS124_1_*      # Previews + CSVs + overlay
    ├── sox2 cy3_growth_AS124_2_*
    ├── ...
    ├── report.html                    # Merged summary report
    └── _examples/                     # Curated example outputs (kept in Git)
```

---

## Notes 
- Use only run_batch_from_labels.sh for batch analysis — it automatically runs both StarDist segmentation and nuclei analysis for all .tif files.

- The script uses CPU-only TensorFlow by default (no GPU setup needed).

- The output structure is nested, meaning each sample gets its own folder:
outputs/growth/<sample_name>/<sample_name>_*

- You can adjust parameters (e.g., --channel, --mode, --n_tiles) directly inside the script call in run_batch_from_labels.sh.

- src/run_multi.py is no longer used — it can be ignored or archived.

- Heavy outputs in outputs/growth/ should not be committed to GitHub.
Keep them local or add to .gitignore.
