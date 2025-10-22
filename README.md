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
python -m src.nuclei_pipeline --config configs/example.yaml --plugins organelle_puncta timeseries_tracking
```

Open interactive review (Napari):
```bash
python -m src.nuclei_pipeline --config configs/example.yaml --review
```

---

## Batch Processing (Multiple Images)

You can process entire folders of `.tif` or `.tiff` images automatically.  
The batch system combines three steps:

### **Step 1 — Prepare Input Folder**
Place all your TIFFs in a folder, e.g.:
```
C:\data\tiff\
├── sox2 cy3_growth_AS124_1.tif
├── sox2 cy3_growth_AS124_2.tif
├── sox2 cy3_growth_GP125_1.tif
...
```

### **Step 2 — Generate StarDist Labels**

Before running the full analysis, generate segmentation label files once:

```bash
bash run_batch_from_labels.sh
```

or manually:

```bash
find "C:/data/tiff" -maxdepth 1 -type f -iname "*.tif" -print0 |
while IFS= read -r -d '' f; do
  stem="$(basename "$f")"; stem="${stem%.*}"
  python -m src.make_stardist_labels     --input "$(cygpath -m "$f")"     --output "C:/projects/fiji-stardist-pipeline/outputs/growth/${stem}_stardist_labels.tif"     --channel 0 --mode mip --prob_thresh 0.58 --nms_thresh 0.30
done
```

Each image will produce its `_stardist_labels.tif` in `outputs/growth/`.

---

### **Step 3 — Batch Analysis via `run_multi.py`**

After labels exist, run all analyses (previews + CSVs + overlays) in one command:

```bash
python -m src.run_multi   --config configs/sox2_single.yaml   --input_dir "C:/data/tiff"   --outputs_dir "C:/projects/fiji-stardist-pipeline/outputs/growth"   --mode mip   --jobs 1
```

- `--input_dir` → folder with TIFFs  
- `--outputs_dir` → destination for all CSVs, overlays, and report  
- `--mode mip` → use max intensity projection for Z-stacks  
- `--jobs` → parallel processing threads (use 1–4 depending on memory)

This will:
- Skip images without labels  
- Run the full post-analysis (CSV generation, metrics, overlays)  
- Merge outputs into one HTML report (`outputs/growth/report.html`)

---

### **Step 4 — Generate Merged Report Only**

If per-sample CSVs already exist (e.g., in `outputs/growth/_examples`), you can build a combined report without re-running StarDist:

```bash
python -m src.merge_reports --outputs_dir outputs/growth/_examples
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
- Use `make_stardist_labels.py` first to generate masks once per dataset.  
- Always provide flat save prefixes (no trailing folder name) in YAML configs.  
- `run_multi.py` automatically detects existing labels and performs analysis only.  
- Heavy batch outputs (like `outputs/growth`) should stay local, not pushed to Git.
