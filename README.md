# Fiji–StarDist Analysis Pipeline

This repository provides a modular, Python-based pipeline for quantitative image analysis of pituitary tissue and related fluorescence microscopy data.  
It integrates **StarDist** segmentation with structured post-analysis, organelle quantification, and optional interactive review.

---

## Overview

The pipeline performs:
1. **Segmentation** – nuclei detection using StarDist (3D or MIP mode)
2. **Channel preprocessing** – illumination and crosstalk correction
3. **Per-cell feature extraction** – morphometrics, intensities, and QC filtering
4. **Region assignment** – anterior vs. marginal using Sox2 and intensity thresholds
5. **Plugin-based extensions** – organelle analysis, time-series tracking, and more
6. **Summary export** – compact CSVs and overlay figures for each sample

---

## Key Features

- **Modular plugin system**  
  Plugins are discovered automatically from the YAML configuration or can be specified via CLI:
  ```bash
  python -m src.nuclei_pipeline --config configs/example.yaml --plugins organelle_puncta timeseries_tracking

## Installation

**Prerequisites**
- Python 3.9–3.12 (tested on 3.10/3.11)
- Git
- (Optional) CUDA/cuDNN if you plan to train/run DL models yourself
- (Optional) Git LFS if you intend to version large TIFFs

**1) Clone**
```bash
git clone https://github.com/mugetekin/fiji-stardist-pipeline.git
cd fiji-stardist-pipeline
```

**2) Create an isolated environment**

Using Conda (recommended):
```bash
conda create -n fiji-stardist python=3.11 -y
conda activate fiji-stardist
```

Or using venv:
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```
3) Install dependencies

If a requirements.txt exists:
```bash
pip install -U pip
pip install -r requirements.txt
```

Otherwise, install the minimal stack:
```bash
pip install -U pip
pip install numpy pandas scikit-image tifffile matplotlib pyyaml
# Optional (plugin/UI):
pip install napari[all]  # for --review
# If you will use Git LFS for large files:
git lfs install
```

## Run the pipeline
```bash
python -m src.nuclei_pipeline --config configs/example.yaml
```

Override plugins at runtime:
```bash
# Use exactly these two plugins
python -m src.nuclei_pipeline --config configs/example.yaml --plugins organelle_puncta timeseries_tracking

# Disable all plugins for this run
python -m src.nuclei_pipeline --config configs/example.yaml --plugins
```

Open the interactive Napari review after processing:
```bash
python -m src.nuclei_pipeline --config configs/example.yaml --review
```

## The system supports:

organelle_puncta – quantifies subcellular puncta per cell

timeseries_tracking – links labeled cells across timepoints

### Optional Napari review interface

Add --review to open an interactive QC window for manual correction:

python -m src.nuclei_pipeline --config configs/example.yaml --review

## Automatic illumination and crosstalk correction
Flat-field correction (Gaussian blur) and linear unmixing between Alexa488 and Cy3 channels.

## Quality control filtering
Removes artifacts based on area, aspect ratio, circularity, and intensity validity.

Outputs

Each processed sample (e.g., outputs/AP231_1) produces:
---
| File	|  Description |

*_per_cell.csv |	Full per-cell morphometric and intensity table

*_summary.csv |	Region-level Sox2 summary

*_counts_overall.csv / *_counts_by_region.csv | Compact summary tables

*_overlay.png	| Color-coded overlay (Alexa+, Cy3+, co-expression)

report.html	(optional) | Combined visual summary report


