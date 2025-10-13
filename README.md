# Fiji–StarDist Pipeline

A modular **Python pipeline** for quantitative fluorescence microscopy image analysis — built around **StarDist** segmentation and extendable with plugins for organelle analysis, tracking, and custom metrics.

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

# Create a fresh environment (Conda example)
conda create -n fiji-stardist python=3.11 -y
conda activate fiji-stardist

# Install dependencies
pip install -U pip
pip install -r requirements.txt  # or manually if not provided
```

Minimal manual install:
```bash
pip install numpy pandas scikit-image tifffile matplotlib pyyaml
# Optional extras
pip install napari[all]  # for --review mode
git lfs install
```

---

##  Run the Pipeline

Process a dataset:
```bash
python -m src.nuclei_pipeline --config configs/example.yaml
```

Run specific plugins:
```bash
python -m src.nuclei_pipeline --config configs/example.yaml   --plugins organelle_puncta timeseries_tracking
```

Skip all plugins:
```bash
python -m src.nuclei_pipeline --config configs/example.yaml --plugins
```

Open an interactive Napari review:
```bash
python -m src.nuclei_pipeline --config configs/example.yaml --review
```

---

## Built-in Plugins
| Plugin | Purpose |
|--------|----------|
| `organelle_puncta` | Detects puncta-like organelles using LoG blob detection |
| `timeseries_tracking` | Tracks labeled cells over time using per-frame labels |

Custom plugins can be added easily under `src/plugins/`.

---

## Outputs

Each processed sample creates an `outputs/<sample>` folder with:

| File | Description |
|------|--------------|
| `*_per_cell.csv` | Per-cell morphology and intensity data |
| `*_summary.csv` | Region-level summary |
| `*_counts_overall.csv`, `*_counts_by_region.csv` | Compact summaries |
| `*_overlay.png` | Color-coded overlay (Alexa+, Cy3+, co-expressing) |

---

## 💡 Tips
- Keep large TIFFs out of Git with `.gitignore` and Git LFS.
- Extend analysis by writing your own plugin (see `src/plugins/`).
- Use batch mode via `src/run_multi.py` for multiple samples.
