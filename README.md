\# Fiji + StarDist Nuclei Pipeline



This project provides a reproducible pipeline for microscopy images in OIB/OIF/TIFF format:



\- **Max Intensity Projection (ZProject MAX)** using Fiji  

\- **StarDist** nucleus segmentation on the DAPI channel  

\- Measurement of nuclei (area, centroid, bounding box)  

\- Colorized single-channel previews and RGB merges  



\## Installation

```bash

conda env create -f environment.yml

conda activate fiji-stardist

python -m ipykernel install --user --name=fiji-stardist --display-name "Python (fiji-stardist)"



