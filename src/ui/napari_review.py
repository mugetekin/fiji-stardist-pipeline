"""
Quick visual quality control (QC) viewer using napari.

This script lets you visually inspect the output images and labels
from the Fiji–StarDist pipeline.

Usage:
  python -m src.ui.napari_review --prefix outputs/AP231_1

It loads and overlays:
  <prefix>_DAPI_blue.jpg
  <prefix>_Alexa488_green.jpg
  <prefix>_Cy3_red.jpg
  <prefix>_stardist_labels.tif
"""


import argparse
from pathlib import Path
import numpy as np
import tifffile
from skimage import io as skio

def _read_gray(p):
    """Read an image and convert to single grayscale float channel."""
    img = skio.imread(str(p))
    if img.ndim == 3: img = img[...,0]   # convert RGB to single channel
    return img.astype(np.float32)

def main():
  # Parse CLI argument: prefix = common path for outputs (without suffix)
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    args = ap.parse_args()

    try:  # Try to import napari only if available (it's optional)
        import napari
    except Exception:
        print("[napari] not installed. pip install napari[all]  (optional)")
        return

    # Resolve file paths based on the given prefix
    px = Path(args.prefix)
    dapi   = _read_gray(px.with_name(px.name + "_DAPI_blue.jpg"))
    alexa  = _read_gray(px.with_name(px.name + "_Alexa488_green.jpg"))
    cy3    = _read_gray(px.with_name(px.name + "_Cy3_red.jpg"))
    labels = tifffile.imread(str(px.with_name(px.name + "_stardist_labels.tif")))

    # Launch napari viewer and overlay channels + segmentation labels
    v = napari.Viewer()
    v.add_image(dapi,  name="DAPI",  colormap="blue", blending="additive")
    v.add_image(alexa, name="Alexa", colormap="green", blending="additive")
    v.add_image(cy3,   name="Cy3",   colormap="red", blending="additive")
    # If label stack is 3D (Z), use maximum projection for 2D view
    if labels.ndim == 3:
        labels = labels.max(axis=0)
    v.add_labels(labels.astype(int), name="Labels")
    # Start napari GUI
    napari.run()

if __name__ == "__main__":
    main()
