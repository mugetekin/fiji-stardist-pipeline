# src/plugins/organelle_puncta.py
from __future__ import annotations
import numpy as np
import pandas as pd
from skimage.feature import blob_log
from skimage.measure import regionprops_table

PLUGIN_NAME = "organelle_puncta"

def _blob_mask_from_points(points, shape, radius=2):
    """Create a binary mask by drawing tiny discs at blob locations."""
    if points is None or len(points) == 0:
        return np.zeros(shape, dtype=bool)
    rr = int(max(1, radius))
    mask = np.zeros(shape, dtype=bool)
    H, W = shape
    for (y, x, s) in points:
        yi, xi = int(round(y)), int(round(x))
        y0, y1 = max(0, yi-rr), min(H, yi+rr+1)
        x0, x1 = max(0, xi-rr), min(W, xi+rr+1)
        patch = np.fromfunction(lambda yy, xx: (yy-yi)**2 + (xx-xi)**2 <= rr*rr,
                                (y1-y0, x1-x0), dtype=int)
        mask[y0:y1, x0:x1] |= patch
    return mask

def augment(df: pd.DataFrame, images, labels, ctx):
    """
    Detect puncta/vesicle-like organelles on a chosen channel using LoG blobs.
    Config via ctx:
      ctx['organelle'] = {
         'channel': 'Alexa'|'Cy3'|'DAPI',
         'min_sigma': 1.5, 'max_sigma': 3.5, 'num_sigma': 5,
         'threshold': 0.02, 'overlap': 0.5, 'disc_radius': 2
      }
    """
    cfg = (ctx.get("organelle") or {})
    ch = cfg.get("channel", "Cy3")
    min_sigma = float(cfg.get("min_sigma", 1.5))
    max_sigma = float(cfg.get("max_sigma", 3.5))
    num_sigma = int(cfg.get("num_sigma", 5))
    threshold = float(cfg.get("threshold", 0.02))
    overlap   = float(cfg.get("overlap", 0.5))
    disc_r    = int(cfg.get("disc_radius", 2))

    img = images.get(ch)
    if img is None:
        print(f"[organelle_puncta] channel '{ch}' not provided.")
        return {}

    # Detect blobs (y, x, sigma)
    blobs = blob_log(img, min_sigma=min_sigma, max_sigma=max_sigma,
                     num_sigma=num_sigma, threshold=threshold, overlap=overlap)
    # optional: area mask from discs
    blob_mask = _blob_mask_from_points(blobs, img.shape, radius=disc_r)

    # Per-cell stats
    props = regionprops_table(labels.astype(int),
                              intensity_image=blob_mask.astype(np.uint8),
                              properties=("label", "area"))
    # area here counts label area in pixels, not blobs. We need counts by overlay.
    # Count puncta per label:
    label_ids = np.unique(labels[labels>0])
    counts = []
    for lab in label_ids:
        counts.append([int(lab), int(blob_mask[labels==lab].sum())])
    counts_df = pd.DataFrame(counts, columns=["Cell", f"{ch}_organel_area_px"])

    # approximate count via raw blob centers landing in label
    # (faster than connected components on mask for small discs)
    puncta_counts = []
    for lab in label_ids:
        mask = (labels == lab)
        # naive center-in-polygon test (mask indexing)
        c = 0
        for (y, x, s) in blobs:
            yi, xi = int(round(y)), int(round(x))
            if 0 <= yi < mask.shape[0] and 0 <= xi < mask.shape[1] and mask[yi, xi]:
                c += 1
        puncta_counts.append([int(lab), int(c)])
    puncta_df = pd.DataFrame(puncta_counts, columns=["Cell", f"{ch}_puncta_count"])

    out_df = counts_df.merge(puncta_df, on="Cell", how="outer")
    # density (per 1000 px) if 'area' exists in main df
    if "area" in df.columns:
        out_df[f"{ch}_puncta_density_perkpx"] = (out_df[f"{ch}_puncta_count"] / df["area"]) * 1000.0

    return {"df_add": out_df, "log": f"puncta detected={len(blobs)}"}
