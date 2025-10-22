"""
Organelle puncta detection (LoG-based) for per-cell quantification.

WHY THIS EXISTS
---------------
Some experiments have small, dot-like organelles (puncta/vesicles) in specific
channels (e.g., Alexa488 or Cy3). This plugin detects those puncta with a
Laplacian-of-Gaussian (LoG) blob detector and summarizes them **per cell**,
using the nucleus/cell labels produced upstream (StarDist) to aggregate
metrics. It returns a small DataFrame to be merged into the main per-cell
table.

WHERE IT'S USED
---------------
This file is loaded by the generic plugin runner at
`src/plugins/runner.py::run_plugins()`, which is called from
`src/post_analysis.py::run_analysis(...)`. In that analysis step, we already
have:
  - 2D images per channel (e.g., DAPI, Alexa, Cy3) after Z selection/MIP,
  - a label image (same XY shape) with one integer label per cell,
  - an existing per-cell DataFrame (morphometry & intensity features),
  - a config dict `cfg` converted into an `AnalysisContext` `ctx`.

This plugin adds columns like:
  - <channel>_organel_area_px      (approx area covered by tiny discs)
  - <channel>_puncta_count         (count of blob centers falling in each cell)
  - <channel>_puncta_density_perkpx (optional; normalized by each cell's area)

HOW TO ENABLE IT
----------------
In your YAML config (e.g., configs/sox2_single.yaml), make sure `analysis.run`
is true and list the plugin under `analysis.plugins`. You can also override the
detection parameters under `analysis.organelle`:

analysis:
  run: true
  plugins: [organelle_puncta]
  organelle:
    channel: "Cy3"        # one of: "DAPI", "Alexa", "Cy3" (keys of the images dict)
    min_sigma: 1.5        # LoG min sigma
    max_sigma: 3.5        # LoG max sigma
    num_sigma: 5          # number of intermediate sigmas
    threshold: 0.02       # LoG response threshold
    overlap: 0.5          # allow overlap between blobs
    disc_radius: 2        # pixel radius for drawing discs around blobs

TYPICAL CLI (batch over TIFFs)
------------------------------
python -m src.run_multi \
  --config configs/sox2_single.yaml \
  --inputs "data/growth_examples/*.tif" \
  --outputs_dir "outputs/growth" \
  --mode mip \
  --jobs 1

INPUTS (what augment() expects)
-------------------------------
- df:      pandas.DataFrame with a "Cell" column (and optionally "area")
- images:  dict with 2D float images in [0,1], keys like {"DAPI","Alexa","Cy3"}
- labels:  2D integer label image (same HxW as the selected channel)
- ctx:     AnalysisContext (dict-like), with ctx["organelle"] holding params

OUTPUT (what augment() returns)
-------------------------------
A dict with:
- "df_add": DataFrame that has at least ["Cell", <metrics...>] to be merged
            on "Cell" into the main per-cell table by the plugin runner.
- "log":    short string message for pipeline logs.

NOTES / TRADE-OFFS
------------------
- We offer two related measures:
  (1) approximate area via drawing small discs around each detected blob
      and intersecting with labels (good proxy for total puncta area).
  (2) simple count of blob centers per label (fast, robust).
- If `df` already includes per-cell "area", we also compute a density metric:
  <channel>_puncta_density_perkpx = count / area * 1000.
- The LoG detector returns (y, x, sigma) per blob. We do not currently use
  sigma to scale the disc radius; we use a fixed `disc_radius` from config
  to keep it simple/robust across images.

"""


from __future__ import annotations
import numpy as np
import pandas as pd
from skimage.feature import blob_log
from skimage.measure import regionprops_table

# The plugin runner imports modules by name and expects PLUGIN_NAME + augment()
PLUGIN_NAME = "organelle_puncta"

def _blob_mask_from_points(points, shape, radius=2):
    """
    Rasterize tiny discs at blob center locations to approximate "puncta area".

    Parameters
    ----------
    points : array-like of shape (N, 3)
        LoG blob tuples (y, x, sigma). sigma is ignored for rasterization here.
    shape : tuple[int, int]
        (H, W) of the output mask; must match image/labels spatial size.
    radius : int, default=2
        Disc radius in pixels. Use small values (1–3) for dot-like puncta.

    Returns
    -------
    mask : (H, W) bool
        True where the drawn discs fall.
    """
    if points is None or len(points) == 0:
        return np.zeros(shape, dtype=bool)
    rr = int(max(1, radius))
    mask = np.zeros(shape, dtype=bool)
    H, W = shape
    
    # Draw a small disc at each detected punctum. This gives a proxy "area".
    for (y, x, s) in points:
        yi, xi = int(round(y)), int(round(x))
        # Clip a small ROI window around the center to avoid index errors
        y0, y1 = max(0, yi-rr), min(H, yi+rr+1)
        x0, x1 = max(0, xi-rr), min(W, xi+rr+1)
        # Inside the ROI, paint a filled disc using the circle equation
        patch = np.fromfunction(lambda yy, xx: (yy-yi)**2 + (xx-xi)**2 <= rr*rr,
                                (y1-y0, x1-x0), dtype=int)
        mask[y0:y1, x0:x1] |= patch
    return mask

def augment(df: pd.DataFrame, images, labels, ctx):
    """
    Detect puncta/vesicle-like organelles via LoG and summarize per cell.

    This function is called by src/plugins/runner.py. It reads parameters from
    ctx["organelle"] (if present) and returns a dict containing "df_add" with
    new per-cell columns to merge on "Cell".

    Parameters
    ----------
    df : pandas.DataFrame
        Existing per-cell table. Must include a "Cell" column. If it also
        includes "area", we will compute a density-per-kpx metric.
    images : dict[str, np.ndarray]
        Mapping from channel name (e.g., "DAPI", "Alexa", "Cy3") to a 2D float
        image in [0,1] (same shape as labels).
    labels : np.ndarray (H, W), int
        Label image where each object/cell has a unique integer id > 0.
    ctx : dict-like (AnalysisContext)
        Contains analysis settings; this plugin expects an "organelle" section.

    Returns
    -------
    dict with keys:
      - "df_add": DataFrame with ["Cell", f"{ch}_organel_area_px", f"{ch}_puncta_count", ...]
      - "log":    A short string for logs, e.g., "puncta detected=37"
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

    # Detect blobs (y, x, sigma) with Laplacian-of-Gaussian
    # Returns array of shape (N, 3): (y, x, sigma)
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
            # simple "point-in-mask" test
            if 0 <= yi < mask.shape[0] and 0 <= xi < mask.shape[1] and mask[yi, xi]:
                c += 1
        puncta_counts.append([int(lab), int(c)])
    puncta_df = pd.DataFrame(puncta_counts, columns=["Cell", f"{ch}_puncta_count"])

    # Merge both measures on Cell; left join onto the existing per-cell df later
    out_df = counts_df.merge(puncta_df, on="Cell", how="outer")
    # Optional density normalization if caller's df includes per-cell "area"
    if "area" in df.columns:
        out_df[f"{ch}_puncta_density_perkpx"] = (out_df[f"{ch}_puncta_count"] / df["area"]) * 1000.0

    # NOTE: regionprops_table() was previously called with blob_mask as
    # intensity_image; we don't need those values now, so we omit that call
    # to keep things lean. If you ever need per-cell "mean puncta coverage",
    # you could compute it via regionprops_table(labels, intensity_image=blob_mask, ...).
    
    return {"df_add": out_df, "log": f"puncta detected={len(blobs)}"}
