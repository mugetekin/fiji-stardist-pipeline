# src/post_analysis.py
import os
import re
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from skimage import io as skio
from skimage.util import img_as_float, img_as_ubyte
from skimage.measure import regionprops_table, label as cc_label
from tifffile import imread as tifread
import matplotlib.pyplot as plt

# ---------------------------
# IO helpers
# ---------------------------

def read_gray_float(path: str) -> np.ndarray:
    """Read image (JPG/PNG/TIF) as grayscale float32 in [0,1]."""
    img = skio.imread(path)
    if img.ndim == 3:  # RGB(A)
        img = img[..., 0]
    img = img_as_float(img).astype(np.float32)
    return img

def read_label_tif(path: str) -> np.ndarray:
    """Read label image (uint16/uint32) and return int32 array."""
    lbl = tifread(path)
    if lbl.ndim == 3:
        # if it's Z-stack labels, max project
        lbl = lbl.max(axis=0)
    if lbl.ndim == 3 and lbl.shape[-1] == 1:
        lbl = lbl[..., 0]
    return lbl.astype(np.int32)

def find_existing(path_no_exts: str) -> Optional[str]:
    """Return an existing file path by trying common image extensions."""
    for ext in (".jpg",".JPG",".jpeg",".JPEG",".png",".PNG",".tif",".tiff",".TIF",".TIFF"):
        p = path_no_exts + ext
        if os.path.exists(p):
            return p
    return None

# ---------------------------
# Per-cell measurements
# ---------------------------

def region_means_by_label(label_img: np.ndarray, intensity_img: np.ndarray, colname: str) -> pd.DataFrame:
    """Return DataFrame with per-label mean intensity for the given image."""
    props = regionprops_table(label_img.astype(int), intensity_image=intensity_img, properties=("label","mean_intensity"))
    df = pd.DataFrame(props).rename(columns={"label":"Cell", "mean_intensity":colname})
    return df

def compute_threshold_low10p(series: pd.Series) -> Tuple[float,float,float]:
    """Background ≈ lowest 10% → median + 2*SD → threshold."""
    n = max(1, int(len(series) * 0.10))
    low = series.nsmallest(n)
    bg_med = float(low.median())
    bg_sd  = float(low.std(ddof=1)) if n > 1 else 0.0
    if not np.isfinite(bg_sd):
        bg_sd = 0.0
    th = bg_med + 2.0 * bg_sd
    return float(th), bg_med, bg_sd

def compute_threshold_median_plus_sd(series: pd.Series) -> Tuple[float,float,float]:
    """Median + 1*SD heuristic."""
    med = float(series.median())
    sd  = float(series.std(ddof=1)) if len(series) > 1 else 0.0
    th  = med + sd
    return float(th), med, sd

# ---------------------------
# Morphometry & intensity stats
# ---------------------------

def region_geometry_features(labels: np.ndarray) -> pd.DataFrame:
    """
    Basic morphometry per label: area, perimeter, major/minor axis, eccentricity, solidity, etc.
    Also derives circularity and aspect ratio.
    """
    props = regionprops_table(
        labels.astype(int),
        properties=(
            "label", "area", "perimeter",
            "major_axis_length", "minor_axis_length",
            "eccentricity", "solidity", "extent"
        ),
    )
    g = pd.DataFrame(props).rename(columns={"label": "Cell"})

    # Derived metrics
    # circularity = 4*pi*area / perimeter^2  (robustness: avoid div by zero)
    g["circularity"] = (4.0 * np.pi * g["area"]) / np.maximum(g["perimeter"]**2, 1e-8)
    # aspect_ratio = major / minor
    g["aspect_ratio"] = g["major_axis_length"] / np.maximum(g["minor_axis_length"], 1e-8)

    return g


def region_intensity_stats(label_img: np.ndarray, intensity_img: np.ndarray, prefix: str) -> pd.DataFrame:
    """
    Per-cell intensity stats for a given channel. Columns prefixed by `prefix` (e.g., 'DAPI_', 'Alexa_', 'Cy3_').
    Stats: mean, median, std, p10, p90, sum.
    """
    # mean via regionprops_table (fast)
    base = regionprops_table(
        label_img.astype(int),
        intensity_image=intensity_img,
        properties=("label", "mean_intensity"),
    )
    df = pd.DataFrame(base).rename(columns={"label": "Cell", "mean_intensity": f"{prefix}mean"})

    # For other stats we iterate (ok for per-cell)
    stats = []
    labels_uni = np.unique(label_img[label_img > 0])
    for lab in labels_uni:
        mask = (label_img == lab)
        vals = intensity_img[mask].ravel()
        if vals.size == 0:
            med = std = p10 = p90 = ssum = np.nan
        else:
            med = float(np.median(vals))
            std = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
            p10 = float(np.percentile(vals, 10))
            p90 = float(np.percentile(vals, 90))
            ssum = float(vals.sum())
        stats.append([int(lab), med, std, p10, p90, ssum])

    s = pd.DataFrame(stats, columns=["Cell", f"{prefix}median", f"{prefix}std", f"{prefix}p10", f"{prefix}p90", f"{prefix}sum"])
    out = df.merge(s, on="Cell", how="outer")
    return out

# ---------------------------
# Illumination / Crosstalk correction
# ---------------------------

from skimage.filters import gaussian
from skimage.restoration import estimate_sigma

def flatfield_correct(img: np.ndarray, sigma: float = 50.0) -> np.ndarray:
    """
    Correct uneven illumination by dividing by a blurred 'flatfield' image.
    sigma: blur radius (adjust based on image size)
    """
    smooth = gaussian(img, sigma=sigma, preserve_range=True)
    smooth[smooth == 0] = np.median(smooth)
    corrected = img / smooth
    corrected = np.clip(corrected / np.percentile(corrected, 99.5), 0, 1)
    return corrected.astype(np.float32)

def crosstalk_correct(
    alexa_img: np.ndarray,
    cy3_img: np.ndarray,
    bleed_factor: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple linear unmixing between Alexa (green) and Cy3 (red) channels.
    bleed_factor: estimated fraction of green leaking into red or vice versa.
    """
    a_corr = alexa_img - bleed_factor * cy3_img
    c_corr = cy3_img - bleed_factor * alexa_img
    a_corr[a_corr < 0] = 0
    c_corr[c_corr < 0] = 0
    return a_corr, c_corr

def preprocess_channels_for_analysis(dapi, alexa, cy3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply illumination + crosstalk correction before per-cell quantification."""
    dapi_corr  = flatfield_correct(dapi, sigma=40)
    alexa_corr = flatfield_correct(alexa, sigma=40)
    cy3_corr   = flatfield_correct(cy3, sigma=40)
    alexa_corr, cy3_corr = crosstalk_correct(alexa_corr, cy3_corr, bleed_factor=0.07)
    return dapi_corr, alexa_corr, cy3_corr


# ---------------------------
# QC filtering
# ---------------------------

def apply_qc_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove cells that are likely segmentation artifacts.
    Rules (adjust as needed):
      - area too small or too large
      - aspect_ratio extremely high (elongated artifacts)
      - circularity too low (weird shapes)
      - intensity values zero or NaN
    Returns a cleaned copy of df.
    """
    qc = df.copy()

    # drop if area is invalid
    if "area" in qc.columns:
        area_low, area_high = qc["area"].quantile([0.01, 0.99])  # keep middle 98%
        qc = qc[(qc["area"] > area_low) & (qc["area"] < area_high)]

    # aspect ratio sanity
    if "aspect_ratio" in qc.columns:
        qc = qc[qc["aspect_ratio"] < 5.0]

    # circularity sanity
    if "circularity" in qc.columns:
        qc = qc[qc["circularity"] > 0.2]

    # remove NaNs and zeros in intensity means
    for col in [c for c in qc.columns if c.endswith("_mean")]:
        qc = qc[np.isfinite(qc[col])]
        qc = qc[qc[col] > 0]

    # reset index
    qc = qc.reset_index(drop=True)
    print(f"[QC] filtered: kept {len(qc)} / {len(df)} cells ({len(qc)/len(df)*100:.1f}%)")
    return qc


# ---------------------------
# Region assignment (Anterior/Marginal)
# ---------------------------

def assign_regions_with_darkness(
    df: pd.DataFrame,
    darkness_image: np.ndarray,
    intensity_col_for_darkness: str,
    sox2_flag_col: str,
    min_cells: int = 5,
    dark_percentile: float = 20.0,
) -> pd.DataFrame:
    """
    Assign 'Anterior' vs 'Marginal' based on:
      - Sox2 positivity
      - Y median split (top vs bottom)
      - Darkness fraction in the region (top tends to be darker)
    """
    dark_thresh = float(np.percentile(darkness_image, dark_percentile))
    df = df.copy()
    df["Above_bg"] = df[intensity_col_for_darkness] > dark_thresh

    pos_cells = df[df[sox2_flag_col] & df["Above_bg"]]
    if len(pos_cells) < min_cells:
        df["Region"] = "Unknown"
        return df

    y_thresh = float(pos_cells["Y"].median())
    df["Region"] = np.where(df["Y"] < y_thresh, "Anterior", "Marginal")

    # Fraction of dark cells in each region
    anterior_mask = df["Region"] == "Anterior"
    marginal_mask = df["Region"] == "Marginal"
    frac_dark_anterior = float((df.loc[anterior_mask, intensity_col_for_darkness] < dark_thresh).mean()) if anterior_mask.any() else 0.0
    frac_dark_marginal = float((df.loc[marginal_mask, intensity_col_for_darkness] < dark_thresh).mean()) if marginal_mask.any() else 0.0

    # If the top (Anterior) is not darker than the bottom, swap assignments
    if frac_dark_anterior < frac_dark_marginal:
        df["Region"] = np.where(df["Y"] < y_thresh, "Marginal", "Anterior")

    # If no Sox2+ in the (current) 'Anterior', set all to Marginal
    if int(df[df["Region"]=="Anterior"][sox2_flag_col].sum()) == 0:
        df["Region"] = "Marginal"

    return df

# ---------------------------
# Overlay rendering
# ---------------------------

def make_overlay_png(
    dapi_img: np.ndarray,
    labels: np.ndarray,
    pos_mask_A: np.ndarray,
    pos_mask_B: np.ndarray,
    out_path: str,
    title: Optional[str] = None,
) -> None:
    """
    Build an RGB overlay:
      - background: DAPI grayscale
      - B channel: DAPI
      - label areas:
          * red   for B+ (e.g., Cy3+)
          * green for A+ (e.g., Alexa488+)
          * magenta for A+ & B+ (co-expression)
    """
    # Normalize DAPI to [0,1] robustly
    dmin = float(dapi_img.min())
    drng = float(np.ptp(dapi_img)) + 1e-8
    dapi01 = (dapi_img - dmin) / drng

    H, W = dapi01.shape
    rgb = np.stack([dapi01, dapi01, dapi01], axis=-1).astype(np.float32)

    # per-label masks → per-pixel masks
    max_label = int(labels.max())
    posA = np.zeros(max_label + 1, dtype=bool)
    posB = np.zeros(max_label + 1, dtype=bool)
    posA[:len(pos_mask_A)] = pos_mask_A
    posB[:len(pos_mask_B)] = pos_mask_B

    mA  = posA[labels]
    mB  = posB[labels]
    mAB = mA & mB

    # dim inside labels then colorize
    rgb[labels>0] = rgb[labels>0] * 0.35
    rgb[(labels>0) & mB]  = [1.0, 0.0, 0.0]
    rgb[(labels>0) & mA]  = [0.0, 1.0, 0.0]
    rgb[(labels>0) & mAB] = [1.0, 0.0, 1.0]

    plt.figure(figsize=(7,7))
    plt.imshow(rgb)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# ---------------------------
# CSV summary emitters
# ---------------------------

def _safe_pct(numer, denom):
    return float(numer) / float(denom) * 100.0 if denom else 0.0

def emit_summary_csvs(df: pd.DataFrame, save_prefix: str, sox2_col: str):
    """
    Write compact CSV summaries:
      - {prefix}_counts_overall.csv       (single row)
      - {prefix}_counts_by_region.csv     (one row per Region)
    Assumes df has columns: ["Cell","X","Y","DAPI_mean","Alexa_mean","Cy3_mean",
                             "Alexa_pos","Cy3_pos","Region", ...]
    """
    prefix = save_prefix

    # Overall counts
    total = int(len(df))
    alexa_pos = int(df["Alexa_pos"].sum())
    cy3_pos   = int(df["Cy3_pos"].sum())
    coexp     = int((df["Alexa_pos"] & df["Cy3_pos"]).sum())
    sox2_pos  = int(df[sox2_col].sum())

    overall = {
        "Prefix": prefix,
        "Total_cells": total,

        # Per-channel positivity (counts + %)
        "Alexa_pos_count": alexa_pos,
        "Alexa_pos_percent": _safe_pct(alexa_pos, total),
        "Cy3_pos_count": cy3_pos,
        "Cy3_pos_percent": _safe_pct(cy3_pos, total),

        # Co-expression
        "Coexpress_count": coexp,
        "Coexpress_percent": _safe_pct(coexp, total),

        # Sox2 (whichever channel is configured as Sox2)
        "Sox2_channel": "Alexa488" if sox2_col == "Alexa_pos" else "Cy3",
        "Sox2_pos_count": sox2_pos,
        "Sox2_pos_percent": _safe_pct(sox2_pos, total),

        # DAPI intensity snapshot (QC)
        "DAPI_mean_median": float(df["DAPI_mean"].median()) if "DAPI_mean" in df.columns else float("nan"),
        "DAPI_mean_mean": float(df["DAPI_mean"].mean()) if "DAPI_mean" in df.columns else float("nan"),

        # Alexa/Cy3 central tendency (QC)
        "Alexa_mean_median": float(df["Alexa_mean"].median()),
        "Alexa_mean_mean": float(df["Alexa_mean"].mean()),
        "Cy3_mean_median": float(df["Cy3_mean"].median()),
        "Cy3_mean_mean": float(df["Cy3_mean"].mean()),
    }

    # By-region breakdown
    by_region = (
        df.assign(
            Alexa_pos=df["Alexa_pos"].astype(bool),
            Cy3_pos=df["Cy3_pos"].astype(bool),
            Coexpress=(df["Alexa_pos"] & df["Cy3_pos"]).astype(bool),
            Sox2_pos=df[sox2_col].astype(bool),
        )
        .groupby("Region", dropna=False)
        .agg(
            N=("Cell", "count"),
            Alexa_pos_count=("Alexa_pos", "sum"),
            Cy3_pos_count=("Cy3_pos", "sum"),
            Coexpress_count=("Coexpress", "sum"),
            Sox2_pos_count=("Sox2_pos", "sum"),
            Alexa_mean_median=("Alexa_mean", "median"),
            Cy3_mean_median=("Cy3_mean", "median"),
            DAPI_mean_median=("DAPI_mean", "median") if "DAPI_mean" in df.columns else ("Alexa_mean", "median")
        )
        .reset_index()
    )
    # add percents per region
    by_region["Alexa_pos_percent"]  = by_region.apply(lambda r: _safe_pct(r["Alexa_pos_count"], r["N"]), axis=1)
    by_region["Cy3_pos_percent"]    = by_region.apply(lambda r: _safe_pct(r["Cy3_pos_count"], r["N"]), axis=1)
    by_region["Coexpress_percent"]  = by_region.apply(lambda r: _safe_pct(r["Coexpress_count"], r["N"]), axis=1)
    by_region["Sox2_pos_percent"]   = by_region.apply(lambda r: _safe_pct(r["Sox2_pos_count"], r["N"]), axis=1)

    # Write CSVs
    overall_df = pd.DataFrame([overall])
    overall_csv = f"{prefix}_counts_overall.csv"
    by_region_csv = f"{prefix}_counts_by_region.csv"

    overall_df.to_csv(overall_csv, index=False)
    by_region.to_csv(by_region_csv, index=False)

    print(f"[analysis] wrote {overall_csv} and {by_region_csv}")

# ---------------------------
# Main one-sample analysis
# ---------------------------

def analyze_one_prefix(
    save_prefix: str,
    *,
    sox2_channel: str = "Cy3",           # "Alexa488" or "Cy3"
    darkness_channel: str = "Cy3",       # channel probed for background/darkness
    threshold_method: str = "low10p",    # "low10p" or "median_plus_sd"
    dark_percentile: float = 20.0,
    min_cells_for_regions: int = 5,
    make_overlay: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Run per-cell analysis for a single sample given a `save_prefix`.
    Expects that the pipeline has already created:
      - {prefix}_stardist_labels.tif
      - {prefix}_DAPI_blue.jpg
      - {prefix}_Alexa488_green.jpg
      - {prefix}_Cy3_red.jpg
    """
    label_path = save_prefix + "_stardist_labels.tif"
    dapi_path  = find_existing(save_prefix + "_DAPI_blue")
    alexa_path = find_existing(save_prefix + "_Alexa488_green")
    cy3_path   = find_existing(save_prefix + "_Cy3_red")

    if not (os.path.exists(label_path) and dapi_path and alexa_path and cy3_path):
        raise FileNotFoundError(
            f"Missing required files for prefix={save_prefix!r}:\n"
            f"  labels: {label_path} (exists={os.path.exists(label_path)})\n"
            f"  DAPI  : {dapi_path}\n"
            f"  Alexa : {alexa_path}\n"
            f"  Cy3   : {cy3_path}"
        )

    # read images
    labels = read_label_tif(label_path)
    dapi   = read_gray_float(dapi_path)
    alexa  = read_gray_float(alexa_path)
    cy3    = read_gray_float(cy3_path)

    # --- Illumination & Crosstalk correction
    dapi, alexa, cy3 = preprocess_channels_for_analysis(dapi, alexa, cy3)


    H,W = labels.shape
    assert dapi.shape == (H,W) and alexa.shape == (H,W) and cy3.shape == (H,W), \
        f"Shape mismatch: labels{labels.shape}, dapi{dapi.shape}, alexa{alexa.shape}, cy3{cy3.shape}"

    # geometry for centroids
    geom = regionprops_table(labels, properties=("label","area","centroid"))
    geom_df = pd.DataFrame(geom).rename(columns={"label":"Cell", "centroid-0":"Y", "centroid-1":"X"})

    # per-cell morphology
    geom_more = region_geometry_features(labels)  # area, perimeter, eccentricity, solidity, etc.

    # per-cell intensity stats for each channel
    dapi_stats  = region_intensity_stats(labels, dapi,  "DAPI_")
    alexa_stats = region_intensity_stats(labels, alexa, "Alexa_")
    cy3_stats   = region_intensity_stats(labels, cy3,   "Cy3_")

    df = (geom_df
      .merge(geom_more,  on="Cell", how="left")
      .merge(dapi_stats, on="Cell", how="left")
      .merge(alexa_stats,on="Cell", how="left")
      .merge(cy3_stats,  on="Cell", how="left"))



    # positivity thresholds
    if threshold_method == "low10p":
        th_a, bg_a_med, bg_a_sd = compute_threshold_low10p(df["Alexa_mean"])
        th_c, bg_c_med, bg_c_sd = compute_threshold_low10p(df["Cy3_mean"])
    elif threshold_method == "median_plus_sd":
        th_a, bg_a_med, bg_a_sd = compute_threshold_median_plus_sd(df["Alexa_mean"])
        th_c, bg_c_med, bg_c_sd = compute_threshold_median_plus_sd(df["Cy3_mean"])
    else:
        raise ValueError(f"Unknown threshold_method={threshold_method!r}")

    df["Alexa_pos"] = df["Alexa_mean"] > th_a
    df["Cy3_pos"]   = df["Cy3_mean"]   > th_c

    # which column is Sox2?
    sox2_col = "Cy3_pos" if sox2_channel.lower().startswith("cy3") else "Alexa_pos"

    # which column is used for darkness/background?
    dark_img  = cy3 if darkness_channel.lower().startswith("cy3") else alexa
    dark_col  = "Cy3_mean" if darkness_channel.lower().startswith("cy3") else "Alexa_mean"
    
    # --- QC filtering to drop problematic cells
    df = apply_qc_filters(df)

    # --- Plugins (optional)
    from src.plugins.runner import run_plugins, AnalysisContext

    ctx = AnalysisContext({
        "prefix": save_prefix,
        "organelle": {
        "channel": "Cy3",
        "min_sigma": 1.5, "max_sigma": 3.5, "num_sigma": 5,
        "threshold": 0.02, "overlap": 0.5, "disc_radius": 2
        }
    })
    
    plugin_names = (cfg.get("plugins") if isinstance(cfg, dict) else None) or ["organelle_puncta"]
    images_dict = {"DAPI": dapi, "Alexa": alexa, "Cy3": cy3}
    df, plugin_outputs = run_plugins(plugin_names, df, images_dict, labels, ctx)
    
    # region assignment
    df = assign_regions_with_darkness(
        df, darkness_image=dark_img, intensity_col_for_darkness=dark_col,
        sox2_flag_col=sox2_col, min_cells=min_cells_for_regions, dark_percentile=dark_percentile
    )

    # summary stats
    summary = {
        "Prefix": save_prefix,
        "Total_cells": int(len(df)),
        "Alexa_threshold": float(th_a),
        "Cy3_threshold": float(th_c),
        "Alexa_bg_median": float(bg_a_med),
        "Alexa_bg_sd": float(bg_a_sd),
        "Cy3_bg_median": float(bg_c_med),
        "Cy3_bg_sd": float(bg_c_sd),
        "Alexa_pos_%": float(df["Alexa_pos"].mean()*100.0),
        "Cy3_pos_%": float(df["Cy3_pos"].mean()*100.0),
        "Coexpress_%": float((df["Alexa_pos"] & df["Cy3_pos"]).mean()*100.0),
    }

    # write per-cell table
    per_cell_csv = save_prefix + "_per_cell.csv"
    df.to_csv(per_cell_csv, index=False)

    # region-level Sox2 summary
    summary_df = (
        df.assign(Sample=os.path.basename(save_prefix))
          .groupby("Region", dropna=False)
          .agg(Sox2_percent=(sox2_col, lambda s: float(s.mean()*100.0)),
               Sox2_count=(sox2_col, "sum"),
               N=("Cell", "count"))
          .reset_index()
    )
    summary_csv = save_prefix + "_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    # overlay
    overlay_path = ""
    if make_overlay:
        max_label = int(labels.max())
        posA = np.zeros(max_label + 1, dtype=bool)  # Alexa
        posB = np.zeros(max_label + 1, dtype=bool)  # Cy3
        posA[df["Cell"].values.astype(int)] = df["Alexa_pos"].values
        posB[df["Cell"].values.astype(int)] = df["Cy3_pos"].values

        overlay_path = save_prefix + "_overlay.png"
        make_overlay_png(
            dapi_img=dapi,
            labels=labels,
            pos_mask_A=posA,
            pos_mask_B=posB,
            out_path=overlay_path,
            title=f"Overlay — {os.path.basename(save_prefix)}",
        )

    # NEW: compact CSV summaries (overall + by-region, including counts for overlay colors)
    emit_summary_csvs(df, save_prefix, sox2_col)

    summary.update({
        "Per_cell_csv": per_cell_csv,
        "Per_sample_summary_csv": summary_csv,
        "Overlay_png": overlay_path
    })

    return df, summary

# ---------------------------
# Public entrypoint
# ---------------------------

def run_analysis(save_prefix: str, cfg: Optional[Dict] = None) -> Dict[str, float]:
    """
    Thin wrapper so nuclei_pipeline can call this easily.
    """
    cfg = cfg or {}
    _, summary = analyze_one_prefix(
        save_prefix,
        sox2_channel       = cfg.get("sox2_channel", "Cy3"),
        darkness_channel   = cfg.get("darkness_channel", "Cy3"),
        threshold_method   = cfg.get("threshold_method", "low10p"),
        dark_percentile    = float(cfg.get("dark_percentile", 20.0)),
        min_cells_for_regions = int(cfg.get("min_cells_for_regions", 5)),
        make_overlay       = bool(cfg.get("make_overlay", True)),
    )
    print("[analysis] summary:", summary)
    return summary

if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser(description="Run post-count per-cell analysis for a single pipeline prefix.")
    ap.add_argument("--prefix", required=True, help="Path prefix used by the pipeline (e.g., outputs/AP231_1)")
    ap.add_argument("--sox2", choices=["Alexa488","Cy3"], default="Cy3")
    ap.add_argument("--darkness", choices=["Alexa488","Cy3"], default="Cy3")
    ap.add_argument("--method", choices=["low10p","median_plus_sd"], default="low10p")
    ap.add_argument("--dark", type=float, default=20.0, help="Dark percentile used to split top/bottom regions")
    ap.add_argument("--min-cells", type=int, default=5)
    ap.add_argument("--no-overlay", action="store_true")
    args = ap.parse_args()

    _, sm = analyze_one_prefix(
        args.prefix,
        sox2_channel=args.sox2,
        darkness_channel=args.darkness,
        threshold_method=args.method,
        dark_percentile=args.dark,
        min_cells_for_regions=args.min_cells,
        make_overlay=(not args.no_overlay),
    )
    print(json.dumps(sm, indent=2))
