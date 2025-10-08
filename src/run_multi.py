# src/run_multi.py
import argparse
import os
from pathlib import Path
import glob
import base64
from io import BytesIO

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import single-sample helpers from your existing pipeline
from src.nuclei_pipeline import show_channels_from_tiff, parse_mode_to_zindex, load_config
from src.post_analysis import run_analysis

TIFF_EXTS = {".tif", ".tiff", ".TIF", ".TIFF"}

def is_tiff(p: Path) -> bool:
    return p.suffix in TIFF_EXTS and p.is_file()

def expand_inputs(inputs, input_dir, glob_pattern):
    files = []
    # 1) explicit paths
    for s in (inputs or []):
        p = Path(s)
        if p.is_dir():
            files += [q for q in p.rglob("*") if is_tiff(q)]
        elif p.is_file() and is_tiff(p):
            files.append(p)
        else:
            files += [Path(x) for x in glob.glob(s) if is_tiff(Path(x))]
    # 2) input_dir
    if input_dir:
        d = Path(input_dir)
        if d.is_dir():
            files += [q for q in d.rglob("*") if is_tiff(q)]
    # 3) glob
    if glob_pattern:
        files += [Path(x) for x in glob.glob(glob_pattern) if is_tiff(Path(x))]
    # de-dup & sort
    uniq = sorted({str(p.resolve()) for p in files})
    return [Path(u) for u in uniq]

def make_prefix_for(path: Path, out_dir: Path) -> Path:
    """outputs/<stem> (safe). Append _2, _3 ... if clash."""
    base = path.stem
    prefix = out_dir / base
    i = 2
    while (prefix.parent / f"{prefix.name}_RGBmerge.jpg").exists() or \
          any((prefix.parent / f"{prefix.name}{s}").exists() for s in ("_per_cell.csv","_counts_overall.csv","_stardist_labels.tif")):
        prefix = out_dir / f"{base}_{i}"
        i += 1
    return prefix

def read_if_exists(path: Path) -> pd.DataFrame | None:
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception:
        pass
    return None

# ----------------- plotting helpers -----------------

def save_bar(df, x, y, title, out_png):
    plt.figure(figsize=(10,5))
    plt.bar(df[x].astype(str), df[y].values)
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def save_scatter(df, x, y, title, out_png):
    plt.figure(figsize=(6,5))
    plt.scatter(df[x].values, df[y].values)
    plt.xlabel(x); plt.ylabel(y); plt.title(title)
    plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()

def save_stacked_regions(region_df, out_png):
    # Expect columns: Sample, Region, N
    pvt = region_df.pivot_table(index="Sample", columns="Region", values="N", aggfunc="sum", fill_value=0)
    pvt = pvt.sort_index()
    plt.figure(figsize=(10,6))
    bottom = np.zeros(len(pvt))
    for col in pvt.columns:
        plt.bar(pvt.index.astype(str), pvt[col].values, bottom=bottom, label=str(col))
        bottom += pvt[col].values
    plt.xticks(rotation=45, ha="right")
    plt.title("Cell counts per region (stacked)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

# ----------------- batch mapping & normalization -----------------

def load_batch_map(batch_map_path: str | None, samples: list[str]) -> pd.DataFrame:
    """
    Returns a DataFrame with columns: Sample, Batch
    If no map provided, infer Batch from sample prefix before first '_' as a simple heuristic.
    """
    if batch_map_path and Path(batch_map_path).exists():
        bm = pd.read_csv(batch_map_path)
        if not {"Sample","Batch"}.issubset(bm.columns):
            raise ValueError("batch_map must have columns: Sample,Batch")
        return bm[["Sample","Batch"]].drop_duplicates()
    # heuristic inference
    rows = []
    for s in samples:
        # e.g., AP231_1 -> AP231 ; else whole stem
        parts = s.split("_")
        batch = parts[0] if len(parts) > 1 else s
        rows.append({"Sample": s, "Batch": batch})
    return pd.DataFrame(rows)

def add_batch_and_normalize(all_overall: pd.DataFrame, batch_map: pd.DataFrame, method: str = "median_center") -> pd.DataFrame:
    """
    Adds Batch column and normalized metrics.
    Supported methods:
      - 'median_center': per-batch median-centering to global median (robust)
      - 'zscore': per-batch z-score (mean 0, std 1)
    Columns normalized: Sox2_pos_percent, Alexa_pos_percent, Cy3_pos_percent (if present)
    """
    df = all_overall.copy()
    # attach Sample column if needed
    if "Sample" not in df.columns:
        if "Prefix" in df.columns:
            df["Sample"] = df["Prefix"].apply(lambda p: Path(p).name)
        else:
            raise ValueError("Cannot infer Sample column in ALL_counts_overall.csv")
    df = df.merge(batch_map, on="Sample", how="left")

    norm_cols = [c for c in ["Sox2_pos_percent","Alexa_pos_percent","Cy3_pos_percent"] if c in df.columns]
    if not norm_cols:
        return df  # nothing to normalize

    if method == "median_center":
        for c in norm_cols:
            global_med = df[c].median()
            df[c + "_norm"] = df[c]  # init
            # per-batch median shift
            for b, sub in df.groupby("Batch"):
                med = sub[c].median()
                df.loc[sub.index, c + "_norm"] = sub[c] - med + global_med
    elif method == "zscore":
        for c in norm_cols:
            df[c + "_norm"] = df[c]
            for b, sub in df.groupby("Batch"):
                mu = sub[c].mean()
                sd = sub[c].std(ddof=1) if len(sub) > 1 else 1.0
                sd = sd if np.isfinite(sd) and sd > 0 else 1.0
                df.loc[sub.index, c + "_norm"] = (sub[c] - mu) / sd
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return df

# ----------------- HTML report -----------------

def img_to_data_uri(path: Path) -> str:
    with open(path, "rb") as f:
        b = f.read()
    b64 = base64.b64encode(b).decode("ascii")
    mime = "image/png" if path.suffix.lower()==".png" else "image/jpeg"
    return f"data:{mime};base64,{b64}"

def write_html_report(out_dir: Path, overall_csv: Path | None, regions_csv: Path | None, plot_paths: list[Path]):
    html = []
    html.append("<html><head><meta charset='utf-8'><title>Confocal Batch Report</title>")
    html.append("""
    <style>
      body{font-family: Arial, sans-serif; margin: 24px; }
      h1,h2{margin: 0 0 12px 0;}
      .grid{display:grid; grid-template-columns: repeat(auto-fit,minmax(320px,1fr)); gap:12px;}
      .card{border:1px solid #ddd; border-radius:8px; padding:12px;}
      table{border-collapse: collapse; width:100%;}
      th,td{border:1px solid #ddd; padding:6px; font-size: 13px;}
      th{background:#f6f6f6;}
      code{background:#f2f2f2; padding:2px 4px; border-radius:4px;}
    </style>
    """)
    html.append("</head><body>")
    html.append("<h1>Confocal Analysis – Batch Report</h1>")

    # Links to CSVs
    html.append("<div class='card'><h2>Downloads</h2><ul>")
    if overall_csv and overall_csv.exists():
        html.append(f"<li><a href='{overall_csv.name}' download>{overall_csv.name}</a></li>")
    if regions_csv and regions_csv.exists():
        html.append(f"<li><a href='{regions_csv.name}' download>{regions_csv.name}</a></li>")
    html.append("</ul></div>")

    # Plots
    if plot_paths:
        html.append("<h2>Plots</h2><div class='grid'>")
        for p in plot_paths:
            if p.exists():
                uri = img_to_data_uri(p)
                html.append(f"<div class='card'><h3>{p.name}</h3><img src='{uri}' style='width:100%'></div>")
        html.append("</div>")

    html.append("<p style='margin-top:24px;color:#777'>Generated by run_multi.py</p>")
    html.append("</body></html>")

    out_html = out_dir / "report.html"
    with open(out_html, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    print(f"[report] wrote {out_html}")

# ----------------- per-file worker (for parallel) -----------------

def process_one_file(fpath: Path, out_dir: Path, z_index, rolling_radius, analysis_cfg) -> dict:
    """Run preview and optional analysis for one file. Returns dict with file/prefix and status."""
    prefix = make_prefix_for(fpath, out_dir)
    prefix_str = str(prefix)
    status = {"file": str(fpath), "prefix": prefix_str, "ok": True, "error": ""}

    try:
        show_channels_from_tiff(
            str(fpath),
            z_index=z_index,
            rolling_radius=rolling_radius,
            save_prefix=prefix_str,
            show_plots=False,  # never block here
        )
    except Exception as e:
        status["ok"] = False
        status["error"] = f"preview failed: {e}"
        return status

    # Optional analysis (needs labels; if not present, skip gracefully)
    if analysis_cfg.get("run", False):
        label_path = Path(prefix_str + "_stardist_labels.tif")
        if not label_path.exists():
            status["ok"] = False
            status["error"] = f"analysis skipped (no labels): {label_path}"
            return status
        try:
            run_analysis(prefix_str, analysis_cfg)
        except Exception as e:
            status["ok"] = False
            status["error"] = f"analysis failed: {e}"
            return status

    return status

# ----------------- main -----------------

def main():
    ap = argparse.ArgumentParser(description="Batch runner for multiple TIFF inputs.")
    ap.add_argument("--config", type=str, help="YAML config (can also include analysis block)")
    ap.add_argument("--inputs", nargs="*", help="Explicit files/dirs/globs (space-separated)")
    ap.add_argument("--input_dir", type=str, help="Directory to scan recursively for .tif/.tiff")
    ap.add_argument("--glob", dest="glob_pattern", type=str, help="Glob pattern, e.g. 'data/*.tif'")
    ap.add_argument("--outputs_dir", type=str, default="outputs", help="Base outputs directory")
    ap.add_argument("--mode", type=str, default=None, help="z selection: 'middle' (default), 'mip'/'none', or 'z=<int>'")
    ap.add_argument("--rolling_radius", type=int, default=None, help="Rolling-ball radius")
    ap.add_argument("--jobs", type=int, default=1, help="Parallel workers (>=1).")
    ap.add_argument("--batch_map", type=str, default=None, help="CSV with columns Sample,Batch for normalization")
    ap.add_argument("--norm", type=str, default="median_center", choices=["none","median_center","zscore"], help="Batch normalization for overall % metrics.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.outputs_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Pull defaults from config if not set on CLI
    mode = args.mode or cfg.get("mode", "middle")
    z_index = parse_mode_to_zindex(mode)
    rolling_radius = args.rolling_radius if args.rolling_radius is not None else cfg.get("rolling_radius", 50)

    # analysis block (optional)
    analysis_cfg = cfg.get("analysis", {})

    # expand inputs
    files = expand_inputs(args.inputs, args.input_dir, args.glob_pattern)
    if not files:
        raise SystemExit("No input .tif/.tiff files found. Use --inputs / --input_dir / --glob.")

    print(f"[batch] found {len(files)} TIFF(s). jobs={args.jobs}")

    # Run (optionally in parallel)
    statuses = []
    if args.jobs and args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = [ex.submit(process_one_file, f, out_dir, z_index, rolling_radius, analysis_cfg) for f in files]
            for fut in as_completed(futs):
                statuses.append(fut.result())
                s = statuses[-1]
                print(("OK  " if s["ok"] else "FAIL") + f" :: {s['file']} :: {s['error']}")
    else:
        for f in files:
            s = process_one_file(f, out_dir, z_index, rolling_radius, analysis_cfg)
            statuses.append(s)
            print(("OK  " if s["ok"] else "FAIL") + f" :: {s['file']} :: {s['error']}")

    # Collect per-sample CSVs
    overall_rows, region_rows = [], []
    for s in statuses:
        if not s["ok"]:  # still try to collect if preview OK but analysis missing
            pass
        prefix_str = s["prefix"]
        ov_csv  = Path(prefix_str + "_counts_overall.csv")
        reg_csv = Path(prefix_str + "_counts_by_region.csv")

        ov = read_if_exists(ov_csv)
        if ov is not None:
            if "Prefix" in ov.columns and "Sample" not in ov.columns:
                ov = ov.copy()
                ov["Sample"] = Path(ov["Prefix"].iloc[0]).name
            overall_rows.append(ov)

        rg = read_if_exists(reg_csv)
        if rg is not None:
            rg = rg.copy()
            if "Sample" not in rg.columns:
                rg["Sample"] = Path(prefix_str).name
            region_rows.append(rg)

    # Aggregate ALL_* CSVs
    all_overall_path = None
    all_regions_path = None
    if overall_rows:
        all_overall = pd.concat(overall_rows, ignore_index=True)
        all_overall_path = out_dir / "ALL_counts_overall.csv"
        all_overall.to_csv(all_overall_path, index=False)
        print(f"[batch] wrote {all_overall_path}")

        # Batch mapping + normalization
        if args.norm != "none":
            batch_map = load_batch_map(args.batch_map, samples=list(all_overall["Sample"].astype(str)))
            all_overall_norm = add_batch_and_normalize(all_overall, batch_map, method=args.norm)
            all_overall_norm_path = out_dir / "ALL_counts_overall_normalized.csv"
            all_overall_norm.to_csv(all_overall_norm_path, index=False)
            print(f"[batch] wrote {all_overall_norm_path}")
        else:
            all_overall_norm_path = None

        # plots (raw)
        plots = []
        if "Sox2_pos_percent" in all_overall.columns:
            p = out_dir / "plot_sox2_percent_per_sample.png"
            save_bar(all_overall, x="Sample", y="Sox2_pos_percent", title="Sox2+ (%) per sample", out_png=str(p))
            plots.append(p)
        if "Total_cells" in all_overall.columns:
            p = out_dir / "plot_total_cells_per_sample.png"
            save_bar(all_overall, x="Sample", y="Total_cells", title="Total cells per sample", out_png=str(p))
            plots.append(p)
        # optional scatter: Alexa vs Cy3 %
        if {"Alexa_pos_percent","Cy3_pos_percent"}.issubset(all_overall.columns):
            p = out_dir / "plot_alexa_vs_cy3_percent.png"
            save_scatter(all_overall, x="Alexa_pos_percent", y="Cy3_pos_percent", title="Alexa% vs Cy3% (raw)", out_png=str(p))
            plots.append(p)

        # plots (normalized)
        if args.norm != "none" and all_overall_norm_path:
            aon = pd.read_csv(all_overall_norm_path)
            if "Sox2_pos_percent_norm" in aon.columns:
                p = out_dir / "plot_sox2_percent_norm_per_sample.png"
                save_bar(aon, x="Sample", y="Sox2_pos_percent_norm", title=f"Sox2+ ({args.norm}) per sample", out_png=str(p))
                plots.append(p)

    if region_rows:
        all_regions = pd.concat(region_rows, ignore_index=True)
        all_regions_path = out_dir / "ALL_counts_by_region.csv"
        all_regions.to_csv(all_regions_path, index=False)
        print(f"[batch] wrote {all_regions_path}")

        p = out_dir / "plot_cells_by_region_stacked.png"
        save_stacked_regions(all_regions, out_png=str(p))
        if 'plots' in locals():
            plots.append(p)
        else:
            plots = [p]

    # HTML report with embedded images (base64)
    write_html_report(out_dir,
                      overall_csv=all_overall_path,
                      regions_csv=all_regions_path,
                      plot_paths=plots if 'plots' in locals() else [])

    print("\n[batch] done.")

if __name__ == "__main__":
    main()
