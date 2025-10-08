# src/run_multi.py
import argparse
import os
from pathlib import Path
import glob
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
            # treat as glob relative to CWD
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
    """outputs/<stem> (safe). If duplicated names exist, append _2, _3 ..."""
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

def plot_bar(df, x, y, title, out_png):
    plt.figure(figsize=(10,5))
    plt.bar(df[x].astype(str), df[y].values)
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def plot_stacked_regions(region_df, out_png):
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

def main():
    ap = argparse.ArgumentParser(description="Batch runner for multiple TIFF inputs.")
    ap.add_argument("--config", type=str, help="YAML config (can also include analysis block)")
    ap.add_argument("--inputs", nargs="*", help="Explicit files/dirs/globs (space-separated)")
    ap.add_argument("--input_dir", type=str, help="Directory to scan recursively for .tif/.tiff")
    ap.add_argument("--glob", dest="glob_pattern", type=str, help="Glob pattern, e.g. 'data/*.tif'")
    ap.add_argument("--outputs_dir", type=str, default="outputs", help="Base outputs directory")
    ap.add_argument("--mode", type=str, default=None, help="z selection: 'middle' (default), 'mip'/'none', or 'z=<int>'")
    ap.add_argument("--rolling_radius", type=int, default=None, help="Rolling-ball radius")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.outputs_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Pull some defaults from config if not set on CLI
    mode = args.mode or cfg.get("mode", "middle")
    z_index = parse_mode_to_zindex(mode)
    rolling_radius = args.rolling_radius if args.rolling_radius is not None else cfg.get("rolling_radius", 50)

    # analysis block (optional)
    analysis_cfg = cfg.get("analysis", {})
    run_analysis_flag = bool(analysis_cfg.get("run", False))

    # expand inputs
    files = expand_inputs(args.inputs, args.input_dir, args.glob_pattern)
    if not files:
        raise SystemExit("No input .tif/.tiff files found. Use --inputs / --input_dir / --glob.")

    print(f"[batch] found {len(files)} TIFF(s).")

    # Per-sample bookkeeping for aggregation
    overall_rows = []
    region_rows  = []

    for f in files:
        print(f"\n=== Processing: {f} ===")
        prefix = make_prefix_for(f, out_dir)
        prefix_str = str(prefix)

        # 1) Preview channels (DAPI/Alexa/Cy3) & save jpgs
        try:
            show_channels_from_tiff(
                str(f),
                z_index=z_index,
                rolling_radius=rolling_radius,
                save_prefix=prefix_str
            )
        except Exception as e:
            print(f"[batch] preview failed for {f}: {e}")
            continue

        # 2) Optional post-analysis (requires stardist labels already produced in your flow)
        if run_analysis_flag:
            # If label is missing, the run_analysis will complain—so skip kindly
            label_path = Path(prefix_str + "_stardist_labels.tif")
            if not label_path.exists():
                print(f"[batch] analysis skipped (no labels): {label_path}")
            else:
                try:
                    run_analysis(prefix_str, analysis_cfg)
                except Exception as e:
                    print(f"[batch] analysis failed for {f}: {e}")

        # 3) Try to collect per-sample CSVs to aggregate
        ov_csv  = Path(prefix_str + "_counts_overall.csv")
        reg_csv = Path(prefix_str + "_counts_by_region.csv")

        ov = read_if_exists(ov_csv)
        if ov is not None:
            # Add Sample column for clarity
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

    # 4) Aggregate and write ALL_* CSVs + simple plots
    if overall_rows:
        all_overall = pd.concat(overall_rows, ignore_index=True)
        all_overall_path = out_dir / "ALL_counts_overall.csv"
        all_overall.to_csv(all_overall_path, index=False)
        print(f"[batch] wrote {all_overall_path}")

        # plots
        if "Sox2_pos_percent" in all_overall.columns:
            plot_bar(all_overall, x="Sample", y="Sox2_pos_percent",
                     title="Sox2+ (%) per sample",
                     out_png=str(out_dir / "plot_sox2_percent_per_sample.png"))
        if "Total_cells" in all_overall.columns:
            plot_bar(all_overall, x="Sample", y="Total_cells",
                     title="Total cells per sample",
                     out_png=str(out_dir / "plot_total_cells_per_sample.png"))

    if region_rows:
        all_regions = pd.concat(region_rows, ignore_index=True)
        all_regions_path = out_dir / "ALL_counts_by_region.csv"
        all_regions.to_csv(all_regions_path, index=False)
        print(f"[batch] wrote {all_regions_path}")

        # stacked bar by region
        if {"Sample","Region","N"}.issubset(all_regions.columns):
            plot_stacked_regions(all_regions, out_png=str(out_dir / "plot_cells_by_region_stacked.png"))

    print("\n[batch] done.")

if __name__ == "__main__":
    main()
