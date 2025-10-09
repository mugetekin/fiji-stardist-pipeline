# src/plugins/timeseries_tracking.py
from __future__ import annotations
import subprocess
from pathlib import Path
import pandas as pd

PLUGIN_NAME = "timeseries_tracking"

def augment(df, images, labels, ctx):
    """
    Lightweight plugin wrapper around src.tracking.track_cells.

    Expects in ctx["analysis_cfg"]["tracking"]:
      - labels_dir (str) : directory with per-timepoint label TIFFs
      - sample     (str) : sample/prefix key to track (used for output name)
      - max_dist   (float, optional): link distance threshold

    Returns:
      dict with any of:
        - df_add: pd.DataFrame of tracks
        - artifact: str path to produced CSV
        - log: message string
    """
    analysis_cfg = ctx.get("analysis_cfg", {}) or {}
    tcfg = analysis_cfg.get("tracking", {}) or {}

    labels_dir = Path(tcfg.get("labels_dir", "data/timeseries")).expanduser()
    sample     = tcfg.get("sample", None)
    max_dist   = str(tcfg.get("max_dist", 20.0))

    if not sample or not labels_dir.exists():
        print("[tracking-plugin] Skipped (missing sample or labels_dir).")
        return {}

    out_csv = Path("outputs") / f"{sample}_tracks.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "-m", "src.tracking.track_cells",
        "--labels_dir", str(labels_dir),
        "--sample", sample,
        "--out", str(out_csv),
        "--max_dist", max_dist,
    ]
    print(f"[tracking-plugin] Running: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
        if out_csv.exists():
            df_track = pd.read_csv(out_csv)
            print(f"[tracking-plugin] OK: {len(df_track)} rows -> {out_csv}")
            return {"df_add": df_track, "artifact": str(out_csv), "log": "tracking OK"}
        else:
            print("[tracking-plugin] No output CSV produced.")
    except Exception as e:
        print(f"[tracking-plugin] failed: {e}")

    return {}
