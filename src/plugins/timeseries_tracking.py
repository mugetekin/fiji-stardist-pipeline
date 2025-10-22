"""
Time-series Tracking Plugin
This plugin is a lightweight wrapper around the `src.tracking.track_cells`
module. It links segmented cells across timepoints in a time-lapse (live-cell)
dataset — that is, it tracks cells from one frame (t) to the next.

Instead of implementing tracking logic directly here, this plugin simply
**calls** the external tracking script via subprocess and loads its output.
That design keeps the main pipeline modular and clean.

WHEN IT RUNS
------------
The plugin is automatically executed by `src/plugins/runner.py` during
post-analysis, after segmentation and per-cell feature extraction.
It reads its configuration from:
    ctx["analysis_cfg"]["tracking"]

EXPECTED CONFIGURATION (inside your YAML or analysis context)
-------------------------------------------------------------
tracking:
  labels_dir: "data/timeseries/AP231_1"
  sample: "AP231_1"
  max_dist: 25.0

FIELDS:
- **labels_dir** : str  
  Directory containing per-timepoint label TIFFs
  (e.g. t000_labels.tif, t001_labels.tif, ...)

- **sample** : str  
  Sample or prefix used to name the output track file  
  (e.g. "AP231_1" → output will be outputs/AP231_1_tracks.csv)

- **max_dist** : float (optional, default = 20.0)  
  Maximum distance allowed between cell centroids when linking them
  across consecutive frames (in pixels).

OUTPUTS
-------
The plugin returns a dictionary (like all plugins) with optional keys:
- `"df_add"`   : pandas DataFrame of tracking results, merged later on "Cell"
- `"artifact"` : path (str) to the generated CSV file
- `"log"`      : short message string for pipeline logs

NOTES
-----
- This plugin does *not* use the `images` or `labels` inputs directly; it
  relies on the label TIFFs stored in `labels_dir`.
- It calls:
    `python -m src.tracking.track_cells --labels_dir ... --sample ... --out ...`
- The external script must write a CSV with per-cell tracking data,
  ideally containing a `"Cell"` column to merge with the main per-cell table.
- If a plugin fails or produces no output, it exits gracefully without
  interrupting the rest of the analysis.

EXAMPLE WORKFLOW
----------------
1. You segment each frame of a time-lapse dataset (StarDist → labels per t).
2. You enable this plugin in your config:
    analysis:
      run: true
      plugins: [timeseries_tracking]
    analysis_cfg:
      tracking:
        labels_dir: "data/timeseries/AP231_1"
        sample: "AP231_1"
3. During post-analysis, this plugin calls the tracking script and merges
   its output (per-cell track IDs, trajectories, etc.) into the main table.

"""

from __future__ import annotations
import subprocess
from pathlib import Path
import pandas as pd

PLUGIN_NAME = "timeseries_tracking"

def augment(df, images, labels, ctx):
    """
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
    # Read configuration from context
    analysis_cfg = ctx.get("analysis_cfg", {}) or {}
    tcfg = analysis_cfg.get("tracking", {}) or {}

    labels_dir = Path(tcfg.get("labels_dir", "data/timeseries")).expanduser()
    sample     = tcfg.get("sample", None)
    max_dist   = str(tcfg.get("max_dist", 20.0))

    if not sample or not labels_dir.exists():
        print("[tracking-plugin] Skipped (missing sample or labels_dir).")
        return {}

    # Prepare output file path
    out_csv = Path("outputs") / f"{sample}_tracks.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Build command line for tracking module
    cmd = [
        "python", "-m", "src.tracking.track_cells",
        "--labels_dir", str(labels_dir),
        "--sample", sample,
        "--out", str(out_csv),
        "--max_dist", max_dist,
    ]
    print(f"[tracking-plugin] Running: {' '.join(cmd)}")

    # Run the command and load the resulting CSV
    try:
        subprocess.run(cmd, check=True)
        if out_csv.exists():
            df_track = pd.read_csv(out_csv)
            print(f"[tracking-plugin] OK: {len(df_track)} rows -> {out_csv}")

            # Optional: ensure it has a "Cell" column for merging
            # if "track_id" in df_track.columns and "Cell" not in df_track.columns:
            #     df_track = df_track.rename(columns={"track_id": "Cell"})
            
            return {"df_add": df_track, "artifact": str(out_csv), "log": "tracking OK"}
        else:
            print("[tracking-plugin] No output CSV produced.")
    except Exception as e:
        # Catch subprocess or file reading errors
        print(f"[tracking-plugin] failed: {e}")

    # If nothing worked, return empty dict (runner will safely skip merge)
    return {}
