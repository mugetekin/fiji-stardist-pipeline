# src/tracking/track_cells.py
"""
Minimal cell tracking across timepoints using **centroid-distance gating**.
- Reads a sequence of **label TIFFs** for consecutive timepoints (t = 0..T-1).
- Extracts per-object **centroids** and **areas** at each timepoint.
- Links objects between t-1 and t using a **greedy nearest-neighbour** rule:
  if the closest previous centroid is within `--max_dist` pixels and not
  already assigned, the current object inherits that previous object's TrackID;
  otherwise, a **new TrackID** is created.
- Writes a CSV with columns: Time, Cell, TrackID, Y, X, Area.

ASSUMPTIONS & LIMITATIONS
-------------------------
- Tracks are created with a simple nearest-centroid rule; there is **no IoU** check
  or motion model here (keeps it minimal and fast for small-to-medium scenes).
- If your TIFF is 3D (Z, H, W), we **max-project** along Z before measuring centroids.
- This script assumes filenames follow:
    <sample>_t000_labels.tif, <sample>_t001_labels.tif, ...
  (You can change the regex `patt` below if your pattern differs.)
- TrackIDs are **global integers** starting at 1 for the first frame, then
  increasing as new objects appear. TrackIDs are not re-used.

USAGE
-----
python -m src.tracking.track_cells \
  --labels_dir data/timeseries/AP231_1 \
  --sample AP231_1 \
  --out outputs/AP231_1_tracks.csv \
  --max_dist 20

ARGUMENTS
---------
--labels_dir : folder that contains time-labeled TIFFs (see naming pattern).
--sample     : sample stem (used to filter & order files).
--out        : output CSV path (default: outputs/<sample>_tracks.csv).
--max_dist   : max centroid distance (pixels) to link across frames.

POSSIBLE EXTENSIONS
-------------------
- Add IoU gating: load both label masks for (t-1, t), compute overlaps for
  candidate pairs, and require IoU >= threshold in addition to distance.
- Add cost-matrix + Hungarian assignment for globally optimal matching.
- Add disappearance/appearance handling windows, gap closing, divisions, etc.
"""


import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import tifffile
from skimage.measure import regionprops_table

def _centroids(lbl):
    props = regionprops_table(lbl.astype(int), properties=("label","area","centroid"))
    df = pd.DataFrame(props).rename(columns={"label":"Cell","centroid-0":"Y","centroid-1":"X"})
    return df

def _match(prev_df, curr_df, max_dist=20.0):
    """Greedy centroid matching with distance gating."""
    # Work on copies to avoid mutating the original inputs
    prev_df = prev_df.copy(); curr_df = curr_df.copy()

    # Mark previous objects as available/unassigned; current will get TrackID
    prev_df["__assigned"] = False
    curr_df["TrackID"] = -1   # -1 = not assigned yet

    # Array views for vectorized distance computations
    P = prev_df[["Y","X"]].values  # shape (N, 2)
    C = curr_df[["Y","X"]].values  # shape (M, 2)

    # For each current object, find the closest previous object
    for j in range(len(curr_df)):
        yx = C[j]                         # current centroid (Y, X) 
        d2 = np.sum((P - yx)**2, axis=1)  # squared distances to all prev
        i = np.argmin(d2)                 # best previous index
        # Accept the match if within threshold AND the prev object isn't taken
        if np.sqrt(d2[i]) <= max_dist and not prev_df.loc[i,"__assigned"]:
            curr_df.loc[j, "TrackID"] = int(prev_df.loc[i,"TrackID"])
            prev_df.loc[i,"__assigned"] = True
        else:
            # new track, assign later
            pass
          
    # new tracks for unassigned
    new_track_base = (prev_df["TrackID"].max() if "TrackID" in prev_df.columns else 0) + 1
    for j in range(len(curr_df)):
        if curr_df.loc[j,"TrackID"] < 0:
            curr_df.loc[j,"TrackID"] = int(new_track_base)
            new_track_base += 1
    return curr_df.drop(columns=["__assigned"], errors="ignore"), prev_df.drop(columns=["__assigned"], errors="ignore")

def main():
    """
    CLI entry point: parses args, finds files, tracks frame-by-frame, writes CSV.
    """
    ap = argparse.ArgumentParser(description="Simple centroid-based cell tracking.")
    ap.add_argument("--labels_dir", required=True, help="Folder with *_t###_labels.tif files")
    ap.add_argument("--sample", required=True, help="Sample stem, e.g., AP231_1")
    ap.add_argument("--out", default=None, help="Output CSV path")
    ap.add_argument("--max_dist", type=float, default=20.0)
    args = ap.parse_args()

    d = Path(args.labels_dir)
    # Regex to select and order timepoint files for the given sample name.
    # Example: AP231_1_t003_labels.tif  → captures "003" as the time index
    patt = re.compile(rf"^{re.escape(args.sample)}_t(\d+)_labels\.tif$", re.IGNORECASE)
    # Collect matching files and sort by extracted integer time index
    files = sorted([p for p in d.iterdir() if patt.match(p.name)], key=lambda p: int(patt.match(p.name).group(1)))
    if not files:
        raise SystemExit("No time-labeled TIFFs found.")

    tracks = []    # list of per-frame DataFrames to concat at the end
    prev = None    # will hold ["Y", "X", "TrackID"] from previous frame
  
    for t, f in enumerate(files):
        # Read label image; if 3D, max-project along Z (shape: (Z, H, W) → (H, W))
        lbl = tifffile.imread(str(f))
        if lbl.ndim == 3:
            lbl = lbl.max(axis=0)

        # Measure centroids and area for the current frame
        cur = _centroids(lbl)
        if prev is None:
            # First frame: initialize TrackIDs as 1..N
            cur["TrackID"] = np.arange(1, len(cur)+1, dtype=int)
        else:
            # Match current objects to previous objects
            cur, prev = _match(prev, cur, max_dist=args.max_dist)

        # Stamp time index and select a consistent column order
        cur["Time"] = t
        tracks.append(cur[["Time","Cell","TrackID","Y","X","area"]])

        # For the next iteration, keep only what's needed to match: (Y, X, TrackID)
        prev = cur[["Y","X","TrackID"]].copy()

    # Concatenate all frames into one table and write CSV
    out = pd.concat(tracks, ignore_index=True)
    out_path = Path(args.out) if args.out else Path("outputs") / f"{args.sample}_tracks.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[tracking] wrote {out_path}")

if __name__ == "__main__":
    main()
