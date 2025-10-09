# src/tracking/track_cells.py
"""
Minimal cell tracking across timepoints using centroid distance + IoU gating.
Input:
  --labels_dir data/timeseries/
     expects files like: <sample>_t000_labels.tif, <sample>_t001_labels.tif ...
Output:
  outputs/<sample>_tracks.csv with columns: Time, Cell, TrackID, Y, X, Area
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
    prev_df = prev_df.copy(); curr_df = curr_df.copy()
    prev_df["__assigned"] = False
    curr_df["TrackID"] = -1

    P = prev_df[["Y","X"]].values
    C = curr_df[["Y","X"]].values

    for j in range(len(curr_df)):
        yx = C[j]
        d2 = np.sum((P - yx)**2, axis=1)
        i = np.argmin(d2)
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
    ap = argparse.ArgumentParser(description="Simple centroid-based cell tracking.")
    ap.add_argument("--labels_dir", required=True, help="Folder with *_t###_labels.tif files")
    ap.add_argument("--sample", required=True, help="Sample stem, e.g., AP231_1")
    ap.add_argument("--out", default=None, help="Output CSV path")
    ap.add_argument("--max_dist", type=float, default=20.0)
    args = ap.parse_args()

    d = Path(args.labels_dir)
    patt = re.compile(rf"^{re.escape(args.sample)}_t(\d+)_labels\.tif$", re.IGNORECASE)
    files = sorted([p for p in d.iterdir() if patt.match(p.name)], key=lambda p: int(patt.match(p.name).group(1)))
    if not files:
        raise SystemExit("No time-labeled TIFFs found.")

    tracks = []
    prev = None
    for t, f in enumerate(files):
        lbl = tifffile.imread(str(f))
        if lbl.ndim == 3:
            lbl = lbl.max(axis=0)
        cur = _centroids(lbl)
        if prev is None:
            cur["TrackID"] = np.arange(1, len(cur)+1, dtype=int)
        else:
            cur, prev = _match(prev, cur, max_dist=args.max_dist)
        cur["Time"] = t
        tracks.append(cur[["Time","Cell","TrackID","Y","X","area"]])
        prev = cur[["Y","X","TrackID"]].copy()
    out = pd.concat(tracks, ignore_index=True)
    out_path = Path(args.out) if args.out else Path("outputs") / f"{args.sample}_tracks.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[tracking] wrote {out_path}")

if __name__ == "__main__":
    main()
