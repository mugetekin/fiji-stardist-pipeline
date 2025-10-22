"""
Plugin Runner
This module is a lightweight plugin management system for the Fiji-StarDist
pipeline. It dynamically imports and executes analysis plugins (like
`organelle_puncta`, `timeseries_tracking`, etc.), merges their outputs, and
returns an updated per-cell DataFrame.

WHY THIS EXISTS
---------------
After nuclei segmentation (StarDist) and image processing, we often want to
perform *additional* per-cell analyses: e.g.
  - counting puncta (see organelle_puncta.py),
  - tracking intensity across timepoints,
  - computing morphological ratios, etc.

Instead of hard-coding all these analyses inside one big monolithic script,
the pipeline uses a **plugin architecture**. Each plugin is an independent
Python file under `src/plugins/` that defines:
  - `PLUGIN_NAME`: a short identifier string,
  - `augment(df, images, labels, ctx)`: a function that receives all data and
    returns new per-cell columns or additional results.

WHEN IT RUNS
------------
It is invoked automatically by:
  - `src/post_analysis.py` → inside the function that runs postprocessing,
  - after segmentation and per-cell table creation,
  - before writing final CSV results.
So you usually don’t call it manually — it’s part of the standard analysis flow.

OUTPUT STRUCTURE
----------------
Returns a tuple: `(df, outputs)`
- `df`: The merged per-cell DataFrame (original + all plugin columns)
- `outputs`: List of tuples `(plugin_name, output_dict)` where output_dict may include:
      "df_add"     → plugin’s DataFrame with "Cell" column
      "artifacts"  → optional images or overlays
      "log"        → short log message (e.g., "puncta detected=123")

"""


from __future__ import annotations
from typing import Dict, List, Any
import importlib

class AnalysisContext(dict):
    """Arbitrary metadata container (prefix, thresholds, etc.)."""

def run_plugins(plugin_names: List[str],
                df,
                images: Dict[str, Any],
                labels,
                ctx: AnalysisContext):
    """
    Each plugin module must expose:
      - PLUGIN_NAME: str
      - augment(df, images, labels, ctx) -> dict
        returns {
          "df_add": DataFrame with "Cell" + new columns   (optional)
          "artifacts": List[(path, np.ndarray | PIL)]     (optional)
          "log": str                                      (optional)
        }
    We merge df_add into df on 'Cell'; save artifacts handled by caller if desired.
    """
    outputs = []
    for name in plugin_names or []:
        try:
            mod = importlib.import_module(f"src.plugins.{name}")
        except Exception as e:
            print(f"[plugins] skip '{name}': cannot import ({e})")
            continue
        if not hasattr(mod, "augment"):
            print(f"[plugins] skip '{name}': no augment()")
            continue
        try:
            out = mod.augment(df, images, labels, ctx) or {}
            outputs.append((name, out))
        except Exception as e:
            print(f"[plugins] '{name}' failed: {e}")
    # merge
    for name, out in outputs:
        add = out.get("df_add")
        if add is not None and "Cell" in add.columns:
            df = df.merge(add, on="Cell", how="left")
    return df, outputs
