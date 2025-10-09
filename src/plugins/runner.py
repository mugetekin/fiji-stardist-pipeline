# src/plugins/runner.py
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
