"""
Nuclei preview + colorized export pipeline (DAPI/Alexa/Cy3)

What this does (high level):
- Loads a multi-channel confocal TIFF (from Fiji, etc.).
- Selects a z-slice (middle, explicit z index) or computes a MIP.
- Performs background subtraction (rolling-ball) per channel.
- Normalizes to [0,1], colorizes (DAPI=Blue, Alexa=Green, Cy3=Red),
  saves per-channel previews + an RGB merge (JPG).
- (Optionally) triggers a Napari-based review UI (if installed).
- (Optionally) runs downstream post-analysis if labels exist.

Inputs can come from CLI flags or a YAML config.
CLI takes precedence over config; sensible defaults are provided.
"""

import argparse
import os
from pathlib import Path
import yaml

import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import rolling_ball
from skimage import exposure, img_as_ubyte
import tifffile

# Prefer imageio for writing; fall back to skimage if needed.
# Using imageio avoids some legacy warnings and has broad codec support.
try:
    import imageio.v3 as iio
    _HAS_IMAGEIO = True
except Exception:
    from skimage import io as skio
    _HAS_IMAGEIO = False


# ---------------- helpers ----------------
def _normalize01(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    vmin, vmax = float(img.min()), float(img.max())
    if vmax <= vmin + 1e-12:
        # Flat image → return zeros to avoid divide-by-zero.
        return np.zeros_like(img, dtype=np.float32)
    return (img - vmin) / (vmax - vmin)


def process_zstack_slice(channel_stack: np.ndarray, z_index="middle", rolling_radius=50) -> np.ndarray:
    """
    Process a single-channel Z stack into a 2D image.

    Parameters
    ----------
    channel_stack : np.ndarray
        Array of shape (Z, Y, X) for a single fluorescence channel.
    z_index : "middle" | int | None
        "middle" → pick the central slice,
        int      → pick that explicit Z,
        None     → compute Max-Intensity Projection (MIP).
    rolling_radius : int
        Radius (in pixels) for rolling-ball background subtraction.

    Returns
    -------
    np.ndarray (float32)
        2D image in [0,1] after background subtraction + rescaling.
    """
    Z = channel_stack.shape[0]
    if z_index == "middle":
        zi = Z // 2
        img = channel_stack[zi]
    elif isinstance(z_index, int):
        # Clamp to valid range to avoid IndexError.
        zi = max(0, min(Z - 1, z_index))
        img = channel_stack[zi]
    else:  # None => MIP
        img = channel_stack.max(axis=0)

    # # Background subtraction to remove uneven illumination. (rolling ball)
    bg = rolling_ball(img, radius=rolling_radius)
    img_corr = img - bg
    img_corr[img_corr < 0] = 0.0
    # Rescale to [0,1] for consistent downstream visualization.
    return exposure.rescale_intensity(img_corr, out_range=(0, 1)).astype(np.float32)


def infer_channel_indices(C: int):
    """
    Map channel indices to expected dyes (heuristic).
    Assumes acquisition order: 0=DAPI(Blue), 1=Alexa488(Green), 2=Cy3(Red).
    If fewer than 3 channels are present, some entries may be None.
    """
    dapi = 0 if C >= 1 else None
    alexa = 1 if C >= 2 else None
    cy3 = 2 if C >= 3 else None
    return dapi, alexa, cy3


def _ensure_parent_dir(path: Path):
    """Create parent directory for a file path if it doesn’t exist."""
    parent = path.parent
    if str(parent) not in ("", "."):
        parent.mkdir(parents=True, exist_ok=True)


def _imwrite_safe(path: Path, arr_uint8: np.ndarray):
    """
    Write an RGB uint8 image to disk.
    Uses imageio if available; falls back to skimage.io otherwise.
    Raises RuntimeError with context on failure.
    """
    _ensure_parent_dir(path)
    try:
        if _HAS_IMAGEIO:
            iio.imwrite(str(path), arr_uint8)
        else:
            from skimage import io as skio
            skio.imsave(str(path), arr_uint8)
    except Exception as e:
        raise RuntimeError(f"Failed to save image to '{path}': {e}")
    if not path.exists():
        # Extra guard for network mounts or silent failures.
        print(f"[WARN] Save reported ok, but file not found on disk: {path}")


def _to_uint8(img01: np.ndarray) -> np.ndarray:
    """Convert [0,1] float image to uint8 safely (clipping included)."""
    return img_as_ubyte(np.clip(img01, 0, 1))


def show_channels_colorized(dapi, alexa, cy3, save_prefix="TEST", show: bool = False):
    """
    Show & save each channel in its own color (Blue/Green/Red) and also RGB merge.
    dapi/alexa/cy3: [0,1] normalized 2D numpy arrays (or None)
    show: if True, opens a blocking matplotlib window; else just saves files.
    """
    imgs, titles = [], []
    saved_paths = []

    # Base path objects (not strictly needed but keeps prints clean).
    base = Path(save_prefix).resolve()

    # DAPI (Blue) → B channel
    if dapi is not None:
        rgb = np.zeros((*dapi.shape, 3), dtype=np.float32)
        rgb[..., 2] = dapi
        imgs.append(rgb); titles.append("DAPI (Blue)")
        p = Path(f"{save_prefix}_DAPIcolor.jpg").resolve()
        _imwrite_safe(p, _to_uint8(rgb)); saved_paths.append(p)

    # Alexa488 (Green) → G channel
    if alexa is not None:
        rgb = np.zeros((*alexa.shape, 3), dtype=np.float32)
        rgb[..., 1] = alexa
        imgs.append(rgb); titles.append("Alexa488 (Green)")
        p = Path(f"{save_prefix}_Alexacolor.jpg").resolve()
        _imwrite_safe(p, _to_uint8(rgb)); saved_paths.append(p)

    # Cy3 (Red) → R channel
    if cy3 is not None:
        rgb = np.zeros((*cy3.shape, 3), dtype=np.float32)
        rgb[..., 0] = cy3
        imgs.append(rgb); titles.append("Cy3 (Red)")
        p = Path(f"{save_prefix}_Cy3color.jpg").resolve()
        _imwrite_safe(p, _to_uint8(rgb)); saved_paths.append(p)

    # Build an RGB merge in the expected order (R=Cy3, G=Alexa, B=DAPI).
    ref = dapi if dapi is not None else (alexa if alexa is not None else cy3)
    if ref is None:
        raise ValueError("No channels to display; all are None.")
    H, W = ref.shape
    merge = np.zeros((H, W, 3), dtype=np.float32)
    if cy3 is not None:   merge[..., 0] = cy3
    if alexa is not None: merge[..., 1] = alexa
    if dapi is not None:  merge[..., 2] = dapi
    imgs.append(merge); titles.append("RGB Merge (DAPI=Blue, Alexa=Green, Cy3=Red)")

    # Save merge as JPG (optionally also as TIFF if needed).
    p_merge_jpg = Path(f"{save_prefix}_RGBmerge.jpg").resolve()
    _imwrite_safe(p_merge_jpg, _to_uint8(merge)); saved_paths.append(p_merge_jpg)
    # p_merge_tif = Path(f"{save_prefix}_RGBmerge.tif").resolve()
    # tifffile.imwrite(str(p_merge_tif), _to_uint8(merge))
    # saved_paths.append(p_merge_tif)

    # Create a quick side-by-side figure for visual QC.
    n = len(imgs)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, img, ttl in zip(axes, imgs, titles):
        ax.imshow(img)
        ax.set_title(ttl)
        ax.axis("off")
    plt.tight_layout()
    if show:
        plt.show()  # blocking
    else:
        plt.close(fig)

    print("Saved files:")
    for p in saved_paths:
        print(" -", p)

# -------- path resolution --------
def _strip_leading_slashes(p: str) -> str:
    # Avoid absolute path from accidental leading '/' or '\'
    if p.startswith("\\") or p.startswith("/"):
        return p.lstrip("\\/")  # make relative
    return p


def resolve_input_path(user_path: str | os.PathLike) -> Path:
    """
    Try to resolve input path robustly on Windows/Linux:
    - Accept absolute paths as-is (if they exist).
    - If the path starts with '/' or '\\', strip and treat as relative.
    - Try relative to CWD, script dir, and project root (script_dir/..).
    """
    candidates: list[Path] = []
    raw = Path(str(user_path))
    if raw.is_absolute() and raw.is_file():
        return raw

    stripped = Path(_strip_leading_slashes(str(raw)))

    cwd = Path.cwd()
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent  # assuming src/ layout

    for base in [cwd, script_dir, project_root]:
        candidates.append((base / stripped).resolve())

    # Also try as-is (relative to CWD)
    candidates.append((cwd / raw).resolve())

    for cand in candidates:
        if cand.is_file():
            return cand

    tried = "\n  - ".join(str(c) for c in dict.fromkeys(candidates))
    raise FileNotFoundError(
        f"TIFF not found. I tried resolving these locations:\n  - {tried}\n"
        "Tip: In configs, prefer relative paths like 'data/AP231_1.tif' (without a leading slash)."
    )


# -------- main processing --------
def show_channels_from_tiff(path, z_index="middle", rolling_radius=50, save_prefix="TEST", show_plots: bool = False):
    """
    Open multi-channel TIFF from Fiji; take DAPI/Alexa/Cy3 slice (or z_index/MIP),
    background-subtract (rolling ball), colorize + merge, and save.
    Returns: (dapi, alexa, cy3) as [0,1] float32 arrays (or None if channel missing).
    show_plots: forward to show_channels_colorized(show=...), default False to avoid blocking.
    """
    resolved_path = resolve_input_path(path)
    print("Resolved TIFF path:", resolved_path)

    arr = tifffile.imread(str(resolved_path))  # (Z,C,Y,X) or (C,Z,Y,X) or other
    print("Original TIFF shape:", arr.shape)

    # Axis fix: if first axis is channels (<=5), move to (Z,C,Y,X)
    if arr.ndim == 4 and arr.shape[0] <= 5:
        arr = np.moveaxis(arr, 0, 1)  # -> (Z,C,Y,X)

    if arr.ndim != 4:
        raise ValueError(f"Expected a 4D stack, got shape {arr.shape}")

    Z, C, Y, X = arr.shape
    dapi_c, alexa_c, cy3_c = infer_channel_indices(C)

    dapi = alexa = cy3 = None
    if dapi_c is not None and dapi_c < C:
        dapi = process_zstack_slice(arr[:, dapi_c], z_index, rolling_radius)
    if alexa_c is not None and alexa_c < C:
        alexa = process_zstack_slice(arr[:, alexa_c], z_index, rolling_radius)
    if cy3_c is not None and cy3_c < C:
        cy3 = process_zstack_slice(arr[:, cy3_c], z_index, rolling_radius)

    # Ensure output dir exists for the prefix
    _ensure_parent_dir(Path(save_prefix))

    show_channels_colorized(dapi, alexa, cy3, save_prefix=save_prefix, show=show_plots)
    return dapi, alexa, cy3


# -------- CLI / config --------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=False, help="YAML config file")
    ap.add_argument("--input_path", type=str, required=False, help="TIFF path (overrides config)")
    ap.add_argument("--save_prefix", type=str, required=False, help="Prefix for saved images")
    ap.add_argument("--mode", type=str, default=None,
                    help="z selection: 'middle' (default), 'mip'/'none' for MIP, or 'z=<int>'")
    ap.add_argument("--rolling_radius", type=int, default=None, help="Rolling-ball radius")
    ap.add_argument("--show", action="store_true", help="Show matplotlib figure (blocks). Off by default.")
    ap.add_argument("--plugins", nargs="*", help="Optional list of plugin module names to activate.")
    ap.add_argument("--review", action="store_true",
                help="Open Napari-based review UI after processing (if installed).")
    return ap.parse_args()


def load_config(path: str | None):
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_mode_to_zindex(mode_str: str | None):
    if not mode_str or mode_str.lower() == "middle":
        return "middle"
    m = mode_str.lower()
    if m in ("mip", "none"):
        return None
    if m.startswith("z="):
        try:
            return int(m.split("=", 1)[1])
        except Exception:
            pass
    # fallback
    return "middle"


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Gather parameters with precedence: CLI > config > defaults
    input_path = args.input_path or cfg.get("input_path")
    save_prefix = args.save_prefix or cfg.get("save_prefix", "outputs/AP231_1")
    mode = args.mode or cfg.get("mode", "middle")
    rolling_radius = args.rolling_radius if args.rolling_radius is not None else cfg.get("rolling_radius", 50)

    if not input_path:
        raise SystemExit("Please provide --input_path or set 'input_path' in the config YAML.")

    z_index = parse_mode_to_zindex(mode)

    print("[pipeline] config:", args.config or "(none)")
    print("[pipeline] input_path:", input_path)
    print("[pipeline] save_prefix:", save_prefix)
    print("[pipeline] mode:", "mip" if z_index is None else z_index)
    dapi, alexa, cy3 = show_channels_from_tiff(
        input_path,
        z_index=z_index,
        rolling_radius=rolling_radius,
        save_prefix=save_prefix,
        show_plots=args.show,   # <-- non-blocking by default
    )
    print("Preview images saved with prefix:", Path(save_prefix).resolve())

    # ----------------------------
    # Optional post-count analysis
    # ----------------------------
    analysis_cfg = cfg.get("analysis", {}) if isinstance(cfg, dict) else {}
    
    # Merge CLI-provided plugin list into config (overrides YAML)
    if args.plugins is not None:
        analysis_cfg.setdefault("plugins", [])
        # If user wrote --plugins with no names, that disables all plugins
        analysis_cfg["plugins"] = args.plugins


    # Prefix variable used by post_analysis
    prefix = save_prefix

    if analysis_cfg.get("run", False):
        # Require the StarDist label file to exist; otherwise skip gracefully
        label_path = f"{prefix}_stardist_labels.tif"
        if not Path(label_path).exists():
            print(f"[analysis] skipped: missing label file: {label_path}")
        else:
            try:
                # lazy import so pipeline works even if post_analysis is absent
                try:
                    from src.post_analysis import run_analysis
                except Exception:
                    from .post_analysis import run_analysis  # if run as a package
                print("[analysis] running post-count analysis…")
                run_analysis(prefix, analysis_cfg)
            except Exception as e:
                print(f"[analysis] skipped due to error: {e}")
    else:
        print("[analysis] disabled (analysis.run is false or missing).")


if __name__ == "__main__":
    main()
