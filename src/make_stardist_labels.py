"""
StarDist Segmentation Script
----------------------------
Loads a microscopy TIFF, extracts the nuclear channel, projects/slices to 2D,
normalizes, and segments nuclei with a pretrained StarDist 2D model.

Example:
python -m src.make_stardist_labels \
    --input "C:/data/tiff/AP231_1.tif" \
    --output "C:/projects/.../AP231_1_stardist_labels.tif" \
    --channel 0 --mode mip --n_tiles 2,2
"""

import os
# Force CPU and quiet logs BEFORE TensorFlow loads
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # no GPU
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # 0=all, 1=info, 2=warn, 3=error

import argparse
import numpy as np
import tifffile as tiff
from stardist.models import StarDist2D

# ----- helpers -----

def load_channel_2d(path, channel=0, mode="mip"):
    """Return a 2D image from a TIFF by selecting channel and MIP/middle."""
    arr = tiff.imread(path)

    # Common shapes:
    # (Z, C, H, W)   -> take channel -> (Z,H,W)
    # (Z, H, W)      -> already single-channel z-stack
    # (H, W)         -> already 2D
    if arr.ndim == 4:
        Z, C, H, W = arr.shape
        if channel < 0 or channel >= C:
            raise ValueError(f"Channel {channel} out of range for shape {arr.shape}")
        stack = arr[:, channel, :, :]  # (Z,H,W)
    elif arr.ndim == 3:
        stack = arr  # assume (Z,H,W)
    elif arr.ndim == 2:
        img2d = arr.astype(np.float32)
        return normalize_025_985(img2d)
    else:
        raise ValueError(f"Unsupported TIFF shape {arr.shape}")

    if mode == "mip":
        img2d = stack.max(axis=0).astype(np.float32)
    elif mode == "middle":
        img2d = stack[stack.shape[0] // 2].astype(np.float32)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return normalize_025_985(img2d)

def normalize_025_985(img):
    """Robust percentile normalization to [0,1]."""
    p2, p985 = np.percentile(img, (2.0, 98.5))
    if p985 > p2:
        img = (img - p2) / (p985 - p2)
    return np.clip(img, 0, 1)

# ----- main -----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to input .tif")
    ap.add_argument("--output", required=True, help="Path to write *_stardist_labels.tif")
    ap.add_argument("--channel", type=int, default=0, help="Channel index for nuclei (default 0)")
    ap.add_argument("--mode", choices=["mip", "middle"], default="mip", help="Z→2D strategy")
    ap.add_argument("--prob_thresh", type=float, default=0.48, help="Probability threshold")
    ap.add_argument("--nms_thresh", type=float, default=0.30, help="NMS threshold")
    ap.add_argument("--n_tiles", default="2,2", help="Tiling as rows,cols (e.g. 2,2 or 3,3)")
    args = ap.parse_args()

    # Parse tiling
    try:
        n_tiles = tuple(int(v) for v in str(args.n_tiles).split(","))
        if len(n_tiles) != 2 or any(v <= 0 for v in n_tiles):
            raise ValueError
    except Exception:
        raise SystemExit("Invalid --n_tiles. Use e.g. '2,2' or '3,3'.")

    # Load image to 2D
    img = load_channel_2d(args.input, channel=args.channel, mode=args.mode)

    # Load model (downloaded on first run)
    model = StarDist2D.from_pretrained("2D_versatile_fluo")

    # Predict instances with tiling (reduces peak RAM)
    labels, _ = model.predict_instances(
        img,
        n_tiles=n_tiles,
        prob_thresh=args.prob_thresh,
        nms_thresh=args.nms_thresh,
    )

    # Save as uint16 label image
    tiff.imwrite(args.output, labels.astype(np.uint16))
    print(f"[saved] {args.output}")

if __name__ == "__main__":
    main()
